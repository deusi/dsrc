from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import random
from typing import Any

from src.controllers import BaseController, ControllerMetadata
from src.envs.base_ctde_env import AVActionMap, AVObservationMap, GlobalState
from src.envs.wrappers import HEADWAY_BINS, LANE_PREFERENCES, MERGE_MODES, SPEED_BINS


@dataclass(frozen=True)
class LocalBaselineConfig:
    safe_lane_gap_m: float = 25.0
    short_headway_s: float = 1.0
    larger_headway_s: float = 1.8
    short_leader_gap_m: float = 15.0
    cautious_leader_gap_m: float = 30.0
    queue_trigger: int = 1
    medium_density_bin: int = 1
    high_density_bin: int = 2
    low_speed_bin: int = 0
    speed_mismatch_tolerance_mps: float = 3.0
    low_flow_speed_mps: float = 12.0
    high_downstream_congestion: float = 0.5
    high_merge_pressure: float = 0.5
    bottleneck_near_m: float = 120.0

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> LocalBaselineConfig:
        cfg = dict(config or {})
        return cls(
            safe_lane_gap_m=float(cfg.get("safe_lane_gap_m", cls.safe_lane_gap_m)),
            short_headway_s=float(cfg.get("short_headway_s", cls.short_headway_s)),
            larger_headway_s=float(cfg.get("larger_headway_s", cls.larger_headway_s)),
            short_leader_gap_m=float(cfg.get("short_leader_gap_m", cls.short_leader_gap_m)),
            cautious_leader_gap_m=float(cfg.get("cautious_leader_gap_m", cls.cautious_leader_gap_m)),
            queue_trigger=int(cfg.get("queue_trigger", cls.queue_trigger)),
            medium_density_bin=int(cfg.get("medium_density_bin", cls.medium_density_bin)),
            high_density_bin=int(cfg.get("high_density_bin", cls.high_density_bin)),
            low_speed_bin=int(cfg.get("low_speed_bin", cls.low_speed_bin)),
            speed_mismatch_tolerance_mps=float(
                cfg.get("speed_mismatch_tolerance_mps", cls.speed_mismatch_tolerance_mps)
            ),
            low_flow_speed_mps=float(cfg.get("low_flow_speed_mps", cls.low_flow_speed_mps)),
            high_downstream_congestion=float(cfg.get("high_downstream_congestion", cls.high_downstream_congestion)),
            high_merge_pressure=float(cfg.get("high_merge_pressure", cls.high_merge_pressure)),
            bottleneck_near_m=float(cfg.get("bottleneck_near_m", cls.bottleneck_near_m)),
        )


def action(
    speed: str = "nominal",
    headway: str = "normal",
    lane: str = "keep",
    merge: str = "normal",
) -> dict[str, str]:
    return {
        "desired_speed_bin": speed,
        "desired_headway_bin": headway,
        "lane_preference": lane,
        "merge_mode": merge,
    }


class LocalBaselineController(BaseController):
    """Base class for non-learning baselines that forbid global sensing."""

    def __init__(
        self,
        metadata: ControllerMetadata,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(metadata)
        self.config = LocalBaselineConfig.from_mapping(config)

    def act(
        self,
        local_obs: AVObservationMap,
        global_state: GlobalState | None = None,
    ) -> AVActionMap:
        if global_state is not None:
            raise ValueError(f"{self.name} is infrastructure-free and must not receive global_state")
        return self._act_local(local_obs)

    def _act_local(self, local_obs: AVObservationMap) -> AVActionMap:
        raise NotImplementedError


class NoAVController(LocalBaselineController):
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            ControllerMetadata(name="no_av", family="baseline", safety_mode="simulator_default"),
            config,
        )

    def _act_local(self, local_obs: AVObservationMap) -> AVActionMap:
        return {}


class RandomAVController(LocalBaselineController):
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            ControllerMetadata(name="random_av", family="baseline", safety_mode="simulator_default"),
            config,
        )
        self._rng = random.Random()

    def reset(self, env_metadata: Mapping[str, Any] | None = None, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def _act_local(self, local_obs: AVObservationMap) -> AVActionMap:
        return {
            agent_id: action(
                speed=self._rng.choice(SPEED_BINS),
                headway=self._rng.choice(HEADWAY_BINS),
                lane=self._rng.choice(LANE_PREFERENCES),
                merge=self._rng.choice(MERGE_MODES),
            )
            for agent_id in local_obs
        }


class SelfishAVController(LocalBaselineController):
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            ControllerMetadata(name="selfish_av", family="baseline", safety_mode="simulator_default"),
            config,
        )

    def _act_local(self, local_obs: AVObservationMap) -> AVActionMap:
        return {agent_id: self._action_for(obs) for agent_id, obs in local_obs.items()}

    def _action_for(self, obs: Mapping[str, Any]) -> dict[str, str]:
        leader_gap = _finite_float(obs.get("leader_gap"), float("inf"))
        headway = _finite_float(obs.get("ego_headway_s"), float("inf"))
        leader_rel = _finite_float(obs.get("leader_relative_speed"), 0.0)
        speed = "fast"
        if leader_gap < self.config.short_leader_gap_m or (leader_rel < -3.0 and leader_gap < self.config.cautious_leader_gap_m):
            speed = "nominal"
        if leader_gap < self.config.short_leader_gap_m and leader_rel < -5.0:
            speed = "slow"
        headway_bin = _headway_from_context(obs, self.config)
        lane = _safe_preferred_lane(obs, self.config)
        return action(speed=speed, headway=headway_bin, lane=lane)


class DensityLookupController(LocalBaselineController):
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            ControllerMetadata(
                name="density_lookup",
                family="baseline",
                cooperation_mode="local_aggregate",
                safety_mode="integrated_rl",
            ),
            config,
        )

    def _act_local(self, local_obs: AVObservationMap) -> AVActionMap:
        return {agent_id: _density_action(obs, self.config) for agent_id, obs in local_obs.items()}


class DynamicSpeedLimitController(LocalBaselineController):
    """Local AV dynamic speed advisory, not infrastructure speed-limit control."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            ControllerMetadata(
                name="dynamic_speed_limit",
                family="baseline",
                cooperation_mode="local_aggregate",
                safety_mode="integrated_rl",
            ),
            config,
        )

    def _act_local(self, local_obs: AVObservationMap) -> AVActionMap:
        return {agent_id: self._action_for(obs) for agent_id, obs in local_obs.items()}

    def _action_for(self, obs: Mapping[str, Any]) -> dict[str, str]:
        density_bin = _int(obs.get("local_density_bin"))
        queue = _int(obs.get("local_queue_estimate"))
        downstream = _downstream_congestion(obs)
        if queue >= self.config.queue_trigger or density_bin >= self.config.high_density_bin or downstream >= self.config.high_downstream_congestion:
            return action(speed="slow", headway="largest", merge="hold_lane")
        if density_bin >= self.config.medium_density_bin:
            return action(speed="nominal", headway="larger")
        return action(speed="fast", headway="normal")


class AVMediatedSpeedHarmonizationController(LocalBaselineController):
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            ControllerMetadata(
                name="av_mediated_speed_harmonization",
                family="baseline",
                cooperation_mode="local_aggregate",
                safety_mode="integrated_rl",
            ),
            config,
        )

    def _act_local(self, local_obs: AVObservationMap) -> AVActionMap:
        return {agent_id: self._action_for(obs) for agent_id, obs in local_obs.items()}

    def _action_for(self, obs: Mapping[str, Any]) -> dict[str, str]:
        ego_speed = _finite_float(obs.get("ego_speed"), 0.0)
        flow_speed = _flow_speed(obs, ego_speed)
        mismatch = ego_speed - flow_speed
        density_bin = _int(obs.get("local_density_bin"))
        queue = _int(obs.get("local_queue_estimate"))
        tolerance = self.config.speed_mismatch_tolerance_mps
        if queue >= self.config.queue_trigger or flow_speed < self.config.low_flow_speed_mps:
            speed = "slow" if mismatch > tolerance else "nominal"
        elif mismatch > 2.0 * tolerance:
            speed = "slow"
        elif mismatch > tolerance:
            speed = "nominal"
        elif mismatch < -tolerance and density_bin < self.config.medium_density_bin:
            speed = "fast"
        else:
            speed = "nominal"
        headway = "larger" if queue or density_bin >= self.config.medium_density_bin else "normal"
        return action(speed=speed, headway=headway)


class BackpressureController(LocalBaselineController):
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            ControllerMetadata(
                name="backpressure",
                family="baseline",
                cooperation_mode="local_aggregate",
                safety_mode="integrated_rl",
            ),
            config,
        )

    def _act_local(self, local_obs: AVObservationMap) -> AVActionMap:
        return {agent_id: self._action_for(obs) for agent_id, obs in local_obs.items()}

    def _action_for(self, obs: Mapping[str, Any]) -> dict[str, str]:
        downstream = _downstream_congestion(obs)
        merge_pressure = _merge_pressure(obs)
        bottleneck_distance = _finite_float(obs.get("distance_to_downstream_bottleneck"), float("inf"))
        queue = _int(obs.get("local_queue_estimate"))
        leader_gap = _finite_float(obs.get("leader_gap"), float("inf"))
        if downstream >= self.config.high_downstream_congestion or bottleneck_distance <= self.config.bottleneck_near_m:
            return action(speed="slow", headway="largest", merge="hold_lane")
        if merge_pressure >= self.config.high_merge_pressure:
            return action(speed="nominal", headway="largest", merge="create_gap")
        if queue <= 0 and leader_gap >= self.config.cautious_leader_gap_m:
            return action(speed="fast", headway="normal")
        return action(speed="nominal", headway="larger")


class CooperativeSmoothingController(LocalBaselineController):
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            ControllerMetadata(
                name="cooperative_smoothing",
                family="baseline",
                cooperation_mode="local_aggregate",
                safety_mode="integrated_rl",
            ),
            config,
        )

    def _act_local(self, local_obs: AVObservationMap) -> AVActionMap:
        return {agent_id: self._action_for(obs) for agent_id, obs in local_obs.items()}

    def _action_for(self, obs: Mapping[str, Any]) -> dict[str, str]:
        density_bin = _int(obs.get("local_density_bin"))
        queue = _int(obs.get("local_queue_estimate"))
        leader_gap = _finite_float(obs.get("leader_gap"), float("inf"))
        leader_rel = _finite_float(obs.get("leader_relative_speed"), 0.0)
        merge_pressure = _merge_pressure(obs)
        downstream = _downstream_congestion(obs)
        if (
            queue >= self.config.queue_trigger
            or density_bin >= self.config.high_density_bin
            or leader_gap < self.config.short_leader_gap_m
            or downstream >= self.config.high_downstream_congestion
        ):
            speed = "slow"
            headway = "largest"
        elif density_bin >= self.config.medium_density_bin or leader_rel < -2.0 or merge_pressure > 0.0:
            speed = "nominal"
            headway = "larger"
        else:
            speed = "fast"
            headway = "normal"
        merge = "create_gap" if merge_pressure >= self.config.high_merge_pressure else "normal"
        return action(speed=speed, headway=headway, lane="keep", merge=merge)


def _density_action(obs: Mapping[str, Any], config: LocalBaselineConfig) -> dict[str, str]:
    density_bin = _int(obs.get("local_density_bin"))
    queue = _int(obs.get("local_queue_estimate"))
    speed_bin = _int(obs.get("local_mean_speed_bin"))
    if queue >= config.queue_trigger or density_bin >= config.high_density_bin or speed_bin <= config.low_speed_bin:
        return action(speed="slow", headway="largest")
    if density_bin >= config.medium_density_bin:
        return action(speed="nominal", headway="larger")
    return action(speed="fast", headway="normal")


def _headway_from_context(obs: Mapping[str, Any], config: LocalBaselineConfig) -> str:
    leader_gap = _finite_float(obs.get("leader_gap"), float("inf"))
    headway = _finite_float(obs.get("ego_headway_s"), float("inf"))
    if leader_gap < config.short_leader_gap_m or headway < config.short_headway_s:
        return "largest"
    if leader_gap < config.cautious_leader_gap_m or headway < config.larger_headway_s:
        return "larger"
    return "normal"


def _safe_preferred_lane(obs: Mapping[str, Any], config: LocalBaselineConfig) -> str:
    left_safe = (
        _finite_float(obs.get("left_lane_front_gap"), 0.0) >= config.safe_lane_gap_m
        and _finite_float(obs.get("left_lane_rear_gap"), 0.0) >= config.safe_lane_gap_m
    )
    right_safe = (
        _finite_float(obs.get("right_lane_front_gap"), 0.0) >= config.safe_lane_gap_m
        and _finite_float(obs.get("right_lane_rear_gap"), 0.0) >= config.safe_lane_gap_m
    )
    if left_safe:
        return "prefer_left_if_safe"
    if right_safe:
        return "prefer_right_if_safe"
    return "keep"


def _flow_speed(obs: Mapping[str, Any], fallback: float) -> float:
    if _int(obs.get("nearby_av_count")) > 0:
        return _finite_float(obs.get("nearby_av_mean_speed"), fallback)
    cooperation = obs.get("cooperation", {})
    if isinstance(cooperation, Mapping):
        return _finite_float(cooperation.get("segment_target_speed"), fallback)
    return _finite_float(obs.get("segment_target_speed"), fallback)


def _downstream_congestion(obs: Mapping[str, Any]) -> float:
    cooperation = obs.get("cooperation", {})
    if isinstance(cooperation, Mapping) and "downstream_congestion_estimate" in cooperation:
        return _finite_float(cooperation.get("downstream_congestion_estimate"), 0.0)
    return _finite_float(obs.get("downstream_congestion_estimate"), 0.0)


def _merge_pressure(obs: Mapping[str, Any]) -> float:
    cooperation = obs.get("cooperation", {})
    if isinstance(cooperation, Mapping) and "merge_pressure" in cooperation:
        return _finite_float(cooperation.get("merge_pressure"), 0.0)
    return _finite_float(obs.get("merge_pressure"), 0.0)


def _finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
