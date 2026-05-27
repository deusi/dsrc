from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.envs.base_ctde_env import AVAction
from src.envs.wrappers import decode_headway_bin, decode_speed_bin, lane_preference_to_action
from src.safety.constraints import SafetyConstraints
from src.safety.etiquette import is_all_lane_low_speed_occupancy, is_low_speed_uncongested, is_passing_lane_slow_hold


@dataclass
class SafetyState:
    last_lane_change_time_s: float | None = None
    lane_changes_last_km: int = 0
    distance_since_window_start_m: float = 0.0
    last_lane_index: Any | None = None


@dataclass(frozen=True)
class SafetyContext:
    time_s: float
    free_flow_speed_mps: float = 30.0
    min_contextual_speed_mps: float = 12.0
    local_density_veh_per_km: float = 0.0
    downstream_congested: bool = False
    target_lane_exists: bool = True
    target_lane_front_gap_m: float = float("inf")
    target_lane_rear_gap_m: float = float("inf")
    target_lane_rear_required_decel_mps2: float = 0.0
    all_lanes_av_occupied: bool = False
    av_mean_speed_mps: float = 30.0
    in_passing_lane: bool = False
    local_mean_speed_mps: float = 30.0
    near_merge: bool = False


@dataclass(frozen=True)
class SafetyDecision:
    target_speed_mps: float
    target_headway_s: float
    lane_action: str | None
    diagnostics: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def empty_diagnostics() -> dict[str, list[dict[str, Any]]]:
    return {
        "safety_masked_action": [],
        "etiquette_blocked_action": [],
        "follower_disruption_blocked": [],
        "simulator_blocked_action": [],
    }


def apply_safety_layer(
    action: AVAction,
    state: SafetyState,
    context: SafetyContext,
    constraints: SafetyConstraints | None = None,
    agent_id: str | None = None,
) -> SafetyDecision:
    constraints = constraints or SafetyConstraints()
    diagnostics = empty_diagnostics()
    target_speed = decode_speed_bin(
        action["desired_speed_bin"],
        free_flow_speed_mps=context.free_flow_speed_mps,
        min_contextual_speed_mps=context.min_contextual_speed_mps,
    )
    target_headway = decode_headway_bin(action["desired_headway_bin"])
    if action["merge_mode"] == "create_gap":
        target_headway += constraints.merge_gap_headway_bonus_s
    lane_action = None if action["merge_mode"] == "hold_lane" else lane_preference_to_action(action["lane_preference"])

    if is_low_speed_uncongested(target_speed, context.free_flow_speed_mps, context.local_density_veh_per_km, constraints):
        target_speed = max(target_speed, context.free_flow_speed_mps - constraints.low_speed_free_flow_delta_mps)
        diagnostics["etiquette_blocked_action"].append({"agent_id": agent_id, "reason": "low_speed_uncongested"})

    if is_all_lane_low_speed_occupancy(
        context.all_lanes_av_occupied,
        context.av_mean_speed_mps,
        context.free_flow_speed_mps,
        context.downstream_congested,
        constraints,
    ):
        lane_action = None
        diagnostics["etiquette_blocked_action"].append({"agent_id": agent_id, "reason": "all_lane_low_speed_occupancy"})

    if is_passing_lane_slow_hold(context.in_passing_lane, target_speed, context.local_mean_speed_mps, constraints):
        diagnostics["etiquette_blocked_action"].append({"agent_id": agent_id, "reason": "passing_lane_slow_hold"})

    if lane_action is not None:
        dwell = float("inf") if state.last_lane_change_time_s is None else context.time_s - state.last_lane_change_time_s
        lane_changes_per_km = state.lane_changes_last_km / max(state.distance_since_window_start_m / 1000.0, 1e-9)
        if dwell < constraints.lane_change_dwell_s:
            lane_action = None
            diagnostics["safety_masked_action"].append({"agent_id": agent_id, "reason": "lane_change_dwell"})
        elif lane_changes_per_km >= constraints.max_lane_changes_per_km:
            lane_action = None
            diagnostics["safety_masked_action"].append({"agent_id": agent_id, "reason": "lane_changes_per_km"})
        elif not context.target_lane_exists:
            lane_action = None
            diagnostics["safety_masked_action"].append({"agent_id": agent_id, "reason": "target_lane_missing"})
        elif context.target_lane_rear_required_decel_mps2 > constraints.max_follower_braking_mps2:
            lane_action = None
            diagnostics["follower_disruption_blocked"].append({"agent_id": agent_id, "reason": "target_lane_rear_braking"})

    return SafetyDecision(
        target_speed_mps=target_speed,
        target_headway_s=target_headway,
        lane_action=lane_action,
        diagnostics=diagnostics,
    )

