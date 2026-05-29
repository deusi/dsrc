from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BurstProfile:
    enabled: bool = False
    start_s: float = 0.0
    end_s: float = 0.0
    multiplier: float = 1.0

    def multiplier_at(self, time_s: float) -> float:
        if not self.enabled:
            return 1.0
        if self.start_s <= time_s < self.end_s:
            return self.multiplier
        return 1.0


@dataclass(frozen=True)
class SpeedDistribution:
    mean_mps: float = 24.0
    std_mps: float = 2.5
    min_mps: float = 0.0
    max_mps: float = 40.0


@dataclass(frozen=True)
class DemandProfile:
    profile_id: str
    enabled: bool
    total_vehicles_per_hour: float
    av_penetration: float
    branch_split: Mapping[str, float]
    burst: BurstProfile
    speed_distribution: SpeedDistribution
    spawn_min_gap_m: float

    def vehicles_per_hour_at(self, time_s: float) -> float:
        if not self.enabled:
            return 0.0
        return self.total_vehicles_per_hour * self.burst.multiplier_at(time_s)


def load_demand_profile(config: Mapping[str, Any] | None, *, enabled: bool = True) -> DemandProfile:
    cfg = dict(config or {})
    total = float(cfg.get("total_vehicles_per_hour", cfg.get("total_veh_per_hour", 0.0)))
    av_penetration = float(cfg.get("av_penetration", cfg.get("role_distribution", {}).get("av", 0.0)))
    spawn_min_gap_m = float(cfg.get("spawn_min_gap_m", 12.0))
    if total < 0:
        raise ValueError("total_vehicles_per_hour must be non-negative")
    if not 0.0 <= av_penetration <= 1.0:
        raise ValueError("av_penetration must be in [0, 1]")
    if spawn_min_gap_m < 0:
        raise ValueError("spawn_min_gap_m must be non-negative")

    burst_cfg = cfg.get("burst", {})
    if not isinstance(burst_cfg, Mapping):
        raise ValueError("burst must be a mapping")
    burst = BurstProfile(
        enabled=bool(burst_cfg.get("enabled", False)),
        start_s=float(burst_cfg.get("start_s", 0.0)),
        end_s=float(burst_cfg.get("end_s", 0.0)),
        multiplier=float(burst_cfg.get("multiplier", 1.0)),
    )
    if burst.end_s < burst.start_s:
        raise ValueError("burst.end_s must be >= burst.start_s")
    if burst.multiplier < 0:
        raise ValueError("burst.multiplier must be non-negative")

    speed_cfg = cfg.get("speed_distribution", {})
    if not isinstance(speed_cfg, Mapping):
        raise ValueError("speed_distribution must be a mapping")
    speed_distribution = SpeedDistribution(
        mean_mps=float(speed_cfg.get("mean_mps", 24.0)),
        std_mps=float(speed_cfg.get("std_mps", 2.5)),
        min_mps=float(speed_cfg.get("min_mps", 0.0)),
        max_mps=float(speed_cfg.get("max_mps", 40.0)),
    )
    if speed_distribution.std_mps < 0:
        raise ValueError("speed_distribution.std_mps must be non-negative")
    if speed_distribution.max_mps < speed_distribution.min_mps:
        raise ValueError("speed_distribution.max_mps must be >= min_mps")

    return DemandProfile(
        profile_id=str(cfg.get("id", "custom")),
        enabled=enabled and bool(cfg.get("enabled", True)) and total > 0,
        total_vehicles_per_hour=total,
        av_penetration=av_penetration,
        branch_split=normalize_branch_split(cfg.get("branch_split", {"main": 1.0})),
        burst=burst,
        speed_distribution=speed_distribution,
        spawn_min_gap_m=spawn_min_gap_m,
    )


def normalize_branch_split(raw_split: Any) -> dict[str, float]:
    if not isinstance(raw_split, Mapping) or not raw_split:
        raise ValueError("branch_split must be a non-empty mapping")
    weights = {str(branch): float(weight) for branch, weight in raw_split.items()}
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("branch_split weights must be non-negative")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("branch_split must have positive total weight")
    return {branch: weight / total for branch, weight in weights.items()}
