from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyConstraints:
    """Conservative defaults for hard safety and etiquette checks."""

    lane_change_dwell_s: float = 15.0
    max_lane_changes_per_km: float = 2.0
    max_follower_braking_mps2: float = 2.5
    comfortable_decel_mps2: float = 3.0
    max_accel_mps2: float = 2.0
    max_decel_mps2: float = 3.0
    emergency_decel_mps2: float = 6.0
    min_front_gap_m: float = 5.0
    min_rear_gap_m: float = 5.0
    min_forward_ttc_s: float = 2.0
    min_lane_change_ttc_s: float = 2.5
    speed_control_kp: float = 0.6
    min_uncongested_speed_mps: float = 12.0
    uncongested_density_threshold_veh_per_km: float = 12.0
    low_speed_free_flow_delta_mps: float = 8.0
    merge_gap_headway_bonus_s: float = 0.8
