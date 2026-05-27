from __future__ import annotations

from src.safety.constraints import SafetyConstraints


def is_low_speed_uncongested(
    target_speed_mps: float,
    free_flow_speed_mps: float,
    density_veh_per_km: float,
    constraints: SafetyConstraints,
) -> bool:
    return (
        density_veh_per_km < constraints.uncongested_density_threshold_veh_per_km
        and target_speed_mps < free_flow_speed_mps - constraints.low_speed_free_flow_delta_mps
    )


def is_all_lane_low_speed_occupancy(
    all_lanes_av_occupied: bool,
    av_mean_speed_mps: float,
    free_flow_speed_mps: float,
    downstream_congested: bool,
    constraints: SafetyConstraints,
) -> bool:
    return (
        all_lanes_av_occupied
        and not downstream_congested
        and av_mean_speed_mps < free_flow_speed_mps - constraints.low_speed_free_flow_delta_mps
    )


def is_passing_lane_slow_hold(
    in_passing_lane: bool,
    ego_speed_mps: float,
    local_mean_speed_mps: float,
    constraints: SafetyConstraints,
) -> bool:
    return in_passing_lane and ego_speed_mps < local_mean_speed_mps - constraints.low_speed_free_flow_delta_mps

