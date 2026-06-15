from __future__ import annotations

import math
import time

import numpy as np
import pytest

from perception.distance import TrackedVehicle
from perception.observation_builder import BuilderConfig, ObservationBuilder, PeerState
from policy import sim_contract
from sensors.gps_reader import GpsFix


def make_vehicle(track_id: int, dist: float, lateral: float, rel: float = 0.0, rel_valid: bool = True) -> TrackedVehicle:
    return TrackedVehicle(
        track_id=track_id,
        xyxy=np.array([100, 100, 200, 200], dtype=np.float32),
        cls=2,
        conf=0.9,
        distance_m=dist,
        lateral_m=lateral,
        rel_speed_mps=rel,
        rel_speed_valid=rel_valid,
        method="ground_plane",
    )


def fresh_fix(speed: float = 20.0) -> GpsFix:
    now = time.monotonic()
    return GpsFix(
        valid=True, lat=40.0, lon=-74.0, speed_mps=speed, heading_deg=90.0,
        fix_quality=1, num_sats=8, hdop=1.0, altitude_m=5.0,
        utc_epoch_s=time.time(), t_mono=now, t_wall=time.time(),
    )


@pytest.fixture
def builder() -> ObservationBuilder:
    return ObservationBuilder(BuilderConfig())


def test_empty_scene_uses_spec_neutral_fallbacks(builder: ObservationBuilder) -> None:
    result = builder.build([], GpsFix(), time.monotonic())
    obs = result.obs
    # spec: neutral fallback values when nothing is sensed
    assert obs["leader_gap"] == math.inf
    assert obs["follower_gap"] == math.inf
    assert obs["nearby_av_count"] == 0
    assert obs["nearby_av_density"] == 0.0
    assert obs["nearby_av_mean_speed"] == builder.config.free_flow_speed_mps
    assert obs["nearby_av_lane_distribution"] == {}
    assert obs["cooperation"]["segment_target_speed"] == builder.config.free_flow_speed_mps
    assert obs["cooperation"]["merge_pressure"] == 0.0
    assert obs["cooperation"]["downstream_congestion_estimate"] == 0.0
    assert obs["active_vehicle_count_local"] == 0
    assert result.encoded.shape == (sim_contract.local_obs_dim(),)
    assert result.field_sources["leader_gap"] == "fallback_neutral"
    assert result.diagnostics["gps_fresh"] is False


def test_leader_selection_and_lane_split(builder: ObservationBuilder) -> None:
    vehicles = [
        make_vehicle(1, 60.0, 0.2),    # ego lane, far
        make_vehicle(2, 35.0, -0.4, rel=-2.0),  # ego lane, near -> leader
        make_vehicle(3, 25.0, -3.6),   # left lane
        make_vehicle(4, 50.0, 3.9),    # right lane
    ]
    result = builder.build(vehicles, fresh_fix(20.0), time.monotonic())
    obs = result.obs
    assert obs["leader_gap"] == 35.0
    assert obs["leader_relative_speed"] == -2.0
    assert obs["left_lane_front_gap"] == 25.0
    assert obs["right_lane_front_gap"] == 50.0
    assert obs["target_lane_front_gap"] == obs["leader_gap"]
    assert obs["ego_headway_s"] == pytest.approx(35.0 / 20.0)
    assert result.field_sources["leader_gap"] == "measured"


def test_density_and_bins_use_sim_formula(builder: ObservationBuilder) -> None:
    # 3 forward vehicles in 80 m, symmetrized to 6 over +-80 m
    vehicles = [make_vehicle(i, 20.0 + i * 10, 0.0 if i == 0 else (-3.7 if i == 1 else 3.7)) for i in range(3)]
    result = builder.build(vehicles, fresh_fix(20.0), time.monotonic())
    obs = result.obs
    assert obs["active_vehicle_count_local"] == 6
    expected_density = 6 / (2 * 80.0 / 1000.0)  # 37.5 veh/km
    assert result.diagnostics["density_veh_per_km"] == pytest.approx(expected_density, abs=0.01)
    # edges (12, 30) -> 37.5 lands in bin 2
    assert obs["local_density_bin"] == 2
    assert obs["local_density_bin"] == sim_contract.bin_index(
        expected_density, builder.config.density_bin_edges_veh_per_km
    )


def test_queue_estimate_counts_slow_vehicles(builder: ObservationBuilder) -> None:
    # ego 6 m/s; leader rel -3 -> abs 3 m/s < 5 -> queued
    vehicles = [make_vehicle(1, 30.0, 0.0, rel=-3.0), make_vehicle(2, 50.0, 0.0, rel=10.0)]
    result = builder.build(vehicles, fresh_fix(6.0), time.monotonic())
    assert result.obs["local_queue_estimate"] == 2  # 1 slow, symmetrized x2


def test_gps_speed_hold_on_staleness(builder: ObservationBuilder) -> None:
    t = time.monotonic()
    builder.build([], fresh_fix(22.0), t)
    stale = GpsFix(valid=True, speed_mps=99.0, t_mono=t - 10.0, t_wall=time.time())
    result = builder.build([], stale, t + 0.1)
    # stale fix is not trusted; last fresh speed is held
    assert result.obs["ego_speed"] == 22.0
    assert result.field_sources["ego_speed"] == "fallback_neutral"


def test_peers_populate_cooperation_fields(builder: ObservationBuilder) -> None:
    peers = [
        PeerState(peer_id="a", distance_m=80.0, speed_mps=24.0, lane_id=1),
        PeerState(peer_id="b", distance_m=120.0, speed_mps=26.0, lane_id=2),
    ]
    result = builder.build([], fresh_fix(20.0), time.monotonic(), peers)
    obs = result.obs
    assert obs["nearby_av_count"] == 2
    assert obs["nearby_av_mean_speed"] == pytest.approx(25.0)
    assert obs["cooperation"]["segment_target_speed"] == pytest.approx(25.0)
    assert obs["nearby_av_lane_distribution"] == {"1": 0.5, "2": 0.5}


def test_uncongested_low_speed_flag_mirrors_etiquette(builder: ObservationBuilder) -> None:
    # empty road (density 0 < 12), speed 10 < 30 - 8 -> flag on
    result = builder.build([], fresh_fix(10.0), time.monotonic())
    assert result.obs["uncongested_low_speed_flag"] is True
    result = builder.build([], fresh_fix(28.0), time.monotonic())
    assert result.obs["uncongested_low_speed_flag"] is False


def test_target_headway_feedback(builder: ObservationBuilder) -> None:
    builder.set_target_headway(2.2)
    result = builder.build([], fresh_fix(20.0), time.monotonic())
    assert result.obs["target_headway_s"] == 2.2
