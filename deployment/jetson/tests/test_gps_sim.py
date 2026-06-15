"""GPS simulator: dead-reckoning, profile semantics, fix emission."""

import json
import math
import time

import pytest

from sensors.gps_sim import M_PER_DEG_LAT, GpsSimProfile, GpsSimulator, SimulatedGps


def east_north_m(profile: GpsSimProfile, state: dict) -> tuple[float, float]:
    east = (state["lon"] - profile.start_lon) * M_PER_DEG_LAT * math.cos(
        math.radians(profile.start_lat)
    )
    north = (state["lat"] - profile.start_lat) * M_PER_DEG_LAT
    return east, north


def test_const_spec_dead_reckons_east():
    profile = GpsSimProfile.from_spec("const:20@40.0,-75.0,90")
    sim = GpsSimulator(profile)
    east, north = east_north_m(profile, sim.state_at(100.0))
    assert east == pytest.approx(2000.0, rel=0.01)
    assert abs(north) < 1.0
    assert sim.speed_at(100.0) == 20.0


def test_piecewise_interpolation_and_hold_past_end():
    profile = GpsSimProfile.from_spec(
        {"start": {"lat": 0.0, "lon": 0.0, "heading_deg": 0.0},
         "speed_profile_mps": [[0, 10], [10, 30]]}
    )
    sim = GpsSimulator(profile)
    assert sim.speed_at(5.0) == pytest.approx(20.0)
    assert sim.speed_at(50.0) == pytest.approx(30.0)  # hold last value
    _, north10 = east_north_m(profile, sim.state_at(10.0))
    _, north20 = east_north_m(profile, sim.state_at(20.0))
    assert north10 == pytest.approx(200.0, rel=0.01)   # trapezoid of the ramp
    assert north20 == pytest.approx(500.0, rel=0.01)   # + 10 s straight at 30


def test_loop_carries_position_across_cycles():
    profile = GpsSimProfile.from_spec(
        {"start": {"lat": 0.0, "lon": 0.0, "heading_deg": 0.0},
         "speed_profile_mps": [[0, 10], [10, 10]], "loop": True}
    )
    sim = GpsSimulator(profile)
    _, north = east_north_m(profile, sim.state_at(25.0))
    assert north == pytest.approx(250.0, rel=0.01)
    assert sim.speed_at(25.0) == pytest.approx(10.0)


def test_dropouts_and_cold_start_suppress_fixes():
    profile = GpsSimProfile.from_spec(
        {"speed_profile_mps": 25, "dropouts_s": [[5, 8]], "cold_start_s": 1.0}
    )
    sim = GpsSimulator(profile)
    assert sim.fix_at(0.5, 0.0, 0.0) is None        # cold start
    assert sim.fix_at(6.0, 0.0, 0.0) is None        # dropout
    assert sim.fix_at(8.0, 0.0, 0.0) is not None    # window is half-open
    assert sim.fix_at(3.0, 0.0, 0.0) is not None


def test_noise_is_seeded_per_emission():
    profile = GpsSimProfile.from_spec(
        {"speed_profile_mps": 25, "noise": {"speed_std_mps": 0.5, "pos_std_m": 2.0},
         "seed": 7}
    )
    sim = GpsSimulator(profile)
    a1, a2 = sim.fix_at(4.0, 0.0, 0.0), sim.fix_at(4.0, 0.0, 0.0)
    assert (a1.lat, a1.lon, a1.speed_mps) == (a2.lat, a2.lon, a2.speed_mps)
    b = sim.fix_at(4.2, 0.0, 0.0)  # next emission index at 5 Hz
    assert (a1.lat, a1.speed_mps) != (b.lat, b.speed_mps)


def test_fix_fields_are_plausible():
    sim = GpsSimulator(GpsSimProfile.from_spec("const:25"))
    fix = sim.fix_at(10.0, 123.0, 456.0)
    assert fix.valid and fix.fix_quality == 1 and fix.num_sats == 10
    assert math.isfinite(fix.lat) and math.isfinite(fix.lon)
    assert fix.speed_mps == pytest.approx(25.0)
    assert fix.t_mono == 123.0 and fix.t_wall == 456.0 and fix.utc_epoch_s == 456.0


def test_profile_roundtrip_through_to_dict():
    original = GpsSimProfile.from_spec(
        {"start": {"lat": 1.0, "lon": 2.0, "heading_deg": 3.0},
         "speed_profile_mps": [[0, 5], [10, 9]],
         "dropouts_s": [[2, 4]], "noise": {"speed_std_mps": 0.1, "pos_std_m": 1.0},
         "seed": 5, "loop": True}
    )
    restored = GpsSimProfile.from_spec(original.to_dict())
    assert restored == original


def test_from_spec_reads_json_file(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"speed_profile_mps": 17.5, "rate_hz": 2}))
    profile = GpsSimProfile.from_spec(str(path))
    assert profile.speed_points == [(0.0, 17.5)]
    assert profile.rate_hz == 2.0


def test_threaded_publisher_emits_at_rate():
    gps = SimulatedGps({"speed_profile_mps": 20, "rate_hz": 10})
    gps.start()
    try:
        time.sleep(1.05)
    finally:
        gps.stop()
    fix = gps.latest()
    assert fix.valid and fix.speed_mps == pytest.approx(20.0)
    assert 8 <= gps.diagnostics.sentences_parsed <= 13
    assert gps.diagnostics.observed_rate_hz() == pytest.approx(10.0, rel=0.3)
    assert not gps.is_stale()
