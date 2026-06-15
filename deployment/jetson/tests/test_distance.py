from __future__ import annotations

import numpy as np
import pytest

from perception.distance import DistanceEstimator
from perception.tracker import Track

FX, CX, HORIZON, CAM_H = 800.0, 640.0, 360.0, 1.25


def track_at(z_m: float, x_m: float = 0.0, track_id: int = 1, veh_width_m: float = 1.8) -> Track:
    """Project a vehicle at (z, x) through the same pinhole the estimator inverts."""
    w_px = FX * veh_width_m / z_m
    h_px = 0.85 * w_px
    u = CX + x_m * FX / z_m
    v_bottom = HORIZON + CAM_H * FX / z_m
    return Track(
        track_id=track_id,
        xyxy=np.array([u - w_px / 2, v_bottom - h_px, u + w_px / 2, v_bottom], dtype=np.float32),
        cls=2,
        conf=0.9,
    )


def make_estimator(method: str = "ground_plane", ema_alpha: float = 1.0) -> DistanceEstimator:
    return DistanceEstimator(
        fx_px=FX, cx_px=CX, horizon_y_px=HORIZON, camera_height_m=CAM_H,
        method=method, ema_alpha=ema_alpha,
    )


@pytest.mark.parametrize("z", [10.0, 25.0, 60.0, 100.0])
def test_ground_plane_inverts_projection(z: float) -> None:
    est = make_estimator("ground_plane")
    [v] = est.update([track_at(z)], t_mono=0.0)
    assert v.distance_m == pytest.approx(z, rel=0.02)
    assert v.method == "ground_plane"


@pytest.mark.parametrize("z", [10.0, 25.0, 60.0])
def test_width_prior_inverts_projection(z: float) -> None:
    est = make_estimator("width_prior")
    [v] = est.update([track_at(z)], t_mono=0.0)
    assert v.distance_m == pytest.approx(z, rel=0.02)
    assert v.method == "width_prior"


def test_lateral_offset_recovered() -> None:
    est = make_estimator()
    [v] = est.update([track_at(40.0, x_m=3.7)], t_mono=0.0)
    assert v.lateral_m == pytest.approx(3.7, rel=0.05)
    [v] = est.update([track_at(40.0, x_m=-3.7, track_id=2)], t_mono=0.0)
    assert v.lateral_m == pytest.approx(-3.7, rel=0.05)


def test_relative_speed_slope() -> None:
    est = make_estimator(ema_alpha=1.0)  # no smoothing: isolate the slope
    z = 50.0
    result = []
    for i in range(8):
        t = i * 0.1
        result = est.update([track_at(z)], t_mono=t)
        z -= 0.2  # -2 m/s closing
    [v] = result
    assert v.rel_speed_valid
    assert v.rel_speed_mps == pytest.approx(-2.0, abs=0.3)


def test_rel_speed_invalid_until_window_spans() -> None:
    est = make_estimator()
    [v] = est.update([track_at(30.0)], t_mono=0.0)
    assert not v.rel_speed_valid
    [v] = est.update([track_at(30.0)], t_mono=0.05)
    assert not v.rel_speed_valid  # span 0.05 s < 0.2 s minimum


def test_ema_smooths_jumps() -> None:
    est = make_estimator(ema_alpha=0.5)
    est.update([track_at(40.0)], t_mono=0.0)
    [v] = est.update([track_at(48.0)], t_mono=0.1)  # 8 m jump
    assert 40.0 < v.distance_m < 48.0  # smoothed, not snapped


def test_above_horizon_falls_back_to_width_prior() -> None:
    est = make_estimator("ground_plane")
    track = track_at(40.0)
    track.xyxy[3] = HORIZON - 5  # bottom above horizon (e.g. uphill)
    [v] = est.update([track], t_mono=0.0)
    assert v.method == "width_prior"


def test_contact_cutoff_switches_to_width_prior() -> None:
    # a vehicle whose bbox bottom lands in the hood zone must not use the
    # (occluded) contact row; the width prior keeps the estimate sane
    est = DistanceEstimator(
        fx_px=FX, cx_px=CX, horizon_y_px=HORIZON, camera_height_m=CAM_H,
        method="ground_plane", ema_alpha=1.0, contact_cutoff_y_px=650.0,
    )
    near = track_at(3.0)        # v_bottom = 360 + 1000/3 = 693 >= cutoff
    far = track_at(20.0)        # v_bottom = 410 < cutoff
    [v_near] = est.update([near], t_mono=0.0)
    assert v_near.method == "width_prior"
    assert v_near.distance_m == pytest.approx(3.0, rel=0.05)
    [v_far] = est.update([far], t_mono=1.0)
    assert v_far.method == "ground_plane"
