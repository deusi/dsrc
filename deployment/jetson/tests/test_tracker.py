from __future__ import annotations

import numpy as np

from perception.detector import Detection
from perception.tracker import IouTracker


def det(x1: float, y1: float, x2: float, y2: float, cls: int = 2) -> Detection:
    return Detection(xyxy=np.array([x1, y1, x2, y2], dtype=np.float32), conf=0.9, cls=cls)


def test_track_id_stable_across_frames() -> None:
    tracker = IouTracker(min_hits=2)
    assert tracker.update([det(100, 100, 200, 200)], t_mono=0.0) == []  # below min_hits
    [track] = tracker.update([det(105, 100, 205, 200)], t_mono=0.033)
    first_id = track.track_id
    [track] = tracker.update([det(110, 100, 210, 200)], t_mono=0.066)
    assert track.track_id == first_id


def test_two_vehicles_get_distinct_ids() -> None:
    tracker = IouTracker(min_hits=1)
    tracks = tracker.update([det(100, 100, 200, 200), det(400, 100, 500, 200)], t_mono=0.0)
    assert len({t.track_id for t in tracks}) == 2


def test_track_dropped_after_max_age() -> None:
    tracker = IouTracker(min_hits=1, max_age_frames=3)
    tracker.update([det(100, 100, 200, 200)], t_mono=0.0)
    for i in range(5):
        tracker.update([], t_mono=0.033 * (i + 1))
    assert tracker.active_track_count == 0


def test_missed_track_not_reported_but_kept() -> None:
    tracker = IouTracker(min_hits=1, max_age_frames=10)
    tracker.update([det(100, 100, 200, 200)], t_mono=0.0)
    assert tracker.update([], t_mono=0.033) == []  # missed -> not visible
    assert tracker.active_track_count == 1        # but retained
    [track] = tracker.update([det(102, 100, 202, 200)], t_mono=0.066)
    assert track.misses == 0


def test_velocity_prediction_carries_fast_mover() -> None:
    tracker = IouTracker(min_hits=1, iou_match_threshold=0.2)
    base = None
    # vehicle sliding 40 px/frame; prediction must keep association
    for i in range(6):
        x = 100 + 40 * i
        [track] = tracker.update([det(x, 100, x + 120, 200)], t_mono=0.033 * i)
        if base is None:
            base = track.track_id
        assert track.track_id == base
