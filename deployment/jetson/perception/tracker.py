"""SORT-lite multi-object tracker.

Latency-first choice: greedy IoU association with a constant-velocity
prediction on (cx, cy, w, h). No Kalman covariance bookkeeping, no
Hungarian solver, no appearance features - with the ~3-15 vehicles a
windshield camera sees, greedy matching on predicted boxes is stable
and costs well under a millisecond. ByteTrack (the plan's suggestion)
is an upgrade path if ID switches become a problem on real footage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from perception.detector import Detection


@dataclass
class Track:
    track_id: int
    xyxy: np.ndarray
    cls: int
    conf: float
    hits: int = 1
    misses: int = 0
    t_last: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    # velocity is d(cx, cy, w, h)/dt, EMA-smoothed

    @property
    def cxcywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float32)

    def predicted_xyxy(self, t_now: float) -> np.ndarray:
        dt = max(0.0, t_now - self.t_last)
        cx, cy, w, h = self.cxcywh + self.velocity * dt
        w, h = max(w, 1.0), max(h, 1.0)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) -> (N,M) IoU."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


class IouTracker:
    def __init__(
        self,
        iou_match_threshold: float = 0.3,
        max_age_frames: int = 10,
        min_hits: int = 2,
        velocity_alpha: float = 0.4,
    ) -> None:
        self.iou_match_threshold = iou_match_threshold
        self.max_age_frames = max_age_frames
        self.min_hits = min_hits
        self.velocity_alpha = velocity_alpha
        self._tracks: list[Track] = []
        self._next_id = 1
        self.id_switch_guard = 0  # diagnostic: tracks dropped at max age

    def update(self, detections: list[Detection], t_mono: float) -> list[Track]:
        """Associate detections to tracks; return confirmed, currently-visible tracks."""
        if self._tracks and detections:
            det_boxes = np.stack([d.xyxy for d in detections])
            trk_boxes = np.stack([t.predicted_xyxy(t_mono) for t in self._tracks])
            iou = _iou_matrix(det_boxes, trk_boxes)
            matched_det: set[int] = set()
            matched_trk: set[int] = set()
            order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
            for di, ti in order:
                if iou[di, ti] < self.iou_match_threshold:
                    break
                if di in matched_det or ti in matched_trk:
                    continue
                matched_det.add(int(di))
                matched_trk.add(int(ti))
                self._update_track(self._tracks[ti], detections[di], t_mono)
        else:
            matched_det, matched_trk = set(), set()

        for ti, track in enumerate(self._tracks):
            if ti not in matched_trk:
                track.misses += 1
        for di, det in enumerate(detections):
            if di not in matched_det:
                self._tracks.append(
                    Track(
                        track_id=self._next_id,
                        xyxy=det.xyxy.copy(),
                        cls=det.cls,
                        conf=det.conf,
                        t_last=t_mono,
                    )
                )
                self._next_id += 1

        survivors = []
        for track in self._tracks:
            if track.misses > self.max_age_frames:
                self.id_switch_guard += 1
                continue
            survivors.append(track)
        self._tracks = survivors

        return [
            t for t in self._tracks if t.misses == 0 and t.hits >= self.min_hits
        ]

    def _update_track(self, track: Track, det: Detection, t_mono: float) -> None:
        dt = max(1e-3, t_mono - track.t_last)
        new_state = np.array(
            [
                (det.xyxy[0] + det.xyxy[2]) / 2,
                (det.xyxy[1] + det.xyxy[3]) / 2,
                det.xyxy[2] - det.xyxy[0],
                det.xyxy[3] - det.xyxy[1],
            ],
            dtype=np.float32,
        )
        instantaneous = (new_state - track.cxcywh) / dt
        track.velocity = (
            self.velocity_alpha * instantaneous + (1 - self.velocity_alpha) * track.velocity
        )
        track.xyxy = det.xyxy.copy()
        track.cls = det.cls
        track.conf = det.conf
        track.hits += 1
        track.misses = 0
        track.t_last = t_mono

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)
