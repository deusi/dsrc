"""Monocular distance and relative speed from tracked bounding boxes.

Latency-first deviation from plan_deployment.md: the plan suggests a
lightweight depth network (Depth Anything small). On an Orin Nano that
would roughly double the GPU budget per frame, so v0 uses closed-form
pinhole geometry instead (~0.1 ms for all tracks). A depth model can be
slotted in later behind this same interface and A/B-benchmarked with
bench_latency.py.

Two estimators, chosen by config:
  ground_plane (default): distance from the bbox bottom edge's image row,
      assuming a flat road:  Z = camera_height * fx / (v_bottom - v_horizon)
      Needs horizon_y_px + camera_height_m calibration; degrades on
      hills/dips. Independent of vehicle size.
  width_prior: Z = fx * real_width(class) / bbox_width_px. No horizon
      needed, but biased by vehicle-width variance (+-15%).
The ground-plane path falls back to width_prior for boxes at/above the
horizon (e.g. trucks seen uphill).

Relative speed is the least-squares slope of smoothed distance over a
short window. The sign convention matches the sim:
  leader_relative_speed = leader_speed - ego_speed = d(gap)/dt
so a closing leader gives a negative value.

Lateral offset comes from the same pinhole model:
  X = (u_center - cx) * Z / fx   (X > 0 means right of camera axis)
which the observation builder turns into lane assignments.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from perception.tracker import Track


@dataclass
class TrackedVehicle:
    track_id: int
    xyxy: np.ndarray
    cls: int
    conf: float
    distance_m: float          # smoothed longitudinal distance
    lateral_m: float           # + right of camera axis
    rel_speed_mps: float       # d(distance)/dt; sim leader_relative_speed convention
    rel_speed_valid: bool      # False until the window has enough span
    method: str                # ground_plane | width_prior


class _TrackDistanceState:
    __slots__ = ("ema", "history", "t_seen")

    def __init__(self) -> None:
        self.ema: float | None = None
        self.history: deque[tuple[float, float]] = deque(maxlen=32)
        self.t_seen = 0.0


class DistanceEstimator:
    def __init__(
        self,
        fx_px: float,
        cx_px: float,
        horizon_y_px: float,
        camera_height_m: float,
        method: str = "ground_plane",
        ema_alpha: float = 0.45,
        # window must span > rel_speed_min_span_s at camera rate:
        # 8 samples at 30 fps = 0.233 s > 0.2 s
        rel_speed_window: int = 8,
        rel_speed_min_span_s: float = 0.2,
        class_widths_m: dict[int, float] | None = None,
        max_range_m: float = 120.0,
        # bbox bottoms at/below this row don't show the true road contact
        # point (ego hood occludes it, or the box is clipped by the frame
        # edge) -> ground_plane is invalid there, use the width prior.
        # Typically the hood line, else frame height minus a margin.
        contact_cutoff_y_px: float | None = None,
    ) -> None:
        if method not in ("ground_plane", "width_prior"):
            raise ValueError(f"unknown distance method '{method}'")
        self.fx = fx_px
        self.cx = cx_px
        self.horizon_y = horizon_y_px
        self.camera_height = camera_height_m
        self.method = method
        self.ema_alpha = ema_alpha
        self.rel_speed_window = rel_speed_window
        self.rel_speed_min_span_s = rel_speed_min_span_s
        self.class_widths = class_widths_m or {2: 1.8, 3: 0.8, 5: 2.55, 7: 2.5}
        self.max_range = max_range_m
        self.contact_cutoff_y = contact_cutoff_y_px
        self._state: dict[int, _TrackDistanceState] = {}

    def update(self, tracks: list[Track], t_mono: float) -> list[TrackedVehicle]:
        results: list[TrackedVehicle] = []
        for track in tracks:
            raw, method = self._estimate_raw(track)
            if raw is None:
                continue
            state = self._state.setdefault(track.track_id, _TrackDistanceState())
            state.t_seen = t_mono
            if state.ema is None:
                state.ema = raw
            else:
                state.ema += self.ema_alpha * (raw - state.ema)
            state.history.append((t_mono, state.ema))
            rel_speed, rel_valid = self._slope(state.history)

            x1, _, x2, _ = track.xyxy
            u_center = (x1 + x2) / 2.0
            lateral = (u_center - self.cx) * state.ema / self.fx

            results.append(
                TrackedVehicle(
                    track_id=track.track_id,
                    xyxy=track.xyxy,
                    cls=track.cls,
                    conf=track.conf,
                    distance_m=float(state.ema),
                    lateral_m=float(lateral),
                    rel_speed_mps=float(rel_speed),
                    rel_speed_valid=rel_valid,
                    method=method,
                )
            )
        self._prune(t_mono)
        return results

    def _estimate_raw(self, track: Track) -> tuple[float | None, str]:
        x1, y1, x2, y2 = track.xyxy
        if self.method == "ground_plane":
            contact_visible = self.contact_cutoff_y is None or y2 < self.contact_cutoff_y
            dv = y2 - self.horizon_y
            if contact_visible and dv > 2.0:  # below the horizon -> on the road plane
                z = self.camera_height * self.fx / dv
                if 1.0 <= z <= self.max_range:
                    return float(z), "ground_plane"
            # contact occluded/clipped, above horizon, or out of range:
            # fall through to the width prior
        width_px = max(x2 - x1, 1.0)
        real_width = self.class_widths.get(track.cls, 1.8)
        z = self.fx * real_width / width_px
        if 1.0 <= z <= self.max_range:
            return float(z), "width_prior"
        return None, "none"

    def _slope(self, history: deque[tuple[float, float]]) -> tuple[float, bool]:
        if len(history) < 3:
            return 0.0, False
        pts = list(history)[-self.rel_speed_window :]
        t = np.array([p[0] for p in pts])
        z = np.array([p[1] for p in pts])
        span = t[-1] - t[0]
        if span < self.rel_speed_min_span_s:
            return 0.0, False
        t = t - t.mean()
        slope = float((t * (z - z.mean())).sum() / max((t * t).sum(), 1e-9))
        return slope, True

    def _prune(self, t_mono: float, ttl_s: float = 2.0) -> None:
        dead = [tid for tid, s in self._state.items() if t_mono - s.t_seen > ttl_s]
        for tid in dead:
            del self._state[tid]
