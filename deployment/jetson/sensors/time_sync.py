"""Time bases for the deployment.

Two clocks are used everywhere:
  - t_mono: time.monotonic(), for all latency math and sensor freshness.
    Never compare monotonic values across processes or reboots.
  - t_wall: time.time() (UTC epoch), for log records and cross-device
    correlation (e.g. matching GPS UTC timestamps offline).

GPS sentences carry UTC; ``GpsUtcOffsetTracker`` keeps a running estimate
of (wall clock - GPS UTC) so logs can be corrected offline if the Jetson
RTC drifts while off-network in the car.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


def now_mono() -> float:
    return time.monotonic()


def now_wall() -> float:
    return time.time()


@dataclass
class Stamp:
    t_mono: float
    t_wall: float

    @classmethod
    def now(cls) -> "Stamp":
        return cls(t_mono=now_mono(), t_wall=now_wall())


class GpsUtcOffsetTracker:
    """EMA of (system wall clock - GPS UTC) in seconds."""

    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha
        self._offset_s: float | None = None

    def update(self, gps_utc_epoch_s: float, wall_s: float) -> None:
        sample = wall_s - gps_utc_epoch_s
        if self._offset_s is None:
            self._offset_s = sample
        else:
            self._offset_s += self._alpha * (sample - self._offset_s)

    @property
    def offset_s(self) -> float | None:
        return self._offset_s
