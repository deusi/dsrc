"""Simulated GPS source for desk testing against dashcam video.

Drop-in replacement for sensors.gps_reader.GpsReader (same consumer
surface: start/stop/latest/is_stale/diagnostics), driven by a scripted
speed/heading profile instead of a serial port. Position is
dead-reckoned from the profile on a local flat-earth tangent, fixes are
emitted at GPS rate (default 5 Hz, like the real u-blox after
UBX-CFG-RATE) with optional measurement noise and scripted dropout
windows - dropouts simply stop publishing, so the observation builder's
hold-on-stale path is exercised exactly as with a real antenna outage.

The core (GpsSimulator) is pure and deterministic: state_at(t) /
fix_at(t) never touch the clock, so tests and the offline evaluator can
query ground truth directly. SimulatedGps wraps it in the publisher
thread for live runs.

Profile JSON (also embedded as "gps" inside scenario files):
  {
    "start": {"lat": 39.0339, "lon": -77.1773, "heading_deg": 105.0},
    "rate_hz": 5,
    "speed_profile_mps": [[0, 24], [60, 27], [120, 20]],   // or a number
    "heading_profile_deg": [[0, 105]],                     // optional
    "dropouts_s": [[120, 124]],                            // optional
    "cold_start_s": 0.0,                                   // optional
    "noise": {"speed_std_mps": 0.1, "pos_std_m": 1.5},     // optional
    "seed": 0,                                             // optional
    "loop": false                                          // optional
  }
Profiles interpolate piecewise-linearly and hold the last value past the
end (or wrap, with position carried across cycles, if loop is true).
Shorthand spec strings: "const:25" (25 m/s forever at a default start)
or "const:25@39.03,-77.18,105" (lat,lon,heading).
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sensors.gps_reader import GpsDiagnostics, GpsFix
from sensors.time_sync import now_mono, now_wall

M_PER_DEG_LAT = 111_320.0
_DEFAULT_START = (39.0339, -77.1773, 105.0)  # I-495 @ I-270 spur, eastbound
_INTEGRATION_DT_S = 0.05


@dataclass
class GpsSimProfile:
    start_lat: float = _DEFAULT_START[0]
    start_lon: float = _DEFAULT_START[1]
    start_heading_deg: float = _DEFAULT_START[2]
    rate_hz: float = 5.0
    speed_points: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 25.0)])
    heading_points: list[tuple[float, float]] | None = None  # None -> constant start heading
    dropouts_s: list[tuple[float, float]] = field(default_factory=list)
    cold_start_s: float = 0.0
    speed_std_mps: float = 0.0
    pos_std_m: float = 0.0
    seed: int = 0
    loop: bool = False

    @property
    def duration_s(self) -> float:
        return max(self.speed_points[-1][0], 1e-9)

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable form, logged into metadata.jsonl for the evaluator."""
        return {
            "start": {
                "lat": self.start_lat,
                "lon": self.start_lon,
                "heading_deg": self.start_heading_deg,
            },
            "rate_hz": self.rate_hz,
            "speed_profile_mps": [[t, v] for t, v in self.speed_points],
            "heading_profile_deg": (
                [[t, h] for t, h in self.heading_points] if self.heading_points else None
            ),
            "dropouts_s": [[a, b] for a, b in self.dropouts_s],
            "cold_start_s": self.cold_start_s,
            "noise": {"speed_std_mps": self.speed_std_mps, "pos_std_m": self.pos_std_m},
            "seed": self.seed,
            "loop": self.loop,
        }

    @classmethod
    def from_spec(cls, spec: str | dict[str, Any]) -> "GpsSimProfile":
        if isinstance(spec, str):
            if spec.startswith("const:"):
                return cls._from_const(spec)
            with open(Path(spec).expanduser()) as f:
                spec = json.load(f)
        return cls._from_dict(spec)

    @classmethod
    def _from_const(cls, spec: str) -> "GpsSimProfile":
        body = spec[len("const:"):]
        lat, lon, heading = _DEFAULT_START
        if "@" in body:
            body, at = body.split("@", 1)
            lat, lon, heading = (float(x) for x in at.split(","))
        return cls(
            start_lat=lat,
            start_lon=lon,
            start_heading_deg=heading,
            speed_points=[(0.0, float(body))],
        )

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> "GpsSimProfile":
        start = raw.get("start", {})
        speed = raw.get("speed_profile_mps", 25.0)
        if isinstance(speed, (int, float)):
            speed_points = [(0.0, float(speed))]
        else:
            speed_points = [(float(t), float(v)) for t, v in speed]
        heading = raw.get("heading_profile_deg")
        noise = raw.get("noise", {})
        return cls(
            start_lat=float(start.get("lat", _DEFAULT_START[0])),
            start_lon=float(start.get("lon", _DEFAULT_START[1])),
            start_heading_deg=float(start.get("heading_deg", _DEFAULT_START[2])),
            rate_hz=float(raw.get("rate_hz", 5.0)),
            speed_points=speed_points,
            heading_points=(
                [(float(t), float(h)) for t, h in heading] if heading else None
            ),
            dropouts_s=[(float(a), float(b)) for a, b in raw.get("dropouts_s", [])],
            cold_start_s=float(raw.get("cold_start_s", 0.0)),
            speed_std_mps=float(noise.get("speed_std_mps", 0.0)),
            pos_std_m=float(noise.get("pos_std_m", 0.0)),
            seed=int(raw.get("seed", 0)),
            loop=bool(raw.get("loop", False)),
        )


class GpsSimulator:
    """Pure deterministic trajectory: profile time -> position/speed/fix."""

    def __init__(self, profile: GpsSimProfile) -> None:
        self.profile = profile
        p = profile
        self._speed_t = np.array([t for t, _ in p.speed_points])
        self._speed_v = np.array([v for _, v in p.speed_points])
        if p.heading_points:
            self._head_t = np.array([t for t, _ in p.heading_points])
            self._head_v = np.array([h for _, h in p.heading_points])
        else:
            self._head_t = np.array([0.0])
            self._head_v = np.array([p.start_heading_deg])

        # cumulative north/east displacement on a fixed grid over one cycle
        grid = np.arange(0.0, p.duration_s + _INTEGRATION_DT_S, _INTEGRATION_DT_S)
        speeds = np.interp(grid, self._speed_t, self._speed_v)
        headings = np.radians(np.interp(grid, self._head_t, self._head_v))
        d_north = speeds * np.cos(headings) * _INTEGRATION_DT_S
        d_east = speeds * np.sin(headings) * _INTEGRATION_DT_S
        self._grid = grid
        self._cum_north = np.concatenate([[0.0], np.cumsum(d_north)[:-1]])
        self._cum_east = np.concatenate([[0.0], np.cumsum(d_east)[:-1]])
        self._cycle_north = float(self._cum_north[-1] + d_north[-1])
        self._cycle_east = float(self._cum_east[-1] + d_east[-1])

    # -- ground truth ----------------------------------------------------

    def speed_at(self, t_s: float) -> float:
        return float(np.interp(self._cycle_time(t_s), self._speed_t, self._speed_v))

    def heading_at(self, t_s: float) -> float:
        return float(np.interp(self._cycle_time(t_s), self._head_t, self._head_v))

    def state_at(self, t_s: float) -> dict[str, float]:
        """Noiseless (lat, lon, speed, heading) at elapsed profile time t_s."""
        p = self.profile
        t_cyc = self._cycle_time(t_s)
        north = float(np.interp(t_cyc, self._grid, self._cum_north))
        east = float(np.interp(t_cyc, self._grid, self._cum_east))
        if p.loop and t_s > p.duration_s:
            cycles = int(t_s // p.duration_s)
            north += cycles * self._cycle_north
            east += cycles * self._cycle_east
        elif not p.loop and t_s > p.duration_s:
            # hold final speed/heading, keep moving in a straight line
            extra = t_s - p.duration_s
            v = float(self._speed_v[-1])
            h = math.radians(float(self._head_v[-1]))
            north = float(np.interp(p.duration_s, self._grid, self._cum_north)) + v * math.cos(h) * extra
            east = float(np.interp(p.duration_s, self._grid, self._cum_east)) + v * math.sin(h) * extra
        lat = p.start_lat + north / M_PER_DEG_LAT
        lon = p.start_lon + east / (M_PER_DEG_LAT * math.cos(math.radians(p.start_lat)))
        return {
            "lat": lat,
            "lon": lon,
            "speed_mps": self.speed_at(t_s),
            "heading_deg": self.heading_at(t_s) % 360.0,
        }

    def _cycle_time(self, t_s: float) -> float:
        p = self.profile
        if p.loop and t_s > p.duration_s:
            return t_s % p.duration_s
        return min(t_s, p.duration_s) if p.loop else t_s  # np.interp clamps anyway

    # -- fix emission ------------------------------------------------------

    def in_dropout(self, t_s: float) -> bool:
        if t_s < self.profile.cold_start_s:
            return True
        t_cyc = self._cycle_time(t_s) if self.profile.loop else t_s
        return any(a <= t_cyc < b for a, b in self.profile.dropouts_s)

    def fix_at(self, t_s: float, t_mono: float, t_wall: float) -> GpsFix | None:
        """The fix a receiver would emit at elapsed time t_s, or None in a
        dropout. Noise is seeded per emission index, so identical t_s
        sequences reproduce identical fixes."""
        if self.in_dropout(t_s):
            return None
        p = self.profile
        state = self.state_at(t_s)
        rng = np.random.default_rng((p.seed, int(t_s * p.rate_hz)))
        speed = max(0.0, state["speed_mps"] + rng.normal(0.0, p.speed_std_mps))
        north_err = rng.normal(0.0, p.pos_std_m)
        east_err = rng.normal(0.0, p.pos_std_m)
        return GpsFix(
            valid=True,
            lat=state["lat"] + north_err / M_PER_DEG_LAT,
            lon=state["lon"]
            + east_err / (M_PER_DEG_LAT * math.cos(math.radians(p.start_lat))),
            speed_mps=speed,
            heading_deg=state["heading_deg"],
            fix_quality=1,
            num_sats=10,
            hdop=0.8,
            altitude_m=50.0,
            utc_epoch_s=t_wall,
            t_mono=t_mono,
            t_wall=t_wall,
        )


class SimulatedGps:
    """Threaded publisher with GpsReader's consumer interface."""

    def __init__(
        self,
        spec: str | dict[str, Any] | GpsSimProfile,
        stale_after_s: float = 2.0,
        raw_log_path: str | None = None,
    ) -> None:
        profile = spec if isinstance(spec, GpsSimProfile) else GpsSimProfile.from_spec(spec)
        self.sim = GpsSimulator(profile)
        self.stale_after_s = stale_after_s
        self.raw_log_path = raw_log_path
        self.diagnostics = GpsDiagnostics(port_open=True, rate_configured=True)
        self.start_mono: float | None = None
        self.start_wall: float | None = None
        self._fix = GpsFix()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._raw_log = None
        self._last_emit_mono: float | None = None

    # -- lifecycle (mirrors GpsReader) ------------------------------------

    def start(self) -> "SimulatedGps":
        self.start_mono = now_mono()
        self.start_wall = now_wall()
        if self.raw_log_path:
            self._raw_log = open(self.raw_log_path, "a", buffering=1)
            self._raw_log.write(
                f"# simulated gps, profile: {json.dumps(self.sim.profile.to_dict())}\n"
            )
            self.diagnostics.raw_log_path = self.raw_log_path
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="gps-sim", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._raw_log is not None:
            self._raw_log.close()

    def _loop(self) -> None:
        period = 1.0 / max(self.sim.profile.rate_hz, 0.1)
        emission = 0
        while not self._stop.is_set():
            target = self.start_mono + emission * period
            delay = target - now_mono()
            if delay > 0 and self._stop.wait(delay):
                return
            t_mono, t_wall = now_mono(), now_wall()
            elapsed = t_mono - self.start_mono
            fix = self.sim.fix_at(elapsed, t_mono, t_wall)
            if fix is not None:
                with self._lock:
                    self._fix = fix
                self.diagnostics.sentences_parsed += 1
                if self._last_emit_mono is not None:
                    self.diagnostics.recent_intervals_s.append(t_mono - self._last_emit_mono)
                    if len(self.diagnostics.recent_intervals_s) > 20:
                        self.diagnostics.recent_intervals_s.pop(0)
                self._last_emit_mono = t_mono
                if self._raw_log is not None:
                    self._raw_log.write(
                        f"{t_wall:.3f} SIM {fix.lat:.6f} {fix.lon:.6f} "
                        f"{fix.speed_mps:.2f} {fix.heading_deg:.1f}\n"
                    )
            emission += 1

    # -- consumers ---------------------------------------------------------

    def latest(self) -> GpsFix:
        with self._lock:
            return self._fix

    def is_stale(self, t_mono_now: float | None = None) -> bool:
        fix = self.latest()
        return fix.age_s(t_mono_now if t_mono_now is not None else now_mono()) > self.stale_after_s
