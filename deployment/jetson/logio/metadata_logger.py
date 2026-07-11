"""Per-run logging: JSONL metadata, UDP telemetry, system stats.

Directory layout per run (under paths.log_dir):
  run_YYYYmmdd_HHMMSS/
    metadata.jsonl    one record per pipeline tick (+ system stat records)
    video.avi         raw frames as processed (if logio.video)
    nmea.log          raw NMEA sentences with wall timestamps (if logio.nmea)
    run_config.yaml   the exact config used
    summary.json      end-of-run aggregates

Notes:
  - JSONL is written with Python's default JSON, which emits Infinity/NaN
    literals for the sim's "no vehicle" gaps. Read logs back with Python
    json / pandas (both accept them), not strict-JSON parsers.
  - The writer runs on its own thread with an unbounded-ish queue; if the
    SD card stalls, ticks are never blocked - records are dropped past
    50k pending and counted in drop stats.

(This package is named logio, not logging as in plan_deployment.md,
because a sibling directory named ``logging`` shadows the stdlib module
for every script in this folder.)
"""

from __future__ import annotations

import datetime
import json
import queue
import shutil
import socket
import threading
from pathlib import Path
from typing import Any


def make_run_dir(log_root: str) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(log_root).expanduser() / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class MetadataLogger:
    def __init__(self, run_dir: Path, config_path: str | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "metadata.jsonl"
        if config_path:
            shutil.copy(config_path, self.run_dir / "run_config.yaml")
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=50_000)
        self.dropped_records = 0
        self._file = open(self.path, "a", buffering=1024 * 1024)
        self._thread = threading.Thread(target=self._loop, name="metadata-log", daemon=True)
        self._thread.start()

    def write(self, record: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(json.dumps(record, default=_json_default))
        except queue.Full:
            self.dropped_records += 1

    def write_summary(self, summary: dict[str, Any]) -> None:
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=_json_default)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        self._file.flush()
        self._file.close()

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            self._file.write(item + "\n")


def _json_default(value: Any) -> Any:
    import numpy as np

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class TelemetrySender:
    """Fire-and-forget UDP JSON tick summaries (local dashboard / dev laptop)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 47900) -> None:
        self.addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, record: dict[str, Any]) -> None:
        try:
            payload = json.dumps(record, default=_json_default).encode()
            if len(payload) < 60_000:
                self._sock.sendto(payload, self.addr)
        except OSError:
            pass  # telemetry must never break the loop

    def close(self) -> None:
        self._sock.close()


class SystemStatsSampler:
    """Optional jtop-based power/utilization sampling into the metadata log.

    Requires the jetson-stats service; degrades silently to a no-op if
    jtop is unavailable so desk machines and CI behave.
    """

    def __init__(self, logger: MetadataLogger, interval_s: float = 5.0) -> None:
        self.logger = logger
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = False
        try:
            from jtop import jtop  # noqa: F401

            self.available = True
        except Exception:
            return

    def start(self) -> "SystemStatsSampler":
        if not self.available:
            return self
        self._thread = threading.Thread(target=self._loop, name="jtop-stats", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        from jtop import jtop

        try:
            with jtop(interval=self.interval_s) as j:
                while not self._stop.is_set() and j.ok():
                    stats = j.stats
                    self.logger.write(
                        {
                            "type": "system",
                            "t_wall": stats.get("time").timestamp() if stats.get("time") else None,
                            "cpu_pct": _mean_cpu(stats),
                            "gpu_pct": stats.get("GPU"),
                            "ram": stats.get("RAM"),
                            "temp_cpu_c": stats.get("Temp cpu"),
                            "temp_gpu_c": stats.get("Temp gpu"),
                            "power_mw": stats.get("Power TOT"),
                            "nvp_mode": stats.get("nvp model"),
                        }
                    )
                    self._stop.wait(self.interval_s)
        except Exception as exc:  # jtop service quirks must not kill the run
            self.logger.write({"type": "system_error", "error": str(exc)})


def _mean_cpu(stats: dict[str, Any]) -> float | None:
    cores = [v for k, v in stats.items() if k.startswith("CPU") and isinstance(v, (int, float))]
    return round(sum(cores) / len(cores), 1) if cores else None
