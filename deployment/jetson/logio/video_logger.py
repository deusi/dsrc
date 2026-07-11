"""Raw frame recording for offline replay.

Writes the frames exactly as the pipeline consumed them (pre-annotation),
so replay_demo.py can re-run perception on identical input. One video
frame is appended per pipeline tick, keeping video frame index == tick
index == metadata record index.

MJPG/AVI is used instead of H.264 because the pip OpenCV wheel has no
hardware encoder access; MJPG encode of 720p costs ~4-6 ms on a worker
thread, which never blocks the pipeline (frames are queued, and dropped
under back-pressure - drops are counted and logged).
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import cv2
import numpy as np


class VideoLogger:
    def __init__(self, run_dir: Path, fps: float, fourcc: str = "MJPG") -> None:
        self.path = str(Path(run_dir) / "video.avi")
        self.fps = max(1.0, fps)
        self.fourcc = fourcc
        self._writer: cv2.VideoWriter | None = None
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=60)
        self.dropped_frames = 0
        self.written_frames = 0
        self._thread = threading.Thread(target=self._loop, name="video-log", daemon=True)
        self._thread.start()

    def write(self, frame_bgr: np.ndarray) -> None:
        try:
            self._queue.put_nowait(frame_bgr)
        except queue.Full:
            self.dropped_frames += 1

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=10.0)
        if self._writer is not None:
            self._writer.release()

    def _loop(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:
                return
            if self._writer is None:
                h, w = frame.shape[:2]
                self._writer = cv2.VideoWriter(
                    self.path, cv2.VideoWriter_fourcc(*self.fourcc), self.fps, (w, h)
                )
            self._writer.write(frame)
            self.written_frames += 1
