"""File-source decode-error recovery in CameraStream.

A mid-file decode failure must trigger reopen+seek (not end-of-stream);
a failure at the true end of the file must still terminate; recovery
attempts are bounded for files that never decode again.
"""

import time

import numpy as np

from sensors.camera_stream import CameraStream


class FakeCapture:
    """Sequential capture over `total` frames that fails on `poison` frames.

    Mimics the FFmpeg behavior seen with one bad VP9 cluster: once a read
    fails, the handle is poisoned and every later read fails too; a fresh
    handle (new FakeCapture) seeked past the bad frame works again.
    """

    def __init__(self, total: int, poison: set[int]):
        self.total = total
        self.poison = poison
        self.pos = 0
        self.dead = False

    def isOpened(self):
        return True

    def read(self):
        if self.dead or self.pos >= self.total or self.pos in self.poison:
            self.dead = True
            return False, None
        self.pos += 1
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def get(self, prop):
        import cv2

        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.total)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.pos)
        if prop == cv2.CAP_PROP_FPS:
            return 0.0  # no pacing in tests
        return 0.0

    def set(self, prop, value):
        import cv2

        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.pos = int(value)
        return True

    def release(self):
        pass


def run_stream(total: int, poison: set[int], timeout_s: float = 5.0) -> CameraStream:
    cam = CameraStream(
        source="file:/nonexistent.avi", pace_file_source=False, max_file_recoveries=8
    )
    cam._open = lambda: FakeCapture(total, poison)  # type: ignore[method-assign]
    cam.start()
    consumed = 0
    deadline = time.monotonic() + timeout_s
    while not cam.end_of_stream and time.monotonic() < deadline:
        if cam.wait_for_fresh(timeout=0.2) is not None:
            consumed += 1
    cam.stop()
    cam.frames_consumed = consumed  # type: ignore[attr-defined]
    return cam


def test_mid_file_poison_recovers_and_reaches_end():
    cam = run_stream(total=100, poison={40})
    assert cam.end_of_stream
    assert cam.file_recoveries == 1
    # frame 40 was skipped, every other frame was produced
    assert cam._frame_counter == 99


def test_clean_end_of_file_does_not_recover():
    cam = run_stream(total=50, poison=set())
    assert cam.end_of_stream
    assert cam.file_recoveries == 0
    assert cam._frame_counter == 50


def test_unrecoverable_region_exhausts_bounded_retries():
    # every frame from 20 on is bad: retries advance one frame each, then stop
    cam = run_stream(total=1000, poison=set(range(20, 1000)))
    assert cam.end_of_stream
    assert cam.file_recoveries == 8
    assert cam._frame_counter == 20
