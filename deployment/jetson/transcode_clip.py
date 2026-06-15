#!/usr/bin/env python3
"""One-time clean-up transcode of a test clip to MJPG AVI.

Why this exists: there is no ffmpeg CLI on the device, and some Commons
VP9 transcodes carry periodic decode glitches (chunked-transcode seams)
that poison OpenCV's sequential decoder mid-stream. This reads every
decodable frame - reopening and seeking past each glitch, same recovery
as CameraStream - and rewrites the clip as MJPG, which decodes cheaply
(less CPU stolen from detection during paced runs) and has no
inter-frame decoder state to poison.

Lossless in count except the glitch frames themselves (reported).

  python3 transcode_clip.py data/clips/i495_eastbound_480p.webm
  -> data/clips/i495_eastbound_480p.avi (same fps, MJPG)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

JETSON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(JETSON_DIR))

import cv2  # noqa: E402

from sensors.camera_stream import reopen_past_failure  # noqa: E402

MAX_RECOVERIES = 256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--out", help="default: input path with .avi extension")
    args = parser.parse_args()

    src = Path(args.video)
    out = Path(args.out) if args.out else src.with_suffix(".avi")
    capture = cv2.VideoCapture(str(src))
    if not capture.isOpened():
        print(f"cannot open {src}")
        return 1
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    written = 0
    recoveries = 0
    t0 = time.monotonic()
    while True:
        ok, image = capture.read()
        if not ok:
            if recoveries >= MAX_RECOVERIES:
                break
            fresh = reopen_past_failure(capture, lambda: cv2.VideoCapture(str(src)))
            if fresh is None:
                break  # genuine end of stream
            capture = fresh
            recoveries += 1
            continue
        if writer is None:
            h, w = image.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
            if not writer.isOpened():
                print(f"cannot open writer for {out}")
                return 1
        writer.write(image)
        written += 1
    capture.release()
    if writer is not None:
        writer.release()

    elapsed = time.monotonic() - t0
    size_mb = out.stat().st_size / 1e6 if out.exists() else 0.0
    print(
        f"{written}/{n_frames} frames -> {out} ({size_mb:.0f} MB) in {elapsed:.0f} s; "
        f"glitches skipped: {recoveries}"
    )
    return 0 if written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
