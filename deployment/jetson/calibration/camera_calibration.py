#!/usr/bin/env python3
"""Camera calibration helpers for the distance estimator.

Three ways to get the numbers config.yaml needs (fx_px, cx_px, cy_px,
horizon_y_px, camera_height_m):

fov       quickest: compute fx from the camera's spec-sheet horizontal
          FOV. Good to ~5% which the width-prior already exceeds.
            python3 calibration/camera_calibration.py fov --width 1280 --hfov-deg 78

intrinsics proper checkerboard calibration from a directory of images
          (9x6 inner corners, ~20 views).
            python3 calibration/camera_calibration.py intrinsics --images ~/calib/*.jpg

horizon   interactive: shows the live camera (or a file), click the
          horizon line (or use up/down arrows), prints horizon_y_px.
          Mount the camera level, point down a flat road or use the
          true horizon. Re-do after every remount.
            python3 calibration/camera_calibration.py horizon --source 0

camera_height_m is a tape measure job (lens center to road).
"""

from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np


def cmd_fov(args: argparse.Namespace) -> int:
    fx = (args.width / 2.0) / math.tan(math.radians(args.hfov_deg) / 2.0)
    print(f"fx_px: {fx:.1f}")
    print(f"cx_px: {args.width / 2.0:.1f}")
    print("(cy_px: image_height / 2 unless calibrated)")
    return 0


def cmd_intrinsics(args: argparse.Namespace) -> int:
    pattern = (args.cols, args.rows)
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2)
    obj_points, img_points = [], []
    size = None
    paths = sorted(glob.glob(args.images))
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
        if found:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            obj_points.append(objp)
            img_points.append(corners)
    if len(obj_points) < 5:
        print(f"only {len(obj_points)}/{len(paths)} images had a detectable "
              f"{pattern[0]}x{pattern[1]} checkerboard; need >= 5")
        return 1
    rms, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    print(f"# calibration RMS reprojection error: {rms:.3f} px ({len(obj_points)} views)")
    print(f"fx_px: {mtx[0, 0]:.1f}")
    print(f"cx_px: {mtx[0, 2]:.1f}")
    print(f"cy_px: {mtx[1, 2]:.1f}")
    print(f"# distortion (k1 k2 p1 p2 k3): {dist.ravel().round(4).tolist()}")
    return 0


def cmd_horizon(args: argparse.Namespace) -> int:
    source = args.source
    cap = (
        cv2.VideoCapture(source[len("file:"):])
        if source.startswith("file:")
        else cv2.VideoCapture(int(source), cv2.CAP_V4L2)
    )
    if not cap.isOpened():
        print(f"cannot open source {source}")
        return 1
    state = {"y": None}

    def on_mouse(event, _x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["y"] = y

    win = "horizon-calibration (click horizon; arrows nudge; s=save q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop files
            continue
        if state["y"] is None:
            state["y"] = frame.shape[0] // 2
        view = frame.copy()
        cv2.line(view, (0, state["y"]), (view.shape[1], state["y"]), (0, 200, 255), 1)
        cv2.putText(view, f"horizon_y_px: {state['y']}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            print(f"horizon_y_px: {state['y']}")
            break
        if key == 82:  # up
            state["y"] = max(0, state["y"] - 1)
        if key == 84:  # down
            state["y"] = min(frame.shape[0] - 1, state["y"] + 1)
    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fov = sub.add_parser("fov", help="fx from spec-sheet horizontal FOV")
    p_fov.add_argument("--width", type=int, required=True)
    p_fov.add_argument("--hfov-deg", type=float, required=True)
    p_fov.set_defaults(func=cmd_fov)

    p_int = sub.add_parser("intrinsics", help="checkerboard calibration")
    p_int.add_argument("--images", required=True, help="glob, e.g. '~/calib/*.jpg'")
    p_int.add_argument("--cols", type=int, default=9)
    p_int.add_argument("--rows", type=int, default=6)
    p_int.set_defaults(func=cmd_intrinsics)

    p_hor = sub.add_parser("horizon", help="interactive horizon line picker")
    p_hor.add_argument("--source", default="0")
    p_hor.set_defaults(func=cmd_horizon)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
