#!/usr/bin/env python3
"""Estimate horizon row + camera height for a dashcam clip from detections.

Ground-plane geometry gives, for a vehicle of true width W at distance Z:
    v_bottom - y_horizon = h_cam * fx / Z      and      w_px = fx * W / Z
so   v_bottom = y_horizon + h_cam * (w_px / W)
which is linear in x = w_px / W_class: the intercept is the horizon row,
the slope is the camera height in meters. A Theil-Sen fit over a few
hundred real detections recovers both without any manual marking; fx
still has to come from an assumed horizontal FOV (it only scales
absolute distance, not ordering or lane assignment).

Use it to fill the "camera" block of a scenario file:
    python3 calibration/auto_horizon.py data/clips/i495_eastbound_480p.webm

Approximations: flat road, negligible lens distortion, class width
priors (config distance.class_widths_m). Good enough for simulated-drive
testing; calibrate properly (README "Calibration") before trusting
absolute distances from a real mounted camera.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

JETSON_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JETSON_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402


def theil_sen(x: np.ndarray, y: np.ndarray, max_pairs: int = 200_000):
    n = len(x)
    i, j = np.triu_indices(n, k=1)
    if len(i) > max_pairs:
        keep = np.random.default_rng(0).choice(len(i), max_pairs, replace=False)
        i, j = i[keep], j[keep]
    dx = x[j] - x[i]
    ok = np.abs(dx) > 1e-9
    slope = float(np.median((y[j] - y[i])[ok] / dx[ok]))
    intercept = float(np.median(y - slope * x))
    return slope, intercept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--config", default=str(JETSON_DIR / "config.yaml"))
    parser.add_argument("--samples", type=int, default=150, help="frames to sample uniformly")
    parser.add_argument("--conf", type=float, default=0.5, help="min detection confidence")
    parser.add_argument("--hfov-deg", type=float, default=100.0, help="assumed horizontal FOV for fx")
    parser.add_argument(
        "--hood-line", type=float, default=None,
        help="row where the ego hood starts; excludes hood/reflection boxes from the fit",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    widths = {int(k): float(v) for k, v in config["distance"]["class_widths_m"].items()}

    from perception.detector import TrtYoloDetector

    engine = Path(config["detector"]["engine"])
    detector = TrtYoloDetector(
        engine_path=str(engine if engine.is_absolute() else JETSON_DIR / engine),
        input_size=config["detector"]["input_size"],
        conf_threshold=args.conf,
        vehicle_classes=tuple(config["detector"]["vehicle_classes"]),
        hood_line_y_px=args.hood_line,
    )

    cap = cv2.VideoCapture(args.video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n_frames <= 0 or img_w <= 0:
        print(f"cannot read {args.video}")
        return 1

    xs, vs = [], []
    used_frames = 0
    for idx in np.linspace(0, n_frames - 1, args.samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, image = cap.read()
        if not ok:
            continue
        used_frames += 1
        for det in detector.infer(image):
            x1, y1, x2, y2 = det.xyxy
            w_px = x2 - x1
            h_px = y2 - y1
            if w_px < 14 or h_px < 12:
                continue  # too small to localize the bottom edge
            if x1 <= 2 or x2 >= img_w - 2 or y2 >= img_h - 2:
                continue  # cropped by the frame border -> wrong width/bottom
            if y2 < img_h * 0.25:
                continue  # in the sky / overpass region, not on the road plane
            if det.cls not in widths:
                continue
            xs.append(w_px / widths[det.cls])
            vs.append(y2)
    cap.release()
    detector.close()

    if len(xs) < 30:
        print(f"only {len(xs)} usable detections from {used_frames} frames - "
              "clip too empty for auto-calibration; mark the horizon manually "
              "(calibration/camera_calibration.py horizon)")
        return 1

    x = np.asarray(xs, dtype=np.float64)
    v = np.asarray(vs, dtype=np.float64)
    height_m, horizon_y = theil_sen(x, v)
    residuals = v - (horizon_y + height_m * x)
    fx = (img_w / 2.0) / math.tan(math.radians(args.hfov_deg) / 2.0)

    print(f"{len(x)} detections from {used_frames} frames of {img_w}x{img_h}")
    print(f"fit: horizon_y = {horizon_y:.1f} px, camera_height = {height_m:.2f} m, "
          f"residual MAD {np.median(np.abs(residuals)):.1f} px")
    if not (0.8 <= height_m <= 2.5):
        print("WARNING: implausible camera height - treat this fit with suspicion")
    camera_block = {
        "fx_px": round(fx, 1),
        "cx_px": img_w / 2.0,
        "cy_px": img_h / 2.0,
        "horizon_y_px": round(horizon_y, 1),
        "camera_height_m": round(height_m, 2),
        "width": img_w,
        "height": img_h,
    }
    print("scenario camera block (assumed HFOV "
          f"{args.hfov_deg:.0f} deg -> fx {fx:.0f} px):")
    print(json.dumps({"camera": camera_block}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
