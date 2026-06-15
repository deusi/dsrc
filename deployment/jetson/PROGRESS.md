# Deployment Progress — resume here

Status against `../../plans/plan_deployment.md` tasks. Updated 2026-06-12.

## Snapshot

Code-complete, benchmarked, and now **validated end-to-end on simulated
drives**: real dashcam footage + scripted GPS through the full pipeline in
real time, scored by `eval_run.py` — **all gates PASS** on both shipped
scenarios. Camera and road testing still pending hardware. Policy bundle is
**random-init** until simulation training delivers a checkpoint.

Synthetic bench: **e2e p50 19.8 ms / p95 20.2 ms, 48.5 FPS** — `models/bench_results.md`.
Simulated drives (paced 30 fps, real TRT detector + MJPG file decode on CPU):
e2e p50 ~22–25 ms / p95 ~34–38 ms. Target < 200 ms.
56/57 tests pass on-device (`python3 -m pytest tests/`); 1 skip = sim-env
comparisons that need highway_env (run those on the dev machine).

Reference simulated runs (each has `report.md` + plots from `eval_run.py`):
- `~/dsrc_logs/run_20260612_163118` — i495_cruise, 8382 ticks / 280 s, PASS
  (GPS speed RMSE 0.121 m/s = injected noise floor; dropout handled)
- `~/dsrc_logs/run_20260612_162817` — highway_decel, 901 ticks / 30 s, PASS
  (leader acquired at ~42 m, closes to ~5 m as scripted braking ends)

## Plan tasks

| task | status | notes |
|---|---|---|
| P1 Jetson setup | DONE (software) | deps installed; engine built; `run_demo.py --selfcheck` implemented. Pending on hardware: **camera not yet attached** (no /dev/video0); **GPS blocked by dialout group** (fix below). |
| P2 detection + tracking | DONE (code + engine) | YOLOv8n FP16 TRT @640 = 3.9 ms GPU / 17.7 ms incl. pre/post; SORT-lite. Pending: demo video on real footage. |
| P3 distance estimation | DONE (geometric v0) | ground-plane + width-prior instead of depth net (latency call - see ARCHITECTURE §7). Pinhole math unit-tested; needs real-world sanity pass after calibration. |
| P4 observation builder | DONE | all 39 fields with provenance tags; encoder bit-identical to sim (tested); neutral fallbacks per spec. |
| P5 actor export + inference | DONE (untrained weights) | export_policy.py (checkpoint or --random); numpy runtime 0.5 ms. **Blocked on a trained checkpoint from the sim side.** |
| P6 dashboard | DONE (code) | HUD + advisory panel + UNTRAINED banner; not yet seen on a physical display. |
| P7 replay mode | DONE (code) | `replay_demo.py --log <run_dir>`; needs a recorded run (requires camera) to exercise for real. |
| P8 in-vehicle demo | NOT STARTED | needs camera, mount, calibration, drive plan. |
| (extra) simulated-drive harness | DONE + EXERCISED | scenarios = dashcam clip + camera geometry + scripted GPS (`sensors/gps_sim.py`); scored by `eval_run.py` gates. Both shipped scenarios PASS on-device. See README "Simulated drives", ARCHITECTURE §10. |

## Hardware blockers (one-time, need the operator)

1. **GPS permissions**: `sudo usermod -aG dialout $USER` then re-login.
   (`/dev/ttyACM0` is root:dialout; the u-blox 8 is enumerated and waiting.)
2. **Attach the USB camera**, then `python3 run_demo.py --selfcheck`.
3. Optional for stable benchmarks: `sudo nvpmodel -q` to record the power
   mode and `sudo jetson_clocks` before timing runs (couldn't be captured
   this session - no passwordless sudo; jtop logs the mode during runs).

## Next actions (in order)

1. Operator runs the dialout fix; re-run `--selfcheck`; confirm GPS sentences
   and the 5 Hz rate config (selfcheck prints observed rate).
2. Attach camera → desk run `python3 run_demo.py` pointed out a window;
   verify detections/distances look sane; record a clip and run
   `replay_demo.py` once over it (closes the P7 loop on real data).
3. Calibrate (README "Calibration") and set `camera:` values in config.yaml;
   if the mount sees the hood, also set `camera.hood_line_y_px` (phantom-
   leader filter — see ARCHITECTURE §10).
4. When training finishes: export the real checkpoint
   (`python3 policy/export_policy.py --checkpoint ... --out models/actor_policy`)
   — the UNTRAINED banner disappears by itself. Then re-run both scenarios
   (`run_demo.py --scenario ...` + `eval_run.py`) — first end-to-end look at
   trained advisories on real footage; head-switch rate should drop far
   below the random bundle's ~160–180/min.
5. Re-run `bench_latency.py` + a live logged run for final paper numbers
   (live e2e expected ~25–55 ms including camera frame interval; the
   simulated drives already measured p95 ~34–38 ms paced at 30 fps).
6. Windshield mount + P8 drive (passenger-operated, advisory-only protocol
   from the plan).

## Session log

- **2026-06-12** — simulated-drive harness (Claude session). New:
  `sensors/gps_sim.py` (profile-scripted GPS twin, dropouts/noise/loop),
  `--scenario`/`--sim-gps` in run_demo, `eval_run.py` (gated reports +
  plots), `calibration/auto_horizon.py` (Theil–Sen horizon/height fit from
  detections), `transcode_clip.py`, 2 scenarios + 2 CC dashcam clips
  (`data/`, licenses in SOURCES.md). Both scenarios PASS all gates.
  Fixed along the way: FFmpeg VP9 state poisoning on one bad cluster →
  CameraStream reopen+seek recovery (the I-495 Commons transcode has a
  glitch at frame ~4153 and a dead tail past ~8390, hence the 280 s AVI);
  ego hood detected as phantom leader at minimum range →
  `camera.hood_line_y_px` detector filter + ground-plane→width-prior
  fallback for hood/frame-clipped bboxes. Tests 37→56. Known cosmetics:
  matplotlib Axes3D warning in eval_run (same ultralytics/matplotlib
  duplication, harmless).
- **2026-06-11** — initial build (Claude session). Probed device (JetPack 6 /
  L4T R36.4.7, TRT 10.3, CPU-only torch → TRT-direct detection + CPU actor
  design). Vendored sim contract from commit `d477dba` with equality tests.
  Implemented full pipeline, entry points, tests, docs. Built YOLOv8n FP16
  engine (3.9 ms GPU). Optimized: numpy preprocess → blobFromImage (−5.5 ms),
  TorchScript actor → verified numpy mirror (−3.5 ms, jitter gone);
  e2e 29.2 → 19.8 ms p50. Fixed rel_speed_window default (6→8: at 30 fps the
  window must span the 0.2 s minimum or relative speed never validates).
  Known cosmetics: ultralytics prints a matplotlib Axes3D warning (harmless);
  ONNX exported at opset 18 (requested 12, converter fell back - TRT 10.3
  handles 18 fine).
