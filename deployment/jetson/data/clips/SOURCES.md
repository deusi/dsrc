# Test clip sources and licenses

Dashcam footage used by the simulated-drive scenarios (`data/scenarios/`).
These files are test inputs only - check the license before redistributing
them or publishing stills (paper figures from the I-495 clip need the
CC BY-SA attribution below).

The scenarios point at local `.avi` (MJPG) versions produced by
`transcode_clip.py` from the `.webm` originals: MJPG decodes cheaply
during paced runs and is immune to the VP9 decode glitches described
below. Re-run the transcode if a `.avi` is missing.

## i495_eastbound_480p.webm

- Source: https://commons.wikimedia.org/wiki/File:Driving_eastbound_on_I-495_from_the_I-270_Spur_to_Cedar_Lane_(1_June_2026).webm
- Author: Illegitimate Barrister (Wikimedia Commons)
- License: CC BY-SA 4.0
- 854x480 @ 30 fps, 300.5 s. Capital Beltway (I-495) eastbound, Maryland,
  multi-lane interstate with dense flowing traffic. This is the Commons
  480p VP9 transcode; the 720p original is AV1, which this Jetson cannot
  decode (no hardware AV1 decoder on Orin Nano, no libdav1d in the
  system/OpenCV FFmpeg).
- Known defects of the Commons VP9 transcode: a decoder-poisoning glitch
  at frame ~4153 and an undecodable tail from frame ~8390 (93%) on. The
  local `i495_eastbound_480p.avi` therefore holds the 8384 clean frames
  (279.5 s) and is what the scenario uses.

## highway_decel_event_720p.webm

- Source: https://commons.wikimedia.org/wiki/File:Multiple_cars_rear-end_collision_on_highway.webm
- License: Public domain (CCTV/dashcam footage, no copyright)
- 1280x720 @ 30 fps, 30.1 s. Expressway dashcam approaching a slowdown that
  ends in a multi-car rear-end collision ahead - exercises leader-gap
  closing and the advisory slowdown path. Burned-in timestamp overlay
  bottom-left (harmless to the detector).
