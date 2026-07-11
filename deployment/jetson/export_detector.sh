#!/usr/bin/env bash
# Build the vehicle-detection TensorRT engine for THIS device.
#
# Engines are specific to the GPU + TensorRT version: rebuild after any
# JetPack upgrade and on every new unit. Takes ~10 min on an Orin Nano
# (mostly trtexec kernel autotuning).
#
# Usage: ./export_detector.sh [model] [imgsz]
#   model  ultralytics weight name (default yolov8n.pt)
#   imgsz  square input size (default 640; try 448 for extra headroom)

set -euo pipefail
cd "$(dirname "$0")/models"

MODEL="${1:-yolov8n.pt}"
IMGSZ="${2:-640}"
BASE="${MODEL%.pt}"
TRTEXEC=/usr/src/tensorrt/bin/trtexec

echo "== exporting ${MODEL} -> ONNX (imgsz=${IMGSZ}) =="
python3 - "$MODEL" "$IMGSZ" <<'PY'
import sys
from ultralytics import YOLO
model, imgsz = sys.argv[1], int(sys.argv[2])
path = YOLO(model).export(format="onnx", imgsz=imgsz, dynamic=False, simplify=False)
print("ONNX:", path)
PY

echo "== building TensorRT FP16 engine =="
"$TRTEXEC" --onnx="${BASE}.onnx" \
  --saveEngine="${BASE}_${IMGSZ}_fp16.engine" \
  --fp16 --memPoolSize=workspace:2048

echo "== done: models/${BASE}_${IMGSZ}_fp16.engine =="
echo "Update detector.engine in config.yaml if you changed model/imgsz."
