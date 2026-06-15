"""YOLOv8n vehicle detection through TensorRT 10 (no torch on the GPU path).

The system torch on this device is a CPU build, so detection talks to
TensorRT directly via cuda-python. The engine is built offline by
export_detector.sh (ONNX -> trtexec --fp16); engine files are
device-specific and must be rebuilt per Jetson / TensorRT version.

Engine I/O (ultralytics YOLOv8 ONNX, no embedded NMS):
  input  "images"  float32 (1, 3, S, S)   letterboxed RGB / 255
  output "output0" float32 (1, 84, 8400)  [cx, cy, w, h, 80 class scores]
"""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass

import cv2
import numpy as np

# TensorRT / CUDA are imported lazily in TrtYoloDetector so that the rest
# of this module (Detection, letterbox, nms) stays importable on machines
# without the Jetson GPU stack (dev laptops, CI).
trt = None
cudart = None


def _load_gpu_stack() -> None:
    global trt, cudart
    if trt is not None:
        return
    import tensorrt as _trt

    try:  # cuda-python >= 12.8 layout
        from cuda.bindings import runtime as _cudart
    except ImportError:  # older cuda-python
        from cuda import cudart as _cudart  # type: ignore[no-redef]
    trt, cudart = _trt, _cudart


COCO_VEHICLE_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Detection:
    xyxy: np.ndarray  # (4,) float32, original-image pixels
    conf: float
    cls: int


def _check(err) -> None:
    if isinstance(err, tuple):
        err = err[0]
    if int(err) != 0:
        raise RuntimeError(f"CUDA runtime error {err}")


def _pinned_array(shape: tuple[int, ...], dtype: np.dtype) -> tuple[np.ndarray, int]:
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    err, ptr = cudart.cudaHostAlloc(nbytes, cudart.cudaHostAllocDefault)
    _check(err)
    buf = (ctypes.c_byte * nbytes).from_address(ptr)
    return np.frombuffer(buf, dtype=dtype).reshape(shape), ptr


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    h, w = image.shape[:2]
    gain = min(size / h, size / w)
    new_w, new_h = int(round(w * gain)), int(round(h * gain))
    pad_x, pad_y = (size - new_w) / 2.0, (size - new_h) / 2.0
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, gain, (pad_x, pad_y)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


class TrtYoloDetector:
    def __init__(
        self,
        engine_path: str,
        input_size: int = 640,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        vehicle_classes: tuple[int, ...] = (2, 3, 5, 7),
        max_detections: int = 64,
        hood_line_y_px: float | None = None,
    ) -> None:
        _load_gpu_stack()
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.vehicle_classes = np.asarray(vehicle_classes, dtype=np.int64)
        self.max_detections = max_detections
        # rows below this are the ego vehicle's own hood: YOLO happily boxes
        # the hood and reflections riding on it as "car", which then becomes
        # a phantom leader at minimum range. Boxes whose CENTER falls below
        # the line are dropped; a real close car truncated by the hood keeps
        # its center above it and survives.
        self.hood_line_y_px = hood_line_y_px
        self.last_timings: dict[str, float] = {}

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"failed to load TensorRT engine {engine_path} - engines are "
                "device/TRT-version specific; rebuild with export_detector.sh"
            )
        self.context = self.engine.create_execution_context()
        err, self.stream = cudart.cudaStreamCreate()
        _check(err)

        self._io: dict[str, dict] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            host, host_ptr = _pinned_array(shape, dtype)
            nbytes = host.nbytes
            err, dev_ptr = cudart.cudaMalloc(nbytes)
            _check(err)
            self.context.set_tensor_address(name, int(dev_ptr))
            mode = self.engine.get_tensor_mode(name)
            self._io[name] = {
                "host": host,
                "host_ptr": host_ptr,
                "dev_ptr": dev_ptr,
                "nbytes": nbytes,
                "is_input": mode == trt.TensorIOMode.INPUT,
            }
        inputs = [n for n, io in self._io.items() if io["is_input"]]
        outputs = [n for n, io in self._io.items() if not io["is_input"]]
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"expected 1 input / 1 output tensor, got {inputs} / {outputs}")
        self._in_name, self._out_name = inputs[0], outputs[0]

    def warmup(self, iterations: int = 3) -> float:
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        start = time.monotonic()
        for _ in range(iterations):
            self.infer(dummy)
        return (time.monotonic() - start) / iterations * 1000.0

    def infer(self, image_bgr: np.ndarray) -> list[Detection]:
        t0 = time.monotonic()
        canvas, gain, (pad_x, pad_y) = letterbox(image_bgr, self.input_size)
        # blobFromImage does BGR->RGB + /255 + HWC->CHW in optimized C++
        # (~4x faster than the numpy equivalent on the Orin's A78 cores)
        blob = cv2.dnn.blobFromImage(canvas, scalefactor=1.0 / 255.0, swapRB=True)
        np.copyto(self._io[self._in_name]["host"], blob)

        t1 = time.monotonic()
        io_in, io_out = self._io[self._in_name], self._io[self._out_name]
        _check(
            cudart.cudaMemcpyAsync(
                io_in["dev_ptr"], io_in["host_ptr"], io_in["nbytes"],
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream,
            )
        )
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        _check(
            cudart.cudaMemcpyAsync(
                io_out["host_ptr"], io_out["dev_ptr"], io_out["nbytes"],
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream,
            )
        )
        _check(cudart.cudaStreamSynchronize(self.stream))

        t2 = time.monotonic()
        detections = self._postprocess(
            io_out["host"][0], gain, pad_x, pad_y, image_bgr.shape[1], image_bgr.shape[0]
        )
        t3 = time.monotonic()
        self.last_timings = {
            "pre_ms": (t1 - t0) * 1000.0,
            "infer_ms": (t2 - t1) * 1000.0,
            "post_ms": (t3 - t2) * 1000.0,
        }
        return detections

    def _postprocess(
        self, raw: np.ndarray, gain: float, pad_x: float, pad_y: float, img_w: int, img_h: int
    ) -> list[Detection]:
        # raw: (84, 8400) -> boxes (8400, 4) + class scores (8400, 80)
        boxes_cxcywh = raw[:4].T
        scores_vehicle = raw[4:].T[:, self.vehicle_classes]  # (8400, n_vehicle)
        conf = scores_vehicle.max(axis=1)
        mask = conf >= self.conf_threshold
        if not mask.any():
            return []
        boxes_cxcywh = boxes_cxcywh[mask]
        conf = conf[mask]
        cls = self.vehicle_classes[scores_vehicle[mask].argmax(axis=1)]

        xyxy = np.empty_like(boxes_cxcywh)
        xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / gain
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / gain
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, img_w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, img_h)

        if self.hood_line_y_px is not None:
            above_hood = (xyxy[:, 1] + xyxy[:, 3]) / 2 < self.hood_line_y_px
            if not above_hood.any():
                return []
            xyxy, conf, cls = xyxy[above_hood], conf[above_hood], cls[above_hood]

        keep = nms(xyxy, conf, self.iou_threshold)[: self.max_detections]
        return [
            Detection(xyxy=xyxy[i].astype(np.float32), conf=float(conf[i]), cls=int(cls[i]))
            for i in keep
        ]

    def close(self) -> None:
        for io in self._io.values():
            cudart.cudaFree(io["dev_ptr"])
            cudart.cudaFreeHost(io["host_ptr"])
        cudart.cudaStreamDestroy(self.stream)
