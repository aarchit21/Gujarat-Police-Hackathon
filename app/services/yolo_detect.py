"""YOLOv8n vehicle detector. Crops cars/buses/trucks/motos. Ignores people.

Optional: if ultralytics/torch is missing, detect_vehicles returns [] and
prepare_live_anpr_frame stays the OpenCV blob fallback.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import ROOT, settings

# COCO ids. Never include person (0).
VEHICLE_CLASS_IDS = {2, 3, 5, 7}
VEHICLE_TYPE_BY_ID = {
    2: "car",
    3: "two_wheeler",
    5: "bus",
    7: "truck",
}

_lock = threading.Lock()
_model = None
_load_error = ""
_device = ""


@dataclass
class VehicleDet:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    vehicle_type: str
    crop: np.ndarray
    crop_box: tuple[int, int, int, int] = (0, 0, 0, 0)


def _weights_path() -> Path:
    raw = (settings.yolo_weights or "yolov8n.pt").strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    name = path.name
    for candidate in (ROOT / "data" / "models" / name, ROOT / name, path):
        if candidate.is_file():
            return candidate
    return ROOT / "data" / "models" / name


def _pick_device() -> str:
    configured = (settings.yolo_device or "auto").strip().lower()
    if configured and configured != "auto":
        return configured
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def yolo_status() -> dict:
    enabled = bool(settings.yolo_enabled)
    available = False
    err = _load_error
    try:
        import ultralytics  # noqa: F401

        available = True
    except Exception as exc:
        err = err or f"ultralytics not installed: {exc}"
    return {
        "enabled": enabled,
        "available": available,
        "loaded": _model is not None,
        "device": _device or _pick_device(),
        "weights": str(_weights_path()),
        "classes": ["car", "motorcycle", "bus", "truck"],
        "ignores": ["person"],
        "error": err,
        "label": (
            "YOLO off"
            if not enabled
            else (
                f"YOLOv8n · {_device or _pick_device()}"
                if _model is not None
                else (f"YOLO not ready · {err}" if err else "YOLOv8n · not loaded yet")
            )
        ),
    }


def _load_model():
    global _model, _load_error, _device
    if _model is not None or not settings.yolo_enabled:
        return _model
    with _lock:
        if _model is not None:
            return _model
        try:
            from ultralytics import YOLO

            _device = _pick_device()
            weights = _weights_path()
            weights.parent.mkdir(parents=True, exist_ok=True)
            _model = YOLO(str(weights))
            _load_error = ""
        except Exception as exc:
            _model = None
            _load_error = str(exc)[:300]
    return _model


def expand_vehicle_box(
    frame_shape: tuple[int, ...],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[int, int, int, int]:
    """Pad the YOLO box so bumper plates are not clipped. Native pixels only."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    bw = max(1.0, float(x2) - float(x1))
    bh = max(1.0, float(y2) - float(y1))
    pad_x = float(getattr(settings, "vehicle_crop_pad_x", 0.20) or 0.20) * bw
    pad_top = float(getattr(settings, "vehicle_crop_pad_top", 0.15) or 0.15) * bh
    pad_bottom = float(getattr(settings, "vehicle_crop_pad_bottom", 0.45) or 0.45) * bh
    xa = max(0, int(x1 - pad_x))
    ya = max(0, int(y1 - pad_top))
    xb = min(w, int(x2 + pad_x))
    yb = min(h, int(y2 + pad_bottom))
    if xb <= xa or yb <= ya:
        return max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    return xa, ya, xb, yb


def _crop(bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    xa, ya, xb, yb = expand_vehicle_box(bgr.shape, x1, y1, x2, y2)
    crop = bgr[ya:yb, xa:xb]
    if crop.size == 0:
        return bgr, (0, 0, int(bgr.shape[1]), int(bgr.shape[0]))
    return crop, (xa, ya, xb - xa, yb - ya)


def detect_vehicles(bgr: np.ndarray, *, predict_fn=None, max_detections: int | None = None) -> list[VehicleDet]:
    """Return vehicle crops only. Person boxes are dropped."""
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return []
    runner = predict_fn
    if runner is None:
        if not settings.yolo_enabled:
            return []
        model = _load_model()
        if model is None:
            return []

        def runner(frame):
            with _lock:
                return model.predict(
                    frame,
                    conf=float(settings.yolo_conf or 0.25),
                    classes=sorted(VEHICLE_CLASS_IDS),
                    verbose=False,
                    device=_device or "cpu",
                )

    try:
        results = runner(bgr)
    except Exception as exc:
        global _load_error
        _load_error = str(exc)[:300]
        return []
    if not results:
        return []
    first = results[0]
    boxes = getattr(first, "boxes", None)
    if boxes is None:
        return []
    dets: list[VehicleDet] = []
    xyxy = getattr(boxes, "xyxy", None)
    cls = getattr(boxes, "cls", None)
    conf = getattr(boxes, "conf", None)
    if xyxy is None:
        return []
    arr = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
    cls_arr = cls.cpu().numpy() if cls is not None and hasattr(cls, "cpu") else np.asarray(cls if cls is not None else [])
    conf_arr = conf.cpu().numpy() if conf is not None and hasattr(conf, "cpu") else np.asarray(conf if conf is not None else [])
    for i, row in enumerate(arr):
        class_id = int(cls_arr[i]) if i < len(cls_arr) else -1
        if class_id not in VEHICLE_CLASS_IDS:
            continue
        x1, y1, x2, y2 = [float(v) for v in row[:4]]
        score = float(conf_arr[i]) if i < len(conf_arr) else 0.0
        crop, crop_box = _crop(bgr, x1, y1, x2, y2)
        dets.append(
            VehicleDet(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                confidence=score,
                vehicle_type=VEHICLE_TYPE_BY_ID.get(class_id, "car"),
                crop=crop,
                crop_box=crop_box,
            )
        )
    dets.sort(key=lambda d: (-(d.x2 - d.x1) * (d.y2 - d.y1), -d.confidence))
    limit = max(1, int(max_detections if max_detections is not None else settings.yolo_max_crops or 2))
    return dets[:limit]
