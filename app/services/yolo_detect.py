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


def _weights_path() -> Path:
    raw = (settings.yolo_weights or "yolov8n.pt").strip()
    path = Path(raw)
    if not path.is_absolute():
        local = ROOT / "data" / "models" / path.name
        if local.is_file():
            return local
    return path


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


def _crop(bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = 0.12 * bw
    pad_y = 0.10 * bh
    xa = max(0, int(x1 - pad_x))
    ya = max(0, int(y1 - pad_y))
    xb = min(w, int(x2 + pad_x))
    yb = min(h, int(y2 + pad_y + 0.25 * bh))
    crop = bgr[ya:yb, xa:xb]
    if crop.size == 0:
        return bgr
    if crop.shape[1] < 320:
        import cv2

        scale = 320.0 / max(crop.shape[1], 1)
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
    return crop


def detect_vehicles(bgr: np.ndarray, *, predict_fn=None) -> list[VehicleDet]:
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
        dets.append(
            VehicleDet(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                confidence=score,
                vehicle_type=VEHICLE_TYPE_BY_ID.get(class_id, "car"),
                crop=_crop(bgr, x1, y1, x2, y2),
            )
        )
    dets.sort(key=lambda d: (-(d.x2 - d.x1) * (d.y2 - d.y1), -d.confidence))
    limit = max(1, int(settings.yolo_max_crops or 2))
    return dets[:limit]
