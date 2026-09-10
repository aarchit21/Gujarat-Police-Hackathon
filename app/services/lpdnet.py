"""NVIDIA TAO LPDNet plate detector on vehicle crops.

Runs the USA pruned ONNX via ONNX Runtime (TensorRT EP, then CUDA, then CPU).
Not DeepStream. Not LPRNet OCR. Missing weights → empty detections.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from app.config import ROOT, settings

INPUT_W = 640
INPUT_H = 480
_lock = threading.Lock()
_session = None
_input_name = ""
_load_error = ""
_provider = ""


def _weights_path() -> Path:
    raw_dir = str(getattr(settings, "lpdnet_weights_dir", "") or "data/models/lpdnet")
    name = str(getattr(settings, "lpdnet_onnx", "") or "LPDNet_usa_pruned_tao5.onnx")
    folder = Path(raw_dir)
    if not folder.is_absolute():
        folder = ROOT / folder
    return folder / name


def lpdnet_ready() -> bool:
    return bool(getattr(settings, "lpdnet_enabled", True)) and _weights_path().is_file()


def lpdnet_status() -> dict:
    path = _weights_path()
    return {
        "enabled": bool(getattr(settings, "lpdnet_enabled", True)),
        "ready": lpdnet_ready(),
        "loaded": _session is not None,
        "weights": str(path),
        "provider": _provider or "",
        "error": _load_error,
        "label": (
            "LPDNet off"
            if not getattr(settings, "lpdnet_enabled", True)
            else (
                f"LPDNet · {_provider}"
                if _session is not None
                else ("LPDNet weights missing" if not path.is_file() else f"LPDNet not loaded · {_load_error}")
            )
        ),
    }


def _providers() -> list[str]:
    requested = str(getattr(settings, "lpdnet_device", "auto") or "auto").strip().lower()
    try:
        import onnxruntime as ort

        available = list(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    preferred: list[str] = []
    if requested in {"auto", "tensorrt", "trt"} and "TensorrtExecutionProvider" in available:
        preferred.append("TensorrtExecutionProvider")
    if requested in {"auto", "cuda", "tensorrt", "trt"} and "CUDAExecutionProvider" in available:
        preferred.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        preferred.append("CPUExecutionProvider")
    return preferred or available


def _load():
    global _session, _input_name, _load_error, _provider
    if _session is not None or not lpdnet_ready():
        return _session
    with _lock:
        if _session is not None:
            return _session
        path = _weights_path()
        try:
            import onnxruntime as ort

            last = None
            for group in (_providers(), ["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]):
                try:
                    sess = ort.InferenceSession(str(path), providers=group)
                    _session = sess
                    _provider = sess.get_providers()[0] if sess.get_providers() else group[0]
                    _input_name = sess.get_inputs()[0].name
                    _load_error = ""
                    return _session
                except Exception as exc:
                    last = exc
                    _session = None
            _load_error = str(last)[:400] if last else "load failed"
        except Exception as exc:
            _load_error = str(exc)[:400]
            _session = None
    return _session


def _nms(boxes: list[tuple[int, int, int, int, float]], iou_thresh: float = 0.5) -> list[tuple[int, int, int, int, float]]:
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept: list[tuple[int, int, int, int, float]] = []
    while ordered:
        best = ordered.pop(0)
        kept.append(best)
        rest = []
        bx, by, bw, bh, _ = best
        for other in ordered:
            ox, oy, ow, oh, _ = other
            ix = max(0, min(bx + bw, ox + ow) - max(bx, ox))
            iy = max(0, min(by + bh, oy + oh) - max(by, oy))
            inter = ix * iy
            union = bw * bh + ow * oh - inter
            if union <= 0 or inter / union < iou_thresh:
                rest.append(other)
        ordered = rest
    return kept


def _as_nchw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4 and arr.shape[1] <= 16:
        return arr
    if arr.ndim == 4:
        return np.transpose(arr, (0, 3, 1, 2))
    if arr.ndim == 3:
        if arr.shape[0] <= 16:
            return arr[None, ...]
        return np.transpose(arr, (2, 0, 1))[None, ...]
    return arr


def _postprocess_detectnet(
    cov: np.ndarray,
    bbox: np.ndarray,
    *,
    conf: float,
    src_w: int,
    src_h: int,
) -> list[tuple[int, int, int, int, float]]:
    cov = _as_nchw(cov)[0]
    bbox = _as_nchw(bbox)[0]
    channels, grid_h, grid_w = cov.shape
    stride_x = INPUT_W / float(grid_w)
    stride_y = INPUT_H / float(grid_h)
    scale_x = src_w / float(INPUT_W)
    scale_y = src_h / float(INPUT_H)
    clusters = max(1, bbox.shape[0] // 4)
    hits: list[tuple[int, int, int, int, float]] = []
    for c in range(min(channels, clusters)):
        cover = cov[c]
        x1m, y1m, x2m, y2m = bbox[c * 4 : c * 4 + 4]
        ys, xs = np.where(cover >= conf)
        for i, j in zip(ys.tolist(), xs.tolist()):
            score = float(cover[i, j])
            cx = (j + 0.5) * stride_x
            cy = (i + 0.5) * stride_y
            x1 = (cx - float(x1m[i, j])) * scale_x
            y1 = (cy - float(y1m[i, j])) * scale_y
            x2 = (cx + float(x2m[i, j])) * scale_x
            y2 = (cy + float(y2m[i, j])) * scale_y
            xa = int(max(0, min(src_w - 1, round(min(x1, x2)))))
            ya = int(max(0, min(src_h - 1, round(min(y1, y2)))))
            xb = int(max(0, min(src_w, round(max(x1, x2)))))
            yb = int(max(0, min(src_h, round(max(y1, y2)))))
            if xb - xa < 4 or yb - ya < 4:
                continue
            hits.append((xa, ya, xb - xa, yb - ya, score))
    return _nms(hits)


def detect_plates(
    bgr: np.ndarray,
    *,
    session=None,
    conf: float | None = None,
) -> list[tuple[int, int, int, int, float]]:
    """Return (x, y, w, h, conf) in the input image's pixel coordinates."""
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return []
    if not bool(getattr(settings, "lpdnet_enabled", True)):
        return []
    runner = session if session is not None else _load()
    if runner is None:
        return []
    import cv2

    src_h, src_w = bgr.shape[:2]
    resized = cv2.resize(bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
    inp = _input_name or runner.get_inputs()[0].name
    try:
        outputs = runner.run(None, {inp: blob})
    except Exception:
        return []
    names = [o.name.lower() for o in runner.get_outputs()]
    cov = bbox = None
    if len(outputs) >= 2:
        for name, arr in zip(names, outputs):
            if "cov" in name or "sigmoid" in name:
                cov = arr
            elif "bbox" in name or "bias" in name:
                bbox = arr
        if cov is None or bbox is None:
            a, b = outputs[0], outputs[1]
            if a.size <= b.size:
                cov, bbox = a, b
            else:
                cov, bbox = b, a
        threshold = float(conf if conf is not None else getattr(settings, "lpdnet_conf", 0.30) or 0.30)
        return _postprocess_detectnet(cov, bbox, conf=threshold, src_w=src_w, src_h=src_h)
    return []
