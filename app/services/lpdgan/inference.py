"""Optional LPDGAN inference on plate crops.

Disabled unless weights exist and `lpdgan_enabled` is set. Untrained or
missing weights must not replace the deterministic CLAHE preprocessor —
generative deblur can invent strokes and is not evidence by itself.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np

from app.config import ROOT, settings
from app.services.lpdgan.config import FULL_HW, LPDGANConfig

_LOCK = Lock()
_RUNTIME: "_LPDGANRuntime | None" = None


def _device_name() -> str:
    requested = str(getattr(settings, "lpdgan_device", "auto") or "auto").strip().lower()
    try:
        import torch
    except ImportError:
        return "cpu"
    if requested in {"cpu", "cuda"}:
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _weights_path() -> Path:
    raw = str(getattr(settings, "lpdgan_weights", "") or "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path
    return ROOT / "data" / "models" / "lpdgan_generator.pth"


class _LPDGANRuntime:
    def __init__(self) -> None:
        import torch

        from app.services.lpdgan.generator import LPDGANGenerator, build_image_pyramid

        self.torch = torch
        self.build_image_pyramid = build_image_pyramid
        self.device = torch.device(_device_name())
        self.cfg = LPDGANConfig()
        self.generator = LPDGANGenerator(self.cfg)
        self.weights = _weights_path()
        payload = torch.load(self.weights, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "generator" in payload:
            payload = payload["generator"]
        if not isinstance(payload, dict):
            raise ValueError("LPDGAN weights must be a state_dict")
        missing, unexpected = self.generator.load_state_dict(payload, strict=False)
        if missing:
            raise RuntimeError(f"LPDGAN weights missing keys: {missing[:8]}")
        self.generator.to(self.device).eval()
        for p in self.generator.parameters():
            p.requires_grad = False
        self.unexpected = list(unexpected)

    def deblur(self, rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        h, w = rgb.shape[:2]
        tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float().div_(255.0).mul_(2.0).sub_(1.0)
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            full, half, quarter = self.build_image_pyramid(tensor, self.cfg)
            out = self.generator(full, half, quarter).y_full
        out = out.squeeze(0).clamp(-1, 1).add_(1.0).mul_(0.5)
        out = out.mul_(255.0).round().byte().permute(1, 2, 0).cpu().numpy()
        if (out.shape[0], out.shape[1]) != (h, w):
            import cv2

            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
        return np.ascontiguousarray(out)


def lpdgan_status() -> dict:
    enabled = bool(getattr(settings, "lpdgan_enabled", False))
    weights = _weights_path()
    try:
        import torch
    except ImportError:
        return {
            "enabled": enabled,
            "available": False,
            "loaded": False,
            "reason": "torch_not_installed",
            "weights": str(weights),
            "device": "cpu",
            "generative": True,
            "input_hw": list(FULL_HW),
        }
    reason = ""
    if not enabled:
        reason = "disabled"
    elif not weights.is_file():
        reason = "weights_missing"
    loaded = _RUNTIME is not None
    return {
        "enabled": enabled,
        "available": True,
        "loaded": loaded,
        "ready": enabled and weights.is_file(),
        "reason": reason,
        "weights": str(weights),
        "device": _device_name(),
        "cuda": bool(torch.cuda.is_available()),
        "generative": True,
        "input_hw": list(FULL_HW),
        "torch": torch.__version__,
    }


def _runtime() -> _LPDGANRuntime | None:
    global _RUNTIME
    status = lpdgan_status()
    if not status.get("ready"):
        return None
    with _LOCK:
        if _RUNTIME is None:
            _RUNTIME = _LPDGANRuntime()
        return _RUNTIME


def reset_runtime() -> None:
    global _RUNTIME
    with _LOCK:
        _RUNTIME = None


def deblur_plate_bgr(bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Deblur a BGR plate crop. Returns (image, meta); identity if not ready."""
    meta = {**lpdgan_status(), "applied": False, "method": "lpdgan"}
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return bgr, meta
    runtime = _runtime()
    if runtime is None:
        return bgr, meta
    try:
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        restored = runtime.deblur(rgb)
        out = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
        meta["applied"] = True
        meta["loaded"] = True
        meta["output_width"] = int(out.shape[1])
        meta["output_height"] = int(out.shape[0])
        return np.ascontiguousarray(out), meta
    except Exception as exc:
        meta["reason"] = f"inference_failed:{type(exc).__name__}"
        return bgr, meta
