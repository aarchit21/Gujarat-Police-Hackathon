from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.config import settings


def save_crop(crop_bgr: np.ndarray | None, camera_id: str, plate_norm: str, *, label: str = "original") -> str:
    """Store a plate crop only — never full video."""
    if crop_bgr is None or crop_bgr.size == 0:
        return ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    safe_cam = "".join(ch for ch in camera_id if ch.isalnum() or ch in "-_") or "camera"
    folder = settings.evidence_dir / safe_cam
    folder.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch for ch in (plate_norm or "unread") if ch.isalnum()) or "unread"
    safe_label = "".join(ch for ch in (label or "original") if ch.isalnum() or ch in "-_") or "original"
    path = folder / f"{stamp}_{safe}_{safe_label}.jpg"
    ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return ""
    Path(path).write_bytes(buf.tobytes())
    return str(path.relative_to(settings.evidence_dir.parent)).replace("\\", "/")


def save_context_crop(
    frame_bgr: np.ndarray | None,
    box: tuple[int, int, int, int] | None,
    camera_id: str,
    plate_norm: str,
) -> str:
    """Save a bounded annotated context crop, never a full camera recording."""
    if frame_bgr is None or not getattr(frame_bgr, "size", 0) or not box:
        return ""
    h, w = frame_bgr.shape[:2]
    try:
        x, y, bw, bh = [int(value) for value in box]
    except (TypeError, ValueError):
        return ""
    if bw <= 0 or bh <= 0:
        return ""
    pad = max(24, int(max(bw, bh) * 1.25))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    context = frame_bgr[y0:y1, x0:x1].copy()
    if context.size == 0:
        return ""
    cv2.rectangle(context, (x - x0, y - y0), (x + bw - x0, y + bh - y0), (0, 0, 255), 2)
    return save_crop(context, camera_id, plate_norm, label="context")
