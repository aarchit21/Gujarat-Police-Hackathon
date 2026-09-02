from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.config import settings


def save_crop(crop_bgr: np.ndarray | None, camera_id: str, plate_norm: str) -> str:
    """Store a plate crop only — never full video."""
    if crop_bgr is None or crop_bgr.size == 0:
        return ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    safe_cam = "".join(ch for ch in camera_id if ch.isalnum() or ch in "-_") or "camera"
    folder = settings.evidence_dir / safe_cam
    folder.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch for ch in (plate_norm or "unread") if ch.isalnum()) or "unread"
    path = folder / f"{stamp}_{safe}.jpg"
    ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return ""
    Path(path).write_bytes(buf.tobytes())
    return str(path.relative_to(settings.evidence_dir.parent)).replace("\\", "/")
