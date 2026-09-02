"""Operator live-frame snapshot. Overwrites one JPEG per camera. Not a VMS archive."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from app.config import settings
from app.security import redact_secrets

_SAVE_GAP_S = 2.0
_CACHE_FRESH_S = 20.0
_last_save: dict[str, float] = {}
_lock = threading.Lock()


def preview_dir() -> Path:
    path = settings.frames_dir / "live_preview"
    path.mkdir(parents=True, exist_ok=True)
    return path


def preview_path(camera_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in (camera_id or "cam"))[:64]
    return preview_dir() / f"{safe}.jpg"


def encode_jpeg(frame, *, max_width: int = 640) -> bytes:
    import cv2

    if frame is None:
        raise ValueError("empty frame")
    height, width = frame.shape[:2]
    if width > max_width and width > 0:
        scale = max_width / float(width)
        frame = cv2.resize(frame, (max_width, max(1, int(height * scale))))
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        raise ValueError("jpeg encode failed")
    return buf.tobytes()


def maybe_save_live_preview(camera_id: str, frame) -> None:
    now = time.monotonic()
    last = _last_save.get(camera_id, 0.0)
    if now - last < _SAVE_GAP_S:
        return
    try:
        data = encode_jpeg(frame)
        preview_path(camera_id).write_bytes(data)
        _last_save[camera_id] = now
    except Exception:
        return


def _cached_jpeg(camera_id: str, *, max_age: float = _CACHE_FRESH_S) -> bytes | None:
    path = preview_path(camera_id)
    if not path.is_file():
        return None
    age = time.time() - path.stat().st_mtime
    if age > max_age:
        return None
    return path.read_bytes()


def _from_image_dir(camera) -> bytes | None:
    uri = Path((getattr(camera, "source_uri", None) or "")).expanduser()
    if getattr(camera, "source_type", "") == "file" and uri.is_file():
        return uri.read_bytes()
    if not uri.is_dir():
        return None
    files = sorted(p for p in uri.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        return None
    return files[-1].read_bytes()


def grab_snapshot(camera, *, open_fn=None) -> dict:
    """Return one operator JPEG. Credentials never leave the server."""
    cached = _cached_jpeg(camera.id)
    if getattr(camera, "source_type", "") in {"image_dir", "file"}:
        data = _from_image_dir(camera)
        if data:
            return {"ok": True, "jpeg": data, "source": "own_feed_file"}
    if cached:
        return {"ok": True, "jpeg": cached, "source": "cached_worker_frame"}
    if not _lock.acquire(blocking=False):
        stale = _cached_jpeg(camera.id, max_age=120)
        if stale:
            return {"ok": True, "jpeg": stale, "source": "cached_worker_frame"}
        return {"ok": False, "error": "snapshot busy"}
    try:
        from app.services.ingest import iter_live_frames

        for _idx, frame, _pts in iter_live_frames(camera, open_fn=open_fn, max_frames=2, max_seconds=8):
            jpeg = encode_jpeg(frame)
            preview_path(camera.id).write_bytes(jpeg)
            return {"ok": True, "jpeg": jpeg, "source": "live_grab"}
    except Exception as exc:
        return {"ok": False, "error": redact_secrets(str(exc))}
    finally:
        _lock.release()
    return {"ok": False, "error": "no frame decoded"}
