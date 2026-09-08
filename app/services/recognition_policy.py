"""Runtime policy for the optional plate-recognition branch.

The app is a single-process P0, so a tiny locked cache lets running camera
workers see an operator change at their next sampled frame.  The durable value
remains in SystemState and is restored after restart.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone

from sqlalchemy import select

from app.config import settings
from app.models import Camera, SystemState

STATE_KEY = "plate_recognition_enabled"
VALID_CAMERA_MODES = {"inherit", "on", "off"}

_lock = threading.Lock()
_enabled: bool | None = None
_camera_modes: dict[str, str] = {}


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def hydrate(db) -> bool:
    """Load the persisted setting, creating the documented safe first-run value."""
    global _enabled
    with _lock:
        if _enabled is not None:
            return _enabled
        row = db.get(SystemState, STATE_KEY)
        if row is None:
            row = SystemState(key=STATE_KEY, value="true" if settings.plate_recognition_enabled else "false")
            db.add(row)
            db.flush()
        _enabled = _as_bool(row.value, bool(settings.plate_recognition_enabled))
        return _enabled


def set_global(db, enabled: bool) -> bool:
    global _enabled
    row = db.get(SystemState, STATE_KEY)
    if row is None:
        row = SystemState(key=STATE_KEY, value="false")
        db.add(row)
    row.value = "true" if enabled else "false"
    row.updated_at = datetime.now(timezone.utc)
    with _lock:
        _enabled = bool(enabled)
    return bool(enabled)


def set_camera_mode(camera_id: str, mode: str) -> str:
    cleaned = str(mode or "inherit").strip().lower()
    if cleaned not in VALID_CAMERA_MODES:
        raise ValueError("plate_recognition_mode must be inherit, on, or off")
    with _lock:
        _camera_modes[camera_id] = cleaned
    return cleaned


def effective(db, camera: Camera | None) -> bool:
    # Global off is a hard safety stop. Per-camera settings only narrow or
    # explicitly allow the branch after the global operator setting is on.
    if not hydrate(db):
        return False
    if camera is None:
        return True
    with _lock:
        mode = _camera_modes.get(camera.id, getattr(camera, "plate_recognition_mode", "inherit"))
    return str(mode).lower() != "off"


def snapshot(db, cameras: list[Camera] | None = None) -> dict:
    enabled = hydrate(db)
    rows = cameras if cameras is not None else list(db.scalars(select(Camera).order_by(Camera.id)))
    items = []
    for camera in rows:
        with _lock:
            mode = _camera_modes.get(camera.id, getattr(camera, "plate_recognition_mode", "inherit"))
        items.append({"camera_id": camera.id, "mode": mode, "effective": bool(enabled and mode != "off")})
    return {
        "enabled": enabled,
        "first_run_default": bool(settings.plate_recognition_enabled),
        "cameras": items,
        "effective_camera_count": sum(1 for row in items if row["effective"]),
    }

