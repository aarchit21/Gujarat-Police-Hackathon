"""All-day demo helpers. Do not mark untested cameras analytics-active."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import settings
from app.services.capacity import promote_decode_ok_cameras, start_accessible_workers


def prepare_decode_ok_cameras(db: Session) -> list[str]:
    return promote_decode_ok_cameras(db)


def autostart_if_configured(manager, db: Session) -> dict | None:
    if not settings.demo_autostart_workers:
        return None
    promoted = prepare_decode_ok_cameras(db)
    started = start_accessible_workers(
        manager,
        db,
        actor="demo",
        decode_ok_only=bool(settings.demo_decode_ok_only),
    )
    started["promoted"] = promoted
    return started
