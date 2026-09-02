"""All-day demo helpers. Do not mark untested cameras analytics-active."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import AuditEvent, Camera
from app.services.capacity import start_accessible_workers
from app.services.coverage import camera_origin


def prepare_decode_ok_cameras(db: Session) -> list[str]:
    promoted = []
    for cam in db.scalars(select(Camera)):
        if cam.decode_status != "ok":
            continue
        if cam.processing_mode in {"deferred", "", None}:
            cam.processing_mode = "local_worker"
            cam.analytics_policy = "continuous"
            if camera_origin(cam) == "government_catalogue" and (cam.priority_class or "D") in {"C", "D", ""}:
                cam.priority_class = "B"
            promoted.append(cam.id)
        cam.analytics_active = False
    if promoted:
        db.add(AuditEvent(action="demo_promote", detail=",".join(promoted)))
    db.commit()
    return promoted


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
