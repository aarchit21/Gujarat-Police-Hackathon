"""Time-multiplex analytics across all live catalogue cameras.

This host holds 4 RTSP captures. Hunt rotates those slots so every live
government feed is visited. It does not open 30 streams at once and is
not a central VMS.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Alert, AuditEvent, Camera, Sighting
from app.services.coverage import camera_origin
from app.services.ingest import rtsp_url_for
from app.services.vehicle_event import is_recordable_plate


PIN_DEFAULT = ("cam01", "cam02", "cam03", "cam05")


def hunt_targets(db: Session, *, pinned_only: bool = False, pin_ids: list[str] | None = None) -> list[Camera]:
    rows = list(db.scalars(select(Camera).order_by(Camera.id)))
    allow = {str(x).strip() for x in (pin_ids or PIN_DEFAULT) if str(x).strip()}
    out = []
    for cam in rows:
        if not cam.catalogue_live and camera_origin(cam) != "government_catalogue":
            continue
        if not (rtsp_url_for(cam) or cam.hls_url):
            continue
        if pinned_only and cam.id not in allow:
            continue
        out.append(cam)
    out.sort(
        key=lambda c: (
            0 if c.id in allow else 1,
            0 if c.decode_status == "ok" else 1 if (c.decode_status or "untested") == "untested" else 2,
            c.id,
        )
    )
    return out


def promote_hunt_camera(cam: Camera) -> bool:
    if cam.processing_mode in {"deferred", "", None}:
        cam.processing_mode = "local_worker"
        cam.analytics_policy = "continuous"
        return True
    return False


def start_hunt(
    manager,
    db: Session,
    *,
    actor: str = "operator",
    pinned_only: bool = False,
    pin_ids: list[str] | None = None,
) -> dict:
    targets = hunt_targets(db, pinned_only=pinned_only, pin_ids=pin_ids)
    promoted = []
    for cam in targets:
        if promote_hunt_camera(cam):
            promoted.append(cam.id)
    db.commit()

    manager.begin_hunt([c.id for c in targets])

    running = [
        w.get("camera_id")
        for w in manager.snapshot().get("workers", [])
        if w.get("camera_id") in manager.hunt_target_ids
    ]
    for camera_id in running:
        manager.stop(db, camera_id, actor=actor)

    started, queued, failed = [], [], []
    for cam in targets:
        out = manager.start(db, cam.id, actor=actor)
        state = out.get("state") or ""
        if not out.get("ok"):
            failed.append({"id": cam.id, "error": out.get("error")})
        elif state == "queued":
            queued.append(cam.id)
        else:
            started.append(cam.id)

    db.add(
        AuditEvent(
            actor=actor,
            action="hunt_start",
            detail=f"targets={len(targets)} started={len(started)} queued={len(queued)} dwell={settings.hunt_dwell_seconds}s",
        )
    )
    db.commit()
    status = hunt_status(manager, db)
    status.update(
        {
            "ok": True,
            "promoted": promoted,
            "started": started,
            "queued": queued,
            "failed": failed,
            "pinned_only": pinned_only,
            "disclaimer": (
                "Pinned to 4 working live cameras."
                if pinned_only
                else "This host hunts 4 government streams at a time and visits all live catalogue "
                "cameras each cycle. Not 30 simultaneous decodes. Not a central VMS."
            ),
        }
    )
    return status


def stop_hunt(manager, db: Session, *, actor: str = "operator") -> dict:
    ids = list(manager.hunt_target_ids)
    manager.end_hunt()
    for camera_id in ids:
        manager.stop(db, camera_id, actor=actor)
    db.add(AuditEvent(actor=actor, action="hunt_stop", detail=f"stopped={len(ids)}"))
    db.commit()
    return {"ok": True, "stopped": ids, **hunt_status(manager, db)}


def hunt_status(manager, db: Session) -> dict:
    snap = manager.snapshot()
    targets = list(manager.hunt_target_ids)
    visited = sorted(manager.hunt_visited)
    gov_ids = targets or [c.id for c in hunt_targets(db)]
    vehicles = 0
    plates = 0
    alerts = 0
    if gov_ids:
        rows = list(db.scalars(select(Sighting).where(Sighting.camera_id.in_(gov_ids))))
        vehicles = len(rows)
        plates = sum(1 for s in rows if is_recordable_plate(s.plate_norm))
        alerts = db.scalar(select(func.count(Alert.id)).where(Alert.camera_id.in_(gov_ids))) or 0
    hunting = [
        w.get("camera_id")
        for w in snap.get("workers", [])
        if w.get("status") in {"running", "starting"}
    ]
    last_hunted = [
        {"id": c.id, "last_hunted_at": c.last_hunted_at.isoformat() if c.last_hunted_at else None}
        for c in db.scalars(select(Camera).where(Camera.last_hunted_at.isnot(None)).order_by(Camera.last_hunted_at.desc()).limit(12))
    ]
    total = len(gov_ids)
    return {
        "enabled": bool(manager.hunt_enabled),
        "cycle_id": manager.hunt_cycle_id,
        "cycle": manager.hunt_cycle,
        "hunting": hunting,
        "hunting_count": len(hunting),
        "max_concurrent": snap.get("max_concurrent") or settings.max_concurrent_workers,
        "visited": visited,
        "visited_count": len(visited),
        "total": total,
        "queued": snap.get("queued") or [],
        "queued_count": snap.get("queued_count") or 0,
        "vehicles_seen": vehicles,
        "plates_read": plates,
        "alerts": alerts,
        "dwell_seconds": settings.hunt_dwell_seconds,
        "max_frames": settings.hunt_max_frames,
        "last_hunted": last_hunted,
        "label": (
            f"Hunting {len(hunting)}/{total} · visited {len(visited)}/{total} this cycle · "
            f"{vehicles} vehicles · {plates} plates"
            if manager.hunt_enabled
            else "Hunt idle — this host can run 4 live streams at a time"
        ),
    }


def mark_camera_hunted(db: Session, camera_id: str) -> None:
    cam = db.get(Camera, camera_id)
    if cam is None:
        return
    cam.last_hunted_at = datetime.now(timezone.utc)
    db.add(cam)
