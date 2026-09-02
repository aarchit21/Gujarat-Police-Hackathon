"""Camera analytics activity windows for investigation (not VMS playback)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models import Camera, CameraActivity, Sighting, utcnow
from app.services.coverage import camera_origin
from app.services.serialize import camera_public


def open_activity(db: Session, camera: Camera, *, run_id: str = "", reason: str = "worker") -> CameraActivity:
    row = CameraActivity(
        camera_id=camera.id,
        started_at=utcnow(),
        run_id=run_id,
        protocol=camera.active_protocol or camera.source_type or "",
        reason=reason,
    )
    db.add(row)
    db.flush()
    return row


def close_activity(db: Session, camera_id: str, *, reason: str = "stop") -> int:
    open_rows = list(
        db.scalars(
            select(CameraActivity).where(
                CameraActivity.camera_id == camera_id,
                CameraActivity.stopped_at.is_(None),
            )
        )
    )
    now = utcnow()
    for row in open_rows:
        row.stopped_at = now
        if reason:
            row.reason = reason
    return len(open_rows)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def cameras_active_at(
    db: Session,
    *,
    at: str | None = None,
    start: str | None = None,
    end: str | None = None,
    window_minutes: int = 30,
) -> dict:
    """Return cameras whose analytics window overlapped the query time/range."""
    start_dt = _parse_time(start)
    end_dt = _parse_time(end)
    at_dt = _parse_time(at)
    if start_dt is None and end_dt is None:
        pivot = at_dt or utcnow()
        delta = timedelta(minutes=max(1, int(window_minutes)))
        start_dt = pivot - delta
        end_dt = pivot + delta
    elif start_dt is None:
        start_dt = end_dt - timedelta(minutes=max(1, int(window_minutes)))
    elif end_dt is None:
        end_dt = start_dt + timedelta(minutes=max(1, int(window_minutes)))
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt

    rows = list(
        db.scalars(
            select(CameraActivity).where(
                CameraActivity.started_at <= end_dt,
                or_(CameraActivity.stopped_at.is_(None), CameraActivity.stopped_at >= start_dt),
            )
        )
    )
    by_cam: dict[str, list[CameraActivity]] = {}
    for row in rows:
        by_cam.setdefault(row.camera_id, []).append(row)

    sighting_counts: dict[str, int] = {}
    for s in db.scalars(
        select(Sighting).where(Sighting.source_time >= start_dt, Sighting.source_time <= end_dt)
    ):
        sighting_counts[s.camera_id] = sighting_counts.get(s.camera_id, 0) + 1

    cameras = []
    for camera_id in sorted(set(by_cam) | set(sighting_counts)):
        cam = db.get(Camera, camera_id)
        if cam is None:
            continue
        windows = [
            {
                "started_at": w.started_at.isoformat() if w.started_at else None,
                "stopped_at": w.stopped_at.isoformat() if w.stopped_at else None,
                "run_id": w.run_id,
                "protocol": w.protocol,
                "open": w.stopped_at is None,
            }
            for w in by_cam.get(camera_id, [])
        ]
        cameras.append(
            {
                **camera_public(cam),
                "origin": camera_origin(cam),
                "activity_windows": windows,
                "sightings_in_range": sighting_counts.get(camera_id, 0),
                "analytics_was_active": camera_id in by_cam,
                "disclaimer": "analytics active on this host — not a full-video archive",
            }
        )
    return {
        "from": start_dt.isoformat(),
        "to": end_dt.isoformat(),
        "camera_count": len(cameras),
        "cameras": cameras,
        "disclaimer": "These cameras had analytics running or a persisted sighting in the window. Full departmental video is not stored here.",
    }
