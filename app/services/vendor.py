"""Authorised vendor ANPR metadata ingest. Sighting first, then exact match."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AuditEvent, Camera, Sighting
from app.services.pipeline import persist_sighting
from app.services.plates import normalize, syntax_ok, vote


class VendorIngestError(ValueError):
    pass


def ingest_vendor_event(db: Session, payload: dict, *, actor: str = "vendor") -> dict:
    event_id = str(payload.get("event_id") or payload.get("id") or "").strip()
    camera_id = str(payload.get("camera_id") or "").strip()
    plate_raw = str(payload.get("plate_raw") or payload.get("plate_text") or "").strip()
    vendor_model_id = str(payload.get("vendor_model_id") or payload.get("model_id") or "").strip()
    if not event_id:
        raise VendorIngestError("event_id is required")
    if not camera_id:
        raise VendorIngestError("camera_id is required")
    if not plate_raw:
        raise VendorIngestError("plate_raw is required")
    if not vendor_model_id:
        raise VendorIngestError("vendor_model_id is required")
    if payload.get("source_time") in (None, ""):
        raise VendorIngestError("source_time is required")
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise VendorIngestError("confidence is required") from exc

    camera = db.get(Camera, camera_id)
    if camera is None:
        raise VendorIngestError("unknown camera_id")

    existing = db.scalar(select(Sighting).where(Sighting.vendor_event_id == event_id))
    if existing:
        return {
            "ok": False,
            "duplicate": True,
            "sighting_id": existing.id,
            "event_id": event_id,
        }

    source_time = _parse_time(payload.get("source_time"))
    pts = payload.get("pts_ms") or payload.get("source_pts_ms")
    try:
        pts_ms = float(pts) if pts is not None else None
    except (TypeError, ValueError):
        pts_ms = None

    plate_norm = normalize(plate_raw)
    body = json.dumps(payload, sort_keys=True, default=str)
    payload_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:32]
    box = _box(payload.get("bbox"))
    passage = str(payload.get("passage_id") or f"vendor-{camera_id}-{event_id}")[:64]
    run_id = str(payload.get("run_id") or f"vendor-{event_id}")[:64]

    sighting, alert, created = persist_sighting(
        db,
        camera,
        plate_raw=plate_raw,
        plate_norm=plate_norm,
        plate_voted=vote([plate_raw]),
        syntax=syntax_ok(plate_norm),
        confidence=confidence,
        model_id=f"vendor:{vendor_model_id}",
        model_hash=str(payload.get("model_hash") or payload_hash),
        evidence_path=str(payload.get("evidence_path") or ""),
        run_id=run_id,
        frame_index=int(payload.get("frame_index") or 0),
        passage_id=passage,
        source_pts_ms=pts_ms,
        provider="vendor_metadata",
        ingest_time=datetime.now(timezone.utc),
        box=box,
        vendor_event_id=event_id,
        vendor_payload_hash=payload_hash,
    )
    # Honour vendor source_time after persist helper (which uses ingest+offset).
    sighting.source_time = source_time
    db.add(AuditEvent(actor=actor, action="vendor_event", detail=f"{camera_id} {event_id} sighting={sighting.id}"))
    db.commit()
    return {
        "ok": True,
        "duplicate": False,
        "sighting_id": sighting.id,
        "alert_id": getattr(alert, "id", None),
        "alert_created": created,
        "event_id": event_id,
        "plate_norm": plate_norm,
    }


def _parse_time(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _box(value) -> tuple[int, int, int, int] | None:
    if isinstance(value, dict):
        try:
            return int(value["x"]), int(value["y"]), int(value["w"]), int(value["h"])
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return int(value[0]), int(value[1]), int(value[2]), int(value[3])
        except (TypeError, ValueError):
            return None
    return None
