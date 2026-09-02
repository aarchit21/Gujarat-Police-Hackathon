from __future__ import annotations

import csv
import io
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Alert, Camera, Sighting
from app.services.coverage import camera_origin


def sighting_rows(db: Session) -> list[dict]:
    alerts_by_sighting: dict[int, Alert] = {}
    for alert in db.scalars(select(Alert)):
        alerts_by_sighting[alert.sighting_id] = alert
    cameras = {c.id: c for c in db.scalars(select(Camera))}
    rows = []
    for s in db.scalars(select(Sighting).order_by(Sighting.source_time)):
        cam = cameras.get(s.camera_id)
        alert = alerts_by_sighting.get(s.id)
        rows.append(
            {
                "plate_number": s.plate_norm,
                "plate_raw": s.plate_raw,
                "raw_ocr": s.plate_raw,
                "plate_normalised": s.plate_norm,
                "voted_ocr": s.plate_voted,
                "plate_voted": s.plate_voted,
                "camera": s.camera_id,
                "camera_id": s.camera_id,
                "department": cam.department if cam else "",
                "location": cam.city if cam else "",
                "protocol": cam.active_protocol if cam else "",
                "codec": cam.codec if cam else "",
                "source_pts_ms": s.source_pts_ms,
                "source_timestamp": s.source_time.isoformat() if s.source_time else "",
                "ingest_utc": s.ingest_time.isoformat() if s.ingest_time else "",
                "ingest_timestamp": s.ingest_time.isoformat() if s.ingest_time else "",
                "confidence": s.confidence,
                "model_id": s.model_id,
                "model_hash": s.model_hash,
                "run_id": s.run_id,
                "evidence_reference": s.evidence_path,
                "alert_review_status": alert.status if alert else "",
                "catalogue_live": bool(cam.catalogue_live) if cam else False,
                "decode_status": cam.decode_status if cam else "",
                "origin": camera_origin(cam) if cam else "",
            }
        )
    return rows


REPORT_FIELDS = [
    "camera_id",
    "department",
    "location",
    "protocol",
    "codec",
    "plate_raw",
    "plate_normalised",
    "plate_voted",
    "source_pts_ms",
    "ingest_utc",
    "confidence",
    "model_id",
    "model_hash",
    "run_id",
    "evidence_reference",
    "alert_review_status",
    "catalogue_live",
    "decode_status",
]


def as_json(rows: list[dict]) -> str:
    return json.dumps(rows, indent=2)


def as_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=REPORT_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()
