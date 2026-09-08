"""Vehicle-first persistence and investigation search.

These observations describe one local camera track.  They are not a visual
identity system and must never be connected into a claimed cross-camera route.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from sqlalchemy import func, select

from app.models import Camera, VehicleObservation
from app.services.evidence import save_context_crop, save_crop
from app.services.serialize import ist_label, utc_iso
from app.services.timing import source_time_from_ingest

VEHICLE_TYPES = (
    "car", "suv", "two_wheeler", "truck", "bus", "van", "auto_rickshaw", "taxi_cab", "unknown",
)
VEHICLE_COLORS = (
    "white", "black", "silver", "gray", "red", "blue", "green", "yellow", "orange", "brown", "other", "unknown",
)


def clean_vehicle_type(value: str | None) -> str:
    raw = str(value or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "motorcycle": "two_wheeler", "bike": "two_wheeler", "scooter": "two_wheeler",
        "auto": "auto_rickshaw", "rickshaw": "auto_rickshaw", "autorickshaw": "auto_rickshaw",
        "cab": "taxi_cab", "taxi": "taxi_cab", "pickup": "truck", "lorry": "truck",
        "sedan": "car", "hatchback": "car", "jeep": "suv",
    }
    return aliases.get(raw, raw) if aliases.get(raw, raw) in VEHICLE_TYPES else "unknown"


def clean_vehicle_color(value: str | None) -> str:
    raw = str(value or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"grey": "gray", "maroon": "red", "beige": "brown", "gold": "brown"}
    return aliases.get(raw, raw) if aliases.get(raw, raw) in VEHICLE_COLORS else "unknown"


def color_estimate(crop) -> tuple[str, float]:
    """Deterministic colour estimate; unknown is safer than a weak label."""
    from app.services.anpr import estimate_vehicle_color

    color = clean_vehicle_color(estimate_vehicle_color(crop))
    if color == "unknown":
        return color, 0.0
    # The current HSV estimator has no calibrated distribution.  Keep this
    # intentionally modest and expose it as an estimate in the UI/API.
    return color, 0.65


def _quality(box: tuple[int, int, int, int] | None, detector_confidence: float) -> float:
    if not box:
        return max(0.0, float(detector_confidence or 0.0))
    return max(0.0, float(box[2]) * float(box[3])) * max(0.01, float(detector_confidence or 0.0))


def upsert_observation(
    db,
    camera: Camera,
    *,
    run_id: str,
    track_id: str,
    frame_index: int,
    source_pts_ms: float | None,
    box: tuple[int, int, int, int] | None,
    frame_shape: tuple[int, int],
    frame,
    crop,
    detector: str,
    detector_confidence: float,
    vehicle_type: str,
) -> tuple[VehicleObservation, bool]:
    now = datetime.now(timezone.utc)
    source_time, _ = source_time_from_ingest(now, camera.clock_offset_ms)
    row = db.scalar(select(VehicleObservation).where(
        VehicleObservation.camera_id == camera.id,
        VehicleObservation.run_id == run_id,
        VehicleObservation.track_id == track_id,
    ))
    canonical_type = clean_vehicle_type(vehicle_type)
    color, color_conf = color_estimate(crop)
    score = _quality(box, detector_confidence)
    if row is None:
        evidence = save_crop(crop, camera.id, track_id, label="vehicle")
        context = save_context_crop(frame, box, camera.id, track_id)
        row = VehicleObservation(
            camera_id=camera.id, run_id=run_id, track_id=track_id,
            first_seen_at=source_time, last_seen_at=source_time,
            first_pts_ms=source_pts_ms, last_pts_ms=source_pts_ms,
            best_frame_index=frame_index,
            bbox_x=box[0] if box else None, bbox_y=box[1] if box else None,
            bbox_w=box[2] if box else None, bbox_h=box[3] if box else None,
            frame_width=frame_shape[1], frame_height=frame_shape[0],
            detector=detector, detector_confidence=float(detector_confidence or 0.0),
            vehicle_type=canonical_type, type_confidence=float(detector_confidence or 0.0), type_source="yolov8n",
            vehicle_color=color, color_confidence=color_conf, color_source="opencv_hsv",
            evidence_path=evidence, context_evidence_path=context,
            metadata_json={"quality_score": score, "attribute_status": "local_estimate"},
        )
        db.add(row)
        db.flush()
        return row, True
    row.last_seen_at = source_time
    row.last_pts_ms = source_pts_ms
    meta = dict(row.metadata_json or {})
    old_score = float(meta.get("quality_score") or 0.0)
    if score > old_score:
        row.best_frame_index = frame_index
        row.bbox_x = box[0] if box else None
        row.bbox_y = box[1] if box else None
        row.bbox_w = box[2] if box else None
        row.bbox_h = box[3] if box else None
        row.frame_width, row.frame_height = frame_shape[1], frame_shape[0]
        row.detector, row.detector_confidence = detector, float(detector_confidence or 0.0)
        row.vehicle_type = canonical_type
        row.type_confidence, row.type_source = float(detector_confidence or 0.0), "yolov8n"
        row.vehicle_color, row.color_confidence, row.color_source = color, color_conf, "opencv_hsv"
        row.evidence_path = save_crop(crop, camera.id, track_id, label="vehicle") or row.evidence_path
        row.context_evidence_path = save_context_crop(frame, box, camera.id, track_id) or row.context_evidence_path
        meta["quality_score"] = score
        meta["attribute_status"] = "local_estimate"
        row.metadata_json = meta
    db.flush()
    return row, False


def apply_refinement(db, observation_id: int, payload: dict) -> bool:
    """Cautiously merge a cloud estimate into a persisted local observation."""
    row = db.get(VehicleObservation, observation_id)
    if row is None:
        return False
    meta = dict(row.metadata_json or {})
    cloud_type = clean_vehicle_type(payload.get("vehicle_type"))
    cloud_color = clean_vehicle_color(payload.get("vehicle_color"))
    confidence = max(0.0, min(1.0, float(payload.get("confidence") or 0.0)))
    meta["cloud_refinement"] = {
        "vehicle_type": cloud_type, "vehicle_color": cloud_color, "confidence": confidence,
        "model_id": payload.get("model_id") or "", "status": "review",
    }
    # Cloud may refine a YOLO car into SUV/taxi/auto, or fill an unknown.  It
    # never replaces a strong local colour/type with a conflicting guess.
    if confidence >= float(settings_vehicle_attribute_min_confidence()):
        if row.vehicle_type == "unknown" or (row.vehicle_type == "car" and cloud_type in {"suv", "taxi_cab", "auto_rickshaw", "van"}):
            if cloud_type != "unknown":
                row.vehicle_type, row.type_confidence, row.type_source = cloud_type, confidence, "ollama_refined"
                meta["cloud_refinement"]["status"] = "applied"
        if row.vehicle_color == "unknown" and cloud_color != "unknown":
            row.vehicle_color, row.color_confidence, row.color_source = cloud_color, confidence, "ollama_refined"
            meta["cloud_refinement"]["status"] = "applied"
    row.metadata_json = meta
    db.flush()
    return True


def settings_vehicle_attribute_min_confidence() -> float:
    from app.config import settings
    return float(settings.vehicle_attribute_min_confidence)


def observation_json(row: VehicleObservation) -> dict:
    camera = row.camera
    return {
        "id": row.id, "camera_id": row.camera_id, "camera_name": camera.name if camera else "",
        "city": camera.city if camera else "", "department": camera.department if camera else "",
        "lat": camera.lat if camera else None, "lng": camera.lng if camera else None,
        "track_id": row.track_id, "run_id": row.run_id,
        "first_seen_at": utc_iso(row.first_seen_at), "first_seen_at_ist": ist_label(row.first_seen_at),
        "last_seen_at": utc_iso(row.last_seen_at), "last_seen_at_ist": ist_label(row.last_seen_at),
        "vehicle_type": row.vehicle_type, "type_confidence": row.type_confidence, "type_source": row.type_source,
        "vehicle_color": row.vehicle_color, "color_confidence": row.color_confidence, "color_source": row.color_source,
        "detector": row.detector, "detector_confidence": row.detector_confidence,
        "evidence_path": row.evidence_path, "context_evidence_path": row.context_evidence_path,
        "metadata": row.metadata_json or {},
        "disclaimer": "Type and colour are estimates. Matching attributes do not confirm the same vehicle.",
    }


def search_observations(db, *, start, end, vehicle_type=None, vehicle_color=None, camera_id=None, min_confidence=0.0, limit=100, offset=0, sort="asc") -> dict:
    query = select(VehicleObservation).where(
        VehicleObservation.first_seen_at >= start,
        VehicleObservation.first_seen_at <= end,
        VehicleObservation.detector_confidence >= float(min_confidence or 0.0),
    )
    if vehicle_type:
        query = query.where(VehicleObservation.vehicle_type == clean_vehicle_type(vehicle_type))
    if vehicle_color:
        query = query.where(VehicleObservation.vehicle_color == clean_vehicle_color(vehicle_color))
    if camera_id:
        query = query.where(VehicleObservation.camera_id == camera_id)
    count_query = select(func.count()).select_from(query.subquery())
    total = int(db.scalar(count_query) or 0)
    ordered = VehicleObservation.first_seen_at.desc() if sort == "desc" else VehicleObservation.first_seen_at.asc()
    rows = list(db.scalars(query.order_by(ordered).offset(offset).limit(limit)))
    camera_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    color_counts: dict[str, int] = {}
    for row in rows:
        camera_counts[row.camera_id] = camera_counts.get(row.camera_id, 0) + 1
        type_counts[row.vehicle_type] = type_counts.get(row.vehicle_type, 0) + 1
        color_counts[row.vehicle_color] = color_counts.get(row.vehicle_color, 0) + 1
    return {
        "from": utc_iso(start), "to": utc_iso(end), "total": total, "offset": offset, "limit": limit,
        "observations": [observation_json(row) for row in rows],
        "camera_counts": camera_counts, "type_counts": type_counts, "color_counts": color_counts,
        "disclaimer": "Matches share selected attributes; identity is not confirmed. No route is inferred.",
    }


def observations_csv(payload: dict) -> str:
    output = io.StringIO()
    fields = ["id", "camera_id", "camera_name", "city", "first_seen_at", "first_seen_at_ist", "vehicle_type", "type_confidence", "vehicle_color", "color_confidence", "detector", "detector_confidence", "evidence_path"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(payload.get("observations") or [])
    return output.getvalue()
