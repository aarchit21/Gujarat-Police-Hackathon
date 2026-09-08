"""Durable recognition attempts plus cheap process-level pipeline counters."""
from __future__ import annotations

import threading
from collections import Counter, deque
from datetime import datetime, timezone

from sqlalchemy import delete, func, select

from app.config import settings

from app.models import RecognitionAttempt, Sighting

_lock = threading.Lock()
_counters: Counter[str] = Counter()
_latencies: deque[float] = deque(maxlen=1000)


def count_event(reason: str, amount: int = 1) -> None:
    if not reason:
        return
    with _lock:
        _counters[reason] += amount


def observe_latency(latency_ms: float) -> None:
    if latency_ms < 0:
        return
    with _lock:
        _latencies.append(float(latency_ms))


def runtime_diagnostics() -> dict:
    with _lock:
        values = sorted(_latencies)
        counters = dict(_counters)

    def percentile(p: float) -> float | None:
        if not values:
            return None
        return round(values[min(len(values) - 1, int((len(values) - 1) * p))], 3)

    return {
        "counters": counters,
        "latency_ms": {"count": len(values), "p50": percentile(0.50), "p95": percentile(0.95)},
    }


def add_attempt(
    db,
    *,
    camera_id: str,
    run_id: str,
    track_id: str,
    frame_index: int,
    source_pts_ms: float | None,
    stage: str,
    reason_code: str,
    detector: str = "",
    recognizer: str = "",
    model_id: str = "",
    model_hash: str = "",
    box: tuple[int, int, int, int] | None = None,
    quality: dict | None = None,
    raw_output: str = "",
    plate_norm: str = "",
    syntax: bool = False,
    character_confidences: list[float] | None = None,
    confidence: float = 0.0,
    latency_ms: float = 0.0,
    accepted: bool = False,
    evidence_path: str = "",
    enhanced_shape: tuple[int, int] | None = None,
) -> RecognitionAttempt:
    quality = quality or {}
    native_w = quality.get("width")
    native_h = quality.get("height")
    row = RecognitionAttempt(
        camera_id=camera_id, run_id=run_id, track_id=track_id, frame_index=frame_index,
        source_pts_ms=source_pts_ms, stage=stage, reason_code=reason_code,
        detector=detector, recognizer=recognizer, model_id=model_id, model_hash=model_hash,
        bbox_x=box[0] if box else None, bbox_y=box[1] if box else None,
        bbox_w=box[2] if box else None, bbox_h=box[3] if box else None,
        native_width=int(native_w) if native_w is not None else None,
        native_height=int(native_h) if native_h is not None else None,
        enhanced_width=enhanced_shape[1] if enhanced_shape else None,
        enhanced_height=enhanced_shape[0] if enhanced_shape else None,
        quality_json=quality or None, raw_output=(raw_output or "")[:2000],
        plate_norm=plate_norm, syntax_ok=syntax,
        character_confidences=character_confidences or None,
        confidence=float(confidence or 0.0), latency_ms=float(latency_ms or 0.0),
        accepted=bool(accepted), evidence_path=evidence_path,
    )
    db.add(row)
    db.flush()
    keep = max(1, int(settings.recognition_attempt_retention_per_track))
    stale_ids = list(db.scalars(
        select(RecognitionAttempt.id).where(
            RecognitionAttempt.camera_id == camera_id,
            RecognitionAttempt.track_id == track_id,
        ).order_by(RecognitionAttempt.id.desc()).offset(keep)
    ))
    if stale_ids:
        db.execute(delete(RecognitionAttempt).where(RecognitionAttempt.id.in_(stale_ids)))
    count_event(reason_code)
    observe_latency(latency_ms)
    return row


def diagnostics_snapshot(db, *, limit: int = 100) -> dict:
    reasons = {
        reason: int(count)
        for reason, count in db.execute(
            select(RecognitionAttempt.reason_code, func.count(RecognitionAttempt.id))
            .group_by(RecognitionAttempt.reason_code)
        )
        if reason
    }
    last_success = None
    for candidate in db.scalars(
        select(Sighting).where(Sighting.plate_norm != "").order_by(Sighting.id.desc()).limit(500)
    ):
        blob = candidate.vehicle_json if isinstance(candidate.vehicle_json, dict) else {}
        if (blob.get("confirmation") or {}).get("status") == "confirmed":
            last_success = candidate
            break
    return {
        "attempt_count": int(db.scalar(select(func.count(RecognitionAttempt.id))) or 0),
        "reason_counts": reasons,
        "runtime": runtime_diagnostics(),
        "last_success": None if last_success is None else {
            "sighting_id": last_success.id,
            "camera_id": last_success.camera_id,
            "plate_norm": last_success.plate_norm,
            "at": last_success.source_time.isoformat() if last_success.source_time else None,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "limit": limit,
    }


def serialize_attempt(row: RecognitionAttempt) -> dict:
    quality = row.quality_json or {}
    return {
        "id": row.id, "camera_id": row.camera_id, "run_id": row.run_id,
        "track_id": row.track_id, "frame_index": row.frame_index,
        "source_pts_ms": row.source_pts_ms,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "stage": row.stage, "reason_code": row.reason_code,
        "detector": row.detector, "recognizer": row.recognizer,
        "model_id": row.model_id, "model_hash": row.model_hash,
        "bbox": [row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h] if row.bbox_w is not None else None,
        "native_size": [row.native_width, row.native_height] if row.native_width is not None else None,
        "enhanced_size": [row.enhanced_width, row.enhanced_height] if row.enhanced_width is not None else None,
        "quality": quality, "context_evidence_path": quality.get("context_evidence_path", ""),
        "raw_output": row.raw_output,
        "plate_norm": row.plate_norm, "syntax_ok": row.syntax_ok,
        "character_confidences": row.character_confidences or [],
        "confidence": row.confidence, "latency_ms": row.latency_ms,
        "accepted": row.accepted, "evidence_path": row.evidence_path,
    }
