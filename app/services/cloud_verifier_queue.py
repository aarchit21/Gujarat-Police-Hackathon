"""Bounded single-worker queue so cloud vision never blocks live capture."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from app.config import settings


@dataclass
class CloudJob:
    kind: str
    camera_id: str
    run_id: str
    track_id: str
    frame_index: int
    source_pts_ms: float | None
    image: np.ndarray
    box: tuple[int, int, int, int] | None
    frame_shape: tuple[int, int]
    detector: str = ""
    vehicle_type: str = ""
    vehicle_color: str = ""
    quality: dict = field(default_factory=dict)
    priority: float = 0.0
    observation_id: int | None = None
    queued_at: float = field(default_factory=time.monotonic)


class CloudVerifierQueue:
    def __init__(self):
        self._jobs: list[CloudJob] = []
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self.processed = 0
        self.dropped = 0
        self.errors = 0
        self.last_error = ""

    def enqueue(self, job: CloudJob) -> bool:
        if not settings.cloud_verifier_queue_enabled or not settings.ollama_vision_enabled:
            return False
        with self._condition:
            # One bounded queue protects capture workers.  Attribute jobs use
            # the same cap instead of silently expanding cloud concurrency.
            limit = max(1, int(settings.cloud_verifier_queue_size))
            self._jobs.append(job)
            self._jobs.sort(key=lambda item: (item.priority, item.queued_at), reverse=True)
            if len(self._jobs) > limit:
                self._jobs.pop()
                self.dropped += 1
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="cloud-plate-verifier", daemon=True)
                self._thread.start()
            self._condition.notify()
        return True

    def _run(self) -> None:
        while True:
            with self._condition:
                if not self._jobs:
                    self._condition.wait(timeout=5.0)
                if not self._jobs:
                    return
                job = self._jobs.pop(0)
            if job.kind in {"plate", "vision_only"} and not _plate_job_allowed(job):
                self.dropped += 1
                continue
            max_age = (
                float(settings.vehicle_attribute_max_age_seconds)
                if job.kind == "vehicle_attribute"
                else float(settings.cloud_verifier_max_age_seconds)
            )
            if time.monotonic() - job.queued_at > max_age:
                self.dropped += 1
                continue
            try:
                _process_job(job)
                self.processed += 1
            except Exception as exc:
                self.errors += 1
                self.last_error = str(exc)[:500]
                _record_job_failure(job, exc)

    def status(self) -> dict:
        with self._condition:
            return {
                "enabled": bool(settings.cloud_verifier_queue_enabled),
                "worker_count": 1,
                "depth": len(self._jobs),
                "capacity": max(1, int(settings.cloud_verifier_queue_size)),
                "processed": self.processed,
                "dropped": self.dropped,
                "errors": self.errors,
                "last_error": self.last_error,
            }


def _process_job(job: CloudJob) -> None:
    from app.database import SessionLocal
    from app.models import Camera
    from app.services.anpr import enhance_for_vision
    from app.services.evidence import save_crop
    from app.services.ollama_vision import infer_bgr, infer_vision_only_frame, infer_vehicle
    from app.services.pipeline import persist_sighting
    from app.services.plates import normalize, syntax_ok
    from app.services.recognition_diagnostics import add_attempt
    from app.services.vehicle_event import is_recordable_plate

    started = time.perf_counter()
    if job.kind == "vehicle_attribute":
        from app.services.vehicle_observations import apply_refinement

        enhanced, enhancement = enhance_for_vision(job.image, profile="vehicle", min_width=320)
        payload = infer_vehicle(
            enhanced,
            camera_id=job.camera_id,
            prepared=True,
            enhancement=enhancement,
            model=settings.vehicle_attribute_model,
            attributes_only=True,
        )
        latency = (time.perf_counter() - started) * 1000.0
        db = SessionLocal()
        try:
            if db.get(Camera, job.camera_id) is None or not job.observation_id:
                return
            apply_refinement(db, job.observation_id, payload)
            add_attempt(
                db, camera_id=job.camera_id, run_id=job.run_id, track_id=job.track_id,
                frame_index=job.frame_index, source_pts_ms=job.source_pts_ms,
                stage="vehicle_attribute", reason_code="attribute_refined" if payload.get("vehicle_type") or payload.get("vehicle_color") else "attribute_empty",
                detector=job.detector, recognizer="ollama_vehicle_attribute",
                model_id=payload.get("model_id") or "", box=job.box, quality=job.quality,
                raw_output=payload.get("raw_response") or "", confidence=float(payload.get("confidence") or 0.0),
                latency_ms=latency, accepted=False,
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return
    if job.kind == "vision_only":
        payload = infer_vision_only_frame(job.image, camera_id=job.camera_id)
        evidence_image = payload.pop("_evidence_bgr", None)
        raw = str(payload.get("chosen_plate") or "")
        norm = normalize(raw)
        selected = next(
            (
                value for value in (payload.get("models") or {}).values()
                if normalize(str(value.get("plate_text") or "")) == norm
            ),
            {},
        )
        confidence = float(payload.get("confidence") or selected.get("confidence") or 0.0)
        model_id = f"ollama:{payload.get('chosen_model') or 'vision-only'}"
        enhancement = payload.get("enhancement") or {}
        evidence_crop = evidence_image if evidence_image is not None else job.image
        vision_only = payload
        recognizer = "ollama_vision_only"
        raw_output = str(selected.get("raw_response") or raw)
        parse_status = str(selected.get("parse_status") or "")
        skipped = str(selected.get("skipped") or "")
        if not selected:
            skipped = next(
                (str(value.get("skipped")) for value in (payload.get("models") or {}).values() if value.get("skipped")),
                "",
            )
    else:
        enhanced, enhancement = enhance_for_vision(
            job.image, profile="plate" if job.detector in {"fast_alpr", "lpdnet", "opencv_plate"} else "vehicle",
            min_width=400 if job.detector in {"fast_alpr", "lpdnet", "opencv_plate"} else 320,
        )
        if job.detector in {"fast_alpr", "lpdnet", "opencv_plate"}:
            read = infer_bgr(enhanced, prepared=True, enhancement=enhancement)
            raw, norm, confidence = read.plate_raw, read.plate_norm, float(read.confidence or 0.0)
            model_id = read.model_id
        else:
            payload = infer_vehicle(enhanced, camera_id=job.camera_id, prepared=True, enhancement=enhancement)
            raw = str(payload.get("plate_raw") or payload.get("plate_norm") or "")
            norm = normalize(str(payload.get("plate_norm") or raw))
            confidence = float(payload.get("confidence") or 0.0)
            model_id = str(payload.get("model_id") or "ollama:vehicle")
        evidence_crop = job.image
        vision_only = None
        recognizer = "ollama_vision"
        raw_output = str(payload.get("raw_response") or raw) if job.detector not in {"fast_alpr", "lpdnet", "opencv_plate"} else raw
        parse_status = str(payload.get("parse_status") or "") if job.detector not in {"fast_alpr", "lpdnet", "opencv_plate"} else "json"
        skipped = str(payload.get("skipped") or "") if job.detector not in {"fast_alpr", "lpdnet", "opencv_plate"} else ""
    latency = (time.perf_counter() - started) * 1000.0
    skip_reason = {
        "error": "cloud_error", "cloud_disabled": "authentication_failed",
        "disabled": "model_unavailable", "unavailable": "model_unavailable",
    }.get(skipped, skipped)
    reason, accepted, persist = classify_cloud_read(
        norm, confidence, skipped=skip_reason, parse_status=parse_status,
    )
    if job.kind == "vision_only":
        evidence_original = ""
        evidence_processed = save_crop(evidence_crop, job.camera_id, norm or "unread-cloud", label="processed-context")
    else:
        evidence_original = save_crop(job.image, job.camera_id, norm or "unread-cloud", label="original")
        processed_image = enhanced
        evidence_processed = save_crop(processed_image, job.camera_id, norm or "unread-cloud", label="processed")
    evidence = evidence_original or evidence_processed
    db = SessionLocal()
    try:
        camera = db.get(Camera, job.camera_id)
        if camera is None:
            return
        add_attempt(
            db, camera_id=job.camera_id, run_id=job.run_id, track_id=job.track_id,
            frame_index=job.frame_index, source_pts_ms=job.source_pts_ms,
            stage="cloud_verification", reason_code=reason, detector=job.detector,
            recognizer=recognizer, model_id=model_id, box=job.box, quality=job.quality,
            raw_output=raw_output, plate_norm=norm, syntax=syntax_ok(norm), confidence=confidence,
            latency_ms=latency, accepted=accepted, evidence_path=evidence_processed or evidence,
            enhanced_shape=evidence_crop.shape[:2],
        )
        if persist:
            persist_sighting(
                db, camera, plate_raw=raw, plate_norm=norm, plate_voted=norm,
                syntax=True, confidence=confidence, model_id=model_id, model_hash="cloud",
                evidence_path=evidence, run_id=job.run_id, frame_index=job.frame_index,
                passage_id=job.track_id, source_pts_ms=job.source_pts_ms,
                provider=recognizer, ingest_time=datetime.now(timezone.utc), box=job.box,
                frame_shape=job.frame_shape, vehicle_type=job.vehicle_type,
                vehicle_color=job.vehicle_color, vision_only=vision_only,
                enhancement=enhancement, unreadable_reason="" if accepted else reason,
                recognition={
                    "detector": job.detector, "recognizer": recognizer,
                    "quality": job.quality, "reason": reason, "latency_ms": latency,
                    "evidence_original": evidence_original,
                    "evidence_processed": evidence_processed,
                    "processed_is_non_generative": True,
                },
            )
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


_SKIP_REASONS = {
    "busy",
    "throttled",
    "timeout",
    "model_unavailable",
    "authentication_failed",
    "cloud_error",
}


def classify_cloud_read(
    plate_norm: str,
    confidence: float,
    *,
    skipped: str = "",
    parse_status: str = "",
) -> tuple[str, bool, bool]:
    """Map a cloud OCR result to (reason, auto-accept, persist as sighting).

    A Gujarat plate such as GJ08AV5178 is syntax-valid. Rejection at conf 0.60
    is low_confidence, not syntax_invalid. Auto-alerts still require the
    confirmation threshold; the sighting is stored for review.
    """
    from app.services.plates import syntax_ok
    from app.services.vehicle_event import is_recordable_plate

    recordable = is_recordable_plate(plate_norm)
    accepted = recordable and float(confidence or 0.0) >= float(settings.plate_confirmation_median_confidence)
    if accepted:
        return "candidate", True, True
    if skipped in _SKIP_REASONS:
        return skipped, False, False
    if parse_status == "invalid_json":
        return "invalid_json", False, False
    if not plate_norm:
        return "ocr_empty", False, False
    if not syntax_ok(plate_norm):
        return "syntax_invalid", False, False
    if float(confidence or 0.0) < float(settings.plate_confirmation_median_confidence):
        return "low_confidence", False, True
    return "review", False, True


def _failure_reason(exc: Exception) -> str:
    text = str(exc).lower()
    if "timeout" in text:
        return "timeout"
    if "401" in text or "403" in text or "api_key" in text or "auth" in text:
        return "authentication_failed"
    if "404" in text or "410" in text or "unavailable" in text or "no model" in text:
        return "model_unavailable"
    if "busy" in text:
        return "busy"
    if "thrott" in text:
        return "throttled"
    return "cloud_error"


def _record_job_failure(job: CloudJob, exc: Exception) -> None:
    from app.database import SessionLocal
    from app.models import Camera
    from app.services.recognition_diagnostics import add_attempt

    db = SessionLocal()
    try:
        if db.get(Camera, job.camera_id) is None:
            return
        add_attempt(
            db, camera_id=job.camera_id, run_id=job.run_id, track_id=job.track_id,
            frame_index=job.frame_index, source_pts_ms=job.source_pts_ms,
            stage="cloud_verification", reason_code=_failure_reason(exc), detector=job.detector,
            recognizer="ollama_vision", box=job.box, quality=job.quality,
            raw_output=str(exc)[:500], accepted=False,
        )
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


cloud_verifier = CloudVerifierQueue()


def _plate_job_allowed(job: CloudJob) -> bool:
    """Queued plate work must honour a newly-disabled runtime safety switch."""
    from app.database import SessionLocal
    from app.models import Camera
    from app.services.recognition_policy import effective

    db = SessionLocal()
    try:
        return effective(db, db.get(Camera, job.camera_id))
    finally:
        db.close()
