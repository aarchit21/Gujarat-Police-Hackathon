"""Process a camera source into persisted sightings and exact-match alerts."""
from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import AuditEvent, Camera, Sighting
from app.services.anpr import (
    anpr_crops,
    crop_box_width,
    enhance_for_vision,
    estimate_vehicle_color,
    load_bgr,
    local_model_hash,
)
from app.services.evidence import save_context_crop, save_crop
from app.services.ingest import (
    SourceOpenError,
    diagnostics,
    iter_live_frames,
    open_video_source,
    resize_for_inference,
    scale_box,
)
from app.services.match import approved_special_format, match_sighting
from app.services.cpu_anpr import confidence_gate, cpu_plate_candidates, localization_gate
from app.services.cloud_verifier_queue import CloudJob, cloud_verifier
from app.services.plate_tracking import PlateTrackManager
from app.services.recognition_diagnostics import add_attempt, count_event
from app.services.ollama_vision import OllamaVisionError, infer_bgr, infer_vehicle, infer_vision_only_frame
from app.services.vehicle_event import build_vehicle_event, is_recordable_plate
from app.services.vehicle_observations import upsert_observation
from app.services.recognition_policy import effective as plate_recognition_effective
from app.services.yolo_detect import detect_vehicles
from app.services.plates import normalize, syntax_ok, vote
from app.services.processing import select_processing_route, target_fps
from app.services.remote import RemoteInferenceError, infer_jpeg
from app.services.timing import PassageClock, PtsSampler, source_time_from_ingest
import time as _time_mod


def _vision_only_due(proc: FrameProcessor) -> bool:
    try:
        from app.services.workers import manager

        if not getattr(manager, "vision_only", False) and not bool(getattr(settings, "vision_only_enabled", False)):
            return False
    except Exception:
        if not bool(getattr(settings, "vision_only_enabled", False)):
            return False
    gap = float(getattr(settings, "vision_only_interval_seconds", 8.0) or 8.0)
    now = _time_mod.monotonic()
    if now - float(getattr(proc, "_last_vision_only", 0.0) or 0.0) < gap:
        return False
    proc._last_vision_only = now
    return True


def persist_sighting(
    db: Session,
    camera: Camera,
    *,
    plate_raw: str,
    plate_norm: str,
    plate_voted: str,
    syntax: bool,
    confidence: float,
    model_id: str,
    model_hash: str,
    evidence_path: str,
    run_id: str,
    frame_index: int,
    passage_id: str,
    source_pts_ms: float | None,
    provider: str,
    ingest_time: datetime | None = None,
    box: tuple[int, int, int, int] | None = None,
    frame_shape: tuple[int, int] | None = None,
    vendor_event_id: str | None = None,
    vendor_payload_hash: str = "",
    vehicle_type: str = "",
    vehicle_make: str = "",
    vehicle_model: str = "",
    vehicle_color: str = "",
    unreadable_reason: str = "",
    gemma: dict | None = None,
    vision_only: dict | None = None,
    enhancement: dict | None = None,
    character_confidences: list[float] | None = None,
    recognition: dict | None = None,
    vehicle_observation_id: int | None = None,
) -> tuple[Sighting, object | None, bool]:
    ingest = ingest_time or datetime.now(timezone.utc)
    source_time, _offset_applied = source_time_from_ingest(ingest, camera.clock_offset_ms)
    sighting = Sighting(
        camera_id=camera.id,
        passage_id=passage_id,
        source_time=source_time,
        ingest_time=ingest,
        source_pts_ms=source_pts_ms,
        plate_raw=plate_raw,
        plate_norm=plate_norm or plate_voted,
        plate_voted=plate_voted,
        syntax_ok=syntax,
        confidence=confidence,
        model_id=model_id,
        model_hash=model_hash,
        evidence_path=evidence_path,
        run_id=run_id,
        frame_index=frame_index,
        provider=provider,
        vendor_event_id=vendor_event_id,
        vendor_payload_hash=vendor_payload_hash,
        bbox_x=box[0] if box else None,
        bbox_y=box[1] if box else None,
        bbox_w=box[2] if box else None,
        bbox_h=box[3] if box else None,
        frame_width=frame_shape[1] if frame_shape else camera.width,
        frame_height=frame_shape[0] if frame_shape else camera.height,
        vehicle_type=vehicle_type,
        vehicle_make=vehicle_make,
        vehicle_model=vehicle_model,
        vehicle_color=vehicle_color,
        vehicle_observation_id=vehicle_observation_id,
    )
    db.add(sighting)
    db.flush()
    if not sighting.id:
        raise RuntimeError("sighting row was not persisted")
    _apply_track_vote(db, sighting, plate_norm, bool(syntax))
    extras = {
        "vehicle_type": vehicle_type,
        "vehicle_make": vehicle_make,
        "vehicle_model": vehicle_model,
        "vehicle_color": vehicle_color,
        "unreadable_reason": unreadable_reason,
        "gemma": gemma or {},
        "vision_only": vision_only or {},
        "enhancement": enhancement or {},
        "recognition": recognition or {},
    }
    eligible_confidence = _confidence_is_eligible(confidence, character_confidences)
    identity_eligible = bool(sighting.syntax_ok or approved_special_format(db, sighting.plate_norm))
    supports = list(
        db.scalars(
            select(Sighting).where(
                Sighting.camera_id == camera.id,
                Sighting.passage_id == passage_id,
                Sighting.plate_norm == sighting.plate_norm,
                Sighting.confidence >= settings.plate_confirmation_median_confidence,
            ).order_by(Sighting.id)
        )
    ) if identity_eligible and eligible_confidence and sighting.plate_norm else []
    supports = [
        support for support in supports
        if support.id == sighting.id
        or (
            isinstance(support.vehicle_json, dict)
            and (support.vehicle_json.get("confirmation") or {}).get("status") in {"pending", "confirmed"}
        )
    ]
    distinct: list[Sighting] = []
    for support in supports:
        if any(_same_observation(support, prior) for prior in distinct):
            continue
        distinct.append(support)
    confirmed = len(distinct) >= max(2, int(settings.plate_confirmation_min_frames))
    confirmation = {
        "status": "confirmed" if confirmed else ("pending" if identity_eligible and eligible_confidence else "review"),
        "support_count": len(distinct),
        "required_frames": max(2, int(settings.plate_confirmation_min_frames)),
        "minimum_gap_ms": float(settings.plate_confirmation_min_gap_ms),
        "supporting_sighting_ids": [row.id for row in distinct[-5:]],
        "reader_agreement": (recognition or {}).get("reader_agreement", ""),
    }
    extras["confirmation"] = confirmation
    event = build_vehicle_event(camera=camera, sighting=sighting, extras=extras)
    event["watchlist_matched"] = False
    sighting.vehicle_json = event
    db.flush()
    alert, created = None, False
    if confirmed and (is_recordable_plate(plate_norm or plate_voted) or approved_special_format(db, plate_norm or plate_voted)):
        alert, created = match_sighting(db, sighting)
    event["watchlist_matched"] = bool(created or alert)
    sighting.vehicle_json = event
    camera.last_frame_at = ingest
    camera.source_pts_ms = source_pts_ms
    camera.last_pts_ms = source_pts_ms
    return sighting, alert, created


def _apply_track_vote(db: Session, sighting: Sighting, plate_norm: str, syntax: bool) -> None:
    """Character-wise majority across syntax-valid reads of the same track."""
    reads: list[str] = []
    if syntax and plate_norm:
        reads.append(plate_norm)
    priors = list(
        db.scalars(
            select(Sighting.plate_norm).where(
                Sighting.camera_id == sighting.camera_id,
                Sighting.passage_id == sighting.passage_id,
                Sighting.id != sighting.id,
                Sighting.syntax_ok.is_(True),
                Sighting.plate_norm != "",
            ).order_by(Sighting.id.desc()).limit(7)
        )
    )
    reads.extend(str(item) for item in priors if item)
    if len(reads) < 2:
        return
    consensus = vote(reads)
    if not (syntax_ok(consensus) or approved_special_format(db, consensus)):
        return
    sighting.plate_voted = consensus
    if consensus == plate_norm or reads.count(consensus) >= 2:
        sighting.plate_norm = consensus
        sighting.syntax_ok = True


def _confidence_is_eligible(confidence: float, char_confidences: list[float] | None) -> bool:
    import statistics

    values = [max(0.0, min(1.0, float(v))) for v in (char_confidences or [])]
    if values:
        return (
            statistics.median(values) >= settings.plate_confirmation_median_confidence
            and min(values) >= settings.plate_confirmation_min_char_confidence
        )
    return float(confidence or 0.0) >= settings.plate_confirmation_median_confidence


def _same_observation(a: Sighting, b: Sighting) -> bool:
    if a.frame_index == b.frame_index:
        return True
    if a.source_pts_ms is None or b.source_pts_ms is None:
        return False
    return abs(float(a.source_pts_ms) - float(b.source_pts_ms)) < float(settings.plate_confirmation_min_gap_ms)


def iter_image_dir_frames(camera: Camera) -> Iterator[tuple[int, Any, float]]:
    uri = Path(camera.source_uri)
    if not uri.is_dir():
        return
    files = sorted(p for p in uri.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    interval = settings.own_feed_synthetic_frame_interval_ms
    for i, path in enumerate(files):
        bgr = load_bgr(path)
        if bgr is None:
            continue
        yield i, bgr, float(i * interval)


def iter_file_frames(camera: Camera) -> Iterator[tuple[int, Any, float]]:
    import cv2

    uri = Path(camera.source_uri)
    if not uri.is_file():
        return
    cap = cv2.VideoCapture(str(uri))
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pts = cap.get(cv2.CAP_PROP_POS_MSEC)
            yield idx, frame, float(pts) if pts is not None else float(idx * settings.own_feed_synthetic_frame_interval_ms)
            idx += 1
    finally:
        cap.release()


def iter_camera_frames(camera: Camera) -> Iterator[tuple[int, Any, float]]:
    if camera.source_type == "image_dir":
        yield from iter_image_dir_frames(camera)
    elif camera.source_type == "file":
        yield from iter_file_frames(camera)


class FrameProcessor:
    """Keeps PTS sampling, passage id and character-vote state across frames."""

    def __init__(self, db: Session, camera: Camera, *, run_id: str | None = None, read_fn=None, remote_client=None):
        self.db = db
        self.camera = camera
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.fps = max(target_fps(camera), 0.1)
        self.sampler = PtsSampler(interval_ms=1000.0 / self.fps)
        self.base_sample_interval_ms = self.sampler.interval_ms
        self.active_track_until_pts: float | None = None
        self.clock = PassageClock(gap_ms=settings.passage_gap_ms, jump_ms=settings.pts_jump_reset_ms)
        self.passage = f"{camera.id}-{self.run_id}-{self.clock.passage_serial}"
        self.raws: list[str] = []
        self.created = 0
        self.vehicle_created = 0
        self.alerts = 0
        self.seen = 0
        self.sampled = 0
        self.passage_logged_unreadable = False
        self.unreadable_tracks: set[str] = set()
        self.route = select_processing_route(camera)
        self.provider_kind = self.route["worker_kind"]
        if camera.source_type in {"image_dir", "file"} and self.provider_kind == "deferred":
            self.provider_kind = "local_worker"
            self.route = {
                **self.route,
                "worker_kind": "local_worker",
                "selected": "local_worker",
                "reason": "explicit file/own-feed analysis on this host",
            }
        self.local_hash = local_model_hash()
        self.reader = read_fn
        self.remote_client = remote_client
        self._last_vision_only = 0.0
        self._last_cloud_queue = 0.0
        self.tracker = PlateTrackManager(camera.id, self.run_id)
        self.vehicle_tracker = PlateTrackManager(camera.id, f"{self.run_id}-vehicle")
        # Per-track attribute evidence, keyed by vehicle track id.
        self.attribute_tracks: dict[str, object] = {}

    def reset_passage(self, pts_ms: float | None, reason: str) -> None:
        self.raws.clear()
        self.tracker.reset()
        self.vehicle_tracker.reset()
        self.attribute_tracks.clear()
        self.sampler.reset()
        self.passage_logged_unreadable = False
        self.unreadable_tracks.clear()
        self.passage = f"{self.camera.id}-{self.run_id}-{self.clock.passage_serial}"
        self.db.add(
            AuditEvent(
                action="scene_discontinuity",
                detail=json.dumps({"camera_id": self.camera.id, "pts_ms": pts_ms, "reason": reason, "events": self.clock.events[-3:]}),
            )
        )

    def observe_attributes(self, *, observation, track_id, frame, box, det, frame_index, pts_ms,
                           frame_boxes=None) -> None:
        """Score this crop, classify it if it is good enough, and re-aggregate.

        Runs the OpenVINO attribute model locally. Never calls an LLM or VLM,
        and never blocks the frame loop on a network request. A failure here
        leaves the observation in place with its existing attributes.
        """
        if not settings.vattr_enabled:
            return
        from app.services.crop_quality import score_crop
        from app.services.track_aggregate import CropObservation, TrackAccumulator
        from app.services.vehicle_attributes import classify_crop, extend_box
        from app.services.vehicle_observations import apply_attributes

        accumulator = self.attribute_tracks.get(track_id)
        if accumulator is None:
            accumulator = TrackAccumulator(track_id)
            self.attribute_tracks[track_id] = accumulator
        accumulator.note_frame(frame_index, pts_ms)

        x1, y1, x2, y2 = extend_box(box, frame.shape)
        body = frame[y1:y2, x1:x2]
        quality = score_crop(
            body, box=box, frame_shape=frame.shape, detector_confidence=det.confidence,
            crop_box=(x1, y1, x2 - x1, y2 - y1), other_boxes=frame_boxes,
        )
        if not quality.eligible:
            accumulator.reject(quality.reason)
            return
        candidate = CropObservation(
            frame_index=frame_index, pts_ms=pts_ms, quality=quality,
            detector_type=det.vehicle_type, detector_confidence=float(det.confidence or 0.0),
            crop=body.copy(), box=box,
        )
        if not accumulator.offer(candidate):
            return
        try:
            for pending in accumulator.needs_attributes():
                if pending.crop is not None:
                    pending.attributes = classify_crop(pending.crop)
        except Exception as exc:
            count_event("vehicle_attribute_error")
            self.db.add(AuditEvent(
                action="vehicle_attribute_error",
                detail=json.dumps({"camera_id": self.camera.id, "track_id": track_id, "error": str(exc)[:200]}),
            ))
            return
        # Provisional while the track is live; it is re-written on each better
        # crop and settles once the track stops producing them.
        apply_attributes(self.db, observation.id, accumulator.aggregate())

    def push(self, frame_index: int, bgr, pts_ms: float | None) -> Sighting | None:
        self.seen += 1
        if bgr is None:
            count_event("no_frame")
            return None
        action = self.clock.observe(pts_ms)
        if action == "reset":
            self.reset_passage(pts_ms, "pts_discontinuity")
        if (
            pts_ms is not None and self.active_track_until_pts is not None
            and pts_ms > self.active_track_until_pts
        ):
            self.sampler.interval_ms = self.base_sample_interval_ms
            self.active_track_until_pts = None
        if not self.sampler.should_take(pts_ms):
            count_event("sample_not_due")
            return None
        self.sampled += 1
        h, w = bgr.shape[:2]
        if not self.camera.width:
            self.camera.width = w
            self.camera.height = h
        live = self.camera.source_type in {"rtsp", "hls", "onvif"}
        max_w = 1920 if live else settings.inference_max_width
        small, scale = resize_for_inference(bgr, max_width=max_w)
        # Vehicle observations are the primary live-feed output.  Detect all
        # configured vehicle boxes once per sampled frame and retain one best
        # representative record per local track.
        vehicle_rows = []
        detections = detect_vehicles(small, max_detections=settings.vehicle_max_detections)
        # Native-pixel boxes for every vehicle in this frame, so each crop can
        # be checked for how much of a *different* vehicle it contains.
        frame_boxes = [
            scale_box((d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1), scale) for d in detections
        ]
        for det in detections:
            small_box = (det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1)
            vehicle_box = scale_box(small_box, scale)
            vehicle_track = self.vehicle_tracker.assign(
                vehicle_box,
                pts_ms,
                {"quality": {"score": float(det.confidence or 0.0)}},
            )
            observation, created = upsert_observation(
                self.db,
                self.camera,
                run_id=self.run_id,
                track_id=vehicle_track,
                frame_index=frame_index,
                source_pts_ms=pts_ms,
                box=vehicle_box,
                frame_shape=(h, w),
                frame=bgr,
                crop=det.crop,
                detector="yolov8n",
                detector_confidence=det.confidence,
                vehicle_type=det.vehicle_type,
            )
            vehicle_rows.append((vehicle_box, observation, det))
            if created:
                self.vehicle_created += 1
            # Deterministic attributes, computed locally on quality-gated crops
            # and aggregated across the track. This replaces the Ollama VLM
            # refinement job that used to be enqueued here.
            self.observe_attributes(
                observation=observation,
                track_id=vehicle_track,
                frame=bgr,
                box=vehicle_box,
                det=det,
                frame_index=frame_index,
                pts_ms=pts_ms,
                frame_boxes=frame_boxes,
            )
        if detections:
            boosted_fps = max(
                self.fps,
                min(float(settings.active_track_max_fps), self.fps * max(1.0, float(settings.active_track_fps_multiplier))),
            )
            self.sampler.interval_ms = 1000.0 / max(0.1, boosted_fps)
            if pts_ms is not None:
                self.active_track_until_pts = pts_ms + 1000.0 * float(settings.active_track_hold_seconds)
        # ANPR is optional and has a persisted operator safety switch.  The
        # vehicle-first path above remains active when it is disabled.
        if not plate_recognition_effective(self.db, self.camera):
            if not detections:
                count_event("no_vehicle")
            return None
        result = _read_plate(
            self.camera,
            small,
            provider_kind=self.provider_kind,
            reader=self.reader,
            remote_client=self.remote_client,
            local_hash=self.local_hash,
            allow_cloud=not live,
        )
        plate_raw = result["plate_raw"]
        plate_norm = result["plate_norm"]
        conf = result["confidence"]
        crop = result["crop"]
        box = scale_box(result["box"], scale)
        vehicle_observation_id = None
        if vehicle_rows:
            def _distance(item):
                candidate = item[0]
                if not box or not candidate:
                    return 0
                return abs((candidate[0] + candidate[2] / 2) - (box[0] + box[2] / 2)) + abs((candidate[1] + candidate[3] / 2) - (box[1] + box[3] / 2))
            vehicle_observation_id = min(vehicle_rows, key=_distance)[1].id
        track_id = self.tracker.assign(box, pts_ms, result)
        if box:
            self.passage = track_id
        recordable = is_recordable_plate(plate_norm)
        vehicle_or_plate_seen = result.get("detector") in {"yolov8n", "fast_alpr", "opencv_plate"} or bool(result.get("vehicle_type"))
        if vehicle_or_plate_seen:
            boosted_fps = max(
                self.fps,
                min(
                    float(settings.active_track_max_fps),
                    self.fps * max(1.0, float(settings.active_track_fps_multiplier)),
                ),
            )
            self.sampler.interval_ms = 1000.0 / max(0.1, boosted_fps)
            if pts_ms is not None:
                self.active_track_until_pts = pts_ms + 1000.0 * float(settings.active_track_hold_seconds)
        unreadable = str(result.get("unreadable_reason") or "")
        vision_only = None
        if self.reader is None and live and _vision_only_due(self):
            queued = cloud_verifier.enqueue(CloudJob(
                kind="vision_only", camera_id=self.camera.id, run_id=self.run_id,
                # Vision-only has no localization/tracker evidence. Keep each
                # frame isolated so two different vehicles can never confirm.
                track_id=f"{track_id}-vo-{frame_index}", frame_index=frame_index, source_pts_ms=pts_ms,
                image=bgr.copy(), box=box, frame_shape=(h, w), detector="vision_only",
                priority=float(result.get("quality", {}).get("score", 0.0)),
            ))
            vision_only = {"queued": queued, "asynchronous": True}
        now = _time_mod.monotonic()
        if (
            self.reader is None and live and not recordable and crop is not None
            and result.get("detector") in {"fast_alpr", "lpdnet"}
            and bool(result.get("localization_accepted"))
            and now - self._last_cloud_queue >= float(settings.ollama_live_interval_seconds)
        ):
            queued = cloud_verifier.enqueue(CloudJob(
                kind="plate", camera_id=self.camera.id, run_id=self.run_id,
                track_id=track_id, frame_index=frame_index, source_pts_ms=pts_ms,
                image=crop.copy(), box=box, frame_shape=(h, w),
                detector=result.get("detector") or "",
                vehicle_type=result.get("vehicle_type") or "",
                vehicle_color=result.get("vehicle_color") or "",
                quality=result.get("quality") or {},
                priority=float(result.get("quality", {}).get("score", 0.0)),
            ))
            if queued:
                self._last_cloud_queue = now
                unreadable = unreadable or "consensus_pending"
        if live:
            if recordable:
                pass
            elif vehicle_or_plate_seen:
                if track_id in self.unreadable_tracks and not vision_only:
                    count_event(unreadable or "ocr_empty")
                    return None
                self.passage_logged_unreadable = True
                self.unreadable_tracks.add(track_id)
                unreadable = unreadable or "no_plate"
            elif vision_only:
                unreadable = unreadable or "vision_only"
            else:
                count_event(unreadable or "no_plate_candidate")
                return None
        elif not plate_raw and not plate_norm:
            count_event(unreadable or "ocr_empty")
            return None
        if recordable:
            self.raws.append(plate_raw or plate_norm)
        voted = vote(self.raws[-5:]) if self.raws else (plate_norm or "")
        quality = result.get("quality") if isinstance(result.get("quality"), dict) else {}
        context_evidence = save_context_crop(bgr, box, self.camera.id, plate_norm or voted or "unread")
        if context_evidence:
            quality = {**quality, "context_evidence_path": context_evidence}
            result["quality"] = quality
        localized_plate = bool(
            result.get("detector") in {"fast_alpr", "lpdnet"} and result.get("localization_accepted")
        )
        # In live diagnostics, evidence_path is reserved for a detector-backed
        # native plate crop. YOLO-only regions are context, not plate evidence.
        plate_evidence = save_crop(crop, self.camera.id, plate_norm or voted or "unread") if (not live or localized_plate) else ""
        sighting_evidence = plate_evidence or context_evidence
        reason_code = (
            "candidate" if recordable else unreadable or result.get("quality", {}).get("reason") or "ocr_empty"
        )
        add_attempt(
            self.db,
            camera_id=self.camera.id,
            run_id=self.run_id,
            track_id=track_id,
            frame_index=frame_index,
            source_pts_ms=pts_ms,
            stage="decision" if recordable else "recognition",
            reason_code=reason_code,
            detector=result.get("detector") or "",
            recognizer=result.get("recognizer") or result.get("provider") or "",
            model_id=result.get("model_id") or "",
            model_hash=result.get("model_hash") or "",
            box=box,
            quality=quality or None,
            raw_output=result.get("raw_output") or plate_raw,
            plate_norm=plate_norm,
            syntax=syntax_ok(plate_norm),
            character_confidences=result.get("character_confidences") or [],
            confidence=conf,
            latency_ms=float(result.get("latency_ms") or 0.0),
            accepted=recordable,
            evidence_path=plate_evidence,
            enhanced_shape=crop.shape[:2] if crop is not None and getattr(crop, "shape", None) is not None else None,
        )
        sighting, _alert, alert_created = persist_sighting(
            self.db,
            self.camera,
            plate_raw=plate_raw,
            plate_norm=plate_norm or voted,
            plate_voted=voted,
            syntax=syntax_ok(plate_norm or voted),
            confidence=conf,
            model_id=result["model_id"],
            model_hash=result["model_hash"],
            evidence_path=sighting_evidence,
            run_id=self.run_id,
            frame_index=frame_index,
            passage_id=self.passage,
            source_pts_ms=pts_ms,
            provider=result["provider"],
            box=box,
            frame_shape=(h, w),
            vehicle_type=result.get("vehicle_type") or "",
            vehicle_make=result.get("vehicle_make") or "",
            vehicle_model=result.get("vehicle_model") or "",
            vehicle_color=result.get("vehicle_color") or "",
            unreadable_reason="" if recordable else unreadable,
            gemma=result.get("gemma") if isinstance(result.get("gemma"), dict) else None,
            vision_only=vision_only,
            enhancement=result.get("enhancement") if isinstance(result.get("enhancement"), dict) else None,
            character_confidences=result.get("character_confidences") or [],
            recognition={
                "detector": result.get("detector") or "",
                "recognizer": result.get("recognizer") or result.get("provider") or "",
                "quality": result.get("quality") or {},
                "character_confidences": result.get("character_confidences") or [],
                "raw_output": result.get("raw_output") or plate_raw,
                "reason": reason_code,
                "latency_ms": float(result.get("latency_ms") or 0.0),
                "reader_agreement": result.get("reader_agreement") or "",
            },
            vehicle_observation_id=vehicle_observation_id,
        )
        self.created += 1
        if alert_created:
            self.alerts += 1
        return sighting

    def summary(self) -> dict:
        return {
            "ok": True,
            "camera_id": self.camera.id,
            "run_id": self.run_id,
            "frames_seen": self.seen,
            "frames_sampled": self.sampled,
            "analysis_fps_hypothesis": self.fps,
            "timing": "pts",
            "sightings": self.created,
            "vehicle_observations": self.vehicle_created,
            "alerts": self.alerts,
            "analytics_active": bool(self.camera.analytics_active),
            "route": self.route,
        }


def process_frame_iter(
    db: Session,
    camera: Camera,
    frames: Iterator[tuple[int, Any, float | None]],
    *,
    run_id: str | None = None,
    read_fn: Callable[..., Any] | None = None,
    remote_client=None,
    stop_check: Callable[[], bool] | None = None,
    max_frames: int | None = None,
    keep_active: bool = False,
) -> dict:
    proc = FrameProcessor(db, camera, run_id=run_id, read_fn=read_fn, remote_client=remote_client)
    camera.analytics_active = True
    camera.last_error = ""
    try:
        for frame_index, bgr, pts_ms in frames:
            if stop_check and stop_check():
                break
            if max_frames is not None and proc.seen >= max_frames:
                break
            proc.push(frame_index, bgr, pts_ms)
        camera.decode_status = "ok" if proc.seen else camera.decode_status
        camera.decode_tested_at = datetime.now(timezone.utc)
        if proc.seen:
            camera.status = "connected"
            camera.status_reason = "decoded on this host"
        db.add(
            AuditEvent(
                action="analyze_camera",
                detail=f"{camera.id} seen={proc.seen} sampled={proc.sampled} sightings={proc.created} alerts={proc.alerts} run={proc.run_id}",
            )
        )
        if not keep_active:
            camera.analytics_active = False
        db.commit()
        out = proc.summary()
        out["analytics_active"] = bool(camera.analytics_active)
        return out
    except Exception as exc:
        db.rollback()
        camera.last_error = str(exc)
        camera.analytics_active = False
        db.add(AuditEvent(action="analyze_error", detail=f"{camera.id}: {exc}"[:2000]))
        db.commit()
        raise
    finally:
        if not keep_active:
            camera.analytics_active = False
            if db.is_active:
                try:
                    db.commit()
                except Exception:
                    db.rollback()


def _empty_read() -> dict:
    return {
        "plate_raw": "",
        "plate_norm": "",
        "confidence": 0.0,
        "crop": None,
        "box": None,
        "model_id": "",
        "model_hash": "",
        "provider": "",
        "vehicle_type": "",
        "vehicle_make": "",
        "vehicle_model": "",
        "vehicle_color": "",
        "detector": "",
        "unreadable_reason": "",
        "enhancement": {},
        "recognizer": "",
        "quality": {},
        "character_confidences": [],
        "latency_ms": 0.0,
        "raw_output": "",
        "reader_agreement": "",
        "localization_accepted": False,
    }


def _yolo_only_read(item: dict, *, reason: str, vision: dict | None = None) -> dict:
    vis = vision or {}
    crop = item.get("crop")
    color = vis.get("vehicle_color") or estimate_vehicle_color(crop) or estimate_vehicle_color(item.get("body_crop"))
    vtype = vis.get("vehicle_type") or item.get("vehicle_type") or "unknown"
    if vtype in {"", "unknown"} and item.get("vehicle_type"):
        vtype = item["vehicle_type"]
    raw = vis.get("plate_raw") or vis.get("plate_norm") or ""
    norm = vis.get("plate_norm") or ""
    if raw and not is_recordable_plate(norm):
        reason = "partial"
    out = _empty_read()
    out.update(
        {
            "crop": crop,
            "box": item.get("box"),
            "plate_raw": raw,
            "plate_norm": norm if is_recordable_plate(norm) else "",
            "vehicle_type": vtype,
            "vehicle_make": vis.get("vehicle_make") or "",
            "vehicle_model": vis.get("vehicle_model") or "",
            "vehicle_color": color or "",
            "detector": "yolov8n",
            "unreadable_reason": reason,
            "model_id": vis.get("model_id") or "yolov8n",
            "model_hash": vis.get("model_hash") or "",
            "provider": vis.get("provider") or "yolo",
            "confidence": float(vis.get("confidence") or 0.0),
            "gemma": vis.get("gemma")
            or {
                "called": bool(vis.get("model_id") or vis.get("provider") == "ollama_vision"),
                "skipped": reason if reason in {"busy", "throttled", "too_small"} else "",
                "plate_text": raw or vis.get("plate_text") or "",
                "type": vtype,
                "color": color or "",
                "crop_w": int(crop.shape[1]) if crop is not None and getattr(crop, "shape", None) is not None else 0,
                "crop_h": int(crop.shape[0]) if crop is not None and getattr(crop, "shape", None) is not None else 0,
                "enhancement": vis.get("enhancement") or item.get("enhancement") or {},
            },
            "enhancement": vis.get("enhancement") or item.get("enhancement") or {},
        }
    )
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Small dependency-free overlap check for duplicate local detector passes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    if not ix or not iy:
        return False
    intersection = ix * iy
    return intersection / max(1, min(aw * ah, bw * bh)) >= 0.5


def _read_plate(
    camera: Camera,
    bgr,
    *,
    provider_kind: str,
    reader,
    remote_client,
    local_hash: str,
    allow_cloud: bool = True,
) -> dict:
    if reader is not None:
        read = reader(bgr)
        out = _empty_read()
        out.update(
            {
                "plate_raw": read.plate_raw,
                "plate_norm": read.plate_norm,
                "confidence": read.confidence,
                "crop": read.crop_bgr,
                "box": read.box,
                "model_id": "test-reader",
                "model_hash": local_hash,
                "provider": "local",
            }
        )
        return out
    live = camera.source_type in {"rtsp", "hls", "onvif"}
    crops = anpr_crops(bgr, live=live)

    # On live CCTV a white/yellow contour anywhere in the frame is often a
    # lamp, HUD, advert or road marking. Only the semantic FastALPR detector
    # may localise a live plate. Run it once on the native frame and again on
    # each YOLO vehicle region so distant vehicles have a tighter search area.
    cpu_candidates = cpu_plate_candidates(bgr, allow_opencv_fallback=not live)
    if live:
        for item in crops:
            if item.get("detector") != "yolov8n":
                continue
            vehicle = item.get("body_crop")
            vehicle_box = item.get("crop_box") or item.get("box")
            if vehicle is None or not getattr(vehicle, "size", 0) or not vehicle_box:
                continue
            ox, oy = int(vehicle_box[0]), int(vehicle_box[1])
            for candidate in cpu_plate_candidates(vehicle, allow_opencv_fallback=False):
                if candidate.box:
                    x, y, bw, bh = candidate.box
                    candidate.box = (x + ox, y + oy, bw, bh)
                cpu_candidates.append(candidate)
            bumper = None
            try:
                from app.services.anpr import bumper_roi

                bumper = bumper_roi(bgr, item.get("box"))
            except Exception:
                bumper = None
            if bumper is not None:
                bumper_crop, bumper_box = bumper
                bx, by = int(bumper_box[0]), int(bumper_box[1])
                for candidate in cpu_plate_candidates(bumper_crop, allow_opencv_fallback=False):
                    if candidate.box:
                        x, y, bw, bh = candidate.box
                        candidate.box = (x + bx, y + by, bw, bh)
                    cpu_candidates.append(candidate)
        # A full-frame and vehicle-region inference may produce the same plate.
        unique: list[Any] = []
        for candidate in cpu_candidates:
            if candidate.box and any(_boxes_overlap(candidate.box, prior.box) for prior in unique if prior.box):
                continue
            unique.append(candidate)
        cpu_candidates = unique
    cpu_candidates.sort(
        key=lambda c: (localization_gate(c) if live else confidence_gate(c), c.quality.get("score", 0), c.confidence),
        reverse=True,
    )
    best_cpu = cpu_candidates[0] if cpu_candidates else None
    if best_cpu is not None and confidence_gate(best_cpu) and (not live or localization_gate(best_cpu)):
        out = _empty_read()
        out.update(
            {
                "plate_raw": best_cpu.plate_raw,
                "plate_norm": best_cpu.plate_norm,
                "confidence": best_cpu.confidence,
                "crop": best_cpu.crop_bgr,
                "box": best_cpu.box,
                "model_id": best_cpu.model_id,
                "model_hash": best_cpu.model_hash,
                "provider": "cpu_anpr",
                "detector": best_cpu.detector,
                "recognizer": best_cpu.recognizer,
                "quality": best_cpu.quality,
                "character_confidences": best_cpu.character_confidences,
                "latency_ms": best_cpu.latency_ms,
                "raw_output": best_cpu.raw_output,
                "reader_agreement": best_cpu.reader_agreement,
                "localization_accepted": localization_gate(best_cpu) if live else best_cpu.detector in {"fast_alpr", "lpdnet"},
            }
        )
        return out
    if best_cpu is not None and best_cpu.crop_bgr is not None:
        crops.insert(0, {
            "crop": best_cpu.crop_bgr,
            "body_crop": best_cpu.crop_bgr,
            "plate_crops": [best_cpu.crop_bgr],
            "box": best_cpu.box,
            "vehicle_type": "",
            "detector": best_cpu.detector,
            "det_conf": best_cpu.confidence,
            "cpu_candidate": best_cpu,
        })
    min_w = int(getattr(settings, "min_vision_box_px", None) or settings.min_plate_width_px or 24)
    best_yolo = next((c for c in crops if c.get("detector") == "yolov8n"), None)
    last_skip = ""
    last_vision: dict | None = None
    last_item: dict | None = None
    if settings.ollama_vision_enabled and allow_cloud:
        for i, item in enumerate(crops):
            width = crop_box_width(item)
            if item.get("detector") == "yolov8n" and width < min_w:
                last_skip = "too_small"
                continue
            crop = item.get("crop")
            if crop is None:
                crop = item.get("body_crop")
            if crop is None:
                continue
            native = item.get("body_crop")
            if native is None:
                native = crop
            enhanced, enhancement = enhance_for_vision(native, profile="vehicle", min_width=320)
            item = {**item, "crop": enhanced, "enhancement": enhancement}
            vision: dict = {}
            try:
                plate_hits = list(item.get("plate_crops") or [])[:2]
                for patch in plate_hits:
                    tight, tight_enhancement = enhance_for_vision(patch, profile="plate", min_width=400)
                    read = infer_bgr(tight, prepared=True, enhancement=tight_enhancement)
                    if is_recordable_plate(read.plate_norm):
                        body_crop = item.get("body_crop")
                        vision = {
                            "plate_raw": read.plate_raw,
                            "plate_norm": read.plate_norm,
                            "confidence": read.confidence,
                            "vehicle_type": item.get("vehicle_type") or "",
                            "vehicle_color": estimate_vehicle_color(body_crop if body_crop is not None else enhanced),
                            "model_id": read.model_id,
                            "model_hash": read.model_hash,
                            "provider": "ollama_vision",
                            "enhancement": read.enhancement,
                            "recognizer": "ollama_vision",
                            "raw_output": read.plate_raw,
                        }
                        item = {**item, "crop": tight, "enhancement": tight_enhancement}
                        break
                if not is_recordable_plate((vision or {}).get("plate_norm")):
                    vision = infer_vehicle(
                        enhanced,
                        camera_id=camera.id if i == 0 else "",
                        prepared=True,
                        enhancement=enhancement,
                    )
                    if isinstance(vision, dict) and "gemma" not in vision:
                        vision["gemma"] = {
                            "called": not bool(vision.get("skipped")),
                            "skipped": vision.get("skipped") or "",
                            "plate_text": vision.get("plate_raw") or vision.get("plate_norm") or "",
                            "type": vision.get("vehicle_type") or "",
                            "color": vision.get("vehicle_color") or "",
                            "crop_w": int(enhanced.shape[1]) if enhanced is not None else 0,
                            "crop_h": int(enhanced.shape[0]) if enhanced is not None else 0,
                            "enhancement": vision.get("enhancement") or enhancement,
                        }
            except OllamaVisionError as exc:
                camera.last_error = str(exc)
                if best_yolo:
                    return _yolo_only_read(best_yolo, reason="vision_error")
                return _empty_read()
            if vision.get("skipped"):
                last_skip = str(vision.get("skipped"))
                last_item = item
                continue
            last_vision = vision
            last_item = item
            if is_recordable_plate(vision.get("plate_norm")):
                vtype = vision.get("vehicle_type") or item.get("vehicle_type") or "unknown"
                if vtype in {"", "unknown"} and item.get("vehicle_type"):
                    vtype = item["vehicle_type"]
                color = vision.get("vehicle_color") or estimate_vehicle_color(enhanced)
                out = _empty_read()
                out.update(
                    {
                        "plate_raw": vision.get("plate_raw") or vision.get("plate_norm") or "",
                        "plate_norm": vision.get("plate_norm") or "",
                        "confidence": float(vision.get("confidence") or 0.0),
                        "crop": item.get("crop") if item.get("crop") is not None else enhanced,
                        "box": item.get("box"),
                        "model_id": vision.get("model_id") or "",
                        "model_hash": vision.get("model_hash") or "",
                        "provider": "ollama_vision",
                        "vehicle_type": vtype,
                        "vehicle_make": vision.get("vehicle_make") or "",
                        "vehicle_model": vision.get("vehicle_model") or "",
                        "vehicle_color": color or "",
                        "detector": item.get("detector") or "",
                        "recognizer": vision.get("recognizer") or "ollama_vision",
                        "quality": (
                            item.get("cpu_candidate").quality
                            if item.get("cpu_candidate") is not None else {}
                        ),
                        "character_confidences": [],
                        "raw_output": vision.get("raw_output") or vision.get("plate_raw") or "",
                        "reader_agreement": "cloud_only",
                        "enhancement": vision.get("enhancement") or item.get("enhancement") or {},
                        "gemma": vision.get("gemma")
                        or {
                            "called": True,
                            "skipped": "",
                            "plate_text": vision.get("plate_raw") or vision.get("plate_norm") or "",
                            "type": vtype,
                            "color": color or "",
                            "crop_w": int(enhanced.shape[1]) if enhanced is not None else 0,
                            "crop_h": int(enhanced.shape[0]) if enhanced is not None else 0,
                            "enhancement": vision.get("enhancement") or item.get("enhancement") or {},
                        },
                    }
                )
                return out
        chosen = last_item or best_yolo
        if chosen and chosen.get("detector") == "yolov8n":
            camera.last_error = f"yolo: vehicle seen, plate {last_skip or 'unreadable'}"
            return _yolo_only_read(chosen, reason=last_skip or ("no_plate_localized" if live else "no_plate"), vision=last_vision)
        if best_cpu is not None:
            out = _empty_read()
            out.update({
                "plate_raw": best_cpu.plate_raw,
                "plate_norm": best_cpu.plate_norm if is_recordable_plate(best_cpu.plate_norm) else "",
                "confidence": best_cpu.confidence,
                "crop": best_cpu.crop_bgr,
                "box": best_cpu.box,
                "model_id": best_cpu.model_id,
                "model_hash": best_cpu.model_hash,
                "provider": "cpu_anpr",
                "detector": best_cpu.detector,
                "recognizer": best_cpu.recognizer,
                "quality": best_cpu.quality,
                "character_confidences": best_cpu.character_confidences,
                "latency_ms": best_cpu.latency_ms,
                "raw_output": best_cpu.raw_output,
                "reader_agreement": best_cpu.reader_agreement,
                "unreadable_reason": (
                    "no_plate_localized" if live and not localization_gate(best_cpu) else best_cpu.reason
                ),
                "localization_accepted": localization_gate(best_cpu) if live else best_cpu.detector in {"fast_alpr", "lpdnet"},
            })
            return out
        camera.last_error = "yolo+ollama: no vehicle crop large enough for a plate"
        return _empty_read()
    if best_cpu is not None:
        out = _empty_read()
        out.update({
            "plate_raw": best_cpu.plate_raw,
            "plate_norm": best_cpu.plate_norm if is_recordable_plate(best_cpu.plate_norm) else "",
            "confidence": best_cpu.confidence,
            "crop": best_cpu.crop_bgr,
            "box": best_cpu.box,
            "model_id": best_cpu.model_id,
            "model_hash": best_cpu.model_hash,
            "provider": "cpu_anpr",
            "detector": best_cpu.detector,
            "recognizer": best_cpu.recognizer,
            "quality": best_cpu.quality,
            "character_confidences": best_cpu.character_confidences,
            "latency_ms": best_cpu.latency_ms,
            "raw_output": best_cpu.raw_output,
            "reader_agreement": best_cpu.reader_agreement,
            "unreadable_reason": (
                "no_plate_localized" if live and not localization_gate(best_cpu) else best_cpu.reason
            ),
            "localization_accepted": localization_gate(best_cpu) if live else best_cpu.detector == "fast_alpr",
        })
        return out
    if best_yolo:
        return _yolo_only_read(best_yolo, reason="no_plate_localized" if live else "no_plate")
    if provider_kind == "remote_gpu" and settings.remote_inference_url:
        try:
            import cv2

            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                raise RemoteInferenceError("jpeg encode failed")
            remote = infer_jpeg(buf.tobytes(), camera_id=camera.id, client=remote_client)
            crop = None
            if remote.box:
                x, y, w, h = remote.box
                crop = bgr[max(0, y) : y + h, max(0, x) : x + w].copy()
            out = _empty_read()
            out.update(
                {
                    "plate_raw": remote.plate_raw,
                    "plate_norm": normalize(remote.plate_raw),
                    "confidence": remote.confidence,
                    "crop": crop,
                    "box": remote.box,
                    "model_id": remote.model_id,
                    "model_hash": remote.model_hash,
                    "provider": "remote_gpu",
                }
            )
            return out
        except RemoteInferenceError as exc:
            camera.last_error = str(exc)
            return _empty_read()
    camera.last_error = "no dedicated plate or vehicle candidate on sampled frame"
    out = _empty_read()
    out["unreadable_reason"] = "no_vehicle"
    return out


def analyze_camera(
    db: Session,
    camera_id: str,
    run_id: str | None = None,
    read_fn=None,
    remote_client=None,
    max_frames: int | None = None,
    max_seconds: float | None = None,
) -> dict:
    camera = db.get(Camera, camera_id)
    if camera is None:
        return {"ok": False, "error": "unknown camera"}

    if camera.source_type in {"rtsp", "onvif", "hls"}:
        try:
            frames = list(
                iter_live_frames(
                    camera,
                    open_fn=open_video_source,
                    max_frames=max_frames if max_frames is not None else settings.live_analyze_max_frames,
                    max_seconds=max_seconds if max_seconds is not None else settings.live_analyze_max_seconds,
                )
            )
        except SourceOpenError as exc:
            camera.status = "blocked"
            camera.analytics_active = False
            camera.decode_status = "failed"
            camera.decode_tested_at = datetime.now(timezone.utc)
            camera.last_error = json.dumps(diagnostics(camera, protocol=camera.active_protocol or "rtsp", error=str(exc)))
            db.add(AuditEvent(action="analyze_blocked", detail=f"{camera_id}: {exc}"))
            db.commit()
            return {
                "ok": False,
                "camera_id": camera_id,
                "error": str(exc),
                "frames": 0,
                "analytics_active": False,
            }
        if not frames:
            camera.status = "blocked"
            camera.analytics_active = False
            camera.decode_status = "failed"
            camera.decode_tested_at = datetime.now(timezone.utc)
            camera.last_error = "opened but no usable frame within keyframe wait"
            db.add(AuditEvent(action="analyze_blocked", detail=f"{camera_id}: no live frames"))
            db.commit()
            return {"ok": False, "camera_id": camera_id, "error": camera.last_error, "frames": 0, "analytics_active": False}
        camera.active_protocol = camera.active_protocol or "rtsp"
        return process_frame_iter(db, camera, iter(frames), run_id=run_id, read_fn=read_fn, remote_client=remote_client)

    frames = list(iter_camera_frames(camera))
    if not frames:
        camera.analytics_active = False
        camera.last_error = "no decodable frames on this host"
        db.add(AuditEvent(action="analyze_blocked", detail=f"{camera_id}: no frames"))
        db.commit()
        return {"ok": False, "camera_id": camera_id, "error": camera.last_error, "frames": 0, "analytics_active": False}

    return process_frame_iter(db, camera, iter(frames), run_id=run_id, read_fn=read_fn, remote_client=remote_client)
