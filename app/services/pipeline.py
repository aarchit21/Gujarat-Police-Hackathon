"""Process a camera source into persisted sightings and exact-match alerts."""
from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.models import AuditEvent, Camera, Sighting
from app.services.anpr import MODEL_ID, load_bgr, local_model_hash, read_frame
from app.services.evidence import save_crop
from app.services.ingest import (
    SourceOpenError,
    diagnostics,
    iter_live_frames,
    open_video_source,
    resize_for_inference,
    scale_box,
)
from app.services.match import match_sighting
from app.services.ollama_vision import OllamaVisionError, infer_bgr, should_use_vision
from app.services.plates import normalize, syntax_ok, vote
from app.services.processing import select_processing_route, target_fps
from app.services.remote import RemoteInferenceError, infer_jpeg
from app.services.timing import PassageClock, PtsSampler, source_time_from_ingest


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
    )
    db.add(sighting)
    db.flush()
    if not sighting.id:
        raise RuntimeError("sighting row was not persisted")
    alert, created = match_sighting(db, sighting)
    camera.last_frame_at = ingest
    camera.source_pts_ms = source_pts_ms
    camera.last_pts_ms = source_pts_ms
    return sighting, alert, created


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
        self.clock = PassageClock(gap_ms=settings.passage_gap_ms, jump_ms=settings.pts_jump_reset_ms)
        self.passage = f"{camera.id}-{self.run_id}-{self.clock.passage_serial}"
        self.raws: list[str] = []
        self.created = 0
        self.alerts = 0
        self.seen = 0
        self.sampled = 0
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
        self.reader = read_fn or read_frame
        self.remote_client = remote_client

    def reset_passage(self, pts_ms: float | None, reason: str) -> None:
        self.raws.clear()
        self.sampler.reset()
        self.passage = f"{self.camera.id}-{self.run_id}-{self.clock.passage_serial}"
        self.db.add(
            AuditEvent(
                action="scene_discontinuity",
                detail=json.dumps({"camera_id": self.camera.id, "pts_ms": pts_ms, "reason": reason, "events": self.clock.events[-3:]}),
            )
        )

    def push(self, frame_index: int, bgr, pts_ms: float | None) -> Sighting | None:
        self.seen += 1
        if bgr is None:
            return None
        action = self.clock.observe(pts_ms)
        if action == "reset":
            self.reset_passage(pts_ms, "pts_discontinuity")
        if not self.sampler.should_take(pts_ms):
            return None
        self.sampled += 1
        h, w = bgr.shape[:2]
        if not self.camera.width:
            self.camera.width = w
            self.camera.height = h
        small, scale = resize_for_inference(bgr)
        plate_raw, plate_norm, conf, crop, box, model_id, model_hash, provider = _read_plate(
            self.camera,
            small,
            provider_kind=self.provider_kind,
            reader=self.reader,
            remote_client=self.remote_client,
            local_hash=self.local_hash,
        )
        box = scale_box(box, scale)
        if not plate_raw and not plate_norm:
            return None
        self.raws.append(plate_raw or plate_norm)
        voted = vote(self.raws[-5:])
        evidence = save_crop(crop, self.camera.id, plate_norm or voted)
        sighting, _alert, alert_created = persist_sighting(
            self.db,
            self.camera,
            plate_raw=plate_raw,
            plate_norm=plate_norm or voted,
            plate_voted=voted,
            syntax=syntax_ok(plate_norm or voted),
            confidence=conf,
            model_id=model_id,
            model_hash=model_hash,
            evidence_path=evidence,
            run_id=self.run_id,
            frame_index=frame_index,
            passage_id=self.passage,
            source_pts_ms=pts_ms,
            provider=provider,
            box=box,
            frame_shape=(h, w),
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
        camera.last_error = str(exc)
        camera.analytics_active = False
        db.add(AuditEvent(action="analyze_error", detail=f"{camera.id}: {exc}"[:2000]))
        db.commit()
        raise
    finally:
        if not keep_active:
            camera.analytics_active = False
            if db.is_active:
                db.commit()


def _read_plate(
    camera: Camera,
    bgr,
    *,
    provider_kind: str,
    reader,
    remote_client,
    local_hash: str,
) -> tuple[str, str, float, Any, tuple[int, int, int, int] | None, str, str, str]:
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
            return (
                remote.plate_raw,
                normalize(remote.plate_raw),
                remote.confidence,
                crop,
                remote.box,
                remote.model_id,
                remote.model_hash,
                "remote_gpu",
            )
        except RemoteInferenceError as exc:
            camera.last_error = str(exc)
            if not settings.remote_fallback_local:
                return "", "", 0.0, None, None, "", "", "remote_gpu"
    read = reader(bgr)
    plate_raw, plate_norm, conf, crop, box, model_id, model_hash, provider = (
        read.plate_raw,
        read.plate_norm,
        read.confidence,
        read.crop_bgr,
        read.box,
        MODEL_ID,
        local_hash,
        "local",
    )
    if should_use_vision(camera, syntax_ok(plate_norm)):
        try:
            # False OpenCV contours are often tiny; send the full inference frame.
            use_crop = (
                crop is not None
                and getattr(crop, "size", 0)
                and crop.shape[0] >= 40
                and crop.shape[1] >= 100
            )
            target = crop if use_crop else bgr
            vision = infer_bgr(target)
            if syntax_ok(vision.plate_norm) or (vision.plate_norm and not syntax_ok(plate_norm)):
                plate_raw = vision.plate_raw or vision.plate_norm
                plate_norm = vision.plate_norm
                conf = max(conf, vision.confidence)
                model_id = vision.model_id
                model_hash = vision.model_hash
                provider = "ollama_vision"
        except OllamaVisionError as exc:
            camera.last_error = str(exc)
    return plate_raw, plate_norm, conf, crop, box, model_id, model_hash, provider


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
