"""Bounded in-process worker manager. No Kafka, no distributed scheduler."""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from app.config import settings
from app.database import SessionLocal
from app.models import AuditEvent, Camera
from app.services.activity import close_activity, open_activity
from app.security import hls_requires_server_credential, redact_url
from app.services.ingest import CaptureRegistry, SourceOpenError, diagnostics, open_video_source, rtsp_url_for
from app.services.pipeline import FrameProcessor, iter_camera_frames, process_frame_iter
from app.services.processing import select_processing_route
from app.services.snapshot import maybe_save_live_preview
from app.services.timing import backoff_seconds

PRIORITY_RANK = {"A": 0, "B": 1, "C": 2, "D": 3}


@dataclass
class WorkerState:
    camera_id: str
    status: str = "starting"
    reason: str = ""
    run_id: str = ""
    started_at: str = ""
    last_pts_ms: float | None = None
    frames: int = 0
    error: str = ""
    reconnect_attempt: int = 0


@dataclass
class PreviewState:
    camera_id: str
    protocol: str
    url: str
    opened_at: str


class WorkerManager:
    def __init__(self, *, max_workers: int | None = None, registry: CaptureRegistry | None = None):
        self.max_workers = int(max_workers or settings.max_concurrent_workers)
        self.registry = registry or CaptureRegistry()
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._stops: dict[str, threading.Event] = {}
        self._states: dict[str, WorkerState] = {}
        self._queue: list[str] = []
        self._previews: dict[str, PreviewState] = {}
        self.hunt_enabled = False
        self.hunt_cycle_id = ""
        self.hunt_cycle = 0
        self.hunt_target_ids: set[str] = set()
        self.hunt_visited: set[str] = set()
        self.open_fn = open_video_source
        self.sleep_fn: Callable[[float], None] = time.sleep
        self.now_fn: Callable[[], float] = time.monotonic

    def snapshot(self) -> dict:
        with self._lock:
            running = [s.__dict__.copy() for s in self._states.values() if s.camera_id in self._threads]
            queued = list(self._queue)
            previews = [
                {
                    "camera_id": p.camera_id,
                    "protocol": p.protocol,
                    "url": "" if p.protocol == "hls" and hls_requires_server_credential(p.url) else redact_url(p.url),
                    "opened_at": p.opened_at,
                }
                for p in self._previews.values()
            ]
            return {
                "max_concurrent": self.max_workers,
                "running_count": len(self._threads),
                "queued": queued,
                "queued_count": len(queued),
                "open_captures": self.registry.count(),
                "open_capture_owners": self.registry.owners(),
                "previews": previews,
                "preview_count": len(self._previews),
                "workers": running,
            }

    def preview_active(self, camera_id: str) -> bool:
        with self._lock:
            return camera_id in self._previews

    def worker_state(self, camera_id: str) -> str:
        with self._lock:
            if camera_id in self._threads:
                return self._states.get(camera_id, WorkerState(camera_id)).status
            if camera_id in self._queue:
                return "queued"
            return ""

    def begin_hunt(self, camera_ids: list[str]) -> None:
        with self._lock:
            self.hunt_enabled = True
            self.hunt_cycle_id = uuid.uuid4().hex[:12]
            self.hunt_cycle = 1
            self.hunt_target_ids = set(camera_ids)
            self.hunt_visited = set()
            for cid in camera_ids:
                if cid not in self._queue:
                    self._queue.append(cid)

    def end_hunt(self) -> None:
        with self._lock:
            ids = set(self.hunt_target_ids)
            self.hunt_enabled = False
            self.hunt_target_ids = set()
            self.hunt_visited = set()
            self._queue = [cid for cid in self._queue if cid not in ids]

    def mark_hunted(self, camera_id: str) -> None:
        with self._lock:
            if not self.hunt_enabled:
                return
            self.hunt_visited.add(camera_id)
            if self.hunt_target_ids and self.hunt_target_ids <= self.hunt_visited:
                self.hunt_cycle += 1
                self.hunt_visited = set()
            if camera_id not in self._queue and camera_id in self.hunt_target_ids:
                self._queue.append(camera_id)

    def start_preview(self, camera: Camera, protocol: str) -> dict:
        protocol = protocol.lower()
        if protocol == "whep":
            from app.services.ingest import apply_stream_userinfo

            url = apply_stream_userinfo(camera.whep_url or "")
        elif protocol in {"hls", "snapshot"}:
            url = camera.hls_url
        else:
            return {"ok": False, "error": "protocol must be whep, hls, or snapshot"}
        if protocol == "snapshot" or (protocol == "hls" and url and hls_requires_server_credential(url)):
            with self._lock:
                self._previews[camera.id] = PreviewState(
                    camera_id=camera.id,
                    protocol="snapshot",
                    url="",
                    opened_at=datetime.now(timezone.utc).isoformat(),
                )
            return {
                "ok": True,
                "preview_active": True,
                "analytics_active": bool(camera.analytics_active),
                "protocol": "snapshot",
                "snapshot": True,
                "url": f"/api/cameras/{camera.id}/snapshot",
                "preview_blocked": False,
                "hls_not_sent_to_browser": True,
                "note": "HLS credential stays server-side. Operator snapshot uses the server-side feed.",
            }
        if not url:
            return {"ok": False, "error": f"no {protocol} URL on camera; RTSP is not exposed to the browser — use Live frame"}
        with self._lock:
            self._previews[camera.id] = PreviewState(
                camera_id=camera.id,
                protocol=protocol,
                url=url,
                opened_at=datetime.now(timezone.utc).isoformat(),
            )
        return {
            "ok": True,
            "preview_active": True,
            "analytics_active": bool(camera.analytics_active),
            "protocol": protocol,
            "url": redact_url(url),
        }

    def stop_preview(self, camera_id: str) -> dict:
        with self._lock:
            self._previews.pop(camera_id, None)
        return {"ok": True, "preview_active": False}

    def start(self, db, camera_id: str, *, actor: str = "operator") -> dict:
        camera = db.get(Camera, camera_id)
        if camera is None:
            return {"ok": False, "error": "unknown camera"}
        if camera.processing_mode == "deferred":
            has_source = bool(
                (camera.source_type in {"image_dir", "file"} and camera.source_uri)
                or rtsp_url_for(camera)
                or camera.hls_url
            )
            if has_source:
                camera.processing_mode = "local_worker"
                camera.analytics_policy = "continuous"
                db.add(AuditEvent(actor=actor, action="worker_promote", detail=f"{camera_id} deferred->local_worker"))
        route = select_processing_route(camera)
        if camera.processing_mode == "vendor_metadata":
            camera.analytics_active = False
            db.add(AuditEvent(actor=actor, action="worker_skip", detail=f"{camera_id} vendor_metadata waits for events"))
            db.commit()
            return {"ok": True, "camera_id": camera_id, "state": "vendor_wait", "analytics_active": False}
        if not route["analytics_may_start"] and camera.processing_mode != "central_on_demand":
            camera.analytics_active = False
            camera.status_reason = route["reason"]
            db.commit()
            return {"ok": False, "error": route["reason"], "analytics_active": False}

        with self._lock:
            if camera_id in self._threads:
                return {"ok": True, "camera_id": camera_id, "state": "already_running"}
            needs_capture = camera.source_type in {"rtsp", "hls", "onvif"}
            slots_full = len(self._threads) >= self.max_workers
            captures_full = (
                needs_capture
                and self.registry.count() >= self.registry.max_open
                and camera_id not in self.registry.owners()
            )
            if slots_full or captures_full:
                if camera_id not in self._queue:
                    self._queue.append(camera_id)
                    self._queue.sort(key=lambda cid: PRIORITY_RANK.get(_priority(db, cid), 9))
                camera.analytics_active = False
                db.add(AuditEvent(actor=actor, action="worker_queued", detail=camera_id))
                db.commit()
                return {"ok": True, "camera_id": camera_id, "state": "queued", "analytics_active": False}
            if needs_capture and not self.registry.try_acquire(camera_id):
                if camera_id not in self._queue:
                    self._queue.append(camera_id)
                camera.analytics_active = False
                db.commit()
                return {"ok": True, "camera_id": camera_id, "state": "queued", "analytics_active": False}

            stop = threading.Event()
            self._stops[camera_id] = stop
            state = WorkerState(
                camera_id=camera_id,
                status="starting",
                run_id=uuid.uuid4().hex[:12],
                started_at=datetime.now(timezone.utc).isoformat(),
                reason=route["reason"],
            )
            self._states[camera_id] = state
            thread = threading.Thread(target=self._run, args=(camera_id, stop, state), daemon=True, name=f"anpr-{camera_id}")
            self._threads[camera_id] = thread
            if camera_id in self._queue:
                self._queue.remove(camera_id)
        thread.start()
        open_activity(db, camera, run_id=state.run_id, reason="worker_start")
        db.add(AuditEvent(actor=actor, action="worker_start", detail=camera_id))
        db.commit()
        return {"ok": True, "camera_id": camera_id, "state": "starting", "run_id": state.run_id}

    def stop(self, db, camera_id: str, *, actor: str = "operator") -> dict:
        with self._lock:
            stop = self._stops.get(camera_id)
            if stop:
                stop.set()
            if camera_id in self._queue and not self.hunt_enabled:
                self._queue.remove(camera_id)
        camera = db.get(Camera, camera_id)
        if camera is not None:
            camera.analytics_active = False
        close_activity(db, camera_id, reason="worker_stop")
        db.add(AuditEvent(actor=actor, action="worker_stop", detail=camera_id))
        db.commit()
        return {"ok": True, "camera_id": camera_id, "state": "stopping"}

    def stop_all(self) -> None:
        self.end_hunt()
        with self._lock:
            for ev in self._stops.values():
                ev.set()
            self._queue.clear()
        for thread in list(self._threads.values()):
            thread.join(timeout=2.0)

    def _run(self, camera_id: str, stop: threading.Event, state: WorkerState) -> None:
        try:
            db = SessionLocal()
            try:
                camera = db.get(Camera, camera_id)
                if camera is None:
                    return
                if camera.source_type in {"image_dir", "file"}:
                    self._run_batch(db, camera, stop, state)
                else:
                    self._run_live(db, camera, stop, state)
            finally:
                db.close()
        except Exception as exc:
            state.status = "crashed"
            state.error = str(exc)
        finally:
            db = SessionLocal()
            try:
                camera = db.get(Camera, camera_id)
                if camera is not None:
                    camera.analytics_active = False
                    if state.error and not camera.last_error:
                        camera.last_error = state.error
                    if self.hunt_enabled and camera_id in self.hunt_target_ids:
                        from app.services.hunt import mark_camera_hunted

                        mark_camera_hunted(db, camera_id)
                    close_activity(db, camera_id, reason="worker_exit")
                    db.add(AuditEvent(action="worker_exit", detail=f"{camera_id} {state.status} {state.error}"[:2000]))
                    db.commit()
            finally:
                db.close()
            with self._lock:
                self._threads.pop(camera_id, None)
                self._stops.pop(camera_id, None)
            self.registry.release(camera_id)
            if self.hunt_enabled:
                self.mark_hunted(camera_id)
            self._promote_queue()

    def _run_batch(self, db, camera: Camera, stop: threading.Event, state: WorkerState) -> None:
        state.status = "running"
        camera.analytics_active = True
        db.commit()
        frames = iter_camera_frames(camera)
        process_frame_iter(db, camera, frames, run_id=state.run_id, stop_check=stop.is_set)
        state.status = "idle"
        camera.analytics_active = False
        db.commit()

    def _run_live(self, db, camera: Camera, stop: threading.Event, state: WorkerState) -> None:
        hunting = self.hunt_enabled and camera.id in self.hunt_target_ids
        run_live_loop(
            db,
            camera,
            stop=stop,
            state=state,
            open_fn=self.open_fn,
            sleep_fn=self.sleep_fn,
            now_fn=self.now_fn,
            registry=self.registry,
            max_frames=settings.hunt_max_frames if hunting else None,
            max_seconds=settings.hunt_dwell_seconds if hunting else None,
            reconnect=True,
        )

    def _promote_queue(self) -> None:
        db = SessionLocal()
        try:
            with self._lock:
                pending = list(self._queue)
            for camera_id in pending:
                with self._lock:
                    if len(self._threads) >= self.max_workers:
                        break
                self.start(db, camera_id, actor="scheduler")
        finally:
            db.close()


def _priority(db, camera_id: str) -> str:
    cam = db.get(Camera, camera_id)
    return cam.priority_class if cam else "D"


def run_live_loop(
    db,
    camera: Camera,
    *,
    stop: threading.Event,
    state: WorkerState,
    open_fn=open_video_source,
    sleep_fn=time.sleep,
    now_fn=time.monotonic,
    registry: CaptureRegistry | None = None,
    read_fn=None,
    max_frames: int | None = None,
    max_seconds: float | None = None,
    process_fn=None,
    reconnect: bool = True,
) -> dict:
    """Reconnect with bounded exponential backoff. Cancel immediately on stop.

    Hunt mode sets reconnect=False and a dwell deadline so the slot rotates.
    """
    attempt = 0
    processed = 0
    loop_started = now_fn()
    first_frame_at: dict[str, float | None] = {"t": None}

    def timed_out() -> bool:
        if max_seconds is None:
            return False
        t0 = first_frame_at["t"]
        if t0 is None:
            limit = max(float(max_seconds), float(settings.keyframe_wait_seconds) + 10.0)
            return (now_fn() - loop_started) >= limit
        return (now_fn() - t0) >= float(max_seconds)

    while not stop.is_set() and not timed_out():
        camera.analytics_active = False
        db.commit()
        try:
            opened = open_fn(camera)
        except SourceOpenError as exc:
            attempt += 1
            camera.reconnect_count = (camera.reconnect_count or 0) + 1
            if (camera.decode_status or "") != "ok":
                camera.decode_status = "failed"
            camera.decode_tested_at = datetime.now(timezone.utc)
            camera.analytics_active = False
            camera.last_error = json.dumps(diagnostics(camera, protocol="rtsp", error=str(exc)))
            camera.active_protocol = ""
            state.reconnect_attempt = attempt
            state.error = str(exc)
            state.status = "reconnect_wait"
            db.add(AuditEvent(action="reconnect", detail=f"{camera.id} attempt={attempt} {exc}"[:2000]))
            db.commit()
            if not reconnect:
                break
            _sleep_backoff(stop, sleep_fn, attempt)
            continue
        except Exception as exc:
            attempt += 1
            camera.analytics_active = False
            camera.last_error = str(exc)
            db.commit()
            if not reconnect:
                break
            _sleep_backoff(stop, sleep_fn, attempt)
            continue

        camera.active_protocol = opened.protocol
        if opened.rtsp_error and opened.protocol == "hls":
            camera.last_error = opened.rtsp_error
        camera.decode_tested_at = datetime.now(timezone.utc)
        state.status = "running"
        got_frame = False
        join_started = now_fn()
        stable = 0
        run_id = state.run_id or uuid.uuid4().hex[:12]
        processor = None if process_fn else FrameProcessor(db, camera, run_id=run_id, read_fn=read_fn)

        try:
            while not stop.is_set() and not timed_out():
                ok, frame, pts = opened.read()
                if not ok or frame is None:
                    if not got_frame and (now_fn() - join_started) < settings.keyframe_wait_seconds:
                        continue
                    break
                got_frame = True
                if first_frame_at["t"] is None:
                    first_frame_at["t"] = now_fn()
                maybe_save_live_preview(camera.id, frame)
                camera.decode_status = "ok"
                camera.status = "connected"
                camera.status_reason = f"decoding via {opened.protocol}"
                camera.last_pts_ms = pts
                camera.source_pts_ms = pts
                camera.analytics_active = True
                stable += 1
                if stable > 15:
                    attempt = 0
                if process_fn:
                    process_fn(frame, pts)
                elif processor is not None:
                    processor.push(processed, frame, pts)
                    db.commit()
                processed += 1
                state.frames = processed
                state.last_pts_ms = pts
                if max_frames is not None and processed >= max_frames:
                    break
        finally:
            opened.release()
            camera.analytics_active = False
            db.commit()

        if stop.is_set() or not reconnect or timed_out() or (max_frames is not None and processed >= max_frames):
            break
        attempt += 1
        camera.reconnect_count = (camera.reconnect_count or 0) + 1
        state.reconnect_attempt = attempt
        state.status = "reconnect_wait"
        db.commit()
        _sleep_backoff(stop, sleep_fn, attempt)
    state.status = "stopped"
    camera.analytics_active = False
    db.commit()
    return {"frames": processed, "reconnect_attempt": attempt}


def _sleep_backoff(stop: threading.Event, sleep_fn: Callable[[float], None], attempt: int) -> None:
    delay = backoff_seconds(attempt, settings.reconnect_start_seconds, settings.reconnect_max_seconds)
    slept = 0.0
    while not stop.is_set() and slept < delay:
        step = min(0.2, delay - slept)
        sleep_fn(step)
        slept += step


manager = WorkerManager()


def reset_manager() -> WorkerManager:
    global manager
    manager.stop_all()
    manager = WorkerManager()
    return manager
