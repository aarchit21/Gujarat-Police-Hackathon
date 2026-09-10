"""Day-3 host capacity: measured throughput, calibrated sampling, accessible workers.

Does not open every catalogue stream at once. Does not treat CAP_PROP_FPS as sampling truth.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import AuditEvent, Camera, SystemState
from app.services.coverage import camera_origin
from app.services.ingest import rtsp_url_for
from app.services.network_check import probe_one_rtsp
from app.services.processing import PRIORITY_CLASSES
from app.services.workers import PRIORITY_RANK, manager


def calibrated_target_fps(measured_fps: float, requested_fps: float) -> float:
    """Stay at or below measured safe throughput. Never invent 2 FPS capacity."""
    req = max(0.1, float(requested_fps or 0.1))
    measured = float(measured_fps or 0.0)
    if measured <= 0:
        return req
    safe = measured * 0.8
    return max(0.1, min(req, safe))


def runnable_cameras(db: Session, *, decode_ok_only: bool = True) -> list[Camera]:
    rows = list(db.scalars(select(Camera)))
    out = []
    for cam in rows:
        if (cam.processing_mode or "") in {"deferred", "vendor_metadata"}:
            continue
        has_source = False
        if cam.source_type in {"image_dir", "file"} and cam.source_uri:
            has_source = True
        elif cam.source_type in {"rtsp", "hls", "onvif"} and (
            rtsp_url_for(cam) or cam.hls_url
        ):
            has_source = True
        if not has_source:
            continue
        if decode_ok_only and cam.source_type in {"rtsp", "hls", "onvif"} and cam.decode_status != "ok":
            continue
        out.append(cam)
    out.sort(
        key=lambda c: (
            0 if camera_origin(c) == "government_catalogue" and c.source_type in {"rtsp", "hls", "onvif"} else 1,
            PRIORITY_RANK.get(c.priority_class or "D", 9),
            c.id,
        )
    )
    return out


def promote_decode_ok_cameras(db: Session) -> list[str]:
    promoted = []
    for cam in db.scalars(select(Camera)):
        if cam.decode_status != "ok":
            continue
        if cam.processing_mode in {"deferred", "", None}:
            cam.processing_mode = "local_worker"
            cam.analytics_policy = "continuous"
            if camera_origin(cam) == "government_catalogue" and (cam.priority_class or "D") in {"C", "D", ""}:
                cam.priority_class = "B"
            promoted.append(cam.id)
        cam.analytics_active = False
    if promoted:
        db.add(AuditEvent(action="demo_promote", detail=",".join(promoted)))
    db.commit()
    return promoted


def start_accessible_workers(manager, db: Session, *, actor: str = "operator", decode_ok_only: bool = True) -> dict:
    promoted = promote_decode_ok_cameras(db)
    selected = runnable_cameras(db, decode_ok_only=decode_ok_only)
    started, queued, skipped, failed = [], [], [], []
    for cam in selected:
        out = manager.start(db, cam.id, actor=actor)
        state = out.get("state") or ""
        if not out.get("ok"):
            failed.append({"id": cam.id, "error": out.get("error")})
        elif state == "queued":
            queued.append(cam.id)
        elif state in {"starting", "already_running"}:
            started.append(cam.id)
        else:
            skipped.append({"id": cam.id, "state": state})
    db.add(
        AuditEvent(
            actor=actor,
            action="workers_start_accessible",
            detail=f"started={len(started)} queued={len(queued)} failed={len(failed)} decode_ok_only={decode_ok_only}",
        )
    )
    db.commit()
    return {
        "ok": True,
        "decode_ok_only": decode_ok_only,
        "max_concurrent": manager.max_workers,
        "candidate_count": len(selected),
        "promoted": promoted,
        "started": started,
        "queued": queued,
        "failed": failed,
        "skipped": skipped,
        "disclaimer": "Queued cameras stay analytics_active=false. This is not a 30-camera load test.",
    }


def measure_government_decode(
    db: Session,
    *,
    limit: int | None = None,
    probe_fn=None,
    timeout: float | None = None,
    retest_failed: bool = False,
) -> dict:
    """Sequentially probe a small batch of untested catalogue RTSP URLs.

    Never fans out all streams. Batch size is independent of max concurrent workers
    so raising pin slots does not turn Measure into a 30× timeout walk.
    """
    cap = int(limit or max(1, int(getattr(settings, "measure_batch_size", None) or 4)))
    wait = float(timeout if timeout is not None else getattr(settings, "measure_probe_timeout_seconds", 6.0) or 6.0)
    probe = probe_fn or probe_one_rtsp
    gov = [
        c
        for c in db.scalars(select(Camera).order_by(Camera.id))
        if camera_origin(c) == "government_catalogue" and rtsp_url_for(c)
    ]
    untested = [c for c in gov if (c.decode_status or "untested") == "untested"]
    failed = [c for c in gov if c.decode_status == "failed"]
    already_ok = [c.id for c in gov if c.decode_status == "ok"]
    if untested:
        queue = untested
    elif retest_failed:
        failed_known = [c for c in failed if c.width]
        failed_other = [c for c in failed if not c.width]
        queue = failed_known + failed_other
    else:
        queue = []
    batch = queue[:cap]
    tested = []
    ok_ids = []
    wall = time.monotonic()
    for cam in batch:
        url = rtsp_url_for(cam)
        t0 = time.monotonic()
        result = probe(url, timeout=wait)
        elapsed = time.monotonic() - t0
        cam.decode_tested_at = datetime.now(timezone.utc)
        if result.get("ok") and result.get("frame"):
            cam.decode_status = "ok"
            cam.status = "connected"
            cam.status_reason = f"decode probe ok via {result.get('protocol') or 'rtsp'}"
            cam.active_protocol = "rtsp"
            if result.get("width"):
                cam.width = int(result["width"])
            if result.get("height"):
                cam.height = int(result["height"])
            if result.get("pts_ms") is not None:
                cam.last_pts_ms = float(result["pts_ms"])
            frames = max(1, int(result.get("frames") or 1))
            cam.measured_worker_fps = round(frames / elapsed, 3) if elapsed > 0 else None
            cam.measured_at = cam.decode_tested_at
            ok_ids.append(cam.id)
        else:
            cam.decode_status = "failed"
            cam.analytics_active = False
            cam.last_error = str(result.get("error") or "no frame")
            cam.status_reason = "decode probe failed on this host"
        tested.append(
            {
                "id": cam.id,
                "ok": cam.decode_status == "ok",
                "width": cam.width,
                "height": cam.height,
                "pts_ms": result.get("pts_ms"),
                "elapsed_s": round(elapsed, 3),
                "error": result.get("error") or "",
            }
        )
    elapsed_all = time.monotonic() - wall
    remaining_untested = sum(1 for c in gov if (c.decode_status or "untested") == "untested")
    remaining_failed = sum(1 for c in gov if c.decode_status == "failed")
    fps_values = [c.measured_worker_fps for c in db.scalars(select(Camera)) if c.id in ok_ids and c.measured_worker_fps]
    host_fps = min(fps_values) if fps_values else 0.0
    requested = float(settings.analysis_fps)
    recommended = calibrated_target_fps(host_fps, requested) if fps_values else float(_state_get(db, "recommended_target_fps") or requested)
    now = datetime.now(timezone.utc)
    if tested:
        _state(db, "measured_safe_fps", str(host_fps), now)
        _state(db, "recommended_target_fps", str(recommended), now)
        _state(db, "government_decode_tested_count", str(len(tested)), now)
    _state(db, "government_decode_ok_count", str(len(already_ok) + len(ok_ids)), now)
    if tested and recommended + 1e-9 < requested:
        for cam in db.scalars(select(Camera)):
            if cam.id in ok_ids:
                cam.target_analysis_fps = recommended
                cam.status_reason = (
                    f"sampling lowered to {recommended:.2f} fps from measured {host_fps:.2f} fps "
                    f"(requested hypothesis {requested:.2f})"
                )
    db.add(
        AuditEvent(
            action="capacity_measure",
            detail=f"tested={len(tested)} ok={len(ok_ids)} measured_fps={host_fps} recommended={recommended} retest_failed={retest_failed}",
        )
    )
    db.commit()
    if not batch:
        disclaimer = (
            f"No untested catalogue cameras left. {len(already_ok)} already decode-ok, "
            f"{remaining_failed} previously failed. Measure does not re-wait {wait:.0f}s on failed RTSP. "
            "Use Pin working cameras to run analytics. Pass retest_failed to retry dead feeds."
        )
    elif not untested and retest_failed:
        disclaimer = (
            f"Retested up to {cap} previously-failed cameras ({wait:.0f}s each, sequential). "
            "Not a 30-camera simultaneous test. Use Pin working cameras for decode-ok feeds."
        )
    else:
        disclaimer = (
            f"Sequential decode probe of the next {len(tested)} untested cameras ({wait:.0f}s cap each). "
            "Repeat Measure to walk the rest. Decode-ok is not a running worker — use Pin working cameras."
        )
    return {
        "ok": True,
        "tested": tested,
        "decode_ok": ok_ids,
        "decode_ok_count": len(ok_ids),
        "tested_count": len(tested),
        "limit": cap,
        "probe_timeout_s": wait,
        "retest_failed": retest_failed,
        "already_decode_ok": already_ok,
        "catalogue_remaining_untested": remaining_untested,
        "catalogue_remaining_failed": remaining_failed,
        "measured_safe_fps": host_fps if tested else _state_get(db, "measured_safe_fps"),
        "requested_fps_hypothesis": requested,
        "recommended_target_fps": recommended,
        "sampling_reduced": bool(tested) and recommended + 1e-9 < requested,
        "elapsed_s": round(elapsed_all, 3),
        "disclaimer": disclaimer,
    }


def _state_get(db: Session, key: str) -> str:
    row = db.get(SystemState, key)
    return row.value if row else ""


def _state(db: Session, key: str, value: str, now: datetime) -> None:
    row = db.get(SystemState, key)
    if row is None:
        db.add(SystemState(key=key, value=value, updated_at=now))
    else:
        row.value = value
        row.updated_at = now


def capacity_snapshot(db: Session) -> dict:
    def _val(key: str) -> str:
        row = db.get(SystemState, key)
        return row.value if row else ""

    return {
        "measured_safe_fps": _val("measured_safe_fps"),
        "recommended_target_fps": _val("recommended_target_fps"),
        "government_decode_ok_count": _val("government_decode_ok_count"),
        "government_decode_tested_count": _val("government_decode_tested_count"),
        "analysis_fps_hypothesis": settings.analysis_fps,
        "max_concurrent": manager.max_workers,
        "max_concurrent_captures": manager.registry.max_open,
        "priority_classes": list(PRIORITY_CLASSES),
    }
