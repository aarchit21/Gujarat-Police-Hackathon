from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import require_operator, require_vendor
from app.config import ROOT, settings
from app.database import SessionLocal, database_status, get_db, init_db
from app.models import Alert, AuditEvent, Camera, Sighting, WatchlistEntry
from app.security import evidence_relpath_is_safe, redact_url
from app.services.activity import cameras_active_at
from app.services.capacity import capacity_snapshot, measure_government_decode, start_accessible_workers
from app.services.catalogue import backfill_catalogue_display, sync_catalogue
from app.services.demo import autostart_if_configured
from app.services.snapshot import grab_snapshot
from app.services.cost import estimate as estimate_cost
from app.services.coverage import coverage
from app.services.ollama_vision import vision_status
from app.services.serialize import ist_label, parse_vehicle_blob, utc_iso
from app.services.vehicle_event import build_vehicle_event, is_recordable_plate
from app.services.match import observed_plates, rematch_watchlist_entry
from app.services.pipeline import analyze_camera
from app.services.plates import normalize
from app.services.processing import select_processing_route
from app.services.reports import as_csv, as_json, sighting_rows
from app.services.serialize import alert_json, camera_public, inferred_links, plate_keys, sighting_json
from app.services.map_match import map_match_status
from app.services.vehicle import vehicle_csv, vehicle_day, vehicle_geojson
from app.services.vendor import VendorIngestError, ingest_vendor_event
from app.services.workers import manager

STATIC = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    db = SessionLocal()
    try:
        backfill_catalogue_display(db)
        db.commit()
        autostart_if_configured(manager, db)
    finally:
        db.close()
    yield
    manager.stop_all()


app = FastAPI(title=settings.app_name, version="0.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
init_db()


class AlertPatch(BaseModel):
    status: str = Field(pattern="^(new|acknowledged|confirmed|rejected)$")


class CameraPatch(BaseModel):
    source_type: str | None = None
    source_uri: str | None = None
    substream_uri: str | None = None
    status: str | None = None
    status_reason: str | None = None
    priority_class: str | None = None
    processing_mode: str | None = None
    analytics_policy: str | None = None
    compute_target: str | None = None
    network_class: str | None = None
    target_analysis_fps: float | None = None
    clock_offset_ms: float | None = None


class WatchlistIn(BaseModel):
    plate_raw: str
    purpose: str = "stolen_vehicle"
    priority: str = "high"
    notes: str = ""
    active: bool = True
    rematch: bool = True


class WatchlistPatch(BaseModel):
    active: bool | None = None
    purpose: str | None = None
    priority: str | None = None
    notes: str | None = None
    rematch: bool = False


class CostIn(BaseModel):
    camera_count: float | None = None
    avg_bitrate_kbps: float | None = None
    target_analysis_fps: float | None = None
    active_cameras: float | None = None
    measured_worker_fps: float | None = None
    gpu_hourly_cost: float | None = None
    storage_cost_per_gb: float | None = None
    evidence_events_per_day: float | None = None
    avg_evidence_size_kb: float | None = None
    selected_frame_jpeg_kb: float | None = None
    share_vendor_metadata: float | None = None
    share_local_worker: float | None = None
    share_remote_gpu: float | None = None
    share_shared_regional: float | None = None
    share_central_on_demand: float | None = None


class PreviewIn(BaseModel):
    protocol: str = "hls"


def _cov(db: Session) -> dict:
    snap = manager.snapshot()
    return coverage(
        db,
        open_captures=snap["open_captures"],
        preview_count=snap["preview_count"],
        queued=snap["queued_count"],
    )


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    snap = manager.snapshot()
    return {
        "app": settings.app_name,
        "architecture": settings.architecture,
        "solo_p0": True,
        "analysis_fps_hypothesis": settings.analysis_fps,
        "tesseract": "disabled",
        "database": database_status(),
        "remote_inference_configured": bool(settings.remote_inference_url),
        "ollama_vision": vision_status(),
        "ingest_catalogue_url": redact_url(settings.ingest_catalogue_url),
        "catalogue_host": settings.catalogue_host(),
        "catalogue_auth_mode": settings.cctv_auth_mode or "none",
        "cctv_token_configured": bool(settings.cctv_access_token),
        "map_match": map_match_status(),
        "demo_autostart": settings.demo_autostart_workers,
        "workers": snap,
        "capacity": capacity_snapshot(db),
        **_cov(db),
    }


@app.get("/api/coverage")
def api_coverage(db: Session = Depends(get_db)):
    return _cov(db)


@app.get("/api/cameras")
def list_cameras(db: Session = Depends(get_db)):
    rows = list(db.scalars(select(Camera).order_by(Camera.id)))
    return [
        camera_public(
            c,
            preview_active=manager.preview_active(c.id),
            worker_state=manager.worker_state(c.id),
        )
        for c in rows
    ]


@app.patch("/api/cameras/{camera_id}")
def patch_camera(
    camera_id: str,
    body: CameraPatch,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    camera = db.get(Camera, camera_id)
    if camera is None:
        raise HTTPException(404, "camera not found")
    data = body.model_dump(exclude_none=True)
    if "source_uri" in data:
        camera.protected_rtsp_url_or_reference = data["source_uri"]
    for field, value in data.items():
        setattr(camera, field, value)
    db.add(AuditEvent(actor=actor, action="camera_patch", detail=f"{camera_id} {data}"))
    db.commit()
    return {"ok": True, "id": camera_id, "route": select_processing_route(camera)}


@app.post("/api/cameras/{camera_id}/analyze")
def api_analyze(
    camera_id: str,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    db.add(AuditEvent(actor=actor, action="analyze", detail=camera_id))
    db.commit()
    return analyze_camera(db, camera_id)


@app.post("/api/analyze-active")
def analyze_active(
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    cameras = list(db.scalars(select(Camera)))
    runnable = [c for c in cameras if c.source_type in {"image_dir", "file"} and c.source_uri]
    results = [analyze_camera(db, c.id) for c in runnable]
    db.add(AuditEvent(actor=actor, action="analyze_active", detail=f"ran={len(results)}"))
    db.commit()
    return {"ran": len(results), "results": results, "coverage": _cov(db)}


@app.get("/api/watchlist")
def list_watchlist(db: Session = Depends(get_db)):
    rows = list(db.scalars(select(WatchlistEntry).order_by(WatchlistEntry.id)))
    return [
        {
            "id": w.id,
            "plate_raw": w.plate_raw,
            "plate_norm": w.plate_norm,
            "purpose": w.purpose,
            "priority": w.priority,
            "authority": w.authority,
            "active": w.active,
            "notes": w.notes,
        }
        for w in rows
    ]


@app.post("/api/watchlist")
def add_watchlist(
    body: WatchlistIn,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    key = normalize(body.plate_raw)
    if not key:
        raise HTTPException(400, "plate is empty after normalisation")
    entry = db.scalar(select(WatchlistEntry).where(WatchlistEntry.plate_norm == key))
    created = entry is None
    if entry is None:
        entry = WatchlistEntry(
            plate_raw=body.plate_raw,
            plate_norm=key,
            purpose=body.purpose,
            priority=body.priority,
            notes=body.notes,
            active=body.active,
        )
        db.add(entry)
        db.flush()
    else:
        entry.active = body.active
        entry.purpose = body.purpose or entry.purpose
        entry.priority = body.priority or entry.priority
        if body.notes:
            entry.notes = body.notes
    db.add(AuditEvent(actor=actor, action="watchlist_add", detail=entry.plate_norm))
    rematch = {"scanned": 0, "alerts_created": 0}
    if body.rematch and entry.active:
        rematch = rematch_watchlist_entry(db, entry)
    db.commit()
    return {
        "ok": True,
        "id": entry.id,
        "plate_norm": entry.plate_norm,
        "created": created,
        "rematch": rematch,
    }


@app.patch("/api/watchlist/{watchlist_id}")
def patch_watchlist(
    watchlist_id: int,
    body: WatchlistPatch,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    entry = db.get(WatchlistEntry, watchlist_id)
    if entry is None:
        raise HTTPException(404, "watchlist entry not found")
    data = body.model_dump(exclude_none=True)
    rematch_flag = data.pop("rematch", False)
    for field, value in data.items():
        setattr(entry, field, value)
    db.add(AuditEvent(actor=actor, action="watchlist_patch", detail=f"{watchlist_id} {data}"))
    rematch = {"scanned": 0, "alerts_created": 0}
    if rematch_flag and entry.active:
        rematch = rematch_watchlist_entry(db, entry)
    db.commit()
    return {"ok": True, "id": entry.id, "active": entry.active, "rematch": rematch}


@app.get("/api/observed-plates")
def api_observed_plates(db: Session = Depends(get_db)):
    return observed_plates(db)


@app.get("/api/vehicle-events")
def list_vehicle_events(
    limit: int = Query(default=50, ge=1, le=500),
    valid_only: bool = True,
    db: Session = Depends(get_db),
):
    rows = list(db.scalars(select(Sighting).order_by(Sighting.source_time)))
    records = []
    rejected = []
    for s in rows:
        payload = parse_vehicle_blob(getattr(s, "vehicle_json", None))
        if payload is None:
            payload = build_vehicle_event(camera=s.camera, sighting=s)
        if is_recordable_plate(s.plate_norm):
            records.append(payload)
        else:
            rejected.append(
                {
                    "plate": s.plate_norm or s.plate_raw,
                    "camera_id": s.camera_id,
                    "observed_at": utc_iso(s.source_time),
                    "observed_at_ist": ist_label(s.source_time),
                    "provider": s.model_id or s.provider,
                    "reason": "overlay or not an Indian plate",
                }
            )
    empty_reason = ""
    if not records:
        last = rejected[-1] if rejected else None
        empty_reason = (
            "No valid vehicle JSON yet. A record is stored only when Ollama reads an Indian plate "
            "(e.g. GJ01AB1234). Tesseract is not used."
            + (f" Last discarded {last['plate']} on {last['camera_id']}." if last else "")
        )
    return {
        "records": records[-limit:],
        "valid_count": len(records),
        "rejected_overlay": rejected[-12:],
        "empty_reason": empty_reason,
        "ollama": vision_status(),
    }


@app.get("/api/sightings")
def list_sightings(
    plate: str | None = None,
    limit: int | None = Query(default=None, ge=1, le=500),
    db: Session = Depends(get_db),
):
    rows = list(db.scalars(select(Sighting).order_by(Sighting.source_time)))
    if plate:
        key = normalize(plate)
        rows = [s for s in rows if key in plate_keys(s)]
    if limit:
        rows = rows[-limit:]
    return [sighting_json(s) for s in rows]


@app.get("/api/vehicles/{plate}")
def vehicle_history(
    plate: str,
    day: str | None = None,
    start: str | None = None,
    end: str | None = None,
    routes: bool = False,
    db: Session = Depends(get_db),
):
    return vehicle_day(db, plate, day=day, start=start, end=end, include_routes=routes)


@app.get("/api/vehicles/{plate}/possible-routes")
def vehicle_possible_routes(
    plate: str,
    day: str | None = None,
    start: str | None = None,
    end: str | None = None,
    db: Session = Depends(get_db),
):
    payload = vehicle_day(db, plate, day=day, start=start, end=end, include_routes=True)
    dumped = json.dumps(payload)
    for attr in ("google_maps_api_key", "mapbox_access_token", "geoapify_api_key"):
        secret = (getattr(settings, attr, "") or "").strip()
        if secret and secret in dumped:
            raise HTTPException(500, "route payload leaked a secret")
    return payload


@app.get("/api/vehicles/{plate}/export.csv")
def vehicle_export_csv(
    plate: str,
    day: str | None = None,
    start: str | None = None,
    end: str | None = None,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    payload = vehicle_day(db, plate, day=day, start=start, end=end, include_routes=False)
    db.add(AuditEvent(actor=actor, action="vehicle_csv", detail=payload["plate_norm"]))
    db.commit()
    return PlainTextResponse(vehicle_csv(payload), media_type="text/csv")


@app.get("/api/vehicles/{plate}/export.geojson")
def vehicle_export_geojson(
    plate: str,
    day: str | None = None,
    start: str | None = None,
    end: str | None = None,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    payload = vehicle_day(db, plate, day=day, start=start, end=end, include_routes=True)
    db.add(AuditEvent(actor=actor, action="vehicle_geojson", detail=payload["plate_norm"]))
    db.commit()
    return vehicle_geojson(payload)


@app.get("/api/cameras/active-at")
def api_active_at(
    at: str | None = None,
    start: str | None = None,
    end: str | None = None,
    window_minutes: int = 30,
    db: Session = Depends(get_db),
):
    return cameras_active_at(db, at=at, start=start, end=end, window_minutes=window_minutes)


@app.get("/api/alerts")
def list_alerts(db: Session = Depends(get_db)):
    rows = list(db.scalars(select(Alert).order_by(Alert.created_at.desc())))
    return [alert_json(a, db.get(Camera, a.camera_id)) for a in rows]


@app.patch("/api/alerts/{alert_id}")
def patch_alert(
    alert_id: int,
    body: AlertPatch,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(404, "alert not found")
    alert.status = body.status
    db.add(AuditEvent(actor=actor, action="alert_review", detail=f"{alert_id} -> {body.status}"))
    db.commit()
    return {"ok": True, "id": alert_id, "status": alert.status}


@app.get("/api/audit")
def list_audit(db: Session = Depends(get_db)):
    rows = list(db.scalars(select(AuditEvent).order_by(AuditEvent.id.desc()).limit(100)))
    return [{"id": e.id, "at": e.at.isoformat(), "actor": e.actor, "action": e.action, "detail": e.detail} for e in rows]


@app.get("/api/workers")
def list_workers():
    return manager.snapshot()


@app.post("/api/workers/{camera_id}/start")
def start_worker(
    camera_id: str,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    return manager.start(db, camera_id, actor=actor)


@app.post("/api/workers/{camera_id}/stop")
def stop_worker(
    camera_id: str,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    return manager.stop(db, camera_id, actor=actor)


@app.post("/api/workers/stop-all")
def stop_all_workers(actor: str = Depends(require_operator)):
    manager.stop_all()
    return {"ok": True, "actor": actor}


class StartAccessibleIn(BaseModel):
    decode_ok_only: bool = True


@app.post("/api/workers/start-accessible")
def api_start_accessible(
    body: StartAccessibleIn | None = None,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    payload = body or StartAccessibleIn()
    return start_accessible_workers(manager, db, actor=actor, decode_ok_only=payload.decode_ok_only)


@app.post("/api/capacity/measure")
def api_capacity_measure(
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    db.add(AuditEvent(actor=actor, action="capacity_measure_request", detail="sequential government decode probe"))
    db.commit()
    return measure_government_decode(db)


@app.get("/api/capacity")
def api_capacity(db: Session = Depends(get_db)):
    return capacity_snapshot(db)


@app.post("/api/catalogue/sync")
def api_catalogue_sync(
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    db.add(AuditEvent(actor=actor, action="catalogue_sync_request", detail=redact_url(settings.ingest_catalogue_url)))
    db.commit()
    return sync_catalogue(db)


@app.get("/api/catalogue/status")
def catalogue_status(db: Session = Depends(get_db)):
    cov = _cov(db)
    return {
        "url": redact_url(settings.ingest_catalogue_url),
        "host": settings.catalogue_host(),
        "auth_mode": settings.cctv_auth_mode or "none",
        "synced_at": cov.get("catalogue_synced_at"),
        "last_error": cov.get("catalogue_last_error"),
        "last_http_status": cov.get("catalogue_last_http_status"),
        "government_catalogue_count": cov.get("government_catalogue_count"),
        "catalogue_live_count": cov.get("catalogue_live_count"),
        "analytics_active_count": cov.get("analytics_active_count"),
        "catalogue_live_is_not_analytics_active": True,
        "hardcoded_50": False,
    }


@app.post("/api/vendor/events")
async def vendor_events(
    request: Request,
    db: Session = Depends(get_db),
    actor: str = Depends(require_vendor),
):
    raw = await request.body()
    if len(raw) > settings.vendor_max_payload_bytes:
        raise HTTPException(413, "vendor payload too large")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, "invalid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(400, "vendor event must be a JSON object")
    try:
        return ingest_vendor_event(db, payload, actor=actor)
    except VendorIngestError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/cost/estimate")
def cost_estimate(body: CostIn):
    return estimate_cost(body.model_dump(exclude_none=True))


@app.get("/api/reports/sightings.json")
def report_json(db: Session = Depends(get_db), actor: str = Depends(require_operator)):
    db.add(AuditEvent(actor=actor, action="report_json", detail="sightings"))
    db.commit()
    body = as_json(sighting_rows(db))
    return Response(content=body, media_type="application/json")


@app.get("/api/reports/sightings.csv")
def report_csv(db: Session = Depends(get_db), actor: str = Depends(require_operator)):
    db.add(AuditEvent(actor=actor, action="report_csv", detail="sightings"))
    db.commit()
    return PlainTextResponse(as_csv(sighting_rows(db)), media_type="text/csv")


@app.get("/api/evidence")
def get_evidence(
    rel: str = Query(..., min_length=1, max_length=400),
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    try:
        path = evidence_relpath_is_safe(rel)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not path.is_file():
        raise HTTPException(404, "evidence not found")
    db.add(AuditEvent(actor=actor, action="evidence_access", detail=rel[:200]))
    db.commit()
    return FileResponse(path)


@app.get("/api/cameras/{camera_id}/snapshot")
def camera_snapshot(
    camera_id: str,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    camera = db.get(Camera, camera_id)
    if camera is None:
        raise HTTPException(404, "camera not found")
    result = grab_snapshot(camera)
    if not result.get("ok") or not result.get("jpeg"):
        raise HTTPException(503, result.get("error") or "no snapshot")
    db.add(AuditEvent(actor=actor, action="snapshot", detail=camera_id))
    db.commit()
    return Response(
        content=result["jpeg"],
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Snapshot-Source": str(result.get("source") or ""),
        },
    )


@app.post("/api/cameras/{camera_id}/preview")
def start_preview(
    camera_id: str,
    body: PreviewIn,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    camera = db.get(Camera, camera_id)
    if camera is None:
        raise HTTPException(404, "camera not found")
    result = manager.start_preview(camera, body.protocol)
    db.add(AuditEvent(actor=actor, action="preview_start", detail=f"{camera_id} {body.protocol}"))
    db.commit()
    return result


@app.post("/api/cameras/{camera_id}/preview/stop")
def stop_preview(
    camera_id: str,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    result = manager.stop_preview(camera_id)
    db.add(AuditEvent(actor=actor, action="preview_stop", detail=camera_id))
    db.commit()
    return result


@app.get("/api/diagnostics/{camera_id}")
def camera_diagnostics(camera_id: str, db: Session = Depends(get_db)):
    camera = db.get(Camera, camera_id)
    if camera is None:
        raise HTTPException(404, "camera not found")
    from app.services.ingest import diagnostics as feed_diag

    return {
        **feed_diag(camera, protocol=camera.active_protocol or "", error=camera.last_error),
        "catalogue_live": camera.catalogue_live,
        "decode_status": camera.decode_status,
        "analytics_active": camera.analytics_active,
        "preview_active": manager.preview_active(camera_id),
        "error_time_utc": datetime.now(timezone.utc).isoformat(),
    }
