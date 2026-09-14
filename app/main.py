from __future__ import annotations

import json
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth import require_operator, require_vendor
from app.config import ROOT, settings
from app.database import SessionLocal, database_status, get_db, init_db
from app.models import Alert, AuditEvent, Camera, RecognitionAttempt, Sighting, VehicleObservation, WatchlistEntry
from app.security import evidence_relpath_is_safe, redact_secrets, redact_url
from app.services.activity import cameras_active_at
from app.services.anpr import enhancement_status
from app.services.lpdgan import lpdgan_status
from app.services.lpdnet import lpdnet_status
from app.services.cpu_anpr import cpu_anpr_status
from app.services.cloud_verifier_queue import cloud_verifier
from app.services.recognition_diagnostics import diagnostics_snapshot, serialize_attempt
from app.services.capacity import capacity_snapshot, measure_government_decode, start_accessible_workers
from app.services.catalogue import backfill_catalogue_display, sync_catalogue
from app.services.demo import autostart_if_configured
from app.services.snapshot import grab_snapshot
from app.services.cost import estimate as estimate_cost
from app.services.coverage import camera_origin, coverage
from app.services.ollama_vision import vision_status
from app.services.yolo_detect import yolo_status
from app.services.serialize import ist_label, parse_vehicle_blob, utc_iso
from app.services.vehicle_event import build_vehicle_event, is_recordable_plate
from app.services.match import observed_plates, rematch_watchlist_entry
from app.services.pipeline import analyze_camera
from app.services.plates import normalize
from app.services.processing import select_processing_route
from app.services.registry import (
    ImportError_ as RegistryImportError,
    gap_analysis,
    gap_analysis_csv,
    onboard_cameras,
)
from app.services.registry import parse_csv as parse_registry_csv
from app.services.reports import as_csv, as_json, sighting_rows
from app.services.serialize import (
    alert_json,
    alert_priority_rank,
    camera_public,
    inferred_links,
    plate_keys,
    sighting_json,
)
from app.services.map_match import map_match_status
from app.services.vehicle import parse_time, vehicle_csv, vehicle_day, vehicle_geojson
from app.services.vehicle_observations import (
    VEHICLE_COLORS,
    VEHICLE_TYPES,
    observation_json,
    observations_csv,
    search_observations,
)
from app.services.recognition_policy import (
    hydrate as hydrate_recognition_policy,
    set_camera_mode,
    set_global as set_plate_recognition_global,
    snapshot as recognition_policy_snapshot,
)
from app.services.vendor import VendorIngestError, ingest_vendor_event
from app.services.hunt import hunt_status, start_hunt, stop_hunt
from app.services.workers import hydrate_concurrency, manager

STATIC = Path(__file__).resolve().parent / "static"
#: The developer console lives OUTSIDE the /static mount on purpose. Putting it
#: under /static would leave it downloadable in production no matter what the
#: flag said, which is exactly the "hidden but still served" failure the
#: production split is meant to avoid.
DEVUI = Path(__file__).resolve().parent / "devui"

log = logging.getLogger("app.api")


def require_developer_ui() -> bool:
    """Gate every developer-only route. 404, not 403: in production these
    surfaces do not exist, and saying "forbidden" would advertise them."""
    if not settings.developer_ui_enabled():
        raise HTTPException(404, "not found")
    return True


def _cli_option(name: str) -> str:
    """Read a uvicorn command-line option, so the banner prints the URL that
    actually works rather than the config default. `--port 8010` on the command
    line does not reach Settings, and a banner naming :8000 when the server is
    on :8010 is worse than no banner at all."""
    argv = sys.argv
    for index, arg in enumerate(argv):
        if arg == name and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
    return ""


def _bound_host() -> str:
    host = _cli_option("--host") or settings.host
    # A wildcard bind has no single URL; loopback is the one that always works
    # from the machine running the server.
    return "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host


def _bound_port() -> str:
    return _cli_option("--port") or str(settings.port)


def _print_startup_banner() -> None:
    """Say which consoles are being served, and at which URL.

    This goes to `uvicorn.error` rather than a module logger on purpose:
    uvicorn's logging config only attaches handlers to its own loggers, so an
    `app.api` INFO line is silently discarded and never reaches the terminal.
    That is exactly what happened before -- developer mode was on, /dev was
    being served, and nothing on screen said so.
    """
    banner = logging.getLogger("uvicorn.error")
    base = f"http://{_bound_host()}:{_bound_port()}"
    if settings.developer_ui_enabled():
        banner.warning(
            "%s | APP_ENV=%s | developer console ENABLED\n"
            "    production console   %s/\n"
            "    developer console    %s/dev",
            settings.app_name, settings.app_env, base, base,
        )
    else:
        banner.warning(
            "%s | APP_ENV=%s | developer console NOT served (set APP_ENV=development "
            "and ENABLE_DEVELOPER_UI=true for /dev)\n"
            "    production console   %s/",
            settings.app_name, settings.app_env, base,
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    db = SessionLocal()
    try:
        backfill_catalogue_display(db)
        hydrate_recognition_policy(db)
        hydrate_concurrency(db)
        db.commit()
        autostart_if_configured(manager, db)
    finally:
        db.close()
    _print_startup_banner()
    yield
    manager.stop_all()


app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    lifespan=lifespan,
    # The interactive docs enumerate every route, its parameters and its response
    # shape. That is a useful map for an operator and an equally useful one for
    # anyone else, so it is served only where /dev is.
    docs_url=None if settings.is_production() else "/docs",
    redoc_url=None if settings.is_production() else "/redoc",
    openapi_url=None if settings.is_production() else "/openapi.json",
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
init_db()


# ---------------------------------------------------------------------------
# Authentication boundary.
#
# Read routes used to answer anonymously -- 26 of them, including vehicle
# search, plate history, alerts, the watchlist, the camera inventory with its
# coordinates, and the audit log. The sign-in screen was a client-side overlay
# (app.js hides a div); the server never knew a session existed. Anyone who
# could reach the port could read the whole corpus with curl.
#
# This is enforced here rather than as `Depends(require_operator)` on each route
# for one reason: a route added later inherits the deny, instead of shipping
# public because someone forgot a decorator. The per-route dependencies stay
# where they are -- they still supply the `actor` value for audit rows, and a
# second check costs nothing.
#
# Everything on this list has to work BEFORE anyone can sign in. Nothing else
# belongs on it.
# ---------------------------------------------------------------------------
PUBLIC_PATHS = frozenset({
    "/",           # serves the sign-in page itself
    "/healthz",    # platform liveness probe; says nothing but "ok"
    "/favicon.ico",
    "/api/ui/config",  # the page reads app_env before a token exists
    # Not public: carries its own credential. `require_vendor` checks a separate
    # token (`vendor_ingest_token`), so the operator check here would reject a
    # correctly-authenticated vendor. It is exempt from THIS gate, not from auth.
    "/api/vendor/events",
})
PUBLIC_PREFIXES = (
    "/static/",  # the CSS and JS for the sign-in page
    "/dev",      # already gated by require_developer_ui, which 404s in production
    "/docs",     # only mounted outside production (see the constructor above)
    "/redoc",
    "/openapi.json",
)


def _is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


@app.middleware("http")
async def operator_boundary(request: Request, call_next):
    if request.method == "OPTIONS" or _is_public(request.url.path):
        return await call_next(request)
    try:
        require_operator(
            authorization=request.headers.get("authorization"),
            x_operator_token=request.headers.get("x-operator-token"),
            token=request.query_params.get("token"),
        )
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return await call_next(request)


@app.get("/healthz")
def healthz():
    """Liveness only.

    `/api/health` reports model names, database type, the catalogue host and
    whether a CCTV token is configured. A hosting platform's health check needs
    none of that, and on a public host it should not be able to read it.
    """
    return {"ok": True}


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    """Never let a stack trace or an internal message reach the browser.

    The operator gets a reference they can quote; the detail goes to the server
    log, where it stays available for debugging.
    """
    reference = uuid.uuid4().hex[:12]
    log.exception("unhandled error ref=%s path=%s", reference, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Something went wrong on the server. The action was not completed.",
            "reference": reference,
        },
    )


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
    plate_recognition_mode: str | None = None


class CameraIn(BaseModel):
    """Manual registry entry (reference Model 1 onboarding)."""

    id: str
    name: str | None = None
    department: str | None = None
    city: str | None = None
    lat: float | None = None
    lng: float | None = None
    source_type: str | None = None
    source_uri: str | None = None
    substream_uri: str | None = None
    priority_class: str | None = None
    processing_mode: str | None = None
    analytics_policy: str | None = None
    network_class: str | None = None
    vendor: str | None = None
    model: str | None = None


class CameraImportIn(BaseModel):
    """Bulk onboarding. Supply exactly one of ``csv`` or ``cameras``."""

    csv: str | None = None
    cameras: list[dict] | None = None


class RecognitionSettingsIn(BaseModel):
    enabled: bool


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


# ---------------------------------------------------------------------------
# Developer console (/dev). Off by default; see Settings.developer_ui_enabled.
# The files are read from app/devui/, which is not mounted anywhere public.
# ---------------------------------------------------------------------------
DEV_FILES = {
    "console.html": "text/html; charset=utf-8",
    "console.js": "application/javascript; charset=utf-8",
    "console.css": "text/css; charset=utf-8",
    "label.js": "application/javascript; charset=utf-8",
}


@app.get("/dev")
def dev_console(_dev: bool = Depends(require_developer_ui)):
    return FileResponse(DEVUI / "console.html", media_type="text/html")


@app.get("/dev/{filename}")
def dev_asset(filename: str, _dev: bool = Depends(require_developer_ui)):
    media_type = DEV_FILES.get(filename)
    if media_type is None:
        raise HTTPException(404, "not found")
    return FileResponse(DEVUI / filename, media_type=media_type)


@app.get("/api/ui/config")
def ui_config():
    """What the production dashboard needs to render itself.

    Deliberately narrow: no model identifiers, no hostnames, no credentials,
    no capacity figures. Anything an investigator cannot act on does not
    belong on this route.
    """
    from app.services.vehicle_attributes import type_deployment_gate

    gate_passed, _reason = type_deployment_gate()
    return {
        "app_env": settings.app_env,
        "developer_ui": settings.developer_ui_enabled(),
        # Present so an engineer can tell a mis-set flag from a broken build.
        "developer_ui_requested": bool(settings.enable_developer_ui),
        "automatic_type_available": bool(gate_passed),
        "color_available": bool(settings.vattr_color_enabled),
        "demo_instance": bool(settings.demo_instance),
    }


def _observation_window(db: Session, hours: int = 24) -> dict:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    total = int(db.scalar(
        select(func.count(VehicleObservation.id)).where(VehicleObservation.first_seen_at >= since)
    ) or 0)
    pending = int(db.scalar(
        select(func.count(VehicleObservation.id)).where(
            VehicleObservation.first_seen_at >= since,
            VehicleObservation.review_status != "verified",
        )
    ) or 0)
    return {"window_hours": hours, "observations": total, "awaiting_review": pending}


def _camera_counts(cameras: list[Camera], running: set[str]) -> dict:
    checked_ok = sum(1 for c in cameras if c.decode_status == "ok")
    unreachable = sum(1 for c in cameras if c.decode_status == "failed")
    return {
        "total": len(cameras),
        "monitoring": sum(1 for c in cameras if c.analytics_active or c.id in running),
        "available": checked_ok,
        "unreachable": unreachable,
        "not_checked": len(cameras) - checked_ok - unreachable,
    }


def _feed_activity(db: Session, camera_ids: list[str], since) -> dict:
    """Counted, never estimated. An empty camera list must yield zeros, not a
    query with an empty IN clause whose result is easy to misread."""
    if not camera_ids:
        return {"observations": 0, "sightings": 0, "alerts_open": 0}
    return {
        "observations": int(db.scalar(select(func.count(VehicleObservation.id)).where(
            VehicleObservation.camera_id.in_(camera_ids),
            VehicleObservation.first_seen_at >= since,
        )) or 0),
        "sightings": int(db.scalar(select(func.count(Sighting.id)).where(
            Sighting.camera_id.in_(camera_ids),
            Sighting.source_time >= since,
        )) or 0),
        "alerts_open": int(db.scalar(select(func.count(Alert.id)).where(
            Alert.camera_id.in_(camera_ids), Alert.status == "new",
        )) or 0),
    }


@app.get("/api/ui/overview")
def ui_overview(db: Session = Depends(get_db)):
    """Dashboard figures for the production UI.

    Every number here is counted from the database or the live worker manager.
    Nothing is estimated, and a camera is never reported as online unless this
    host has actually opened its stream (`decode_status == "ok"`), which is why
    `not_checked` is its own bucket rather than being folded into offline.

    The `feeds` block splits the same totals by where the footage came from.
    The two paths demonstrate different things -- the own feed is a synthetic
    source that exercises plate-to-alert end to end, the government catalogue is
    live RTSP that exercises detection at scale -- and presenting one combined
    number invites the reader to credit each path with the other's results.
    """
    cameras = list(db.scalars(select(Camera)))
    snap = manager.snapshot()
    running = {w.get("camera_id") for w in (snap.get("workers") or []) if w.get("status") == "running"}
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    by_origin: dict[str, list[Camera]] = {}
    for camera in cameras:
        by_origin.setdefault(camera_origin(camera), []).append(camera)
    # `local_registry` cameras are neither: they ride with the government group
    # for counting so the two cards still sum to the totals above.
    own = by_origin.get("own_feed", [])
    gov = by_origin.get("government_catalogue", []) + by_origin.get("local_registry", [])
    latest = db.scalar(select(VehicleObservation).order_by(VehicleObservation.id.desc()).limit(1))
    return {
        "cameras": _camera_counts(cameras, running),
        "vehicles": _observation_window(db, 24),
        "alerts_open": int(db.scalar(select(func.count(Alert.id)).where(Alert.status == "new")) or 0),
        "watchlist_active": int(db.scalar(
            select(func.count(WatchlistEntry.id)).where(WatchlistEntry.active.is_(True))
        ) or 0),
        "last_observation_at_ist": ist_label(latest.first_seen_at) if latest else None,
        "plate_recognition_enabled": bool(hydrate_recognition_policy(db)),
        "feeds": {
            "own": {
                "cameras": _camera_counts(own, running),
                **_feed_activity(db, [c.id for c in own], since),
            },
            "government": {
                "cameras": _camera_counts(gov, running),
                **_feed_activity(db, [c.id for c in gov], since),
            },
            "window_hours": 24,
        },
    }


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    snap = manager.snapshot()
    return {
        "app": settings.app_name,
        "architecture": settings.architecture,
        "solo_p0": True,
        "analysis_fps_hypothesis": settings.analysis_fps,
        "tesseract": "enabled as CPU fallback" if settings.cpu_anpr_tesseract_enabled else "disabled",
        "cpu_anpr": cpu_anpr_status(),
        "plate_recognition": recognition_policy_snapshot(db),
        "recognition": diagnostics_snapshot(db),
        "cloud_verifier_queue": cloud_verifier.status(),
        "database": database_status(),
        "remote_inference_configured": bool(settings.remote_inference_url),
        "ollama_vision": vision_status(),
        "vision_enhancement": enhancement_status(),
        "lpdgan": lpdgan_status(),
        "lpdnet": lpdnet_status(),
        "vision_only": {
            "active": bool(getattr(manager, "vision_only", False)),
            "models": [m.strip() for m in (settings.vision_only_models or "").split(",") if m.strip()],
            "interval_seconds": settings.vision_only_interval_seconds,
        },
        "yolo_detector": yolo_status(),
        "ingest_catalogue_url": redact_url(settings.ingest_catalogue_url),
        "catalogue_host": settings.catalogue_host(),
        "catalogue_auth_mode": settings.cctv_auth_mode or "none",
        "cctv_token_configured": bool(settings.cctv_access_token),
        "map_match": map_match_status(),
        "demo_autostart": settings.demo_autostart_workers,
        "hunt": hunt_status(manager, db),
        "workers": snap,
        "capacity": capacity_snapshot(db),
        **_cov(db),
    }


@app.get("/api/coverage")
def api_coverage(db: Session = Depends(get_db)):
    return _cov(db)


@app.get("/api/settings/recognition")
def get_recognition_settings(
    db: Session = Depends(get_db),
    _actor: str = Depends(require_operator),
):
    return recognition_policy_snapshot(db)


@app.patch("/api/settings/recognition")
def patch_recognition_settings(
    body: RecognitionSettingsIn,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    enabled = set_plate_recognition_global(db, body.enabled)
    db.add(AuditEvent(actor=actor, action="plate_recognition_toggle", detail=f"enabled={enabled}"))
    db.commit()
    return recognition_policy_snapshot(db)


def _investigation_time(value: str, label: str):
    try:
        parsed = parse_time(value)
    except ValueError as exc:
        raise HTTPException(400, f"invalid {label} timestamp") from exc
    if parsed is None:
        raise HTTPException(400, f"{label} is required")
    return parsed


@app.get("/api/investigations/vehicles")
def investigate_vehicles(
    start: str,
    end: str,
    vehicle_type: str | None = None,
    vehicle_color: str | None = None,
    camera_id: str | None = None,
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    sort: str = Query(default="asc", pattern="^(asc|desc)$"),
    plate_state: str | None = Query(default=None, pattern="^(ok|plate_unreadable|plate_not_visible|not_checked)$"),
    review: str | None = Query(default=None, pattern="^(pending|verified)$"),
    feed: str | None = Query(default=None, pattern="^(own|government)$"),
    db: Session = Depends(get_db),
):
    from_dt, to_dt = _investigation_time(start, "start"), _investigation_time(end, "end")
    if from_dt > to_dt:
        raise HTTPException(400, "start must be before end")
    return search_observations(
        db, start=from_dt, end=to_dt, vehicle_type=vehicle_type, vehicle_color=vehicle_color,
        camera_id=camera_id, min_confidence=min_confidence, limit=limit, offset=offset, sort=sort,
        plate_state=plate_state, review=review, feed=feed,
        include_diagnostics=settings.developer_ui_enabled(),
    )


@app.get("/api/investigations/vehicles/export.csv")
def investigate_vehicles_csv(
    start: str,
    end: str,
    vehicle_type: str | None = None,
    vehicle_color: str | None = None,
    camera_id: str | None = None,
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    feed: str | None = Query(default=None, pattern="^(own|government)$"),
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    from_dt, to_dt = _investigation_time(start, "start"), _investigation_time(end, "end")
    if from_dt > to_dt:
        raise HTTPException(400, "start must be before end")
    payload = search_observations(
        db, start=from_dt, end=to_dt, vehicle_type=vehicle_type, vehicle_color=vehicle_color,
        camera_id=camera_id, min_confidence=min_confidence, limit=500, offset=0, sort="asc",
        feed=feed,
    )
    db.add(AuditEvent(actor=actor, action="vehicle_investigation_csv", detail=f"{start}..{end}"))
    db.commit()
    return PlainTextResponse(observations_csv(payload), media_type="text/csv")


@app.get("/api/vehicle-observations/{observation_id}")
def vehicle_observation_detail(observation_id: int, db: Session = Depends(get_db)):
    from app.services.vehicle_observations import plate_status_for

    row = db.get(VehicleObservation, observation_id)
    if row is None:
        raise HTTPException(404, "vehicle observation not found")
    return observation_json(
        row,
        plate_status_for(db, [row]).get(row.id),
        include_diagnostics=settings.developer_ui_enabled(),
    )


class VehicleReviewIn(BaseModel):
    """A human verdict. Omit a field to leave it alone; pass "" to clear it."""

    vehicle_type: str | None = None
    vehicle_color: str | None = None
    note: str = ""
    #: How this verdict was produced (surface, whether the model's answer was on
    #: screen, the scope being swept). Kept on the review history so the export
    #: can tell a blind label from a confirmation.
    context: dict | None = None


@app.post("/api/vehicle-observations/{observation_id}/review")
def review_vehicle_observation(
    observation_id: int,
    payload: VehicleReviewIn,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    """Record a manual correction. The raw model output is preserved."""
    if payload.vehicle_type is None and payload.vehicle_color is None:
        raise HTTPException(400, "Choose a vehicle type or a colour before saving.")
    row = _record_review(
        db, observation_id, actor=actor,
        vehicle_type=payload.vehicle_type, vehicle_color=payload.vehicle_color,
        note=payload.note, context=payload.context,
    )
    db.add(AuditEvent(actor=actor, action="vehicle_observation_review",
                      detail=json.dumps({"observation_id": observation_id,
                                         "vehicle_type": payload.vehicle_type,
                                         "vehicle_color": payload.vehicle_color})))
    db.commit()
    return _observation_payload(db, row)


def _record_review(db: Session, observation_id: int, *, actor: str,
                   vehicle_type=None, vehicle_color=None, note="", context=None):
    """Apply one human verdict. Shared by the single-record route and the
    developer labelling queue, so the two cannot drift apart."""
    from app.services.vehicle_observations import review_observation

    try:
        row = review_observation(
            db, observation_id, actor=actor,
            vehicle_type=vehicle_type, vehicle_color=vehicle_color,
            note=note, context=context,
        )
    except ValueError as exc:
        # The message names the rejected value, which the operator typed. It is
        # not internal state, so it is safe and useful to show.
        raise HTTPException(400, f"That is not a value this system records: {exc}") from exc
    if row is None:
        raise HTTPException(404, "vehicle observation not found")
    return row


def _observation_payload(db: Session, row) -> dict:
    from app.services.vehicle_observations import plate_status_for

    return observation_json(
        row,
        plate_status_for(db, [row]).get(row.id),
        include_diagnostics=settings.developer_ui_enabled(),
    )


#: Tracks in the frozen evaluation set. Those 59 blind-labelled tracks are what
#: the published 50%/83% figures were measured on; relabelling them with the
#: model's answer on screen would quietly turn a held-out measurement into a
#: circular one. Loaded once.
_EVAL_MANIFEST = ROOT / "data" / "labels" / "tracks.jsonl"
_eval_track_ids: set[str] | None = None


def frozen_eval_track_ids() -> set[str]:
    global _eval_track_ids
    if _eval_track_ids is None:
        ids: set[str] = set()
        try:
            for line in _EVAL_MANIFEST.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    ids.add(str(json.loads(line).get("track_id") or ""))
        except (OSError, json.JSONDecodeError):
            ids = set()  # no manifest on this host: nothing to protect
        _eval_track_ids = ids - {""}
    return _eval_track_ids


@app.get("/api/dev/label-queue")
def dev_label_queue(
    start: str,
    end: str,
    camera_id: str | None = None,
    vehicle_type: str | None = None,
    vehicle_color: str | None = None,
    type_source: str | None = None,
    color_source: str | None = None,
    min_type_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    max_type_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    min_color_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    max_color_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    review: str = Query(default="pending", pattern="^(pending|verified|any)$"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    feed: str | None = Query(default=None, pattern="^(own|government)$"),
    db: Session = Depends(get_db),
    _actor: str = Depends(require_operator),
    _dev: bool = Depends(require_developer_ui),
):
    """Rows to label, matched against the model's RAW output.

    Developer surface. The production search deliberately cannot find these
    rows -- it refuses to show a vehicle type at all while the gate is closed,
    and hides colours from retired sources -- which is correct for an
    investigator and useless for building training data.
    """
    from_dt, to_dt = _investigation_time(start, "start"), _investigation_time(end, "end")
    if from_dt > to_dt:
        raise HTTPException(400, "start must be before end")
    payload = search_observations(
        db, start=from_dt, end=to_dt, camera_id=camera_id, feed=feed,
        vehicle_type=vehicle_type, vehicle_color=vehicle_color,
        type_source=type_source, color_source=color_source,
        min_type_confidence=min_type_confidence, max_type_confidence=max_type_confidence,
        min_color_confidence=min_color_confidence, max_color_confidence=max_color_confidence,
        min_confidence=min_confidence, limit=limit, offset=offset, sort="asc",
        review=None if review == "any" else review,
        raw_attribute_filters=True, include_diagnostics=True,
    )
    frozen = frozen_eval_track_ids()
    kept = [row for row in payload["observations"] if row["track_id"] not in frozen]
    payload["observations"] = kept
    payload["returned"] = len(kept)
    payload["raw_filters"] = True
    payload["frozen_eval_excluded"] = len(frozen)
    return payload


@app.get("/api/vehicle-attribute-limitations")
def vehicle_attribute_limitations():
    """Measured POC accuracy. Surfaced so the UI cannot hide it."""
    from app.services.vehicle_attributes import attributes_status, type_deployment_gate
    from app.services.vehicle_observations import POC_ACCURACY

    passed, reason = type_deployment_gate()
    return {
        **POC_ACCURACY,
        "type_gate_passed": passed,
        "type_gate_reason": reason,
        "type_operational": passed,
        "color_operational": bool(settings.vattr_color_enabled),
        "attribute_model": attributes_status()["model_id"],
        "safe_claims": [
            "Vehicles are detected and tracked, and one record is kept per track.",
            "Vehicle colour is an estimate that requires human review.",
            "Attributes are produced by deterministic CV models with no LLM or VLM.",
            "A missing or unreadable plate never removes a vehicle observation.",
        ],
        "unsafe_claims": [
            "Automatic vehicle type is reliable.",
            "The system can find every vehicle of a given colour.",
            "Two tracks sharing a type and colour are the same vehicle.",
        ],
    }


@app.get("/api/vehicle-observation-options")
def vehicle_observation_options(db: Session = Depends(get_db)):
    from app.services.vehicle_attributes import attributes_status, type_deployment_gate
    from app.services.vehicle_observations import (
        POC_ACCURACY,
        SUPPORTED_VEHICLE_COLORS,
        SUPPORTED_VEHICLE_TYPES,
    )

    # Every canonical value stays filterable so old rows remain searchable, but
    # each is flagged with whether a model on this host can actually produce it.
    # Offering `suv` or `silver` as if they were predictable would be a lie.
    gate_passed, gate_reason = type_deployment_gate()
    return {
        "type_filter": {
            # Type is optional and matches verified records only. It must never
            # be a mandatory filter while the automatic value is unreliable.
            "enabled": True,
            "required": False,
            "matches": "human_verified_only",
            "reason": gate_reason,
            "warning": (
                "Automatic vehicle type measured 50% precision and is suppressed. "
                "This filter matches only records a person has verified."
            ),
        },
        "color_filter": {
            "enabled": bool(settings.vattr_color_enabled),
            "required": False,
            "matches": "estimated",
            "warning": (
                "Colour is an estimate (83% precision, 71% coverage on one night "
                "camera). It can miss vehicles and include the wrong ones."
            ),
        },
        "accuracy": POC_ACCURACY,
        "type_gate_passed": gate_passed,
        "vehicle_types": VEHICLE_TYPES,
        "vehicle_colors": VEHICLE_COLORS,
        "type_options": [
            {"value": v, "supported": v in SUPPORTED_VEHICLE_TYPES or v == "unknown"} for v in VEHICLE_TYPES
        ],
        "color_options": [
            {"value": v, "supported": v in SUPPORTED_VEHICLE_COLORS or v == "unknown"} for v in VEHICLE_COLORS
        ],
        "supported_vehicle_types": list(SUPPORTED_VEHICLE_TYPES),
        "supported_vehicle_colors": list(SUPPORTED_VEHICLE_COLORS),
        "attribute_model": attributes_status(),
        "cameras": [{"id": c.id, "name": c.name, "city": c.city} for c in db.scalars(select(Camera).order_by(Camera.id))],
    }


@app.get("/api/cameras")
def list_cameras(db: Session = Depends(get_db)):
    rows = list(db.scalars(select(Camera).order_by(Camera.id)))
    return [
        camera_public(
            c,
            preview_active=manager.preview_active(c.id),
            worker_state=manager.worker_state(c.id),
            # Stream URIs and raw decoder errors are connection details for a
            # government feed. They stay out of the payload in production.
            include_connection=settings.developer_ui_enabled(),
        )
        for c in rows
    ]


@app.post("/api/cameras")
def create_camera(
    body: CameraIn,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    """Manual camera onboarding (Model 1). Additive: refuses to clobber."""
    if db.get(Camera, body.id.strip()):
        raise HTTPException(409, f"camera {body.id.strip()!r} already exists; PATCH it instead")
    try:
        result = onboard_cameras(
            db, [body.model_dump(exclude_none=True)], actor=actor, origin="manual_entry"
        )
    except RegistryImportError as exc:
        raise HTTPException(400, str(exc)) from exc
    if result["errors"]:
        raise HTTPException(400, result["errors"][0]["error"])
    db.add(AuditEvent(actor=actor, action="camera_create", detail=f"{body.id} manual_entry"))
    db.commit()
    return {"ok": True, "id": body.id.strip(), **result}


@app.post("/api/cameras/import")
def import_cameras(
    body: CameraImportIn,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    """Bulk camera onboarding from CSV text or a JSON list (Model 1).

    Never deletes. A camera missing from the file keeps its row and history.
    """
    if bool(body.csv) == bool(body.cameras):
        raise HTTPException(400, "supply exactly one of 'csv' or 'cameras'")
    try:
        rows = parse_registry_csv(body.csv) if body.csv else list(body.cameras or [])
    except RegistryImportError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not rows:
        raise HTTPException(400, "no rows to import")
    if len(rows) > 5000:
        raise HTTPException(400, f"import is limited to 5000 rows, got {len(rows)}")
    result = onboard_cameras(db, rows, actor=actor, origin="bulk_import")
    db.add(
        AuditEvent(
            actor=actor,
            action="camera_import",
            detail=f"created={result['created_count']} updated={result['updated_count']} errors={result['error_count']}",
        )
    )
    db.commit()
    return result


@app.get("/api/reports/gap-analysis.json")
def gap_analysis_json(db: Session = Depends(get_db), _actor: str = Depends(require_operator)):
    return gap_analysis(db)


@app.get("/api/reports/gap-analysis.csv")
def gap_analysis_csv_report(db: Session = Depends(get_db), _actor: str = Depends(require_operator)):
    return PlainTextResponse(
        gap_analysis_csv(gap_analysis(db)),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=gap_analysis.csv"},
    )


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
    if "plate_recognition_mode" in data:
        try:
            data["plate_recognition_mode"] = set_camera_mode(camera_id, data["plate_recognition_mode"])
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
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
    include_unreadable: bool = False,
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
        elif include_unreadable and (s.vehicle_type or (payload or {}).get("vehicle", {}).get("type")):
            records.append(payload)
        else:
            rejected.append(
                {
                    "plate": s.plate_norm or s.plate_raw,
                    "camera_id": s.camera_id,
                    "observed_at": utc_iso(s.source_time),
                    "observed_at_ist": ist_label(s.source_time),
                    "provider": s.model_id or s.provider,
                    "reason": ((payload or {}).get("vehicle") or {}).get("unreadable_reason")
                    or "overlay or not an Indian plate",
                }
            )
    empty_reason = ""
    if not records:
        last = rejected[-1] if rejected else None
        empty_reason = (
            "No vehicles logged yet. Hunt all 30 live feeds: YOLO stores each vehicle even when the plate "
            "is unreadable. A plate card appears only when the crop is large enough for Ollama."
            + (f" Last discarded {last['plate']} on {last['camera_id']}." if last else "")
        )
    return {
        "records": records[-limit:],
        "valid_count": sum(1 for r in records if is_recordable_plate((r.get("vehicle") or {}).get("number_ocr") or (r.get("vehicle") or {}).get("number") or "")),
        "rejected_overlay": rejected[-12:],
        "empty_reason": empty_reason,
        "ollama": vision_status(),
        "include_unreadable": include_unreadable,
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


@app.get("/api/recognition/diagnostics")
def recognition_diagnostics(
    camera_id: str | None = None,
    reason: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
    _actor: str = Depends(require_operator),
    _dev: bool = Depends(require_developer_ui),
):
    """Per-attempt detector/OCR decisions, including model ids and hashes.

    Developer surface. Every attempt is still WRITTEN in production -- this
    gates who can read it back, not whether it is recorded. Investigators get
    the operational plate state on the observation itself instead.
    """
    query = select(RecognitionAttempt)
    if camera_id:
        query = query.where(RecognitionAttempt.camera_id == camera_id)
    if reason:
        query = query.where(RecognitionAttempt.reason_code == reason)
    query = query.order_by(RecognitionAttempt.id.desc()).limit(limit)
    rows = list(db.scalars(query))
    return {"summary": diagnostics_snapshot(db, limit=limit), "attempts": [serialize_attempt(row) for row in rows]}


@app.get("/api/vehicles/{plate}")
def vehicle_history(
    plate: str,
    day: str | None = None,
    start: str | None = None,
    end: str | None = None,
    routes: bool = False,
    db: Session = Depends(get_db),
):
    return _without_internals(vehicle_day(db, plate, day=day, start=start, end=end, include_routes=routes))


#: Model identifiers, hashes and run ids belong on an evidence export, not in
#: the browser. The CSV and GeoJSON exports below still carry them, because an
#: exhibit has to say which model produced the reading.
_INTERNAL_SIGHTING_FIELDS = ("model_id", "model_hash", "run_id", "provider", "frame_index", "vehicle", "passage_id")


def _without_internals(payload: dict) -> dict:
    if settings.developer_ui_enabled():
        return payload
    payload["sightings"] = [
        {k: v for k, v in row.items() if k not in _INTERNAL_SIGHTING_FIELDS}
        for row in (payload.get("sightings") or [])
    ]
    return payload


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
    # Severity first, newest first within a severity. Nothing is filtered out --
    # a low-priority hit still appears, just below the urgent ones.
    rows.sort(key=lambda a: (alert_priority_rank(a), -(a.id or 0)))
    payload = [alert_json(a, db.get(Camera, a.camera_id)) for a in rows]
    if settings.developer_ui_enabled():
        return payload
    # Same rule as the plate history: the model identifiers stay on the audited
    # report exports, not on the queue an operator works from.
    return [{k: v for k, v in row.items() if k not in _INTERNAL_SIGHTING_FIELDS} for row in payload]


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


@app.get("/api/hunt")
def api_hunt_status(db: Session = Depends(get_db)):
    return hunt_status(manager, db)


class HuntStartIn(BaseModel):
    pinned_only: bool = False
    pin_ids: list[str] | None = None
    vision_only: bool = False
    max_concurrent: int | None = Field(default=None, ge=1, le=30)


@app.post("/api/hunt/start")
def api_hunt_start(
    body: HuntStartIn | None = None,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    payload = body or HuntStartIn()
    return start_hunt(
        manager,
        db,
        actor=actor,
        pinned_only=payload.pinned_only,
        pin_ids=payload.pin_ids,
        vision_only=payload.vision_only,
        max_concurrent=payload.max_concurrent,
    )


@app.post("/api/hunt/pin")
def api_hunt_pin(
    body: HuntStartIn | None = None,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    payload = body or HuntStartIn()
    return start_hunt(
        manager,
        db,
        actor=actor,
        pinned_only=True,
        pin_ids=payload.pin_ids,
        max_concurrent=payload.max_concurrent,
    )


@app.post("/api/hunt/stop")
def api_hunt_stop(
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    return stop_hunt(manager, db, actor=actor)


class MeasureIn(BaseModel):
    retest_failed: bool = False
    limit: int | None = Field(default=None, ge=1, le=8)


@app.post("/api/capacity/measure")
def api_capacity_measure(
    body: MeasureIn | None = None,
    db: Session = Depends(get_db),
    actor: str = Depends(require_operator),
):
    payload = body or MeasureIn()
    db.add(AuditEvent(actor=actor, action="capacity_measure_request", detail="sequential government decode probe"))
    db.commit()
    return measure_government_decode(db, limit=payload.limit, retest_failed=payload.retest_failed)


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
        # The decoder's message can carry the stream URL and therefore the
        # credentials embedded in it. It goes to the log, never to the browser.
        log.warning("snapshot failed camera=%s: %s", camera_id, redact_secrets(result.get("error")))
        raise HTTPException(503, "No live frame is available from this camera right now.")
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
def camera_diagnostics(
    camera_id: str,
    db: Session = Depends(get_db),
    _actor: str = Depends(require_operator),
    _dev: bool = Depends(require_developer_ui),
):
    """Feed-level decode diagnostics: protocol, codec, decoder errors.

    Developer surface. Production shows a camera's availability in words
    ("Available", "Unreachable", "Not checked") on the Cameras page instead.
    """
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
