from __future__ import annotations

from urllib.parse import urlparse

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Alert, Camera, Sighting, SystemState
from app.security import redact_url


def camera_origin(camera: Camera) -> str:
    if camera.catalogue_camera_id:
        return "government_catalogue"
    if camera.source_type in {"image_dir", "file"}:
        return "own_feed"
    return "local_registry"


def coverage(db: Session, *, open_captures: int = 0, preview_count: int = 0, queued: int = 0) -> dict:
    cameras = list(db.scalars(select(Camera)))
    gov = [c for c in cameras if camera_origin(c) == "government_catalogue"]
    own = [c for c in cameras if camera_origin(c) == "own_feed"]
    onboarded = len(cameras)
    connected = sum(1 for c in cameras if c.status == "connected")
    active = sum(1 for c in cameras if c.analytics_active)
    blocked = sum(1 for c in cameras if c.status in {"blocked", "deferred", "unavailable"})
    deferred = sum(1 for c in cameras if c.processing_mode == "deferred" or c.status == "deferred")
    catalogue_live = sum(1 for c in cameras if c.catalogue_live)
    decode_ok = sum(1 for c in cameras if c.decode_status == "ok")
    gov_active = any(c.analytics_active and camera_origin(c) == "government_catalogue" for c in cameras)
    gov_decoded = any(c.decode_status == "ok" and camera_origin(c) == "government_catalogue" for c in cameras)
    sighting_count = int(db.scalar(select(func.count(Sighting.id))) or 0)
    last_s = db.scalar(select(Sighting).order_by(Sighting.id.desc()).limit(1))
    alert_review = int(db.scalar(select(func.count(Alert.id)).where(Alert.status == "new")) or 0)
    sync = db.get(SystemState, "catalogue_synced_at")
    sync_err = db.get(SystemState, "catalogue_last_error")
    cat_count = db.get(SystemState, "catalogue_count")
    cat_status = db.get(SystemState, "catalogue_last_http_status")
    parsed = urlparse(settings.ingest_catalogue_url)
    actual_catalogue = int(cat_count.value) if cat_count and cat_count.value.isdigit() else len(gov)
    if gov_active:
        gov_status = "active"
        gov_label = "decoded, analytics running"
    elif gov_decoded:
        gov_status = "decoded_idle"
        gov_label = "decoded, worker idle"
    elif actual_catalogue:
        gov_status = "catalogue_synced_decode_untested"
        gov_label = "catalogue synced, decode untested"
    else:
        gov_status = "blocked_until_host_decode"
        gov_label = "blocked until host decode"
    return {
        "onboarded_count": onboarded,
        "own_feed_count": len(own),
        "government_catalogue_count": actual_catalogue,
        "local_registry_count": onboarded - len(own) - len(gov),
        "connected_count": connected,
        "analytics_active_count": active,
        "blocked_count": blocked,
        "deferred_count": deferred,
        "catalogue_live_count": catalogue_live,
        "decode_ok_count": decode_ok,
        "queued_count": queued,
        "open_capture_count": open_captures,
        "preview_active_count": preview_count,
        "sighting_count": sighting_count,
        "last_sighting_plate": last_s.plate_norm if last_s else "",
        "last_sighting_camera": last_s.camera_id if last_s else "",
        "last_sighting_at": last_s.source_time.isoformat() if last_s and last_s.source_time else "",
        "alerts_requiring_review": alert_review,
        "honest_coverage": f"{active}/{onboarded} cameras have analytics running",
        "government_feed_status": gov_status,
        "government_feed_label": gov_label,
        "catalogue_url": redact_url(settings.ingest_catalogue_url),
        "catalogue_host": (parsed.hostname or ""),
        "catalogue_path": parsed.path,
        "catalogue_auth_mode": (settings.cctv_auth_mode or "none"),
        "catalogue_synced_at": sync.value if sync else "",
        "catalogue_last_error": sync_err.value if sync_err else "",
        "catalogue_last_http_status": cat_status.value if cat_status else "",
        "catalogue_live_is_not_analytics_active": True,
        "hardcoded_50": False,
        "measured_safe_fps": (db.get(SystemState, "measured_safe_fps").value if db.get(SystemState, "measured_safe_fps") else ""),
        "recommended_target_fps": (db.get(SystemState, "recommended_target_fps").value if db.get(SystemState, "recommended_target_fps") else ""),
    }
