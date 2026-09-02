"""Discover cameras from GET INGEST_CATALOGUE_URL. Never construct stream URLs."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import AuditEvent, Camera, SystemState
from app.security import redact_secrets, redact_url

GET_ONLY = True  # consume-only: never publish, never call a control API
AUTH_MODES = ("none", "bearer", "basic", "custom_header", "form")
DEFAULT_CATALOGUE_URL = "https://cctv.corp8.cloud/cameras.json"
# Official stream contract. Used only when cameras.json omits a URL for that camera.
DOCUMENTED_RTSP_PREFIX = "rtsp://103.250.160.189:8554/stream/"
DOCUMENTED_WHEP_PREFIX = "http://103.250.160.189:8889/stream/"
DOCUMENTED_WHEP_SUFFIX = "/whep"
DOCUMENTED_HLS_PREFIX = "https://cctv.corp8.cloud/"
DOCUMENTED_HLS_SUFFIX = "/index.m3u8"


class CatalogueError(ValueError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(redact_secrets(message))
        self.status_code = status_code


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _looks_like_html(content_type: str, body: str) -> bool:
    if "text/html" in (content_type or "").lower():
        return True
    head = (body or "").lstrip()[:200].lower()
    return head.startswith("<!doctype") or head.startswith("<html")


def catalogue_auth_headers() -> dict[str, str]:
    """Build request headers. Never log the returned values."""
    mode = (settings.cctv_auth_mode or "none").strip().lower() or "none"
    token = (settings.cctv_access_token or "").strip()
    if mode == "bearer":
        if not token:
            raise CatalogueError("CCTV_ACCESS_TOKEN is required for bearer authentication")
        return {"Authorization": f"Bearer {token}"}
    if mode == "custom_header":
        name = (settings.cctv_auth_header_name or "").strip()
        if not name or not token:
            raise CatalogueError("CCTV_AUTH_HEADER_NAME and CCTV_ACCESS_TOKEN are required for custom_header authentication")
        return {name: token}
    return {}


def catalogue_get(url: str, *, timeout: float | None = None, client: httpx.Client | None = None) -> Any:
    result = fetch_catalogue(url, timeout=timeout, client=client)
    if not result["ok"]:
        raise CatalogueError(result["error"], status_code=result.get("status_code"))
    return result["payload"]


def fetch_catalogue(url: str, *, timeout: float | None = None, client: httpx.Client | None = None) -> dict:
    if not url:
        raise CatalogueError("INGEST_CATALOGUE_URL is empty")
    mode = (settings.cctv_auth_mode or "none").strip().lower() or "none"
    if mode not in AUTH_MODES:
        raise CatalogueError(f"unsupported CCTV_AUTH_MODE: {mode}")
    own = client is None
    http = client or httpx.Client(timeout=timeout or settings.catalogue_sync_timeout_seconds, follow_redirects=True)
    status_code = None
    content_type = ""
    try:
        token = (settings.cctv_access_token or "").strip()
        try:
            if mode == "form":
                if not token:
                    raise CatalogueError("CCTV_ACCESS_TOKEN is required for form authentication")
                login_url = settings.cctv_login_url.strip() or urljoin(_origin(url), "/auth/login")
                if _origin(login_url).lower() != _origin(url).lower():
                    raise CatalogueError("CCTV_LOGIN_URL must use the same origin as INGEST_CATALOGUE_URL")
                login = http.post(login_url, data={"password": token})
                if login.status_code in {401, 403}:
                    raise CatalogueError(f"catalogue HTTP {login.status_code}", status_code=login.status_code)
                login.raise_for_status()
                response = http.get(url)
            elif mode == "bearer":
                response = http.get(url, headers=catalogue_auth_headers())
            elif mode == "basic":
                if not settings.cctv_access_username or not token:
                    raise CatalogueError("CCTV_ACCESS_USERNAME and CCTV_ACCESS_TOKEN are required for basic authentication")
                response = http.get(url, auth=(settings.cctv_access_username, token))
            elif mode == "custom_header":
                response = http.get(url, headers=catalogue_auth_headers())
            else:
                response = http.get(url)
        except httpx.TimeoutException as exc:
            raise CatalogueError("catalogue request timed out") from exc
        except httpx.HTTPError as exc:
            raise CatalogueError(f"catalogue HTTP error: {exc}") from exc

        status_code = response.status_code
        content_type = response.headers.get("content-type", "")
        body = response.text or ""
        if status_code in {401, 403}:
            raise CatalogueError(f"catalogue HTTP {status_code}", status_code=status_code)
        if _looks_like_html(content_type, body):
            raise CatalogueError("catalogue returned HTML/login page instead of JSON", status_code=status_code)
        if status_code >= 400:
            raise CatalogueError(f"catalogue HTTP {status_code}", status_code=status_code)
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise CatalogueError("catalogue returned invalid JSON", status_code=status_code) from exc
        items = parse_catalogue(payload)
        return {
            "ok": True,
            "status_code": status_code,
            "content_type": content_type,
            "payload": payload,
            "camera_count": len(items),
            "auth_mode": mode,
            "url": redact_url(url),
            "error": "",
        }
    except CatalogueError as exc:
        return {
            "ok": False,
            "status_code": exc.status_code or status_code,
            "content_type": content_type,
            "payload": None,
            "camera_count": 0,
            "auth_mode": mode,
            "url": redact_url(url),
            "error": redact_secrets(str(exc)),
        }
    finally:
        if own:
            http.close()


def parse_catalogue(payload: Any) -> list[dict]:
    if payload is None:
        return []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("cameras") or payload.get("items") or payload.get("data") or []
        if isinstance(items, dict):
            items = list(items.values())
    else:
        items = []
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or item.get("camera_id") or item.get("cameraId") or "").strip()
        if not cid:
            continue
        stream = item.get("stream") if isinstance(item.get("stream"), dict) else {}
        out.append(
            {
                "catalogue_camera_id": cid,
                "name": str(item.get("name") or item.get("location") or cid),
                "location": str(item.get("location") or item.get("city") or ""),
                "codec": str(item.get("codec") or item.get("videoCodec") or stream.get("codec") or ""),
                "live": bool(item.get("live") if item.get("live") is not None else item.get("isLive") or False),
                "width": _int(item.get("width") or stream.get("width")),
                "height": _int(item.get("height") or stream.get("height")),
                "reported_fps": _float(item.get("fps") or item.get("reported_fps") or stream.get("fps")),
                "bitrate": _int(item.get("bitrate") or stream.get("bitrate")),
                "rtsp_url": _url_field(item, stream, "rtsp_url", "rtspUrl", "rtsp"),
                "whep_url": _url_field(item, stream, "whep_url", "whepUrl", "whep"),
                "hls_url": _url_field(item, stream, "hls_url", "hlsUrl", "hls"),
                "lat": _float(item.get("lat") or item.get("latitude")),
                "lng": _float(item.get("lng") or item.get("longitude") or item.get("lon")),
                "department": str(item.get("department") or "government-catalogue"),
            }
        )
        apply_documented_stream_contract(out[-1])
    return out


def apply_documented_stream_contract(item: dict) -> dict:
    """Fill missing stream URLs from the organiser contract. Never invent camera IDs.

    Catalogue-provided URLs always win. Patterns are documentation, not a scanner.
    """
    cid = item.get("catalogue_camera_id") or ""
    if not cid:
        return item
    if not item.get("rtsp_url"):
        item["rtsp_url"] = DOCUMENTED_RTSP_PREFIX + cid
        item["rtsp_url_source"] = "documented_contract"
    else:
        item["rtsp_url_source"] = "catalogue"
    if not item.get("whep_url"):
        item["whep_url"] = DOCUMENTED_WHEP_PREFIX + cid + DOCUMENTED_WHEP_SUFFIX
        item["whep_url_source"] = "documented_contract"
    else:
        item["whep_url_source"] = "catalogue"
    if not item.get("hls_url"):
        item["hls_url"] = DOCUMENTED_HLS_PREFIX + cid + DOCUMENTED_HLS_SUFFIX
        item["hls_url_source"] = "documented_contract"
    else:
        item["hls_url_source"] = "catalogue"
    return item


def _url_field(item: dict, stream: dict, *keys: str) -> str:
    """Return a catalogue-provided URL only. Never synthesise one from the camera id."""
    for key in keys:
        value = item.get(key)
        if value:
            return str(value).strip()
        value = stream.get(key)
        if value:
            return str(value).strip()
    return ""


def local_camera_id(catalogue_camera_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._:-]", "-", catalogue_camera_id).strip("-")[:64]
    return safe or "UNKNOWN"


def upsert_from_catalogue(db: Session, item: dict, *, synced_at: datetime | None = None) -> Camera:
    """Use catalogue-provided URLs as-is. Do not build rtsp/whep/hls from the id."""
    now = synced_at or datetime.now(timezone.utc)
    cat_id = item["catalogue_camera_id"]
    cam_id = local_camera_id(cat_id)
    camera = db.get(Camera, cam_id)
    if camera is None:
        camera = db.scalar(select(Camera).where(Camera.catalogue_camera_id == cat_id))
    created = camera is None
    if created:
        has_rtsp = bool(item.get("rtsp_url"))
        camera = Camera(
            id=cam_id,
            name=item.get("name") or cam_id,
            department=item.get("department") or "government-catalogue",
            city=item.get("location") or "",
            lat=item.get("lat") or 22.3,
            lng=item.get("lng") or 71.2,
            source_type="rtsp" if has_rtsp else ("hls" if item.get("hls_url") else "blocked"),
            source_uri="",
            hls_url="",
            whep_url="",
            status="onboarded",
            status_reason="onboarded from ingest catalogue; decode not yet tested",
            processing_mode="local_worker" if has_rtsp or item.get("hls_url") else "deferred",
            analytics_policy="on_demand",
            priority_class="C",
            network_class="limited" if item.get("live") else "offline",
            compute_target="local-host",
            analytics_active=False,
            decode_status="untested",
        )
        db.add(camera)

    camera.catalogue_camera_id = cat_id
    camera.catalogue_live = bool(item.get("live"))
    camera.codec = item.get("codec") or camera.codec
    if item.get("width"):
        camera.width = item["width"]
    if item.get("height"):
        camera.height = item["height"]
    if item.get("reported_fps"):
        camera.reported_fps = item["reported_fps"]
    if item.get("bitrate"):
        camera.bitrate = item["bitrate"]
    if item.get("rtsp_url"):
        camera.protected_rtsp_url_or_reference = item["rtsp_url"]
        camera.source_uri = item["rtsp_url"]
        camera.source_type = "rtsp"
    if item.get("hls_url"):
        camera.hls_url = item["hls_url"]
    if item.get("whep_url"):
        camera.whep_url = item["whep_url"]
    if item.get("name") and created:
        camera.name = item["name"]
    if item.get("location") and created:
        camera.city = item["location"]
    camera.catalogue_synced_at = now
    camera.analytics_active = bool(camera.analytics_active)
    return camera


def sync_catalogue(
    db: Session,
    *,
    url: str | None = None,
    fetcher: Callable[[str], Any] | None = None,
) -> dict:
    target = (url if url is not None else settings.ingest_catalogue_url).strip()
    now = datetime.now(timezone.utc)
    status_code = None
    try:
        if fetcher is None:
            fetched = fetch_catalogue(target)
            if not fetched["ok"]:
                raise CatalogueError(fetched["error"], status_code=fetched.get("status_code"))
            payload = fetched["payload"]
            status_code = fetched.get("status_code")
        else:
            payload = fetcher(target)
        items = parse_catalogue(payload)
    except Exception as exc:
        message = redact_secrets(str(exc))
        _state(db, "catalogue_last_error", message, now)
        _state(db, "catalogue_last_http_status", str(getattr(exc, "status_code", "") or ""), now)
        _state(db, "catalogue_count", "0", now)
        db.add(AuditEvent(action="catalogue_sync_failed", detail=message[:2000]))
        db.commit()
        return {"ok": False, "error": message, "url": redact_url(target), "cameras": 0, "status_code": getattr(exc, "status_code", None)}

    seen: set[str] = set()
    created = 0
    updated = 0
    for item in items:
        existing = db.get(Camera, local_camera_id(item["catalogue_camera_id"]))
        if existing is None:
            existing = db.scalar(select(Camera).where(Camera.catalogue_camera_id == item["catalogue_camera_id"]))
        was_new = existing is None
        upsert_from_catalogue(db, item, synced_at=now)
        seen.add(item["catalogue_camera_id"])
        if was_new:
            created += 1
        else:
            updated += 1

    missing = 0
    catalogue_rows = list(db.scalars(select(Camera).where(Camera.catalogue_camera_id.is_not(None))))
    for cam in catalogue_rows:
        if cam.catalogue_camera_id not in seen:
            cam.catalogue_live = False
            if cam.status not in {"blocked", "deferred"}:
                cam.status = "unavailable"
            cam.status_reason = "missing from latest catalogue; history retained"
            cam.analytics_active = False
            missing += 1

    _state(db, "catalogue_synced_at", now.isoformat(), now)
    _state(db, "catalogue_last_error", "", now)
    _state(db, "catalogue_url", redact_url(target), now)
    _state(db, "catalogue_count", str(len(seen)), now)
    _state(db, "catalogue_last_http_status", str(status_code or 200), now)
    db.add(
        AuditEvent(
            action="catalogue_sync",
            detail=json.dumps({"created": created, "updated": updated, "missing": missing, "seen": len(seen)}),
        )
    )
    db.commit()
    return {
        "ok": True,
        "url": redact_url(target),
        "cameras": len(seen),
        "government_catalogue_count": len(seen),
        "created": created,
        "updated": updated,
        "missing_marked_unavailable": missing,
        "analytics_active_not_implied_by_live": True,
        "status_code": status_code or 200,
        "hardcoded_50": False,
    }


def _state(db: Session, key: str, value: str, now: datetime) -> None:
    row = db.get(SystemState, key)
    if row is None:
        db.add(SystemState(key=key, value=value, updated_at=now))
    else:
        row.value = value
        row.updated_at = now


def _int(value) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
