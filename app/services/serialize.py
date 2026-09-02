from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app.models import Alert, Camera, Sighting

IST = ZoneInfo("Asia/Kolkata")


def utc_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def ist_label(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST")
from app.security import hls_requires_server_credential, redact_url
from app.services.coverage import camera_origin
from app.services.processing import select_processing_route


def camera_public(c: Camera, *, preview_active: bool = False, worker_state: str = "") -> dict:
    return {
        "id": c.id,
        "name": c.name,
        "department": c.department,
        "city": c.city,
        "lat": c.lat,
        "lng": c.lng,
        "latitude": c.lat,
        "longitude": c.lng,
        "coords_source": c.coords_source or "",
        "coords_are_placeholder": (c.coords_source or "") == "placeholder",
        "source_type": c.source_type,
        "source_uri_redacted": redact_url(c.source_uri),
        "has_rtsp": bool(c.protected_rtsp_url_or_reference or (c.source_type == "rtsp" and c.source_uri)),
        "has_substream": bool(c.substream_uri),
        "priority_class": c.priority_class,
        "processing_mode": c.processing_mode,
        "analytics_policy": c.analytics_policy,
        "compute_target": c.compute_target,
        "network_class": c.network_class,
        "target_analysis_fps": c.target_analysis_fps,
        "status": c.status,
        "status_reason": c.status_reason,
        "analytics_active": bool(c.analytics_active),
        "preview_active": bool(preview_active),
        "worker_state": worker_state,
        "last_frame_at": utc_iso(c.last_frame_at),
        "last_frame_at_ist": ist_label(c.last_frame_at),
        "last_error": c.last_error,
        "capabilities": c.capabilities,
        "vendor": c.vendor,
        "model": c.model,
        "clock_offset_ms": c.clock_offset_ms,
        "catalogue_camera_id": c.catalogue_camera_id,
        "catalogue_live": bool(c.catalogue_live),
        "codec": c.codec,
        "width": c.width,
        "height": c.height,
        "reported_fps": c.reported_fps,
        "bitrate": c.bitrate,
        "origin": camera_origin(c),
        "whep_url": redact_url(c.whep_url),
        "hls_url": "" if hls_requires_server_credential(c.hls_url) else redact_url(c.hls_url),
        "hls_preview_blocked": hls_requires_server_credential(c.hls_url),
        "catalogue_synced_at": c.catalogue_synced_at.isoformat() if c.catalogue_synced_at else None,
        "decode_tested_at": c.decode_tested_at.isoformat() if c.decode_tested_at else None,
        "decode_status": c.decode_status,
        "source_pts_ms": c.source_pts_ms,
        "last_pts_ms": c.last_pts_ms,
        "reconnect_count": c.reconnect_count,
        "active_protocol": c.active_protocol,
        "measured_worker_fps": c.measured_worker_fps,
        "measured_at": c.measured_at.isoformat() if c.measured_at else None,
        "route": select_processing_route(c),
    }


def parse_vehicle_blob(raw) -> dict | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return loaded if isinstance(loaded, dict) else None
    return None


def sighting_json(s: Sighting) -> dict:
    event = parse_vehicle_blob(getattr(s, "vehicle_json", None))
    return {
        "id": s.id,
        "camera_id": s.camera_id,
        "passage_id": s.passage_id,
        "source_time": utc_iso(s.source_time),
        "source_time_ist": ist_label(s.source_time),
        "ingest_time": utc_iso(s.ingest_time),
        "ingest_time_ist": ist_label(s.ingest_time),
        "source_pts_ms": s.source_pts_ms,
        "plate_raw": s.plate_raw,
        "plate_norm": s.plate_norm,
        "plate_voted": s.plate_voted,
        "syntax_ok": s.syntax_ok,
        "confidence": s.confidence,
        "model_id": s.model_id,
        "model_hash": s.model_hash,
        "evidence_path": s.evidence_path,
        "run_id": s.run_id,
        "frame_index": s.frame_index,
        "provider": s.provider,
        "vehicle_type": getattr(s, "vehicle_type", "") or "",
        "vehicle_make": getattr(s, "vehicle_make", "") or "",
        "vehicle_model": getattr(s, "vehicle_model", "") or "",
        "vehicle_color": getattr(s, "vehicle_color", "") or "",
        "vehicle": event,
    }


def plate_keys(s: Sighting) -> set[str]:
    from app.services.plates import layout_hint

    return {s.plate_norm, s.plate_voted, layout_hint(s.plate_norm or ""), layout_hint(s.plate_voted or "")} - {""}


def alert_json(a: Alert, cam: Camera | None) -> dict:
    s = a.sighting
    event = parse_vehicle_blob(getattr(s, "vehicle_json", None)) if s else None
    veh = (event or {}).get("vehicle") or {}
    return {
        "id": a.id,
        "plate_norm": a.plate_norm,
        "match_type": a.match_type,
        "status": a.status,
        "camera_id": a.camera_id,
        "camera_name": cam.name if cam else "",
        "city": cam.city if cam else "",
        "department": cam.department if cam else "",
        "location": (event or {}).get("location") or (cam.city if cam else "") or (cam.name if cam else ""),
        "lat": cam.lat if cam else None,
        "lng": cam.lng if cam else None,
        "passage_id": a.passage_id,
        "created_at": utc_iso(a.created_at),
        "created_at_ist": ist_label(a.created_at),
        "evidence_path": s.evidence_path if s else "",
        "source_time": utc_iso(s.source_time) if s else None,
        "source_time_ist": ist_label(s.source_time) if s else None,
        "ingest_time": utc_iso(s.ingest_time) if s else None,
        "confidence": s.confidence if s else None,
        "plate_raw": s.plate_raw if s else "",
        "plate_voted": s.plate_voted if s else "",
        "model_id": s.model_id if s else "",
        "model_hash": s.model_hash if s else "",
        "run_id": s.run_id if s else "",
        "sighting_id": a.sighting_id,
        "vehicle_type": getattr(s, "vehicle_type", "") or veh.get("type") or "",
        "vehicle_color": getattr(s, "vehicle_color", "") or veh.get("color") or "",
        "vehicle_make": getattr(s, "vehicle_make", "") or veh.get("make") or "",
        "vehicle_model": getattr(s, "vehicle_model", "") or veh.get("model") or "",
        "vehicle": event,
    }


def inferred_links(points: list[dict]) -> list[dict]:
    links = []
    for a, b in zip(points, points[1:], strict=False):
        if a.get("camera_id") == b.get("camera_id"):
            continue
        if a.get("lat") is None or b.get("lat") is None:
            continue
        links.append(
            {
                "from_camera": a["camera_id"],
                "to_camera": b["camera_id"],
                "from": [a["lng"], a["lat"]],
                "to": [b["lng"], b["lat"]],
                "from_time": a.get("source_time"),
                "to_time": b.get("source_time"),
                "label": "inferred movement, not a verified road",
            }
        )
    return links
