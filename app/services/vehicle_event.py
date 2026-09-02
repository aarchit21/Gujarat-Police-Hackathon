"""Vehicle passage JSON. Estimates from one frame — not VAHAN, not a proven route."""
from __future__ import annotations

import json
import re

from app.services.plates import layout_hint, normalize, syntax_ok
from app.services.serialize import ist_label, utc_iso

SCHEMA_VERSION = 1
VEHICLE_TYPES = (
    "car",
    "suv",
    "truck",
    "bus",
    "van",
    "two_wheeler",
    "auto_rickshaw",
    "unknown",
)

VEHICLE_PROMPT = (
    "You are reading a still from a traffic CCTV camera in India. "
    "Look for a real vehicle body and its number plate. "
    "Ignore on-screen HUD, channel names, PTZ labels, timestamps, and watermarks. "
    "Return JSON only, no markdown: "
    '{"plate_text":"","vehicle_type":"car|suv|truck|bus|van|two_wheeler|auto_rickshaw|unknown",'
    '"make":"","model":"","color":"","confidence":0.0}. '
    "plate_text must be A-Z and 0-9 only, the registration painted on the vehicle. "
    "If you cannot read a plate, plate_text must be empty. Do not guess a typical plate."
)

OVERLAY_TOKEN = re.compile(
    r"^(CSITMS|PTZ\d*|CAM\d+|HDD|REC|LIVE|OSD|AHD|TVI)$",
    re.I,
)


def is_recordable_plate(plate_norm: str | None) -> bool:
    """Persist only plausible vehicle registrations, not HUD/clock OCR."""
    key = normalize(plate_norm)
    if not key:
        return False
    if syntax_ok(key):
        return True
    hinted = layout_hint(key)
    if hinted != key and syntax_ok(hinted):
        return True
    if OVERLAY_TOKEN.match(key):
        return False
    if key.isdigit() and len(key) <= 8:
        return False
    if key.isalpha() and len(key) <= 8:
        return False
    if len(key) < 7:
        return False
    return bool(re.search(r"[A-Z]", key) and re.search(r"\d", key))


def clean_attr(value: str | None, *, max_len: int = 40) -> str:
    text = re.sub(r"[^A-Za-z0-9 \-_/]", "", str(value or "")).strip()
    return text[:max_len]


def clean_type(value: str | None) -> str:
    raw = str(value or "unknown").strip().lower().replace(" ", "_")
    aliases = {
        "motorcycle": "two_wheeler",
        "bike": "two_wheeler",
        "scooter": "two_wheeler",
        "lorry": "truck",
        "pickup": "truck",
        "jeep": "suv",
        "auto": "auto_rickshaw",
        "rickshaw": "auto_rickshaw",
        "sedan": "car",
        "hatchback": "car",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in VEHICLE_TYPES else "unknown"


def parse_vehicle_payload(text: str) -> dict:
    plate, conf = "", 0.0
    raw = (text or "").strip()
    payload: dict = {}
    blob = raw
    if "```" in blob:
        blob = re.sub(r"```(?:json)?", "", blob).replace("```", "")
    try:
        loaded = json.loads(blob)
        if isinstance(loaded, dict):
            payload = loaded
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]+\}", raw)
        if match:
            try:
                loaded = json.loads(match.group(0))
                if isinstance(loaded, dict):
                    payload = loaded
            except json.JSONDecodeError:
                payload = {}
    plate = normalize(str(payload.get("plate_text") or payload.get("plate") or payload.get("vehicle_number") or ""))
    try:
        conf = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    return {
        "plate_raw": str(payload.get("plate_text") or payload.get("plate") or plate),
        "plate_norm": plate,
        "vehicle_type": clean_type(payload.get("vehicle_type") or payload.get("type")),
        "vehicle_make": clean_attr(payload.get("make") or payload.get("company")),
        "vehicle_model": clean_attr(payload.get("model")),
        "vehicle_color": clean_attr(payload.get("color") or payload.get("colour")),
        "confidence": max(0.0, min(1.0, conf)),
    }


def build_vehicle_event(*, camera, sighting, extras: dict | None = None) -> dict:
    extra = extras or {}
    cam = camera
    raw_norm = sighting.plate_norm or ""
    hinted = layout_hint(raw_norm)
    layout_ok = syntax_ok(hinted)
    display = hinted if layout_ok else raw_norm
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": f"sighting-{sighting.id}",
        "camera_id": sighting.camera_id,
        "camera_name": getattr(cam, "name", "") if cam else "",
        "city": getattr(cam, "city", "") if cam else "",
        "department": getattr(cam, "department", "") if cam else "",
        "lat": getattr(cam, "lat", None) if cam else None,
        "lng": getattr(cam, "lng", None) if cam else None,
        "coords_source": getattr(cam, "coords_source", "") if cam else "",
        "location": (
            getattr(cam, "city", "")
            or getattr(cam, "name", "")
            or (
                f"{cam.lat:.5f}, {cam.lng:.5f}"
                if cam is not None and cam.lat is not None and cam.lng is not None
                else ""
            )
        ),
        "observed_at": utc_iso(sighting.source_time),
        "observed_at_ist": ist_label(sighting.source_time),
        "ingest_at": utc_iso(sighting.ingest_time),
        "ingest_at_ist": ist_label(sighting.ingest_time),
        "timezone": "Asia/Kolkata (IST, UTC+05:30). stored times are UTC.",
        "source_pts_ms": sighting.source_pts_ms,
        "passage_id": sighting.passage_id,
        "vehicle": {
            "number": display,
            "number_raw": sighting.plate_raw,
            "number_ocr": raw_norm,
            "number_layout": hinted if hinted != raw_norm else "",
            "syntax_ok": bool(sighting.syntax_ok),
            "layout_syntax_ok": layout_ok,
            "type": extra.get("vehicle_type") or getattr(sighting, "vehicle_type", "") or "unknown",
            "make": extra.get("vehicle_make") or getattr(sighting, "vehicle_make", "") or "",
            "model": extra.get("vehicle_model") or getattr(sighting, "vehicle_model", "") or "",
            "color": extra.get("vehicle_color") or getattr(sighting, "vehicle_color", "") or "",
        },
        "confidence": {
            "plate": float(sighting.confidence or 0.0),
        },
        "provider": sighting.model_id or sighting.provider,
        "evidence_path": sighting.evidence_path,
        "watchlist_matched": False,
        "disclaimer": (
            "Make/model/color/type are single-frame model estimates, not VAHAN. "
            "Camera-to-camera path is inferred, not a proven road."
        ),
    }
