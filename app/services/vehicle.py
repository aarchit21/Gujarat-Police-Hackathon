"""Vehicle-day history, time filters, and path export."""
from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Sighting
from app.services.map_match import DISCLAIMER, possible_routes_for_points, resolved_provider
from app.services.plates import normalize
from app.services.serialize import inferred_links, plate_keys, sighting_json


def _aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        text += "T00:00:00+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def vehicle_day(
    db: Session,
    plate: str,
    *,
    day: str | None = None,
    start: str | None = None,
    end: str | None = None,
    include_routes: bool = False,
    client=None,
) -> dict:
    key = normalize(plate)
    start_dt = parse_time(start)
    end_dt = parse_time(end)
    if day and start_dt is None:
        start_dt = parse_time(day)
        if start_dt:
            end_dt = start_dt + timedelta(days=1)
    rows = [
        s
        for s in db.scalars(select(Sighting).order_by(Sighting.source_time))
        if key in plate_keys(s)
    ]
    if start_dt:
        rows = [s for s in rows if s.source_time and _aware(s.source_time) >= start_dt]
    if end_dt:
        rows = [s for s in rows if s.source_time and _aware(s.source_time) < end_dt]
    points = []
    for s in rows:
        cam = s.camera
        points.append(
            {
                **sighting_json(s),
                "lat": cam.lat if cam else None,
                "lng": cam.lng if cam else None,
                "city": cam.city if cam else None,
                "department": cam.department if cam else None,
            }
        )
    out = {
        "plate_norm": key,
        "from": start_dt.isoformat() if start_dt else None,
        "to": end_dt.isoformat() if end_dt else None,
        "sightings": points,
        "inferred_links": inferred_links(points),
        "possible_routes": [],
        "map_match_provider": resolved_provider(),
        "path_disclaimer": DISCLAIMER,
    }
    if include_routes:
        routed = possible_routes_for_points(points, client=client)
        out["possible_routes"] = routed.get("routes") or []
        out["map_match_provider"] = routed.get("provider") or resolved_provider()
        if routed.get("error"):
            out["match_error"] = routed["error"]
    return out


def vehicle_csv(payload: dict) -> str:
    buf = io.StringIO()
    fields = [
        "plate_norm",
        "camera_id",
        "city",
        "department",
        "source_time",
        "ingest_time",
        "source_pts_ms",
        "lat",
        "lng",
        "plate_raw",
        "plate_voted",
        "confidence",
        "model_id",
        "evidence_path",
    ]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for s in payload.get("sightings") or []:
        writer.writerow({"plate_norm": payload.get("plate_norm"), **s})
    return buf.getvalue()


def vehicle_geojson(payload: dict) -> dict:
    features = []
    for i, s in enumerate(payload.get("sightings") or [], start=1):
        if s.get("lat") is None or s.get("lng") is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [s["lng"], s["lat"]]},
                "properties": {
                    "seq": i,
                    "plate_norm": payload.get("plate_norm"),
                    "camera_id": s.get("camera_id"),
                    "source_time": s.get("source_time"),
                    "disclaimer": payload.get("path_disclaimer"),
                },
            }
        )
    for route in payload.get("possible_routes") or []:
        path = route.get("path") or []
        if len(path) < 2:
            continue
        coords = [[lng, lat] for lat, lng in path]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "label": route.get("label"),
                    "from_camera": route.get("from_camera"),
                    "to_camera": route.get("to_camera"),
                    "provider": route.get("provider"),
                    "disclaimer": payload.get("path_disclaimer"),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}
