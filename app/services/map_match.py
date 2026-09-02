"""OSM snap-to-road / map matching. Possible paths, never proven roads.

Default: public OSRM Match API (no key, no credit card).
Optional: Mapbox or Geoapify if a token is set. Google Directions is not used.
"""
from __future__ import annotations

from datetime import datetime, timezone
import httpx

from app.config import settings
from app.security import redact_secrets

ROUTE_LABEL = "OSM map-matched possible path, not a verified route"
DISCLAIMER = (
    "Camera-to-camera GIS links are inferred. Snap-to-road uses OpenStreetMap matching "
    "and is a possible path, not a proven route."
)
USER_AGENT = "GujaratCCTV-P0/hackathon (OSRM match; no production SLA)"


def decode_polyline(encoded: str) -> list[list[float]]:
    """Encoded polyline → [[lat, lng], ...]."""
    points: list[list[float]] = []
    index = lat = lng = 0
    length = len(encoded or "")
    while index < length:
        result = shift = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if result & 1 else result >> 1
        lat += dlat
        result = shift = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if result & 1 else result >> 1
        lng += dlng
        points.append([lat / 1e5, lng / 1e5])
    return points


def resolved_provider() -> str:
    raw = (settings.map_match_provider or "osrm").strip().lower()
    mapbox = bool((settings.mapbox_access_token or "").strip())
    geoapify = bool((settings.geoapify_api_key or "").strip())
    if raw == "mapbox" and mapbox:
        return "mapbox"
    if raw == "geoapify" and geoapify:
        return "geoapify"
    if raw == "auto":
        if mapbox:
            return "mapbox"
        if geoapify:
            return "geoapify"
    return "osrm"


def map_match_status() -> dict:
    provider = resolved_provider()
    return {
        "provider": provider,
        "osrm_url": (settings.osrm_match_url or "").rstrip("/"),
        "requires_key": provider in {"mapbox", "geoapify"},
        "mapbox_configured": bool((settings.mapbox_access_token or "").strip()),
        "geoapify_configured": bool((settings.geoapify_api_key or "").strip()),
        "google_used": False,
        "label": ROUTE_LABEL,
    }


def possible_routes_for_points(points: list[dict], *, client: httpx.Client | None = None) -> dict:
    """Snap timestamped CCTV coordinates onto OSM roads. Never a proven path."""
    provider = resolved_provider()
    trace = _trace(points)
    if len(trace) < 2:
        return {
            "ok": True,
            "provider": provider,
            "routes": [],
            "label": ROUTE_LABEL,
            "disclaimer": DISCLAIMER,
        }
    own = client is None
    http = client or httpx.Client(timeout=12.0, headers={"User-Agent": USER_AGENT})
    try:
        path, meta = _match_trace(trace, provider, http)
        if path and len(path) >= 2:
            return {
                "ok": True,
                "provider": meta.get("provider") or provider,
                "routes": [_route_payload(trace[0], trace[-1], path, meta, cameras=[p["camera_id"] for p in trace])],
                "label": ROUTE_LABEL,
                "disclaimer": DISCLAIMER,
            }
        routes = _hop_routes(trace, provider, http)
        return {
            "ok": True,
            "provider": provider,
            "routes": routes,
            "label": ROUTE_LABEL,
            "disclaimer": DISCLAIMER,
        }
    finally:
        if own:
            http.close()


def _trace(points: list[dict]) -> list[dict]:
    out: list[dict] = []
    for point in points:
        if point.get("lat") is None or point.get("lng") is None:
            continue
        if out and out[-1].get("camera_id") == point.get("camera_id"):
            continue
        out.append(point)
    return out


def _unix(point: dict) -> int | None:
    raw = point.get("source_time")
    if raw is None:
        return None
    if isinstance(raw, datetime):
        dt = raw
    else:
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _increasing_unix(trace: list[dict]) -> list[int] | None:
    times: list[int] = []
    last = None
    for point in trace:
        stamp = _unix(point)
        if stamp is None:
            return None
        if last is not None and stamp <= last:
            stamp = last + 1
        times.append(stamp)
        last = stamp
    return times


def _radius_m(provider: str) -> int:
    configured = int(settings.map_match_radius_m or 25)
    if provider == "mapbox":
        return max(1, min(configured, 50))
    # Public OSRM demo returns TooBig at radiuses >= 50.
    return max(5, min(configured, 40))


def _coords(trace: list[dict]) -> str:
    return ";".join(f"{p['lng']},{p['lat']}" for p in trace)


def _geometry_to_path(geom) -> list[list[float]]:
    if isinstance(geom, str) and geom:
        return decode_polyline(geom)
    if not isinstance(geom, dict):
        return []
    kind = geom.get("type")
    if kind == "LineString":
        return [[lat, lon] for lon, lat in (geom.get("coordinates") or [])]
    if kind == "MultiLineString":
        path: list[list[float]] = []
        for part in geom.get("coordinates") or []:
            path.extend([lat, lon] for lon, lat in part)
        return path
    return []


def _osrm_like_path(payload: dict, *, collection: str) -> tuple[list[list[float]], dict]:
    rows = payload.get(collection) or []
    if not rows:
        return [], {}
    first = rows[0] or {}
    path = _geometry_to_path(first.get("geometry"))
    return path, {"confidence": first.get("confidence")}


def _match_trace(trace: list[dict], provider: str, http: httpx.Client) -> tuple[list[list[float]], dict]:
    if provider == "mapbox":
        path, meta = _mapbox_match(trace, http)
        if path:
            return path, {**meta, "provider": "mapbox"}
    elif provider == "geoapify":
        path, meta = _geoapify_match(trace, http)
        if path:
            return path, {**meta, "provider": "geoapify"}
    path, meta = _osrm_match(trace, http)
    if path:
        return path, {**meta, "provider": "osrm_match"}
    path, meta = _osrm_route(trace, http)
    if path:
        return path, {**meta, "provider": "osrm_route"}
    return [], {}


def _hop_routes(trace: list[dict], provider: str, http: httpx.Client) -> list[dict]:
    routes = []
    for a, b in zip(trace, trace[1:], strict=False):
        hop = [a, b]
        path, meta = _match_trace(hop, provider, http)
        used = meta.get("provider") or provider
        if not path or len(path) < 2:
            path = [[a["lat"], a["lng"]], [b["lat"], b["lng"]]]
            used = "fallback_straight"
            meta = {"error": meta.get("error") or "match returned no geometry"}
        routes.append(_route_payload(a, b, path, {**meta, "provider": used}))
    return routes


def _route_payload(a: dict, b: dict, path: list[list[float]], meta: dict, cameras: list | None = None) -> dict:
    out = {
        "from_camera": a.get("camera_id"),
        "to_camera": b.get("camera_id"),
        "from_time": a.get("source_time"),
        "to_time": b.get("source_time"),
        "path": path,
        "label": ROUTE_LABEL,
        "provider": meta.get("provider") or "osrm_match",
    }
    if cameras:
        out["cameras"] = cameras
    if meta.get("confidence") is not None:
        out["confidence"] = meta["confidence"]
    if meta.get("error"):
        out["error"] = redact_secrets(str(meta["error"]))
    return out


def _osrm_match(trace: list[dict], http: httpx.Client) -> tuple[list[list[float]], dict]:
    base = (settings.osrm_match_url or "http://router.project-osrm.org").rstrip("/")
    url = f"{base}/match/v1/driving/{_coords(trace)}"
    stamps = _increasing_unix(trace)
    radii = [_radius_m("osrm")]
    for extra in (25, 15):
        if extra not in radii and extra <= radii[0]:
            radii.append(extra)
    last: tuple[list[list[float]], dict] = [], {}
    for radius in radii:
        params = {
            "overview": "full",
            "geometries": "geojson",
            "gaps": "ignore",
            "radiuses": ";".join(str(radius) for _ in trace),
        }
        if stamps:
            params["timestamps"] = ";".join(str(t) for t in stamps)
        path, meta = _osrm_get(http, url, collection="matchings", params=params)
        last = path, meta
        if path:
            return path, meta
        if str(meta.get("error") or "") != "TooBig":
            break
    return last


def _osrm_route(trace: list[dict], http: httpx.Client) -> tuple[list[list[float]], dict]:
    base = (settings.osrm_match_url or "http://router.project-osrm.org").rstrip("/")
    params = {"overview": "full", "geometries": "geojson"}
    url = f"{base}/route/v1/driving/{_coords(trace)}"
    return _osrm_get(http, url, collection="routes", params=params)


def _osrm_get(http: httpx.Client, url: str, *, collection: str, params: dict | None = None) -> tuple[list[list[float]], dict]:
    try:
        response = http.get(url, params=params)
        payload = response.json()
    except Exception as exc:
        return [], {"error": redact_secrets(str(exc))}
    code = str(payload.get("code") or "")
    if code and code.lower() != "ok":
        return [], {"error": code}
    path, meta = _osrm_like_path(payload, collection=collection)
    if not path:
        return [], {"error": code or "empty geometry"}
    return path, meta


def _mapbox_match(trace: list[dict], http: httpx.Client) -> tuple[list[list[float]], dict]:
    token = (settings.mapbox_access_token or "").strip()
    if not token:
        return [], {"error": "MAPBOX_ACCESS_TOKEN is not set"}
    params = {
        "access_token": token,
        "geometries": "geojson",
        "overview": "full",
        "radiuses": ";".join(str(_radius_m("mapbox")) for _ in trace),
    }
    stamps = _increasing_unix(trace)
    if stamps:
        params["timestamps"] = ";".join(str(t) for t in stamps)
    url = f"https://api.mapbox.com/matching/v5/mapbox/driving/{_coords(trace)}"
    try:
        response = http.get(url, params=params)
        payload = response.json()
    except Exception as exc:
        return [], {"error": redact_secrets(str(exc))}
    code = str(payload.get("code") or "")
    if code and code.lower() != "ok":
        return [], {"error": code}
    path, meta = _osrm_like_path(payload, collection="matchings")
    if not path:
        message = payload.get("message") or code or "empty geometry"
        return [], {"error": redact_secrets(str(message))}
    return path, meta


def _geoapify_match(trace: list[dict], http: httpx.Client) -> tuple[list[list[float]], dict]:
    key = (settings.geoapify_api_key or "").strip()
    if not key:
        return [], {"error": "GEOAPIFY_API_KEY is not set"}
    waypoints = []
    for point in trace:
        item = {"location": [point["lng"], point["lat"]]}
        stamp = _unix(point)
        if stamp is not None:
            item["timestamp"] = stamp
        waypoints.append(item)
    url = "https://api.geoapify.com/v1/mapmatching"
    try:
        response = http.post(url, params={"apiKey": key}, json={"mode": "drive", "waypoints": waypoints})
        payload = response.json()
    except Exception as exc:
        return [], {"error": redact_secrets(str(exc))}
    if payload.get("error") or payload.get("message") and not payload.get("features"):
        return [], {"error": redact_secrets(str(payload.get("message") or payload.get("error")))}
    features = payload.get("features") or []
    if not features:
        return [], {"error": "empty geometry"}
    path = _geometry_to_path((features[0] or {}).get("geometry"))
    if not path:
        return [], {"error": "empty geometry"}
    props = (features[0] or {}).get("properties") or {}
    return path, {"confidence": props.get("confidence")}
