import json
from datetime import datetime, timezone

import httpx

from app.services.activity import cameras_active_at, close_activity, open_activity
from app.services.map_match import decode_polyline, possible_routes_for_points
from app.services.pipeline import persist_sighting
from app.services.vehicle import vehicle_day, vehicle_geojson
from tests.conftest import add_camera, add_watchlist


def test_decode_polyline_roundtrip_shape():
    # encoded "??" is empty-ish; a known tiny polyline for 38.5,-120.2
    path = decode_polyline("_p~iF~ps|U")
    assert path
    assert len(path[0]) == 2


def _points():
    return [
        {"camera_id": "a", "lat": 23.02, "lng": 72.57, "source_time": "2026-09-02T10:00:00+00:00"},
        {"camera_id": "b", "lat": 21.17, "lng": 72.83, "source_time": "2026-09-02T14:00:00+00:00"},
    ]


def test_osrm_match_needs_no_api_key(monkeypatch):
    monkeypatch.setattr("app.services.map_match.settings.map_match_provider", "osrm")
    monkeypatch.setattr("app.services.map_match.settings.mapbox_access_token", "")
    monkeypatch.setattr("app.services.map_match.settings.geoapify_api_key", "")
    monkeypatch.setattr("app.services.map_match.settings.google_maps_api_key", "")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        assert "googleapis.com" not in url
        assert "mapbox.com" not in url
        assert "geoapify.com" not in url
        assert "match/v1/driving" in url
        assert "access_token" not in url
        return httpx.Response(
            200,
            json={
                "code": "Ok",
                "matchings": [
                    {
                        "confidence": 0.88,
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[72.57, 23.02], [72.58, 23.03], [72.83, 21.17]],
                        },
                    }
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = possible_routes_for_points(_points(), client=client)
    assert out["ok"] is True
    assert out["provider"] == "osrm_match"
    assert out["routes"][0]["path"][0] == [23.02, 72.57]
    assert "not a verified route" in out["label"]
    assert "google" not in json.dumps(out).lower()


def test_osrm_toobig_retries_smaller_radius(monkeypatch):
    monkeypatch.setattr("app.services.map_match.settings.map_match_provider", "osrm")
    monkeypatch.setattr("app.services.map_match.settings.map_match_radius_m", 40)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        seen.append(url)
        if "40" in (request.url.params.get("radiuses") or ""):
            return httpx.Response(400, json={"code": "TooBig", "message": "Radius search size is too large"})
        return httpx.Response(
            200,
            json={
                "code": "Ok",
                "matchings": [
                    {"geometry": {"type": "LineString", "coordinates": [[72.57, 23.02], [72.58, 23.03]]}}
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = possible_routes_for_points(_points(), client=client)
    assert out["provider"] == "osrm_match"
    assert any("40" in u for u in seen)
    assert out["routes"][0]["path"][0] == [23.02, 72.57]


def test_osrm_failure_falls_back_to_straight(monkeypatch):
    monkeypatch.setattr("app.services.map_match.settings.map_match_provider", "osrm")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": "NoMatch", "matchings": [], "routes": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = possible_routes_for_points(_points(), client=client)
    assert out["routes"]
    assert out["routes"][0]["provider"] == "fallback_straight"
    assert out["routes"][0]["path"] == [[23.02, 72.57], [21.17, 72.83]]


def test_mapbox_token_never_in_payload(monkeypatch):
    monkeypatch.setattr("app.services.map_match.settings.map_match_provider", "mapbox")
    monkeypatch.setattr("app.services.map_match.settings.mapbox_access_token", "pk.secret-token-value")

    def handler(request: httpx.Request) -> httpx.Response:
        assert "pk.secret-token-value" in str(request.url)
        return httpx.Response(
            200,
            json={
                "code": "Ok",
                "matchings": [
                    {
                        "confidence": 0.9,
                        "geometry": {"type": "LineString", "coordinates": [[72.57, 23.02], [72.83, 21.17]]},
                    }
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = possible_routes_for_points(_points(), client=client)
    dumped = json.dumps(out)
    assert "pk.secret-token-value" not in dumped
    assert out["provider"] == "mapbox"
    assert out["routes"][0]["path"][0] == [23.02, 72.57]


def test_geoapify_key_never_in_payload(monkeypatch):
    monkeypatch.setattr("app.services.map_match.settings.map_match_provider", "geoapify")
    monkeypatch.setattr("app.services.map_match.settings.geoapify_api_key", "geo-secret-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert "geo-secret-key" in str(request.url)
        return httpx.Response(
            200,
            json={
                "features": [
                    {"geometry": {"type": "LineString", "coordinates": [[72.57, 23.02], [72.83, 21.17]]}, "properties": {}}
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = possible_routes_for_points(_points(), client=client)
    assert "geo-secret-key" not in json.dumps(out)
    assert out["provider"] == "geoapify"


def test_vehicle_day_filter(db):
    cam = add_camera(db)
    add_watchlist(db)
    old = datetime(2026, 8, 1, tzinfo=timezone.utc)
    new = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    s1, _, _ = persist_sighting(
        db, cam, plate_raw="GJ01AB1234", plate_norm="GJ01AB1234", plate_voted="GJ01AB1234",
        syntax=True, confidence=0.9, model_id="tesseract-opencv-p0", model_hash="x",
        evidence_path="", run_id="r", frame_index=0, passage_id="p-old", source_pts_ms=1.0, provider="local",
    )
    s1.source_time = old
    s2, _, _ = persist_sighting(
        db, cam, plate_raw="GJ01AB1234", plate_norm="GJ01AB1234", plate_voted="GJ01AB1234",
        syntax=True, confidence=0.9, model_id="tesseract-opencv-p0", model_hash="x",
        evidence_path="", run_id="r", frame_index=1, passage_id="p-new", source_pts_ms=2.0, provider="local",
    )
    s2.source_time = new
    db.commit()
    all_rows = vehicle_day(db, "GJ01AB1234")
    assert len(all_rows["sightings"]) == 2
    day = vehicle_day(db, "GJ01AB1234", day="2026-09-02")
    assert len(day["sightings"]) == 1
    assert "key" not in str(day)


def test_vehicle_day_osrm_path(db, monkeypatch):
    cam_a = add_camera(db, id="CAM-A", lat=23.02, lng=72.57)
    cam_b = add_camera(db, id="CAM-B", lat=21.17, lng=72.83, city="Surat")
    add_watchlist(db)
    persist_sighting(
        db, cam_a, plate_raw="GJ01AB1234", plate_norm="GJ01AB1234", plate_voted="GJ01AB1234",
        syntax=True, confidence=0.9, model_id="tesseract-opencv-p0", model_hash="x",
        evidence_path="", run_id="r", frame_index=0, passage_id="p-a", source_pts_ms=1.0, provider="local",
    )
    persist_sighting(
        db, cam_b, plate_raw="GJ01AB1234", plate_norm="GJ01AB1234", plate_voted="GJ01AB1234",
        syntax=True, confidence=0.9, model_id="tesseract-opencv-p0", model_hash="x",
        evidence_path="", run_id="r", frame_index=1, passage_id="p-b", source_pts_ms=2.0, provider="local",
    )
    db.commit()

    def handler(request: httpx.Request) -> httpx.Response:
        assert "googleapis.com" not in str(request.url)
        return httpx.Response(
            200,
            json={
                "code": "Ok",
                "matchings": [
                    {
                        "confidence": 0.8,
                        "geometry": {"type": "LineString", "coordinates": [[72.57, 23.02], [72.83, 21.17]]},
                    }
                ],
            },
        )

    payload = vehicle_day(
        db,
        "GJ01AB1234",
        include_routes=True,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert payload["possible_routes"]
    assert "google" not in payload["path_disclaimer"].lower()
    assert "not a proven route" in payload["path_disclaimer"].lower()
    geo = vehicle_geojson(payload)
    lines = [f for f in geo["features"] if f["geometry"]["type"] == "LineString"]
    assert lines
    assert "not a proven" in (lines[0]["properties"].get("disclaimer") or "").lower()


def test_activity_window_active_at(db):
    cam = add_camera(db, id="cam01", catalogue_camera_id="cam01")
    t0 = datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc)
    row = open_activity(db, cam, run_id="run1")
    row.started_at = t0
    db.commit()
    hit = cameras_active_at(db, at="2026-09-02T10:15:00+00:00", window_minutes=30)
    assert hit["camera_count"] == 1
    assert hit["cameras"][0]["id"] == "cam01"
    miss = cameras_active_at(db, start="2026-09-01T00:00:00+00:00", end="2026-09-01T01:00:00+00:00")
    assert miss["camera_count"] == 0
    close_activity(db, "cam01", reason="stop")
    db.commit()
    still = cameras_active_at(db, start="2026-09-02T10:00:00+00:00", end="2026-09-02T11:00:00+00:00")
    assert still["camera_count"] == 1


def test_autostart_does_not_activate_untested(db, monkeypatch):
    from app.services.demo import autostart_if_configured

    add_camera(db, id="cam-untested", source_type="rtsp", source_uri="rtsp://x", decode_status="untested", processing_mode="local_worker")
    monkeypatch.setattr("app.services.demo.settings.demo_autostart_workers", True)
    monkeypatch.setattr("app.services.demo.settings.demo_decode_ok_only", True)

    class FakeMgr:
        max_workers = 4

        def start(self, _db, camera_id, actor="demo"):
            cam = _db.get(type(_db.get(camera_id)), camera_id) if False else None
            from app.models import Camera

            c = _db.get(Camera, camera_id)
            assert c.decode_status == "ok"
            return {"ok": True, "state": "starting"}

    # no decode_ok cameras → start should not be called with untested
    class Mgr:
        max_workers = 4
        calls = []

        def start(self, _db, camera_id, actor="demo"):
            self.calls.append(camera_id)
            return {"ok": True, "state": "starting"}

    mgr = Mgr()
    out = autostart_if_configured(mgr, db)
    assert "cam-untested" not in mgr.calls
    assert out is not None
