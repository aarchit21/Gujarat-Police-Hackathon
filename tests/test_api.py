from contextlib import contextmanager

from fastapi.testclient import TestClient

from app.database import get_db, init_db, make_engine, make_session_factory
from app.main import app
from tests.conftest import operator_headers
from app.models import Camera, WatchlistEntry
from app.services.plates import normalize


def _client(tmp_engine):
    Session = make_session_factory(tmp_engine)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    return TestClient(app, headers=operator_headers()), Session


def test_health_exposes_sqlite_fallback_and_coverage():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    client, Session = _client(engine)
    with client:
        db = Session()
        db.add(
            Camera(
                id="CAM-X",
                name="x",
                department="Home",
                city="Ahmedabad",
                lat=23.0,
                lng=72.5,
                source_type="image_dir",
                status="onboarded",
                processing_mode="local_worker",
            )
        )
        db.commit()
        db.close()
        health = client.get("/api/health").json()
        # /api/health reports the CONFIGURED database, not this test's in-memory
        # engine, so it says "postgresql" on a host that has been migrated and
        # "sqlite" on one still on the fallback. Assert the two fields agree
        # rather than pinning a dialect -- pinning one made this test fail purely
        # because the host moved to PostgreSQL for concurrent camera workers.
        assert health["database"]["type"] in {"sqlite", "postgresql"}
        assert health["database"]["sqlite_is_dev_fallback"] is (
            health["database"]["type"] == "sqlite"
        )
        if health["database"]["type"] == "postgresql":
            # Pool must cover every worker slot, or workers queue on connections.
            assert health["database"]["pool_size"] >= health["database"]["concurrent_worker_capacity"]
        assert health["onboarded_count"] == 1
        assert health["analytics_active_count"] == 0
        assert health["catalogue_live_is_not_analytics_active"] is True
        assert health["hardcoded_50"] is False
        assert health["own_feed_count"] == 1
        assert health["government_catalogue_count"] == 0
        assert health["vision_enhancement"]["method"] == "opencv-clahe-unsharp-v1"
        assert health["vision_enhancement"]["generative"] is False
        assert health["cpu_anpr"]["execution_provider"] in {
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
        }
        assert health["recognition"]["attempt_count"] == 0
        # Recognition diagnostics are a developer surface: still recorded in
        # production, but only readable when the developer console is on.
        assert client.get(
            "/api/recognition/diagnostics", headers={"Authorization": "Bearer p0-operator"}
        ).status_code == 404
        with _developer_ui():
            diag = client.get("/api/recognition/diagnostics", headers={"Authorization": "Bearer p0-operator"})
            assert diag.status_code == 200
            assert diag.json()["attempts"] == []
        assert "cctv_access_token" not in health
        dumped = str(health)
        assert settings_token_absent(dumped)
    app.dependency_overrides.clear()


@contextmanager
def _developer_ui():
    """Turn the developer console on for the duration of a block."""
    from app.config import settings

    app_env, enabled = settings.app_env, settings.enable_developer_ui
    settings.app_env, settings.enable_developer_ui = "development", True
    try:
        yield
    finally:
        settings.app_env, settings.enable_developer_ui = app_env, enabled


def test_developer_console_is_not_served_in_production():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    client, _Session = _client(engine)
    with client:
        assert client.get("/dev").status_code == 404
        assert client.get("/dev/console.js").status_code == 404
        # Never reachable through the public static mount, flag or no flag.
        assert client.get("/static/console.js").status_code == 404
        assert client.get("/api/ui/config").json()["developer_ui"] is False
        with _developer_ui():
            assert client.get("/dev").status_code == 200
            assert client.get("/dev/console.js").status_code == 200
            assert client.get("/dev/../config.py").status_code in {403, 404}
            assert client.get("/api/ui/config").json()["developer_ui"] is True
    app.dependency_overrides.clear()


def test_production_camera_payload_hides_stream_connection_details():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    client, Session = _client(engine)
    with client:
        db = Session()
        db.add(
            Camera(
                id="CAM-RTSP", name="gate", department="Home", city="Surat",
                source_type="rtsp", source_uri="rtsp://operator:hunter2@10.0.0.9:554/stream/1",
                hls_url="https://cctv.corp8.cloud/cam01/index.m3u8",
                status="onboarded", processing_mode="local_worker",
                last_error="failed to open rtsp://operator:hunter2@10.0.0.9:554/stream/1",
            )
        )
        db.commit()
        db.close()
        body = client.get("/api/cameras").text
        assert "hunter2" not in body
        assert "rtsp://" not in body
        assert "10.0.0.9" not in body
        camera = client.get("/api/cameras").json()[0]
        assert "source_uri_redacted" not in camera
        assert "last_error" not in camera
        # Availability is measured, never assumed from being catalogued.
        assert camera["availability"] == "not_checked"
        assert camera["availability_label"] == "Not checked"
    app.dependency_overrides.clear()


def settings_token_absent(text: str) -> bool:
    from app.config import settings

    token = (settings.cctv_access_token or "").strip()
    return (not token) or (token not in text)


def test_vendor_api_and_reports(tmp_path):
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    client, Session = _client(engine)
    with client:
        db = Session()
        db.add(
            Camera(
                id="CAM-V",
                name="v",
                department="RTO",
                city="Surat",
                lat=21.1,
                lng=72.8,
                source_type="rtsp",
                processing_mode="vendor_metadata",
            )
        )
        db.add(WatchlistEntry(plate_raw="GJ01AB1234", plate_norm=normalize("GJ01AB1234")))
        db.commit()
        db.close()
        body = {
            "event_id": "e1",
            "camera_id": "CAM-V",
            "source_time": "2026-09-01T10:00:00Z",
            "plate_raw": "GJ01AB1234",
            "confidence": 0.9,
            "vendor_model_id": "vendor-a",
            "passage_id": "vendor-pass",
            "frame_index": 0,
        }
        r = client.post("/api/vendor/events", json=body, headers={"Authorization": "Bearer p0-vendor"})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["sighting_id"]
        dup = client.post("/api/vendor/events", json=body, headers={"Authorization": "Bearer p0-vendor"})
        assert dup.json()["duplicate"] is True
        body2 = {**body, "event_id": "e2", "frame_index": 1, "source_time": "2026-09-01T10:00:00.200000Z"}
        confirmed = client.post("/api/vendor/events", json=body2, headers={"Authorization": "Bearer p0-vendor"})
        assert confirmed.json()["alert_created"] is True
        alerts = client.get("/api/alerts").json()
        assert len(alerts) == 1
        assert alerts[0]["plate_norm"] == "GJ01AB1234"
        csv = client.get("/api/reports/sightings.csv", headers={"Authorization": "Bearer p0-operator"})
        assert csv.status_code == 200
        assert "GJ01AB1234" in csv.text
        assert "model_id" in csv.text
        assert "source_pts_ms" in csv.text
        assert "ingest_utc" in csv.text
        hist = client.get("/api/vehicles/GJ01AB1234").json()
        assert hist["sightings"]
    app.dependency_overrides.clear()


def test_feed_split_partitions_cameras_and_observations():
    """Own feed and government feed must sum to the totals, never overlap.

    The two paths demonstrate different things -- the own feed is a synthetic
    source that proves plate-to-alert end to end, the catalogue is live RTSP
    that proves detection at scale -- so the dashboard reports them separately.
    A split that double-counts or loses rows would misstate both.
    """
    from datetime import datetime, timedelta, timezone

    from app.models import VehicleObservation

    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    client, Session = _client(engine)
    with client:
        db = Session()
        db.add(Camera(id="OWN-1", name="own", department="Home", city="Ahmedabad",
                      source_type="image_dir", source_uri="x", status="onboarded",
                      processing_mode="local_worker", decode_status="ok"))
        db.add(Camera(id="cam01", name="gov", department="government-catalogue", city="Surat",
                      source_type="rtsp", catalogue_camera_id="cam01", status="connected",
                      processing_mode="local_worker", decode_status="ok"))
        db.add(Camera(id="cam02", name="gov2", department="government-catalogue", city="Surat",
                      source_type="rtsp", catalogue_camera_id="cam02", status="onboarded",
                      processing_mode="local_worker", decode_status="failed"))
        now = datetime.now(timezone.utc)
        for camera_id in ("OWN-1", "cam01", "cam01"):
            db.add(VehicleObservation(camera_id=camera_id, run_id="r", track_id=f"t{camera_id}{now.microsecond}",
                                      first_seen_at=now, last_seen_at=now))
            now += timedelta(milliseconds=1)
        db.commit()
        db.close()

        feeds = client.get("/api/ui/overview").json()["feeds"]
        assert feeds["own"]["cameras"]["total"] == 1
        assert feeds["government"]["cameras"]["total"] == 2
        assert feeds["government"]["cameras"]["unreachable"] == 1
        # A camera is only "available" once this host actually opened it.
        assert feeds["own"]["cameras"]["available"] == 1
        assert feeds["government"]["cameras"]["available"] == 1

        window = {"start": (now - timedelta(hours=1)).isoformat(),
                  "end": (now + timedelta(hours=1)).isoformat()}
        both = client.get("/api/investigations/vehicles", params=window).json()
        own = client.get("/api/investigations/vehicles", params={**window, "feed": "own"}).json()
        gov = client.get("/api/investigations/vehicles", params={**window, "feed": "government"}).json()
        assert own["total"] + gov["total"] == both["total"] == 3
        assert {o["feed"] for o in own["observations"]} == {"own"}
        assert {o["feed"] for o in gov["observations"]} == {"government"}
        # The camera payload carries the same label the filter matches on.
        feeds_by_id = {c["id"]: c["feed"] for c in client.get("/api/cameras").json()}
        assert feeds_by_id == {"OWN-1": "own", "cam01": "government", "cam02": "government"}
    app.dependency_overrides.clear()
