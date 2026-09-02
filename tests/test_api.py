from fastapi.testclient import TestClient

from app.database import get_db, init_db, make_engine, make_session_factory
from app.main import app
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
    return TestClient(app), Session


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
        assert health["database"]["type"] == "sqlite"
        assert health["database"]["sqlite_is_dev_fallback"] is True
        assert health["onboarded_count"] == 1
        assert health["analytics_active_count"] == 0
        assert health["catalogue_live_is_not_analytics_active"] is True
        assert health["hardcoded_50"] is False
        assert health["own_feed_count"] == 1
        assert health["government_catalogue_count"] == 0
        assert "cctv_access_token" not in health
        dumped = str(health)
        assert settings_token_absent(dumped)
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
        }
        r = client.post("/api/vendor/events", json=body, headers={"Authorization": "Bearer p0-vendor"})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["sighting_id"]
        dup = client.post("/api/vendor/events", json=body, headers={"Authorization": "Bearer p0-vendor"})
        assert dup.json()["duplicate"] is True
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
