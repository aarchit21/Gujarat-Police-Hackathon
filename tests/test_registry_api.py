"""API surface for Model 1 onboarding, the gap report and alert priority."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.database import get_db, init_db, make_engine, make_session_factory
from app.main import app
from app.models import Alert, Camera, Sighting, WatchlistEntry
from app.services.plates import normalize

AUTH = {"Authorization": "Bearer p0-operator"}


def _client():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    Session = make_session_factory(engine)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    return TestClient(app), Session


# --- manual onboarding ----------------------------------------------------


def test_manual_camera_onboarding_persists_and_appears_in_the_ledger():
    client, _ = _client()
    with client:
        r = client.post(
            "/api/cameras",
            json={"id": "CAM-MANUAL", "name": "Gate 4", "city": "Rajkot",
                  "lat": 22.30, "lng": 70.80, "source_type": "rtsp"},
            headers=AUTH,
        )
        assert r.status_code == 200, r.text
        ledger = client.get("/api/cameras").json()
        row = next(c for c in ledger if c["id"] == "CAM-MANUAL")
        # Onboarded, but never contacted -- the ledger must say so.
        assert row["decode_status"] == "untested"
        assert row["analytics_active"] is False
    app.dependency_overrides.clear()


def test_duplicate_camera_id_is_refused_rather_than_overwritten():
    client, _ = _client()
    with client:
        client.post("/api/cameras", json={"id": "CAM-DUP", "city": "Surat"}, headers=AUTH)
        r = client.post("/api/cameras", json={"id": "CAM-DUP", "city": "Patan"}, headers=AUTH)
        assert r.status_code == 409
        assert client.get("/api/cameras").json()[0]["city"] == "Surat"
    app.dependency_overrides.clear()


def test_onboarding_requires_operator_auth():
    client, _ = _client()
    with client:
        r = client.post("/api/cameras", json={"id": "CAM-X"},
                        headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401
    app.dependency_overrides.clear()


def test_invalid_coordinates_are_rejected_with_a_reason():
    client, _ = _client()
    with client:
        r = client.post("/api/cameras",
                        json={"id": "CAM-BAD", "lat": 900.0, "lng": 10.0}, headers=AUTH)
        assert r.status_code == 400
        assert "out of range" in r.json()["detail"]
    app.dependency_overrides.clear()


# --- bulk import ----------------------------------------------------------


def test_csv_bulk_import_creates_cameras_and_reports_bad_rows():
    client, _ = _client()
    csv_text = (
        "id,name,city,lat,lng,source_type\n"
        "CAM-B1,North Gate,Surat,21.17,72.83,rtsp\n"
        "CAM-B2,South Gate,Rajkot,22.30,70.80,rtsp\n"
        ",No Id,Patan,23.0,71.0,rtsp\n"
    )
    with client:
        r = client.post("/api/cameras/import", json={"csv": csv_text}, headers=AUTH)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["created_count"] == 2
        assert body["error_count"] == 1
        assert body["errors"][0]["row"] == 3
        assert len(client.get("/api/cameras").json()) == 2
    app.dependency_overrides.clear()


def test_json_bulk_import_is_accepted():
    client, _ = _client()
    with client:
        r = client.post(
            "/api/cameras/import",
            json={"cameras": [{"id": "CAM-J1", "city": "Patan"}]},
            headers=AUTH,
        )
        assert r.status_code == 200
        assert r.json()["created_count"] == 1
    app.dependency_overrides.clear()


def test_import_requires_exactly_one_input_form():
    client, _ = _client()
    with client:
        assert client.post("/api/cameras/import", json={}, headers=AUTH).status_code == 400
        both = client.post("/api/cameras/import",
                           json={"csv": "id\nA\n", "cameras": [{"id": "B"}]}, headers=AUTH)
        assert both.status_code == 400
    app.dependency_overrides.clear()


def test_reimport_does_not_delete_cameras_missing_from_the_file():
    client, _ = _client()
    with client:
        client.post("/api/cameras/import",
                    json={"cameras": [{"id": "CAM-K1"}, {"id": "CAM-K2"}]}, headers=AUTH)
        client.post("/api/cameras/import", json={"cameras": [{"id": "CAM-K1"}]}, headers=AUTH)
        ids = {c["id"] for c in client.get("/api/cameras").json()}
        assert ids == {"CAM-K1", "CAM-K2"}
    app.dependency_overrides.clear()


# --- gap analysis ---------------------------------------------------------


def test_gap_analysis_json_reports_untested_cameras():
    client, _ = _client()
    with client:
        client.post("/api/cameras/import",
                    json={"cameras": [{"id": "CAM-G1", "city": "Patan"}]}, headers=AUTH)
        body = client.get("/api/reports/gap-analysis.json", headers=AUTH).json()
        assert body["totals"]["cameras"] == 1
        assert body["totals"]["never_probed"] == 1
        assert "Patan" in body["uncovered_cities_no_working_camera"]
        assert body["caveats"]
    app.dependency_overrides.clear()


def test_gap_analysis_csv_downloads():
    client, _ = _client()
    with client:
        client.post("/api/cameras", json={"id": "CAM-G2"}, headers=AUTH)
        r = client.get("/api/reports/gap-analysis.csv", headers=AUTH)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")
        assert "gap,camera_id" in r.text
    app.dependency_overrides.clear()


# --- alert priority -------------------------------------------------------


def _seed_alert(db, *, plate: str, priority: str, camera_id: str, alert_id: int):
    db.add(Camera(id=camera_id, name=camera_id, department="Home", city="Surat",
                  lat=21.1, lng=72.8, source_type="image_dir", source_uri="",
                  status="connected", processing_mode="local_worker",
                  analytics_policy="continuous", priority_class="A",
                  network_class="good", analytics_active=False))
    wl = WatchlistEntry(plate_raw=plate, plate_norm=normalize(plate),
                        purpose="stolen_vehicle", priority=priority)
    db.add(wl)
    db.commit()
    s = Sighting(camera_id=camera_id, passage_id=f"{camera_id}-p", plate_norm=normalize(plate),
                 plate_raw=plate, confidence=0.9,
                 source_time=datetime(2026, 9, 14, 10, 0, tzinfo=timezone.utc))
    db.add(s)
    db.commit()
    db.add(Alert(id=alert_id, sighting_id=s.id, watchlist_id=wl.id, camera_id=camera_id,
                 passage_id=s.passage_id, plate_norm=normalize(plate), match_type="exact",
                 status="new"))
    db.commit()


def test_alerts_carry_watchlist_priority_and_sort_high_first():
    client, Session = _client()
    with client:
        db = Session()
        _seed_alert(db, plate="GJ05CD9999", priority="low", camera_id="CAM-LOW", alert_id=1)
        _seed_alert(db, plate="GJ01AB1234", priority="high", camera_id="CAM-HIGH", alert_id=2)
        db.close()
        alerts = client.get("/api/alerts").json()
        assert [a["priority"] for a in alerts] == ["high", "low"]
        assert alerts[0]["purpose"] == "stolen_vehicle"
        # Nothing is filtered away by prioritisation.
        assert len(alerts) == 2
    app.dependency_overrides.clear()
