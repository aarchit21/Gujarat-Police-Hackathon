from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.config import settings
from app.database import get_db, init_db, make_engine, make_session_factory
from app.main import app
from app.models import VehicleObservation
from app.services.pipeline import FrameProcessor
from app.services.recognition_policy import effective, hydrate, set_camera_mode, set_global
from app.services.track_aggregate import AggregatedAttributes
from app.services.vehicle_observations import apply_attributes, search_observations, upsert_observation
from app.services.yolo_detect import VehicleDet
from tests.conftest import add_camera


def _aggregated(*, vehicle_type: str, vehicle_color: str) -> AggregatedAttributes:
    """An accepted aggregation, as the deterministic attribute path produces."""
    return AggregatedAttributes(
        vehicle_type=vehicle_type, type_confidence=0.94, type_agreement=1.0,
        type_source="openvino:vehicle-attributes-recognition-barrier-0042", type_reason="accepted",
        vehicle_color=vehicle_color, color_confidence=0.91, color_agreement=1.0,
        color_source="openvino:vehicle-attributes-recognition-barrier-0042", color_reason="accepted",
        observations=3, detector_type=vehicle_type,
    )


def _client(engine):
    Session = make_session_factory(engine)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    return TestClient(app), Session


def test_vehicle_observation_upserts_one_track_and_searches_exact_attributes(db):
    cam = add_camera(db)
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    crop = np.full((60, 120, 3), 230, dtype=np.uint8)
    first, created = upsert_observation(
        db, cam, run_id="run", track_id="track", frame_index=1, source_pts_ms=100.0,
        box=(20, 30, 100, 50), frame_shape=frame.shape[:2], frame=frame, crop=crop,
        detector="yolov8n", detector_confidence=0.9, vehicle_type="car",
    )
    second, created_again = upsert_observation(
        db, cam, run_id="run", track_id="track", frame_index=2, source_pts_ms=250.0,
        box=(20, 30, 130, 70), frame_shape=frame.shape[:2], frame=frame, crop=crop,
        detector="yolov8n", detector_confidence=0.9, vehicle_type="car",
    )
    db.commit()
    assert created is True and created_again is False
    assert first.id == second.id
    assert db.scalar(select(func.count(VehicleObservation.id))) == 1

    # upsert_observation records detection and evidence only. It must NOT
    # assert a type or colour: the COCO class is provenance, not a fine-grained
    # prediction, and type_confidence is no longer a copy of detector_confidence.
    assert first.vehicle_type == "unknown"
    assert first.vehicle_color == "unknown"
    assert first.type_confidence == 0.0
    assert first.type_source == "pending"
    assert first.metadata_json["detector_type"] == "car"

    # Attributes arrive from the deterministic aggregator. Type stays
    # suppressed (no validated model); colour is exposed as an estimate.
    apply_attributes(db, first.id, _aggregated(vehicle_type="car", vehicle_color="white"))
    db.commit()
    assert first.vehicle_type == "unknown"
    assert first.type_source == "suppressed_pending_validation"
    assert first.metadata_json["attributes"]["type_candidate"] == "car"
    assert first.vehicle_color == "white"

    # Colour is searchable as an estimate, and the caveat travels with it.
    payload = search_observations(
        db,
        start=datetime.now(timezone.utc) - timedelta(hours=1),
        end=datetime.now(timezone.utc) + timedelta(hours=1),
        vehicle_color="white", limit=10,
    )
    assert payload["total"] == 1
    assert payload["observations"][0]["camera_id"] == cam.id
    assert payload["observations"][0]["color_state"] == "estimated"
    assert any("ESTIMATE" in w for w in payload["search_warnings"])
    assert "identity is not confirmed" in payload["disclaimer"]

    # A type filter matches only human-verified records, so it finds nothing yet.
    unverified = search_observations(
        db,
        start=datetime.now(timezone.utc) - timedelta(hours=1),
        end=datetime.now(timezone.utc) + timedelta(hours=1),
        vehicle_type="car", limit=10,
    )
    assert unverified["total"] == 0
    assert any("HUMAN-VERIFIED" in w for w in unverified["search_warnings"])


def test_policy_global_off_is_hard_stop_and_camera_off_applies_when_enabled(db, monkeypatch):
    import app.services.recognition_policy as policy

    monkeypatch.setattr(policy, "_enabled", None)
    monkeypatch.setattr(policy, "_camera_modes", {})
    monkeypatch.setattr(settings, "plate_recognition_enabled", False)
    cam = add_camera(db)
    assert hydrate(db) is False
    assert effective(db, cam) is False
    set_global(db, True)
    db.commit()
    monkeypatch.setattr(policy, "_enabled", None)
    assert hydrate(db) is True
    assert effective(db, cam) is True
    set_camera_mode(cam.id, "off")
    assert effective(db, cam) is False


def test_frame_processor_persists_vehicle_when_anpr_disabled(db, monkeypatch):
    import app.services.pipeline as pipeline
    import app.services.recognition_policy as policy

    monkeypatch.setattr(policy, "_enabled", None)
    monkeypatch.setattr(policy, "_camera_modes", {})
    monkeypatch.setattr(settings, "plate_recognition_enabled", False)
    cam = add_camera(db)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    crop = np.full((80, 160, 3), 200, dtype=np.uint8)
    monkeypatch.setattr(
        pipeline,
        "detect_vehicles",
        lambda *_args, **_kwargs: [VehicleDet(30, 40, 190, 120, 0.92, "car", crop)],
    )
    monkeypatch.setattr(pipeline, "_read_plate", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ANPR must be skipped")))
    proc = FrameProcessor(db, cam, run_id="vehicle-only")
    assert proc.push(0, frame, 0.0) is None
    db.commit()
    row = db.scalar(select(VehicleObservation))
    assert row is not None
    assert row.evidence_path
    # The vehicle observation persists without ANPR and without any VLM. The
    # all-black test frame yields no crop good enough to classify, so the
    # attributes correctly abstain instead of inheriting the COCO class.
    assert row.vehicle_type == "unknown"
    assert row.vehicle_color == "unknown"
    assert row.metadata_json["detector_type"] == "car"


def test_investigation_and_recognition_setting_apis(monkeypatch, tmp_path):
    import app.services.recognition_policy as policy

    monkeypatch.setattr(policy, "_enabled", None)
    monkeypatch.setattr(policy, "_camera_modes", {})
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    client, Session = _client(engine)
    with client:
        db = Session()
        cam = add_camera(db, id="OBS-API")
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        crop = np.full((50, 100, 3), 230, dtype=np.uint8)
        row, _ = upsert_observation(
            db, cam, run_id="api", track_id="one", frame_index=0, source_pts_ms=0,
            box=(10, 20, 80, 40), frame_shape=frame.shape[:2], frame=frame, crop=crop,
            detector="yolov8n", detector_confidence=0.9, vehicle_type="car",
        )
        apply_attributes(db, row.id, _aggregated(vehicle_type="car", vehicle_color="white"))
        db.commit()
        db.close()
        token = {"Authorization": "Bearer p0-operator"}
        changed = client.patch("/api/settings/recognition", json={"enabled": False}, headers=token)
        assert changed.status_code == 200 and changed.json()["enabled"] is False
        updated = client.patch("/api/cameras/OBS-API", json={"plate_recognition_mode": "off"}, headers=token)
        assert updated.status_code == 200
        now = datetime.now(timezone.utc)
        result = client.get("/api/investigations/vehicles", params={
            "start": (now - timedelta(hours=1)).isoformat(), "end": (now + timedelta(hours=1)).isoformat(),
            "vehicle_color": "white",
        })
        assert result.status_code == 200
        body = result.json()
        assert body["total"] == 1
        assert "identity is not confirmed" in body["disclaimer"]
        # The measured POC numbers travel with every result set.
        assert body["accuracy"]["vehicle_type"]["selective_precision"] == 0.50
        assert body["accuracy"]["type_usable"] is False

        limits = client.get("/api/vehicle-attribute-limitations")
        assert limits.status_code == 200
        assert limits.json()["type_gate_passed"] is False
        assert "Automatic vehicle type is reliable." in limits.json()["unsafe_claims"]
    app.dependency_overrides.clear()
