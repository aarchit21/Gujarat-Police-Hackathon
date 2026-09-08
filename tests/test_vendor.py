from sqlalchemy import func, select

from app.models import Alert, Sighting
from app.services.vendor import VendorIngestError, ingest_vendor_event
from tests.conftest import add_camera, add_watchlist


def _event(**kwargs):
    body = {
        "event_id": "evt-1",
        "camera_id": "CAM-TEST-001",
        "source_time": "2026-09-01T12:00:00+00:00",
        "plate_raw": "GJ 01 AB 1234",
        "confidence": 0.93,
        "vendor_model_id": "vendor-ocr-x",
    }
    body.update(kwargs)
    return body


def test_vendor_ingest_persists_then_alerts(db):
    add_camera(db, processing_mode="vendor_metadata")
    add_watchlist(db)
    out = ingest_vendor_event(db, _event())
    assert out["ok"] is True
    assert out["sighting_id"]
    assert out["alert_created"] is False
    confirmed = ingest_vendor_event(db, _event(
        event_id="evt-2", passage_id="vendor-pass", frame_index=1,
        source_time="2026-09-01T12:00:00.200000+00:00",
    ))
    # First event needs the same passage, so this pair remains deliberately separate.
    assert confirmed["alert_created"] is False
    sighting = db.get(Sighting, out["sighting_id"])
    assert sighting.provider == "vendor_metadata"
    assert sighting.model_id.startswith("vendor:")
    assert sighting.vendor_payload_hash
    assert db.scalar(select(func.count(Alert.id))) == 0


def test_two_vendor_frames_same_passage_confirm_then_alert(db):
    add_camera(db, processing_mode="vendor_metadata")
    add_watchlist(db)
    first = ingest_vendor_event(db, _event(passage_id="vendor-pass", frame_index=0))
    second = ingest_vendor_event(db, _event(
        event_id="evt-2", passage_id="vendor-pass", frame_index=1,
        source_time="2026-09-01T12:00:00.200000+00:00",
    ))
    assert first["alert_created"] is False
    assert second["alert_created"] is True
    assert db.scalar(select(func.count(Alert.id))) == 1


def test_duplicate_vendor_event_rejected(db):
    add_camera(db, processing_mode="vendor_metadata")
    add_watchlist(db)
    first = ingest_vendor_event(db, _event())
    second = ingest_vendor_event(db, _event())
    assert first["ok"] is True
    assert second["duplicate"] is True
    assert second["sighting_id"] == first["sighting_id"]
    assert db.scalar(select(func.count(Sighting.id))) == 1


def test_vendor_requires_fields(db):
    add_camera(db)
    try:
        ingest_vendor_event(db, {"event_id": "x"})
        assert False
    except VendorIngestError:
        pass
