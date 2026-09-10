from datetime import datetime, timezone

import numpy as np
from sqlalchemy import func, select

from app.models import Alert, RecognitionAttempt, Sighting
from app.services.match import match_sighting
from app.services.cpu_anpr import CpuPlateCandidate
from app.services.pipeline import _read_plate, persist_sighting, process_frame_iter
from app.services.serialize import inferred_links, plate_keys
from tests.conftest import add_camera, add_watchlist, fake_read


def test_alert_only_after_sighting_persistence(db):
    cam = add_camera(db)
    add_watchlist(db)
    sighting, alert, created = persist_sighting(
        db,
        cam,
        plate_raw="GJ01AB1234",
        plate_norm="GJ01AB1234",
        plate_voted="GJ01AB1234",
        syntax=True,
        confidence=0.9,
        model_id="tesseract-opencv-p0",
        model_hash="abc",
        evidence_path="evidence/x.jpg",
        run_id="run1",
        frame_index=0,
        passage_id="p1",
        source_pts_ms=100.0,
        provider="local",
    )
    assert sighting.id is not None
    assert created is False
    assert alert is None
    assert sighting.vehicle_json["confirmation"]["status"] == "pending"
    second, alert, created = persist_sighting(
        db, cam, plate_raw="GJ01AB1234", plate_norm="GJ01AB1234",
        plate_voted="GJ01AB1234", syntax=True, confidence=0.9,
        model_id="tesseract-opencv-p0", model_hash="abc", evidence_path="evidence/y.jpg",
        run_id="run1", frame_index=1, passage_id="p1", source_pts_ms=200.0, provider="local",
    )
    assert created is True
    assert alert is not None
    assert alert.sighting_id == second.id
    assert second.vehicle_json["confirmation"]["support_count"] == 2


def test_track_vote_prefers_majority_syntax_valid_plate(db):
    cam = add_camera(db)
    persist_sighting(
        db, cam, plate_raw="GJ08AV5178", plate_norm="GJ08AV5178", plate_voted="GJ08AV5178",
        syntax=True, confidence=0.6, model_id="ollama-vision-p0", model_hash="x",
        evidence_path="", run_id="run1", frame_index=0, passage_id="cam12-t2",
        source_pts_ms=100.0, provider="ollama_vision",
    )
    second, _alert, created = persist_sighting(
        db, cam, plate_raw="GJ08AV5171", plate_norm="GJ08AV5171", plate_voted="GJ08AV5171",
        syntax=True, confidence=0.6, model_id="ollama-vision-p0", model_hash="x",
        evidence_path="", run_id="run1", frame_index=1, passage_id="cam12-t2",
        source_pts_ms=200.0, provider="ollama_vision",
    )
    assert created is False
    third, _alert, created = persist_sighting(
        db, cam, plate_raw="GJ08AV5178", plate_norm="GJ08AV5178", plate_voted="GJ08AV5178",
        syntax=True, confidence=0.6, model_id="ollama-vision-p0", model_hash="x",
        evidence_path="", run_id="run1", frame_index=2, passage_id="cam12-t2",
        source_pts_ms=300.0, provider="ollama_vision",
    )
    assert created is False
    assert third.plate_norm == "GJ08AV5178"
    assert third.plate_voted == "GJ08AV5178"


def test_alert_deduplication(db):
    cam = add_camera(db)
    add_watchlist(db)
    persist_sighting(
        db,
        cam,
        plate_raw="GJ01AB1234",
        plate_norm="GJ01AB1234",
        plate_voted="GJ01AB1234",
        syntax=True,
        confidence=0.9,
        model_id="tesseract-opencv-p0",
        model_hash="abc",
        evidence_path="",
        run_id="run1",
        frame_index=0,
        passage_id="p1",
        source_pts_ms=100.0,
        provider="local",
    )
    persist_sighting(
        db,
        cam,
        plate_raw="GJ01AB1234",
        plate_norm="GJ01AB1234",
        plate_voted="GJ01AB1234",
        syntax=True,
        confidence=0.8,
        model_id="tesseract-opencv-p0",
        model_hash="abc",
        evidence_path="",
        run_id="run1",
        frame_index=1,
        passage_id="p1",
        source_pts_ms=200.0,
        provider="local",
    )
    db.commit()
    assert db.scalar(select(func.count(Sighting.id))) == 2
    assert db.scalar(select(func.count(Alert.id))) == 1


def test_match_rejects_unpersisted_sighting(db):
    cam = add_camera(db)
    add_watchlist(db)
    ghost = Sighting(
        camera_id=cam.id,
        passage_id="x",
        source_time=datetime.now(timezone.utc),
        plate_raw="GJ01AB1234",
        plate_norm="GJ01AB1234",
    )
    try:
        match_sighting(db, ghost)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_camera_coverage_counts(db):
    from app.services.coverage import coverage

    add_camera(db, id="A", status="connected", analytics_active=True, processing_mode="local_worker")
    add_camera(db, id="B", status="blocked", analytics_active=False, processing_mode="deferred")
    add_camera(db, id="C", status="deferred", analytics_active=False, processing_mode="deferred")
    cov = coverage(db, open_captures=1, queued=1)
    assert cov["onboarded_count"] == 3
    assert cov["connected_count"] == 1
    assert cov["analytics_active_count"] == 1
    assert cov["blocked_count"] == 2
    assert cov["open_capture_count"] == 1
    assert cov["queued_count"] == 1
    assert "government_feed_label" in cov


def test_inferred_gis_ordering():
    points = [
        {"camera_id": "c1", "lat": 23.0, "lng": 72.5, "source_time": "t1"},
        {"camera_id": "c1", "lat": 23.0, "lng": 72.5, "source_time": "t2"},
        {"camera_id": "c2", "lat": 21.1, "lng": 72.8, "source_time": "t3"},
    ]
    links = inferred_links(points)
    assert len(links) == 1
    assert links[0]["from_camera"] == "c1"
    assert links[0]["to_camera"] == "c2"
    assert "not a verified road" in links[0]["label"]


def test_process_frames_pts_sampling_and_irregular_gaps(db):
    cam = add_camera(db, target_analysis_fps=2.0)
    add_watchlist(db)
    frames = []
    for i, pts in enumerate([0, 50, 80, 600, 650, 8000, 8100]):
        frames.append((i, np.zeros((120, 320, 3), dtype=np.uint8), float(pts)))
    out = process_frame_iter(db, cam, iter(frames), read_fn=fake_read)
    assert out["ok"] is True
    assert out["timing"] == "pts"
    assert out["frames_seen"] == 7
    assert out["frames_sampled"] < 7
    assert out["sightings"] >= 1
    assert cam.analytics_active is False
    assert db.scalar(select(func.count(RecognitionAttempt.id))) >= 1


def test_low_confidence_or_same_frame_cannot_confirm(db):
    cam = add_camera(db)
    add_watchlist(db)
    for frame_index, pts, confidence in [(0, 100.0, 0.9), (0, 100.0, 0.9), (1, 300.0, 0.5)]:
        persist_sighting(
            db, cam, plate_raw="GJ01AB1234", plate_norm="GJ01AB1234",
            plate_voted="GJ01AB1234", syntax=True, confidence=confidence,
            model_id="reader", model_hash="x", evidence_path="", run_id="r",
            frame_index=frame_index, passage_id="track", source_pts_ms=pts, provider="local",
        )
    db.commit()
    assert db.scalar(select(func.count(Alert.id))) == 0


def test_layout_hint_never_becomes_automatic_match(db):
    cam = add_camera(db)
    add_watchlist(db, plate="GJ01AB1234")
    # Directly exercising match requires confirmed persisted evidence; even then
    # GJG1... must not be converted to GJ01... for automatic matching.
    for frame, pts in [(0, 0.0), (1, 200.0)]:
        persist_sighting(
            db, cam, plate_raw="GJG1AB1234", plate_norm="GJG1AB1234",
            plate_voted="GJG1AB1234", syntax=False, confidence=0.99,
            model_id="reader", model_hash="x", evidence_path="", run_id="r",
            frame_index=frame, passage_id="track-hint", source_pts_ms=pts, provider="local",
        )
    db.commit()
    assert db.scalar(select(func.count(Alert.id))) == 0


def test_authoritative_special_format_requires_explicit_marker_and_two_frames(db):
    cam = add_camera(db)
    watch = add_watchlist(db, plate="MIL12345")
    watch.authority = "Gujarat Police authorised test"
    watch.notes = "allow-special-format=true"
    db.commit()
    for frame, pts in [(0, 0.0), (1, 200.0)]:
        _sighting, _alert, created = persist_sighting(
            db, cam, plate_raw="MIL12345", plate_norm="MIL12345",
            plate_voted="MIL12345", syntax=False, confidence=0.95,
            model_id="reader", model_hash="x", evidence_path="", run_id="r",
            frame_index=frame, passage_id="special-track", source_pts_ms=pts, provider="local",
        )
    db.commit()
    assert created is True
    assert db.scalar(select(func.count(Alert.id))) == 1


def test_pts_regression_resets_passage(db):
    cam = add_camera(db, target_analysis_fps=10)
    add_watchlist(db)
    frames = [
        (0, np.zeros((80, 200, 3), dtype=np.uint8), 1000.0),
        (1, np.zeros((80, 200, 3), dtype=np.uint8), 1100.0),
        (2, np.zeros((80, 200, 3), dtype=np.uint8), 50.0),
        (3, np.zeros((80, 200, 3), dtype=np.uint8), 80.0),
    ]
    process_frame_iter(db, cam, iter(frames), read_fn=fake_read, run_id="reg")
    passages = {s.passage_id for s in db.scalars(select(Sighting))}
    assert len(passages) >= 2


def test_plate_keys_include_voted():
    s = Sighting(plate_norm="GJG1AB1234", plate_voted="GJ01AB1234")
    assert "GJ01AB1234" in plate_keys(s)


def test_live_plate_localization_uses_fast_alpr_on_vehicle_crop_only(db, monkeypatch):
    cam = add_camera(db, source_type="rtsp")
    frame = np.zeros((160, 320, 3), dtype=np.uint8)
    vehicle = frame[40:140, 80:260]
    calls = []
    candidate = CpuPlateCandidate(
        plate_raw="GJ01AB1234", plate_norm="GJ01AB1234", confidence=0.95,
        character_confidences=[0.95] * 10, crop_bgr=np.full((30, 120, 3), 220, np.uint8),
        box=(20, 50, 120, 30), detector="fast_alpr", recognizer="fast_plate_ocr",
        detector_confidence=0.8,
        quality={"eligible": True, "score": 0.9},
    )

    def fake_cpu(image, *, allow_opencv_fallback=True, **_kwargs):
        calls.append((image.shape, allow_opencv_fallback))
        return [candidate] if image.shape == vehicle.shape else []

    monkeypatch.setattr("app.services.pipeline.cpu_plate_candidates", fake_cpu)
    monkeypatch.setattr(
        "app.services.pipeline.anpr_crops",
        lambda _frame, **_kwargs: [{
            "crop": vehicle, "body_crop": vehicle, "plate_crops": [],
            "box": (80, 40, 180, 100), "vehicle_type": "car", "detector": "yolov8n",
        }],
    )
    out = _read_plate(cam, frame, provider_kind="local_worker", reader=None, remote_client=None, local_hash="x", allow_cloud=False)
    assert calls[0] == (frame.shape, False)
    assert (vehicle.shape, False) in calls
    assert out["detector"] == "fast_alpr"
    assert out["box"] == (100, 90, 120, 30)
    assert out["localization_accepted"] is True
