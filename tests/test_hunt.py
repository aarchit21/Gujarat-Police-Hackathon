import threading

import numpy as np
from sqlalchemy import func, select

from app.models import Alert, Sighting
from app.services.hunt import hunt_targets, PIN_DEFAULT
from app.services.ingest import OpenedSource
from app.services.pipeline import process_frame_iter
from app.services.workers import WorkerManager, run_live_loop
from tests.conftest import FakeCap, add_camera, add_watchlist


def test_hunt_targets_decode_ok_first(db):
    add_camera(
        db,
        id="cam02",
        source_type="rtsp",
        source_uri="rtsp://x",
        catalogue_live=True,
        catalogue_camera_id="cam02",
        decode_status="failed",
    )
    add_camera(
        db,
        id="cam01",
        source_type="rtsp",
        source_uri="rtsp://x",
        catalogue_live=True,
        catalogue_camera_id="cam01",
        decode_status="ok",
    )
    add_camera(db, id="CAM-HOME-AHM-001", source_type="image_dir", source_uri="frames")
    ids = [c.id for c in hunt_targets(db)]
    assert ids[0] == "cam01"
    assert "cam02" in ids
    assert "CAM-HOME-AHM-001" not in ids
    pinned = hunt_targets(db, pinned_only=True, pin_ids=["cam01"])
    assert [c.id for c in pinned] == ["cam01"]
    assert set(PIN_DEFAULT) >= {"cam01", "cam02", "cam03", "cam05"}


def test_hunt_session_does_not_reconnect(db):
    cam = add_camera(db, id="cam01", source_type="rtsp", source_uri="rtsp://x", catalogue_live=True)
    opens = {"n": 0}

    def open_fn(_camera):
        opens["n"] += 1
        cap = FakeCap(True, frames=[np.zeros((40, 80, 3), np.uint8)], pts=[10])
        return OpenedSource(cap, "rtsp", "rtsp://x")

    stop = threading.Event()
    state = type("S", (), {"run_id": "r", "status": "", "frames": 0, "last_pts_ms": None, "reconnect_attempt": 0, "error": ""})()
    run_live_loop(
        db,
        cam,
        stop=stop,
        state=state,
        open_fn=open_fn,
        sleep_fn=lambda _s: None,
        now_fn=lambda: 1.0,
        process_fn=lambda *_a: None,
        max_frames=1,
        reconnect=False,
    )
    assert opens["n"] == 1
    assert cam.analytics_active is False


def test_hunt_requeue_completes_a_cycle():
    mgr = WorkerManager(max_workers=2)
    ids = [f"cam{i:02d}" for i in range(5)]
    mgr.begin_hunt(ids)
    assert mgr.hunt_enabled is True
    assert mgr.snapshot()["queued_count"] == 5
    for cid in ids:
        mgr.mark_hunted(cid)
    assert mgr.hunt_cycle == 2
    snap = mgr.snapshot()
    assert snap["queued_count"] >= 5
    mgr.end_hunt()
    assert mgr.hunt_enabled is False


def test_live_yolo_without_plate_persists_no_alert(db, monkeypatch):
    cam = add_camera(
        db,
        id="cam01",
        source_type="rtsp",
        source_uri="rtsp://x",
        catalogue_live=True,
        catalogue_camera_id="cam01",
        processing_mode="local_worker",
        target_analysis_fps=10.0,
    )
    add_watchlist(db)
    calls = {"n": 0}

    def fake_crops(_bgr, live=False):
        return [
            {
                "crop": np.zeros((40, 40, 3), np.uint8),
                "box": (10, 20, 40, 30),
                "vehicle_type": "car",
                "detector": "yolov8n",
                "det_conf": 0.9,
            }
        ]

    def fake_infer(*_a, **_k):
        calls["n"] += 1
        return {
            "plate_raw": "",
            "plate_norm": "",
            "confidence": 0.2,
            "vehicle_type": "car",
            "vehicle_color": "white",
            "model_id": "ollama:gemma4:31b",
            "provider": "ollama_vision",
        }

    monkeypatch.setattr("app.services.pipeline.anpr_crops", fake_crops)
    monkeypatch.setattr("app.services.pipeline.infer_vehicle", fake_infer)
    frames = [(0, np.zeros((360, 640, 3), np.uint8), 0.0)]
    out = process_frame_iter(db, cam, iter(frames))
    assert out["ok"] is True
    rows = list(db.scalars(select(Sighting)))
    assert len(rows) == 1
    assert rows[0].plate_norm == ""
    assert rows[0].vehicle_type == "car"
    assert rows[0].vehicle_color == "white"
    blob = rows[0].vehicle_json or {}
    assert blob["vehicle"]["unreadable_reason"] == "no_plate"
    assert blob["vehicle"]["color"] == "white"
    assert (blob.get("gemma") or {}).get("called") is True
    assert (blob.get("gemma") or {}).get("plate_text") == ""
    assert db.scalar(select(func.count(Sighting.id))) == 1
    assert db.scalar(select(func.count(Alert.id))) == 0
    assert calls["n"] == 1


def test_large_yolo_crop_calls_ollama(db, monkeypatch):
    cam = add_camera(
        db,
        id="cam03",
        source_type="rtsp",
        source_uri="rtsp://x",
        catalogue_live=True,
        catalogue_camera_id="cam03",
        processing_mode="local_worker",
        target_analysis_fps=10.0,
    )
    add_watchlist(db, plate="GJ18BV7580")
    calls = {"n": 0}

    def fake_crops(_bgr, live=False):
        return [
            {
                "crop": np.zeros((80, 160, 3), np.uint8),
                "box": (20, 40, 160, 80),
                "vehicle_type": "car",
                "detector": "yolov8n",
                "det_conf": 0.88,
            }
        ]

    def fake_infer(*_a, **_k):
        calls["n"] += 1
        return {
            "plate_raw": "GJ18BV7580",
            "plate_norm": "GJ18BV7580",
            "confidence": 0.8,
            "vehicle_type": "car",
            "vehicle_make": "",
            "vehicle_model": "",
            "vehicle_color": "white",
            "model_id": "ollama:gemma4:31b",
            "model_hash": "x",
        }

    monkeypatch.setattr("app.services.pipeline.anpr_crops", fake_crops)
    monkeypatch.setattr("app.services.pipeline.infer_vehicle", fake_infer)
    frames = [(0, np.zeros((360, 640, 3), np.uint8), 0.0)]
    process_frame_iter(db, cam, iter(frames))
    assert calls["n"] == 1
    row = db.scalar(select(Sighting))
    assert row.plate_norm == "GJ18BV7580"
    assert row.vehicle_color == "white"
    assert db.scalar(select(func.count(Alert.id))) == 1


def test_detect_plate_boxes_finds_green_ev_plate():
    from app.services.anpr import detect_plate_boxes

    frame = np.zeros((160, 320, 3), dtype=np.uint8)
    frame[70:95, 40:160] = (40, 180, 40)
    hits = detect_plate_boxes(frame, min_width=20, skip_top=True, skip_bottom=False)
    assert hits
    assert hits[0][2] >= 20


def test_detect_plate_boxes_skips_bottom_hud():
    from app.services.anpr import detect_plate_boxes

    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    frame[184:198, 40:110] = (255, 255, 255)
    assert detect_plate_boxes(frame, min_width=20, skip_top=True, skip_bottom=True) == []
    hits = detect_plate_boxes(frame, min_width=20, skip_top=True, skip_bottom=False)
    assert hits


def test_enhance_upscales_small_crop():
    from app.services.anpr import enhance_plate_crop

    tiny = np.zeros((20, 40, 3), dtype=np.uint8)
    tiny[:] = (40, 40, 200)
    out = enhance_plate_crop(tiny, min_width=320)
    assert out.shape[1] >= 320
