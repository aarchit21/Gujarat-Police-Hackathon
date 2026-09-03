from app.services.capacity import calibrated_target_fps, measure_government_decode, runnable_cameras, start_accessible_workers
from tests.conftest import add_camera


def test_calibrated_fps_drops_below_hypothesis():
    assert calibrated_target_fps(0.5, 2.0) == 0.4
    assert calibrated_target_fps(5.0, 2.0) == 2.0
    assert calibrated_target_fps(0, 2.0) == 2.0


def test_runnable_cameras_skips_untested_rtsp(db):
    add_camera(db, id="cam-ok", source_type="rtsp", source_uri="rtsp://x", decode_status="ok", processing_mode="local_worker", priority_class="B")
    add_camera(db, id="cam-no", source_type="rtsp", source_uri="rtsp://y", decode_status="untested", processing_mode="local_worker", priority_class="A")
    add_camera(db, id="cam-own", source_type="image_dir", source_uri="C:/frames", processing_mode="local_worker", priority_class="A")
    ids = [c.id for c in runnable_cameras(db, decode_ok_only=True)]
    assert "cam-ok" in ids
    assert "cam-own" in ids
    assert "cam-no" not in ids


def test_start_accessible_queues_overflow(db):
    add_camera(db, id="cam-a", source_type="image_dir", source_uri="C:/a", processing_mode="local_worker", priority_class="A")
    add_camera(db, id="cam-b", source_type="image_dir", source_uri="C:/b", processing_mode="local_worker", priority_class="D")
    calls = []

    class FakeMgr:
        max_workers = 1

        def start(self, _db, camera_id, actor="operator"):
            calls.append(camera_id)
            if len([c for c in calls if c == camera_id or True]) == 1 and len(calls) == 1:
                return {"ok": True, "state": "starting"}
            return {"ok": True, "state": "queued", "analytics_active": False}

    out = start_accessible_workers(FakeMgr(), db)
    assert out["started"] == ["cam-a"]
    assert out["queued"] == ["cam-b"]
    assert "analytics_active=false" in out["disclaimer"].lower() or "Queued" in out["disclaimer"]


def test_measure_sequential_limit_and_decode_status(db):
    add_camera(
        db,
        id="cam01",
        source_type="rtsp",
        source_uri="rtsp://103.250.160.189:8554/stream/cam01",
        catalogue_camera_id="cam01",
        processing_mode="local_worker",
        decode_status="untested",
    )
    add_camera(
        db,
        id="cam02",
        source_type="rtsp",
        source_uri="rtsp://103.250.160.189:8554/stream/cam02",
        catalogue_camera_id="cam02",
        processing_mode="local_worker",
        decode_status="untested",
    )

    def probe(url, timeout=12.0):
        ok = url.endswith("cam01")
        return {
            "ok": ok,
            "frame": ok,
            "width": 1920 if ok else None,
            "height": 1080 if ok else None,
            "pts_ms": 100 if ok else None,
            "error": "" if ok else "no frame",
        }

    out = measure_government_decode(db, limit=2, probe_fn=probe)
    assert out["tested_count"] == 2
    assert out["decode_ok_count"] == 1
    assert "cam01" in out["decode_ok"]
    from app.models import Camera

    cam01 = db.get(Camera, "cam01")
    cam02 = db.get(Camera, "cam02")
    assert cam01.decode_status == "ok"
    assert cam01.width == 1920
    assert cam02.decode_status == "failed"
    assert cam02.analytics_active is False


def test_measure_skips_already_ok_and_probes_untested(db):
    add_camera(
        db,
        id="cam01",
        source_type="rtsp",
        source_uri="rtsp://x/cam01",
        catalogue_camera_id="cam01",
        processing_mode="local_worker",
        decode_status="ok",
    )
    add_camera(
        db,
        id="cam05",
        source_type="rtsp",
        source_uri="rtsp://x/cam05",
        catalogue_camera_id="cam05",
        processing_mode="deferred",
        decode_status="untested",
    )
    seen = []

    def probe(url, timeout=12.0):
        seen.append(url.split("@")[-1] if "@" in url else url)
        return {"ok": True, "frame": True, "width": 1280, "height": 720, "pts_ms": 10, "error": ""}

    out = measure_government_decode(db, limit=1, probe_fn=probe)
    assert seen == ["x/cam05"]
    assert out["decode_ok"] == ["cam05"]
    assert "cam01" in (out.get("already_decode_ok") or [])
    from app.models import Camera

    assert db.get(Camera, "cam05").decode_status == "ok"


def test_measure_retries_failed_preferring_known_size(db):
    add_camera(
        db,
        id="cam04",
        source_type="rtsp",
        source_uri="rtsp://x/cam04",
        catalogue_camera_id="cam04",
        processing_mode="local_worker",
        decode_status="failed",
        width=None,
    )
    add_camera(
        db,
        id="cam01",
        source_type="rtsp",
        source_uri="rtsp://x/cam01",
        catalogue_camera_id="cam01",
        processing_mode="local_worker",
        decode_status="failed",
        width=1920,
        height=1080,
    )
    seen = []

    def probe(url, timeout=12.0):
        seen.append(url.split("@")[-1] if "@" in url else url)
        return {"ok": True, "frame": True, "width": 1920, "height": 1080, "pts_ms": 10, "error": ""}

    out = measure_government_decode(db, limit=1, probe_fn=probe)
    assert seen == ["x/cam01"]
    assert out["decode_ok"] == ["cam01"]
    assert out["catalogue_remaining_untested"] == 0


def test_start_accessible_promotes_deferred_decode_ok(db):
    add_camera(
        db,
        id="cam01",
        source_type="rtsp",
        source_uri="rtsp://x",
        catalogue_camera_id="cam01",
        processing_mode="deferred",
        decode_status="ok",
        priority_class="C",
    )

    class FakeMgr:
        max_workers = 4

        def start(self, _db, camera_id, actor="operator"):
            from app.models import Camera

            cam = _db.get(Camera, camera_id)
            assert cam.processing_mode == "local_worker"
            return {"ok": True, "state": "starting"}

    out = start_accessible_workers(FakeMgr(), db)
    assert out["started"] == ["cam01"]
    assert "cam01" in (out.get("promoted") or [])
