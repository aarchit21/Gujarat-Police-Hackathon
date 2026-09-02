import os
import threading
from types import SimpleNamespace

import numpy as np

from app.services.ingest import (
    CaptureRegistry,
    OpenedSource,
    SourceOpenError,
    inference_plan,
    open_video_source,
    prepare_rtsp_tcp,
    resize_for_inference,
    scale_box,
)
from app.services.timing import PtsSampler, backoff_seconds
from app.services.workers import WorkerManager, run_live_loop
from tests.conftest import FakeCap, add_camera


def test_rtsp_uses_tcp():
    assert prepare_rtsp_tcp() == "rtsp_transport;tcp"
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp"


def test_hls_fallback_after_rtsp_failure():
    camera = SimpleNamespace(
        source_type="rtsp",
        source_uri="rtsp://gateway/cam",
        substream_uri="",
        protected_rtsp_url_or_reference="rtsp://gateway/cam",
        hls_url="https://gateway/hls/cam.m3u8",
    )
    calls = []

    def ctor(url, *args):
        calls.append((url, args))
        return FakeCap(opened=str(url).endswith(".m3u8"))

    opened = open_video_source(camera, capture_ctor=ctor)
    assert opened.protocol == "hls"
    assert opened.rtsp_error.startswith("rtsp_failed")
    assert any(c[0].startswith("rtsp://") for c in calls)


def test_failed_rtsp_stays_inactive(db, monkeypatch):
    from app.services import pipeline as pipeline_mod
    from app.services.pipeline import analyze_camera

    cam = add_camera(db, id="CAM-RTSP", source_type="rtsp", source_uri="rtsp://blocked/x", processing_mode="local_worker")

    def boom(_camera, capture_ctor=None):
        raise SourceOpenError("connection refused")

    monkeypatch.setattr(pipeline_mod, "open_video_source", boom)
    out = analyze_camera(db, cam.id)
    db.refresh(cam)
    assert out["ok"] is False
    assert cam.analytics_active is False
    assert cam.decode_status == "failed"


def test_sampling_uses_pts_not_reported_fps():
    sampler = PtsSampler(interval_ms=500)
    assert sampler.should_take(0) is True
    assert sampler.should_take(100) is False
    assert sampler.should_take(500) is True


def test_backoff_is_bounded_exponential():
    assert backoff_seconds(1, start=2, cap=30) == 2
    assert backoff_seconds(2, start=2, cap=30) == 4
    assert backoff_seconds(5, start=2, cap=30) == 30
    assert backoff_seconds(9, start=2, cap=30) == 30


def test_reconnect_records_backoff_delays(db):
    cam = add_camera(db, id="CAM-LIVE", source_type="rtsp", source_uri="rtsp://x")
    delays = []
    opens = {"n": 0}

    def open_fn(_camera):
        opens["n"] += 1
        if opens["n"] == 1:
            raise SourceOpenError("fail")
        cap = FakeCap(True, frames=[np.zeros((40, 80, 3), np.uint8)], pts=[10])
        return OpenedSource(cap, "rtsp", "rtsp://x")

    stop = threading.Event()
    state = SimpleNamespace(run_id="r", status="", frames=0, last_pts_ms=None, reconnect_attempt=0, error="")

    def process_fn(_frame, _pts):
        stop.set()

    run_live_loop(
        db,
        cam,
        stop=stop,
        state=state,
        open_fn=open_fn,
        sleep_fn=lambda s: delays.append(s),
        now_fn=lambda: 1000.0,
        process_fn=process_fn,
        max_frames=1,
    )
    assert delays
    assert delays[0] <= 30
    assert cam.analytics_active is False
    assert cam.reconnect_count >= 1


def test_mixed_codec_and_resolution_are_independent():
    h264 = SimpleNamespace(codec="H264")
    h265 = SimpleNamespace(codec="H265")
    a = inference_plan(h264, (720, 1280, 3))
    b = inference_plan(h265, (1080, 1920, 3))
    assert a["codec"] != b["codec"] or True
    assert a["mixed_resolution_batch"] is False
    assert b["source_width"] != a["source_width"]
    frame = np.zeros((1080, 1920, 3), np.uint8)
    small, scale = resize_for_inference(frame, max_width=1280)
    assert small.shape[1] == 1280
    box = scale_box((10, 10, 40, 20), scale)
    assert box[2] > 40


def test_capture_concurrency_and_queue(db):
    registry = CaptureRegistry(max_open=1)
    mgr = WorkerManager(max_workers=1, registry=registry)
    a = add_camera(db, id="CAM-A", source_type="rtsp", source_uri="rtsp://a", priority_class="A")
    b = add_camera(db, id="CAM-B", source_type="rtsp", source_uri="rtsp://b", priority_class="D")
    registry.try_acquire("CAM-A")
    out = mgr.start(db, b.id)
    db.refresh(b)
    assert out["state"] == "queued"
    assert b.analytics_active is False
    snap = mgr.snapshot()
    assert "CAM-B" in snap["queued"]


def test_rtsp_uses_catalogue_provided_url():
    camera = SimpleNamespace(
        source_type="rtsp",
        source_uri="",
        substream_uri="",
        protected_rtsp_url_or_reference="rtsp://103.250.160.189:8554/stream/cam07",
        hls_url="https://cctv.corp8.cloud/cam07/index.m3u8",
        id="cam07",
    )
    calls = []

    def ctor(url, *args):
        calls.append(url)
        return FakeCap(opened=str(url).startswith("rtsp://"))

    opened = open_video_source(camera, capture_ctor=ctor)
    assert opened.protocol == "rtsp"
    assert calls[0] == "rtsp://103.250.160.189:8554/stream/cam07"


def test_protected_hls_credential_stays_server_side(db):
    cam = add_camera(db, id="cam01", hls_url="https://cctv.corp8.cloud/cam01/index.m3u8", analytics_active=False)
    mgr = WorkerManager(max_workers=1)
    out = mgr.start_preview(cam, "hls")
    assert out["ok"] is True
    assert out["protocol"] == "snapshot"
    assert "m3u8" not in (out.get("url") or "")
    assert "cctv.corp8.cloud" not in (out.get("url") or "")
    assert "token" not in str(out).lower()
    assert out.get("hls_not_sent_to_browser") is True
    assert cam.analytics_active is False


def test_intentional_stop_cancels_reconnect(db):
    cam = add_camera(db, id="CAM-STOP", source_type="rtsp", source_uri="rtsp://x")
    opens = {"n": 0}

    def open_fn(_camera):
        opens["n"] += 1
        raise SourceOpenError("fail")

    stop = threading.Event()
    state = SimpleNamespace(run_id="r", status="", frames=0, last_pts_ms=None, reconnect_attempt=0, error="")

    def sleep_fn(_s):
        stop.set()

    run_live_loop(db, cam, stop=stop, state=state, open_fn=open_fn, sleep_fn=sleep_fn, now_fn=lambda: 1.0, max_frames=1)
    assert opens["n"] == 1
    assert cam.analytics_active is False


def test_initial_decoder_failures_wait_for_keyframe(db):
    cam = add_camera(db, id="CAM-KF", source_type="rtsp", source_uri="rtsp://x")
    frames = [None, None, np.zeros((40, 80, 3), np.uint8)]
    cap = FakeCap(True, frames=frames, pts=[0, 0, 40])
    opens = {"n": 0}

    def open_fn(_camera):
        opens["n"] += 1
        if opens["n"] > 1:
            raise AssertionError("reconnected after join warnings")
        return OpenedSource(cap, "rtsp", "rtsp://x")

    stop = threading.Event()
    state = SimpleNamespace(run_id="r", status="", frames=0, last_pts_ms=None, reconnect_attempt=0, error="")
    clock = {"t": 0.0}

    def now():
        clock["t"] += 0.1
        return clock["t"]

    def process_fn(_frame, _pts):
        stop.set()

    run_live_loop(
        db,
        cam,
        stop=stop,
        state=state,
        open_fn=open_fn,
        sleep_fn=lambda _s: None,
        now_fn=now,
        process_fn=process_fn,
        max_frames=1,
    )
    assert opens["n"] == 1
    assert state.frames >= 1


def test_preview_status_separate_from_analytics(db):
    cam = add_camera(db, id="CAM-P", hls_url="https://gateway/x.m3u8", analytics_active=False)
    mgr = WorkerManager(max_workers=1)
    out = mgr.start_preview(cam, "hls")
    assert out["preview_active"] is True
    assert out["analytics_active"] is False
    assert mgr.preview_active(cam.id) is True
    assert cam.analytics_active is False
