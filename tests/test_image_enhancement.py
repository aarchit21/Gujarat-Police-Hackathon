import base64
import json
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
from sqlalchemy import select

from app.config import settings
from app.models import Alert, Sighting
from app.services.anpr import ENHANCEMENT_METHOD, enhance_for_vision, vision_only_views
from app.services.ollama_vision import infer_named_vision
from app.services.pipeline import _read_plate, process_frame_iter
from tests.conftest import add_camera, add_watchlist, fake_read


def test_enhancement_is_deterministic_non_mutating_and_capped(monkeypatch):
    monkeypatch.setattr(settings, "vision_enhancement_enabled", True)
    crop = np.full((32, 100, 3), 112, dtype=np.uint8)
    cv2.putText(crop, "GJ01", (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (126, 126, 126), 1, cv2.LINE_AA)
    original = crop.copy()

    first, meta = enhance_for_vision(crop, profile="plate", min_width=1000)
    second, second_meta = enhance_for_vision(crop, profile="plate", min_width=1000)

    assert np.array_equal(crop, original)
    assert np.array_equal(first, second)
    assert meta == second_meta
    assert first.shape[1] == 400
    assert meta["scale"] == 4.0
    assert meta["method"] == ENHANCEMENT_METHOD
    assert cv2.cvtColor(first, cv2.COLOR_BGR2GRAY).std() > cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).std()


def test_enhancement_handles_empty_tiny_and_disabled(monkeypatch):
    monkeypatch.setattr(settings, "vision_enhancement_enabled", True)
    empty, empty_meta = enhance_for_vision(np.zeros((0, 0, 3), np.uint8), profile="plate")
    assert empty.size == 0
    assert empty_meta["view_count"] == 0

    tiny = np.full((1, 1, 3), 77, np.uint8)
    tiny_out, tiny_meta = enhance_for_vision(tiny, profile="plate", min_width=400)
    assert np.array_equal(tiny_out, tiny)
    assert tiny_meta["scale"] == 1.0

    monkeypatch.setattr(settings, "vision_enhancement_enabled", False)
    frame = np.full((80, 160, 3), 90, np.uint8)
    views, meta = vision_only_views(frame)
    assert len(views) == 1
    assert meta["method"] == "none"
    # HUD masking is legacy vision-only preprocessing and remains enabled.
    assert np.all(views[0][:9] == 0)


def test_vision_only_sends_enhanced_context_and_zoom_views(monkeypatch):
    monkeypatch.setattr(settings, "vision_enhancement_enabled", True)
    monkeypatch.setattr(settings, "ollama_vision_enabled", True)
    monkeypatch.setattr(settings, "ollama_url", "http://127.0.0.1:11434")
    monkeypatch.setattr(settings, "ollama_api_key", "")
    monkeypatch.setattr("app.services.ollama_vision._cloud_disabled_reason", "")
    frame = np.full((360, 640, 3), 90, np.uint8)
    frame[180:220, 220:420] = 125
    views, meta = vision_only_views(frame, max_width=1280)
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        captured.update(payload)
        return httpx.Response(
            200,
            json={"response": '{"plate_text":"","vehicle_type":"car"}', "model": "llava:7b"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = infer_named_vision(views, model="llava:7b", client=client)

    images = captured.get("images")
    if not images:
        messages = captured.get("messages") or []
        images = (messages[0] or {}).get("images") if messages else []
    assert len(images) == meta["view_count"] == 3
    decoded = [
        cv2.imdecode(np.frombuffer(base64.b64decode(blob), np.uint8), cv2.IMREAD_COLOR)
        for blob in images
    ]
    assert all(image is not None for image in decoded)
    assert decoded[1].shape[1] == 1280
    prompt = captured.get("prompt") or ""
    if not prompt:
        messages = captured.get("messages") or []
        prompt = str((messages[0] or {}).get("content") or "") if messages else ""
    assert "same frame" in prompt
    assert result["enhancement"]["view_count"] == 3


def test_yolo_tight_plate_is_enhanced_once_before_vision(db, monkeypatch):
    monkeypatch.setattr(settings, "vision_enhancement_enabled", True)
    monkeypatch.setattr(settings, "ollama_vision_enabled", True)
    cam = add_camera(db, source_type="rtsp", source_uri="rtsp://example")
    body = np.full((100, 240, 3), 80, np.uint8)
    plate = np.full((28, 120, 3), 120, np.uint8)
    captured = {}

    monkeypatch.setattr(
        "app.services.pipeline.anpr_crops",
        lambda _bgr, live=False: [
            {
                "crop": body,
                "body_crop": body,
                "plate_crops": [plate],
                "box": (10, 20, 240, 100),
                "vehicle_type": "car",
                "detector": "yolov8n",
            }
        ],
    )

    def fake_plate_read(image, *, prepared=False, enhancement=None, **_kwargs):
        captured["shape"] = image.shape
        captured["prepared"] = prepared
        captured["enhancement"] = enhancement
        return SimpleNamespace(
            plate_raw="GJ01AB1234",
            plate_norm="GJ01AB1234",
            confidence=0.9,
            model_id="ollama:test",
            model_hash="test",
            enhancement=enhancement or {},
        )

    monkeypatch.setattr("app.services.pipeline.infer_bgr", fake_plate_read)
    result = _read_plate(
        cam,
        np.zeros((240, 480, 3), np.uint8),
        provider_kind="local_worker",
        reader=None,
        remote_client=None,
        local_hash="local",
    )

    assert captured["prepared"] is True
    assert captured["shape"][1] == 400
    assert captured["enhancement"]["profile"] == "plate"
    assert result["enhancement"]["profile"] == "plate"
    assert result["crop"].shape[1] == 400


def test_own_feed_alert_is_created_from_persisted_sighting(db):
    cam = add_camera(db, source_type="image_dir", target_analysis_fps=10.0)
    add_watchlist(db, "GJ01AB1234")
    frames = [
        (0, np.zeros((120, 320, 3), np.uint8), 0.0),
        (1, np.zeros((120, 320, 3), np.uint8), 100.0),
    ]

    out = process_frame_iter(db, cam, iter(frames), read_fn=fake_read, run_id="enhanced-own-feed")
    sighting = db.scalar(select(Sighting).order_by(Sighting.id.desc()))
    alert = db.scalar(select(Alert))

    assert out["sightings"] == 2
    assert out["alerts"] == 1
    assert sighting is not None and sighting.id is not None
    assert alert is not None and alert.sighting_id == sighting.id
