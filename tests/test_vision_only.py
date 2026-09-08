import json

import httpx
import numpy as np
from sqlalchemy import select

from app.config import settings
from app.models import Sighting
from app.services.ollama_vision import infer_named_vision, infer_vision_only_frame
from app.services.pipeline import process_frame_iter
from app.services.workers import manager
from tests.conftest import add_camera, add_watchlist


def test_vision_only_gemma_then_glm_when_gemma_empty(monkeypatch):
    monkeypatch.setattr(settings, "vision_only_models", "gemma4:31b,glm-5.3-flash")
    calls = []

    def fake_named(_bgr, *, model, max_width=1280, client=None):
        calls.append(model)
        if model.startswith("gemma"):
            return {"plate_norm": "", "plate_raw": "", "skipped": "", "vehicle_type": "car"}
        return {
            "plate_norm": "GJ01AB1234",
            "plate_raw": "GJ01AB1234",
            "vehicle_type": "car",
            "skipped": "",
        }

    monkeypatch.setattr("app.services.ollama_vision.infer_named_vision", fake_named)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = infer_vision_only_frame(frame, camera_id="cam01")
    assert calls[0] == "gemma4:31b"
    assert "glm-5.3-flash" in calls
    assert out["chosen_plate"] == "GJ01AB1234"
    assert out["chosen_model"] == "glm-5.3-flash"


def test_vision_only_glm_404_does_not_disable_gemma(monkeypatch):
    monkeypatch.setattr(settings, "vision_only_models", "gemma4:31b,glm-5.3-flash")

    def fake_named(_bgr, *, model, max_width=1280, client=None):
        if "glm" in model:
            return {"plate_norm": "", "skipped": "unavailable", "model_id": f"ollama:{model}"}
        return {"plate_norm": "", "plate_raw": "", "skipped": "", "vehicle_type": "car"}

    monkeypatch.setattr("app.services.ollama_vision.infer_named_vision", fake_named)
    out = infer_vision_only_frame(np.zeros((80, 80, 3), dtype=np.uint8))
    glm = out["models"].get("glm-5.3-flash") or {}
    assert glm.get("skipped") == "unavailable"
    assert (out["models"].get("gemma4_31b") or {}).get("skipped") == ""


def test_named_vision_retries_glm_cloud_alias(monkeypatch):
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_vision_enabled", True)
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_url", "https://ollama.com")
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_api_key", "unit-ollama-cloud-key")
    monkeypatch.setattr("app.services.ollama_vision._cloud_disabled_reason", "")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        seen.append(payload["model"])
        if payload["model"] == "glm-5.3-flash":
            return httpx.Response(404, json={"error": "model not found"})
        if payload["model"] == "glm-5.3-flash:cloud":
            return httpx.Response(
                200,
                json={
                    "message": {"content": '{"plate_text":"GJ01AB1234","vehicle_type":"car"}'},
                    "model": "glm-5.3-flash:cloud",
                },
            )
        return httpx.Response(410, json={"error": "unexpected model"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = infer_named_vision(
        np.zeros((80, 160, 3), dtype=np.uint8),
        model="glm-5.3-flash",
        client=client,
    )
    assert seen == ["glm-5.3-flash", "glm-5.3-flash:cloud"]
    assert out.get("skipped") == ""
    assert out.get("plate_norm") == "GJ01AB1234"
    assert out.get("model_id") == "ollama:glm-5.3-flash:cloud"
    assert "gemma" not in "".join(seen)


def test_yolo_still_runs_when_vision_only_on(db, monkeypatch):
    manager.vision_only = True
    monkeypatch.setattr(settings, "vision_only_interval_seconds", 0.0)
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
    yolo = {"n": 0}

    def fake_crops(_bgr, live=False):
        yolo["n"] += 1
        return [
            {
                "crop": np.zeros((80, 160, 3), np.uint8),
                "box": (20, 40, 160, 80),
                "vehicle_type": "car",
                "detector": "yolov8n",
                "det_conf": 0.9,
            }
        ]

    def fake_infer(*_a, **_k):
        return {
            "plate_raw": "",
            "plate_norm": "",
            "confidence": 0.2,
            "vehicle_type": "car",
            "vehicle_color": "white",
            "model_id": "ollama:gemma4:31b",
            "provider": "ollama_vision",
        }

    def fake_vo(_bgr, camera_id=""):
        return {
            "models": {
                "gemma4_31b": {"model": "gemma4:31b", "plate_text": "", "skipped": ""},
                "glm-5.3-flash": {"model": "glm-5.3-flash", "plate_text": "", "skipped": "unavailable"},
            },
            "chosen_plate": "",
            "chosen_model": "",
        }

    monkeypatch.setattr("app.services.pipeline.anpr_crops", fake_crops)
    monkeypatch.setattr("app.services.pipeline.infer_vehicle", fake_infer)
    queued = []
    monkeypatch.setattr(settings, "cloud_verifier_queue_enabled", True)
    monkeypatch.setattr("app.services.pipeline.cloud_verifier.enqueue", lambda job: queued.append(job) or True)
    frames = [(0, np.zeros((360, 640, 3), np.uint8), 0.0)]
    process_frame_iter(db, cam, iter(frames))
    manager.vision_only = False
    assert yolo["n"] >= 1
    row = db.scalar(select(Sighting))
    assert row is not None
    blob = row.vehicle_json or {}
    assert "vision_only" in blob
    assert blob["vision_only"]["queued"] is True
    assert any(job.kind == "vision_only" for job in queued)
