import json

import httpx
import numpy as np
import pytest

from app.services.ollama_vision import (
    PLATE_PROMPT,
    OllamaVisionError,
    infer_bgr,
    parse_vision_text,
    resolve_vision_model,
    should_use_vision,
)
from tests.conftest import add_camera


def test_prompt_does_not_mention_watchlist():
    assert "GJ01" not in PLATE_PROMPT
    assert "watchlist" not in PLATE_PROMPT.lower()


def test_parse_vision_json():
    plate, conf = parse_vision_text('{"plate_text":"GJ05CD1234","confidence":0.8}')
    assert plate == "GJ05CD1234"
    assert conf == 0.8


def test_resolve_prefers_installed_candidate(monkeypatch):
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_api_key", "")
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_url", "http://127.0.0.1:11434")
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_vision_model", "llava:7b")
    assert resolve_vision_model(["llama3:8b", "llava:7b"]) == "llava:7b"


def test_should_use_vision_only_when_tesseract_fails_on_rtsp(db, monkeypatch):
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_vision_enabled", True)
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_vision_on_own_feed", False)
    cam = add_camera(db, source_type="rtsp")
    assert should_use_vision(cam, tess_syntax=False) is True
    assert should_use_vision(cam, tess_syntax=True) is False
    own = add_camera(db, id="CAM-OWN", source_type="image_dir")
    assert should_use_vision(own, tess_syntax=False) is False


def test_infer_bgr_success_with_mock(monkeypatch):
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_vision_enabled", True)
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_url", "http://127.0.0.1:11434")
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_api_key", "")
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_vision_model", "llava:7b")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        body = request.read()
        assert b"GJ01AB1234" not in body
        return httpx.Response(200, json={"response": '{"plate_text":"GJ05CD8888","confidence":0.7}', "model": "llava:7b"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    read = infer_bgr(frame, client=client, model="llava:7b")
    assert read.plate_norm == "GJ05CD8888"
    assert read.model_id == "ollama:llava:7b"


def test_infer_rejects_non_local_host(monkeypatch):
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_vision_enabled", True)
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_api_key", "")
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_url", "http://evil.example:11434")
    with pytest.raises(OllamaVisionError):
        infer_bgr(np.zeros((10, 10, 3), dtype=np.uint8), model="llava:7b")


def test_cloud_requires_api_key(monkeypatch):
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_vision_enabled", True)
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_url", "https://ollama.com")
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_api_key", "")
    with pytest.raises(OllamaVisionError, match="OLLAMA_API_KEY"):
        infer_bgr(np.zeros((10, 10, 3), dtype=np.uint8), model="gemma3:4b")


def test_cloud_sends_bearer_and_does_not_use_llava_default(monkeypatch):
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_vision_enabled", True)
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_url", "https://ollama.com")
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_api_key", "unit-ollama-cloud-key")
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_vision_model", "llava:7b")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "ollama.com"
        assert request.headers.get("authorization") == "Bearer unit-ollama-cloud-key"
        assert b"unit-ollama-cloud-key" not in request.content
        payload = json.loads(request.content.decode())
        assert payload["model"] == "gemma3:4b"
        return httpx.Response(200, json={"response": '{"plate_text":"GJ05CD7777","confidence":0.6}', "model": "gemma3:4b"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    read = infer_bgr(np.zeros((80, 160, 3), dtype=np.uint8), client=client)
    assert read.plate_norm == "GJ05CD7777"
    assert read.model_id == "ollama:gemma3:4b"


def test_api_key_not_returned_in_status(monkeypatch):
    from app.services.ollama_vision import vision_status

    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_api_key", "unit-ollama-cloud-key")
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_url", "https://ollama.com")
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_vision_enabled", True)
    monkeypatch.setattr("app.services.ollama_vision.settings.ollama_vision_model", "gemma3:4b")
    monkeypatch.setattr("app.services.ollama_vision.list_ollama_models", lambda: ["gemma3:4b"])
    status = vision_status()
    assert status["api_key_configured"] is True
    assert status["cloud"] is True
    assert status["live"] is True
    assert "Ollama Cloud" in status["label"]
    assert "live" in status["label"]
    assert "unit-ollama-cloud-key" not in str(status)
