"""Optional Ollama vision OCR (local GPU or Ollama Cloud).

Local: OLLAMA_URL=http://127.0.0.1:11434 (no API key).
Cloud: https://ollama.com (requires OLLAMA_API_KEY).

Never put watchlist plates in the prompt. Never log the API key.
model_id is ollama:<actual-model> only when that model is called.
"""
from __future__ import annotations

import base64
import json
import re
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import numpy as np

from app.config import settings
from app.security import redact_secrets
from app.services.plates import normalize
from app.services.vehicle_event import VEHICLE_PROMPT, parse_vehicle_payload

VISION_CANDIDATES = (
    "llava:7b",
    "llava:7b-v1.6",
    "llava:latest",
    "gemma4:e4b",
    "gemma4:e2b",
    "gemma4",
    "qwen3.5:4b",
    "gemma3:4b",
    "gemma3:12b",
    "qwen3-vl",
    "qwen3-vl:4b",
    "moondream:1.8b",
)
# gemma3:4b was retired from Ollama Cloud on 2026-07-15 (HTTP 410).
# This account's /api/tags lists gemma4:31b as the available vision cloud model.
CLOUD_DEFAULT_MODEL = "gemma4:31b"
RETIRED_CLOUD_PREFIXES = ("gemma3", "llava", "qwen3-vl", "moondream")
CLOUD_HOSTS = {"ollama.com", "www.ollama.com"}
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

PLATE_PROMPT = (
    "You are reading a still from a traffic camera. "
    "If a vehicle registration plate is visible, transcribe only the plate characters you can actually see. "
    "Return JSON only, no markdown: {\"plate_text\":\"\",\"confidence\":0.0}. "
    "plate_text must use A-Z and 0-9 only. "
    "If no plate is readable, return {\"plate_text\":\"\",\"confidence\":0.0}. "
    "Do not guess a typical or example plate. Do not describe people."
)


class OllamaVisionError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(redact_secrets(message))


@dataclass
class VisionRead:
    plate_raw: str
    plate_norm: str
    confidence: float
    model_id: str
    model_hash: str
    raw_text: str


def normalize_ollama_base(url: str | None = None) -> str:
    raw = (url or settings.ollama_url or "").strip().rstrip("/")
    if raw.endswith("/api"):
        raw = raw[:-4]
    return raw or "http://127.0.0.1:11434"


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def is_cloud_url(url: str | None = None) -> bool:
    return _host(normalize_ollama_base(url)) in CLOUD_HOSTS


def is_local_url(url: str | None = None) -> bool:
    return _host(normalize_ollama_base(url)) in LOCAL_HOSTS


def effective_ollama_url() -> str:
    """Use Ollama Cloud when an API key is set and the URL is still the local default."""
    base = normalize_ollama_base()
    if is_cloud_url(base):
        return "https://ollama.com"
    if (settings.ollama_api_key or "").strip() and is_local_url(base):
        return "https://ollama.com"
    return base


def ollama_host_allowed(url: str | None = None) -> bool:
    host = _host(url or effective_ollama_url())
    return host in LOCAL_HOSTS or host in CLOUD_HOSTS


def auth_headers() -> dict[str, str]:
    key = (settings.ollama_api_key or "").strip()
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


def list_ollama_models(client: httpx.Client | None = None) -> list[str]:
    base = effective_ollama_url()
    if not ollama_host_allowed(base):
        raise OllamaVisionError("OLLAMA_URL host is not local or ollama.com")
    if is_cloud_url(base) and not (settings.ollama_api_key or "").strip():
        raise OllamaVisionError("OLLAMA_API_KEY is required for Ollama Cloud")
    own = client is None
    http = client or httpx.Client(timeout=8.0)
    try:
        response = http.get(base + "/api/tags", headers=auth_headers())
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        raise OllamaVisionError(f"ollama tags failed: {exc}") from exc
    finally:
        if own:
            http.close()
    names = []
    for item in payload.get("models") or []:
        name = str(item.get("name") or item.get("model") or "").strip()
        if name:
            names.append(name)
    return names


def resolve_vision_model(available: list[str] | None = None) -> str:
    configured = (settings.ollama_vision_model or "").strip()
    cloud = is_cloud_url(effective_ollama_url())
    if cloud:
        name = configured or CLOUD_DEFAULT_MODEL
        if any(name.startswith(prefix) for prefix in RETIRED_CLOUD_PREFIXES):
            name = CLOUD_DEFAULT_MODEL
        if name.startswith("gemma4:e"):
            name = CLOUD_DEFAULT_MODEL
        names = available if available is not None else []
        if names and name not in names:
            for candidate in ("gemma4:31b", "gemma4", "gemma4:e4b", "qwen3.5:4b"):
                if candidate in names:
                    return candidate
        return name
    names = available if available is not None else []
    if configured and (not names or configured in names):
        return configured
    for candidate in VISION_CANDIDATES:
        if candidate in names:
            return candidate
    if configured:
        return configured
    raise OllamaVisionError("no Ollama vision model configured or installed")


def encode_jpeg_b64(bgr: np.ndarray, max_width: int | None = None) -> str:
    import cv2

    if bgr is None or getattr(bgr, "size", 0) == 0:
        raise OllamaVisionError("empty image")
    frame = bgr
    limit = max_width or settings.ollama_vision_max_width
    h, w = frame.shape[:2]
    if w > limit:
        scale = limit / float(w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise OllamaVisionError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def parse_vision_text(text: str) -> tuple[str, float]:
    raw = (text or "").strip()
    if not raw:
        return "", 0.0
    blob = raw
    if "```" in blob:
        blob = re.sub(r"```(?:json)?", "", blob).replace("```", "")
    try:
        payload = json.loads(blob)
        if isinstance(payload, dict):
            plate = str(payload.get("plate_text") or payload.get("plate") or payload.get("text") or "")
            try:
                conf = float(payload.get("confidence") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            return plate.strip(), max(0.0, min(1.0, conf))
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]+\}", raw)
    if match:
        try:
            payload = json.loads(match.group(0))
            plate = str(payload.get("plate_text") or payload.get("plate") or "")
            return plate.strip(), 0.5
        except json.JSONDecodeError:
            pass
    return raw.splitlines()[0].strip(), 0.2


_cloud_disabled_reason = ""
_last_vehicle_infer: dict[str, float] = {}
_vehicle_lock = threading.Lock()


def _disable_cloud(reason: str) -> None:
    global _cloud_disabled_reason
    _cloud_disabled_reason = reason[:200]


def _local_generate(http: httpx.Client, base: str, model: str, image_b64: str, prompt: str) -> dict:
    response = http.post(
        base + "/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "format": "json",
        },
        headers=auth_headers(),
    )
    response.raise_for_status()
    return response.json()


def _cloud_model_candidates(chosen: str) -> list[str]:
    out: list[str] = []
    for name in (chosen, CLOUD_DEFAULT_MODEL, "gemma4:31b", "gemma4"):
        if name and name not in out:
            out.append(name)
    return out


def _cloud_chat_fallback(http: httpx.Client, base: str, chosen: str, image_b64: str, prompt: str) -> tuple[dict, str]:
    last: Exception | None = None
    for model in _cloud_model_candidates(chosen):
        try:
            return _cloud_chat(http, base, model, image_b64, prompt), model
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status in {404, 410}:
                last = exc
                continue
            raise
    if last:
        raise last
    raise OllamaVisionError("no Cloud vision model responded")


def _cloud_chat(http: httpx.Client, base: str, model: str, image_b64: str, prompt: str) -> dict:
    """Ollama Cloud documents /api/chat for vision. /api/generate on retired models returns 410."""
    response = http.post(
        base + "/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "stream": False,
            "format": "json",
        },
        headers=auth_headers(),
    )
    response.raise_for_status()
    return response.json()


def _response_text(payload: dict) -> str:
    message = payload.get("message")
    if isinstance(message, dict) and message.get("content"):
        return str(message.get("content") or "")
    return str(payload.get("response") or payload.get("text") or "")


def infer_bgr(
    bgr: np.ndarray,
    *,
    client: httpx.Client | None = None,
    model: str | None = None,
) -> VisionRead:
    if not settings.ollama_vision_enabled:
        raise OllamaVisionError("Ollama vision is disabled")
    if _cloud_disabled_reason:
        raise OllamaVisionError(_cloud_disabled_reason)
    base = effective_ollama_url()
    if not ollama_host_allowed(base):
        raise OllamaVisionError("OLLAMA_URL host is not local or ollama.com")
    if is_cloud_url(base) and not (settings.ollama_api_key or "").strip():
        raise OllamaVisionError("OLLAMA_API_KEY is required for Ollama Cloud")
    chosen = model or resolve_vision_model()
    if is_cloud_url(base) and any(chosen.startswith(prefix) for prefix in RETIRED_CLOUD_PREFIXES):
        chosen = CLOUD_DEFAULT_MODEL
    image_b64 = encode_jpeg_b64(bgr)
    own = client is None
    http = client or httpx.Client(timeout=settings.ollama_vision_timeout_seconds)
    try:
        if is_cloud_url(base):
            payload, chosen = _cloud_chat_fallback(http, base, chosen, image_b64, PLATE_PROMPT)
        else:
            payload = _local_generate(http, base, chosen, image_b64, PLATE_PROMPT)
    except httpx.TimeoutException as exc:
        raise OllamaVisionError("ollama vision timeout") from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        if status in {401, 403}:
            _disable_cloud(f"Ollama Cloud HTTP {status}; check OLLAMA_API_KEY")
        raise OllamaVisionError(f"ollama vision HTTP error: {exc}") from exc
    except httpx.HTTPError as exc:
        raise OllamaVisionError(f"ollama vision HTTP error: {exc}") from exc
    finally:
        if own:
            http.close()
    text = _response_text(payload)
    plate_raw, confidence = parse_vision_text(text)
    plate_norm = normalize(plate_raw)
    digest = str(payload.get("model") or chosen)
    return VisionRead(
        plate_raw=plate_raw,
        plate_norm=plate_norm,
        confidence=confidence,
        model_id=f"ollama:{chosen}",
        model_hash=digest[:64],
        raw_text=text[:500],
    )


def infer_vehicle(
    bgr: np.ndarray,
    *,
    camera_id: str = "",
    client: httpx.Client | None = None,
    model: str | None = None,
) -> dict:
    """Vision-first vehicle record. Empty plate_norm means do not persist."""
    gap = float(getattr(settings, "ollama_live_interval_seconds", 0.4) or 0.4)
    now = time.monotonic()
    if camera_id and now - _last_vehicle_infer.get(camera_id, 0.0) < gap:
        return {"plate_norm": "", "skipped": "throttled"}
    wait = float(getattr(settings, "ollama_lock_wait_seconds", 25.0) or 25.0)
    if not _vehicle_lock.acquire(timeout=max(1.0, wait)):
        return {"plate_norm": "", "skipped": "busy"}
    try:
        if camera_id:
            _last_vehicle_infer[camera_id] = now
        if not settings.ollama_vision_enabled:
            raise OllamaVisionError("Ollama vision is disabled")
        if _cloud_disabled_reason:
            raise OllamaVisionError(_cloud_disabled_reason)
        base = effective_ollama_url()
        if not ollama_host_allowed(base):
            raise OllamaVisionError("OLLAMA_URL host is not local or ollama.com")
        if is_cloud_url(base) and not (settings.ollama_api_key or "").strip():
            raise OllamaVisionError("OLLAMA_API_KEY is required for Ollama Cloud")
        chosen = model or resolve_vision_model()
        if is_cloud_url(base) and any(chosen.startswith(prefix) for prefix in RETIRED_CLOUD_PREFIXES):
            chosen = CLOUD_DEFAULT_MODEL
        image_b64 = encode_jpeg_b64(bgr, max_width=1024)
        own = client is None
        http = client or httpx.Client(timeout=min(25.0, settings.ollama_vision_timeout_seconds))
        try:
            if is_cloud_url(base):
                payload, chosen = _cloud_chat_fallback(http, base, chosen, image_b64, VEHICLE_PROMPT)
            else:
                payload = _local_generate(http, base, chosen, image_b64, VEHICLE_PROMPT)
        finally:
            if own:
                http.close()
        parsed = parse_vehicle_payload(_response_text(payload))
        parsed["model_id"] = f"ollama:{chosen}"
        parsed["model_hash"] = str(payload.get("model") or chosen)[:64]
        parsed["provider"] = "ollama_vision"
        return parsed
    except httpx.TimeoutException as exc:
        raise OllamaVisionError("ollama vision timeout") from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        if status in {401, 403}:
            _disable_cloud(f"Ollama Cloud HTTP {status}; check OLLAMA_API_KEY")
        raise OllamaVisionError(f"ollama vision HTTP error: {exc}") from exc
    except httpx.HTTPError as exc:
        raise OllamaVisionError(f"ollama vision HTTP error: {exc}") from exc
    finally:
        _vehicle_lock.release()


def should_use_vision(camera, tess_syntax: bool) -> bool:
    if not settings.ollama_vision_enabled:
        return False
    if _cloud_disabled_reason:
        return False
    if tess_syntax:
        return False
    return True


def _status_label(out: dict) -> str:
    if not out.get("enabled"):
        return "Ollama off"
    model = out.get("resolved_model") or out.get("configured_model") or "vision"
    if out.get("cloud"):
        if not out.get("api_key_configured"):
            return "Ollama Cloud · key missing"
        if out.get("reachable"):
            return f"Ollama Cloud · live · {model}"
        err = out.get("error") or "unreachable"
        return f"Ollama Cloud · not live · {err}"
    if out.get("reachable"):
        return f"Ollama local · live · {model}"
    return "Ollama local · not running"


def vision_status() -> dict:
    enabled = bool(settings.ollama_vision_enabled)
    base = effective_ollama_url()
    out = {
        "enabled": enabled,
        "url": base,
        "cloud": is_cloud_url(base),
        "api_key_configured": bool((settings.ollama_api_key or "").strip()),
        "configured_model": settings.ollama_vision_model,
        "resolved_model": "",
        "reachable": False,
        "live": False,
        "available_vision": [],
        "error": "",
        "label": "Ollama off",
    }
    if not enabled:
        out["label"] = _status_label(out)
        return out
    try:
        names = list_ollama_models()
        out["reachable"] = True
        out["live"] = True
        out["available_vision"] = [
            n
            for n in names
            if n in VISION_CANDIDATES or n.startswith(("llava", "gemma3", "gemma4", "qwen3-vl", "moondream"))
        ]
        out["resolved_model"] = resolve_vision_model(names)
    except Exception as exc:
        out["error"] = redact_secrets(str(exc))
        try:
            out["resolved_model"] = resolve_vision_model()
        except Exception:
            pass
    out["label"] = _status_label(out)
    return out
