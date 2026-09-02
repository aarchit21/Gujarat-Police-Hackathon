"""Optional remote GPU inference. Production never invents a plate response."""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.config import settings
from app.security import assert_http_url_allowed, redact_url


class RemoteInferenceError(RuntimeError):
    pass


@dataclass
class RemoteRead:
    plate_raw: str
    confidence: float
    model_id: str
    model_hash: str
    box: tuple[int, int, int, int] | None
    raw_payload: dict


def infer_jpeg(
    jpeg_bytes: bytes,
    *,
    camera_id: str,
    timeout_seconds: float | None = None,
    url: str | None = None,
    token: str | None = None,
    allowed_hosts: set[str] | None = None,
    client: httpx.Client | None = None,
) -> RemoteRead:
    target = (url if url is not None else settings.remote_inference_url).strip()
    if not target:
        raise RemoteInferenceError("REMOTE_INFERENCE_URL is not configured")
    try:
        assert_http_url_allowed(target, allowed_hosts)
    except ValueError as exc:
        raise RemoteInferenceError(str(exc)) from exc

    headers = {}
    auth = token if token is not None else settings.remote_inference_token
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    timeout = timeout_seconds if timeout_seconds is not None else settings.remote_inference_timeout_seconds

    own_client = client is None
    http = client or httpx.Client(timeout=timeout, follow_redirects=False)
    try:
        response = http.post(
            target,
            content=jpeg_bytes,
            headers={**headers, "Content-Type": "image/jpeg", "X-Camera-Id": camera_id},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.TimeoutException as exc:
        raise RemoteInferenceError(f"remote inference timeout contacting {redact_url(target)}") from exc
    except httpx.HTTPError as exc:
        raise RemoteInferenceError(f"remote inference HTTP error: {exc}") from exc
    finally:
        if own_client:
            http.close()

    if not isinstance(payload, dict):
        raise RemoteInferenceError("remote inference returned a non-object JSON body")
    plate = str(payload.get("plate_text") or payload.get("plate_raw") or "").strip()
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    model_id = str(payload.get("model_id") or "remote-unspecified").strip()
    model_hash = str(payload.get("model_hash") or "unpinned").strip()
    box = _bbox(payload.get("bbox"))
    return RemoteRead(plate, confidence, model_id, model_hash, box, payload)


def _bbox(value) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    if isinstance(value, dict):
        try:
            return int(value["x"]), int(value["y"]), int(value["w"]), int(value["h"])
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return int(value[0]), int(value[1]), int(value[2]), int(value[3])
        except (TypeError, ValueError):
            return None
    return None
