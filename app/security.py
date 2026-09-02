"""URL allowlists, redaction, and path checks. Never log secrets."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse, urlunparse

from app.config import settings


PROTECTED_MEDIA_HOST_SUFFIXES = ("corp8.cloud",)


def redact_url(url: str | None) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    if parsed.username or parsed.password:
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        netloc = f"***:***@{host}"
        return urlunparse(parsed._replace(netloc=netloc))
    return raw


def redact_secrets(text: str | None) -> str:
    raw = str(text or "")
    token = (settings.cctv_access_token or "").strip()
    if token and token in raw:
        raw = raw.replace(token, "***")
    user = (settings.cctv_access_username or "").strip()
    if user and len(user) > 1 and user in raw:
        raw = raw.replace(user, "***")
    return raw


def hls_requires_server_credential(url: str | None) -> bool:
    parsed = urlparse(url or "")
    if parsed.username or parsed.password:
        return True
    host = (parsed.hostname or "").lower()
    return any(host == suffix or host.endswith("." + suffix) for suffix in PROTECTED_MEDIA_HOST_SUFFIXES)


def assert_http_url_allowed(url: str, allowed_hosts: set[str] | None = None) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("remote inference URL must be http or https")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("remote inference URL missing host")
    allowed = allowed_hosts if allowed_hosts is not None else settings.remote_allowed_hosts()
    if not allowed:
        raise ValueError("remote inference host allowlist is empty")
    if host not in allowed:
        raise ValueError(f"remote inference host {host} is not in the allowlist")
    return host


def evidence_relpath_is_safe(rel: str, root: Path | None = None) -> Path:
    base = (root or settings.evidence_dir).resolve()
    cleaned = rel.replace("\\", "/").lstrip("/")
    if cleaned.startswith("data/"):
        cleaned = cleaned[len("data/") :]
    if cleaned.startswith("evidence/"):
        cleaned = cleaned[len("evidence/") :]
    path = (base / cleaned).resolve()
    if base not in path.parents and path != base:
        raise ValueError("evidence path escapes evidence directory")
    return path
