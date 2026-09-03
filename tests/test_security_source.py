from pathlib import Path

from app.security import redact_secrets

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"


def _py_files() -> list[Path]:
    return list(APP.rglob("*.py"))


def test_redact_secrets_strips_token(monkeypatch):
    monkeypatch.setattr("app.security.settings.cctv_access_token", "unit-secret-token")
    assert "unit-secret-token" not in redact_secrets("failed unit-secret-token for catalogue")


def test_no_government_download_or_gateway_publish():
    blob = "\n".join(p.read_text(encoding="utf-8") for p in _py_files())
    for needle in (
        "download_footage",
        "save_stream_to_disk",
        "publish_stream",
        "gateway/publish",
        "call_control_api",
    ):
        assert needle not in blob
    cat = (APP / "services" / "catalogue.py").read_text(encoding="utf-8")
    assert "http.get" in cat or "httpx" in cat
    # Optional same-origin form POST is login-only. No camera-control or publish.
    assert cat.count("http.post") >= 1
    assert "GET_ONLY" in cat
    assert "rtsp://{" not in cat
    assert "DOCUMENTED_RTSP_PREFIX" in cat
    blob_js = (APP / "static" / "app.js").read_text(encoding="utf-8")
    assert "CCTV_ACCESS_TOKEN" not in blob_js


def test_configured_token_not_embedded_in_source():
    from app.config import settings

    token = (settings.cctv_access_token or "").strip()
    if len(token) < 8:
        return
    for path in list(APP.rglob("*.py")) + list(APP.rglob("*.js")) + list(APP.rglob("*.html")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert token not in text, f"token leaked in {path}"


def test_no_google_directions_client():
    for path in (APP / "services").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "maps.googleapis.com" not in text
        assert "maps.google.com" not in text
    js = (APP / "static" / "app.js").read_text(encoding="utf-8")
    assert "googleapis.com" not in js
    assert "google_maps" not in js


def test_protected_rtsp_not_in_public_serializer():
    text = (APP / "services" / "serialize.py").read_text(encoding="utf-8")
    assert "protected_rtsp_url_or_reference" not in text or "redact" in text
    assert "source_uri_redacted" in text
