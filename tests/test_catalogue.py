import httpx
from sqlalchemy import select

from app.config import settings
from app.models import Camera, Sighting
from app.services.catalogue import (
    DEFAULT_CATALOGUE_URL,
    catalogue_get,
    parse_catalogue,
    sync_catalogue,
    upsert_from_catalogue,
)
from app.services.pipeline import persist_sighting
from tests.conftest import add_camera, add_watchlist


def test_default_catalogue_url_is_cameras_json():
    assert DEFAULT_CATALOGUE_URL.endswith("/cameras.json")
    assert settings.ingest_catalogue_url.endswith("/cameras.json")


def test_catalogue_uses_provided_urls_never_constructs_from_id(db):
    item = {
        "catalogue_camera_id": "portal-42",
        "name": "Junction",
        "live": True,
        "codec": "H265",
        "rtsp_url": "rtsp://user:pass@gateway/stream/abc",
        "whep_url": "https://gateway/whep/abc",
        "hls_url": "https://gateway/hls/abc.m3u8",
        "lat": 23.0,
        "lng": 72.5,
    }
    cam = upsert_from_catalogue(db, item)
    db.commit()
    assert cam.source_uri == "rtsp://user:pass@gateway/stream/abc"
    assert cam.hls_url.endswith(".m3u8")
    assert "portal-42" not in (cam.hls_url or "") or cam.hls_url == item["hls_url"]
    assert cam.analytics_active is False
    assert cam.catalogue_live is True
    assert cam.decode_status == "untested"


def test_documented_urls_fill_only_when_catalogue_omits_them(db):
    from app.services.catalogue import apply_documented_stream_contract

    item = apply_documented_stream_contract({"catalogue_camera_id": "cam01", "rtsp_url": "", "whep_url": "", "hls_url": ""})
    assert item["rtsp_url"].endswith("/stream/cam01")
    assert item["rtsp_url_source"] == "documented_contract"
    provided = apply_documented_stream_contract(
        {"catalogue_camera_id": "cam01", "rtsp_url": "rtsp://custom/keep", "whep_url": "http://w", "hls_url": "https://h"}
    )
    assert provided["rtsp_url"] == "rtsp://custom/keep"
    assert provided["rtsp_url_source"] == "catalogue"
    cam = upsert_from_catalogue(db, {"catalogue_camera_id": "only-id", "live": True})
    # upsert path without parse_catalogue does not invent IDs; URLs stay empty unless provided
    assert cam.analytics_active is False


def test_catalogue_live_does_not_imply_analytics_active(db):
    payload = {"cameras": [{"id": "g1", "live": True, "rtsp_url": "rtsp://gateway/a"}]}
    out = sync_catalogue(db, url="https://cctv.corp8.cloud/cameras.json", fetcher=lambda _url: payload)
    assert out["ok"] is True
    assert out["analytics_active_not_implied_by_live"] is True
    assert out["hardcoded_50"] is False
    cam = db.get(Camera, "g1")
    assert cam.catalogue_live is True
    assert cam.analytics_active is False


def test_actual_catalogue_count_not_hardcoded_50(db):
    payload = {"cameras": [{"id": f"cam{i:02d}", "live": True, "rtsp_url": f"rtsp://gateway/{i}"} for i in range(1, 4)]}
    out = sync_catalogue(db, url="https://cctv.corp8.cloud/cameras.json", fetcher=lambda _url: payload)
    assert out["cameras"] == 3
    from app.services.coverage import coverage

    cov = coverage(db)
    assert cov["government_catalogue_count"] == 3
    assert cov["hardcoded_50"] is False


def test_missing_catalogue_cameras_retain_history(db):
    first = {"cameras": [{"id": "keep-me", "live": True, "rtsp_url": "rtsp://gateway/a", "location": "Surat"}]}
    sync_catalogue(db, url="https://cctv.corp8.cloud/cameras.json", fetcher=lambda _url: first)
    cam = db.get(Camera, "keep-me")
    add_watchlist(db)
    persist_sighting(
        db,
        cam,
        plate_raw="GJ01AB1234",
        plate_norm="GJ01AB1234",
        plate_voted="GJ01AB1234",
        syntax=True,
        confidence=0.5,
        model_id="tesseract-opencv-p0",
        model_hash="x",
        evidence_path="",
        run_id="r",
        frame_index=0,
        passage_id="p",
        source_pts_ms=1.0,
        provider="local",
    )
    db.commit()
    second = {"cameras": [{"id": "other", "live": False}]}
    sync_catalogue(db, url="https://cctv.corp8.cloud/cameras.json", fetcher=lambda _url: second)
    cam = db.get(Camera, "keep-me")
    assert cam is not None
    assert cam.catalogue_live is False
    assert "history retained" in cam.status_reason
    assert cam.analytics_active is False
    assert db.scalar(select(Sighting).where(Sighting.camera_id == "keep-me")) is not None


def test_parse_catalogue_list():
    items = parse_catalogue([{"id": "a", "rtspUrl": "rtsp://x"}])
    assert items[0]["rtsp_url"] == "rtsp://x"
    assert items[0]["live"] is True


def test_id_name_catalogue_is_live_and_not_stacked(db):
    from app.services.catalogue import backfill_catalogue_display

    payload = {"cameras": [{"id": "cam01", "name": "One"}, {"id": "cam02", "name": "Two"}]}
    sync_catalogue(db, url="https://cctv.corp8.cloud/cameras.json", fetcher=lambda _url: payload)
    a = db.get(Camera, "cam01")
    b = db.get(Camera, "cam02")
    assert a.catalogue_live is True
    assert b.catalogue_live is True
    assert (a.lat, a.lng) != (b.lat, b.lng)
    assert a.coords_source == "placeholder"
    assert b.coords_source == "placeholder"
    stacked = add_camera(db, id="cam99", catalogue_camera_id="cam99", lat=22.3, lng=71.2, catalogue_live=False)
    backfill_catalogue_display(db)
    db.commit()
    stacked = db.get(Camera, "cam99")
    assert stacked.catalogue_live is True
    assert not (stacked.lat == 22.3 and stacked.lng == 71.2)


def test_explicit_live_false_stays_false(db):
    payload = {"cameras": [{"id": "cam-off", "live": False}]}
    sync_catalogue(db, url="https://cctv.corp8.cloud/cameras.json", fetcher=lambda _url: payload)
    assert db.get(Camera, "cam-off").catalogue_live is False


def test_auth_none(monkeypatch):
    seen = {}

    def handler(request: httpx.Request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"cameras": [{"id": "cam01"}]})

    monkeypatch.setattr("app.services.catalogue.settings.cctv_auth_mode", "none")
    monkeypatch.setattr("app.services.catalogue.settings.cctv_access_token", "")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payload = catalogue_get("https://cctv.corp8.cloud/cameras.json", client=client)
    assert payload["cameras"][0]["id"] == "cam01"
    assert seen["auth"] is None


def test_auth_bearer(monkeypatch):
    def handler(request: httpx.Request):
        assert request.headers.get("authorization") == "Bearer unit-token"
        return httpx.Response(200, json={"cameras": [{"id": "cam01"}]})

    monkeypatch.setattr("app.services.catalogue.settings.cctv_auth_mode", "bearer")
    monkeypatch.setattr("app.services.catalogue.settings.cctv_access_token", "unit-token")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payload = catalogue_get("https://cctv.corp8.cloud/cameras.json", client=client)
    assert payload["cameras"][0]["id"] == "cam01"


def test_auth_basic(monkeypatch):
    def handler(request: httpx.Request):
        assert request.headers.get("authorization", "").startswith("Basic ")
        return httpx.Response(200, json={"cameras": [{"id": "cam02"}]})

    monkeypatch.setattr("app.services.catalogue.settings.cctv_auth_mode", "basic")
    monkeypatch.setattr("app.services.catalogue.settings.cctv_access_username", "unit-user")
    monkeypatch.setattr("app.services.catalogue.settings.cctv_access_token", "unit-pass")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payload = catalogue_get("https://cctv.corp8.cloud/cameras.json", client=client)
    assert payload["cameras"][0]["id"] == "cam02"


def test_auth_custom_header(monkeypatch):
    def handler(request: httpx.Request):
        assert request.headers.get("x-api-key") == "unit-header-token"
        return httpx.Response(200, json={"cameras": [{"id": "cam03"}]})

    monkeypatch.setattr("app.services.catalogue.settings.cctv_auth_mode", "custom_header")
    monkeypatch.setattr("app.services.catalogue.settings.cctv_auth_header_name", "X-Api-Key")
    monkeypatch.setattr("app.services.catalogue.settings.cctv_access_token", "unit-header-token")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payload = catalogue_get("https://cctv.corp8.cloud/cameras.json", client=client)
    assert payload["cameras"][0]["id"] == "cam03"


def test_html_login_response_rejected(monkeypatch):
    def handler(_request: httpx.Request):
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<!DOCTYPE html><html>login</html>")

    monkeypatch.setattr("app.services.catalogue.settings.cctv_auth_mode", "none")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        try:
            catalogue_get("https://cctv.corp8.cloud/cameras.json", client=client)
        except ValueError as exc:
            assert "HTML" in str(exc)
        else:
            raise AssertionError("HTML catalogue was accepted")


def test_http_401_rejected(monkeypatch):
    def handler(_request: httpx.Request):
        return httpx.Response(401, text="unauthorized")

    monkeypatch.setattr("app.services.catalogue.settings.cctv_auth_mode", "bearer")
    monkeypatch.setattr("app.services.catalogue.settings.cctv_access_token", "unit-token")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        try:
            catalogue_get("https://cctv.corp8.cloud/cameras.json", client=client)
        except ValueError as exc:
            assert "401" in str(exc)
            assert "unit-token" not in str(exc)
        else:
            raise AssertionError("401 was accepted")


def test_catalogue_form_auth_logs_in_then_uses_session_cookie(monkeypatch):
    calls = []

    def handler(request: httpx.Request):
        calls.append((request.method, str(request.url), request.content))
        if request.url.path == "/auth/login":
            if request.method == "GET":
                return httpx.Response(200, text="<html><form><input name='email'><input name='password'></form></html>")
            assert request.method == "POST"
            body = (request.content or b"").decode()
            assert "email=" in body
            assert "password=" in body
            return httpx.Response(302, headers={"set-cookie": "session=ok; Path=/", "location": "/resource"})
        if request.url.path == "/resource":
            return httpx.Response(200, text="ok")
        assert request.headers.get("cookie") == "session=ok"
        return httpx.Response(200, json={"cameras": [{"id": "cam01"}]})

    monkeypatch.setattr("app.services.catalogue.settings.cctv_auth_mode", "form")
    monkeypatch.setattr("app.services.catalogue.settings.cctv_access_username", "ops@example.com")
    monkeypatch.setattr("app.services.catalogue.settings.cctv_access_token", "secret")
    monkeypatch.setattr("app.services.catalogue.settings.cctv_login_url", "")
    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        payload = catalogue_get("https://cctv.test/cameras.json", client=client)
    assert payload["cameras"][0]["id"] == "cam01"
    assert ("POST", "https://cctv.test/auth/login") in [c[0:2] for c in calls]


def test_catalogue_form_auth_requires_email(monkeypatch):
    monkeypatch.setattr("app.services.catalogue.settings.cctv_auth_mode", "form")
    monkeypatch.setattr("app.services.catalogue.settings.cctv_access_username", "")
    monkeypatch.setattr("app.services.catalogue.settings.cctv_access_token", "secret")
    with httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(500))) as client:
        try:
            catalogue_get("https://cctv.test/cameras.json", client=client)
        except ValueError as exc:
            assert "USERNAME" in str(exc) or "email" in str(exc).lower()
        else:
            raise AssertionError("form login without email was accepted")


def test_catalogue_form_auth_rejects_cross_origin_login(monkeypatch):
    monkeypatch.setattr("app.services.catalogue.settings.cctv_auth_mode", "form")
    monkeypatch.setattr("app.services.catalogue.settings.cctv_access_username", "ops@example.com")
    monkeypatch.setattr("app.services.catalogue.settings.cctv_access_token", "secret")
    monkeypatch.setattr("app.services.catalogue.settings.cctv_login_url", "https://evil.test/auth/login")
    with httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(500))) as client:
        try:
            catalogue_get("https://cctv.test/cameras.json", client=client)
        except ValueError as exc:
            assert "same origin" in str(exc)
        else:
            raise AssertionError("cross-origin credential POST was not rejected")
