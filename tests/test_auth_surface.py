"""The authentication boundary, asserted route by route.

Before this existed, 26 routes answered anonymously -- vehicle search, plate
history, alerts, the watchlist, the camera inventory with coordinates, the audit
log -- while the UI showed a sign-in screen. The screen was a client-side
overlay; the server never knew a session existed.

The test that matters here is `test_every_route_is_either_allowlisted_or_closed`.
It enumerates the live routing table rather than a hand-written list, so a route
added next month is covered the day it is written: either its path is added to
the allowlist in app/main.py deliberately, or this fails.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.database import get_db, init_db, make_engine, make_session_factory
from app.main import PUBLIC_PATHS, PUBLIC_PREFIXES, app, _is_public
from tests.conftest import operator_headers

# Path parameters get a value that is syntactically valid but matches nothing.
# A 404 from a missing row is fine -- it proves the request got past the gate,
# which is what is under test. A 401 is what we assert against.
PARAM_STUB = {
    "filename": "console.js",
    "camera_id": "CAM-DOES-NOT-EXIST",
    "plate": "GJ01ZZ9999",
    "observation_id": "999999",
    "alert_id": "999999",
    "watchlist_id": "999999",
}


def _client():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    Session = make_session_factory(engine)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    return TestClient(app)


def _concrete_paths():
    """Every routable path, with path parameters filled in."""
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        if not path or path.startswith("/static"):
            continue
        for name, stub in PARAM_STUB.items():
            path = path.replace("{" + name + "}", stub)
        if "{" in path:  # an unmapped parameter -- make it loud, not skipped
            raise AssertionError(f"add a PARAM_STUB entry for {path}")
        for method in sorted(methods - {"HEAD", "OPTIONS"}):
            yield method, path


def test_every_route_is_either_allowlisted_or_closed():
    client = _client()
    leaked = []
    for method, path in _concrete_paths():
        if _is_public(path):
            continue
        response = client.request(method, path)
        if response.status_code != 401:
            leaked.append(f"{method} {path} -> {response.status_code}")
    assert not leaked, (
        "these routes answered without a token:\n  " + "\n  ".join(leaked)
        + "\n\nAdd the path to PUBLIC_PATHS in app/main.py only if it must work "
          "before sign-in. Otherwise it is a leak."
    )


def test_the_public_allowlist_is_exactly_what_we_intend():
    """A guard on the allowlist itself.

    Widening the boundary should be a deliberate act that edits this test, not a
    line quietly added to a set in main.py.
    """
    assert PUBLIC_PATHS == frozenset({
        "/", "/healthz", "/favicon.ico", "/api/ui/config", "/api/vendor/events",
    })
    assert PUBLIC_PREFIXES == ("/static/", "/dev", "/docs", "/redoc", "/openapi.json")


def test_the_sign_in_page_and_its_assets_load_without_a_token():
    client = _client()
    assert client.get("/").status_code == 200
    assert client.get("/api/ui/config").status_code == 200
    assert client.get("/healthz").json() == {"ok": True}


def test_healthz_says_nothing_about_the_deployment():
    """/api/health reports models, database type and the catalogue host.

    A hosting platform's liveness probe needs none of it, so the probe endpoint
    is a separate route that cannot grow those fields by accident.
    """
    client = _client()
    assert client.get("/healthz").json() == {"ok": True}
    assert client.get("/api/health").status_code == 401


def test_a_wrong_token_is_rejected():
    client = _client()
    assert client.get("/api/ui/overview", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/ui/overview", headers=operator_headers()).status_code == 200


def test_query_string_token_is_refused_in_production(monkeypatch):
    """`?token=` puts the credential in history, proxy logs and Referer.

    The developer console needs it -- `<img src>` and `window.open` cannot send a
    header -- so it is allowed exactly where /dev is served, and nowhere else.
    """
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "enable_developer_ui", False)
    client = _client()
    assert client.get(f"/api/ui/overview?token={settings.admin_token}").status_code == 401

    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "enable_developer_ui", True)
    assert client.get(f"/api/ui/overview?token={settings.admin_token}").status_code == 200


def test_authentication_cannot_be_switched_off_by_configuration():
    """`require_auth=False` used to make a tokenless request succeed.

    The setting is gone. This asserts it cannot come back by accident: there is
    no attribute to set, and setting one has no effect.
    """
    assert not hasattr(settings, "require_auth")
    client = _client()
    settings.__dict__["require_auth"] = False  # the old escape hatch
    try:
        assert client.get("/api/ui/overview").status_code == 401
    finally:
        settings.__dict__.pop("require_auth", None)


@pytest.mark.parametrize("path", [
    "/api/investigations/vehicles",
    "/api/vehicles/GJ01AB1234",
    "/api/alerts",
    "/api/watchlist",
    "/api/cameras",
    "/api/audit",
    "/api/sightings",
    "/api/observed-plates",
    "/api/health",
])
def test_the_routes_that_used_to_be_open(path):
    """Named individually because each was a specific disclosure.

    Kept alongside the generic sweep: if someone allowlists one of these, the
    sweep goes quiet but this does not.
    """
    assert _client().get(path).status_code == 401
