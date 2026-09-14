"""The developer labelling queue, and the production invariants it must not break.

The queue exists because the production search cannot find rows worth labelling:
while the type deployment gate is closed every row carries vehicle_type="unknown"
in the column, and the colour filter deliberately hides values from retired
sources. Both are correct for an investigator and useless for building training
data, so the raw matching lives on a separate developer-gated route.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi.testclient import TestClient

from app.config import settings
from app.database import get_db, init_db, make_engine, make_session_factory
from app.main import app
from app.models import VehicleObservation
from app.services.vehicle_observations import (
    observation_json,
    review_observation,
    search_observations,
    upsert_observation,
)
from tests.conftest import add_camera

MODEL_SOURCE = "openvino:vehicle-attributes-recognition-barrier-0042"


@contextmanager
def developer_ui():
    app_env, enabled = settings.app_env, settings.enable_developer_ui
    settings.app_env, settings.enable_developer_ui = "development", True
    try:
        yield
    finally:
        settings.app_env, settings.enable_developer_ui = app_env, enabled


def _row(db, camera, track_id, **fields) -> VehicleObservation:
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    crop = np.full((60, 120, 3), 220, dtype=np.uint8)
    row, _ = upsert_observation(
        db, camera, run_id="run", track_id=track_id, frame_index=1, source_pts_ms=1.0,
        box=(10, 10, 80, 40), frame_shape=frame.shape[:2], frame=frame, crop=crop,
        detector="yolov8n", detector_confidence=0.9, vehicle_type="car",
    )
    meta = dict(row.metadata_json or {})
    if "attributes" in fields:
        meta["attributes"] = fields.pop("attributes")
        row.metadata_json = meta
    for key, value in fields.items():
        setattr(row, key, value)
    db.commit()
    return row


def _window():
    now = datetime.now(timezone.utc)
    return {"start": now - timedelta(hours=1), "end": now + timedelta(hours=1)}


def test_raw_type_filter_finds_the_gated_candidate_the_column_hides(db):
    """The whole reason this surface exists.

    apply_attributes() writes vehicle_type="unknown" on every gated row, so the
    model's answer survives only in metadata_json. A queue filtering the column
    would return nothing and look broken.
    """
    camera = add_camera(db)
    _row(db, camera, "gated", vehicle_type="unknown", type_source="suppressed_pending_validation",
         attributes={"type_candidate": "truck", "type_candidate_confidence": 0.83})

    production = search_observations(db, **_window(), vehicle_type="truck")
    assert production["total"] == 0          # correct for an investigator

    queue = search_observations(db, **_window(), vehicle_type="truck", raw_attribute_filters=True)
    assert [r["track_id"] for r in queue["observations"]] == ["gated"]
    assert queue["observations"][0]["vehicle_type"] == "unknown"        # column untouched
    assert queue["observations"][0]["type_candidate"] == "truck"        # what the model said


def test_raw_colour_filter_reaches_the_sources_production_hides(db):
    camera = add_camera(db)
    _row(db, camera, "legacy", vehicle_color="red", color_source="legacy_sighting")
    _row(db, camera, "hsv", vehicle_color="red", color_source="opencv_hsv")
    _row(db, camera, "model", vehicle_color="red", color_source=MODEL_SOURCE)

    production = search_observations(db, **_window(), vehicle_color="red")
    assert [r["track_id"] for r in production["observations"]] == ["model"]

    queue = search_observations(db, **_window(), vehicle_color="red", raw_attribute_filters=True)
    assert sorted(r["track_id"] for r in queue["observations"]) == ["hsv", "legacy", "model"]
    assert any("Developer labelling mode" in w for w in queue["search_warnings"])


def test_source_and_confidence_bands_narrow_the_queue(db):
    camera = add_camera(db)
    _row(db, camera, "low", color_source=MODEL_SOURCE, color_confidence=0.30,
         attributes={"type_candidate": "car", "type_candidate_confidence": 0.20})
    _row(db, camera, "high", color_source=MODEL_SOURCE, color_confidence=0.95,
         attributes={"type_candidate": "car", "type_candidate_confidence": 0.95})
    _row(db, camera, "legacy", color_source="legacy_sighting", color_confidence=0.95)

    by_source = search_observations(db, **_window(), raw_attribute_filters=True, color_source="legacy")
    assert [r["track_id"] for r in by_source["observations"]] == ["legacy"]

    # Type confidence reads JSON, colour confidence reads the column. The
    # asymmetry is deliberate -- the gate zeroes the type column.
    by_type_conf = search_observations(db, **_window(), raw_attribute_filters=True,
                                       min_type_confidence=0.9)
    assert [r["track_id"] for r in by_type_conf["observations"]] == ["high"]
    by_colour_conf = search_observations(db, **_window(), raw_attribute_filters=True,
                                         max_color_confidence=0.5)
    assert [r["track_id"] for r in by_colour_conf["observations"]] == ["low"]


def test_review_context_is_recorded_on_the_verdict(db):
    """A label taken with the model on screen must stay distinguishable from a
    blind one, or the whole corpus becomes unusable for evaluation."""
    camera = add_camera(db)
    row = _row(db, camera, "ctx")
    review_observation(db, row.id, actor="operator", vehicle_type="bus",
                       note="annotator:amit",
                       context={"ui": "bulk-v1", "prediction_visible": True})
    db.commit()
    entry = row.metadata_json["review_history"][-1]
    assert entry["context"]["prediction_visible"] is True
    assert entry["note"] == "annotator:amit"
    assert observation_json(row)["type_display"] == "Bus"


def _client(engine):
    Session = make_session_factory(engine)

    def override():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override
    return TestClient(app), Session


def test_label_queue_route_is_developer_only_and_skips_the_frozen_eval_set():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    client, _Session = _client(engine)
    now = datetime.now(timezone.utc)
    params = {"start": (now - timedelta(hours=1)).isoformat(),
              "end": (now + timedelta(hours=1)).isoformat()}
    headers = {"Authorization": "Bearer p0-operator"}
    with client:
        assert client.get("/api/dev/label-queue", params=params, headers=headers).status_code == 404
        with developer_ui():
            response = client.get("/api/dev/label-queue", params=params, headers=headers)
            assert response.status_code == 200
            body = response.json()
            assert body["raw_filters"] is True
            # The 59 blind-labelled tracks the published accuracy was measured
            # on are never served for relabelling.
            assert body["frozen_eval_excluded"] >= 0
            from app.main import frozen_eval_track_ids
            frozen = frozen_eval_track_ids()
            assert not [r for r in body["observations"] if r["track_id"] in frozen]
    app.dependency_overrides.clear()


def test_production_route_ignores_the_developer_filter_parameters():
    """Not merely unused in production -- unreachable. The raw filters live on a
    route that 404s, so no query string can switch them on."""
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    client, Session = _client(engine)
    with client:
        db = Session()
        camera = add_camera(db, id="CAM-F")
        _row(db, camera, "legacy-red", vehicle_color="red", color_source="legacy_sighting")
        db.close()
        now = datetime.now(timezone.utc)
        params = {"start": (now - timedelta(hours=1)).isoformat(),
                  "end": (now + timedelta(hours=1)).isoformat(), "vehicle_color": "red"}
        assert client.get("/api/investigations/vehicles", params=params).json()["total"] == 0
        assert client.get("/api/investigations/vehicles",
                          params={**params, "raw_attribute_filters": "true"}).json()["total"] == 0
    app.dependency_overrides.clear()
