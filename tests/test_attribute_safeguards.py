"""Safeguards around unreliable automatic attributes.

Measured on this deployment: vehicle type reached 50% selective precision at
64% coverage on 28 blind-labelled tracks, and precision FELL as the confidence
floor rose. These tests pin the consequences of that measurement so a later
change cannot quietly start presenting type as a result again.
"""
from __future__ import annotations

import numpy as np
from sqlalchemy import select

from app.config import settings
from app.models import VehicleObservation
from app.services.track_aggregate import AggregatedAttributes
from app.services.vehicle_attributes import type_deployment_gate
from app.services.vehicle_observations import (
    ATTRIBUTE_DISCLAIMER,
    POC_ACCURACY,
    apply_attributes,
    attribute_states,
    observation_json,
    review_observation,
    search_observations,
    upsert_observation,
)
from tests.conftest import add_camera

MODEL_SOURCE = "openvino:vehicle-attributes-recognition-barrier-0042"


def _aggregated(vehicle_type="car", vehicle_color="white") -> AggregatedAttributes:
    return AggregatedAttributes(
        vehicle_type=vehicle_type, type_confidence=0.97, type_agreement=1.0,
        type_source=MODEL_SOURCE, type_reason="accepted",
        vehicle_color=vehicle_color, color_confidence=0.91, color_agreement=0.88,
        color_source=MODEL_SOURCE, color_reason="accepted",
        observations=3, detector_type=vehicle_type,
        type_probs={"car": 0.97, "van": 0.01, "truck": 0.01, "bus": 0.01},
        color_probs={"white": 0.91, "gray": 0.09},
    )


def _observation(db, camera, track_id="t1") -> VehicleObservation:
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    crop = np.full((60, 120, 3), 230, dtype=np.uint8)
    row, _ = upsert_observation(
        db, camera, run_id="run", track_id=track_id, frame_index=1, source_pts_ms=100.0,
        box=(20, 30, 100, 50), frame_shape=frame.shape[:2], frame=frame, crop=crop,
        detector="yolov8n", detector_confidence=0.9, vehicle_type="car",
    )
    return row


# -- 1/3. Unverified type defaults to unknown ----------------------------


def test_type_deployment_gate_fails_closed():
    passed, reason = type_deployment_gate()
    assert passed is False
    assert "validated" in reason


def test_unverified_type_is_unknown_even_when_the_model_is_confident(db):
    camera = add_camera(db)
    row = _observation(db, camera)
    apply_attributes(db, row.id, _aggregated(vehicle_type="car"))
    db.commit()
    assert row.vehicle_type == "unknown"
    assert row.type_confidence == 0.0
    assert row.type_source == "suppressed_pending_validation"
    assert attribute_states(row)[0] == "unknown"


def test_type_is_exposed_only_when_a_model_clears_the_gate(db, monkeypatch):
    camera = add_camera(db)
    row = _observation(db, camera)
    monkeypatch.setattr(settings, "vattr_type_validated_model_id", MODEL_SOURCE)
    apply_attributes(db, row.id, _aggregated(vehicle_type="car"))
    db.commit()
    assert row.vehicle_type == "car"
    assert row.type_source == MODEL_SOURCE


def test_gate_rejects_a_model_id_that_is_not_the_one_running(db, monkeypatch):
    camera = add_camera(db)
    row = _observation(db, camera)
    monkeypatch.setattr(settings, "vattr_type_validated_model_id", "some-other-model")
    apply_attributes(db, row.id, _aggregated(vehicle_type="car"))
    db.commit()
    assert row.vehicle_type == "unknown"


# -- 2. Raw type prediction stays in metadata ----------------------------


def test_raw_type_candidate_confidence_and_source_are_kept(db):
    camera = add_camera(db)
    row = _observation(db, camera)
    apply_attributes(db, row.id, _aggregated(vehicle_type="truck"))
    db.commit()
    attributes = row.metadata_json["attributes"]
    assert attributes["type_candidate"] == "truck"
    assert attributes["type_candidate_confidence"] == 0.97
    assert attributes["type_model_source"] == MODEL_SOURCE
    assert attributes["type_agreement"] == 1.0
    assert attributes["probabilities"]["type"]["car"] == 0.97
    assert attributes["verified"] is False
    assert attributes["review_required"] is True
    # Suppression must not erase the diagnostic record.
    assert row.vehicle_type == "unknown"


def test_observation_json_exposes_candidate_separately_from_the_result(db):
    camera = add_camera(db)
    row = _observation(db, camera)
    apply_attributes(db, row.id, _aggregated(vehicle_type="bus"))
    db.commit()
    payload = observation_json(row)
    assert payload["vehicle_type"] == "unknown"
    assert payload["type_candidate"] == "bus"
    assert payload["type_suppressed"] is True
    assert payload["type_state"] == "unknown"
    assert payload["review_required"] is True
    assert "not a result" in payload["disclaimer"]


# -- 5/6. Colour is estimated, never verified ----------------------------


def test_colour_is_estimated_and_flagged_for_review(db):
    camera = add_camera(db)
    row = _observation(db, camera)
    apply_attributes(db, row.id, _aggregated(vehicle_color="white"))
    db.commit()
    assert row.vehicle_color == "white"
    assert row.color_source == MODEL_SOURCE
    assert attribute_states(row)[1] == "estimated"
    payload = observation_json(row)
    assert payload["color_state"] == "estimated"
    assert payload["color_verified"] is False
    assert payload["review_required"] is True
    attributes = row.metadata_json["attributes"]
    for key in ("color_candidate", "color_candidate_confidence", "color_agreement", "color_model_source"):
        assert key in attributes


def test_estimated_colour_is_never_marked_verified(db):
    camera = add_camera(db)
    row = _observation(db, camera)
    apply_attributes(db, row.id, _aggregated())
    db.commit()
    assert row.review_status == "unreviewed"
    assert row.verified_vehicle_color == ""
    assert observation_json(row)["color_verified"] is False


def test_colour_can_be_disabled_entirely(db, monkeypatch):
    camera = add_camera(db)
    row = _observation(db, camera)
    monkeypatch.setattr(settings, "vattr_color_enabled", False)
    apply_attributes(db, row.id, _aggregated(vehicle_color="white"))
    db.commit()
    assert row.vehicle_color == "unknown"
    assert row.color_source == "disabled"


# -- 7. Manual review overrides display, keeps raw output ----------------


def test_manual_review_overrides_type_without_losing_the_model_candidate(db):
    camera = add_camera(db)
    row = _observation(db, camera)
    apply_attributes(db, row.id, _aggregated(vehicle_type="truck"))
    db.commit()
    assert row.vehicle_type == "unknown"

    review_observation(db, row.id, actor="officer-1", vehicle_type="bus", note="clearly an AMTS bus")
    db.commit()

    assert row.vehicle_type == "bus"
    assert row.verified_vehicle_type == "bus"
    assert row.type_source == "human_verified"
    assert row.type_confidence == 1.0
    assert row.review_status == "verified"
    assert row.verified_by == "officer-1"
    # The model's wrong answer survives for audit.
    assert row.metadata_json["attributes"]["type_candidate"] == "truck"
    history = row.metadata_json["review_history"]
    assert history[-1]["vehicle_type"] == {"from": "", "to": "bus", "model_candidate": "truck"}
    assert history[-1]["note"] == "clearly an AMTS bus"
    assert attribute_states(row)[0] == "verified"


def test_manual_review_can_correct_colour(db):
    camera = add_camera(db)
    row = _observation(db, camera)
    apply_attributes(db, row.id, _aggregated(vehicle_color="red"))
    db.commit()
    review_observation(db, row.id, actor="officer-2", vehicle_color="white")
    db.commit()
    assert row.vehicle_color == "white"
    assert row.color_source == "human_verified"
    assert attribute_states(row)[1] == "verified"
    assert row.metadata_json["attributes"]["color_candidate"] == "red"


def test_clearing_a_verdict_returns_the_attribute_to_the_automatic_value(db):
    camera = add_camera(db)
    row = _observation(db, camera)
    apply_attributes(db, row.id, _aggregated(vehicle_color="red"))
    db.commit()
    review_observation(db, row.id, actor="officer-2", vehicle_color="white")
    db.commit()
    assert row.vehicle_color == "white"
    review_observation(db, row.id, actor="officer-2", vehicle_color="")
    db.commit()
    assert row.verified_vehicle_color == ""
    assert row.vehicle_color == "red"
    assert row.color_source == MODEL_SOURCE


def test_a_later_frame_does_not_overwrite_a_human_verdict(db):
    camera = add_camera(db)
    row = _observation(db, camera)
    apply_attributes(db, row.id, _aggregated(vehicle_type="truck", vehicle_color="red"))
    db.commit()
    review_observation(db, row.id, actor="officer-3", vehicle_type="bus", vehicle_color="white")
    db.commit()
    # A clearer crop arrives and re-runs aggregation.
    apply_attributes(db, row.id, _aggregated(vehicle_type="car", vehicle_color="gray"))
    db.commit()
    assert row.vehicle_type == "bus"
    assert row.vehicle_color == "white"
    # ...and the newest model candidate is still recorded.
    assert row.metadata_json["attributes"]["type_candidate"] == "car"
    assert row.metadata_json["attributes"]["color_candidate"] == "gray"


# -- 9/10. Search behaviour ----------------------------------------------


def test_type_filter_matches_only_verified_records(db):
    camera = add_camera(db)
    from datetime import datetime, timedelta, timezone

    window = {
        "start": datetime.now(timezone.utc) - timedelta(hours=1),
        "end": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    unverified = _observation(db, camera, track_id="t1")
    apply_attributes(db, unverified.id, _aggregated(vehicle_type="bus"))
    verified = _observation(db, camera, track_id="t2")
    apply_attributes(db, verified.id, _aggregated(vehicle_type="car"))
    review_observation(db, verified.id, actor="officer", vehicle_type="bus")
    db.commit()

    found = search_observations(db, **window, vehicle_type="bus", limit=10)
    assert found["total"] == 1
    assert found["observations"][0]["track_id"] == "t2"
    assert any("HUMAN-VERIFIED" in w for w in found["search_warnings"])


def test_colour_filter_carries_a_miss_and_false_match_warning(db):
    camera = add_camera(db)
    from datetime import datetime, timedelta, timezone

    row = _observation(db, camera)
    apply_attributes(db, row.id, _aggregated(vehicle_color="white"))
    db.commit()
    found = search_observations(
        db,
        start=datetime.now(timezone.utc) - timedelta(hours=1),
        end=datetime.now(timezone.utc) + timedelta(hours=1),
        vehicle_color="white", limit=10,
    )
    warning = " ".join(found["search_warnings"])
    assert "MISS" in warning and "INCLUDE" in warning
    assert "ESTIMATE" in warning


def test_search_without_filters_needs_neither_type_nor_colour(db):
    """Neither attribute may be a mandatory filter."""
    camera = add_camera(db)
    from datetime import datetime, timedelta, timezone

    row = _observation(db, camera)
    apply_attributes(db, row.id, _aggregated())
    db.commit()
    found = search_observations(
        db,
        start=datetime.now(timezone.utc) - timedelta(hours=1),
        end=datetime.now(timezone.utc) + timedelta(hours=1),
        limit=10,
    )
    assert found["total"] == 1
    assert found["search_warnings"] == []


# -- 11/12. Plate isolation and published limitations --------------------


def test_plate_failure_does_not_remove_the_vehicle_observation(db):
    """ANPR is optional; its failure must never delete evidence."""
    camera = add_camera(db)
    row = _observation(db, camera)
    apply_attributes(db, row.id, _aggregated())
    db.commit()
    assert db.scalar(select(VehicleObservation).where(VehicleObservation.id == row.id)) is not None
    # No plate was ever attached, and the record is complete regardless.
    assert row.vehicle_color == "white"
    assert observation_json(row)["evidence_path"]


def test_published_accuracy_matches_the_measured_evaluation():
    assert POC_ACCURACY["vehicle_type"]["selective_precision"] == 0.50
    assert POC_ACCURACY["vehicle_type"]["coverage"] == 0.64
    assert POC_ACCURACY["vehicle_color"]["selective_precision"] == 0.83
    assert POC_ACCURACY["vehicle_color"]["coverage"] == 0.71
    assert POC_ACCURACY["type_usable"] is False
    joined = " ".join(POC_ACCURACY["notes"])
    assert "systematic" in joined
    assert "buses" in joined
    assert "single annotator" in joined


def test_disclaimer_never_calls_type_a_result():
    assert "suppressed" in ATTRIBUTE_DISCLAIMER
    assert "ESTIMATE" in ATTRIBUTE_DISCLAIMER


# -- Production presentation --------------------------------------------
# The console must not show a value it cannot stand behind, whatever a row
# happens to hold. These pin the three demotions: the failed type gate, a
# non-deterministic source, and a class no weight on this host can predict.


def _legacy(db, camera, **fields) -> VehicleObservation:
    row = _observation(db, camera, track_id=fields.pop("track_id", "legacy"))
    for key, value in fields.items():
        setattr(row, key, value)
    db.commit()
    return row


def test_legacy_type_is_not_shown_while_the_deployment_gate_is_closed(db):
    camera = add_camera(db)
    row = _legacy(db, camera, vehicle_type="truck", type_source="legacy_sighting", type_confidence=0.0)
    payload = observation_json(row)
    assert payload["type_display"] == "Unknown"
    assert payload["type_state"] == "unknown"
    assert "not in operational use" in payload["type_note"]
    # The stored value is untouched; only the presentation changed.
    assert row.vehicle_type == "truck"
    assert payload["vehicle_type"] == "truck"


def test_colour_from_a_retired_source_is_not_shown_as_an_estimate(db):
    camera = add_camera(db)
    row = _legacy(db, camera, vehicle_color="yellow", color_source="legacy_sighting")
    payload = observation_json(row)
    assert payload["color_display"] == "Unknown"
    assert payload["color_state"] == "unknown"
    assert "predates" in payload["color_note"]

    row.color_source = "opencv_hsv"
    assert observation_json(row)["color_state"] == "unknown"

    row.color_source = MODEL_SOURCE
    assert observation_json(row)["color_state"] == "estimated"
    assert observation_json(row)["color_display"] == "Yellow"


def test_unsupported_class_is_never_presented_as_a_prediction(db):
    camera = add_camera(db)
    row = _legacy(db, camera, vehicle_color="silver", color_source=MODEL_SOURCE)
    payload = observation_json(row)
    assert payload["color_display"] == "Unknown"
    assert payload["color_state"] == "unknown"
    assert "cannot be reproduced" in payload["color_note"]


def test_a_human_verdict_is_shown_even_for_a_manual_only_class(db):
    camera = add_camera(db)
    row = _observation(db, camera, track_id="verified")
    review_observation(db, row.id, actor="operator", vehicle_type="auto_rickshaw", vehicle_color="silver")
    db.commit()
    payload = observation_json(row)
    assert payload["type_display"] == "Auto rickshaw"
    assert payload["type_state"] == "verified"
    assert payload["color_display"] == "Silver"
    assert payload["color_state"] == "verified"


def test_colour_search_ignores_rows_the_console_refuses_to_show(db):
    from datetime import datetime, timedelta, timezone

    camera = add_camera(db)
    shown = _legacy(db, camera, track_id="shown", vehicle_color="white", color_source=MODEL_SOURCE)
    _legacy(db, camera, track_id="legacy-white", vehicle_color="white", color_source="legacy_sighting")
    now = datetime.now(timezone.utc)
    payload = search_observations(db, start=now - timedelta(hours=1), end=now + timedelta(hours=1),
                                  vehicle_color="white")
    assert [row["id"] for row in payload["observations"]] == [shown.id]


def test_plate_failure_never_removes_the_vehicle_and_says_why(db):
    from app.models import RecognitionAttempt

    camera = add_camera(db)
    row = _observation(db, camera, track_id="noplate")
    db.add(RecognitionAttempt(camera_id=camera.id, track_id=row.track_id,
                              reason_code="insufficient_pixels"))
    db.commit()
    payload = observation_json(row, __import__(
        "app.services.vehicle_observations", fromlist=["plate_status_for"]
    ).plate_status_for(db, [row]).get(row.id))
    assert payload["plate_state"] == "plate_unreadable"
    assert payload["plate_text"] == ""
    assert payload["plate_detail"] == "Plate too small in frame"
    assert payload["evidence_path"]


def test_production_payload_withholds_the_raw_probability_vectors(db):
    camera = add_camera(db)
    row = _observation(db, camera, track_id="probs")
    apply_attributes(db, row.id, _aggregated())
    db.commit()
    assert "metadata" not in observation_json(row)
    # Still recorded, and still readable by the developer console.
    assert row.metadata_json["attributes"]["probabilities"]["color"]["white"] == 0.91
    assert observation_json(row, include_diagnostics=True)["metadata"]["attributes"]["probabilities"]


def test_raw_block_is_developer_only_and_undemoted(db):
    """The developer console needs what the model said; production must not
    receive it. Both halves matter, so both are asserted together."""
    camera = add_camera(db)
    row = _legacy(db, camera, track_id="rawblock", vehicle_color="yellow", color_source=MODEL_SOURCE)
    apply_attributes(db, row.id, _aggregated(vehicle_type="bus", vehicle_color="yellow"))
    db.commit()

    production = observation_json(row)
    assert "raw" not in production
    assert "metadata" not in production
    assert production["type_display"] == "Unknown"

    developer = observation_json(row, include_diagnostics=True)
    # Same row, same request, two different answers -- by design.
    assert developer["type_state"] == "unknown"           # what production shows
    assert developer["raw"]["type_candidate"] == "bus"    # what the model said
    assert developer["raw"]["type_probs"]["car"] == 0.97
    assert developer["raw"]["color_candidate"] == "yellow"
    assert developer["raw"]["vehicle_color"] == "yellow"
    assert developer["raw"]["color_state"] == "estimated"


def test_clearing_a_verdict_never_leaves_a_claim_of_human_verification(db):
    """Withdrawing a verdict must withdraw the claim that a person made it.

    The colour branch of _recompute_operational had no else, so clearing a
    verdict on a row with no model candidate left color_source="human_verified"
    and confidence 1.0 in place. The record then asserted a human confirmation
    that no longer existed -- the precise failure every other safeguard here
    exists to prevent.
    """
    camera = add_camera(db)
    row = _observation(db, camera, track_id="cleared")
    # A legacy row: a human verdict, and no model candidate to fall back to.
    row.metadata_json = {"quality_score": 1.0}
    review_observation(db, row.id, actor="operator", vehicle_color="black")
    db.commit()
    assert row.color_source == "human_verified" and row.color_confidence == 1.0

    review_observation(db, row.id, actor="operator", vehicle_color="")
    db.commit()
    assert row.verified_vehicle_color == ""
    assert row.color_source != "human_verified"
    assert row.color_confidence == 0.0
    assert row.vehicle_color == "unknown"
    payload = observation_json(row)
    assert payload["color_state"] == "unknown"
    assert payload["color_display"] == "Unknown"
    assert payload["color_verified"] is False
