"""Vehicle-first persistence and investigation search.

These observations describe one local camera track.  They are not a visual
identity system and must never be connected into a claimed cross-camera route.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from sqlalchemy import func, or_, select

from app.models import Camera, VehicleObservation
from app.services.coverage import OWN_FEED_SOURCE_TYPES, feed_of
from app.services.evidence import save_context_crop, save_crop
from app.services.serialize import ist_label, utc_iso
from app.services.timing import source_time_from_ingest

VEHICLE_TYPES = (
    "car", "suv", "two_wheeler", "truck", "bus", "van", "auto_rickshaw", "taxi_cab", "unknown",
)
VEHICLE_COLORS = (
    "white", "black", "silver", "gray", "red", "blue", "green", "yellow", "orange", "brown", "other", "unknown",
)

# The vocabularies above are what the API and DB accept, including values kept
# for backwards compatibility. These are what any weight on this host can
# actually predict. Everything else abstains to "unknown" -- see
# app/services/vehicle_attributes.py.
#   types   : car/van/truck/bus from the OMZ model, two_wheeler from COCO id 3
#   colours : the OMZ model's own seven classes
SUPPORTED_VEHICLE_TYPES = ("car", "van", "truck", "bus", "two_wheeler")
SUPPORTED_VEHICLE_COLORS = ("white", "gray", "yellow", "red", "green", "blue", "black")


def clean_vehicle_type(value: str | None) -> str:
    raw = str(value or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "motorcycle": "two_wheeler", "bike": "two_wheeler", "scooter": "two_wheeler",
        "auto": "auto_rickshaw", "rickshaw": "auto_rickshaw", "autorickshaw": "auto_rickshaw",
        "cab": "taxi_cab", "taxi": "taxi_cab", "pickup": "truck", "lorry": "truck",
        "sedan": "car", "hatchback": "car", "jeep": "suv",
    }
    return aliases.get(raw, raw) if aliases.get(raw, raw) in VEHICLE_TYPES else "unknown"


def clean_vehicle_color(value: str | None) -> str:
    raw = str(value or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"grey": "gray", "maroon": "red", "beige": "brown", "gold": "brown"}
    return aliases.get(raw, raw) if aliases.get(raw, raw) in VEHICLE_COLORS else "unknown"


def color_estimate(crop) -> tuple[str, float]:
    """HSV colour baseline. NOT the primary source, and not equivalent to one.

    Retained only for comparison against the trained classifier. It has no
    calibrated probability -- the 0.65 below is a constant, not a confidence --
    so it is written with ``color_source="opencv_hsv_fallback"`` and is never
    presented as a classifier result.
    """
    from app.services.anpr import estimate_vehicle_color

    color = clean_vehicle_color(estimate_vehicle_color(crop))
    if color == "unknown":
        return color, 0.0
    return color, 0.65


def _quality(box: tuple[int, int, int, int] | None, detector_confidence: float) -> float:
    if not box:
        return max(0.0, float(detector_confidence or 0.0))
    return max(0.0, float(box[2]) * float(box[3])) * max(0.01, float(detector_confidence or 0.0))


def upsert_observation(
    db,
    camera: Camera,
    *,
    run_id: str,
    track_id: str,
    frame_index: int,
    source_pts_ms: float | None,
    box: tuple[int, int, int, int] | None,
    frame_shape: tuple[int, int],
    frame,
    crop,
    detector: str,
    detector_confidence: float,
    vehicle_type: str,
) -> tuple[VehicleObservation, bool]:
    now = datetime.now(timezone.utc)
    source_time, _ = source_time_from_ingest(now, camera.clock_offset_ms)
    row = db.scalar(select(VehicleObservation).where(
        VehicleObservation.camera_id == camera.id,
        VehicleObservation.run_id == run_id,
        VehicleObservation.track_id == track_id,
    ))
    # The detector's COCO class is recorded as provenance only. It is NOT a
    # type confidence: copying detector_confidence into type_confidence (as
    # this used to) claims a certainty about `car` vs `van` that a 4-class
    # COCO head cannot support. The real attributes arrive from
    # apply_attributes() once enough good crops exist.
    detector_type = clean_vehicle_type(vehicle_type)
    score = _quality(box, detector_confidence)
    if row is None:
        evidence = save_crop(crop, camera.id, track_id, label="vehicle")
        context = save_context_crop(frame, box, camera.id, track_id)
        row = VehicleObservation(
            camera_id=camera.id, run_id=run_id, track_id=track_id,
            first_seen_at=source_time, last_seen_at=source_time,
            first_pts_ms=source_pts_ms, last_pts_ms=source_pts_ms,
            best_frame_index=frame_index,
            bbox_x=box[0] if box else None, bbox_y=box[1] if box else None,
            bbox_w=box[2] if box else None, bbox_h=box[3] if box else None,
            frame_width=frame_shape[1], frame_height=frame_shape[0],
            detector=detector, detector_confidence=float(detector_confidence or 0.0),
            vehicle_type="unknown", type_confidence=0.0, type_source="pending",
            vehicle_color="unknown", color_confidence=0.0, color_source="pending",
            evidence_path=evidence, context_evidence_path=context,
            metadata_json={
                "quality_score": score,
                "attribute_status": "pending",
                "detector_type": detector_type,
            },
        )
        db.add(row)
        db.flush()
        return row, True
    row.last_seen_at = source_time
    row.last_pts_ms = source_pts_ms
    meta = dict(row.metadata_json or {})
    old_score = float(meta.get("quality_score") or 0.0)
    if score > old_score:
        row.best_frame_index = frame_index
        row.bbox_x = box[0] if box else None
        row.bbox_y = box[1] if box else None
        row.bbox_w = box[2] if box else None
        row.bbox_h = box[3] if box else None
        row.frame_width, row.frame_height = frame_shape[1], frame_shape[0]
        row.detector, row.detector_confidence = detector, float(detector_confidence or 0.0)
        # A clearer crop replaces the stored evidence and the detector fields.
        # Type and colour are deliberately NOT touched here: they belong to
        # apply_attributes(), which aggregates across the whole track rather
        # than overwriting from whichever single frame is currently best.
        row.evidence_path = save_crop(crop, camera.id, track_id, label="vehicle") or row.evidence_path
        row.context_evidence_path = save_context_crop(frame, box, camera.id, track_id) or row.context_evidence_path
        meta["quality_score"] = score
        meta["detector_type"] = detector_type
        row.metadata_json = meta
    db.flush()
    return row, False


VERIFIED = "verified"
ESTIMATED = "estimated"
UNKNOWN = "unknown"


def apply_attributes(db, observation_id: int, aggregated) -> bool:
    """Store the raw model output, then derive the operational attributes.

    Two separate things are recorded, and conflating them is the failure this
    function exists to prevent:

    * ``metadata_json["attributes"]`` keeps the **raw candidate** for every
      attribute — label, confidence, temporal agreement, model source, reason
      and the full probability vectors. This is diagnostic evidence and is
      never suppressed, overwritten by a reviewer, or presented as a result.
    * the ``vehicle_type`` / ``vehicle_color`` columns are the **operational**
      values an operator sees and searches. Vehicle type is forced to
      ``unknown`` unless a human verified it or a model cleared
      ``type_deployment_gate``; colour is exposed but only ever as an estimate.

    A human verdict always wins and is never clobbered by a later frame.
    """
    from app.services.vehicle_attributes import type_deployment_gate

    row = db.get(VehicleObservation, observation_id)
    if row is None:
        return False
    payload = aggregated.as_dict()
    meta = dict(row.metadata_json or {})

    type_candidate = clean_vehicle_type(payload["vehicle_type"])
    color_candidate = clean_vehicle_color(payload["vehicle_color"])

    # -- raw model output, for diagnostics only --------------------------
    meta["attribute_status"] = "deterministic"
    meta["attributes"] = {
        "type_candidate": type_candidate,
        "type_candidate_confidence": payload["vehicle_type_confidence"],
        "type_agreement": payload["vehicle_type_agreement"],
        "type_model_source": payload["type_source"] or "unknown",
        "type_reason": payload["type_reason"],
        "color_candidate": color_candidate,
        "color_candidate_confidence": payload["vehicle_color_confidence"],
        "color_agreement": payload["vehicle_color_agreement"],
        "color_model_source": payload["color_source"] or "unknown",
        "color_reason": payload["color_reason"],
        "observations": payload["attribute_observations"],
        "detector_type": payload["detector_type"],
        "type_rejected": payload.get("type_rejected"),
        "color_rejected": payload.get("color_rejected"),
        "probabilities": {"type": aggregated.type_probs, "color": aggregated.color_probs},
        # Automatic type is not a verified result on this deployment.
        "verified": False,
        "review_required": True,
    }
    # A stale VLM verdict must not survive next to a deterministic result.
    meta.pop("cloud_refinement", None)
    row.metadata_json = meta

    # -- operational values ----------------------------------------------
    gate_passed, gate_reason = type_deployment_gate()
    meta["attributes"]["type_gate_passed"] = gate_passed
    meta["attributes"]["type_gate_reason"] = gate_reason

    if row.verified_vehicle_type:
        row.vehicle_type = clean_vehicle_type(row.verified_vehicle_type)
        row.type_confidence = 1.0
        row.type_source = "human_verified"
    elif gate_passed:
        row.vehicle_type = type_candidate
        row.type_confidence = float(payload["vehicle_type_confidence"])
        row.type_source = payload["type_source"] or "unknown"
    else:
        # Suppressed, not missing. The candidate is in metadata above.
        row.vehicle_type = UNKNOWN
        row.type_confidence = 0.0
        row.type_source = "suppressed_pending_validation"

    if row.verified_vehicle_color:
        row.vehicle_color = clean_vehicle_color(row.verified_vehicle_color)
        row.color_confidence = 1.0
        row.color_source = "human_verified"
    elif not bool(getattr(settings_module(), "vattr_color_enabled", True)):
        row.vehicle_color = UNKNOWN
        row.color_confidence = 0.0
        row.color_source = "disabled"
    else:
        row.vehicle_color = color_candidate
        row.color_confidence = float(payload["vehicle_color_confidence"])
        row.color_source = payload["color_source"] or "unknown"

    db.flush()
    return True


def settings_module():
    from app.config import settings

    return settings


def attribute_states(row: VehicleObservation) -> tuple[str, str]:
    """Return (type_state, color_state), each verified | estimated | unknown."""
    if row.verified_vehicle_type:
        type_state = VERIFIED
    elif row.vehicle_type and row.vehicle_type != UNKNOWN:
        type_state = ESTIMATED
    else:
        type_state = UNKNOWN

    if row.verified_vehicle_color:
        color_state = VERIFIED
    elif row.vehicle_color and row.vehicle_color != UNKNOWN:
        color_state = ESTIMATED
    else:
        color_state = UNKNOWN
    return type_state, color_state


def review_observation(
    db,
    observation_id: int,
    *,
    actor: str,
    vehicle_type: str | None = None,
    vehicle_color: str | None = None,
    note: str = "",
    context: dict | None = None,
) -> VehicleObservation | None:
    """Record a human verdict without destroying the raw model output.

    Passing ``""`` clears a previous verdict for that attribute and returns it
    to the automatic value. The model's candidate stays in
    ``metadata_json["attributes"]`` either way, so a correction is always
    auditable against what the model actually said.
    """
    from datetime import datetime, timezone

    row = db.get(VehicleObservation, observation_id)
    if row is None:
        return None

    meta = dict(row.metadata_json or {})
    attributes = dict(meta.get("attributes") or {})
    history = list(meta.get("review_history") or [])
    entry: dict = {"actor": actor, "at": datetime.now(timezone.utc).isoformat(), "note": note[:500]}
    # How the label was produced: which surface, whether the annotator could see
    # the model's answer, what scope they were sweeping. A label taken with the
    # prediction on screen is confirmation-prone, and the export has to be able
    # to tell those apart from blind ones rather than treating all 5,000 the
    # same. Structured, not parsed back out of a truncated note string.
    if context:
        entry["context"] = context

    if vehicle_type is not None:
        cleaned = clean_vehicle_type(vehicle_type) if vehicle_type else ""
        if vehicle_type and cleaned == UNKNOWN and vehicle_type.strip().lower() != UNKNOWN:
            raise ValueError(f"unknown vehicle_type: {vehicle_type!r}")
        entry["vehicle_type"] = {"from": row.verified_vehicle_type, "to": cleaned,
                                 "model_candidate": attributes.get("type_candidate", "")}
        row.verified_vehicle_type = cleaned
    if vehicle_color is not None:
        cleaned = clean_vehicle_color(vehicle_color) if vehicle_color else ""
        if vehicle_color and cleaned == UNKNOWN and vehicle_color.strip().lower() != UNKNOWN:
            raise ValueError(f"unknown vehicle_color: {vehicle_color!r}")
        entry["vehicle_color"] = {"from": row.verified_vehicle_color, "to": cleaned,
                                  "model_candidate": attributes.get("color_candidate", "")}
        row.verified_vehicle_color = cleaned

    row.verified_by = actor[:64]
    row.verified_at = datetime.now(timezone.utc)
    row.review_status = VERIFIED if (row.verified_vehicle_type or row.verified_vehicle_color) else "unreviewed"

    history.append(entry)
    meta["review_history"] = history[-20:]
    row.metadata_json = meta

    _recompute_operational(row, attributes)
    db.flush()
    return row


def _recompute_operational(row: VehicleObservation, attributes: dict) -> None:
    """Re-derive the operational columns after a review, gate unchanged."""
    from app.services.vehicle_attributes import type_deployment_gate

    gate_passed, _ = type_deployment_gate()
    if row.verified_vehicle_type:
        row.vehicle_type = clean_vehicle_type(row.verified_vehicle_type)
        row.type_confidence, row.type_source = 1.0, "human_verified"
    elif gate_passed and attributes.get("type_candidate"):
        row.vehicle_type = clean_vehicle_type(attributes["type_candidate"])
        row.type_confidence = float(attributes.get("type_candidate_confidence") or 0.0)
        row.type_source = attributes.get("type_model_source") or "unknown"
    else:
        row.vehicle_type, row.type_confidence = UNKNOWN, 0.0
        row.type_source = "suppressed_pending_validation"

    if row.verified_vehicle_color:
        row.vehicle_color = clean_vehicle_color(row.verified_vehicle_color)
        row.color_confidence, row.color_source = 1.0, "human_verified"
    elif attributes.get("color_candidate"):
        row.vehicle_color = clean_vehicle_color(attributes["color_candidate"])
        row.color_confidence = float(attributes.get("color_candidate_confidence") or 0.0)
        row.color_source = attributes.get("color_model_source") or "unknown"
    else:
        # Clearing a verdict on a row with no model candidate to fall back to
        # used to leave `human_verified` and confidence 1.0 in place, so the
        # record went on asserting that a person had confirmed a colour after
        # that person's verdict was withdrawn. There is nothing to show here,
        # and "nothing" is what it must say. (The type branch above already
        # had this case; colour did not.)
        row.vehicle_color, row.color_confidence = UNKNOWN, 0.0
        row.color_source = "cleared_no_candidate"


ATTRIBUTE_DISCLAIMER = (
    "Vehicle colour is an ESTIMATE and needs review. Automatic vehicle type is "
    "suppressed as unknown on this deployment and is not a result. Matching "
    "attributes do not confirm the same physical vehicle."
)

# Measured on 59 blind-labelled cam01 night tracks. Surfaced through the API so
# the UI cannot show attributes without showing what they are worth.
POC_ACCURACY = {
    "measured": True,
    "camera": "cam01 (Ahmedabad, night)",
    "labelled_tracks": 59,
    "excluded_unclear": 14,
    "vehicle_type": {"scored_tracks": 28, "coverage": 0.64, "selective_precision": 0.50},
    "vehicle_color": {"scored_tracks": 42, "coverage": 0.71, "selective_precision": 0.83},
    "type_usable": False,
    # Re-measured after the multi-vehicle crop fix, on the 35-track offline
    # segment only (the other segment's track ids no longer align). A DIFFERENT
    # basis from the headline numbers above, so the two are not comparable
    # row-for-row -- it is reported separately rather than folded in.
    "after_multi_vehicle_crop_fix": {
        "basis": "offline segment only, 35 tracks",
        "vehicle_type": {"coverage_before": 0.55, "precision_before": 0.36,
                         "coverage_after": 0.60, "precision_after": 0.67},
        "vehicle_color": {"coverage_before": 0.78, "precision_before": 0.81,
                          "coverage_after": 0.63, "precision_after": 0.71},
        "note": (
            "Type improved materially (4/11 -> 8/12 correct): contaminated crops "
            "were corrupting it. Colour moved 17/21 -> 12/17, a two-track "
            "difference on n<=21 that is not distinguishable from noise. "
            "Thresholds were deliberately NOT retuned to improve this, since "
            "tuning on the evaluation set would invalidate it."
        ),
    },
    "notes": [
        "Vehicle type is NOT fit for operational use: 50% selective precision, and "
        "precision FELL to 30% when the confidence floor was raised to 0.95, so the "
        "errors are systematic rather than noisy.",
        "Indian city buses are frequently classified as trucks; this is a barrier/"
        "toll-gate model applied out of distribution.",
        "No vehicle-type threshold reaches the 95% precision objective.",
        "The `detector` conflict policy beat `abstain` and `classifier`, but it was "
        "selected on the same 20-track segment and still reached only 50% precision. "
        "Winning that comparison does not make it a production classifier.",
        "Colour's dominant error is white vehicles read as red, from brake lights and "
        "red signage at night.",
        "Labels were blind but come from a single annotator and are not independently "
        "verified.",
        "Dense traffic contaminates crops: 243 of 276 crops came from frames holding "
        "more than one vehicle and 20% contained >25% of a neighbour. Such crops are "
        "now rejected as `multi_vehicle_crop`, which raised type precision to 67% on "
        "the offline segment but still leaves it below any usable bar.",
    ],
}


# ---------------------------------------------------------------------------
# Plate presentation
#
# A plate failure must never remove a vehicle observation, so the plate is a
# *state* on the record rather than a precondition for it. The five states below
# are what an investigator sees; each one is a different operational fact and
# collapsing them into a blank cell would hide why the plate is missing.
# ---------------------------------------------------------------------------
#: Values match app/services/vehicle_runner.py so one vocabulary describes a
#: plate everywhere it is recorded or displayed.
PLATE_READ = "ok"                    # characters recognised -> still an estimate
PLATE_UNREADABLE = "plate_unreadable"  # a plate was found but not resolvable
PLATE_NOT_VISIBLE = "plate_not_visible"  # no plate region in the vehicle's frames
PLATE_NOT_CHECKED = "not_checked"      # recognition did not run for this vehicle

#: Internal recogniser reason codes -> the operational plate state. Codes that
#: are not about the plate itself (attribute stage) are ignored entirely.
PLATE_REASON_STATE = {
    "candidate": PLATE_READ,
    "no_plate_localized": PLATE_NOT_VISIBLE,
    "insufficient_pixels": PLATE_UNREADABLE,
    "ocr_empty": PLATE_UNREADABLE,
    "syntax_invalid": PLATE_UNREADABLE,
    "underexposed": PLATE_UNREADABLE,
    "glare": PLATE_UNREADABLE,
    "blur": PLATE_UNREADABLE,
    "multi_vehicle_crop": PLATE_UNREADABLE,
    "authentication_failed": PLATE_NOT_CHECKED,
}

#: Plain language for the operator. No reason code ever reaches the screen.
PLATE_STATE_TEXT = {
    PLATE_READ: "Read by the system — not verified by a person",
    PLATE_UNREADABLE: "A plate was seen but could not be read",
    PLATE_NOT_VISIBLE: "No number plate visible in this vehicle's frames",
    PLATE_NOT_CHECKED: "Number-plate recognition did not run for this vehicle",
}

#: Why a plate could not be read, in words an investigator can act on.
PLATE_DETAIL_TEXT = {
    "insufficient_pixels": "Plate too small in frame",
    "ocr_empty": "No characters could be resolved",
    "syntax_invalid": "Characters did not form a valid plate",
    "underexposed": "Too dark",
    "glare": "Glare on the plate",
    "blur": "Motion blur",
    "multi_vehicle_crop": "Another vehicle overlapped the crop",
    "authentication_failed": "Camera feed could not be opened",
}


def _plate_rank(state: str) -> int:
    """Best-known-first. A read beats an unreadable, which beats no plate."""
    return {PLATE_READ: 0, PLATE_UNREADABLE: 1, PLATE_NOT_VISIBLE: 2, PLATE_NOT_CHECKED: 3}.get(state, 4)


def plate_status_for(db, rows: list[VehicleObservation]) -> dict[int, dict]:
    """Resolve one plate state per observation, in two batched queries.

    Persisted sightings win, because those are the reads that were accepted and
    could raise an alert. Recognition attempts fill in *why* there is no plate
    for everything else, which is the difference between "not visible" and
    "unreadable" -- an investigator needs that to decide whether re-checking the
    footage is worth it.
    """
    from app.models import RecognitionAttempt, Sighting

    ids = [r.id for r in rows if r.id is not None]
    out: dict[int, dict] = {
        r.id: {"plate_text": "", "plate_state": PLATE_NOT_CHECKED, "plate_detail": ""}
        for r in rows
        if r.id is not None
    }
    if not ids:
        return out

    # Track keys, so an observation with no persisted sighting can still explain
    # itself from the recognition attempts recorded against the same track.
    track_keys = {(r.camera_id, r.track_id): r.id for r in rows if r.track_id}
    if track_keys:
        attempts = db.scalars(
            select(RecognitionAttempt).where(
                RecognitionAttempt.camera_id.in_({c for c, _ in track_keys}),
                RecognitionAttempt.track_id.in_({t for _, t in track_keys}),
            )
        )
        for attempt in attempts:
            obs_id = track_keys.get((attempt.camera_id, attempt.track_id))
            if obs_id is None:
                continue
            state = PLATE_REASON_STATE.get(attempt.reason_code or "")
            if state is None:
                continue  # attribute-stage or unknown code: not a plate fact
            current = out[obs_id]
            if _plate_rank(state) > _plate_rank(current["plate_state"]):
                continue
            if state == PLATE_READ and not (attempt.plate_norm and attempt.syntax_ok):
                state = PLATE_UNREADABLE
            out[obs_id] = {
                "plate_text": attempt.plate_norm if state == PLATE_READ else "",
                "plate_state": state,
                "plate_detail": PLATE_DETAIL_TEXT.get(attempt.reason_code or "", ""),
            }

    for sighting in db.scalars(select(Sighting).where(Sighting.vehicle_observation_id.in_(ids))):
        obs_id = sighting.vehicle_observation_id
        if obs_id not in out:
            continue
        plate = (sighting.plate_voted or sighting.plate_norm or "").strip()
        if plate and sighting.syntax_ok:
            out[obs_id] = {"plate_text": plate, "plate_state": PLATE_READ, "plate_detail": ""}
        elif _plate_rank(PLATE_UNREADABLE) < _plate_rank(out[obs_id]["plate_state"]):
            out[obs_id] = {
                "plate_text": "",
                "plate_state": PLATE_UNREADABLE,
                "plate_detail": out[obs_id].get("plate_detail") or "Characters did not form a valid plate",
            }
    return out


# ---------------------------------------------------------------------------
# Supported-class presentation
#
# Historic rows (type_source="legacy_sighting") hold classes such as `suv`,
# `auto_rickshaw`, `silver` and `orange` that NO weight on this host can
# predict. Showing those as if they were predictions would advertise a
# capability that does not exist, so an unsupported value from a model is
# presented as Unknown. A value a PERSON verified is a human fact and stands.
# The stored value is never rewritten -- only how it is displayed.
# ---------------------------------------------------------------------------
UNSUPPORTED_CLASS_NOTE = (
    "An earlier version of this system recorded a value here that no model on "
    "this host can produce. It is not shown, because it cannot be reproduced "
    "or relied on."
)
TYPE_GATE_NOTE = (
    "Automatic vehicle type is not in operational use on this deployment. It "
    "stays Unknown until a person confirms it."
)
LEGACY_SOURCE_NOTE = (
    "This record predates the current vehicle-attribute model. Its automatic "
    "reading is kept for audit but is not shown as a result."
)

#: The ONLY automatic source a colour may be shown from. The OpenCV HSV
#: baseline and the retired VLM path both wrote into the same column; neither
#: is a calibrated classifier, and presenting either as an estimate would put a
#: number on screen that nothing behind it supports.
DETERMINISTIC_SOURCE_PREFIX = "openvino:"


def presentation(row: VehicleObservation) -> dict:
    """Derive what the production UI shows for type and colour.

    Three things can demote an automatic value to Unknown, and all three are
    live on this deployment:

    1. the type deployment gate has not been cleared, so NO automatic vehicle
       type is shown, whatever a row happens to hold;
    2. the value came from something other than the deterministic attribute
       model (a legacy row, the HSV baseline, a pending write);
    3. the class is one no weight on this host can predict (`suv`, `silver`,
       `orange` and friends).

    A human verdict is exempt from all three: it is a person's observation, not
    a model's, and it is the only thing this console shows as confirmed.
    """
    from app.services.vehicle_attributes import type_deployment_gate

    type_state, color_state = attribute_states(row)
    out = {
        "type_display": "Unknown",
        "type_state": UNKNOWN,
        "type_note": "",
        "color_display": "Unknown",
        "color_state": UNKNOWN,
        "color_note": "",
    }

    if type_state == VERIFIED:
        out["type_display"] = _humanise(row.verified_vehicle_type or row.vehicle_type)
        out["type_state"] = VERIFIED
    elif type_state == ESTIMATED:
        gate_passed, _reason = type_deployment_gate()
        if not gate_passed:
            out["type_note"] = TYPE_GATE_NOTE
        elif row.vehicle_type not in SUPPORTED_VEHICLE_TYPES:
            out["type_note"] = UNSUPPORTED_CLASS_NOTE
        elif not str(row.type_source or "").startswith(DETERMINISTIC_SOURCE_PREFIX):
            out["type_note"] = LEGACY_SOURCE_NOTE
        else:
            out["type_display"] = _humanise(row.vehicle_type)
            out["type_state"] = ESTIMATED

    if color_state == VERIFIED:
        out["color_display"] = _humanise(row.verified_vehicle_color or row.vehicle_color)
        out["color_state"] = VERIFIED
    elif color_state == ESTIMATED:
        if row.vehicle_color not in SUPPORTED_VEHICLE_COLORS:
            out["color_note"] = UNSUPPORTED_CLASS_NOTE
        elif not str(row.color_source or "").startswith(DETERMINISTIC_SOURCE_PREFIX):
            out["color_note"] = LEGACY_SOURCE_NOTE
        else:
            out["color_display"] = _humanise(row.vehicle_color)
            out["color_state"] = ESTIMATED
    return out


#: Vocabulary values that are not ordinary words.
_DISPLAY_NAMES = {"suv": "SUV", "two_wheeler": "Two-wheeler", "taxi_cab": "Taxi"}


def _humanise(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if raw in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[raw]
    text = raw.replace("_", " ")
    return text[:1].upper() + text[1:] if text else "Unknown"


def observation_json(
    row: VehicleObservation,
    plate: dict | None = None,
    *,
    include_diagnostics: bool = False,
) -> dict:
    """Serialise one observation.

    ``include_diagnostics`` controls the raw model payload -- probability
    vectors, rejected candidates, per-crop observations. That blob is evidence
    and is never deleted, but production has no use for it and shipping it to
    a browser that will not render it would be hiding rather than withholding.
    The developer console asks for it explicitly.
    """
    camera = row.camera
    attributes = (row.metadata_json or {}).get("attributes") or {}
    type_state, color_state = attribute_states(row)
    plate = plate or {"plate_text": "", "plate_state": PLATE_NOT_CHECKED, "plate_detail": ""}
    payload = {
        "id": row.id, "camera_id": row.camera_id, "camera_name": camera.name if camera else "",
        # own | government -- which of the two feed paths produced this record.
        "feed": feed_of(camera),
        "city": camera.city if camera else "", "department": camera.department if camera else "",
        "lat": camera.lat if camera else None, "lng": camera.lng if camera else None,
        "track_id": row.track_id, "run_id": row.run_id,
        "first_seen_at": utc_iso(row.first_seen_at), "first_seen_at_ist": ist_label(row.first_seen_at),
        "last_seen_at": utc_iso(row.last_seen_at), "last_seen_at_ist": ist_label(row.last_seen_at),
        "vehicle_type": row.vehicle_type, "type_confidence": row.type_confidence, "type_source": row.type_source,
        "vehicle_color": row.vehicle_color, "color_confidence": row.color_confidence, "color_source": row.color_source,
        "detector": row.detector, "detector_confidence": row.detector_confidence,
        "evidence_path": row.evidence_path, "context_evidence_path": row.context_evidence_path,
        # verified | estimated | unknown -- what the UI badges off.
        "type_state": type_state,
        "color_state": color_state,
        "type_verified": bool(row.verified_vehicle_type),
        "color_verified": bool(row.verified_vehicle_color),
        "verified_vehicle_type": row.verified_vehicle_type,
        "verified_vehicle_color": row.verified_vehicle_color,
        "review_status": row.review_status,
        "review_required": not (row.verified_vehicle_type and row.verified_vehicle_color),
        "verified_by": row.verified_by,
        "verified_at": utc_iso(row.verified_at) if row.verified_at else None,
        # Raw model output, kept visible for diagnostics and never presented
        # as the operational answer.
        "type_candidate": attributes.get("type_candidate", ""),
        "type_candidate_confidence": attributes.get("type_candidate_confidence", 0.0),
        "type_model_source": attributes.get("type_model_source", ""),
        "type_agreement": attributes.get("type_agreement"),
        "color_agreement": attributes.get("color_agreement"),
        "type_reason": attributes.get("type_reason", ""),
        "color_reason": attributes.get("color_reason", ""),
        "attribute_observations": attributes.get("observations", 0),
        "type_suppressed": row.type_source == "suppressed_pending_validation",
        "disclaimer": ATTRIBUTE_DISCLAIMER,
    }
    if include_diagnostics:
        payload["metadata"] = row.metadata_json or {}
        # What the model ACTUALLY produced, before presentation() demotes it.
        # Deliberately a nested block rather than top-level keys: it cannot
        # collide with the fields presentation() overwrites, and "is production
        # leaking raw state?" is then a one-line assertion.
        #
        # There is NO switch to skip presentation() -- that would be one
        # careless keyword argument away from production rendering a vehicle
        # type the deployment gate exists to suppress. This block rides inside
        # include_diagnostics, which is only ever true in developer mode.
        probabilities = attributes.get("probabilities") or {}
        payload["raw"] = {
            "type_state": type_state, "color_state": color_state,
            "vehicle_type": row.vehicle_type, "vehicle_color": row.vehicle_color,
            "type_confidence": row.type_confidence, "color_confidence": row.color_confidence,
            "type_source": row.type_source, "color_source": row.color_source,
            "type_candidate": attributes.get("type_candidate", ""),
            "type_candidate_confidence": attributes.get("type_candidate_confidence", 0.0),
            # Not exposed at top level anywhere else, so this closes a real gap.
            "color_candidate": attributes.get("color_candidate", ""),
            "color_candidate_confidence": attributes.get("color_candidate_confidence", 0.0),
            "type_probs": probabilities.get("type") or {},
            "color_probs": probabilities.get("color") or {},
            "type_rejected": attributes.get("type_rejected"),
            "color_rejected": attributes.get("color_rejected"),
            "review_history": (row.metadata_json or {}).get("review_history") or [],
        }
    # Presentation last: it decides what type_state/color_state the operator is
    # shown, including demoting a class no model on this host can predict.
    payload.update(presentation(row))
    payload.update(plate)
    payload["plate_state_text"] = PLATE_STATE_TEXT.get(plate.get("plate_state", PLATE_NOT_CHECKED), "")
    return payload


def _feed_clause(feed: str):
    """SQL twin of app/services/coverage.py:feed_of. Keep the two in step."""
    own = Camera.catalogue_camera_id.is_(None) & Camera.source_type.in_(OWN_FEED_SOURCE_TYPES)
    return own if feed == "own" else ~own


#: JSON path to the model's own type answer. It is NOT in the vehicle_type
#: column: while the deployment gate is closed apply_attributes() writes
#: vehicle_type="unknown", type_confidence=0.0 on every row it touches, so a
#: labelling queue that filtered the column would find only the handful of rows
#: a human had already labelled. Measured on the demo database: 3,569 gated rows
#: hold `unknown` in the column and car/truck/bus/van/two_wheeler in here.
_TYPE_CANDIDATE = VehicleObservation.metadata_json["attributes"]["type_candidate"]
_TYPE_CANDIDATE_CONF = VehicleObservation.metadata_json["attributes"]["type_candidate_confidence"]


def search_observations(db, *, start, end, vehicle_type=None, vehicle_color=None, camera_id=None,
                        min_confidence=0.0, limit=100, offset=0, sort="asc",
                        plate_state=None, review=None, feed=None, include_diagnostics=False,
                        # ---- developer labelling only. Production passes none of these,
                        # and they are not reachable from /api/investigations/vehicles.
                        raw_attribute_filters=False, type_source=None, color_source=None,
                        min_type_confidence=None, max_type_confidence=None,
                        min_color_confidence=None, max_color_confidence=None,
                        exclude_ids=None) -> dict:
    query = select(VehicleObservation).where(
        VehicleObservation.first_seen_at >= start,
        VehicleObservation.first_seen_at <= end,
        VehicleObservation.detector_confidence >= float(min_confidence or 0.0),
    )
    warnings: list[str] = []
    if raw_attribute_filters:
        warnings.append(
            "Developer labelling mode: filters match the RAW model output, including "
            "values production refuses to display. This result set is not what an "
            "investigator would see."
        )
        if vehicle_type:
            wanted = clean_vehicle_type(vehicle_type)
            query = query.where(or_(
                _TYPE_CANDIDATE.as_string() == wanted,   # gated rows
                VehicleObservation.vehicle_type == wanted,  # legacy_sighting rows
                VehicleObservation.verified_vehicle_type == wanted,
            ))
        if vehicle_color:
            # No color_source restriction here: the rows production hides are
            # exactly the ones worth labelling.
            query = query.where(VehicleObservation.vehicle_color == clean_vehicle_color(vehicle_color))
        if type_source:
            query = query.where(VehicleObservation.type_source.startswith(type_source))
        if color_source:
            query = query.where(VehicleObservation.color_source.startswith(color_source))
        # Confidence is deliberately asymmetric: colour's lives in a column,
        # type's only in JSON, for the same gate reason as above.
        if min_type_confidence is not None:
            query = query.where(_TYPE_CANDIDATE_CONF.as_float() >= float(min_type_confidence))
        if max_type_confidence is not None:
            query = query.where(_TYPE_CANDIDATE_CONF.as_float() <= float(max_type_confidence))
        if min_color_confidence is not None:
            query = query.where(VehicleObservation.color_confidence >= float(min_color_confidence))
        if max_color_confidence is not None:
            query = query.where(VehicleObservation.color_confidence <= float(max_color_confidence))
        if exclude_ids:
            query = query.where(VehicleObservation.id.notin_(list(exclude_ids)))
        vehicle_type = vehicle_color = None  # handled above; skip the production clauses
    if vehicle_type:
        # Automatic type is suppressed, so a type filter can only match records
        # a human verified. Silently matching model guesses would make an
        # unreliable attribute behave like a search key.
        wanted = clean_vehicle_type(vehicle_type)
        query = query.where(VehicleObservation.verified_vehicle_type == wanted)
        warnings.append(
            f"Vehicle type filter {wanted!r} matches only HUMAN-VERIFIED records. "
            "Automatic vehicle type measured 50% precision and is not used for search."
        )
    if vehicle_color:
        # Matches only colours this console is willing to SHOW: a human verdict,
        # or the deterministic attribute model. Without this, a colour search
        # would return legacy rows whose colour the results table then renders
        # as Unknown -- a list that contradicts itself.
        query = query.where(
            VehicleObservation.vehicle_color == clean_vehicle_color(vehicle_color),
            or_(
                VehicleObservation.color_source == "human_verified",
                VehicleObservation.color_source.startswith(DETERMINISTIC_SOURCE_PREFIX),
            ),
        )
        warnings.append(
            "Colour is an ESTIMATE (measured 83% precision at 71% coverage on one "
            "night camera). This filter will MISS vehicles whose colour was not "
            "recognised and will INCLUDE vehicles of other colours -- white vehicles "
            "are most often mis-read as red under brake lights and signage. "
            "Treat results as a shortlist to review, never as a definitive set."
        )
    if camera_id:
        query = query.where(VehicleObservation.camera_id == camera_id)
    if feed in {"own", "government"}:
        query = query.join(Camera, Camera.id == VehicleObservation.camera_id).where(_feed_clause(feed))
    if review == "pending":
        query = query.where(VehicleObservation.review_status != VERIFIED)
    elif review == "verified":
        query = query.where(VehicleObservation.review_status == VERIFIED)
    count_query = select(func.count()).select_from(query.subquery())
    total = int(db.scalar(count_query) or 0)
    ordered = VehicleObservation.first_seen_at.desc() if sort == "desc" else VehicleObservation.first_seen_at.asc()
    rows = list(db.scalars(query.order_by(ordered).offset(offset).limit(limit)))
    plates = plate_status_for(db, rows)
    records = [
        observation_json(row, plates.get(row.id), include_diagnostics=include_diagnostics)
        for row in rows
    ]
    if plate_state:
        # Plate state is derived from sightings and recognition attempts rather
        # than stored on the row, so it filters the page that was fetched. The
        # count below says so instead of implying it filtered the whole range.
        records = [r for r in records if r["plate_state"] == plate_state]
        warnings.append(
            "The plate filter was applied to the results on this page only. "
            "Widen the time range or narrow the camera to see more."
        )
    camera_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    color_counts: dict[str, int] = {}
    plate_counts: dict[str, int] = {}
    for row in records:
        camera_counts[row["camera_id"]] = camera_counts.get(row["camera_id"], 0) + 1
        type_counts[row["vehicle_type"]] = type_counts.get(row["vehicle_type"], 0) + 1
        color_counts[row["vehicle_color"]] = color_counts.get(row["vehicle_color"], 0) + 1
        plate_counts[row["plate_state"]] = plate_counts.get(row["plate_state"], 0) + 1
    return {
        "from": utc_iso(start), "to": utc_iso(end), "total": total, "offset": offset, "limit": limit,
        "returned": len(records),
        "observations": records,
        "camera_counts": camera_counts, "type_counts": type_counts, "color_counts": color_counts,
        "plate_counts": plate_counts,
        "search_warnings": warnings,
        "accuracy": POC_ACCURACY,
        "disclaimer": (
            "Matches share selected attributes; identity is not confirmed and no route "
            "is inferred. Colour is an estimate that can both miss and wrongly include "
            "vehicles. Automatic vehicle type is suppressed and is not searchable."
        ),
    }


def observations_csv(payload: dict) -> str:
    output = io.StringIO()
    fields = [
        "id", "camera_id", "camera_name", "city",
        "first_seen_at", "first_seen_at_ist", "last_seen_at_ist",
        # Both the raw stored value and what an operator was shown. Exporting
        # only the stored value would hand an analyst `silver` for a record the
        # console correctly refused to present as a prediction.
        "vehicle_type", "type_display", "type_state", "type_confidence",
        "vehicle_color", "color_display", "color_state", "color_confidence",
        "plate_text", "plate_state", "review_status", "verified_by",
        "detector", "detector_confidence", "evidence_path",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(payload.get("observations") or [])
    return output.getvalue()
