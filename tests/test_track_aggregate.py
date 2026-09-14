"""Track aggregation and the abstention gate."""
from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.services.crop_quality import CropQuality
from app.services.track_aggregate import CropObservation, TrackAccumulator
from app.services.vehicle_attributes import COLOR_CLASSES, TYPE_CLASSES, AttributeResult


def quality(score: float = 0.5, eligible: bool = True, reason: str = "") -> CropQuality:
    return CropQuality(
        width=200, height=200, sharpness=300.0, exposure=0.9, dark_fraction=0.0,
        clipped_fraction=0.0, visible_fraction=1.0, detector_confidence=0.9,
        score=score, eligible=eligible, reason=reason,
    )


def probs(mapping: dict, classes: tuple) -> np.ndarray:
    vector = np.full(len(classes), (1.0 - sum(mapping.values())) / max(1, len(classes) - len(mapping)))
    for label, value in mapping.items():
        vector[classes.index(label)] = value
    return vector.astype(np.float32)


def observation(type_map: dict, color_map: dict, *, score: float = 0.5, detector_type: str = "car", frame: int = 0):
    return CropObservation(
        frame_index=frame, pts_ms=float(frame * 100), quality=quality(score),
        detector_type=detector_type, detector_confidence=0.9,
        attributes=AttributeResult(
            type_probs=probs(type_map, TYPE_CLASSES),
            color_probs=probs(color_map, COLOR_CLASSES),
            model_id="test",
        ),
    )


def accumulator_with(observations: list[CropObservation], track_id: str = "t1") -> TrackAccumulator:
    acc = TrackAccumulator(track_id, best_crops=10)
    for obs in observations:
        acc.note_frame(obs.frame_index, obs.pts_ms)
        acc.offer(obs)
    return acc


# -- happy path ---------------------------------------------------------


def test_confident_agreeing_frames_are_accepted():
    acc = accumulator_with([
        observation({"car": 0.95}, {"red": 0.95}, frame=0),
        observation({"car": 0.93}, {"red": 0.91}, frame=1),
        observation({"car": 0.96}, {"red": 0.94}, frame=2),
    ])
    out = acc.aggregate()
    assert out.vehicle_type == "car"
    assert out.vehicle_color == "red"
    assert out.type_reason == "accepted"
    assert out.type_agreement == 1.0
    assert out.type_confidence > 0.9


# -- abstention ---------------------------------------------------------


def test_too_few_observations_abstains():
    acc = accumulator_with([observation({"car": 0.99}, {"red": 0.99})])
    out = acc.aggregate()
    assert out.vehicle_type == "unknown"
    assert out.type_reason == "too_few_observations"


def test_abstained_record_reports_zero_confidence_not_the_rejected_probability():
    """An `unknown` must never carry a high confidence number."""
    acc = accumulator_with([observation({"car": 0.99}, {"red": 0.99})])
    out = acc.aggregate()
    assert out.vehicle_type == "unknown"
    assert out.type_confidence == 0.0
    payload = out.as_dict()
    assert payload["vehicle_type_confidence"] == 0.0
    # The rejected class is still recorded, but under its own key.
    assert payload["type_rejected"]["label"] == "car"
    assert payload["type_rejected"]["probability"] > 0.9


def test_below_min_probability_abstains():
    acc = accumulator_with([
        observation({"car": 0.40, "van": 0.35}, {"red": 0.90}, frame=0),
        observation({"car": 0.41, "van": 0.34}, {"red": 0.91}, frame=1),
    ])
    out = acc.aggregate()
    assert out.vehicle_type == "unknown"
    assert out.type_reason == "below_min_probability"
    # Colour is decided independently and survives.
    assert out.vehicle_color == "red"


def test_small_top_two_margin_abstains(monkeypatch):
    monkeypatch.setattr(settings, "vattr_type_min_prob", 0.30)
    acc = accumulator_with([
        observation({"car": 0.40, "van": 0.38}, {"red": 0.9}, frame=0),
        observation({"car": 0.41, "van": 0.37}, {"red": 0.9}, frame=1),
    ])
    out = acc.aggregate()
    assert out.vehicle_type == "unknown"
    assert out.type_reason == "top_two_margin_too_small"


def test_temporal_disagreement_abstains(monkeypatch):
    monkeypatch.setattr(settings, "vattr_type_min_agreement", 0.9)
    acc = accumulator_with([
        observation({"car": 0.80}, {"red": 0.9}, score=0.9, frame=0),
        observation({"van": 0.80}, {"red": 0.9}, score=0.1, frame=1),
        observation({"car": 0.75}, {"red": 0.9}, score=0.1, frame=2),
    ])
    out = acc.aggregate()
    assert out.type_agreement < 0.9
    assert out.vehicle_type == "unknown"
    assert out.type_reason == "temporal_disagreement"


def test_type_and_color_abstain_independently():
    acc = accumulator_with([
        observation({"car": 0.97}, {"red": 0.30, "blue": 0.28}, frame=0),
        observation({"car": 0.96}, {"red": 0.31, "blue": 0.27}, frame=1),
    ])
    out = acc.aggregate()
    assert out.vehicle_type == "car"
    assert out.vehicle_color == "unknown"
    assert out.color_confidence == 0.0


def test_no_usable_crop_abstains_on_both():
    out = TrackAccumulator("empty").aggregate()
    assert out.vehicle_type == "unknown"
    assert out.vehicle_color == "unknown"
    assert out.type_reason == "no_usable_crop"


# -- detector interaction ------------------------------------------------


def _bus_called_truck() -> TrackAccumulator:
    """Measured on cam01: this model calls Indian buses `truck` at 0.99."""
    return accumulator_with([
        observation({"truck": 0.99}, {"blue": 0.9}, detector_type="bus", frame=0),
        observation({"truck": 0.99}, {"blue": 0.9}, detector_type="bus", frame=1),
    ])


def test_conflict_policy_detector_keeps_the_coco_class_by_default():
    """Default. Measured best of the three: 90% coverage at 50% precision."""
    out = _bus_called_truck().aggregate()
    assert out.vehicle_type == "bus"
    assert out.type_source == "yolo_coco"
    assert out.type_reason == "detector_wins_conflict"


def test_conflict_policy_abstain_returns_unknown(monkeypatch):
    monkeypatch.setattr(settings, "vattr_conflict_policy", "abstain")
    out = _bus_called_truck().aggregate()
    assert out.vehicle_type == "unknown"
    assert out.type_reason == "detector_classifier_conflict"
    assert out.type_confidence == 0.0


def test_conflict_policy_classifier_needs_a_decisive_probability(monkeypatch):
    monkeypatch.setattr(settings, "vattr_conflict_policy", "classifier")
    # 0.99 clears vattr_type_conflict_min_prob, so the classifier wins...
    assert _bus_called_truck().aggregate().vehicle_type == "truck"
    # ...but a marginal probability still abstains.
    monkeypatch.setattr(settings, "vattr_type_min_prob", 0.30)
    marginal = accumulator_with([
        observation({"truck": 0.55, "car": 0.20}, {"blue": 0.9}, detector_type="bus", frame=0),
        observation({"truck": 0.55, "car": 0.20}, {"blue": 0.9}, detector_type="bus", frame=1),
    ])
    out = marginal.aggregate()
    assert out.vehicle_type == "unknown"
    assert out.type_reason == "detector_classifier_conflict"


def test_agreement_between_detector_and_classifier_is_accepted():
    acc = accumulator_with([
        observation({"truck": 0.95}, {"white": 0.9}, detector_type="truck", frame=0),
        observation({"truck": 0.93}, {"white": 0.9}, detector_type="truck", frame=1),
    ])
    out = acc.aggregate()
    assert out.vehicle_type == "truck"
    assert out.type_reason == "accepted"


def test_two_wheeler_comes_from_the_detector_and_colour_abstains():
    """The 4-class model has no two-wheeler class and is OOD for its colour."""
    acc = accumulator_with([
        observation({"car": 0.99}, {"red": 0.99}, detector_type="two_wheeler", frame=0),
        observation({"car": 0.99}, {"red": 0.99}, detector_type="two_wheeler", frame=1),
    ])
    out = acc.aggregate()
    assert out.vehicle_type == "two_wheeler"
    assert out.type_source == "yolo_coco"
    assert out.vehicle_color == "unknown"
    assert out.color_reason == "two_wheeler_out_of_distribution"


def test_two_wheeler_colour_can_be_enabled_explicitly(monkeypatch):
    monkeypatch.setattr(settings, "vattr_two_wheeler_color_enabled", True)
    acc = accumulator_with([
        observation({"car": 0.99}, {"red": 0.99}, detector_type="two_wheeler", frame=0),
        observation({"car": 0.99}, {"red": 0.99}, detector_type="two_wheeler", frame=1),
    ])
    out = acc.aggregate()
    assert out.vehicle_color == "red"


# -- weighting and crop selection ---------------------------------------


def test_quality_weighting_beats_plain_majority_vote():
    """Two poor crops must not outvote one good one in the aggregate."""
    acc = accumulator_with([
        observation({"car": 0.97}, {"white": 0.97}, score=0.95, frame=0),
        observation({"van": 0.55}, {"black": 0.55}, score=0.02, frame=1),
        observation({"van": 0.55}, {"black": 0.55}, score=0.02, frame=2),
    ])
    out = acc.aggregate()
    # A plain majority vote would say `van` 2-1. Quality weighting says `car`.
    assert max(out.type_probs, key=out.type_probs.get) == "car"


def test_agreement_is_unweighted_so_it_stays_a_real_second_signal():
    """Agreement deliberately ignores quality.

    If it were also quality-weighted it would just restate the aggregate and
    could never contradict it. Here weighting picks `car` but only 1 of 3
    frames agrees, so the gate abstains rather than emit a 1-frame answer.
    """
    acc = accumulator_with([
        observation({"car": 0.97}, {"white": 0.97}, score=0.95, frame=0),
        observation({"van": 0.55}, {"black": 0.55}, score=0.02, frame=1),
        observation({"van": 0.55}, {"black": 0.55}, score=0.02, frame=2),
    ])
    out = acc.aggregate()
    assert out.type_agreement == pytest.approx(1 / 3)
    assert out.vehicle_type == "unknown"
    assert out.type_reason == "temporal_disagreement"


def test_only_best_n_crops_are_kept():
    acc = TrackAccumulator("t", best_crops=2)
    for i, score in enumerate([0.1, 0.9, 0.5, 0.7]):
        acc.note_frame(i, float(i))
        acc.offer(observation({"car": 0.9}, {"red": 0.9}, score=score, frame=i))
    assert len(acc.kept) == 2
    assert [round(o.quality.score, 1) for o in acc.kept] == [0.9, 0.7]


def test_ineligible_crops_are_rejected_and_counted():
    acc = TrackAccumulator("t")
    bad = observation({"car": 0.9}, {"red": 0.9})
    bad.quality = quality(eligible=False, reason="blurred")
    acc.offer(bad)
    assert acc.kept == []
    assert acc.rejected == {"blurred": 1}


def test_min_observations_is_configurable(monkeypatch):
    monkeypatch.setattr(settings, "vattr_min_observations", 1)
    acc = accumulator_with([observation({"car": 0.99}, {"red": 0.99})])
    out = acc.aggregate()
    assert out.vehicle_type == "car"


def test_unsupported_labels_can_never_be_emitted():
    acc = accumulator_with([
        observation({"car": 0.99}, {"white": 0.99}, frame=0),
        observation({"car": 0.99}, {"white": 0.99}, frame=1),
    ])
    out = acc.aggregate()
    assert out.vehicle_type not in {"suv", "auto_rickshaw", "taxi_cab"}
    assert out.vehicle_color not in {"silver", "brown", "orange", "other"}


# -- one vehicle per record ----------------------------------------------


def test_type_and_colour_share_one_crop_weighting():
    """Regression: type and colour must be driven by identical evidence.

    The weight used to be `quality.score * probs.max()` computed PER ATTRIBUTE.
    Because the type head and the colour head peak on different crops, the two
    attributes were weighted differently over the same track -- in dense
    traffic that produced one vehicle's type beside another vehicle's colour.
    Here crop 0 is type-peaked and crop 1 is colour-peaked by the same amount,
    so their shared weights must come out equal.
    """
    acc = accumulator_with([
        observation({"car": 0.99}, {"white": 0.30}, score=0.5, frame=0),
        observation({"truck": 0.30}, {"red": 0.99}, score=0.5, frame=1),
    ])
    scored = [o for o in acc.kept if o.attributes is not None]
    weights = acc._crop_weights(scored)
    assert weights[0] == pytest.approx(weights[1])


def test_a_colour_confident_crop_does_not_outweigh_others_for_colour_alone():
    """A poor crop must not gain influence just because one head is confident.

    Crop 1 is low quality but its colour head peaks at 0.99. Under the old
    per-attribute weighting that peak bought it extra pull over colour only.
    Now both attributes use the crop's single shared weight, so the good crop
    dominates the colour aggregate as well.
    """
    acc = accumulator_with([
        observation({"car": 0.95}, {"white": 0.95}, score=0.9, frame=0),
        observation({"bus": 0.26}, {"red": 0.99}, score=0.1, frame=1),
    ])
    scored = [o for o in acc.kept if o.attributes is not None]
    weights = acc._crop_weights(scored)
    assert weights[0] > weights[1]
    colour_probs, agreement = acc._weighted(scored, "color", weights)
    assert COLOR_CLASSES[int(np.argmax(colour_probs))] == "white"
    # The two crops still disagree, so the gate abstains rather than emitting
    # a colour off a single frame. Weighting and gating are separate concerns.
    assert agreement == pytest.approx(0.5)
    assert acc.aggregate().vehicle_color == "unknown"
