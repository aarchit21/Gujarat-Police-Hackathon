"""Per-track aggregation of vehicle attributes, with explicit abstention.

One vehicle track produces many crops.  This module keeps only the best few,
aggregates their probability vectors with quality weighting, measures temporal
agreement, and then *refuses to answer* unless several independent conditions
hold.  ``unknown`` is decided separately for type and for colour.

Abstention rules (each configurable, none calibrated on this deployment):

* fewer than ``vattr_min_observations`` usable crops;
* aggregated top probability below the per-attribute minimum;
* top-two margin too small to separate the classes;
* frame-to-frame agreement below the per-attribute minimum;
* detector and classifier disagree on a class the detector is good at
  (``bus``/``truck``) without a decisive classifier probability.

Deliberately *not* done here: no bounding-box aspect-ratio rule is used to
break ties between classes, and no attribute is ever copied from another track
just because it looks similar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.config import settings
from app.services.crop_quality import CropQuality
from app.services.vehicle_attributes import (
    COLOR_CLASSES,
    TYPE_CLASSES,
    AttributeResult,
    model_id,
)

# COCO classes the detector is genuinely reliable on; used only to flag a
# disagreement with the attribute model, never to overwrite it silently.
DETECTOR_STRONG_TYPES = frozenset({"bus", "truck"})
# The attribute model has no two-wheeler class at all.
DETECTOR_ONLY_TYPES = frozenset({"two_wheeler"})

UNKNOWN = "unknown"


@dataclass
class CropObservation:
    frame_index: int
    pts_ms: float | None
    quality: CropQuality
    detector_type: str
    detector_confidence: float
    attributes: AttributeResult | None = None
    crop: np.ndarray | None = None
    context: np.ndarray | None = None
    box: tuple[int, int, int, int] | None = None


@dataclass
class AggregatedAttributes:
    vehicle_type: str = UNKNOWN
    type_confidence: float = 0.0
    type_agreement: float = 0.0
    type_source: str = ""
    type_reason: str = ""
    vehicle_color: str = UNKNOWN
    color_confidence: float = 0.0
    color_agreement: float = 0.0
    color_source: str = ""
    color_reason: str = ""
    observations: int = 0
    detector_type: str = UNKNOWN
    type_probs: dict = field(default_factory=dict)
    color_probs: dict = field(default_factory=dict)
    # Peak aggregated probability of the class the model *would* have picked.
    # Kept separate so an abstained record never carries a high "confidence".
    type_rejected_label: str = ""
    type_rejected_prob: float = 0.0
    color_rejected_label: str = ""
    color_rejected_prob: float = 0.0

    def as_dict(self) -> dict:
        payload = {
            "vehicle_type": self.vehicle_type,
            # Confidence describes the label actually emitted. When the gate
            # abstains this is 0.0, not the rejected class's probability.
            "vehicle_type_confidence": round(self.type_confidence, 4),
            "vehicle_type_agreement": round(self.type_agreement, 4),
            "type_source": self.type_source,
            "type_reason": self.type_reason,
            "vehicle_color": self.vehicle_color,
            "vehicle_color_confidence": round(self.color_confidence, 4),
            "vehicle_color_agreement": round(self.color_agreement, 4),
            "color_source": self.color_source,
            "color_reason": self.color_reason,
            "attribute_observations": self.observations,
            "detector_type": self.detector_type,
        }
        if self.vehicle_type == UNKNOWN and self.type_rejected_label:
            payload["type_rejected"] = {
                "label": self.type_rejected_label,
                "probability": round(self.type_rejected_prob, 4),
            }
        if self.vehicle_color == UNKNOWN and self.color_rejected_label:
            payload["color_rejected"] = {
                "label": self.color_rejected_label,
                "probability": round(self.color_rejected_prob, 4),
            }
        return payload

    def abstain_type(self, reason: str) -> None:
        self.vehicle_type = UNKNOWN
        self.type_reason = reason
        self.type_confidence = 0.0

    def abstain_color(self, reason: str) -> None:
        self.vehicle_color = UNKNOWN
        self.color_reason = reason
        self.color_confidence = 0.0


def _top_two(probs: np.ndarray) -> tuple[int, float, float]:
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    first = float(probs[top])
    second = float(probs[order[1]]) if probs.size > 1 else 0.0
    return top, first, second


class TrackAccumulator:
    """Collects the best crops for one track and aggregates their attributes."""

    def __init__(self, track_id: str, *, best_crops: int | None = None):
        self.track_id = track_id
        self.best_crops = int(
            best_crops if best_crops is not None else getattr(settings, "vattr_best_crops_per_track", 3) or 3
        )
        self.kept: list[CropObservation] = []
        self.seen_frames = 0
        self.rejected: dict[str, int] = {}
        self.first_pts_ms: float | None = None
        self.last_pts_ms: float | None = None
        self.first_frame_index: int | None = None
        self.last_frame_index: int | None = None

    # -- collection -----------------------------------------------------
    def note_frame(self, frame_index: int, pts_ms: float | None) -> None:
        self.seen_frames += 1
        if self.first_frame_index is None:
            self.first_frame_index = frame_index
            self.first_pts_ms = pts_ms
        self.last_frame_index = frame_index
        self.last_pts_ms = pts_ms

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def offer(self, observation: CropObservation) -> bool:
        """Keep this crop if it is among the best N seen for the track."""
        if not observation.quality.eligible:
            self.reject(observation.quality.reason or "ineligible")
            return False
        self.kept.append(observation)
        self.kept.sort(key=lambda o: o.quality.score, reverse=True)
        dropped = self.kept[self.best_crops :]
        del self.kept[self.best_crops :]
        for item in dropped:
            item.crop = None  # release pixels we will not use
            item.context = None
        return observation in self.kept

    @property
    def best(self) -> CropObservation | None:
        return self.kept[0] if self.kept else None

    def needs_attributes(self) -> list[CropObservation]:
        return [o for o in self.kept if o.attributes is None]

    # -- aggregation ----------------------------------------------------
    def _detector_type(self) -> str:
        votes: dict[str, float] = {}
        for obs in self.kept:
            votes[obs.detector_type] = votes.get(obs.detector_type, 0.0) + obs.quality.score
        if not votes:
            return UNKNOWN
        return max(votes.items(), key=lambda kv: kv[1])[0]

    def aggregate(self) -> AggregatedAttributes:
        scored = [o for o in self.kept if o.attributes is not None]
        detector_type = self._detector_type()
        out = AggregatedAttributes(observations=len(scored), detector_type=detector_type)

        min_obs = int(getattr(settings, "vattr_min_observations", 2) or 1)

        if not scored:
            reason = "no_usable_crop"
            out.type_reason = out.color_reason = reason
            # A two-wheeler is still a detector-supported class even with no
            # attribute inference, so report it rather than discarding it.
            if detector_type in DETECTOR_ONLY_TYPES:
                out.vehicle_type = detector_type
                out.type_confidence = self._detector_confidence()
                out.type_agreement = 1.0
                out.type_source = "yolo_coco"
                out.type_reason = "detector_only_class"
            return out

        # One weight vector, computed once, used for both attributes so they
        # can never be driven by different crops of a multi-vehicle scene.
        weights = self._crop_weights(scored)
        type_agg, type_agree = self._weighted(scored, "type", weights)
        color_agg, color_agree = self._weighted(scored, "color", weights)
        out.type_probs = {c: round(float(p), 6) for c, p in zip(TYPE_CLASSES, type_agg)}
        out.color_probs = {c: round(float(p), 6) for c, p in zip(COLOR_CLASSES, color_agg)}

        self._decide_type(out, scored, type_agg, type_agree, detector_type, min_obs)
        self._decide_color(out, color_agg, color_agree, min_obs)
        return out

    def _detector_confidence(self) -> float:
        if not self.kept:
            return 0.0
        return float(max(o.detector_confidence for o in self.kept))

    def _crop_weights(self, scored: list[CropObservation]) -> np.ndarray:
        """One weight per crop, shared by BOTH attributes.

        This used to be computed per attribute as
        ``quality.score * probs.max()``. Because ``probs.max()`` differs between
        the type head and the colour head, type and colour were weighted
        differently over the same crops -- so in dense traffic the type could be
        dominated by the frame where one vehicle filled the crop while the
        colour was dominated by a frame showing its neighbour. That is how a
        single record ended up with one vehicle's type and another's colour.

        A single shared weight makes both attributes describe the same
        evidence. Model confidence still enters through the averaged
        probability vectors themselves, and through the mean peak below.
        """
        weights = []
        for obs in scored:
            type_peak = float(np.asarray(obs.attributes.type_probs, dtype=np.float64).max())
            color_peak = float(np.asarray(obs.attributes.color_probs, dtype=np.float64).max())
            confidence = 0.5 * (type_peak + color_peak)
            weights.append(max(1e-6, float(obs.quality.score)) * max(1e-6, confidence))
        return np.asarray(weights, dtype=np.float64)

    def _weighted(
        self,
        scored: list[CropObservation],
        kind: str,
        weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Quality x model-confidence weighted mean of probability vectors."""
        vectors = [
            np.asarray(obs.attributes.type_probs if kind == "type" else obs.attributes.color_probs,
                       dtype=np.float64)
            for obs in scored
        ]
        stack = np.vstack(vectors)
        w = self._crop_weights(scored) if weights is None else weights
        aggregated = (stack * w[:, None]).sum(axis=0) / w.sum()
        total = aggregated.sum()
        if total > 0:
            aggregated = aggregated / total
        winner = int(np.argmax(aggregated))
        agreement = float(np.mean([int(np.argmax(v)) == winner for v in stack]))
        return aggregated, agreement

    def _decide_type(
        self,
        out: AggregatedAttributes,
        scored: list[CropObservation],
        probs: np.ndarray,
        agreement: float,
        detector_type: str,
        min_obs: int,
    ) -> None:
        out.type_agreement = agreement
        out.type_source = model_id()

        # The attribute model has no two-wheeler class. Trust the detector for
        # that one class and do not let a 4-class head overwrite it.
        if detector_type in DETECTOR_ONLY_TYPES:
            out.vehicle_type = detector_type
            out.type_confidence = self._detector_confidence()
            out.type_agreement = 1.0
            out.type_source = "yolo_coco"
            out.type_reason = "detector_only_class"
            return

        idx, first, second = _top_two(probs)
        label = TYPE_CLASSES[idx]
        out.type_rejected_label, out.type_rejected_prob = label, first

        min_prob = float(getattr(settings, "vattr_type_min_prob", 0.60) or 0.0)
        min_margin = float(getattr(settings, "vattr_type_min_margin", 0.15) or 0.0)
        min_agreement = float(getattr(settings, "vattr_type_min_agreement", 0.60) or 0.0)
        conflict_min = float(getattr(settings, "vattr_type_conflict_min_prob", 0.75) or 0.0)

        if len(scored) < min_obs:
            out.abstain_type("too_few_observations")
            return
        if first < min_prob:
            out.abstain_type("below_min_probability")
            return
        if (first - second) < min_margin:
            out.abstain_type("top_two_margin_too_small")
            return
        if agreement < min_agreement:
            out.abstain_type("temporal_disagreement")
            return
        # Detector/classifier conflict on a class the detector is good at.
        #
        # Measured on this deployment: COCO YOLO calls Indian city buses `bus`
        # confidently, while this barrier-trained classifier gives them a `bus`
        # probability of ~0.002 and says `truck` at 0.99. The vendor's own
        # figures already flag `bus` as its weakest class (68.57%). A high
        # classifier probability is therefore NOT evidence it is right here, so
        # the default refuses to let either unvalidated model win.
        if detector_type in DETECTOR_STRONG_TYPES and label != detector_type:
            policy = str(getattr(settings, "vattr_conflict_policy", "abstain") or "abstain").lower()
            if policy == "detector":
                out.vehicle_type = detector_type
                out.type_confidence = self._detector_confidence()
                out.type_source = "yolo_coco"
                out.type_reason = "detector_wins_conflict"
                out.type_rejected_label, out.type_rejected_prob = label, first
                return
            if policy == "classifier":
                if first < conflict_min:
                    out.abstain_type("detector_classifier_conflict")
                    return
            else:  # "abstain" -- the default
                out.abstain_type("detector_classifier_conflict")
                return

        out.vehicle_type = label
        out.type_confidence = first
        out.type_reason = "accepted"
        out.type_rejected_label, out.type_rejected_prob = "", 0.0

    def _decide_color(
        self,
        out: AggregatedAttributes,
        probs: np.ndarray,
        agreement: float,
        min_obs: int,
    ) -> None:
        out.color_agreement = agreement
        out.color_source = model_id()
        idx, first, second = _top_two(probs)
        out.color_rejected_label, out.color_rejected_prob = COLOR_CLASSES[idx], first

        # A barrier-trained 4-class car model is out of distribution on a
        # motorcycle; its colour output there is not evidence.
        if out.detector_type in DETECTOR_ONLY_TYPES and not bool(
            getattr(settings, "vattr_two_wheeler_color_enabled", False)
        ):
            out.abstain_color("two_wheeler_out_of_distribution")
            return

        min_prob = float(getattr(settings, "vattr_color_min_prob", 0.55) or 0.0)
        min_margin = float(getattr(settings, "vattr_color_min_margin", 0.12) or 0.0)
        min_agreement = float(getattr(settings, "vattr_color_min_agreement", 0.60) or 0.0)

        if out.observations < min_obs:
            out.abstain_color("too_few_observations")
            return
        if first < min_prob:
            out.abstain_color("below_min_probability")
            return
        if (first - second) < min_margin:
            out.abstain_color("top_two_margin_too_small")
            return
        if agreement < min_agreement:
            out.abstain_color("temporal_disagreement")
            return

        # COLOR_CLASSES is the model's own vocabulary; 'silver'/'brown'/'orange'
        # are not in it and are therefore unreachable rather than guessed.
        out.vehicle_color = COLOR_CLASSES[idx]
        out.color_confidence = first
        out.color_reason = "accepted"
        out.color_rejected_label, out.color_rejected_prob = "", 0.0
