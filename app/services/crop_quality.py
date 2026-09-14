"""Crop-quality scoring and rejection for vehicle attribute inference.

Deliberately mirrors the shape of ``cpu_anpr.plate_quality`` so the two gates
read the same way.  The point of this module is to *refuse* work: running a
classifier on a 20-pixel, motion-blurred, half-clipped crop produces a
confident-looking number with nothing behind it.

All thresholds come from configuration.  None of them are calibrated against
labelled data for this deployment -- they are conservative starting values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from app.config import settings


@dataclass
class CropQuality:
    width: int
    height: int
    sharpness: float
    exposure: float
    dark_fraction: float
    clipped_fraction: float
    visible_fraction: float
    detector_confidence: float
    score: float
    eligible: bool
    reason: str = ""
    # Fraction of the classifier's crop covered by OTHER detected vehicles.
    # Last, with a default, so existing constructors keep working.
    foreign_fraction: float = 0.0

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def foreign_fraction(
    crop_box: tuple[int, int, int, int],
    subject_box: tuple[int, int, int, int],
    other_boxes: list[tuple[int, int, int, int]] | None,
) -> float:
    """How much of the classifier's crop belongs to a different vehicle.

    Measured on cam01 night traffic: 243 of 276 crops came from frames holding
    more than one vehicle, 20% of crops contained >25% of a neighbouring
    vehicle, and 5% contained more foreign vehicle than subject. A classifier
    fed that crop is not describing the vehicle the track is about, which is how
    one vehicle's type ends up beside another's colour.

    Overlap with the subject's own box is excluded, and the result is capped at
    1.0 so several overlapping neighbours cannot exceed the crop area.
    """
    if not other_boxes:
        return 0.0
    cx, cy, cw, ch = (float(v) for v in crop_box)
    crop_area = max(1.0, cw * ch)

    def overlap(box: tuple[int, int, int, int]) -> float:
        bx, by, bw, bh = (float(v) for v in box)
        left, top = max(cx, bx), max(cy, by)
        right, bottom = min(cx + cw, bx + bw), min(cy + ch, by + bh)
        return max(0.0, right - left) * max(0.0, bottom - top)

    subject = tuple(int(v) for v in subject_box)
    foreign = sum(overlap(b) for b in other_boxes if tuple(int(v) for v in b) != subject)
    return float(min(1.0, foreign / crop_area))


def visible_fraction(box: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> float:
    """Fraction of the detector box actually inside the frame.

    A vehicle entering or leaving the scene is clipped at the boundary; its
    visible body is not representative, so it should not drive the attributes.
    """
    height, width = int(frame_shape[0]), int(frame_shape[1])
    x, y, w, h = (float(v) for v in box)
    if w <= 0 or h <= 0:
        return 0.0
    inside_w = max(0.0, min(x + w, width) - max(x, 0.0))
    inside_h = max(0.0, min(y + h, height) - max(y, 0.0))
    return float(max(0.0, min(1.0, (inside_w * inside_h) / (w * h))))


def _normalised_sharpness(value: float) -> float:
    """Map raw Laplacian variance onto 0..1 for a comparable composite score."""
    reference = max(1.0, float(getattr(settings, "vattr_sharpness_reference", 300.0) or 300.0))
    return float(max(0.0, min(1.0, value / reference)))


def score_crop(
    crop_bgr: np.ndarray | None,
    *,
    box: tuple[int, int, int, int] | None = None,
    frame_shape: tuple[int, ...] | None = None,
    detector_confidence: float = 0.0,
    crop_box: tuple[int, int, int, int] | None = None,
    other_boxes: list[tuple[int, int, int, int]] | None = None,
) -> CropQuality:
    """Score one vehicle crop and decide whether it may drive attribute inference.

    ``crop_box`` is the extended region actually handed to the classifier, and
    ``other_boxes`` every vehicle box in the same frame; together they measure
    how much of the crop belongs to a different vehicle.
    """
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return CropQuality(0, 0, 0.0, 0.0, 1.0, 0.0, 0.0, float(detector_confidence or 0.0), 0.0, False, "empty_crop")

    height, width = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_level = float(np.mean(gray))
    dark = float(np.mean(gray <= 28))
    clipped = float(np.mean(gray >= 245))
    seen = visible_fraction(box, frame_shape) if box and frame_shape else 1.0
    confidence = float(detector_confidence or 0.0)
    foreign = foreign_fraction(crop_box, box, other_boxes) if (crop_box and box) else 0.0

    # Exposure peaks at mid-grey and falls off towards crushed or blown-out.
    exposure = float(max(0.0, 1.0 - abs(mean_level - 128.0) / 128.0))

    min_px = int(getattr(settings, "vattr_min_crop_px", 40) or 40)
    min_sharp = float(getattr(settings, "vattr_min_sharpness", 25.0) or 0.0)
    min_visible = float(getattr(settings, "vattr_min_visible_fraction", 0.70) or 0.0)
    max_dark = float(getattr(settings, "vattr_max_dark_fraction", 0.60) or 1.0)
    max_clipped = float(getattr(settings, "vattr_max_clipped_fraction", 0.35) or 1.0)
    min_conf = float(getattr(settings, "vattr_min_detector_confidence", 0.30) or 0.0)
    max_foreign = float(getattr(settings, "vattr_max_foreign_fraction", 0.25) or 1.0)

    reason = ""
    if width < min_px or height < min_px:
        reason = "crop_too_small"
    elif seen < min_visible:
        reason = "clipped_at_frame_edge"
    elif foreign > max_foreign:
        # Another vehicle occupies too much of this crop to attribute anything
        # to the tracked one.
        reason = "multi_vehicle_crop"
    elif confidence < min_conf:
        reason = "low_detector_confidence"
    elif sharpness < min_sharp:
        reason = "blurred"
    elif dark > max_dark:
        reason = "underexposed"
    elif clipped > max_clipped:
        reason = "overexposed"

    # Composite score ranks *eligible* crops against each other; it is not a
    # probability and is never reported as a confidence.
    area_term = min(1.0, (width * height) / float(max(1, min_px * min_px * 16)))
    score = float(
        area_term
        * max(0.05, confidence)
        * max(0.05, _normalised_sharpness(sharpness))
        * max(0.05, exposure)
        * max(0.05, seen)
        # A crop that is partly someone else's vehicle ranks below a clean one
        # even when it survives the gate.
        * max(0.05, 1.0 - foreign)
    )

    return CropQuality(
        width=int(width),
        height=int(height),
        sharpness=sharpness,
        exposure=exposure,
        dark_fraction=dark,
        clipped_fraction=clipped,
        visible_fraction=seen,
        detector_confidence=confidence,
        foreign_fraction=foreign,
        score=score,
        eligible=not reason,
        reason=reason,
    )
