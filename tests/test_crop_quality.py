"""Crop-quality gate: the cases that must be refused."""
from __future__ import annotations

import cv2
import numpy as np

from app.config import settings
from app.services.crop_quality import score_crop, visible_fraction


def _textured(width: int, height: int, level: int = 128) -> np.ndarray:
    """A sharp, well-exposed crop that should pass every gate."""
    rng = np.random.default_rng(0)
    noise = rng.integers(-60, 60, size=(height, width, 3))
    return np.clip(noise + level, 0, 255).astype(np.uint8)


def test_good_crop_is_eligible():
    quality = score_crop(_textured(200, 200), box=(0, 0, 200, 200), frame_shape=(480, 640, 3), detector_confidence=0.8)
    assert quality.eligible
    assert quality.reason == ""
    assert quality.score > 0


def test_empty_crop_rejected():
    quality = score_crop(None)
    assert not quality.eligible
    assert quality.reason == "empty_crop"


def test_too_small_crop_rejected():
    quality = score_crop(_textured(20, 20), box=(0, 0, 20, 20), frame_shape=(480, 640, 3), detector_confidence=0.9)
    assert not quality.eligible
    assert quality.reason == "crop_too_small"


def test_clipped_at_frame_edge_rejected():
    """Half the vehicle outside the frame is not a representative body view."""
    quality = score_crop(
        _textured(200, 200), box=(-150, 10, 200, 200), frame_shape=(480, 640, 3), detector_confidence=0.9
    )
    assert not quality.eligible
    assert quality.reason == "clipped_at_frame_edge"


def test_blurred_crop_rejected():
    blurred = cv2.GaussianBlur(_textured(200, 200), (31, 31), 0)
    quality = score_crop(blurred, box=(0, 0, 200, 200), frame_shape=(480, 640, 3), detector_confidence=0.9)
    assert not quality.eligible
    assert quality.reason == "blurred"


def test_underexposed_crop_rejected():
    dark = np.full((200, 200, 3), 5, dtype=np.uint8)
    quality = score_crop(dark, box=(0, 0, 200, 200), frame_shape=(480, 640, 3), detector_confidence=0.9)
    assert not quality.eligible
    assert quality.reason in {"blurred", "underexposed"}


def test_overexposed_crop_rejected():
    rng = np.random.default_rng(1)
    blown = np.clip(rng.integers(-40, 40, size=(200, 200, 3)) + 252, 0, 255).astype(np.uint8)
    quality = score_crop(blown, box=(0, 0, 200, 200), frame_shape=(480, 640, 3), detector_confidence=0.9)
    assert not quality.eligible
    assert quality.reason == "overexposed"


def test_low_detector_confidence_rejected():
    quality = score_crop(
        _textured(200, 200), box=(0, 0, 200, 200), frame_shape=(480, 640, 3), detector_confidence=0.05
    )
    assert not quality.eligible
    assert quality.reason == "low_detector_confidence"


def test_visible_fraction_math():
    assert visible_fraction((0, 0, 100, 100), (480, 640, 3)) == 1.0
    assert visible_fraction((-50, 0, 100, 100), (480, 640, 3)) == 0.5
    assert visible_fraction((-200, 0, 100, 100), (480, 640, 3)) == 0.0
    assert visible_fraction((0, 0, 0, 0), (480, 640, 3)) == 0.0


def test_thresholds_are_configurable(monkeypatch):
    small = _textured(30, 30)
    assert score_crop(small, detector_confidence=0.9).reason == "crop_too_small"
    monkeypatch.setattr(settings, "vattr_min_crop_px", 10)
    assert score_crop(small, detector_confidence=0.9).reason != "crop_too_small"


def test_bigger_and_sharper_scores_higher():
    small = score_crop(_textured(60, 60), box=(0, 0, 60, 60), frame_shape=(480, 640, 3), detector_confidence=0.9)
    large = score_crop(_textured(300, 300), box=(0, 0, 300, 300), frame_shape=(480, 640, 3), detector_confidence=0.9)
    assert large.score > small.score


# -- multi-vehicle crops (dense traffic) --------------------------------


def test_foreign_fraction_measures_only_other_vehicles():
    from app.services.crop_quality import foreign_fraction

    crop = (0, 0, 100, 100)
    subject = (0, 0, 50, 100)
    # The subject's own box never counts against it.
    assert foreign_fraction(crop, subject, [subject]) == 0.0
    # A neighbour covering the right half of the crop is 50%.
    assert foreign_fraction(crop, subject, [subject, (50, 0, 50, 100)]) == 0.5
    # Overlapping neighbours are capped at the crop area, never above 1.
    assert foreign_fraction(crop, subject, [(0, 0, 100, 100), (0, 0, 100, 100)]) == 1.0
    assert foreign_fraction(crop, subject, None) == 0.0


def test_crop_containing_another_vehicle_is_rejected():
    """Measured on cam01: 20% of crops held >25% of a neighbouring vehicle."""
    quality = score_crop(
        _textured(200, 200),
        box=(0, 0, 100, 200),
        frame_shape=(480, 640, 3),
        detector_confidence=0.9,
        crop_box=(0, 0, 200, 200),
        other_boxes=[(0, 0, 100, 200), (100, 0, 100, 200)],  # neighbour fills half
    )
    assert not quality.eligible
    assert quality.reason == "multi_vehicle_crop"
    assert quality.foreign_fraction == 0.5


def test_clean_crop_in_a_busy_frame_still_passes():
    """A nearby vehicle that does not intrude on the crop must not reject it."""
    quality = score_crop(
        _textured(200, 200),
        box=(0, 0, 200, 200),
        frame_shape=(480, 640, 3),
        detector_confidence=0.9,
        crop_box=(0, 0, 200, 200),
        other_boxes=[(0, 0, 200, 200), (400, 0, 100, 200)],
    )
    assert quality.eligible
    assert quality.foreign_fraction == 0.0


def test_contamination_lowers_the_ranking_score_even_when_allowed(monkeypatch):
    monkeypatch.setattr(settings, "vattr_max_foreign_fraction", 1.0)
    common = dict(box=(0, 0, 200, 200), frame_shape=(480, 640, 3), detector_confidence=0.9,
                  crop_box=(0, 0, 200, 200))
    clean = score_crop(_textured(200, 200), **common, other_boxes=[(0, 0, 200, 200)])
    dirty = score_crop(_textured(200, 200), **common,
                       other_boxes=[(0, 0, 200, 200), (100, 0, 100, 200)])
    assert dirty.eligible and clean.eligible
    assert dirty.score < clean.score


def test_foreign_threshold_is_configurable(monkeypatch):
    args = dict(box=(0, 0, 100, 200), frame_shape=(480, 640, 3), detector_confidence=0.9,
                crop_box=(0, 0, 200, 200), other_boxes=[(0, 0, 100, 200), (100, 0, 100, 200)])
    monkeypatch.setattr(settings, "vattr_max_foreign_fraction", 0.6)
    assert score_crop(_textured(200, 200), **args).eligible
    monkeypatch.setattr(settings, "vattr_max_foreign_fraction", 0.4)
    assert not score_crop(_textured(200, 200), **args).eligible
