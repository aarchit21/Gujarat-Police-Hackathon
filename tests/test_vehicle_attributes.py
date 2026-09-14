"""Vehicle attribute model: class-order pinning, preprocessing, fail-loud."""
from __future__ import annotations

import numpy as np
import pytest

from app.services import vehicle_attributes as va


def test_class_orders_are_the_model_output_order():
    """Pin the README output order.

    The published accuracy-check.yml carries an ALPHABETICAL label_map
    (bus, car, truck, van / black, blue, gray, green, red, white, yellow).
    That is not the output index order. Using it silently swaps car and bus.
    """
    assert va.TYPE_CLASSES == ("car", "van", "truck", "bus")
    assert va.COLOR_CLASSES == ("white", "gray", "yellow", "red", "green", "blue", "black")
    # Guard against someone "sorting" them.
    assert list(va.TYPE_CLASSES) != sorted(va.TYPE_CLASSES)
    assert list(va.COLOR_CLASSES) != sorted(va.COLOR_CLASSES)


def test_unsupported_classes_are_absent_from_the_vocabulary():
    for label in ("suv", "auto_rickshaw", "taxi_cab"):
        assert label not in va.TYPE_CLASSES
    for label in ("silver", "brown", "orange", "other"):
        assert label not in va.COLOR_CLASSES


def test_gray_is_a_real_class_and_silver_is_not():
    """gray must never be rewritten to silver: the model cannot tell them apart."""
    assert "gray" in va.COLOR_CLASSES
    assert "silver" not in va.COLOR_CLASSES


def test_preprocess_shape_dtype_and_no_normalization():
    crop = np.full((120, 90, 3), 200, dtype=np.uint8)
    tensor = va.preprocess(crop)
    assert tensor.shape == (1, 3, va.INPUT_SIZE, va.INPUT_SIZE)
    assert tensor.dtype == np.float32
    # Raw 0-255 range: a constant 200 crop must stay 200, not be scaled to ~0.78.
    assert np.allclose(tensor, 200.0)


def test_preprocess_preserves_bgr_channel_order():
    """A pure-blue BGR crop must stay on channel 0 after CHW transpose."""
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[:, :, 0] = 255  # blue in BGR
    tensor = va.preprocess(crop)
    assert tensor[0, 0].mean() == pytest.approx(255.0)
    assert tensor[0, 1].mean() == pytest.approx(0.0)
    assert tensor[0, 2].mean() == pytest.approx(0.0)


def test_preprocess_rejects_empty_crop():
    with pytest.raises(ValueError):
        va.preprocess(np.zeros((0, 0, 3), dtype=np.uint8))


def test_extend_box_is_symmetric_and_clipped():
    box = (100, 100, 100, 100)
    x1, y1, x2, y2 = va.extend_box(box, (480, 640, 3), extend=0.30)
    assert (x1, y1, x2, y2) == (70, 70, 230, 230)
    # Clipped at the frame edge rather than running negative.
    x1, y1, x2, y2 = va.extend_box((0, 0, 50, 50), (480, 640, 3), extend=0.5)
    assert x1 == 0 and y1 == 0


def test_extend_box_does_not_use_the_plate_padding():
    """yolo_detect pads the bottom by 0.45 for bumper plates; this must not."""
    box = (100, 100, 100, 100)
    x1, y1, x2, y2 = va.extend_box(box, (480, 640, 3), extend=0.30)
    top_pad, bottom_pad = 100 - y1, y2 - 200
    assert top_pad == bottom_pad


def test_missing_weights_raise_rather_than_substitute(monkeypatch, tmp_path):
    monkeypatch.setattr(va.settings, "vattr_weights", str(tmp_path / "absent"))
    monkeypatch.setattr(va, "_compiled", None)
    assert va.missing_weight_files()
    with pytest.raises(va.AttributeModelUnavailable) as excinfo:
        va.load_model()
    message = str(excinfo.value)
    assert "missing" in message
    assert "pull_vehicle_attributes" in message


def test_extract_rejects_wrong_output_sizes():
    with pytest.raises(va.AttributeModelUnavailable):
        va._extract({"type": np.zeros(9), "color": np.zeros(7)})


def test_extract_rejects_missing_outputs():
    with pytest.raises(va.AttributeModelUnavailable):
        va._extract({"type": np.zeros(4)})


def test_classify_crop_with_injected_inference_reports_pinned_labels():
    """car index 0 and bus index 3 must map by position, not alphabetically."""
    type_probs = np.array([0.90, 0.04, 0.03, 0.03], dtype=np.float32)
    color_probs = np.array([0.05, 0.05, 0.02, 0.80, 0.03, 0.03, 0.02], dtype=np.float32)

    def fake_infer(_tensor):
        return {"type": type_probs, "color": color_probs}

    result = va.classify_crop(np.full((80, 80, 3), 120, dtype=np.uint8), infer_fn=fake_infer)
    assert result.type_top[0] == "car"
    assert result.color_top[0] == "red"

    bus_first = np.array([0.05, 0.05, 0.05, 0.85], dtype=np.float32)
    result = va.classify_crop(
        np.full((80, 80, 3), 120, dtype=np.uint8),
        infer_fn=lambda _t: {"type": bus_first, "color": color_probs},
    )
    assert result.type_top[0] == "bus"


def test_attributes_status_never_raises_without_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(va.settings, "vattr_weights", str(tmp_path / "absent"))
    status = va.attributes_status()
    assert status["weights_present"] is False
    assert status["missing_files"]
    assert "suv" in status["unsupported_types"]
    assert "silver" in status["unsupported_colors"]
