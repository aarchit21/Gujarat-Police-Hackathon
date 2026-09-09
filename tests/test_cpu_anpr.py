import numpy as np

import app.services.cpu_anpr as cpu_anpr
from app.services.cpu_anpr import (
    CpuPlateCandidate,
    aligned_median_fusion,
    confidence_gate,
    detect_with_fast_alpr,
    localization_gate,
    plate_quality,
    rectify_plate,
)
from app.services.plate_tracking import PlateTrackManager, box_iou


def _textured_plate(width=160, height=40):
    crop = np.full((height, width, 3), 210, np.uint8)
    crop[:, ::8] = 20
    return crop


def test_plate_quality_rejects_missing_and_tiny_pixels():
    assert plate_quality(None).reason == "no_plate_candidate"
    tiny = plate_quality(np.zeros((8, 20, 3), np.uint8))
    assert tiny.eligible is False
    assert tiny.reason == "insufficient_pixels"


def test_fast_alpr_rereads_upscaled_crop_when_native_ocr_is_junk(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "cpu_anpr_ocr_min_width", 100)
    monkeypatch.setattr(settings, "vision_enhancement_enabled", True)
    frame = np.zeros((80, 200, 3), np.uint8)
    frame[20:40, 20:71] = 200

    class OCR:
        def predict(self, crop):
            if crop.shape[1] >= 100:
                return {"text": "GJ08AV5178", "confidence": [0.9] * 10}
            return {"text": "6DDANT7", "confidence": [0.6] * 7}

    class Predictor:
        ocr = OCR()

        def predict(self, _image):
            return [{
                "detection": {"bounding_box": {"x1": 20, "y1": 20, "x2": 71, "y2": 40}, "confidence": 0.7},
                "ocr": {"text": "6DDANT7", "confidence": [0.6] * 7},
            }]

    rows = detect_with_fast_alpr(frame, predictor=Predictor())
    assert len(rows) == 1
    assert rows[0].plate_norm == "GJ08AV5178"
    assert rows[0].recognizer == "fast_plate_ocr_enhanced"


def test_onnx_providers_cpu_override(monkeypatch):
    from app.config import settings
    from app.services.cpu_anpr import onnx_execution_providers

    monkeypatch.setattr(settings, "cpu_anpr_device", "cpu")
    assert onnx_execution_providers()[0] == "CPUExecutionProvider"


def test_fast_alpr_adapter_preserves_raw_and_character_confidence():
    frame = np.zeros((120, 300, 3), np.uint8)
    frame[40:80, 60:220] = _textured_plate()

    class Predictor:
        def predict(self, _image):
            return [{
                "detection": {"bounding_box": {"x1": 60, "y1": 40, "x2": 220, "y2": 80}},
                "ocr": {"text": "GJ01AB1234", "confidence": [0.91] * 10},
            }]

    rows = detect_with_fast_alpr(frame, predictor=Predictor())
    assert len(rows) == 1
    assert rows[0].plate_raw == "GJ01AB1234"
    assert rows[0].box == (60, 40, 160, 40)
    assert confidence_gate(rows[0]) is True
    assert localization_gate(rows[0]) is True


def test_live_candidate_path_never_falls_back_to_colour_contours(monkeypatch):
    frame = np.zeros((120, 300, 3), np.uint8)
    fake_contour = CpuPlateCandidate(
        crop_bgr=_textured_plate(), detector="opencv_plate", recognizer="tesseract",
        quality={"eligible": True, "score": 1.0},
    )
    monkeypatch.setattr(cpu_anpr, "detect_with_fast_alpr", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cpu_anpr, "opencv_tesseract_candidates", lambda _image: [fake_contour])

    assert cpu_anpr.cpu_plate_candidates(frame, allow_opencv_fallback=False) == []
    assert cpu_anpr.cpu_plate_candidates(frame, allow_opencv_fallback=True) == [fake_contour]
    assert localization_gate(fake_contour) is False


def test_perspective_rectification_and_guarded_median_fusion():
    crop = _textured_plate()
    warped = rectify_plate(crop, [(4, 3), (155, 1), (158, 37), (2, 39)])
    assert warped.size > 0
    assert warped is not crop
    fused = aligned_median_fusion([crop, crop.copy()])
    assert fused is not None
    assert fused.shape == crop.shape
    assert aligned_median_fusion([crop]) is None


def test_plate_tracker_associates_nearby_boxes_and_splits_distant_ones():
    manager = PlateTrackManager("cam", "run")
    first = manager.assign((10, 10, 100, 30), 0.0, {"quality": {"score": 0.5}})
    second = manager.assign((15, 12, 100, 30), 200.0, {"quality": {"score": 0.8}})
    distant = manager.assign((500, 300, 100, 30), 300.0)
    assert first == second
    assert distant != first
    assert len(manager.tracks[first].best) == 2
    assert box_iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
