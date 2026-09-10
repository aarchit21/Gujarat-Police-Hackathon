from __future__ import annotations

import numpy as np

from app.services.cpu_anpr import CpuPlateCandidate, _union_candidates, localization_gate
from app.services.lpdnet import _nms, _postprocess_detectnet, detect_plates, lpdnet_ready
from app.services.yolo_detect import expand_vehicle_box, detect_vehicles


def test_expanded_vehicle_box_includes_pixels_below_yolo():
    frame = np.zeros((200, 200, 3), np.uint8)
    xa, ya, xb, yb = expand_vehicle_box(frame.shape, 40, 40, 120, 120)
    assert ya <= 40
    assert yb > 120
    assert xb - xa >= 80


def test_clipped_plate_is_inside_expanded_crop_not_tight_box():
    frame = np.zeros((180, 240, 3), dtype=np.uint8)
    frame[100:115, 50:110] = 200
    xa, ya, xb, yb = expand_vehicle_box(frame.shape, 40, 20, 120, 100)
    tight = frame[20:100, 40:120]
    expanded = frame[ya:yb, xa:xb]
    assert tight.shape[0] == 80
    assert ya <= 100 < yb
    assert expanded.max() == 200


def test_lpdnet_without_weights_returns_empty(monkeypatch, tmp_path):
    from app.config import settings

    monkeypatch.setattr(settings, "lpdnet_weights_dir", str(tmp_path / "missing"))
    assert lpdnet_ready() is False
    assert detect_plates(np.zeros((80, 120, 3), np.uint8)) == []


def test_detectnet_postprocess_and_nms():
    cov = np.zeros((1, 3, 30, 40), np.float32)
    bbox = np.zeros((1, 12, 30, 40), np.float32)
    cov[0, 0, 10, 20] = 0.9
    bbox[0, 0, 10, 20] = 20
    bbox[0, 1, 10, 20] = 10
    bbox[0, 2, 10, 20] = 20
    bbox[0, 3, 10, 20] = 10
    boxes = _postprocess_detectnet(cov, bbox, conf=0.3, src_w=640, src_h=480)
    assert boxes
    assert boxes[0][4] >= 0.3
    merged = _nms(boxes + boxes, iou_thresh=0.5)
    assert len(merged) == 1


def test_union_keeps_non_overlapping_lpdnet_box():
    a = CpuPlateCandidate(box=(10, 10, 40, 12), detector="fast_alpr", detector_confidence=0.8)
    b = CpuPlateCandidate(box=(80, 40, 40, 12), detector="lpdnet", detector_confidence=0.7)
    out = _union_candidates([a], [b])
    assert len(out) == 2
    assert localization_gate(
        CpuPlateCandidate(
            crop_bgr=np.zeros((16, 48, 3), np.uint8),
            detector="lpdnet",
            detector_confidence=0.8,
        )
    ) is True


def test_yolo_crop_uses_expanded_origin():
    frame = np.zeros((180, 240, 3), dtype=np.uint8)

    class _Arr:
        def __init__(self, data):
            self._data = np.asarray(data, dtype=float)

        def cpu(self):
            return self

        def numpy(self):
            return self._data

    class _Boxes:
        xyxy = _Arr([[40, 20, 120, 100]])
        cls = _Arr([2])
        conf = _Arr([0.9])

    class _Result:
        boxes = _Boxes()

    dets = detect_vehicles(frame, predict_fn=lambda _f: [_Result()])
    assert dets
    cx, cy, cw, ch = dets[0].crop_box
    assert cy + ch > 100
    assert dets[0].crop.shape[0] == ch
