import numpy as np

from app.services.anpr import anpr_crops
from app.services.yolo_detect import VEHICLE_CLASS_IDS, detect_vehicles, yolo_status


class _Arr:
    def __init__(self, data):
        self._data = np.asarray(data, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class _Boxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = _Arr(xyxy)
        self.cls = _Arr(cls)
        self.conf = _Arr(conf)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


def test_yolo_drops_person_keeps_car():
    frame = np.zeros((180, 240, 3), dtype=np.uint8)

    def predict(_frame):
        return [
            _Result(
                _Boxes(
                    xyxy=[[10, 20, 120, 100], [0, 0, 40, 80]],
                    cls=[2, 0],
                    conf=[0.91, 0.99],
                )
            )
        ]

    dets = detect_vehicles(frame, predict_fn=predict)
    assert len(dets) == 1
    assert dets[0].vehicle_type == "car"
    assert 0 not in VEHICLE_CLASS_IDS
    assert dets[0].crop.size > 0


def test_yolo_status_does_not_require_weights():
    status = yolo_status()
    assert "person" in status["ignores"]
    assert "car" in status["classes"]


def test_anpr_crops_uses_yolo_when_present(monkeypatch):
    frame = np.zeros((180, 240, 3), dtype=np.uint8)

    def fake_detect(_bgr):
        from app.services.yolo_detect import VehicleDet

        crop = frame[20:100, 10:120].copy()
        return [VehicleDet(10, 20, 120, 100, 0.9, "truck", crop)]

    monkeypatch.setattr("app.services.yolo_detect.detect_vehicles", fake_detect)
    crops = anpr_crops(frame, live=True)
    assert crops[0]["detector"] == "yolov8n"
    assert crops[0]["vehicle_type"] == "truck"
