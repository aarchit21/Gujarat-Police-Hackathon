"""Vehicle tracking behind one interface.

Primary path is Ultralytics ByteTrack (``model.track(persist=True)``), which
needs ``lap`` for its linear-assignment step.  When ``lap`` is absent we fall
back to a two-band greedy IoU associator built on the existing
``plate_tracking`` helpers.

The fallback is **not** ByteTrack and is never reported as such: it has no
Kalman motion model, so it is weaker through occlusion and fast motion.
``tracker_source`` always says which one actually ran.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.services.plate_tracking import box_iou
from app.services.yolo_detect import VEHICLE_CLASS_IDS, VEHICLE_TYPE_BY_ID, VehicleDet, _crop


@dataclass
class TrackedDetection:
    """A detection with a persistent track id attached."""

    track_id: str
    detection: VehicleDet
    box: tuple[int, int, int, int]  # x, y, w, h in the coordinate space given


def bytetrack_available() -> tuple[bool, str]:
    try:
        import lap  # noqa: F401
    except ImportError as exc:
        return False, f"lap not installed ({exc})"
    try:
        import ultralytics  # noqa: F401
    except ImportError as exc:
        return False, f"ultralytics not installed ({exc})"
    return True, ""


class GreedyIouTracker:
    """Two-band greedy IoU associator. A fallback, not ByteTrack."""

    tracker_source = "iou_fallback"

    def __init__(self, camera_id: str, run_id: str):
        self.camera_id = camera_id
        self.run_id = run_id
        self.serial = 0
        self.tracks: dict[str, dict] = {}

    def _new_track(self, box, pts_ms) -> str:
        self.serial += 1
        track_id = f"{self.camera_id}-{self.run_id}-t{self.serial}"
        self.tracks[track_id] = {"box": box, "last_pts_ms": pts_ms, "misses": 0}
        return track_id

    def update(self, detections: list[VehicleDet], pts_ms: float | None) -> list[TrackedDetection]:
        max_gap = float(settings.plate_track_max_gap_seconds) * 1000.0
        for track_id, track in list(self.tracks.items()):
            if pts_ms is not None and track["last_pts_ms"] is not None and pts_ms - track["last_pts_ms"] > max_gap:
                del self.tracks[track_id]

        high_band = float(getattr(settings, "vattr_track_high_conf", 0.50) or 0.5)
        strong_iou = float(settings.plate_track_iou_threshold)
        weak_iou = float(getattr(settings, "vattr_track_low_iou", 0.20) or 0.2)

        boxes = [(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1) for d in detections]
        order = sorted(range(len(detections)), key=lambda i: -float(detections[i].confidence or 0.0))
        assigned: dict[int, str] = {}
        used: set[str] = set()

        # Pass 1 high-confidence detections at the strict IoU, pass 2 the rest
        # at a looser IoU. That two-band idea is borrowed from ByteTrack; the
        # motion model is not.
        for band_strict in (True, False):
            for i in order:
                if i in assigned:
                    continue
                conf = float(detections[i].confidence or 0.0)
                if band_strict != (conf >= high_band):
                    continue
                threshold = strong_iou if band_strict else weak_iou
                best_id, best_overlap = None, threshold
                for track_id, track in self.tracks.items():
                    if track_id in used:
                        continue
                    overlap = box_iou(track["box"], boxes[i])
                    if overlap >= best_overlap:
                        best_id, best_overlap = track_id, overlap
                if best_id is not None:
                    assigned[i] = best_id
                    used.add(best_id)

        out: list[TrackedDetection] = []
        for i, det in enumerate(detections):
            track_id = assigned.get(i) or self._new_track(boxes[i], pts_ms)
            self.tracks[track_id]["box"] = boxes[i]
            self.tracks[track_id]["last_pts_ms"] = pts_ms
            out.append(TrackedDetection(track_id=track_id, detection=det, box=boxes[i]))
        return out

    def reset(self) -> None:
        self.tracks.clear()


class ByteTrackTracker:
    """Ultralytics ByteTrack/BoT-SORT, driven by ``model.track(persist=True)``."""

    def __init__(self, camera_id: str, run_id: str, *, tracker_cfg: str | None = None):
        self.camera_id = camera_id
        self.run_id = run_id
        self.tracker_cfg = tracker_cfg or (getattr(settings, "vattr_tracker", "bytetrack.yaml") or "bytetrack.yaml")
        self.tracker_source = "bytetrack" if "byte" in self.tracker_cfg else "botsort"

    def track_frame(
        self,
        model,
        frame: np.ndarray,
        pts_ms: float | None,
        *,
        device: str,
        imgsz: int,
        precision: dict | None = None,
    ) -> list[TrackedDetection]:
        results = model.track(
            frame,
            persist=True,
            tracker=self.tracker_cfg,
            conf=float(settings.yolo_conf or 0.25),
            iou=float(getattr(settings, "yolo_iou", 0.45) or 0.45),
            classes=sorted(VEHICLE_CLASS_IDS),
            imgsz=int(imgsz),
            device=device,
            verbose=False,
            **(precision or {}),
        )
        if not results:
            return []
        return self._convert(results[0], frame)

    def _convert(self, result, frame: np.ndarray) -> list[TrackedDetection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return []

        def to_numpy(value):
            if value is None:
                return None
            return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)

        xyxy = to_numpy(boxes.xyxy)
        cls = to_numpy(getattr(boxes, "cls", None))
        conf = to_numpy(getattr(boxes, "conf", None))
        ids = to_numpy(getattr(boxes, "id", None))

        out: list[TrackedDetection] = []
        for i, row in enumerate(xyxy):
            class_id = int(cls[i]) if cls is not None and i < len(cls) else -1
            if class_id not in VEHICLE_CLASS_IDS:
                continue
            # No ByteTrack id yet means the track is unconfirmed. Skipping it
            # keeps one observation per *confirmed* track.
            if ids is None or i >= len(ids) or ids[i] is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in row[:4])
            score = float(conf[i]) if conf is not None and i < len(conf) else 0.0
            crop, crop_box = _crop(frame, x1, y1, x2, y2)
            det = VehicleDet(
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                confidence=score,
                vehicle_type=VEHICLE_TYPE_BY_ID.get(class_id, "unknown"),
                crop=crop,
                crop_box=crop_box,
            )
            out.append(
                TrackedDetection(
                    track_id=f"{self.camera_id}-{self.run_id}-t{int(ids[i])}",
                    detection=det,
                    box=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                )
            )
        return out

    def reset(self) -> None:  # pragma: no cover - ultralytics owns the state
        pass


def build_tracker(camera_id: str, run_id: str) -> tuple[object, str, str]:
    """Return (tracker, tracker_source, note). Never silently misreports which one."""
    if bool(getattr(settings, "vattr_force_iou_tracker", False)):
        return GreedyIouTracker(camera_id, run_id), "iou_fallback", "forced by vattr_force_iou_tracker"
    ok, why = bytetrack_available()
    if ok:
        tracker = ByteTrackTracker(camera_id, run_id)
        return tracker, tracker.tracker_source, ""
    return (
        GreedyIouTracker(camera_id, run_id),
        "iou_fallback",
        f"ByteTrack unavailable: {why}. Using greedy IoU fallback (no motion model).",
    )
