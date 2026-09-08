"""Small IoU/centroid plate tracker; it never tracks people or identities."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.config import settings


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    overlap = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - overlap
    return float(overlap / union) if union > 0 else 0.0


def _center_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    distance = math.hypot((ax + aw / 2) - (bx + bw / 2), (ay + ah / 2) - (by + bh / 2))
    return distance <= 0.5 * max(math.hypot(aw, ah), math.hypot(bw, bh))


@dataclass
class PlateTrack:
    track_id: str
    box: tuple[int, int, int, int]
    last_pts_ms: float | None
    best: list[tuple[float, object]] = field(default_factory=list)

    def add(self, score: float, candidate: object) -> None:
        self.best.append((float(score), candidate))
        self.best.sort(key=lambda item: item[0], reverse=True)
        del self.best[max(1, int(settings.plate_track_best_crops)) :]


class PlateTrackManager:
    def __init__(self, camera_id: str, run_id: str):
        self.camera_id = camera_id
        self.run_id = run_id
        self.serial = 0
        self.tracks: dict[str, PlateTrack] = {}

    def assign(self, box: tuple[int, int, int, int] | None, pts_ms: float | None, candidate=None) -> str:
        if box is None:
            return f"{self.camera_id}-{self.run_id}-unlocated"
        max_gap = float(settings.plate_track_max_gap_seconds) * 1000.0
        best_id, best_iou = None, -1.0
        for track_id, track in list(self.tracks.items()):
            if pts_ms is not None and track.last_pts_ms is not None and pts_ms - track.last_pts_ms > max_gap:
                del self.tracks[track_id]
                continue
            overlap = box_iou(track.box, box)
            if overlap >= settings.plate_track_iou_threshold or _center_close(track.box, box):
                if overlap > best_iou:
                    best_id, best_iou = track_id, overlap
        if best_id is None:
            self.serial += 1
            best_id = f"{self.camera_id}-{self.run_id}-t{self.serial}"
            self.tracks[best_id] = PlateTrack(best_id, box, pts_ms)
        track = self.tracks[best_id]
        track.box, track.last_pts_ms = box, pts_ms
        if candidate is not None:
            quality = candidate.get("quality", {}) if isinstance(candidate, dict) else getattr(candidate, "quality", {})
            score = float((quality or {}).get("score", 0.0))
            track.add(score, candidate)
        return best_id

    def reset(self) -> None:
        self.tracks.clear()
