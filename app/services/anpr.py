"""OpenCV frame helpers for ANPR. Plate text is read by Ollama vision, not Tesseract."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config import settings

MODEL_ID = "ollama-vision-p0"


def local_model_hash() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    except OSError:
        return "unpinned"


@dataclass
class PlateRead:
    plate_raw: str
    plate_norm: str
    syntax_ok: bool
    confidence: float
    crop_bgr: np.ndarray | None
    box: tuple[int, int, int, int] | None


def _yellow_white_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Tight white: skip grey asphalt / lane paint (V~210).
    white = cv2.inRange(hsv, (0, 0, 230), (180, 40, 255))
    yellow = cv2.inRange(hsv, (10, 40, 80), (45, 255, 255))
    return cv2.bitwise_or(white, yellow)


def detect_plate_boxes(bgr: np.ndarray, min_width: int | None = None) -> list[tuple[int, int, int, int, float]]:
    min_w = min_width or settings.min_plate_width_px
    h, w = bgr.shape[:2]
    mask = _yellow_white_mask(bgr)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 11), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored: list[tuple[int, int, int, int, float]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        # CCTV HUD / timestamp overlays sit in the top band. Do not OCR them as plates.
        if y < int(0.18 * h):
            continue
        if bw < min_w or bh < 18 or bw > 0.72 * w:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 1.6 or aspect > 6.5:
            continue
        area = cv2.contourArea(cnt)
        fill = area / max(bw * bh, 1)
        if fill < 0.35:
            continue
        sharpness = _sharpness(bgr[y : y + bh, x : x + bw])
        score = fill * min(aspect / 4.0, 1.2) * (sharpness / 400.0) * (bw / w)
        scored.append((x, y, bw, bh, float(score)))
    scored.sort(key=lambda t: t[4], reverse=True)
    return scored[:3]


def _sharpness(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mask_hud(bgr: np.ndarray) -> np.ndarray:
    """Blank channel-name / timestamp bars so OCR/vision do not read CSITMS/PTZ/clock."""
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return bgr
    out = bgr.copy()
    h, w = out.shape[:2]
    out[: max(1, int(0.12 * h)), :] = 0
    out[max(0, int(0.88 * h)) :, :] = 0
    return out


def prepare_live_anpr_frame(bgr: np.ndarray) -> np.ndarray:
    """Zoom the strongest vehicle-like region on a wide PTZ view, else HUD-masked full frame."""
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return bgr
    h, w = bgr.shape[:2]
    y0, y1 = int(0.12 * h), int(0.90 * h)
    region = bgr[y0:y1, :]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, hot = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(blur, 40, 120)
    comb = cv2.dilate(cv2.bitwise_or(hot, edges), np.ones((9, 15), np.uint8))
    contours, _ = cv2.findContours(comb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    rh, rw = region.shape[:2]
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if bw < 48 or bh < 28 or area < 0.004 * rw * rh or area > 0.5 * rw * rh:
            continue
        if area > best_area:
            best_area = area
            best = (x, y + y0, bw, bh)
    if best is None:
        return mask_hud(bgr)
    x, y, bw, bh = best
    pad_x, pad_y = int(bw * 0.2), int(bh * 0.45)
    x0 = max(0, x - pad_x)
    y0b = max(0, y - pad_y)
    x1 = min(w, x + bw + pad_x)
    y1b = min(h, y + bh + pad_y + int(bh * 0.5))
    crop = bgr[y0b:y1b, x0:x1]
    if crop.size == 0:
        return mask_hud(bgr)
    if crop.shape[1] < 360:
        scale = 360.0 / max(crop.shape[1], 1)
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
    return crop


def load_bgr(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
