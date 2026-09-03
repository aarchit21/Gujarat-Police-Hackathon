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


def _indian_plate_mask(bgr: np.ndarray) -> np.ndarray:
    """White, yellow, green (EV), blue (diplomatic/commercial). Not HUD grey."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 230), (180, 40, 255))
    yellow = cv2.inRange(hsv, (10, 40, 80), (45, 255, 255))
    green = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
    blue = cv2.inRange(hsv, (90, 40, 40), (140, 255, 255))
    return cv2.bitwise_or(cv2.bitwise_or(white, yellow), cv2.bitwise_or(green, blue))


def _yellow_white_mask(bgr: np.ndarray) -> np.ndarray:
    return _indian_plate_mask(bgr)


def detect_plate_boxes(
    bgr: np.ndarray,
    min_width: int | None = None,
    *,
    skip_top: bool = True,
    skip_bottom: bool = False,
) -> list[tuple[int, int, int, int, float]]:
    min_w = min_width or settings.min_plate_width_px
    h, w = bgr.shape[:2]
    mask = _indian_plate_mask(bgr)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 11), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored: list[tuple[int, int, int, int, float]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        # CCTV HUD / timestamp overlays sit in the top and bottom bands of the full frame.
        if skip_top and y < int(0.12 * h):
            continue
        if skip_bottom and (y + bh) > int(0.90 * h):
            continue
        if bw < min_w or bh < 12 or bw > 0.72 * w:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 1.5 or aspect > 7.0:
            continue
        area = cv2.contourArea(cnt)
        fill = area / max(bw * bh, 1)
        if fill < 0.30:
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


def _clip_box(bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(w, x1), min(h, y1)
    if xb - xa < 8 or yb - ya < 8:
        return None
    crop = bgr[ya:yb, xa:xb]
    return None if crop.size == 0 else crop


def plate_focus_crops(bgr: np.ndarray, box: tuple[int, int, int, int] | None) -> list[np.ndarray]:
    """Native-frame plate patches: colour boxes, then nose/tail (side view), then bumper."""
    if bgr is None or getattr(bgr, "size", 0) == 0 or not box:
        return []
    fh, fw = bgr.shape[:2]
    x, y, bw, bh = [int(v) for v in box[:4]]
    pad = max(2, int(0.08 * bw))
    x0 = max(0, x - pad)
    x1 = min(fw, x + bw + pad)
    y0 = max(int(0.10 * fh), y)
    y1 = min(int(0.90 * fh), y + bh + max(2, int(0.06 * bh)))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return []
    roi = bgr[y0:y1, x0:x1]
    out: list[np.ndarray] = []
    for px, py, pw, ph, _score in detect_plate_boxes(roi, min_width=12, skip_top=True, skip_bottom=False):
        patch = roi[py : py + ph, px : px + pw]
        if patch.size:
            out.append(patch)
    left = _clip_box(bgr, x, y + int(0.22 * bh), x + int(0.34 * bw), y + bh)
    right = _clip_box(bgr, x + int(0.66 * bw), y + int(0.22 * bh), x + bw, y + bh)
    ends = [p for p in (left, right) if p is not None]
    if len(ends) == 2:
        ends.sort(key=_sharpness, reverse=True)
        out.append(ends[0])
    elif ends:
        out.append(ends[0])
    bumper = bumper_crop(bgr, box)
    if bumper is not None and bumper.size:
        out.append(bumper)
    return out[:3]


def bumper_crop(bgr: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray | None:
    """Lower half of a vehicle box — where Indian plates sit — from the native frame."""
    if bgr is None or getattr(bgr, "size", 0) == 0 or not box:
        return None
    h, w = bgr.shape[:2]
    x, y, bw, bh = [int(v) for v in box[:4]]
    if bw < 8 or bh < 8:
        return None
    pad_x = max(2, int(0.06 * bw))
    x0 = max(0, x - pad_x)
    x1 = min(w, x + bw + pad_x)
    y0 = max(0, y + int(0.48 * bh))
    y1 = min(h, y + bh + max(2, int(0.08 * bh)))
    if y1 - y0 < 8 or x1 - x0 < 8:
        y0 = max(0, y)
        y1 = min(h, y + bh)
    crop = bgr[y0:y1, x0:x1]
    return None if crop.size == 0 else crop


def enhance_plate_crop(bgr: np.ndarray, min_width: int = 320) -> np.ndarray:
    """Light upsample + CLAHE. Heavy denoise/unsharp smears plate strokes; do not invent pixels."""
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return bgr
    out = bgr
    try:
        h, w = out.shape[:2]
        if w < 16 or h < 10:
            return bgr
        target = max(int(min_width or 320), 160)
        if w < target:
            scale = min(target / float(w), 8.0)
            out = cv2.resize(
                out,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_ch)
        out = cv2.cvtColor(cv2.merge((l2, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
    except Exception:
        return bgr
    return out


def estimate_vehicle_color(bgr: np.ndarray) -> str:
    """Dominant body colour from a crop. Not VAHAN."""
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return ""
    try:
        h, w = bgr.shape[:2]
        roi = bgr[int(0.18 * h) : int(0.72 * h), int(0.12 * w) : int(0.88 * w)]
        if roi.size == 0:
            roi = bgr
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)
        if pixels.size == 0:
            return ""
        s = pixels[:, 1].astype(np.int32)
        v = pixels[:, 2].astype(np.int32)
        hue = pixels[:, 0].astype(np.int32)
        if np.mean(v) >= 165 and np.mean(s) <= 45:
            return "white"
        if np.mean(v) <= 55:
            return "black"
        if np.mean(s) <= 40:
            return "silver"
        mask = (s > 40) & (v > 50)
        if not np.any(mask):
            return "silver"
        h_mean = float(np.mean(hue[mask]))
        if h_mean < 12 or h_mean >= 170:
            return "red"
        if h_mean < 28:
            return "orange"
        if h_mean < 38:
            return "yellow"
        if h_mean < 85:
            return "green"
        if h_mean < 135:
            return "blue"
        return "red"
    except Exception:
        return ""


def crop_box_width(item: dict) -> int:
    box = item.get("box") if item else None
    if not box or len(box) < 3:
        return 0
    try:
        return int(box[2])
    except (TypeError, ValueError):
        return 0


def anpr_crops(bgr: np.ndarray, *, live: bool) -> list[dict]:
    """YOLO vehicle crops first; OpenCV blob or full frame if YOLO finds nothing."""
    from app.services.yolo_detect import detect_vehicles

    dets = detect_vehicles(bgr)
    if dets:
        ranked = sorted(dets, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1), reverse=True)
        out = []
        for d in ranked[: max(1, int(settings.yolo_max_crops or 2))]:
            box = (d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1)
            focuses = plate_focus_crops(bgr, box)
            plate = focuses[0] if focuses else bumper_crop(bgr, box)
            out.append(
                {
                    "crop": plate if plate is not None and getattr(plate, "size", 0) else d.crop,
                    "body_crop": d.crop,
                    "plate_crops": focuses,
                    "box": box,
                    "vehicle_type": d.vehicle_type,
                    "detector": "yolov8n",
                    "det_conf": d.confidence,
                }
            )
        return out
    if live:
        return [
            {
                "crop": prepare_live_anpr_frame(bgr),
                "box": None,
                "vehicle_type": "",
                "detector": "opencv_blob",
                "det_conf": 0.0,
            }
        ]
    return [{"crop": bgr, "box": None, "vehicle_type": "", "detector": "full_frame", "det_conf": 0.0}]


def load_bgr(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
