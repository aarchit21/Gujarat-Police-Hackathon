"""Local ANPR: OpenCV plate-like crop + Tesseract OCR.

Awiros / PaddleOCR / YOLO are candidates only. This provider records
model_id=tesseract-opencv-p0 and must produce a real crop and raw string.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

from app.config import settings
from app.services.plates import normalize, syntax_ok

pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

TESS_WHITELIST = (
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    "-c load_system_dawg=0 -c load_freq_dawg=0"
)
TESS_CONFIGS = [
    f"--oem 3 --psm 7 {TESS_WHITELIST}",
    f"--oem 3 --psm 8 {TESS_WHITELIST}",
]
MODEL_ID = "tesseract-opencv-p0"


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


def ocr_crop(crop_bgr: np.ndarray) -> tuple[str, float]:
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() < 127:
        bw = cv2.bitwise_not(bw)
    kernel = np.ones((2, 2), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    img = Image.fromarray(bw)
    best_raw, best_conf, best_norm_len = "", 0.0, -1
    try:
        for cfg in TESS_CONFIGS:
            raw = pytesseract.image_to_string(img, config=cfg).strip()
            data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
            confs = []
            for conf in data.get("conf", []):
                try:
                    c = float(conf)
                except (TypeError, ValueError):
                    continue
                if c >= 0:
                    confs.append(c)
            conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
            n = normalize(raw)
            score = (2 if syntax_ok(n) else 0) + len(n) + conf
            if score > best_norm_len:
                best_norm_len = score
                best_raw, best_conf = raw, conf
    except pytesseract.TesseractNotFoundError:
        return "", 0.0
    return best_raw, best_conf


def read_frame(bgr: np.ndarray) -> PlateRead:
    boxes = detect_plate_boxes(bgr)
    if not boxes:
        return PlateRead("", "", False, 0.0, None, None)
    x, y, w, h, det_score = boxes[0]
    pad = 4
    crop = bgr[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad].copy()
    raw, ocr_conf = ocr_crop(crop)
    norm = normalize(raw)
    conf = float(min(1.0, 0.4 * det_score + 0.6 * ocr_conf))
    return PlateRead(raw, norm, syntax_ok(norm), conf, crop, (x, y, w, h))


def load_bgr(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
