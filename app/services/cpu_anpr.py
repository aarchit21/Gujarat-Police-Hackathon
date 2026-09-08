"""CPU-first plate detection/OCR with optional FastALPR and Tesseract fallback.

FastALPR is loaded only after an explicit setup step marks models ready. This
prevents a live camera worker from downloading model weights unexpectedly.
"""
from __future__ import annotations

import hashlib
import inspect
import shutil
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import settings
from app.services.anpr import detect_plate_boxes, enhance_for_vision
from app.services.plates import layout_hint, normalize, syntax_ok

_lock = threading.Lock()
_inference_slots = threading.BoundedSemaphore(max(1, int(settings.cpu_anpr_workers)))
_alpr = None
_load_error = ""
_loaded_model_hash = ""


@dataclass
class PlateQuality:
    width: int
    height: int
    sharpness: float
    brightness: float
    dark_fraction: float
    clipped_fraction: float
    score: float
    eligible: bool
    reason: str = ""


@dataclass
class CpuPlateCandidate:
    plate_raw: str = ""
    plate_norm: str = ""
    confidence: float = 0.0
    character_confidences: list[float] = field(default_factory=list)
    crop_bgr: np.ndarray | None = None
    box: tuple[int, int, int, int] | None = None
    detector: str = ""
    recognizer: str = ""
    model_id: str = ""
    model_hash: str = ""
    quality: dict = field(default_factory=dict)
    reason: str = "ocr_empty"
    latency_ms: float = 0.0
    raw_output: str = ""
    secondary_raw: str = ""
    secondary_norm: str = ""
    reader_agreement: str = ""
    detector_confidence: float = 0.0


def plate_quality(crop: np.ndarray | None) -> PlateQuality:
    if crop is None or getattr(crop, "size", 0) == 0:
        return PlateQuality(0, 0, 0.0, 0.0, 1.0, 0.0, 0.0, False, "no_plate_candidate")
    height, width = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    dark = float(np.mean(gray <= 28))
    clipped = float(np.mean(gray >= 245))
    eligible = width >= settings.cpu_anpr_min_plate_width_px and height >= settings.cpu_anpr_min_plate_height_px
    reason = ""
    if not eligible:
        reason = "insufficient_pixels"
    elif dark > 0.72 or brightness < 30:
        reason = "underexposed"
    elif clipped > 0.72:
        reason = "glare"
    elif sharpness < 12:
        reason = "blur"
    size_score = min(1.0, width / 160.0) * min(1.0, height / 48.0)
    exposure_score = max(0.0, 1.0 - dark - clipped)
    sharp_score = min(1.0, sharpness / 180.0)
    score = 0.45 * size_score + 0.35 * sharp_score + 0.20 * exposure_score
    return PlateQuality(
        width=int(width), height=int(height), sharpness=round(sharpness, 3),
        brightness=round(brightness, 3), dark_fraction=round(dark, 4),
        clipped_fraction=round(clipped, 4), score=round(float(score), 4),
        eligible=eligible, reason=reason,
    )


def rectify_plate(crop: np.ndarray, corners: list[tuple[float, float]] | None) -> np.ndarray:
    """Perspective-rectify four observed corners; return a copy when unreliable."""
    if crop is None or not getattr(crop, "size", 0) or not corners or len(corners) != 4:
        return crop.copy() if crop is not None else crop
    points = np.asarray(corners, dtype=np.float32)
    if not np.isfinite(points).all():
        return crop.copy()
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered = np.array([
        points[np.argmin(sums)], points[np.argmin(diffs)],
        points[np.argmax(sums)], points[np.argmax(diffs)],
    ], dtype=np.float32)
    tl, tr, br, bl = ordered
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 8 or height < 6 or width / max(height, 1) < 1.2:
        return crop.copy()
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(ordered, target)
    return cv2.warpPerspective(crop, matrix, (width, height), flags=cv2.INTER_CUBIC)


def aligned_median_fusion(crops: list[np.ndarray], *, min_ecc: float = 0.85) -> np.ndarray | None:
    """Fuse aligned observations only; reject a track whose alignment is weak."""
    valid = [crop for crop in crops if crop is not None and getattr(crop, "size", 0)]
    if len(valid) < 2:
        return None
    base = valid[0]
    height, width = base.shape[:2]
    base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if base.ndim == 3 else base
    aligned = [base.astype(np.float32)]
    for crop in valid[1:]:
        resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            score, warp = cv2.findTransformECC(
                base_gray.astype(np.float32), gray.astype(np.float32), warp,
                cv2.MOTION_TRANSLATION,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4),
            )
        except cv2.error:
            return None
        if float(score) < min_ecc:
            return None
        registered = cv2.warpAffine(
            resized, warp, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        aligned.append(registered.astype(np.float32))
    return np.median(np.stack(aligned, axis=0), axis=0).clip(0, 255).astype(np.uint8)


def confidence_gate(candidate: CpuPlateCandidate) -> bool:
    confs = [float(v) for v in candidate.character_confidences if v is not None]
    if (
        not syntax_ok(candidate.plate_norm)
        or not candidate.quality.get("eligible", False)
        or candidate.reader_agreement == "disagree"
    ):
        return False
    if confs:
        return (
            statistics.median(confs) >= settings.plate_confirmation_median_confidence
            and min(confs) >= settings.plate_confirmation_min_char_confidence
        )
    return candidate.confidence >= settings.plate_confirmation_median_confidence


def localization_gate(candidate: CpuPlateCandidate) -> bool:
    """Accept only plausible, detector-backed native plate observations.

    The colour-contour locator has no semantic detector confidence, so it must
    never pass this gate for live ANPR. Two-line Indian plates are retained by
    the conservative 1.2 aspect lower bound.
    """
    crop = candidate.crop_bgr
    if crop is None or not getattr(crop, "size", 0):
        return False
    height, width = crop.shape[:2]
    aspect = width / max(height, 1)
    if aspect < 1.2 or aspect > 8.0:
        return False
    if candidate.detector != "fast_alpr":
        return False
    return float(candidate.detector_confidence or 0.0) >= float(settings.cpu_anpr_min_detector_confidence)


def _model_hash() -> str:
    if _loaded_model_hash:
        return _loaded_model_hash
    value = f"{settings.cpu_anpr_detector_model}|{settings.cpu_anpr_ocr_model}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def cpu_anpr_status() -> dict:
    try:
        import onnxruntime
        providers = list(onnxruntime.get_available_providers())
        onnx_available = True
    except Exception as exc:
        providers = []
        onnx_available = False
        err = str(exc)
    else:
        err = ""
    try:
        import fast_alpr  # noqa: F401
        fast_available = True
    except Exception as exc:
        fast_available = False
        err = err or str(exc)
    configured_tesseract = Path(settings.tesseract_cmd) if settings.tesseract_cmd else None
    tesseract_available = bool(
        (configured_tesseract and configured_tesseract.is_file()) or shutil.which("tesseract")
    )
    return {
        "enabled": bool(settings.cpu_anpr_enabled),
        "models_ready": bool(settings.cpu_anpr_models_ready),
        "fast_alpr_available": fast_available,
        "onnx_available": onnx_available,
        "onnx_providers": providers,
        "execution_provider": "CPUExecutionProvider",
        "inference_workers": max(1, int(settings.cpu_anpr_workers)),
        "tesseract_available": tesseract_available,
        "loaded": _alpr is not None,
        "detector_model": settings.cpu_anpr_detector_model,
        "ocr_model": settings.cpu_anpr_ocr_model,
        "model_hash": _model_hash(),
        "error": _load_error or err,
    }


def _load_alpr():
    global _alpr, _load_error, _loaded_model_hash
    if _alpr is not None:
        return _alpr
    if not settings.cpu_anpr_enabled or not settings.cpu_anpr_models_ready:
        return None
    with _lock:
        if _alpr is not None:
            return _alpr
        try:
            from fast_alpr import ALPR
            kwargs = {
                "detector_model": settings.cpu_anpr_detector_model,
                "ocr_model": settings.cpu_anpr_ocr_model,
                "detector_providers": ["CPUExecutionProvider"],
                "ocr_device": "cpu",
                "ocr_providers": ["CPUExecutionProvider"],
            }
            accepted = inspect.signature(ALPR).parameters
            _alpr = ALPR(**{k: v for k, v in kwargs.items() if k in accepted})
            paths = [
                Path(_alpr.detector.detector.model._model_path),
                Path(_alpr.ocr.ocr_model.model._model_path),
            ]
            digest = hashlib.sha256()
            for path in paths:
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            _loaded_model_hash = digest.hexdigest()[:16]
            _load_error = ""
        except Exception as exc:
            _load_error = str(exc)[:500]
            _alpr = None
    return _alpr


def _value(obj: Any, *names: str, default=None):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _parse_box(result: Any) -> tuple[int, int, int, int] | None:
    det = _value(result, "detection", default=result)
    box = _value(det, "bounding_box", "bbox", "box")
    if box is None:
        return None
    if not isinstance(box, dict):
        box = {name: getattr(box, name, None) for name in ("x1", "y1", "x2", "y2", "x", "y", "w", "h")}
    try:
        if box.get("x1") is not None:
            x1, y1, x2, y2 = (int(box[k]) for k in ("x1", "y1", "x2", "y2"))
            return x1, y1, max(0, x2 - x1), max(0, y2 - y1)
        return tuple(int(box[k]) for k in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError):
        return None


def detect_with_fast_alpr(bgr: np.ndarray, *, predictor=None) -> list[CpuPlateCandidate]:
    started = time.perf_counter()
    runner = predictor or _load_alpr()
    if runner is None:
        return []
    try:
        results = runner.predict(bgr) if hasattr(runner, "predict") else runner(bgr)
    except Exception as exc:
        global _load_error
        _load_error = str(exc)[:500]
        return []
    out: list[CpuPlateCandidate] = []
    height, width = bgr.shape[:2]
    for result in results or []:
        det = _value(result, "detection", default=result)
        box = _parse_box(result)
        if not box:
            continue
        x, y, w, h = box
        x, y = max(0, x), max(0, y)
        crop = bgr[y:min(height, y + h), x:min(width, x + w)].copy()
        if crop.size == 0:
            continue
        ocr = _value(result, "ocr", default=result)
        raw = str(_value(ocr, "text", "plate_text", default="") or "")
        norm = normalize(raw)
        confs = _value(ocr, "character_confidences", "char_confidences", default=[]) or []
        raw_confidence = _value(ocr, "confidence", default=0.0)
        if isinstance(raw_confidence, (list, tuple, np.ndarray)) and not confs:
            confs = list(raw_confidence)
        confs = [max(0.0, min(1.0, float(v))) for v in confs]
        conf = statistics.mean(confs) if confs else float(raw_confidence or 0.0)
        quality = asdict(plate_quality(crop))
        raw_detector_confidence = _value(det, "confidence", "score", default=None)
        # Predictors in unit tests and older FastALPR adapters may omit a score.
        # Treat those as an explicitly trusted adapter result, not as a silent
        # contour fallback. Production FastALPR supplies this value.
        detector_confidence = float(raw_detector_confidence) if raw_detector_confidence is not None else 1.0
        quality.update({
            "aspect_ratio": round(float(w) / max(float(h), 1.0), 3),
            "detector_confidence": round(detector_confidence, 4),
        })
        reason = "candidate" if norm else quality.get("reason") or "ocr_empty"
        out.append(CpuPlateCandidate(
            plate_raw=raw, plate_norm=norm, confidence=max(0.0, min(1.0, conf)),
            character_confidences=confs, crop_bgr=crop, box=(x, y, w, h),
            detector="fast_alpr", recognizer="fast_plate_ocr",
            model_id=f"{settings.cpu_anpr_detector_model}+{settings.cpu_anpr_ocr_model}",
            model_hash=_model_hash(), quality=quality, reason=reason,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3), raw_output=raw,
            detector_confidence=detector_confidence,
        ))
    return out


def _tesseract_variant(crop: np.ndarray, name: str) -> np.ndarray:
    if name == "original":
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if name == "gray":
        return gray
    if name == "adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 7)
    enhanced, _ = enhance_for_vision(crop, profile="plate", min_width=400)
    return enhanced


def recognize_with_tesseract(crop: np.ndarray) -> CpuPlateCandidate:
    quality = asdict(plate_quality(crop))
    base = CpuPlateCandidate(
        crop_bgr=crop.copy() if crop is not None and crop.size else crop,
        detector="opencv_plate", recognizer="tesseract",
        model_id="tesseract-opencv-multiview-v1", model_hash=_model_hash(), quality=quality,
        reason=quality.get("reason") or "ocr_empty",
    )
    if not settings.cpu_anpr_tesseract_enabled or crop is None or not crop.size:
        return base
    if not quality.get("eligible"):
        return base
    started = time.perf_counter()
    try:
        import pytesseract
        from pytesseract import Output
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    except Exception:
        base.reason = "model_unavailable"
        return base
    reads: list[CpuPlateCandidate] = []
    for name in ("original", "enhanced", "gray", "adaptive"):
        try:
            image = _tesseract_variant(crop, name)
            data = pytesseract.image_to_data(
                image, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                output_type=Output.DICT,
            )
            tokens, token_conf = [], []
            for text, conf in zip(data.get("text", []), data.get("conf", [])):
                norm = normalize(text)
                try:
                    score = float(conf) / 100.0
                except (TypeError, ValueError):
                    score = -1.0
                if norm and score >= 0:
                    tokens.append(norm)
                    token_conf.append(max(0.0, min(1.0, score)))
            raw = "".join(tokens)
            norm = normalize(raw)
            char_conf = [c for token, c in zip(tokens, token_conf) for _ in token]
            reads.append(CpuPlateCandidate(
                plate_raw=raw, plate_norm=norm,
                confidence=statistics.mean(char_conf) if char_conf else 0.0,
                character_confidences=char_conf, crop_bgr=base.crop_bgr,
                detector=base.detector, recognizer=f"tesseract:{name}",
                model_id=base.model_id, model_hash=base.model_hash,
                quality=quality, reason="candidate" if norm else base.reason, raw_output=raw,
            ))
        except Exception:
            continue
    if not reads:
        base.latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return base
    reads.sort(key=lambda r: (syntax_ok(r.plate_norm), r.confidence, len(r.plate_norm)), reverse=True)
    best = reads[0]
    best.latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    if best.plate_norm and not syntax_ok(best.plate_norm):
        best.reason = "syntax_invalid"
    elif best.plate_norm:
        best.reason = "candidate"
    return best


def opencv_tesseract_candidates(bgr: np.ndarray) -> list[CpuPlateCandidate]:
    out: list[CpuPlateCandidate] = []
    for x, y, w, h, _score in detect_plate_boxes(bgr, min_width=12, skip_top=True, skip_bottom=True):
        crop = bgr[y:y + h, x:x + w].copy()
        candidate = recognize_with_tesseract(crop)
        candidate.box = (x, y, w, h)
        out.append(candidate)
    return out


def cpu_plate_candidates(
    bgr: np.ndarray,
    *,
    predictor=None,
    allow_opencv_fallback: bool = True,
) -> list[CpuPlateCandidate]:
    if not settings.cpu_anpr_enabled or bgr is None or not getattr(bgr, "size", 0):
        return []
    with _inference_slots:
        fast = detect_with_fast_alpr(bgr, predictor=predictor)
        if fast:
            if settings.cpu_anpr_secondary_tesseract:
                for candidate in fast:
                    if (
                        syntax_ok(candidate.plate_norm)
                        and candidate.confidence >= settings.cpu_anpr_secondary_trigger_confidence
                    ):
                        candidate.reader_agreement = "primary_high_confidence"
                        continue
                    secondary = recognize_with_tesseract(candidate.crop_bgr)
                    candidate.secondary_raw = secondary.plate_raw
                    candidate.secondary_norm = secondary.plate_norm
                    if secondary.plate_norm:
                        if secondary.plate_norm == candidate.plate_norm:
                            candidate.reader_agreement = "agree"
                        elif layout_hint(secondary.plate_norm) == candidate.plate_norm:
                            # The primary strict value is unchanged. This only records
                            # that Tesseract's positional ambiguity explains the delta.
                            candidate.reader_agreement = "agree_layout_review"
                        else:
                            candidate.reader_agreement = "disagree"
                    else:
                        candidate.reader_agreement = "primary_only"
                    candidate.raw_output = f"primary={candidate.plate_raw};secondary={secondary.plate_raw}"
                    candidate.latency_ms += secondary.latency_ms
                    if candidate.reader_agreement == "disagree":
                        candidate.reason = "reader_disagreement"
            return sorted(fast, key=lambda c: (confidence_gate(c), c.quality.get("score", 0), c.confidence), reverse=True)
        if not allow_opencv_fallback:
            return []
        return sorted(
            opencv_tesseract_candidates(bgr),
            key=lambda c: (confidence_gate(c), c.quality.get("score", 0), c.confidence), reverse=True,
        )
