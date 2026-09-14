"""End-to-end vehicle observation runner: detect -> track -> attributes -> record.

One structured record per *track*, never one per frame.  Attribute inference
runs only on the best few crops of each track, and the result is aggregated
with explicit abstention (see ``track_aggregate``).

ANPR is optional and strictly downstream: a missing, unreadable or failed plate
can never remove a vehicle observation or change its type or colour.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from app.config import ROOT, settings
from app.services.crop_quality import score_crop
from app.services.track_aggregate import CropObservation, TrackAccumulator
from app.services.vehicle_attributes import (
    AttributeModelUnavailable,
    classify_crop,
    extend_box,
    missing_weight_files,
)
from app.services.vehicle_tracking import GreedyIouTracker, build_tracker

def redact_source(source: str) -> str:
    """Strip userinfo from a stream URL. RTSP passwords must never be logged."""
    text = str(source or "")
    if "://" not in text:
        return text
    scheme, rest = text.split("://", 1)
    if "@" not in rest.split("/", 1)[0]:
        return text
    netloc, _, path = rest.partition("/")
    host = netloc.rsplit("@", 1)[1]
    return f"{scheme}://<redacted>@{host}" + (f"/{path}" if path else "")


PLATE_DISABLED = "disabled"
PLATE_NOT_VISIBLE = "plate_not_visible"
PLATE_UNREADABLE = "plate_unreadable"
PLATE_OCR_EMPTY = "ocr_empty"
PLATE_OCR_ERROR = "ocr_error"
PLATE_OK = "ok"


@dataclass
class RunnerConfig:
    camera_id: str = "camera-01"
    run_id: str = "run"
    device: str = "auto"
    imgsz: int = 640
    stride: int = 1
    max_frames: int = 0
    anpr: bool = False
    evidence_dir: Path | None = None
    track_timeout_seconds: float = 3.0


@dataclass
class RunnerMetrics:
    frames_read: int = 0
    frames_processed: int = 0
    detections: int = 0
    tracks_started: int = 0
    tracks_emitted: int = 0
    attribute_inferences: int = 0
    crops_rejected: dict = field(default_factory=dict)
    detect_ms: list = field(default_factory=list)
    attribute_ms: list = field(default_factory=list)
    anpr_ms: list = field(default_factory=list)
    wall_seconds: float = 0.0
    peak_vram_allocated_bytes: int = 0
    peak_vram_reserved_bytes: int = 0
    device: str = ""
    half_requested: bool = False
    half_verified: bool = False
    tracker_source: str = ""
    tracker_note: str = ""

    def summary(self) -> dict:
        def stats(values: list) -> dict:
            if not values:
                return {"count": 0}
            arr = np.asarray(values, dtype=np.float64)
            return {
                "count": int(arr.size),
                "mean_ms": round(float(arr.mean()), 2),
                "p50_ms": round(float(np.percentile(arr, 50)), 2),
                "p95_ms": round(float(np.percentile(arr, 95)), 2),
            }

        fps = (self.frames_processed / self.wall_seconds) if self.wall_seconds > 0 else 0.0
        # Wall-clock FPS on a live RTSP source is bounded by the stream's own
        # real-time rate, not by compute. Report the compute-only rate too so
        # the two are never confused.
        detect_p50 = float(np.percentile(self.detect_ms, 50)) if self.detect_ms else 0.0
        compute_fps = (1000.0 / detect_p50) if detect_p50 > 0 else 0.0
        return {
            "processed_fps_wallclock": round(fps, 2),
            "compute_fps_detect_only": round(compute_fps, 1),
            "throughput_note": (
                "processed_fps_wallclock is bounded by the source (a live RTSP feed "
                "delivers at real time); compute_fps_detect_only is 1000/p50 detect latency."
            ),
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "detections": self.detections,
            "tracks_started": self.tracks_started,
            "tracks_emitted": self.tracks_emitted,
            "attribute_inferences": self.attribute_inferences,
            "crops_rejected": dict(sorted(self.crops_rejected.items())),
            "wall_seconds": round(self.wall_seconds, 3),
            "processed_fps": round(fps, 2),
            "detect": stats(self.detect_ms),
            "attribute": stats(self.attribute_ms),
            "anpr": stats(self.anpr_ms),
            "device": self.device,
            # Requested vs. read back off the live predictor after a real
            # forward pass -- the second is the one that means anything.
            "fp16_requested": self.half_requested,
            "fp16_verified": self.half_verified,
            "tracker": self.tracker_source,
            "tracker_note": self.tracker_note,
            "peak_vram_allocated_mb": round(self.peak_vram_allocated_bytes / 1e6, 1),
            "peak_vram_reserved_mb": round(self.peak_vram_reserved_bytes / 1e6, 1),
        }


# ---------------------------------------------------------------- sources


def _check_local_source(source: str) -> None:
    """Fail with the actual cause when a relative path misses.

    The usual reason is being in the wrong directory: every documented command
    uses paths relative to the repository root. "could not open source" on its
    own sends people looking for a codec problem.
    """
    if "://" in source:
        return
    path = Path(source)
    if path.exists():
        return
    hint = f"source not found: {source}\n  working directory: {Path.cwd()}"
    if not path.is_absolute():
        candidate = ROOT / source
        if candidate.exists():
            hint += (
                f"\n  but it DOES exist at: {candidate}"
                f"\n  Run from the repository root: cd {ROOT}"
            )
        else:
            hint += f"\n  also not found at: {candidate}"
    raise FileNotFoundError(hint)


def iter_frames(source: str, *, max_frames: int = 0, stride: int = 1, record: Path | None = None) -> Iterator[tuple[int, np.ndarray, float | None]]:
    """Yield (frame_index, bgr, pts_ms) from a video file, RTSP URL or image dir."""
    _check_local_source(source)
    path = Path(source)
    if path.is_dir():
        images = sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        writer = None
        for index, image_path in enumerate(images):
            if max_frames and index >= max_frames:
                break
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            if index % max(1, stride):
                continue
            if record is not None and writer is None:
                writer = _open_writer(record, frame, 10.0)
            if writer is not None:
                writer.write(frame)
            yield index, frame, float(index * 100.0)
        if writer is not None:
            writer.release()
        return

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"could not open source: {redact_source(source)}")
    writer = None
    try:
        index = 0
        emitted = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            pts = capture.get(cv2.CAP_PROP_POS_MSEC)
            pts_ms = float(pts) if pts and pts > 0 else None
            if record is not None and writer is None:
                fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
                writer = _open_writer(record, frame, float(fps) if 0 < fps < 121 else 15.0)
            if writer is not None:
                writer.write(frame)
            if index % max(1, stride) == 0:
                yield index, frame, pts_ms
                emitted += 1
                if max_frames and emitted >= max_frames:
                    break
            index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def context_crop(frame: np.ndarray, box: tuple[int, int, int, int], *, pad_scale: float = 1.25) -> np.ndarray | None:
    """Bounded, annotated context region around a vehicle box.

    A bounded crop, never a full camera recording, and a copy so the original
    frame is never drawn on.
    """
    if frame is None or not getattr(frame, "size", 0) or not box:
        return None
    height, width = frame.shape[:2]
    x, y, bw, bh = (int(v) for v in box)
    if bw <= 0 or bh <= 0:
        return None
    pad = max(24, int(max(bw, bh) * pad_scale))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(width, x + bw + pad), min(height, y + bh + pad)
    region = frame[y0:y1, x0:x1].copy()
    if region.size == 0:
        return None
    cv2.rectangle(region, (x - x0, y - y0), (x + bw - x0, y + bh - y0), (0, 0, 255), 2)
    return region


def _open_writer(record: Path, frame: np.ndarray, fps: float):
    record.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(record), fourcc, max(1.0, fps), (width, height))


# ---------------------------------------------------------------- runner


class VehicleRunner:
    """Detect, track, classify and emit one observation per track."""

    def __init__(self, config: RunnerConfig):
        self.config = config
        self.metrics = RunnerMetrics()
        self.accumulators: dict[str, TrackAccumulator] = {}
        self.finalized: set[str] = set()
        self.first_seen: dict[str, datetime] = {}
        self.last_seen: dict[str, datetime] = {}
        self._model = None
        self._device = "cpu"
        self._half = False
        self.tracker, self.tracker_source, self.tracker_note = build_tracker(config.camera_id, config.run_id)
        self.metrics.tracker_source = self.tracker_source
        self.metrics.tracker_note = self.tracker_note
        self.detector_source = ""

    # -- model ----------------------------------------------------------
    def _resolve_device(self) -> tuple[str, bool]:
        requested = (self.config.device or "auto").strip().lower()
        try:
            import torch

            cuda = torch.cuda.is_available()
        except Exception:
            cuda = False
        if requested in {"cpu"}:
            return "cpu", False
        if requested in {"auto", ""}:
            device = "cuda:0" if cuda else "cpu"
        else:
            device = requested
        half = bool(settings.yolo_half) and device.startswith("cuda")
        return device, half

    def load_detector(self):
        if self._model is not None:
            return self._model
        from ultralytics import YOLO

        from app.services.yolo_detect import _weights_path

        weights = _weights_path()
        if not weights.is_file():
            raise FileNotFoundError(
                f"detector weights not found: {weights}. Run: python scripts/pull_yolo.py"
            )
        self._device, self._half = self._resolve_device()
        self._model = YOLO(str(weights))
        self.metrics.device = self._device
        self.metrics.half_requested = self._half
        self.detector_source = f"ultralytics:{weights.name}"
        return self._model

    def precision_kwargs(self) -> dict:
        """Ultralytics renamed `half` to `quantize`; support both without warnings."""
        if not self._half:
            return {}
        import ultralytics

        try:
            major, minor = (int(p) for p in ultralytics.__version__.split(".")[:2])
        except ValueError:
            return {"half": True}
        return {"quantize": "fp16"} if (major, minor) >= (8, 4) else {"half": True}

    def _note_precision(self) -> None:
        """Read the precision actually in force off the compiled predictor."""
        predictor = getattr(self._model, "predictor", None)
        backend = getattr(predictor, "model", None)
        self.metrics.half_verified = bool(getattr(backend, "fp16", False))

    # -- per frame ------------------------------------------------------
    def process_frame(self, frame_index: int, frame: np.ndarray, pts_ms: float | None) -> None:
        model = self.load_detector()
        started = time.perf_counter()
        if isinstance(self.tracker, GreedyIouTracker):
            from app.services.yolo_detect import detect_vehicles

            def predict(image):
                return model.predict(
                    image,
                    conf=float(settings.yolo_conf or 0.25),
                    iou=float(settings.yolo_iou or 0.45),
                    classes=[2, 3, 5, 7],
                    imgsz=int(self.config.imgsz),
                    device=self._device,
                    verbose=False,
                    **self.precision_kwargs(),
                )

            dets = detect_vehicles(frame, predict_fn=predict, max_detections=settings.vehicle_max_detections)
            tracked = self.tracker.update(dets, pts_ms)
        else:
            tracked = self.tracker.track_frame(
                model,
                frame,
                pts_ms,
                device=self._device,
                imgsz=self.config.imgsz,
                precision=self.precision_kwargs(),
            )
        self.metrics.detect_ms.append((time.perf_counter() - started) * 1000.0)
        self._note_precision()
        self.metrics.frames_processed += 1
        self.metrics.detections += len(tracked)

        now = datetime.now(timezone.utc)
        # Every vehicle box in this frame, so each crop can be checked for how
        # much of a *different* vehicle it contains.
        frame_boxes = [t.box for t in tracked]
        for item in tracked:
            accumulator = self.accumulators.get(item.track_id)
            if accumulator is None:
                accumulator = TrackAccumulator(item.track_id)
                self.accumulators[item.track_id] = accumulator
                self.metrics.tracks_started += 1
                self.first_seen[item.track_id] = now
            self.last_seen[item.track_id] = now
            accumulator.note_frame(frame_index, pts_ms)

            # Attribute crop is taken from the *raw* detector box with a
            # symmetric extension, not from the plate-padded crop.
            x1, y1, x2, y2 = extend_box(item.box, frame.shape)
            body = frame[y1:y2, x1:x2]
            quality = score_crop(
                body,
                box=item.box,
                frame_shape=frame.shape,
                detector_confidence=item.detection.confidence,
                crop_box=(x1, y1, x2 - x1, y2 - y1),
                other_boxes=frame_boxes,
            )
            if not quality.eligible:
                accumulator.reject(quality.reason)
                self.metrics.crops_rejected[quality.reason] = (
                    self.metrics.crops_rejected.get(quality.reason, 0) + 1
                )
                continue
            accumulator.offer(
                CropObservation(
                    frame_index=frame_index,
                    pts_ms=pts_ms,
                    quality=quality,
                    detector_type=item.detection.vehicle_type,
                    detector_confidence=float(item.detection.confidence or 0.0),
                    crop=body.copy(),
                    context=context_crop(frame, item.box),
                    box=item.box,
                )
            )

    def expired_tracks(self, pts_ms: float | None) -> list[str]:
        if pts_ms is None:
            return []
        timeout = float(self.config.track_timeout_seconds) * 1000.0
        return [
            track_id
            for track_id, acc in self.accumulators.items()
            if track_id not in self.finalized
            and acc.last_pts_ms is not None
            and pts_ms - acc.last_pts_ms > timeout
        ]

    # -- finalize -------------------------------------------------------
    def finalize_track(self, track_id: str) -> dict | None:
        accumulator = self.accumulators.get(track_id)
        if accumulator is None or track_id in self.finalized:
            return None
        self.finalized.add(track_id)

        for observation in accumulator.needs_attributes():
            if observation.crop is None:
                continue
            started = time.perf_counter()
            observation.attributes = classify_crop(observation.crop)
            self.metrics.attribute_ms.append((time.perf_counter() - started) * 1000.0)
            self.metrics.attribute_inferences += 1

        aggregated = accumulator.aggregate()
        plate = self.read_plate(accumulator)
        record = self.build_record(accumulator, aggregated, plate)
        self.metrics.tracks_emitted += 1
        return record

    def finalize_all(self) -> list[dict]:
        out = []
        for track_id in list(self.accumulators):
            record = self.finalize_track(track_id)
            if record is not None:
                out.append(record)
        return out

    # -- ANPR (strictly optional, strictly downstream) -------------------
    def read_plate(self, accumulator: TrackAccumulator) -> dict:
        if not self.config.anpr:
            return {"plate_text": None, "plate_confidence": None, "plate_status": PLATE_DISABLED}
        best = accumulator.best
        if best is None or best.crop is None:
            return {"plate_text": None, "plate_confidence": None, "plate_status": PLATE_NOT_VISIBLE}
        started = time.perf_counter()
        try:
            from app.services.cpu_anpr import confidence_gate, cpu_plate_candidates, localization_gate

            candidates = cpu_plate_candidates(best.crop, allow_opencv_fallback=False)
            self.metrics.anpr_ms.append((time.perf_counter() - started) * 1000.0)
            if not candidates:
                return {"plate_text": None, "plate_confidence": None, "plate_status": PLATE_NOT_VISIBLE}
            top = candidates[0]
            if not top.plate_raw:
                return {
                    "plate_text": None,
                    "plate_confidence": None,
                    "plate_status": PLATE_OCR_EMPTY,
                    "plate_raw": "",
                }
            accepted = localization_gate(top) and confidence_gate(top)
            return {
                # Raw OCR is retained before Indian-plate normalization.
                "plate_raw": top.plate_raw,
                "plate_text": top.plate_norm if accepted else None,
                "plate_confidence": round(float(top.confidence), 4) if accepted else None,
                "plate_status": PLATE_OK if accepted else PLATE_UNREADABLE,
                "plate_recognizer": top.recognizer or top.detector,
            }
        except Exception as exc:
            # An ANPR failure must never remove or alter the vehicle record.
            self.metrics.anpr_ms.append((time.perf_counter() - started) * 1000.0)
            return {
                "plate_text": None,
                "plate_confidence": None,
                "plate_status": PLATE_OCR_ERROR,
                "plate_error": f"{type(exc).__name__}: {exc}"[:200],
            }

    # -- record ----------------------------------------------------------
    def build_record(self, accumulator: TrackAccumulator, aggregated, plate: dict) -> dict:
        best = accumulator.best
        quality = best.quality if best else None
        crop_path, context_path = self._save_evidence(accumulator)
        first = self.first_seen.get(accumulator.track_id)
        last = self.last_seen.get(accumulator.track_id)
        from app.services.vehicle_attributes import type_deployment_gate

        attributes = aggregated.as_dict()
        gate_passed, gate_reason = type_deployment_gate()
        # The raw candidate is kept verbatim; the operational value is
        # suppressed until a model is validated or a human verifies the record.
        type_candidate = attributes["vehicle_type"]
        operational_type = type_candidate if gate_passed else "unknown"
        record = {
            "camera_id": self.config.camera_id,
            "track_id": accumulator.track_id,
            "run_id": self.config.run_id,
            "first_seen": first.isoformat().replace("+00:00", "Z") if first else None,
            "last_seen": last.isoformat().replace("+00:00", "Z") if last else None,
            **attributes,
            "vehicle_type": operational_type,
            "vehicle_type_confidence": attributes["vehicle_type_confidence"] if gate_passed else 0.0,
            "type_candidate": type_candidate,
            "type_candidate_confidence": attributes["vehicle_type_confidence"],
            "type_state": "estimated" if gate_passed and operational_type != "unknown" else "unknown",
            "type_suppressed": not gate_passed,
            "type_gate_reason": gate_reason,
            "color_state": "estimated" if attributes["vehicle_color"] != "unknown" else "unknown",
            "verified": False,
            "review_required": True,
            **plate,
            "detector_source": self.detector_source,
            "tracker_source": self.tracker_source,
            "best_vehicle_crop": crop_path,
            "context_image": context_path,
            "review_status": "unreviewed",
            "quality": quality.as_dict() if quality else {},
            "frames_seen": accumulator.seen_frames,
            "crops_kept": len(accumulator.kept),
            "crops_rejected": dict(sorted(accumulator.rejected.items())),
            "status": "complete",
            "probabilities": {"type": aggregated.type_probs, "color": aggregated.color_probs},
            "disclaimer": (
                "Type and colour are model estimates with explicit abstention. "
                "Matching attributes do not confirm the same physical vehicle."
            ),
        }
        return record

    def _save_evidence(self, accumulator: TrackAccumulator) -> tuple[str | None, str | None]:
        """Write the best crop and its context frame. Originals are never overwritten."""
        directory = self.config.evidence_dir
        best = accumulator.best
        if directory is None or best is None or best.crop is None:
            return None, None
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        safe = accumulator.track_id.replace("/", "_").replace(":", "_")
        crop_path = directory / f"{safe}_vehicle.jpg"
        cv2.imwrite(str(crop_path), best.crop)
        context_path = None
        if best.context is not None and best.context.size:
            candidate = directory / f"{safe}_context.jpg"
            cv2.imwrite(str(candidate), best.context)
            context_path = str(candidate)
        return str(crop_path), context_path


def contact_sheet(records: list[dict], out_path: Path, *, columns: int = 4, cell: int = 200) -> Path | None:
    """Grid of best crops annotated with predicted type, colour and confidence."""
    usable = [r for r in records if r.get("best_vehicle_crop")]
    if not usable:
        return None
    rows = (len(usable) + columns - 1) // columns
    label_h = 46
    sheet = np.full((rows * (cell + label_h), columns * cell, 3), 24, dtype=np.uint8)
    for i, record in enumerate(usable):
        image = cv2.imread(str(record["best_vehicle_crop"]))
        if image is None:
            continue
        r, c = divmod(i, columns)
        y0, x0 = r * (cell + label_h), c * cell
        h, w = image.shape[:2]
        scale = min(cell / max(1, w), cell / max(1, h))
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
        rh, rw = resized.shape[:2]
        sheet[y0 : y0 + rh, x0 : x0 + rw] = resized
        text_type = f"{record.get('vehicle_type')} {record.get('vehicle_type_confidence', 0):.2f}"
        text_color = f"{record.get('vehicle_color')} {record.get('vehicle_color_confidence', 0):.2f}"
        cv2.putText(sheet, str(record.get("track_id", ""))[-18:], (x0 + 4, y0 + cell + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(sheet, text_type, (x0 + 4, y0 + cell + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(sheet, text_color, (x0 + 4, y0 + cell + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return out_path


def run(source: str, config: RunnerConfig, *, record_video: Path | None = None) -> tuple[list[dict], dict]:
    """Run the full pipeline over a source and return (records, metrics)."""
    missing = missing_weight_files()
    if settings.vattr_enabled and missing:
        raise AttributeModelUnavailable(
            "vehicle attribute weights missing:\n  "
            + "\n  ".join(missing)
            + "\nRun: python scripts/pull_vehicle_attributes.py"
        )

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        torch = None

    runner = VehicleRunner(config)
    records: list[dict] = []
    started = time.perf_counter()
    grad_ctx = torch.no_grad() if torch is not None else _NullContext()
    with grad_ctx:
        for frame_index, frame, pts_ms in iter_frames(
            source, max_frames=config.max_frames, stride=config.stride, record=record_video
        ):
            runner.metrics.frames_read += 1
            runner.process_frame(frame_index, frame, pts_ms)
            for track_id in runner.expired_tracks(pts_ms):
                finished = runner.finalize_track(track_id)
                if finished is not None:
                    records.append(finished)
        records.extend(runner.finalize_all())
    runner.metrics.wall_seconds = time.perf_counter() - started

    if torch is not None and torch.cuda.is_available():
        runner.metrics.peak_vram_allocated_bytes = int(torch.cuda.max_memory_allocated())
        runner.metrics.peak_vram_reserved_bytes = int(torch.cuda.max_memory_reserved())
    return records, runner.metrics.summary()


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def write_jsonl(records: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
