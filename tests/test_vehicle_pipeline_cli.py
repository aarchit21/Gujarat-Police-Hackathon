"""Runner and CLI: plate isolation, evidence, CPU fallback, fail-loud modes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.services.crop_quality import CropQuality  # noqa: E402
from app.services.track_aggregate import CropObservation, TrackAccumulator  # noqa: E402
from app.services.vehicle_attributes import COLOR_CLASSES, TYPE_CLASSES, AttributeResult  # noqa: E402
from app.services.vehicle_runner import (  # noqa: E402
    PLATE_DISABLED,
    PLATE_NOT_VISIBLE,
    PLATE_OCR_ERROR,
    RunnerConfig,
    VehicleRunner,
    context_crop,
    iter_frames,
    redact_source,
    write_jsonl,
)
from scripts import vehicle_pipeline  # noqa: E402


def _quality(score: float = 0.6) -> CropQuality:
    return CropQuality(
        width=200, height=200, sharpness=300.0, exposure=0.9, dark_fraction=0.0,
        clipped_fraction=0.0, visible_fraction=1.0, detector_confidence=0.9,
        score=score, eligible=True, reason="",
    )


def _runner(tmp_path, *, anpr: bool = False) -> VehicleRunner:
    return VehicleRunner(RunnerConfig(camera_id="cam-test", run_id="r1", anpr=anpr, evidence_dir=tmp_path))


def _accumulator_with_crop(runner: VehicleRunner, track_id: str = "t1") -> TrackAccumulator:
    acc = TrackAccumulator(track_id)
    type_probs = np.zeros(len(TYPE_CLASSES), dtype=np.float32)
    type_probs[TYPE_CLASSES.index("car")] = 0.97
    color_probs = np.zeros(len(COLOR_CLASSES), dtype=np.float32)
    color_probs[COLOR_CLASSES.index("white")] = 0.95
    for i in range(2):
        acc.note_frame(i, float(i * 100))
        acc.offer(CropObservation(
            frame_index=i, pts_ms=float(i * 100), quality=_quality(),
            detector_type="car", detector_confidence=0.9,
            attributes=AttributeResult(type_probs=type_probs, color_probs=color_probs, model_id="test"),
            crop=np.full((80, 80, 3), 120, dtype=np.uint8),
            context=np.full((160, 160, 3), 100, dtype=np.uint8),
            box=(10, 10, 80, 80),
        ))
    runner.accumulators[track_id] = acc
    return acc


# -- ANPR must never affect the vehicle record --------------------------


def test_plate_disabled_still_produces_a_vehicle_record(tmp_path):
    runner = _runner(tmp_path, anpr=False)
    acc = _accumulator_with_crop(runner)
    record = runner.finalize_track(acc.track_id)
    assert record["plate_status"] == PLATE_DISABLED
    assert record["plate_text"] is None
    # Operational type is suppressed; the raw candidate is still recorded.
    assert record["vehicle_type"] == "unknown"
    assert record["type_candidate"] == "car"
    assert record["type_suppressed"] is True
    assert record["vehicle_color"] == "white"
    assert record["color_state"] == "estimated"
    assert record["verified"] is False and record["review_required"] is True


def test_missing_plate_does_not_discard_the_observation(tmp_path, monkeypatch):
    runner = _runner(tmp_path, anpr=True)
    monkeypatch.setattr("app.services.cpu_anpr.cpu_plate_candidates", lambda *a, **k: [])
    acc = _accumulator_with_crop(runner)
    record = runner.finalize_track(acc.track_id)
    assert record["plate_status"] == PLATE_NOT_VISIBLE
    assert record["type_candidate"] == "car"
    assert record["vehicle_color"] == "white"


def test_ocr_exception_is_reported_not_swallowed_and_record_survives(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("onnx session died")

    monkeypatch.setattr("app.services.cpu_anpr.cpu_plate_candidates", boom)
    runner = _runner(tmp_path, anpr=True)
    acc = _accumulator_with_crop(runner)
    record = runner.finalize_track(acc.track_id)
    assert record["plate_status"] == PLATE_OCR_ERROR
    assert "onnx session died" in record["plate_error"]
    # The whole point: type and colour are untouched by the OCR failure.
    assert record["type_candidate"] == "car"
    assert record["vehicle_color"] == "white"


def test_plate_failure_cannot_change_type_or_colour(tmp_path, monkeypatch):
    runner_ok = _runner(tmp_path / "a", anpr=False)
    record_ok = runner_ok.finalize_track(_accumulator_with_crop(runner_ok).track_id)

    monkeypatch.setattr(
        "app.services.cpu_anpr.cpu_plate_candidates",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("nope")),
    )
    runner_bad = _runner(tmp_path / "b", anpr=True)
    record_bad = runner_bad.finalize_track(_accumulator_with_crop(runner_bad).track_id)

    for field in ("vehicle_type", "type_candidate", "vehicle_color", "vehicle_color_confidence"):
        assert record_ok[field] == record_bad[field]


# -- one record per track, with evidence --------------------------------


def test_one_record_per_track_not_per_frame(tmp_path):
    runner = _runner(tmp_path)
    acc = _accumulator_with_crop(runner)
    assert acc.seen_frames == 2
    assert runner.finalize_track(acc.track_id) is not None
    # A second finalize must not emit a duplicate.
    assert runner.finalize_track(acc.track_id) is None


def test_best_crop_and_context_are_written(tmp_path):
    runner = _runner(tmp_path)
    acc = _accumulator_with_crop(runner)
    record = runner.finalize_track(acc.track_id)
    assert Path(record["best_vehicle_crop"]).is_file()
    assert Path(record["context_image"]).is_file()


def test_context_crop_does_not_modify_the_original_frame():
    frame = np.full((480, 640, 3), 50, dtype=np.uint8)
    before = frame.copy()
    region = context_crop(frame, (100, 100, 80, 80))
    assert region is not None
    assert np.array_equal(frame, before)


# -- CPU fallback --------------------------------------------------------


def test_device_resolves_to_cpu_when_requested(tmp_path):
    runner = VehicleRunner(RunnerConfig(camera_id="c", run_id="r", device="cpu", evidence_dir=tmp_path))
    device, half = runner._resolve_device()
    assert device == "cpu"
    assert half is False  # FP16 is CUDA-only


def test_device_falls_back_to_cpu_without_cuda(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runner = VehicleRunner(RunnerConfig(camera_id="c", run_id="r", device="auto", evidence_dir=tmp_path))
    device, half = runner._resolve_device()
    assert device == "cpu"
    assert half is False


# -- credentials ---------------------------------------------------------


def test_rtsp_credentials_are_redacted():
    url = "rtsp://user%40x.com:s3cr3t@10.0.0.1:8554/stream/cam01"
    out = redact_source(url)
    assert "s3cr3t" not in out
    assert "user%40x.com" not in out
    assert "10.0.0.1:8554" in out


def test_redact_leaves_plain_sources_alone():
    assert redact_source("clip.mp4") == "clip.mp4"
    assert redact_source("rtsp://10.0.0.1:8554/s") == "rtsp://10.0.0.1:8554/s"


def test_unopenable_source_error_does_not_leak_credentials():
    with pytest.raises(RuntimeError) as excinfo:
        list(iter_frames("rtsp://user:hunter2@127.0.0.1:1/nope"))
    assert "hunter2" not in str(excinfo.value)


# -- modes ---------------------------------------------------------------


def test_custom_mode_names_every_missing_file_and_class_order(monkeypatch, tmp_path):
    monkeypatch.setattr(vehicle_pipeline, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        vehicle_pipeline.require_custom_weights()
    message = str(excinfo.value)
    assert "vehicle_type_classifier.pt" in message
    assert "vehicle_color_classifier.pt" in message
    assert "exact index order" in message
    # It must refuse, not quietly fall back to the pretrained path.
    assert "No substitute weights will be loaded" in message


def test_evaluate_refuses_without_a_labelled_manifest(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        vehicle_pipeline.evaluate(tmp_path / "absent.jsonl", [tmp_path / "p.jsonl"])
    assert "labelled manifest not found" in str(excinfo.value)


def _write_eval_pair(tmp_path, truth_rows, pred_rows):
    manifest = tmp_path / "truth.jsonl"
    predictions = tmp_path / "pred.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in truth_rows))
    predictions.write_text("\n".join(json.dumps(r) for r in pred_rows))
    return manifest, predictions


def test_evaluate_computes_metrics_and_abstention(tmp_path):
    manifest, predictions = _write_eval_pair(
        tmp_path,
        [
            {"track_id": "1", "vehicle_type": "car", "vehicle_color": "white"},
            {"track_id": "2", "vehicle_type": "truck", "vehicle_color": "red"},
            {"track_id": "3", "vehicle_type": "bus", "vehicle_color": "blue"},
        ],
        [
            {"track_id": "1", "vehicle_type": "car", "vehicle_color": "white"},
            {"track_id": "2", "vehicle_type": "truck", "vehicle_color": "unknown"},
            {"track_id": "3", "vehicle_type": "unknown", "vehicle_color": "blue"},
        ],
    )
    assert vehicle_pipeline.evaluate(manifest, [predictions]) == 0


def test_evaluate_separates_unsupported_truth_classes(capsys, tmp_path):
    """A truth class the model cannot represent must be scored as abstention.

    Counting `auto_rickshaw -> unknown` as an error would punish the system for
    behaving correctly; counting `auto_rickshaw -> truck` as merely a wrong
    class would hide that it asserted something it cannot know.
    """
    manifest, predictions = _write_eval_pair(
        tmp_path,
        [
            {"track_id": "1", "vehicle_type": "auto_rickshaw", "vehicle_color": "yellow"},
            {"track_id": "2", "vehicle_type": "auto_rickshaw", "vehicle_color": "yellow"},
            {"track_id": "3", "vehicle_type": "car", "vehicle_color": "white"},
        ],
        [
            {"track_id": "1", "vehicle_type": "unknown", "vehicle_color": "yellow"},
            {"track_id": "2", "vehicle_type": "truck", "vehicle_color": "yellow"},
            {"track_id": "3", "vehicle_type": "car", "vehicle_color": "white"},
        ],
    )
    vehicle_pipeline.evaluate(manifest, [predictions])
    report = json.loads(capsys.readouterr().out)
    unsupported = report["vehicle_type"]["unsupported_classes"]
    assert unsupported["n"] == 2
    assert unsupported["correct_abstention_rate"] == 0.5
    assert unsupported["false_attributions"] == 1
    # The one supported-class track was answered correctly.
    supported = report["vehicle_type"]["supported_classes"]
    assert supported["n"] == 1
    assert supported["selective_precision"] == 1.0


def test_evaluate_excludes_unclear_ground_truth(capsys, tmp_path):
    manifest, predictions = _write_eval_pair(
        tmp_path,
        [
            {"track_id": "1", "vehicle_type": "unclear", "vehicle_color": "unclear"},
            {"track_id": "2", "vehicle_type": "car", "vehicle_color": "white"},
        ],
        [
            {"track_id": "1", "vehicle_type": "truck", "vehicle_color": "red"},
            {"track_id": "2", "vehicle_type": "car", "vehicle_color": "white"},
        ],
    )
    vehicle_pipeline.evaluate(manifest, [predictions])
    report = json.loads(capsys.readouterr().out)
    assert report["vehicle_type"]["excluded_unclear"] == 1
    assert report["vehicle_type"]["supported_classes"]["n"] == 1


def test_evaluate_counts_unmatched_tracks_rather_than_dropping_them(capsys, tmp_path):
    """A labelled track with no prediction must be visible, not silently lost."""
    manifest, predictions = _write_eval_pair(
        tmp_path,
        [
            {"track_id": "1", "vehicle_type": "car", "vehicle_color": "white"},
            {"track_id": "absent", "vehicle_type": "bus", "vehicle_color": "blue"},
        ],
        [{"track_id": "1", "vehicle_type": "car", "vehicle_color": "white"}],
    )
    vehicle_pipeline.evaluate(manifest, [predictions])
    report = json.loads(capsys.readouterr().out)
    assert report["vehicle_type"]["unmatched_tracks"] == 1


def test_write_jsonl_round_trips(tmp_path):
    path = write_jsonl([{"track_id": "a"}, {"track_id": "b"}], tmp_path / "o.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["track_id"] for r in rows] == ["a", "b"]


def test_unsupported_classes_are_declared_not_emitted():
    assert set(vehicle_pipeline.UNSUPPORTED_TYPES) == {"suv", "auto_rickshaw", "taxi_cab"}
    assert "silver" in vehicle_pipeline.UNSUPPORTED_COLORS
    for label in vehicle_pipeline.UNSUPPORTED_TYPES:
        assert label not in TYPE_CLASSES
    for label in vehicle_pipeline.UNSUPPORTED_COLORS:
        assert label not in COLOR_CLASSES


# -- image-dir source ----------------------------------------------------


def test_iter_frames_reads_an_image_directory(tmp_path):
    import cv2

    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i:03d}.jpg"), np.full((60, 80, 3), 10 * i + 20, dtype=np.uint8))
    frames = list(iter_frames(str(tmp_path)))
    assert len(frames) == 3
    assert frames[0][1].shape == (60, 80, 3)
