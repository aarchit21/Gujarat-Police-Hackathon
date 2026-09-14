"""Vehicle detection + type/colour attribution CLI. No LLM, no VLM, no network.

Modes:
  baseline  COCO YOLO detector + OpenVINO OMZ attribute model. Honestly reports
            which classes are supported; everything else abstains to `unknown`.
  custom    Fine-tuned detector/type/colour weights. Exits non-zero naming each
            missing file and its expected class order. Never falls back silently.
  evaluate  Metrics against a labelled JSONL manifest. Implemented, but no
            labelled manifest exists for this deployment yet.

Examples:
  python scripts/vehicle_pipeline.py --source clip.mp4 --out runs/a.jsonl --contact-sheet
  python scripts/vehicle_pipeline.py --camera cam01 --max-frames 300 --anpr
  python scripts/vehicle_pipeline.py --mode evaluate --manifest labels.jsonl --predictions runs/a.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT, settings  # noqa: E402
from app.services.vehicle_attributes import (  # noqa: E402
    COLOR_CLASSES,
    TYPE_CLASSES,
    AttributeModelUnavailable,
    attributes_status,
)
from app.services.vehicle_runner import (  # noqa: E402
    RunnerConfig,
    contact_sheet,
    run,
    write_jsonl,
)

# Canonical vocabulary the API/DB accepts, versus what any weight on disk can
# actually predict. The gap is reported, never quietly filled in.
UNSUPPORTED_TYPES = ("suv", "auto_rickshaw", "taxi_cab")
UNSUPPORTED_COLORS = ("silver", "brown", "orange", "other")


def redact_source(source: str) -> str:
    """Strip RTSP userinfo so credentials never reach stdout or a metrics file."""
    from app.services.vehicle_runner import redact_source as _redact

    return _redact(source)


def resolve_camera_source(camera_id: str) -> str:
    """Resolve a camera id to a playable URL, applying catalogue credentials.

    The returned string may contain RTSP userinfo. It is never printed, logged
    or written to the metrics file -- see ``redact_source``.
    """
    from sqlalchemy import select

    from app.database import SessionLocal
    from app.models import Camera
    from app.services.ingest import prepare_rtsp_tcp, rtsp_url_for

    session = SessionLocal()
    try:
        camera = session.scalar(select(Camera).where(Camera.id == camera_id))
        if camera is None:
            raise SystemExit(f"camera not found in database: {camera_id}")
        prepare_rtsp_tcp()
        source = (rtsp_url_for(camera) or "").strip() or (camera.source_uri or "").strip()
        if not source:
            raise SystemExit(f"camera {camera_id} has no usable source")
        return source
    finally:
        session.close()


def require_custom_weights() -> None:
    """custom mode: stop with a precise message rather than substitute weights."""
    expected = [
        (
            Path(settings.vattr_custom_detector_weights)
            if getattr(settings, "vattr_custom_detector_weights", "")
            else ROOT / "data" / "models" / "custom" / "vehicle_detector.pt",
            "fine-tuned vehicle detector",
            "car, suv, two_wheeler, truck, bus, van, auto_rickshaw, taxi_cab",
        ),
        (
            ROOT / "data" / "models" / "custom" / "vehicle_type_classifier.pt",
            "fine-tuned vehicle-type classifier",
            "car, suv, two_wheeler, truck, bus, van, auto_rickshaw, taxi_cab, unknown",
        ),
        (
            ROOT / "data" / "models" / "custom" / "vehicle_color_classifier.pt",
            "fine-tuned vehicle-colour classifier",
            "white, black, silver, gray, red, blue, green, yellow, orange, brown, other, unknown",
        ),
    ]
    missing = [(p, what, order) for p, what, order in expected if not Path(p).is_file()]
    if not missing:
        return
    lines = ["custom mode requires fine-tuned weights that are not on this host:", ""]
    for path, what, order in missing:
        lines.append(f"  missing : {path}")
        lines.append(f"  purpose : {what}")
        lines.append(f"  classes : {order}   (exact index order)")
        lines.append("")
    lines.append("No substitute weights will be loaded. Use --mode baseline for the")
    lines.append("pretrained path, which reports its supported classes honestly.")
    raise SystemExit("\n".join(lines))


def _predicted_value(prediction: dict, attribute: str) -> str:
    """The value to SCORE, which is the model's raw candidate.

    The operational `vehicle_type` is deliberately suppressed to `unknown` by
    the deployment gate. Scoring that would measure the safeguard, not the
    model, and would report a flattering 0% error rate for a classifier that is
    actually 50% precise. Evaluation must always see the raw candidate.
    """
    if attribute == "vehicle_type":
        candidate = prediction.get("type_candidate")
        if candidate:
            return str(candidate)
    return str(prediction.get(attribute, "unknown"))


def _score_attribute(pairs: list[tuple[str, str]], supported: set[str]) -> dict:
    """Score (truth, prediction) pairs for one attribute.

    Truth classes are partitioned, because the two groups ask different
    questions and averaging them together hides both answers:

    * supported   -- the model has this class, so it can be right or wrong.
                     Measured with coverage and selective precision.
    * unsupported -- no weight on this host has this class (auto_rickshaw,
                     suv, silver, ...), so `unknown` is the ONLY correct
                     answer. Anything else is a false attribution.
    """
    labels = sorted({t for t, _ in pairs} | {p for _, p in pairs})
    confusion = {a: {b: 0 for b in labels} for a in labels}
    for truth, pred in pairs:
        confusion[truth][pred] += 1

    in_vocab = [(t, p) for t, p in pairs if t in supported]
    out_vocab = [(t, p) for t, p in pairs if t not in supported]

    answered = [(t, p) for t, p in in_vocab if p != "unknown"]
    correct = [(t, p) for t, p in answered if t == p]

    per_class, f1s = {}, []
    for label in sorted(supported):
        tp = sum(1 for t, p in pairs if t == label and p == label)
        fp = sum(1 for t, p in pairs if t != label and p == label)
        fn = sum(1 for t, p in pairs if t == label and p != label)
        support = tp + fn
        if not (support or fp):
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "support": support,
        }
        if support:
            f1s.append(f1)

    false_attributions = [(t, p) for t, p in out_vocab if p != "unknown"]
    return {
        "supported_classes": {
            "n": len(in_vocab),
            "coverage": round(len(answered) / len(in_vocab), 4) if in_vocab else 0.0,
            "selective_precision": round(len(correct) / len(answered), 4) if answered else 0.0,
            "correct": len(correct),
            "answered": len(answered),
        },
        "unsupported_classes": {
            "n": len(out_vocab),
            "correct_abstention_rate": round(
                sum(1 for _, p in out_vocab if p == "unknown") / len(out_vocab), 4
            ) if out_vocab else 0.0,
            "false_attributions": len(false_attributions),
            "false_attribution_examples": [
                {"truth": t, "predicted": p} for t, p in false_attributions[:10]
            ],
        },
        "per_class": per_class,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "confusion": confusion,
    }


def _threshold_sweep(rows: list[dict], truths: dict, attribute: str, prob_key: str, supported: set[str]) -> list[dict]:
    """Coverage vs selective precision as the minimum-probability gate moves.

    Re-scores the stored aggregated probabilities offline. Only the minimum
    probability is swept; the margin, agreement and conflict gates stay where
    they are, so this shows the effect of one knob rather than a search over
    all of them.
    """
    out = []
    for threshold in [round(0.05 * i, 2) for i in range(1, 20)]:
        answered = correct = 0
        for row in rows:
            truth = truths.get(row["track_id"])
            if truth is None or truth not in supported:
                continue
            probs = (row.get("probabilities") or {}).get(prob_key) or {}
            if not probs:
                continue
            label, prob = max(probs.items(), key=lambda kv: kv[1])
            # Respect the gates that actually fired, then apply the new floor.
            blocked = row.get(f"{attribute.split('_')[1]}_reason") in {
                "detector_classifier_conflict", "temporal_disagreement",
                "top_two_margin_too_small", "too_few_observations", "no_usable_crop",
                "two_wheeler_out_of_distribution",
            }
            if blocked or prob < threshold:
                continue
            answered += 1
            correct += int(label == truth)
        total = sum(1 for r in rows if truths.get(r["track_id"]) in supported)
        out.append({
            "min_prob": threshold,
            "coverage": round(answered / total, 4) if total else 0.0,
            "selective_precision": round(correct / answered, 4) if answered else None,
            "answered": answered,
        })
    return out


def evaluate(manifest: Path, prediction_paths: list[Path]) -> int:
    """Per-class P/R/F1, coverage, selective precision and threshold sweep."""
    if not manifest.is_file():
        raise SystemExit(
            f"labelled manifest not found: {manifest}\n"
            "Expected JSONL with one object per TRACK: "
            '{"track_id": "...", "vehicle_type": "...", "vehicle_color": "..."}\n'
            "Split by video/camera/track identity -- never put adjacent frames of "
            "one track in both training and test."
        )
    missing = [str(p) for p in prediction_paths if not p.is_file()]
    if missing:
        raise SystemExit("predictions JSONL not found: " + ", ".join(missing))

    truth_rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    predicted: dict[str, dict] = {}
    for path in prediction_paths:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                predicted[str(row["track_id"])] = row

    # two_wheeler is detector-sourced, so it counts as supported for type.
    supported_types = set(TYPE_CLASSES) | {"two_wheeler"}
    supported_colors = set(COLOR_CLASSES)

    report = {
        "manifest": str(manifest),
        "predictions": [str(p) for p in prediction_paths],
        "tracks_labelled": len(truth_rows),
        "annotator": truth_rows[0].get("annotator", "unspecified") if truth_rows else "",
    }

    for attribute, supported, prob_key in (
        ("vehicle_type", supported_types, "type"),
        ("vehicle_color", supported_colors, "color"),
    ):
        # "unclear" means the annotator could not call the crop; scoring the
        # model against a label a human could not read would be meaningless.
        usable = [r for r in truth_rows if r.get(attribute) not in {"", "unclear"}]
        pairs, unmatched = [], 0
        for row in usable:
            prediction = predicted.get(str(row["track_id"]))
            if prediction is None:
                unmatched += 1
                continue
            pairs.append((str(row[attribute]), _predicted_value(prediction, attribute)))

        scored = _score_attribute(pairs, supported)
        scored["excluded_unclear"] = len(truth_rows) - len(usable)
        scored["unmatched_tracks"] = unmatched
        truths = {str(r["track_id"]): str(r[attribute]) for r in usable}
        scored["threshold_sweep"] = _threshold_sweep(
            list(predicted.values()), truths, attribute, prob_key, supported
        )

        by_video: dict[str, list] = {}
        for row in usable:
            prediction = predicted.get(str(row["track_id"]))
            if prediction is not None:
                by_video.setdefault(row.get("video", "?"), []).append(
                    (str(row[attribute]), _predicted_value(prediction, attribute))
                )
        scored["by_video"] = {
            video: _score_attribute(group, supported)["supported_classes"]
            for video, group in sorted(by_video.items())
        }
        report[attribute] = scored

    print(json.dumps(report, indent=2))
    return 0


def export_reviews(out_path: Path, *, min_reviews: int = 1) -> int:
    """Dump human-verified observations as a fine-tuning manifest.

    Exports only what a person actually confirmed, alongside the model's
    original answer so corrections stay auditable. It does NOT train anything.

    Two warnings are emitted deliberately rather than left implicit:

    * a crop rejected as `multi_vehicle_crop` should not become training data,
      because labelling a two-vehicle crop teaches the wrong thing;
    * `video` is the split key. Never put crops of one track, or one video, in
      both train and test -- adjacent frames of a vehicle are near-duplicates
      and will inflate accuracy.
    """
    from sqlalchemy import select

    from app.database import SessionLocal
    from app.models import VehicleObservation

    session = SessionLocal()
    try:
        rows = list(session.scalars(
            select(VehicleObservation).where(VehicleObservation.review_status == "verified")
        ))
    finally:
        session.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = skipped_no_crop = contaminated = 0
    abstentions = blind = assisted = unknown_provenance = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            crop = (row.evidence_path or "").strip()
            if not crop:
                skipped_no_crop += 1
                continue
            attributes = (row.metadata_json or {}).get("attributes") or {}
            quality = (row.metadata_json or {}).get("quality") or {}
            foreign = float(quality.get("foreign_fraction") or 0.0)
            if foreign > 0.0:
                contaminated += 1
            # How the label was produced. A verdict recorded with the model's
            # answer on screen is confirmation-prone: the annotator agrees with
            # "truck 0.83" more often than they would have unprompted, so those
            # labels reproduce the model's systematic errors. Fine for training,
            # not for re-measuring accuracy -- and the only way to honour that
            # distinction later is to record it now.
            history = (row.metadata_json or {}).get("review_history") or []
            last = history[-1] if history else {}
            context = last.get("context") or {}
            note = str(last.get("note") or "")
            annotator = note.split("annotator:", 1)[1].strip() if "annotator:" in note else ""
            # An "unclear" verdict is an abstention, not a training target. It
            # reaches the column as the literal "unknown", which is truthy and
            # therefore marks the row verified -- so it has to be excluded here
            # or it teaches the model that this vehicle IS an unknown.
            label_type = row.verified_vehicle_type or ""
            label_color = row.verified_vehicle_color or ""
            usable = bool(
                (label_type and label_type != "unknown")
                or (label_color and label_color != "unknown")
            )
            if not usable:
                abstentions += 1
            # Three buckets, not two. A row with no recorded context was labelled
            # before provenance was captured -- that is UNKNOWN provenance, not
            # blind. Absence of evidence is not evidence of blindness, and
            # counting those as blind would overstate how much of the corpus is
            # safe to evaluate on.
            provenance = ("assisted" if context.get("prediction_visible")
                          else "blind" if context else "unknown")
            if provenance == "assisted":
                assisted += 1
            elif provenance == "blind":
                blind += 1
            else:
                unknown_provenance += 1
            handle.write(json.dumps({
                "usable": usable,
                "provenance": provenance,
                "prediction_visible": bool(context.get("prediction_visible")),
                "accepted_model": bool(context.get("accepted_model")),
                "context_crop_used": bool(context.get("context_crop_used")),
                "annotator": annotator or row.verified_by,
                "labelling_ui": context.get("ui", ""),
                "scope": context.get("scope") or {},
                "crop": crop,
                "context": row.context_evidence_path or "",
                "camera_id": row.camera_id,
                # Split key: group by this, never shuffle rows.
                "video": f"{row.camera_id}:{row.run_id}",
                "track_id": row.track_id,
                "observed_at": row.first_seen_at.isoformat() if row.first_seen_at else None,
                "bbox": [row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h],
                "frame_size": [row.frame_width, row.frame_height],
                "label_vehicle_type": row.verified_vehicle_type or "",
                "label_vehicle_color": row.verified_vehicle_color or "",
                "verified_by": row.verified_by,
                "verified_at": row.verified_at.isoformat() if row.verified_at else None,
                # What the model said, so a correction is auditable.
                "model_type_candidate": attributes.get("type_candidate", ""),
                "model_type_confidence": attributes.get("type_candidate_confidence", 0.0),
                "model_color_candidate": attributes.get("color_candidate", ""),
                "model_color_confidence": attributes.get("color_candidate_confidence", 0.0),
                "model_source": attributes.get("type_model_source", ""),
                "crop_foreign_fraction": foreign,
            }) + "\n")
            written += 1

    warnings = [
        "Split by `video` (or track_id). Adjacent frames of one vehicle are "
        "near-duplicates; splitting them across train and test inflates accuracy.",
        "Crops with crop_foreign_fraction > 0 contain part of another vehicle. "
        "Filter them out before training or the labels teach the wrong thing.",
        f"{written} verified records is far below what fine-tuning needs "
        "(low thousands, class-balanced). This export is a starting point, not a dataset.",
    ]
    if abstentions:
        warnings.append(
            f"{abstentions} row(s) carry an `unknown` verdict and are marked usable=false. "
            "An abstention is not a training target; drop them before training."
        )
    if unknown_provenance:
        warnings.append(
            f"{unknown_provenance} label(s) predate provenance recording, so it is not "
            "known whether the annotator could see the model's answer. Treat them as "
            "possibly model-assisted rather than as blind ground truth."
        )
    if assisted:
        warnings.append(
            f"{assisted} of {written} label(s) were recorded with the model's own answer "
            "visible (prediction_visible=true) and are confirmation-prone: they tend to "
            "reproduce the model's systematic errors. Use them to FINE-TUNE, never to "
            "re-measure accuracy. The frozen blind set in data/labels/tracks.jsonl "
            "remains the measurement."
        )
    print(json.dumps({
        "exported": written,
        "usable": written - abstentions,
        "abstentions_unknown": abstentions,
        "blind_labels": blind,
        "model_assisted_labels": assisted,
        "unknown_provenance_labels": unknown_provenance,
        "manifest": str(out_path),
        "skipped_missing_crop": skipped_no_crop,
        "crops_with_foreign_vehicle_content": contaminated,
        "split_key": "video",
        "warnings": warnings,
        "trained": False,
    }, indent=2))
    return 0 if written >= min_reviews else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("baseline", "custom", "evaluate", "export-reviews"), default="baseline")
    parser.add_argument("--source", help="video file, RTSP URL, or directory of images")
    parser.add_argument("--camera", help="camera id; resolves source_uri from the database")
    parser.add_argument("--camera-id", default="", help="label written into records (defaults to --camera)")
    parser.add_argument("--run-id", default="", help="run identifier; defaults to a timestamp")
    parser.add_argument("--out", default="data/runs/observations.jsonl")
    parser.add_argument("--evidence-dir", default="")
    parser.add_argument("--contact-sheet", action="store_true")
    parser.add_argument("--record", default="", help="also write the decoded frames to this mp4")
    parser.add_argument("--anpr", action="store_true", help="enable optional plate recognition")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda:0")
    parser.add_argument("--manifest", default="", help="evaluate mode: labelled JSONL")
    parser.add_argument("--predictions", nargs="*", default=[],
                        help="evaluate mode: one or more predictions JSONL files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode == "export-reviews":
        return export_reviews(Path(args.out if args.out != "data/runs/observations.jsonl"
                                   else "data/labels/reviews_export.jsonl"))
    if args.mode == "evaluate":
        paths = [Path(p) for p in (args.predictions or [args.out])]
        return evaluate(Path(args.manifest or "data/labels/tracks.jsonl"), paths)
    if args.mode == "custom":
        require_custom_weights()

    if not args.source and not args.camera:
        raise SystemExit("one of --source or --camera is required")
    source = args.source or resolve_camera_source(args.camera)

    from datetime import datetime, timezone

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = Path(args.out)
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else out_path.parent / "crops"

    config = RunnerConfig(
        camera_id=args.camera_id or args.camera or "camera-01",
        run_id=run_id,
        device=args.device,
        imgsz=int(args.imgsz or settings.yolo_imgsz or 640),
        stride=max(1, int(args.stride)),
        max_frames=max(0, int(args.max_frames)),
        anpr=bool(args.anpr),
        evidence_dir=evidence_dir,
        track_timeout_seconds=float(settings.vattr_track_timeout_seconds),
    )

    status = attributes_status()
    print(f"attribute model : {status['model_id']}  ({status['licence']})", file=sys.stderr)
    print(f"supported types : {', '.join(status['supported_types'])} (+ two_wheeler from COCO)", file=sys.stderr)
    print(f"supported colours: {', '.join(status['supported_colors'])}", file=sys.stderr)
    print(f"NOT supported   : types {', '.join(UNSUPPORTED_TYPES)}; "
          f"colours {', '.join(UNSUPPORTED_COLORS)} -> always unknown", file=sys.stderr)

    try:
        records, metrics = run(source, config, record_video=Path(args.record) if args.record else None)
    except AttributeModelUnavailable as exc:
        raise SystemExit(str(exc)) from exc

    write_jsonl(records, out_path)
    metrics_path = out_path.with_name(out_path.stem + "_metrics.json")
    payload = {
        **metrics,
        "mode": args.mode,
        "source": redact_source(source),
        "attribute_model": status["model_id"],
        "attribute_model_hash": status.get("model_hash", ""),
        "supported_types": status["supported_types"] + ["two_wheeler"],
        "supported_colors": status["supported_colors"],
        "unsupported_types": list(UNSUPPORTED_TYPES),
        "unsupported_colors": list(UNSUPPORTED_COLORS),
        "accuracy_measured": False,
        "accuracy_note": (
            "No labelled validation data exists for this deployment. "
            "These are throughput numbers only, not accuracy."
        ),
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")

    sheet = None
    if args.contact_sheet:
        sheet = contact_sheet(records, out_path.with_name(out_path.stem + "_contact_sheet.jpg"))

    print(json.dumps({
        "records": len(records),
        "jsonl": str(out_path),
        "metrics": str(metrics_path),
        "contact_sheet": str(sheet) if sheet else None,
        "tracker": metrics.get("tracker"),
        "device": metrics.get("device"),
        "fp16_verified": metrics.get("fp16_verified"),
        "processed_fps_wallclock": metrics.get("processed_fps_wallclock"),
        "compute_fps_detect_only": metrics.get("compute_fps_detect_only"),
        "peak_vram_allocated_mb": metrics.get("peak_vram_allocated_mb"),
        "abstained_type": sum(1 for r in records if r.get("vehicle_type") == "unknown"),
        "abstained_color": sum(1 for r in records if r.get("vehicle_color") == "unknown"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
