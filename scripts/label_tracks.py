"""Build a BLIND labelling pack from saved track crops.

Ground truth must not be written while the annotator can see the model's
answer. This script therefore reads only the crop image files and their track
ids from the filenames -- it never opens observations.jsonl, so a prediction
cannot leak onto the sheet the annotator looks at.

Outputs, per run:
  labels/<run>_sheet_NN.jpg   crops with an index and short track id only
  labels/tracks_template.jsonl one row per track with empty label fields

Fill in `vehicle_type` and `vehicle_color` in the template, save it as
tracks.jsonl, then:
  python scripts/vehicle_pipeline.py --mode evaluate \
      --manifest data/labels/tracks.jsonl --predictions data/runs/<run>/observations.jsonl

Ground truth uses the FULL canonical vocabulary, not the model's four classes.
Labelling an auto-rickshaw as `car` just to fit the model would hide the very
gap this evaluation exists to measure. Use `unclear` for a crop you genuinely
cannot call; those rows are excluded from metrics and counted separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CANONICAL_TYPES = (
    "car", "suv", "two_wheeler", "truck", "bus", "van",
    "auto_rickshaw", "taxi_cab", "unclear",
)
CANONICAL_COLORS = (
    "white", "black", "silver", "gray", "red", "blue", "green",
    "yellow", "orange", "brown", "other", "unclear",
)


def track_id_from(path: Path) -> str:
    return path.name[: -len("_vehicle.jpg")]


def collect(run_dir: Path) -> list[tuple[str, Path]]:
    crops = sorted((run_dir / "crops").glob("*_vehicle.jpg"))
    return [(track_id_from(p), p) for p in crops]


def build_sheet(items: list[tuple[int, str, Path]], out_path: Path, *, columns: int = 4, cell: int = 260) -> Path:
    """Grid of crops annotated with index + track id ONLY. No predictions."""
    rows = (len(items) + columns - 1) // columns
    label_h = 30
    sheet = np.full((rows * (cell + label_h), columns * cell, 3), 24, dtype=np.uint8)
    for slot, (index, track_id, path) in enumerate(items):
        image = cv2.imread(str(path))
        if image is None:
            continue
        r, c = divmod(slot, columns)
        y0, x0 = r * (cell + label_h), c * cell
        h, w = image.shape[:2]
        scale = min(cell / max(1, w), cell / max(1, h))
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
        rh, rw = resized.shape[:2]
        sheet[y0 : y0 + rh, x0 : x0 + rw] = resized
        cv2.putText(sheet, f"#{index}", (x0 + 4, y0 + cell + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (90, 220, 90), 1, cv2.LINE_AA)
        cv2.putText(sheet, track_id[-14:], (x0 + 52, y0 + cell + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="run directories containing crops/")
    parser.add_argument("--out", default="data/labels")
    parser.add_argument("--per-sheet", type=int, default=12)
    parser.add_argument("--cell", type=int, default=260)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    template: list[dict] = []
    sheets: list[str] = []
    index = 0

    for run in args.runs:
        run_dir = Path(run)
        items = collect(run_dir)
        if not items:
            print(f"warning: no crops under {run_dir}/crops", file=sys.stderr)
            continue
        numbered = []
        for track_id, path in items:
            index += 1
            numbered.append((index, track_id, path))
            template.append({
                "index": index,
                "track_id": track_id,
                # The source video is the split key: never mix frames of one
                # track, or one video, across evaluation groups.
                "video": run_dir.name,
                "crop": str(path),
                "vehicle_type": "",
                "vehicle_color": "",
            })
        for start in range(0, len(numbered), args.per_sheet):
            chunk = numbered[start : start + args.per_sheet]
            sheet = out_dir / f"{run_dir.name}_sheet_{start // args.per_sheet + 1:02d}.jpg"
            build_sheet(chunk, sheet, cell=args.cell)
            sheets.append(str(sheet))

    template_path = out_dir / "tracks_template.jsonl"
    with template_path.open("w", encoding="utf-8") as handle:
        for row in template:
            handle.write(json.dumps(row) + "\n")

    print(json.dumps({
        "tracks": len(template),
        "template": str(template_path),
        "sheets": sheets,
        "allowed_types": list(CANONICAL_TYPES),
        "allowed_colors": list(CANONICAL_COLORS),
        "note": "Sheets carry no model predictions. Label before looking at any observations.jsonl.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
