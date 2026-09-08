"""Measure local preprocessing/OCR throughput without claiming camera capacity."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.config import ROOT
from app.services.anpr import enhance_for_vision, load_bgr
from app.services.cpu_anpr import cpu_anpr_status, cpu_plate_candidates, recognize_with_tesseract


def measure(name, fn, image, iterations=20):
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn(image)
        timings.append((time.perf_counter() - started) * 1000.0)
    timings.sort()
    mean = statistics.mean(timings)
    return {
        "name": name, "iterations": iterations, "mean_ms": round(mean, 3),
        "p50_ms": round(statistics.median(timings), 3),
        "p95_ms": round(timings[int((len(timings) - 1) * 0.95)], 3),
        "operations_per_second": round(1000.0 / mean, 3) if mean else None,
    }


def main() -> int:
    plate = np.full((32, 128, 3), 125, dtype=np.uint8)
    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
    results = [
        measure("enhancement_plate", lambda x: enhance_for_vision(x, profile="plate", min_width=400), plate),
        measure("enhancement_frame", lambda x: enhance_for_vision(x, profile="frame"), frame, iterations=5),
    ]
    status = cpu_anpr_status()
    if status.get("fast_alpr_available") and status.get("models_ready"):
        results.append(measure("fast_alpr_full_frame", cpu_plate_candidates, frame, iterations=3))
        own_frame = load_bgr(ROOT / "data" / "frames" / "cam-ahmedabad" / "0000.jpg")
        if own_frame is not None:
            results.append(measure("fast_alpr_own_feed_frame", cpu_plate_candidates, own_frame, iterations=3))
        status = cpu_anpr_status()
    if status.get("tesseract_available"):
        results.append(measure("tesseract_plate", recognize_with_tesseract, plate, iterations=3))
    print(json.dumps({
        "host_measurement_only": True,
        "camera_capacity_claim": False,
        "cpu_anpr": status,
        "benchmarks": results,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
