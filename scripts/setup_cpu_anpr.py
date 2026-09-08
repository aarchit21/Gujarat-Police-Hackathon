"""Explicitly download/cache the pinned CPU ANPR models before live runtime."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.config import settings


def main() -> int:
    from fast_alpr import ALPR

    alpr = ALPR(
        detector_model=settings.cpu_anpr_detector_model,
        ocr_model=settings.cpu_anpr_ocr_model,
        detector_providers=["CPUExecutionProvider"],
        ocr_device="cpu",
        ocr_providers=["CPUExecutionProvider"],
    )
    # A tiny call forces model resolution/cache creation without needing a feed.
    alpr.predict(np.zeros((64, 128, 3), dtype=np.uint8))
    model_paths = [
        alpr.detector.detector.model._model_path,
        alpr.ocr.ocr_model.model._model_path,
    ]
    digest = hashlib.sha256()
    for model_path in model_paths:
        with open(model_path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    print(json.dumps({
        "ok": True,
        "execution_provider": "CPUExecutionProvider",
        "detector_model": settings.cpu_anpr_detector_model,
        "ocr_model": settings.cpu_anpr_ocr_model,
        "combined_model_sha256": digest.hexdigest(),
        "model_files": model_paths,
        "next_step": "Set CPU_ANPR_MODELS_READY=true only on this prepared host.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
