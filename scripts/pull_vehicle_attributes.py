"""Download the Intel OMZ vehicle-attributes-recognition-barrier-0042 IR.

Apache-2.0 (openvinotoolkit/open_model_zoo).  This is a *pretrained barrier /
toll-gate* attribute model.  It is not trained on Indian CCTV and this script
makes no accuracy claim for this deployment.

Idempotent: re-running verifies the on-disk SHA-256 and skips the download.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT  # noqa: E402

MODEL_NAME = "vehicle-attributes-recognition-barrier-0042"
PRECISION = "FP16"
BASE_URL = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/"
    f"models_bin/1/{MODEL_NAME}/{PRECISION}"
)
# Published sizes, used only as a sanity check before hashing.
EXPECTED_BYTES = {f"{MODEL_NAME}.xml": 91_261, f"{MODEL_NAME}.bin": 22_354_764}

LICENCE = "Apache-2.0 (openvinotoolkit/open_model_zoo)"
# Authoritative output order, from the model README.  NOT the alphabetical
# label_map in accuracy-check.yml -- see app/services/vehicle_attributes.py.
COLOR_CLASSES = ("white", "gray", "yellow", "red", "green", "blue", "black")
TYPE_CLASSES = ("car", "van", "truck", "bus")


def target_dir() -> Path:
    return ROOT / "data" / "models" / MODEL_NAME / PRECISION


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(name: str, dest: Path) -> None:
    url = f"{BASE_URL}/{name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        tmp.write_bytes(response.read())
    size = tmp.stat().st_size
    expected = EXPECTED_BYTES.get(name)
    if expected and size != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: expected {expected} bytes, got {size}. Refusing to install.")
    tmp.replace(dest)


def main() -> int:
    out = target_dir()
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for name in (f"{MODEL_NAME}.xml", f"{MODEL_NAME}.bin"):
        dest = out / name
        if not dest.is_file() or dest.stat().st_size != EXPECTED_BYTES.get(name, -1):
            _download(name, dest)
        else:
            print(f"present  {dest}")
        files[name] = sha256_of(dest)

    manifest = {
        "model_name": MODEL_NAME,
        "precision": PRECISION,
        "source_url": BASE_URL,
        "licence": LICENCE,
        "input": {"name": "input", "shape": [1, 3, 72, 72], "layout": "NCHW", "color_order": "BGR"},
        "outputs": {"color": list(COLOR_CLASSES), "type": list(TYPE_CLASSES)},
        "preprocessing": "extend_around_rect 0.3 -> crop_rect -> resize 115 -> center crop 72; no mean/scale",
        "sha256": files,
        "vendor_accuracy_note": (
            "Vendor-reported on their barrier/toll dataset: type avg 87.34% (bus 68.57%), "
            "colour avg 82.71% (yellow 61.50%). Not measured on Indian CCTV."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"dir": str(out), "sha256": files, "licence": LICENCE}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
