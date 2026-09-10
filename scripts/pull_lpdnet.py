"""Download NVIDIA TAO LPDNet USA ONNX into data/models/lpdnet/.

NGC may require NGC_API_KEY in the environment if the anonymous URL 401s.
This does not enable LPDGAN and does not install DeepStream.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT, settings

USA_ONNX = "LPDNet_usa_pruned_tao5.onnx"
FILES = (
    USA_ONNX,
    "usa_cal_8.6.1.bin",
)
BASE = "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/lpdnet/pruned_v2.2/files?redirect=true&path="


def _dest() -> Path:
    folder = Path(getattr(settings, "lpdnet_weights_dir", "") or "data/models/lpdnet")
    if not folder.is_absolute():
        folder = ROOT / folder
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _get(url: str, dest: Path) -> None:
    headers = {"User-Agent": "gujarat-cctv-p0"}
    key = (os.environ.get("NGC_API_KEY") or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def main() -> int:
    folder = _dest()
    saved = []
    for name in FILES:
        dest = folder / name
        if dest.is_file() and dest.stat().st_size > 1000:
            saved.append(str(dest))
            continue
        url = BASE + name
        try:
            _get(url, dest)
        except urllib.error.HTTPError as exc:
            if dest.exists():
                dest.unlink()
            if exc.code in {401, 403}:
                print(json.dumps({
                    "ok": False,
                    "error": "ngc_auth",
                    "detail": "NGC returned 401/403. Create a free NVIDIA NGC API key and set NGC_API_KEY.",
                    "url": url,
                }, indent=2))
                return 2
            print(json.dumps({"ok": False, "error": f"HTTP {exc.code}", "url": url}, indent=2))
            return 1
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc), "url": url}, indent=2))
            return 1
        saved.append(str(dest))
    onnx = folder / USA_ONNX
    print(json.dumps({
        "ok": onnx.is_file(),
        "onnx": str(onnx),
        "bytes": onnx.stat().st_size if onnx.is_file() else 0,
        "files": saved,
        "next": "Restart uvicorn. LPDNet loads on first vehicle crop.",
    }, indent=2))
    return 0 if onnx.is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
