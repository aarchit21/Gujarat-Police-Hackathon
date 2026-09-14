"""Host facts. Does not invent GPU capacity or government-feed access."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.database import database_status  # noqa: E402
from app.security import redact_url  # noqa: E402
from app.services.ingest import opencv_version, prepare_rtsp_tcp  # noqa: E402
from app.services.network_check import host_network_report  # noqa: E402


def main() -> None:
    print("python", sys.version)
    configured = Path(settings.tesseract_cmd) if settings.tesseract_cmd else None
    tesseract = (
        str(configured)
        if configured and configured.is_file()
        else (shutil.which("tesseract") or "NOT ON PATH")
    )
    print("tesseract", tesseract)
    print("ffmpeg_executable", shutil.which("ffmpeg") or "NOT ON PATH (not required; OpenCV CAP_FFMPEG may still work)")
    print("node", shutil.which("node") or "NOT ON PATH (not required)")
    print("opencv", opencv_version())
    print("rtsp_tcp", prepare_rtsp_tcp())
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.STDOUT)
        print("gpu", out.strip())
    except Exception as exc:
        print("gpu FAIL", exc)
    print("analysis_fps_hypothesis", settings.analysis_fps)
    print("database", database_status())
    print("ingest_catalogue_url", redact_url(settings.ingest_catalogue_url) or "(not configured)")
    print("catalogue_auth_mode", settings.cctv_auth_mode or "none")
    print("cctv_token_configured", bool(settings.cctv_access_token))
    print("remote_inference_url", redact_url(settings.remote_inference_url) or "(not configured)")
    try:
        from app.services.ollama_vision import vision_status

        print("ollama_vision", vision_status())
    except Exception as exc:
        print("ollama_vision FAIL", exc)
    try:
        from app.services.yolo_detect import yolo_status

        print("yolo_detector", yolo_status())
    except Exception as exc:
        print("yolo_detector FAIL", exc)
    print("--- deterministic vehicle attributes ---")
    try:
        import torch

        cuda = torch.cuda.is_available()
        print("torch", torch.__version__, "built_cuda", torch.version.cuda, "cuda_available", cuda)
        if not cuda:
            # The exact failure this host hit: torch cu130 against a 12.2 driver.
            print("  FP16/CUDA unavailable -> detector will run on CPU. Check that the "
                  "torch CUDA build matches the installed driver (nvidia-smi).")
        else:
            print("  gpu", torch.cuda.get_device_name(0),
                  f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except Exception as exc:
        print("torch FAIL", exc)
    try:
        from app.services.vehicle_attributes import attributes_status

        status = attributes_status()
        print("vehicle_attributes", json.dumps(status, indent=2))
        if not status["weights_present"]:
            print("  run: python scripts/pull_vehicle_attributes.py")
        if not status["runtime_available"]:
            print("  run: pip install openvino")
    except Exception as exc:
        print("vehicle_attributes FAIL", exc)
    try:
        from app.services.vehicle_tracking import bytetrack_available

        ok, why = bytetrack_available()
        print("bytetrack", "available" if ok else f"UNAVAILABLE ({why}) -> weaker greedy-IoU fallback; pip install lap")
    except Exception as exc:
        print("bytetrack FAIL", exc)
    print("max_concurrent_captures", settings.max_open_captures)
    net = host_network_report(include_rtsp_probe=False)
    print("network")
    print(json.dumps(net, indent=2))
    https = net["https_cameras_json"]
    if https.get("ok"):
        print("government_feed", f"catalogue_ok count={https.get('camera_count')} (decode not implied)")
    else:
        print("government_feed", f"catalogue_blocked: {https.get('error') or 'unknown'}")


if __name__ == "__main__":
    main()
