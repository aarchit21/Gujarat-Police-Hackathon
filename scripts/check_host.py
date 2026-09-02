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
    print("tesseract", "disabled (ANPR is Ollama vision)")
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
