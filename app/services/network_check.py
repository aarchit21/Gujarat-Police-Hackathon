"""Safe reachability checks for the organiser catalogue and RTSP gateway.

Does not scan arbitrary hosts or ports. Never logs credentials.
"""
from __future__ import annotations

import socket
import threading
from urllib.parse import urlparse

from app.config import settings
from app.security import redact_secrets, redact_url
from app.services.catalogue import DEFAULT_CATALOGUE_URL, fetch_catalogue
from app.services.ingest import opencv_version, prepare_rtsp_tcp

RTSP_HOST = "103.250.160.189"
RTSP_PORT = 8554
WHEP_PORT = 8889


def _tcp(host: str, port: int, timeout: float = 5.0) -> dict:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return {"ok": True, "host": host, "port": port, "error": ""}
    except OSError as exc:
        return {"ok": False, "host": host, "port": port, "error": redact_secrets(str(exc))}


def _dns(host: str) -> dict:
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        addrs = sorted({item[4][0] for item in infos})
        return {"ok": True, "host": host, "addresses": addrs, "error": ""}
    except OSError as exc:
        return {"ok": False, "host": host, "addresses": [], "error": redact_secrets(str(exc))}


def probe_one_rtsp(url: str, *, timeout: float = 20.0) -> dict:
    """Open one catalogue-provided RTSP URL and try to read a single frame.

    FFmpeg often reports isOpened()=False for a few seconds. Wait for a frame.
    """
    import time as _time

    prepare_rtsp_tcp()
    wait = min(float(timeout or 20.0), max(3.0, float(getattr(settings, "rtsp_open_wait_seconds", 6.0) or 6.0)))
    box: dict = {
        "ok": False,
        "opened": False,
        "frame": False,
        "pts_ms": None,
        "width": None,
        "height": None,
        "backend": opencv_version(),
        "url_redacted": redact_url(url),
        "error": "",
        "protocol": "rtsp",
    }

    def run() -> None:
        cap = None
        try:
            import cv2

            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            deadline = _time.monotonic() + wait
            frame = None
            ok = False
            while _time.monotonic() < deadline:
                box["opened"] = bool(cap.isOpened())
                if box["opened"]:
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        break
                _time.sleep(0.2)
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) if box["opened"] else None
            box["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
            box["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
            box["frame"] = bool(ok and frame is not None)
            box["pts_ms"] = float(pts) if pts is not None else None
            box["ok"] = box["frame"]
            if box["frame"]:
                box["error"] = ""
            elif not box["opened"]:
                box["error"] = "VideoCapture not opened"
            else:
                box["error"] = "no usable frame"
        except Exception as exc:
            box["error"] = redact_secrets(str(exc))
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        box["error"] = f"timeout after {timeout}s waiting for a frame"
        box["ok"] = False
    return box


def host_network_report(*, include_rtsp_probe: bool = False) -> dict:
    url = settings.ingest_catalogue_url or DEFAULT_CATALOGUE_URL
    parsed = urlparse(url)
    host = parsed.hostname or "cctv.corp8.cloud"
    dns = _dns(host)
    catalogue = fetch_catalogue(url)
    report = {
        "catalogue_url": redact_url(url),
        "catalogue_host": host,
        "catalogue_auth_mode": (settings.cctv_auth_mode or "none"),
        "dns_cctv_corp8_cloud": dns,
        "https_cameras_json": {
            "ok": catalogue["ok"],
            "status_code": catalogue.get("status_code"),
            "camera_count": catalogue.get("camera_count", 0),
            "error": catalogue.get("error") or "",
        },
        "tcp_rtsp_8554": _tcp(RTSP_HOST, RTSP_PORT),
        "tcp_whep_8889": _tcp(RTSP_HOST, WHEP_PORT),
        "opencv": opencv_version(),
        "rtsp_tcp": prepare_rtsp_tcp(),
        "rtsp_probe": None,
    }
    if include_rtsp_probe and catalogue.get("ok"):
        from app.services.catalogue import parse_catalogue

        items = parse_catalogue(catalogue.get("payload"))
        first = next((item for item in items if item.get("rtsp_url")), None)
        if first:
            report["rtsp_probe"] = probe_one_rtsp(first["rtsp_url"])
            report["rtsp_probe"]["camera_id"] = first["catalogue_camera_id"]
            report["rtsp_probe"]["codec"] = first.get("codec") or ""
        else:
            report["rtsp_probe"] = {"ok": False, "error": "catalogue returned no RTSP URL"}
    return report
