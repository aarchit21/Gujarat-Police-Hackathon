"""Live ingest: RTSP over TCP, HLS fallback, mixed codecs, capture registry.

Does not download government footage, seek live streams, or publish to the gateway.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from app.config import settings
from app.security import redact_url

RTSP_TCP_OPTIONS = "rtsp_transport;tcp"
CAP_FFMPEG = 1900  # cv2.CAP_FFMPEG; numeric so tests need not import cv2
CAP_PROP_POS_MSEC = 0  # cv2.CAP_PROP_POS_MSEC
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4


def prepare_rtsp_tcp() -> str:
    transport = (getattr(settings, "rtsp_transport", None) or "tcp").strip().lower() or "tcp"
    options = f"rtsp_transport;{transport}"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
    return os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]


def opencv_version() -> str:
    try:
        import cv2

        return str(cv2.__version__)
    except Exception as exc:
        return f"unavailable:{exc}"


@dataclass
class OpenedSource:
    capture: Any
    protocol: str
    url_redacted: str
    rtsp_error: str = ""
    width: int | None = None
    height: int | None = None

    def read(self) -> tuple[bool, Any, float | None]:
        ok, frame = self.capture.read()
        pts = None
        try:
            pts = float(self.capture.get(CAP_PROP_POS_MSEC))
        except Exception:
            pts = None
        return ok, frame, pts

    def release(self) -> None:
        try:
            self.capture.release()
        except Exception:
            pass


CaptureCtor = Callable[..., Any]


def _default_ctor(url: str, *args: Any) -> Any:
    import cv2

    prepare_rtsp_tcp()
    if args:
        return cv2.VideoCapture(url, *args)
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


def rtsp_url_for(camera: Any) -> str:
    return (
        (camera.substream_uri or "").strip()
        or (camera.protected_rtsp_url_or_reference or "").strip()
        or ((camera.source_uri or "").strip() if camera.source_type in {"rtsp", "onvif", "hls"} else "")
    )


def open_video_source(camera: Any, capture_ctor: CaptureCtor | None = None) -> OpenedSource:
    """Open catalogue-provided URLs only. Never construct a stream URL from camera id."""
    prepare_rtsp_tcp()
    ctor = capture_ctor or _default_ctor
    rtsp = rtsp_url_for(camera)
    hls = (camera.hls_url or "").strip()
    errors: list[str] = []

    if rtsp:
        cap = _open(ctor, rtsp, ffmpeg=True)
        if cap is not None and _is_open(cap):
            return _handle(cap, "rtsp", rtsp, "")
        errors.append(f"rtsp_failed:{_safe_err(cap)}")
        _release(cap)

    if hls:
        cap = _open(ctor, hls, ffmpeg=True)
        if cap is not None and _is_open(cap):
            rtsp_error = "; ".join(errors) if errors else ""
            return _handle(cap, "hls", hls, rtsp_error)
        errors.append(f"hls_failed:{_safe_err(cap)}")
        _release(cap)

    raise SourceOpenError("; ".join(errors) or "no rtsp or hls url on camera")


def iter_live_frames(
    camera: Any,
    *,
    open_fn: Callable[..., OpenedSource] | None = None,
    max_frames: int | None = None,
    max_seconds: float | None = None,
    stop_check: Callable[[], bool] | None = None,
    now_fn: Callable[[], float] | None = None,
):
    """Bounded live read. Does not seek, download, or retain the stream."""
    import time as _time

    opener = open_fn or open_video_source
    opened = opener(camera)
    clock = now_fn or _time.monotonic
    limit_frames = max_frames if max_frames is not None else settings.live_analyze_max_frames
    limit_seconds = max_seconds if max_seconds is not None else settings.live_analyze_max_seconds
    started = clock()
    join_started = started
    got_frame = False
    idx = 0
    try:
        while idx < limit_frames and (clock() - started) < limit_seconds:
            if stop_check and stop_check():
                break
            ok, frame, pts = opened.read()
            if not ok or frame is None:
                if not got_frame and (clock() - join_started) < settings.keyframe_wait_seconds:
                    continue
                break
            got_frame = True
            yield idx, frame, pts
            idx += 1
    finally:
        opened.release()


def _open(ctor: CaptureCtor, url: str, ffmpeg: bool) -> Any:
    try:
        if ffmpeg:
            try:
                import cv2

                return ctor(url, cv2.CAP_FFMPEG)
            except TypeError:
                return ctor(url, CAP_FFMPEG)
        return ctor(url)
    except Exception:
        try:
            return ctor(url)
        except Exception:
            return None


def _is_open(cap: Any) -> bool:
    try:
        return bool(cap.isOpened())
    except Exception:
        return False


def _release(cap: Any) -> None:
    if cap is None:
        return
    try:
        cap.release()
    except Exception:
        pass


def _safe_err(cap: Any) -> str:
    if cap is None:
        return "constructor_failed"
    return "not_opened"


def _handle(cap: Any, protocol: str, url: str, rtsp_error: str) -> OpenedSource:
    width = height = None
    try:
        width = int(cap.get(CAP_PROP_FRAME_WIDTH) or 0) or None
        height = int(cap.get(CAP_PROP_FRAME_HEIGHT) or 0) or None
    except Exception:
        pass
    return OpenedSource(
        capture=cap,
        protocol=protocol,
        url_redacted=redact_url(url),
        rtsp_error=rtsp_error,
        width=width,
        height=height,
    )


class SourceOpenError(RuntimeError):
    pass


class CaptureRegistry:
    def __init__(self, max_open: int | None = None):
        self.max_open = int(max_open or settings.max_open_captures)
        self._owners: dict[str, str] = {}

    def count(self) -> int:
        return len(self._owners)

    def owners(self) -> dict[str, str]:
        return dict(self._owners)

    def try_acquire(self, camera_id: str, role: str = "analytics") -> bool:
        if camera_id in self._owners:
            return True
        if len(self._owners) >= self.max_open:
            return False
        self._owners[camera_id] = role
        return True

    def release(self, camera_id: str) -> None:
        self._owners.pop(camera_id, None)


def diagnostics(camera: Any, *, protocol: str, error: str) -> dict:
    return {
        "camera_id": getattr(camera, "id", ""),
        "url_redacted": redact_url(rtsp_url_for(camera) or getattr(camera, "hls_url", "")),
        "protocol": protocol,
        "catalogue_live": bool(getattr(camera, "catalogue_live", False)),
        "codec": getattr(camera, "codec", "") or "",
        "width": getattr(camera, "width", None),
        "height": getattr(camera, "height", None),
        "opencv_version": opencv_version(),
        "last_pts_ms": getattr(camera, "last_pts_ms", None),
        "reconnect_count": getattr(camera, "reconnect_count", 0) or 0,
        "error": error,
    }


def resize_for_inference(bgr: Any, max_width: int | None = None) -> tuple[Any, float]:
    """Resize per-frame. Never batch mixed-resolution cameras together."""
    import cv2

    limit = max_width or settings.inference_max_width
    h, w = bgr.shape[:2]
    if w <= limit:
        return bgr, 1.0
    scale = limit / float(w)
    resized = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    return resized, scale


def scale_box(box: tuple[int, int, int, int] | None, scale: float) -> tuple[int, int, int, int] | None:
    if not box or scale == 0:
        return box
    x, y, w, h = box
    inv = 1.0 / scale
    return int(x * inv), int(y * inv), int(w * inv), int(h * inv)


def inference_plan(camera: Any, frame_shape: tuple[int, ...]) -> dict:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    scale = 1.0
    inf_w, inf_h = w, h
    if w > settings.inference_max_width:
        scale = settings.inference_max_width / float(w)
        inf_w = int(w * scale)
        inf_h = int(h * scale)
    return {
        "codec": getattr(camera, "codec", "") or "unknown",
        "source_width": w,
        "source_height": h,
        "inference_width": inf_w,
        "inference_height": inf_h,
        "scale": scale,
        "mixed_resolution_batch": False,
    }
