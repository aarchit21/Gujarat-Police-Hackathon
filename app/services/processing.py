"""Capability-based processing route selection. No GPU required in every district."""
from __future__ import annotations

from app.config import settings
from app.models import Camera

MODES = (
    "vendor_metadata",
    "local_worker",
    "remote_gpu",
    "shared_regional",
    "central_on_demand",
    "deferred",
)

PRIORITY_CLASSES = ("A", "B", "C", "D")
ANALYTICS_POLICIES = ("continuous", "event_triggered", "scheduled", "on_demand")
NETWORK_CLASSES = ("good", "limited", "intermittent", "offline")


def select_processing_route(camera: Camera, *, remote_url: str | None = None) -> dict:
    """Choose the best available route without claiming unavailable compute."""
    mode = (camera.processing_mode or "deferred").strip()
    if mode not in MODES:
        mode = "deferred"

    remote = (remote_url if remote_url is not None else settings.remote_inference_url).strip()
    network = (camera.network_class or "offline").strip()

    if mode == "deferred" or (network == "offline" and mode not in {"local_worker", "vendor_metadata"}):
        if mode == "deferred":
            return _route("deferred", "deferred", "processing_mode is deferred")
        if network == "offline" and not _has_local_source(camera):
            return _route("deferred", "deferred", "no network and no local compute source")

    if mode == "vendor_metadata":
        return _route("vendor_metadata", "vendor_metadata", "consume authorised vendor ANPR events")

    if mode == "remote_gpu":
        if remote:
            return _route("remote_gpu", "remote_gpu", "remote inference URL is configured")
        if settings.remote_fallback_local and _has_local_source(camera):
            return _route(
                "local_worker",
                "remote_gpu_fallback_local",
                "remote URL missing; local fallback allowed",
            )
        return _route("deferred", "deferred", "remote_gpu selected but REMOTE_INFERENCE_URL is empty")

    if mode in {"local_worker", "shared_regional"}:
        if _has_local_source(camera):
            label = "shared_regional" if mode == "shared_regional" else "local_worker"
            return _route("local_worker", label, "local OpenCV/Tesseract worker")
        return _route("deferred", "deferred", "local_worker selected but no decodable local source")

    if mode == "central_on_demand":
        return _route("on_demand", "central_on_demand", "process only when an operator starts the worker")

    return _route("deferred", "deferred", "no viable processing route")


def _has_local_source(camera: Camera) -> bool:
    if camera.source_type in {"image_dir", "file"} and camera.source_uri:
        return True
    if camera.source_type in {"rtsp", "hls", "onvif"} and (
        camera.source_uri or camera.substream_uri or camera.protected_rtsp_url_or_reference or camera.hls_url
    ):
        return True
    return False


def _route(worker_kind: str, selected: str, reason: str) -> dict:
    return {
        "worker_kind": worker_kind,
        "selected": selected,
        "reason": reason,
        "analytics_may_start": worker_kind in {"local_worker", "remote_gpu"},
    }


def target_fps(camera: Camera) -> float:
    if camera.target_analysis_fps and camera.target_analysis_fps > 0:
        return float(camera.target_analysis_fps)
    return float(settings.analysis_fps)
