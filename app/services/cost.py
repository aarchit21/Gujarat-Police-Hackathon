"""Capacity and cost estimator. Every result is labelled as an estimate."""
from __future__ import annotations

import math
from typing import Any


def estimate(assumptions: dict[str, Any]) -> dict[str, Any]:
    camera_count = _num(assumptions.get("camera_count"), 50)
    avg_bitrate_kbps = _num(assumptions.get("avg_bitrate_kbps"), 1500)
    target_analysis_fps = _num(assumptions.get("target_analysis_fps"), 2.0)
    active_cameras = _num(assumptions.get("active_cameras"), 2)
    measured_worker_fps = _num(assumptions.get("measured_worker_fps"), 0)
    gpu_hourly_cost = _num(assumptions.get("gpu_hourly_cost"), 0)
    storage_cost_per_gb = _num(assumptions.get("storage_cost_per_gb"), 0)
    evidence_events_per_day = _num(assumptions.get("evidence_events_per_day"), 100)
    avg_evidence_kb = _num(assumptions.get("avg_evidence_size_kb"), 40)
    jpeg_frame_kb = _num(assumptions.get("selected_frame_jpeg_kb"), 80)
    vendor_share = _clamp(_num(assumptions.get("share_vendor_metadata"), 0.1))
    local_share = _clamp(_num(assumptions.get("share_local_worker"), 0.2))
    remote_share = _clamp(_num(assumptions.get("share_remote_gpu"), 0.05))
    regional_share = _clamp(_num(assumptions.get("share_shared_regional"), 0.05))
    on_demand_share = _clamp(_num(assumptions.get("share_central_on_demand"), 0.1))
    deferred_share = _clamp(1.0 - (vendor_share + local_share + remote_share + regional_share + on_demand_share))

    source_bandwidth_mbps = camera_count * (avg_bitrate_kbps / 1000.0)

    # Central transfer: vendor metadata ~tiny; local worker sends evidence only;
    # remote_gpu sends selected JPEGs; on-demand sends selected live feeds.
    metadata_kbps = 8.0
    evidence_kbps = (evidence_events_per_day * avg_evidence_kb * 8.0) / (86400.0)
    remote_frame_kbps = remote_share * camera_count * target_analysis_fps * jpeg_frame_kb * 8.0 / 1000.0
    on_demand_live_mbps = on_demand_share * camera_count * (avg_bitrate_kbps / 1000.0)
    regional_mbps = regional_share * camera_count * (avg_bitrate_kbps / 1000.0) * 0.25
    local_central_mbps = (local_share * camera_count * evidence_kbps) / 1000.0
    vendor_central_mbps = (vendor_share * camera_count * metadata_kbps) / 1000.0
    remote_central_mbps = remote_frame_kbps / 1000.0

    central_mbps = (
        vendor_central_mbps
        + local_central_mbps
        + remote_central_mbps
        + regional_mbps
        + on_demand_live_mbps
    )

    evidence_gb_day = evidence_events_per_day * avg_evidence_kb / (1024.0 * 1024.0)
    evidence_gb_month = evidence_gb_day * 30.0

    workers_needed = None
    if measured_worker_fps > 0 and target_analysis_fps > 0:
        per_worker = measured_worker_fps / target_analysis_fps
        if per_worker > 0:
            workers_needed = int(math.ceil(active_cameras / per_worker))

    monthly_compute = None
    if workers_needed is not None and gpu_hourly_cost > 0:
        monthly_compute = workers_needed * gpu_hourly_cost * 24.0 * 30.0

    monthly_storage = evidence_gb_month * storage_cost_per_gb if storage_cost_per_gb else None

    return {
        "disclaimer": "Estimate only. Not a measured production result. Not an 80,000-camera load test.",
        "assumptions": {
            "camera_count": camera_count,
            "avg_bitrate_kbps": avg_bitrate_kbps,
            "target_analysis_fps": target_analysis_fps,
            "active_cameras": active_cameras,
            "measured_worker_fps": measured_worker_fps,
            "gpu_hourly_cost": gpu_hourly_cost,
            "storage_cost_per_gb": storage_cost_per_gb,
            "evidence_events_per_day": evidence_events_per_day,
            "avg_evidence_size_kb": avg_evidence_kb,
            "mode_shares": {
                "vendor_metadata": vendor_share,
                "local_worker": local_share,
                "remote_gpu": remote_share,
                "shared_regional": regional_share,
                "central_on_demand": on_demand_share,
                "deferred": max(0.0, deferred_share),
            },
        },
        "estimated_source_bandwidth_mbps": round(source_bandwidth_mbps, 3),
        "estimated_central_transfer_mbps": round(central_mbps, 3),
        "estimated_evidence_storage_gb_per_day": round(evidence_gb_day, 4),
        "estimated_evidence_storage_gb_per_month": round(evidence_gb_month, 3),
        "approximate_workers_from_measured_throughput": workers_needed,
        "approximate_monthly_compute_cost": monthly_compute,
        "approximate_monthly_storage_cost": monthly_storage,
        "notes": [
            "Full video is not copied into central evidence storage.",
            "Local-worker cameras transfer plate crops and metadata, not statewide video.",
            "Do not hard-code a savings percentage from these numbers.",
        ],
    }


def _num(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
