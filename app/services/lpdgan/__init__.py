"""LPDGAN (Gong et al., IJCAI 2024) — license-plate motion deblurring.

Opt-in generator for CCTV plate crops. Default ANPR enhancement remains
deterministic CLAHE + unsharp. Do not enable without trained LPBlur weights.
"""
from __future__ import annotations

from app.services.lpdgan.config import LPDGANConfig
from app.services.lpdgan.inference import deblur_plate_bgr, lpdgan_status, reset_runtime

__all__ = [
    "LPDGANConfig",
    "deblur_plate_bgr",
    "lpdgan_status",
    "reset_runtime",
]
