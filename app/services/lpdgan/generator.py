"""LPDGAN generator: encoder + latent fusion + decoder (+ text head)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.services.lpdgan.config import LPDGANConfig
from app.services.lpdgan.decoder import DecoderOutput, MultiScaleDecoder
from app.services.lpdgan.encoder import MultiScaleFeatureEncoder
from app.services.lpdgan.fusion import FusionOutput, LatentFusionModule
from app.services.lpdgan.text import TextReconstructionModule


def build_image_pyramid(
    image: torch.Tensor,
    cfg: LPDGANConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resize a BCHW plate tensor to the three LPDGAN input scales."""
    cfg = cfg or LPDGANConfig()
    full = F.interpolate(image, size=cfg.full_hw, mode="bilinear", align_corners=False)
    half = F.interpolate(image, size=cfg.half_hw, mode="bilinear", align_corners=False)
    quarter = F.interpolate(image, size=cfg.quarter_hw, mode="bilinear", align_corners=False)
    return full, half, quarter


@dataclass
class GeneratorOutput:
    y_full: torch.Tensor
    y_half: torch.Tensor
    y_quarter: torch.Tensor
    text_logits: torch.Tensor
    fused_23: torch.Tensor
    decoder_mid: torch.Tensor
    fusion: FusionOutput
    decoder: DecoderOutput


class LPDGANGenerator(nn.Module):
    """End-to-end deblurring generator G = D ∘ F ∘ E."""

    def __init__(self, cfg: LPDGANConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LPDGANConfig()
        self.encoder = MultiScaleFeatureEncoder(self.cfg)
        self.fusion = LatentFusionModule(
            full_dim=self.encoder.enc_full.out_dim,
            half_dim=self.encoder.enc_half.out_dim,
            quarter_dim=self.encoder.enc_quarter.out_dim,
        )
        self.decoder = MultiScaleDecoder(self.cfg)
        self.text_head = TextReconstructionModule(
            fused_dim=self.fusion.fuse_23.out_dim,
            decoder_dim=self.cfg.embed_dim * 4,
            fused_hw=self.encoder.token_hw,
            decoder_hw=self.decoder.hw_14,
            charset_size=len(self.cfg.charset) + 1,
            max_len=self.cfg.plate_max_len,
        )

    def forward(
        self,
        x_full: torch.Tensor,
        x_half: torch.Tensor | None = None,
        x_quarter: torch.Tensor | None = None,
    ) -> GeneratorOutput:
        if x_half is None or x_quarter is None:
            x_full, x_half, x_quarter = build_image_pyramid(x_full, self.cfg)
        encoded = self.encoder(x_full, x_half, x_quarter)
        fused = self.fusion(encoded.full, encoded.half, encoded.quarter, encoded.token_hw)
        decoded = self.decoder(fused.latent)
        text_logits = self.text_head(fused.fused_23, decoded.decoder_mid)
        return GeneratorOutput(
            y_full=decoded.y_full,
            y_half=decoded.y_half,
            y_quarter=decoded.y_quarter,
            text_logits=text_logits,
            fused_23=fused.fused_23,
            decoder_mid=decoded.decoder_mid,
            fusion=fused,
            decoder=decoded,
        )
