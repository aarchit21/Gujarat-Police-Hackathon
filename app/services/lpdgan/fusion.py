"""Latent Fusion Module F based on Spatial Feature Transform (SFT).

Paper eq. (4)-(5) and Figure 3:

    α, β = Conv(E(x_{i+1})_{sp1} + E(x_{i+1})_{sp2})
    F_{i,i+1} = Concat(α ⊙ E(x_i) + β, E(x_i))

Lower-resolution latent codes are split along the channel axis; the two
halves are summed and mapped by 3x3 convolutions to affine parameters that
modulate the higher-resolution codes. The unmodulated higher-res codes are
re-concatenated (identity skip). F_{1,2} and F_{2,3} are then linearly mixed.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from app.services.lpdgan.swin import map_to_tokens, tokens_to_map


class SFTParamHead(nn.Module):
    """Maps a condition map to SFT scale or shift, matching Figure 3 Conv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        mid = max(out_ch, (in_ch + out_ch) // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.net(condition)


class SpatialFeatureTransform(nn.Module):
    """Fuse one higher-res latent with the next coarser latent via SFT."""

    def __init__(self, higher_dim: int, lower_dim: int) -> None:
        super().__init__()
        if lower_dim % 2 != 0:
            raise ValueError(f"lower-res channels {lower_dim} must be even to split")
        cond_dim = lower_dim // 2
        self.alpha_head = SFTParamHead(cond_dim, higher_dim)
        self.beta_head = SFTParamHead(cond_dim, higher_dim)
        self.out_dim = higher_dim * 2

    def forward(
        self,
        higher: torch.Tensor,
        lower: torch.Tensor,
        token_hw: tuple[int, int],
    ) -> torch.Tensor:
        h, w = token_hw
        higher_map = tokens_to_map(higher, h, w)
        lower_map = tokens_to_map(lower, h, w)
        split_a, split_b = torch.chunk(lower_map, 2, dim=1)
        condition = split_a + split_b
        alpha = self.alpha_head(condition)
        beta = self.beta_head(condition)
        modulated = alpha * higher_map + beta
        fused = torch.cat([modulated, higher_map], dim=1)
        return map_to_tokens(fused)


@dataclass(frozen=True)
class FusionOutput:
    fused_12: torch.Tensor  # F_{1,2}: (B, 49, 768)
    fused_23: torch.Tensor  # F_{2,3}: (B, 49, 384)
    latent: torch.Tensor  # decoder start: (B, 49, 384)
    token_hw: tuple[int, int]


class LatentFusionModule(nn.Module):
    def __init__(self, full_dim: int = 384, half_dim: int = 192, quarter_dim: int = 96) -> None:
        super().__init__()
        self.fuse_12 = SpatialFeatureTransform(higher_dim=full_dim, lower_dim=half_dim)
        self.fuse_23 = SpatialFeatureTransform(higher_dim=half_dim, lower_dim=quarter_dim)
        fused_dim = self.fuse_12.out_dim + self.fuse_23.out_dim  # 768 + 384 = 1152
        self.mix = nn.Sequential(
            nn.Linear(fused_dim, full_dim),
            nn.GELU(),
            nn.LayerNorm(full_dim),
        )
        self.out_dim = full_dim

    def forward(
        self,
        e_full: torch.Tensor,
        e_half: torch.Tensor,
        e_quarter: torch.Tensor,
        token_hw: tuple[int, int],
    ) -> FusionOutput:
        fused_12 = self.fuse_12(e_full, e_half, token_hw)
        fused_23 = self.fuse_23(e_half, e_quarter, token_hw)
        latent = self.mix(torch.cat([fused_12, fused_23], dim=-1))
        return FusionOutput(
            fused_12=fused_12,
            fused_23=fused_23,
            latent=latent,
            token_hw=token_hw,
        )
