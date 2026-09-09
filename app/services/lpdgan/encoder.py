"""Multi-scale Swin feature encoder E(x_i), i = 1,2,3.

Full / half / quarter plate crops are encoded independently so elongated
motion-ghosting can be modelled at each capture distance. All three branches
emit an aligned 7x7 token grid for the latent fusion module.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from app.services.lpdgan.config import LPDGANConfig
from app.services.lpdgan.swin import BasicLayer, PatchEmbed, init_swin_weights, stochastic_depth_rates


@dataclass(frozen=True)
class EncoderFeatures:
    full: torch.Tensor  # E(x1): (B, 49, 384)
    half: torch.Tensor  # E(x2): (B, 49, 192)
    quarter: torch.Tensor  # E(x3): (B, 49, 96)
    token_hw: tuple[int, int] = (7, 7)


class ScaleEncoder(nn.Module):
    """Hierarchical Swin encoder for one pyramid level.

    `n_merges` patch-merging stages bring every scale to a 7x7 token grid:
    full 56x56 -> 3 merges, half 28x28 -> 2 merges, quarter 14x14 -> 1 merge.
    """

    def __init__(
        self,
        img_size: tuple[int, int],
        n_merges: int,
        cfg: LPDGANConfig,
        drop_path: list[list[float]],
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim)
        h, w = self.patch_embed.patches_resolution
        stages = n_merges + 1
        layers: list[nn.Module] = []
        for i in range(stages):
            dim = int(cfg.embed_dim * 2**i)
            depth = cfg.encoder_depths[i] if i < len(cfg.encoder_depths) else cfg.encoder_depths[-1]
            heads = cfg.num_heads[i] if i < len(cfg.num_heads) else cfg.num_heads[-1]
            layers.append(
                BasicLayer(
                    dim=dim,
                    input_resolution=(h, w),
                    depth=depth,
                    num_heads=heads,
                    window_size=cfg.window_size,
                    mlp_ratio=cfg.mlp_ratio,
                    qkv_bias=cfg.qkv_bias,
                    drop=cfg.drop_rate,
                    attn_drop=cfg.attn_drop_rate,
                    drop_path=drop_path[i] if i < len(drop_path) else 0.0,
                    downsample=i < n_merges,
                )
            )
            if i < n_merges:
                h, w = h // 2, w // 2
        self.layers = nn.ModuleList(layers)
        self.out_dim = int(cfg.embed_dim * 2**n_merges)
        self.out_hw = (h, w)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MultiScaleFeatureEncoder(nn.Module):
    """Three Swin encoders: E(x1) full, E(x2) half, E(x3) quarter."""

    def __init__(self, cfg: LPDGANConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LPDGANConfig()
        dpr_full = stochastic_depth_rates(self.cfg.drop_path_rate, self.cfg.encoder_depths)
        dpr_half = stochastic_depth_rates(self.cfg.drop_path_rate, self.cfg.encoder_depths[:3])
        dpr_quarter = stochastic_depth_rates(self.cfg.drop_path_rate, self.cfg.encoder_depths[:2])
        self.enc_full = ScaleEncoder(self.cfg.full_hw, n_merges=3, cfg=self.cfg, drop_path=dpr_full)
        self.enc_half = ScaleEncoder(self.cfg.half_hw, n_merges=2, cfg=self.cfg, drop_path=dpr_half)
        self.enc_quarter = ScaleEncoder(self.cfg.quarter_hw, n_merges=1, cfg=self.cfg, drop_path=dpr_quarter)
        self.apply(init_swin_weights)

    @property
    def token_hw(self) -> tuple[int, int]:
        return self.enc_full.out_hw

    def forward(
        self,
        x_full: torch.Tensor,
        x_half: torch.Tensor,
        x_quarter: torch.Tensor,
    ) -> EncoderFeatures:
        return EncoderFeatures(
            full=self.enc_full(x_full),
            half=self.enc_half(x_half),
            quarter=self.enc_quarter(x_quarter),
            token_hw=self.token_hw,
        )
