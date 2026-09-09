"""Multi-scale Swin decoder with patch expanding.

Starts from the fused 7x7x384 latent and emits sharp plates at the three
pyramid sizes used by the encoder: 28x56, 56x112, and 112x224.
The 14x14 decoder stage is exposed as F_D for the text reconstruction module.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from app.services.lpdgan.config import LPDGANConfig
from app.services.lpdgan.swin import BasicLayer, FinalPatchExpand, PatchExpand, init_swin_weights, stochastic_depth_rates


class RGBHead(nn.Module):
    def __init__(self, token_hw: tuple[int, int], dim: int, out_chans: int = 3) -> None:
        super().__init__()
        self.expand = FinalPatchExpand(token_hw, dim)
        self.proj = nn.Conv2d(dim, out_chans, kernel_size=1, bias=False)
        self.act = nn.Tanh()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.expand(tokens).permute(0, 3, 1, 2).contiguous()
        return self.act(self.proj(x))


@dataclass(frozen=True)
class DecoderOutput:
    y_full: torch.Tensor  # ỹ1 (B, 3, 112, 224)
    y_half: torch.Tensor  # ỹ2 (B, 3, 56, 112)
    y_quarter: torch.Tensor  # ỹ3 (B, 3, 28, 56)
    decoder_mid: torch.Tensor  # F_D (B, 196, 192) at 14x14
    decoder_mid_hw: tuple[int, int]


class MultiScaleDecoder(nn.Module):
    def __init__(self, cfg: LPDGANConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LPDGANConfig()
        # Token grid after patch embed of the full image, then 3 merges.
        full_tokens = (
            self.cfg.full_hw[0] // self.cfg.patch_size[0],
            self.cfg.full_hw[1] // self.cfg.patch_size[1],
        )
        d7 = (full_tokens[0] // 8, full_tokens[1] // 8)  # 7x7
        d14 = (d7[0] * 2, d7[1] * 2)
        d28 = (d14[0] * 2, d14[1] * 2)
        d56 = (d28[0] * 2, d28[1] * 2)
        self.hw_7, self.hw_14, self.hw_28, self.hw_56 = d7, d14, d28, d56

        depths = self.cfg.decoder_depths
        heads = self.cfg.num_heads
        dpr = stochastic_depth_rates(self.cfg.drop_path_rate, depths)
        # decoder_depths is coarse-to-fine in config ([2,2,6,2]); apply finest last.
        dim384 = self.cfg.embed_dim * 8
        dim192 = self.cfg.embed_dim * 4
        dim96 = self.cfg.embed_dim * 2
        dim48 = self.cfg.embed_dim

        self.stage_7 = BasicLayer(
            dim=dim384,
            input_resolution=d7,
            depth=depths[3],
            num_heads=heads[3],
            window_size=self.cfg.window_size,
            mlp_ratio=self.cfg.mlp_ratio,
            qkv_bias=self.cfg.qkv_bias,
            drop=self.cfg.drop_rate,
            attn_drop=self.cfg.attn_drop_rate,
            drop_path=dpr[3],
        )
        self.expand_7_to_14 = PatchExpand(d7, dim384)
        self.stage_14 = BasicLayer(
            dim=dim192,
            input_resolution=d14,
            depth=depths[2],
            num_heads=heads[2],
            window_size=self.cfg.window_size,
            mlp_ratio=self.cfg.mlp_ratio,
            qkv_bias=self.cfg.qkv_bias,
            drop=self.cfg.drop_rate,
            attn_drop=self.cfg.attn_drop_rate,
            drop_path=dpr[2],
        )
        self.expand_14_to_28 = PatchExpand(d14, dim192)
        self.stage_28 = BasicLayer(
            dim=dim96,
            input_resolution=d28,
            depth=depths[1],
            num_heads=heads[1],
            window_size=self.cfg.window_size,
            mlp_ratio=self.cfg.mlp_ratio,
            qkv_bias=self.cfg.qkv_bias,
            drop=self.cfg.drop_rate,
            attn_drop=self.cfg.attn_drop_rate,
            drop_path=dpr[1],
        )
        self.expand_28_to_56 = PatchExpand(d28, dim96)
        self.stage_56 = BasicLayer(
            dim=dim48,
            input_resolution=d56,
            depth=depths[0],
            num_heads=heads[0],
            window_size=self.cfg.window_size,
            mlp_ratio=self.cfg.mlp_ratio,
            qkv_bias=self.cfg.qkv_bias,
            drop=self.cfg.drop_rate,
            attn_drop=self.cfg.attn_drop_rate,
            drop_path=dpr[0],
        )
        self.head_quarter = RGBHead(d14, dim192, self.cfg.out_chans)
        self.head_half = RGBHead(d28, dim96, self.cfg.out_chans)
        self.head_full = RGBHead(d56, dim48, self.cfg.out_chans)
        self.norm_mid = nn.LayerNorm(dim192)
        self.apply(init_swin_weights)

    def forward(self, latent: torch.Tensor) -> DecoderOutput:
        x = self.stage_7(latent)
        x = self.expand_7_to_14(x)
        x = self.stage_14(x)
        decoder_mid = self.norm_mid(x)
        y_quarter = self.head_quarter(x)
        x = self.expand_14_to_28(x)
        x = self.stage_28(x)
        y_half = self.head_half(x)
        x = self.expand_28_to_56(x)
        x = self.stage_56(x)
        y_full = self.head_full(x)
        return DecoderOutput(
            y_full=y_full,
            y_half=y_half,
            y_quarter=y_quarter,
            decoder_mid=decoder_mid,
            decoder_mid_hw=self.hw_14,
        )
