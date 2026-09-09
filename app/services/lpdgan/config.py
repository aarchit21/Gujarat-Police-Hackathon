"""LPDGAN defaults from Gong et al., IJCAI 2024 (arXiv:2404.13677).

Input crops are 112x224 (H x W) with a 3-scale pyramid. Swin patch size is
rectangular (2, 4) so the token grid is square (56x56 at full resolution).
"""
from __future__ import annotations

from dataclasses import dataclass

# Paper implementation details: (H, W).
FULL_HW = (112, 224)
HALF_HW = (56, 112)
QUARTER_HW = (28, 56)
PATCH_SIZE = (2, 4)
WINDOW_SIZE = 7
EMBED_DIM = 48
ENCODER_DEPTHS = (2, 2, 2, 2)
DECODER_DEPTHS = (2, 2, 6, 2)
NUM_HEADS = (3, 6, 12, 24)
MLP_RATIO = 4.0
DROP_PATH_RATE = 0.2

# Indian ANPR charset for the text reconstruction head (no faces / no FRS).
INDIAN_PLATE_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PLATE_MAX_LEN = 10
EARLY_PARTITION_COUNT = 7
LATE_PARTITION_COUNT = 3


@dataclass(frozen=True)
class LPDGANConfig:
    full_hw: tuple[int, int] = FULL_HW
    half_hw: tuple[int, int] = HALF_HW
    quarter_hw: tuple[int, int] = QUARTER_HW
    patch_size: tuple[int, int] = PATCH_SIZE
    window_size: int = WINDOW_SIZE
    embed_dim: int = EMBED_DIM
    encoder_depths: tuple[int, ...] = ENCODER_DEPTHS
    decoder_depths: tuple[int, ...] = DECODER_DEPTHS
    num_heads: tuple[int, ...] = NUM_HEADS
    mlp_ratio: float = MLP_RATIO
    drop_path_rate: float = DROP_PATH_RATE
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    qkv_bias: bool = True
    in_chans: int = 3
    out_chans: int = 3
    charset: str = INDIAN_PLATE_CHARSET
    plate_max_len: int = PLATE_MAX_LEN
    early_partition_count: int = EARLY_PARTITION_COUNT
    late_partition_count: int = LATE_PARTITION_COUNT
    lambda_l1: float = 1.0
    lambda_per: float = 0.01
    lambda_g: float = 1.0
    lambda_p: float = 1.0
    lambda_t: float = 0.1
    lambda_gp: float = 10.0
    ndf: int = 64
    disc_layers: int = 3
