"""Swin Transformer primitives used by LPDGAN (Liu et al., ICCV 2021).

Independent implementation of window-MSA, patch merging, and patch expanding.
No timm / einops / yacs dependency.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_2tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(x, tuple):
        return int(x[0]), int(x[1])
    return int(x), int(x)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask.div(keep)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, drop: float = 0.0) -> None:
        super().__init__()
        hidden = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (num_windows*B, window, window, C)."""
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """(num_windows*B, window, window, C) -> (B, H, W, C)."""
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


def _relative_position_index(window_size: tuple[int, int]) -> torch.Tensor:
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative = relative.permute(1, 2, 0).contiguous()
    relative[:, :, 0] += window_size[0] - 1
    relative[:, :, 1] += window_size[1] - 1
    relative[:, :, 0] *= 2 * window_size[1] - 1
    return relative.sum(-1)


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        self.register_buffer("relative_position_index", _relative_position_index(window_size), persistent=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        ).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            n_w = mask.shape[0]
            attn = attn.view(b_ // n_w, n_w, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj_drop(self.proj(x))


def _shift_mask(resolution: tuple[int, int], window_size: int, shift_size: int) -> torch.Tensor:
    h, w = resolution
    img_mask = torch.zeros((1, h, w, 1))
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    cnt = 0
    for hs in h_slices:
        for ws in w_slices:
            img_mask[:, hs, ws, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size).view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        if self.shift_size > 0:
            mask = _shift_mask(input_resolution, self.window_size, self.shift_size)
        else:
            mask = None
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.input_resolution
        b, length, c = x.shape
        if length != h * w:
            raise ValueError(f"token length {length} != {h}*{w}")
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x_windows = window_partition(x, self.window_size).view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        x = window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(b, h * w, c)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: tuple[int, int], dim: int) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.input_resolution
        b, length, c = x.shape
        if length != h * w:
            raise ValueError(f"token length {length} != {h}*{w}")
        if h % 2 or w % 2:
            raise ValueError(f"cannot merge odd spatial size {h}x{w}")
        x = x.view(b, h, w, c)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1).view(b, -1, 4 * c)
        return self.reduction(self.norm(x))


class PatchExpand(nn.Module):
    """2x spatial upsample that halves channel count (Swin-Unet)."""

    def __init__(self, input_resolution: tuple[int, int], dim: int) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.input_resolution
        x = self.expand(x)
        b, length, c = x.shape
        if length != h * w:
            raise ValueError(f"token length {length} != {h}*{w}")
        x = x.view(b, h, w, 2, 2, c // 4)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h * 2 * w * 2, c // 4)
        return self.norm(x)


class FinalPatchExpand(nn.Module):
    """Undo rectangular patch embed (2, 4): 2x height, 4x width, keep channels."""

    def __init__(self, input_resolution: tuple[int, int], dim: int) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 8 * dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.input_resolution
        x = self.expand(x)
        b, length, c = x.shape
        if length != h * w:
            raise ValueError(f"token length {length} != {h}*{w}")
        x = x.view(b, h, w, 2, 4, self.dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h * 2, w * 4, self.dim)
        return self.norm(x)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size: tuple[int, int] = (2, 4),
        in_chans: int = 3,
        embed_dim: int = 48,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if (h, w) != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: list[float] | float,
        downsample: bool = False,
        upsample: bool = False,
    ) -> None:
        super().__init__()
        if downsample and upsample:
            raise ValueError("a Swin stage cannot both merge and expand")
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ]
        )
        self.downsample = PatchMerging(input_resolution, dim) if downsample else None
        self.upsample = PatchExpand(input_resolution, dim) if upsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


def stochastic_depth_rates(drop_path_rate: float, depths: tuple[int, ...]) -> list[list[float]]:
    total = sum(depths)
    if total == 0:
        return [[] for _ in depths]
    dpr = torch.linspace(0, drop_path_rate, total).tolist()
    out: list[list[float]] = []
    cursor = 0
    for depth in depths:
        out.append(dpr[cursor : cursor + depth])
        cursor += depth
    return out


def init_swin_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def tokens_to_map(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(B, H*W, C) -> (B, C, H, W)."""
    b, length, c = tokens.shape
    if length != height * width:
        raise ValueError(f"token length {length} != {height}*{width}")
    return tokens.transpose(1, 2).contiguous().view(b, c, height, width)


def map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H*W, C)."""
    return feature_map.flatten(2).transpose(1, 2).contiguous()


def spatial_hw(num_tokens: int) -> tuple[int, int]:
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"non-square token grid with {num_tokens} tokens")
    return side, side
