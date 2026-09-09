"""Global WGAN-GP discriminator and partition (per-letter) discriminator.

Paper §4.2: D_g scores the whole restored plate; D_p scores n randomly chosen
letter crops. Early training uses n=7 equal-width partitions; later training
can pass detected letter boxes and n=3. Both discriminators are applied at
the three generator scales.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.services.lpdgan.config import EARLY_PARTITION_COUNT, LPDGANConfig


class GlobalDiscriminator(nn.Module):
    """PatchGAN critic used with WGAN-GP (unconditional, as in eq. (5))."""

    def __init__(self, in_chans: int = 3, ndf: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_chans, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf = ndf
        for i in range(1, n_layers):
            nf_prev, nf = nf, min(ndf * 2**i, ndf * 8)
            layers += [
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(nf, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_prev, nf = nf, min(nf * 2, ndf * 8)
        layers += [
            nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(nf, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class PartitionDiscriminator(nn.Module):
    """Letter-crop critic. Adaptive pool keeps it size-agnostic."""

    def __init__(self, in_chans: int = 3, ndf: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ndf * 2, 1, kernel_size=1),
        )

    def forward(self, partitions: torch.Tensor) -> torch.Tensor:
        return self.net(partitions).view(partitions.shape[0], -1)


@dataclass(frozen=True)
class PartitionBatch:
    crops: torch.Tensor  # (B * n, 3, H, w)
    count: int


def equal_width_partitions(images: torch.Tensor, n: int, min_width: int = 8) -> torch.Tensor:
    """Split BCHW plates into n equal-width strips: (B, n, C, H, strip_w)."""
    if n < 1:
        raise ValueError("partition count must be positive")
    b, c, h, w = images.shape
    strip_w = max(min_width, w // n)
    edges = torch.linspace(0, w, n + 1, device=images.device).long()
    strips = []
    for i in range(n):
        x0, x1 = int(edges[i].item()), int(edges[i + 1].item())
        if x1 <= x0:
            x1 = min(w, x0 + strip_w)
        strip = images[:, :, :, x0:x1]
        if strip.shape[-1] != strip_w:
            strip = F.interpolate(strip, size=(h, strip_w), mode="bilinear", align_corners=False)
        strips.append(strip)
    return torch.stack(strips, dim=1)


def _crop_boxes(
    images: torch.Tensor,
    boxes: list[list[tuple[int, int, int, int]]],
    n: int,
    min_width: int,
) -> torch.Tensor:
    b, _, h, w = images.shape
    strip_w = max(min_width, w // max(n, 1))
    crops = []
    for bi in range(b):
        img_boxes = list(boxes[bi]) if bi < len(boxes) else []
        if not img_boxes:
            row = equal_width_partitions(images[bi : bi + 1], n, min_width)
            crops.append(row)
            continue
        chosen = img_boxes[:n]
        while len(chosen) < n:
            chosen.append(chosen[len(chosen) % len(img_boxes)])
        batch_crops = []
        for (x, y, bw, bh) in chosen:
            x0 = max(0, int(x))
            y0 = max(0, int(y))
            x1 = min(w, x0 + max(min_width, int(bw)))
            y1 = min(h, y0 + max(1, int(bh)))
            crop = images[bi : bi + 1, :, y0:y1, x0:x1]
            crop = F.interpolate(crop, size=(h, strip_w), mode="bilinear", align_corners=False)
            batch_crops.append(crop)
        crops.append(torch.stack(batch_crops, dim=1))
    return torch.cat(crops, dim=0)


def extract_partitions(
    images: torch.Tensor,
    n: int = EARLY_PARTITION_COUNT,
    boxes: list[list[tuple[int, int, int, int]]] | None = None,
    min_width: int = 8,
    indices: torch.Tensor | None = None,
) -> PartitionBatch:
    """Letter crops. Same `indices` on real/fake keeps WGAN-GP pairs aligned."""
    if boxes is None:
        strips = equal_width_partitions(images, n, min_width)
    else:
        strips = _crop_boxes(images, boxes, n, min_width)
    n_avail = strips.shape[1]
    take = min(n, n_avail)
    if indices is None:
        indices = torch.randperm(n_avail, device=images.device)[:take]
    selected = strips[:, indices]
    b, k, c, h, w = selected.shape
    return PartitionBatch(crops=selected.reshape(b * k, c, h, w), count=k)


class MultiScaleDiscriminators(nn.Module):
    """One global critic per scale plus a shared partition critic."""

    def __init__(self, cfg: LPDGANConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or LPDGANConfig()
        self.global_full = GlobalDiscriminator(cfg.out_chans, cfg.ndf, cfg.disc_layers)
        self.global_half = GlobalDiscriminator(cfg.out_chans, cfg.ndf, cfg.disc_layers)
        self.global_quarter = GlobalDiscriminator(cfg.out_chans, cfg.ndf, cfg.disc_layers)
        self.partition = PartitionDiscriminator(cfg.out_chans, cfg.ndf)
        self.early_n = cfg.early_partition_count
        self.late_n = cfg.late_partition_count

    def global_scores(
        self,
        full: torch.Tensor,
        half: torch.Tensor,
        quarter: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.global_full(full), self.global_half(half), self.global_quarter(quarter)

    def partition_scores(
        self,
        image: torch.Tensor,
        n: int | None = None,
        boxes: list[list[tuple[int, int, int, int]]] | None = None,
    ) -> torch.Tensor:
        n = self.early_n if n is None else n
        batch = extract_partitions(image, n=n, boxes=boxes)
        return self.partition(batch.crops)
