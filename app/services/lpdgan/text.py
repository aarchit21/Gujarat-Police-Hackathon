"""Text Reconstruction Module T and a CRNN baseline ψ_crnn.

T concatenates F_{2,3} with decoder-mid features F_D, then maps them to a
character-sequence vector. A frozen CRNN on the sharp plate produces the
target vector. Loss is L1 as in paper eq. (8).

Charset is Indian A-Z/0-9 (plus CTC blank). This is independent of any
patent-derived voting scheme.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.services.lpdgan.config import INDIAN_PLATE_CHARSET, PLATE_MAX_LEN
from app.services.lpdgan.swin import tokens_to_map


class TextReconstructionModule(nn.Module):
    """T(F_{2,3}, F_D) -> (B, max_len, charset+blank)."""

    def __init__(
        self,
        fused_dim: int,
        decoder_dim: int,
        fused_hw: tuple[int, int],
        decoder_hw: tuple[int, int],
        charset_size: int,
        max_len: int = PLATE_MAX_LEN,
    ) -> None:
        super().__init__()
        self.fused_hw = fused_hw
        self.decoder_hw = decoder_hw
        self.max_len = max_len
        self.charset_size = charset_size
        self.project_fused = nn.Conv2d(fused_dim, decoder_dim, kernel_size=1)
        merged = decoder_dim * 2
        self.conv = nn.Sequential(
            nn.Conv2d(merged, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, max_len)),
        )
        self.head = nn.Linear(128, charset_size)

    def forward(self, fused_23: torch.Tensor, decoder_mid: torch.Tensor) -> torch.Tensor:
        fused_map = tokens_to_map(fused_23, *self.fused_hw)
        fused_map = F.interpolate(
            self.project_fused(fused_map),
            size=self.decoder_hw,
            mode="bilinear",
            align_corners=False,
        )
        decoder_map = tokens_to_map(decoder_mid, *self.decoder_hw)
        merged = torch.cat([fused_map, decoder_map], dim=1)
        feat = self.conv(merged)  # B, 128, 1, T
        feat = feat.squeeze(2).transpose(1, 2)  # B, T, 128
        return self.head(feat)


class _CRNNConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CRNNBaseline(nn.Module):
    """Shi et al. CRNN used as ψ_crnn. Frozen during LPDGAN training unless thawed."""

    def __init__(
        self,
        charset: str = INDIAN_PLATE_CHARSET,
        max_len: int = PLATE_MAX_LEN,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.charset = charset
        self.max_len = max_len
        self.blank_index = len(charset)
        self.n_class = len(charset) + 1
        self.cnn = _CRNNConv()
        self.rnn = nn.LSTM(512, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, self.n_class)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(image)
        b, c, h, w = feat.shape
        if h != 1:
            feat = F.adaptive_avg_pool2d(feat, (1, w))
        seq = feat.squeeze(2).permute(0, 2, 1)  # B, W, C
        seq, _ = self.rnn(seq)
        logits = self.fc(seq)  # B, W, n_class
        if logits.shape[1] == self.max_len:
            return logits
        return F.interpolate(
            logits.transpose(1, 2),
            size=self.max_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def freeze(self) -> None:
        self.eval()
        for p in self.parameters():
            p.requires_grad = False


def encode_plate_text(
    texts: list[str],
    charset: str = INDIAN_PLATE_CHARSET,
    max_len: int = PLATE_MAX_LEN,
) -> torch.Tensor:
    """Integer targets (B, max_len); blank-padded. For logging / CTC, not L_text."""
    table = {ch: i for i, ch in enumerate(charset)}
    blank = len(charset)
    out = torch.full((len(texts), max_len), blank, dtype=torch.long)
    for i, raw in enumerate(texts):
        cleaned = "".join(ch for ch in raw.upper() if ch in table)[:max_len]
        for j, ch in enumerate(cleaned):
            out[i, j] = table[ch]
    return out
