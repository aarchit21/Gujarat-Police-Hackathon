from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.config import settings
from app.services.anpr import ENHANCEMENT_METHOD, enhance_for_vision, enhancement_status
from app.services.lpdgan.config import LPDGANConfig
from app.services.lpdgan.decoder import MultiScaleDecoder
from app.services.lpdgan.discriminators import (
    GlobalDiscriminator,
    equal_width_partitions,
    extract_partitions,
)
from app.services.lpdgan.encoder import MultiScaleFeatureEncoder
from app.services.lpdgan.fusion import LatentFusionModule, SpatialFeatureTransform
from app.services.lpdgan.generator import LPDGANGenerator, build_image_pyramid
from app.services.lpdgan.inference import deblur_plate_bgr, lpdgan_status, reset_runtime
from app.services.lpdgan.losses import LPDGAN, PixelPerceptualLoss, wgan_gp_penalty
from app.services.lpdgan.swin import tokens_to_map
from app.services.lpdgan.text import CRNNBaseline, TextReconstructionModule


def _rand_plate(batch: int = 2) -> torch.Tensor:
    return torch.randn(batch, 3, 112, 224)


def test_multiscale_encoder_emits_aligned_7x7_latents():
    enc = MultiScaleFeatureEncoder(LPDGANConfig()).eval()
    x1, x2, x3 = build_image_pyramid(_rand_plate(1))
    with torch.no_grad():
        feats = enc(x1, x2, x3)
    assert feats.token_hw == (7, 7)
    assert feats.full.shape == (1, 49, 384)
    assert feats.half.shape == (1, 49, 192)
    assert feats.quarter.shape == (1, 49, 96)


def test_sft_fusion_matches_affine_and_identity_concat():
    fuse = SpatialFeatureTransform(higher_dim=4, lower_dim=2)
    with torch.no_grad():
        fuse.alpha_head.net[-1].weight.fill_(0)
        fuse.alpha_head.net[-1].bias.fill_(2.0)
        fuse.beta_head.net[-1].weight.fill_(0)
        fuse.beta_head.net[-1].bias.fill_(0.5)
    higher = torch.ones(1, 4, 4)
    lower = torch.zeros(1, 4, 2)
    out = fuse(higher, lower, (2, 2))
    mapped = tokens_to_map(out, 2, 2)
    # α=2, β=0.5, higher=1 → modulated=2.5, concat identity 1.
    assert mapped.shape == (1, 8, 2, 2)
    assert torch.allclose(mapped[:, :4], torch.full((1, 4, 2, 2), 2.5))
    assert torch.allclose(mapped[:, 4:], torch.ones(1, 4, 2, 2))


def test_latent_fusion_mixes_three_scales():
    enc = MultiScaleFeatureEncoder().eval()
    fusion = LatentFusionModule()
    x1, x2, x3 = build_image_pyramid(_rand_plate(1))
    with torch.no_grad():
        feats = enc(x1, x2, x3)
        fused = fusion(feats.full, feats.half, feats.quarter, feats.token_hw)
    assert fused.fused_12.shape[-1] == 768
    assert fused.fused_23.shape[-1] == 384
    assert fused.latent.shape == (1, 49, 384)


def test_decoder_emits_three_paper_scales():
    dec = MultiScaleDecoder().eval()
    latent = torch.randn(1, 49, 384)
    with torch.no_grad():
        out = dec(latent)
    assert out.y_full.shape == (1, 3, 112, 224)
    assert out.y_half.shape == (1, 3, 56, 112)
    assert out.y_quarter.shape == (1, 3, 28, 56)
    assert out.decoder_mid.shape == (1, 196, 192)
    assert out.y_full.min() >= -1.0 and out.y_full.max() <= 1.0


def test_generator_forward_and_text_head():
    gen = LPDGANGenerator().eval()
    blur = _rand_plate(1)
    with torch.no_grad():
        out = gen(blur)
    assert out.y_full.shape[-2:] == (112, 224)
    assert out.text_logits.shape == (1, 10, 37)
    assert out.fused_23.shape == (1, 49, 384)


def test_partition_strips_and_global_critic():
    image = torch.randn(2, 3, 112, 224)
    strips = equal_width_partitions(image, n=7)
    assert strips.shape[:2] == (2, 7)
    batch = extract_partitions(image, n=7)
    assert batch.crops.shape[0] == 14
    disc = GlobalDiscriminator()
    score = disc(image)
    assert score.ndim == 4 and score.shape[0] == 2


def test_wgan_gp_and_combined_loss_backward():
    system = LPDGAN(perceptual=PixelPerceptualLoss(), vgg_pretrained=False)
    blur = torch.randn(1, 3, 112, 224, requires_grad=True)
    sharp = torch.randn(1, 3, 112, 224)
    sharp_pyr = system.pyramid(sharp)
    out = system(blur)
    g_loss = system.loss.generator_losses(out, *sharp_pyr)
    g_loss.generator_total.backward(retain_graph=True)
    d_loss = system.loss.discriminator_losses(out, *sharp_pyr)
    d_loss.discriminator_total.backward()
    assert g_loss.l1.item() >= 0
    assert g_loss.text.item() >= 0
    assert torch.isfinite(d_loss.gp_global)


def test_gradient_penalty_is_near_zero_for_unit_norm_critic():
    class UnitNormCritic(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            flat = x.reshape(x.shape[0], -1)
            return flat.sum(dim=1, keepdim=True) / (flat.shape[1] ** 0.5)

    real = torch.randn(4, 3, 8, 8)
    fake = torch.randn(4, 3, 8, 8)
    gp = wgan_gp_penalty(UnitNormCritic(), real, fake, gp_lambda=1.0)
    assert gp.item() < 1e-5


def test_crnn_and_text_module_shapes():
    crnn = CRNNBaseline()
    text = TextReconstructionModule(
        fused_dim=384,
        decoder_dim=192,
        fused_hw=(7, 7),
        decoder_hw=(14, 14),
        charset_size=37,
        max_len=10,
    )
    image = torch.randn(2, 3, 112, 224)
    fused = torch.randn(2, 49, 384)
    mid = torch.randn(2, 196, 192)
    pred = text(fused, mid)
    target = crnn(image)
    assert pred.shape == target.shape == (2, 10, 37)


def test_inference_is_identity_without_weights(monkeypatch):
    reset_runtime()
    monkeypatch.setattr(settings, "lpdgan_enabled", True)
    crop = np.full((40, 80, 3), 90, dtype=np.uint8)
    out, meta = deblur_plate_bgr(crop)
    assert np.array_equal(out, crop)
    assert meta["applied"] is False
    assert meta["reason"] == "weights_missing"


def test_lpdgan_enabled_without_weights_falls_back_to_clahe(monkeypatch):
    monkeypatch.setattr(settings, "lpdgan_enabled", True)
    monkeypatch.setattr(settings, "vision_enhancement_enabled", True)
    crop = np.full((32, 100, 3), 112, dtype=np.uint8)
    out, meta = enhance_for_vision(crop, profile="plate", min_width=100)
    assert meta["method"] == ENHANCEMENT_METHOD
    assert meta.get("generative") is False
    assert meta["lpdgan"]["applied"] is False
    assert out.shape[1] == 100 or out.shape[1] >= 100


def test_default_enhancement_stays_non_generative(monkeypatch):
    monkeypatch.setattr(settings, "lpdgan_enabled", False)
    monkeypatch.setattr(settings, "vision_enhancement_enabled", True)
    crop = np.full((32, 100, 3), 112, dtype=np.uint8)
    out, meta = enhance_for_vision(crop, profile="plate", min_width=100)
    assert meta["method"] == ENHANCEMENT_METHOD
    assert meta.get("generative") is False
    status = enhancement_status()
    assert status["generative"] is False
    assert status["lpdgan"]["enabled"] is False
    assert lpdgan_status()["generative"] is True
