"""LPDGAN training objective, paper eq. (5)–(10).

L = L_rec + λ_g L_{D_g} + λ_p L_{D_p} + λ_t L_text
L_rec = λ_l1 ||y − ỹ||_1 + λ_per ||ψ_vgg(y) − ψ_vgg(ỹ)||_2

VGG-19 compares ReLU feature maps at sequential indices 8, 15, and 22.
Adversarial terms are WGAN-GP. Text term is L1 against a frozen CRNN.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.services.lpdgan.config import LPDGANConfig
from app.services.lpdgan.discriminators import (
    MultiScaleDiscriminators,
    equal_width_partitions,
)
from app.services.lpdgan.generator import GeneratorOutput, LPDGANGenerator, build_image_pyramid
from app.services.lpdgan.text import CRNNBaseline

# torchvision VGG-19 feature indices of ReLU layers 2_2, 3_3, 4_3.
VGG_RELU_INDICES = (8, 15, 22)


def _vgg19_features(pretrained: bool) -> nn.Sequential:
    from torchvision.models import VGG19_Weights, vgg19

    weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
    model = vgg19(weights=weights)
    return model.features


class PixelPerceptualLoss(nn.Module):
    """Fallback perceptual term when VGG-19 weights are not loaded."""

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(fake, real.detach())


class VGG19PerceptualLoss(nn.Module):
    """Paper L_per: L2 on VGG-19 ReLU maps at indices 8, 15, 22."""

    def __init__(self, pretrained: bool = True, resize: bool = True) -> None:
        super().__init__()
        self.resize = resize
        features = _vgg19_features(pretrained)
        slices: list[nn.Sequential] = []
        prev = 0
        for idx in VGG_RELU_INDICES:
            slices.append(nn.Sequential(*list(features.children())[prev : idx + 1]))
            prev = idx + 1
        self.slices = nn.ModuleList(slices)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _prepare(self, image: torch.Tensor) -> torch.Tensor:
        # Generator emits tanh [-1, 1]; VGG expects ImageNet-normalised [0, 1].
        x = (image.clamp(-1, 1) + 1.0) * 0.5
        x = (x - self.mean) / self.std
        if self.resize and (x.shape[-2] < 32 or x.shape[-1] < 32):
            x = F.interpolate(x, size=(112, 224), mode="bilinear", align_corners=False)
        return x

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        fake_x = self._prepare(fake)
        real_x = self._prepare(real)
        loss = fake_x.new_zeros(())
        for block in self.slices:
            fake_x = block(fake_x)
            real_x = block(real_x)
            loss = loss + F.mse_loss(fake_x, real_x.detach())
        return loss / len(self.slices)


def wgan_gp_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    gp_lambda: float,
) -> torch.Tensor:
    """Paper eq. (5)/(6) gradient penalty on ŷ = εỹ + (1−ε)y."""
    batch = real.shape[0]
    epsilon = torch.rand(batch, 1, 1, 1, device=real.device, dtype=real.dtype)
    interpolated = (epsilon * fake.detach() + (1.0 - epsilon) * real.detach()).requires_grad_(True)
    scores = critic(interpolated)
    ones = torch.ones_like(scores)
    grad = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.reshape(batch, -1)
    penalty = ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp_lambda * penalty


def _mean_critic(score: torch.Tensor) -> torch.Tensor:
    return score.reshape(score.shape[0], -1).mean()


@dataclass
class LPDGANLossBreakdown:
    reconstruction: torch.Tensor
    l1: torch.Tensor
    perceptual: torch.Tensor
    global_adv: torch.Tensor
    partition_adv: torch.Tensor
    text: torch.Tensor
    d_global: torch.Tensor
    d_partition: torch.Tensor
    gp_global: torch.Tensor
    gp_partition: torch.Tensor
    generator_total: torch.Tensor
    discriminator_total: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            key: float(val.detach().item())
            for key, val in self.__dict__.items()
            if torch.is_tensor(val)
        }


class LPDGANCriterion(nn.Module):
    def __init__(
        self,
        cfg: LPDGANConfig | None = None,
        perceptual: nn.Module | None = None,
        crnn: CRNNBaseline | None = None,
        vgg_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg or LPDGANConfig()
        self.perceptual = perceptual if perceptual is not None else VGG19PerceptualLoss(pretrained=vgg_pretrained)
        self.crnn = crnn if crnn is not None else CRNNBaseline(self.cfg.charset, self.cfg.plate_max_len)
        self.crnn.freeze()
        self.l1 = nn.L1Loss()

    def reconstruction(
        self,
        fake: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        real: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l1 = fake[0].new_zeros(())
        per = fake[0].new_zeros(())
        for y_hat, y in zip(fake, real):
            l1 = l1 + self.l1(y_hat, y)
            per = per + self.perceptual(y_hat, y)
        n = float(len(fake))
        l1 = (l1 / n) * self.cfg.lambda_l1
        per = (per / n) * self.cfg.lambda_per
        return l1 + per, l1, per

    def text_l1(self, pred_logits: torch.Tensor, sharp_full: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target = self.crnn(sharp_full)
        return F.l1_loss(pred_logits, target) * self.cfg.lambda_t


def _scale_pair(
    fake: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    real: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return list(zip(fake, real))


class LPDGANLoss(nn.Module):
    """Combined generator / discriminator step for one batch."""

    def __init__(
        self,
        generator: LPDGANGenerator,
        discriminators: MultiScaleDiscriminators,
        cfg: LPDGANConfig | None = None,
        perceptual: nn.Module | None = None,
        crnn: CRNNBaseline | None = None,
        vgg_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg or LPDGANConfig()
        self.generator = generator
        self.discs = discriminators
        self.crit = LPDGANCriterion(self.cfg, perceptual=perceptual, crnn=crnn, vgg_pretrained=vgg_pretrained)

    def _global_disc_loss(
        self,
        fake: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        real: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        critics = (self.discs.global_full, self.discs.global_half, self.discs.global_quarter)
        adv = fake[0].new_zeros(())
        gp = fake[0].new_zeros(())
        for critic, (y_hat, y) in zip(critics, _scale_pair(fake, real)):
            fake_score = _mean_critic(critic(y_hat.detach()))
            real_score = _mean_critic(critic(y))
            adv = adv + (fake_score - real_score)
            gp = gp + wgan_gp_penalty(critic, y, y_hat, self.cfg.lambda_gp)
        n = float(len(critics))
        return adv / n, gp / n

    def _partition_disc_loss(
        self,
        fake_full: torch.Tensor,
        real_full: torch.Tensor,
        n: int,
        boxes: list[list[tuple[int, int, int, int]]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real_strips = equal_width_partitions(real_full, n) if boxes is None else None
        fake_strips = equal_width_partitions(fake_full.detach(), n) if boxes is None else None
        if real_strips is None:
            from app.services.lpdgan.discriminators import extract_partitions

            real_batch = extract_partitions(real_full, n=n, boxes=boxes)
            fake_batch = extract_partitions(fake_full.detach(), n=n, boxes=boxes)
            real_crops, fake_crops = real_batch.crops, fake_batch.crops
        else:
            b, k, c, h, w = real_strips.shape
            real_crops = real_strips.reshape(b * k, c, h, w)
            fake_crops = fake_strips.reshape(b * k, c, h, w)
        fake_score = _mean_critic(self.discs.partition(fake_crops))
        real_score = _mean_critic(self.discs.partition(real_crops))
        adv = fake_score - real_score
        gp = wgan_gp_penalty(self.discs.partition, real_crops, fake_crops, self.cfg.lambda_gp)
        return adv, gp

    def _global_gen_adv(
        self,
        fake: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        critics = (self.discs.global_full, self.discs.global_half, self.discs.global_quarter)
        adv = fake[0].new_zeros(())
        for critic, y_hat in zip(critics, fake):
            adv = adv - _mean_critic(critic(y_hat))
        return adv / float(len(critics))

    def _partition_gen_adv(self, fake_full: torch.Tensor, n: int) -> torch.Tensor:
        strips = equal_width_partitions(fake_full, n)
        b, k, c, h, w = strips.shape
        score = self.discs.partition(strips.reshape(b * k, c, h, w))
        return -_mean_critic(score)

    def generator_losses(
        self,
        output: GeneratorOutput,
        real_full: torch.Tensor,
        real_half: torch.Tensor,
        real_quarter: torch.Tensor,
        partition_n: int | None = None,
    ) -> LPDGANLossBreakdown:
        n = partition_n or self.cfg.early_partition_count
        fake = (output.y_full, output.y_half, output.y_quarter)
        real = (real_full, real_half, real_quarter)
        rec, l1, per = self.crit.reconstruction(fake, real)
        g_adv = self._global_gen_adv(fake) * self.cfg.lambda_g
        p_adv = self._partition_gen_adv(output.y_full, n) * self.cfg.lambda_p
        text = self.crit.text_l1(output.text_logits, real_full)
        total = rec + g_adv + p_adv + text
        zero = total.new_zeros(())
        return LPDGANLossBreakdown(
            reconstruction=rec,
            l1=l1,
            perceptual=per,
            global_adv=g_adv,
            partition_adv=p_adv,
            text=text,
            d_global=zero,
            d_partition=zero,
            gp_global=zero,
            gp_partition=zero,
            generator_total=total,
            discriminator_total=zero,
        )

    def discriminator_losses(
        self,
        output: GeneratorOutput,
        real_full: torch.Tensor,
        real_half: torch.Tensor,
        real_quarter: torch.Tensor,
        partition_n: int | None = None,
        letter_boxes: list[list[tuple[int, int, int, int]]] | None = None,
    ) -> LPDGANLossBreakdown:
        n = partition_n or self.cfg.early_partition_count
        fake = (output.y_full, output.y_half, output.y_quarter)
        real = (real_full, real_half, real_quarter)
        d_g, gp_g = self._global_disc_loss(fake, real)
        d_p, gp_p = self._partition_disc_loss(output.y_full, real_full, n, letter_boxes)
        d_g = d_g * self.cfg.lambda_g
        d_p = d_p * self.cfg.lambda_p
        total = d_g + gp_g + d_p + gp_p
        zero = total.new_zeros(())
        return LPDGANLossBreakdown(
            reconstruction=zero,
            l1=zero,
            perceptual=zero,
            global_adv=zero,
            partition_adv=zero,
            text=zero,
            d_global=d_g,
            d_partition=d_p,
            gp_global=gp_g,
            gp_partition=gp_p,
            generator_total=zero,
            discriminator_total=total,
        )


class LPDGAN(nn.Module):
    """Full trainable system: G, D_g, D_p, T, ψ_crnn."""

    def __init__(
        self,
        cfg: LPDGANConfig | None = None,
        vgg_pretrained: bool = True,
        crnn: CRNNBaseline | None = None,
        perceptual: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or LPDGANConfig()
        self.generator = LPDGANGenerator(self.cfg)
        self.discriminators = MultiScaleDiscriminators(self.cfg)
        self.loss = LPDGANLoss(
            self.generator,
            self.discriminators,
            self.cfg,
            perceptual=perceptual,
            crnn=crnn,
            vgg_pretrained=vgg_pretrained,
        )

    def forward(
        self,
        blur_full: torch.Tensor,
        blur_half: torch.Tensor | None = None,
        blur_quarter: torch.Tensor | None = None,
    ) -> GeneratorOutput:
        return self.generator(blur_full, blur_half, blur_quarter)

    def pyramid(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return build_image_pyramid(image, self.cfg)
