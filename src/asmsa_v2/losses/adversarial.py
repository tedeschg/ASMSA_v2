"""Standard non-saturating BCE-with-logits GAN losses.

Pair with ``torch.nn.utils.spectral_norm`` on the discriminator for a stable
drop-in replacement for v1's "random-scaling + raw logits" discriminator.
"""
import torch
import torch.nn.functional as F


def bce_disc_loss(
    logits_real: torch.Tensor,
    logits_fake: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    real_target = torch.full_like(logits_real, 1.0 - label_smoothing)
    fake_target = torch.zeros_like(logits_fake)
    loss_real = F.binary_cross_entropy_with_logits(logits_real, real_target)
    loss_fake = F.binary_cross_entropy_with_logits(logits_fake, fake_target)
    return loss_real + loss_fake


def bce_gen_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss: encoder wants its latents to look 'real'."""
    target = torch.ones_like(logits_fake)
    return F.binary_cross_entropy_with_logits(logits_fake, target)
