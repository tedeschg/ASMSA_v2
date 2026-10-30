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


def r1_penalty(logits_real: torch.Tensor, z_real: torch.Tensor) -> torch.Tensor:
    """R1 gradient penalty on the real (prior) samples.

    Mescheder, Geiger & Nowozin, "Which Training Methods for GANs do actually
    Converge?", ICML 2018, show that penalizing the discriminator's gradient
    norm on the real distribution alone is enough to make the GAN dynamics
    locally convergent, unlike the unregularized game which can orbit the
    equilibrium indefinitely. It is cheaper than WGAN-GP (no interpolates, one
    gradient instead of two) and composes with spectral normalization.

    ``z_real`` must have ``requires_grad=True`` and have been the input that
    produced ``logits_real``.
    """
    if not z_real.requires_grad:
        raise ValueError(
            "r1_penalty needs z_real.requires_grad=True so the gradient of the "
            "critic output with respect to its input can be taken"
        )
    grad = torch.autograd.grad(
        outputs=logits_real.sum(), inputs=z_real,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    return 0.5 * grad.pow(2).sum(dim=1).mean()
