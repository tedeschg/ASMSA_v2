"""Density-matching loss: pull encoder outputs toward a target per-sample density.

Ported semantically from v1's ``with_density`` branch — the user provides
per-sample target densities ``p_target(x_i)`` (e.g. from Boltzmann weights) and
we push ``prior.log_prob(enc(x_i))`` toward the matching log-density.
"""
import torch

from ..priors.base import Prior


def density_matching_loss(
    z: torch.Tensor,
    target_density: torch.Tensor,
    prior: Prior,
) -> torch.Tensor:
    log_target = torch.log(target_density.clamp_min(1e-12))
    log_q = prior.log_prob(z)
    return torch.mean((log_q - log_target) ** 2)
