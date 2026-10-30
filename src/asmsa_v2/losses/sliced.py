"""Sliced-Wasserstein latent penalty.

Kolouri, Pope, Martin & Rohde, "Sliced-Wasserstein Autoencoder", 2018.
The idea: the Wasserstein distance between two distributions on R^d is hard,
but on R^1 it is just the L2 distance between the sorted samples. So project
both samples onto many random directions, average the 1-D Wasserstein
distances, and you get a cheap, differentiable, *non-adversarial* divergence.

For the 2-D latent spaces used as collective variables this is especially
attractive: with d = 2 a few dozen random directions already cover the sphere
densely, so the estimator is nearly exact and costs one sort per direction.
Unlike MMD there is no bandwidth to choose at all.
"""
from __future__ import annotations

import torch


def sliced_wasserstein_loss(
    z_posterior: torch.Tensor,
    z_prior: torch.Tensor,
    n_projections: int = 64,
    p: int = 2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sliced-Wasserstein-``p`` distance between encoder output and prior samples.

    Parameters
    ----------
    z_posterior, z_prior : ``[n, d]`` and ``[m, d]``
        If ``n != m`` the larger sample is linearly resampled onto the smaller
        one's quantile grid, which keeps the sorted comparison well-defined.
    n_projections : int
        Number of random unit directions. For ``d = 2``, 64 is plenty.
    p : int
        Order of the Wasserstein distance (1 or 2 in practice).

    Returns
    -------
    torch.Tensor
        Non-negative scalar.
    """
    if z_posterior.ndim != 2 or z_prior.ndim != 2:
        raise ValueError("sliced_wasserstein_loss expects 2-D [n, d] tensors")
    d = z_posterior.shape[1]
    if z_prior.shape[1] != d:
        raise ValueError(
            f"latent dim mismatch: posterior {d} vs prior {z_prior.shape[1]}"
        )
    n, m = z_posterior.shape[0], z_prior.shape[0]
    if n < 2 or m < 2:
        return z_posterior.new_zeros(())

    theta = torch.randn(
        d, int(n_projections), device=z_posterior.device,
        dtype=z_posterior.dtype, generator=generator,
    )
    theta = theta / theta.norm(dim=0, keepdim=True).clamp_min(1e-12)

    proj_q = z_posterior @ theta           # [n, n_proj]
    proj_p = z_prior @ theta               # [m, n_proj]

    sq, _ = torch.sort(proj_q, dim=0)
    sp, _ = torch.sort(proj_p, dim=0)

    if n != m:
        k = min(n, m)
        # Common quantile grid; interpolate the longer sorted sample onto it.
        grid = torch.linspace(0, 1, k, device=sq.device, dtype=sq.dtype)
        sq = _interp_sorted(sq, grid) if n != k else sq
        sp = _interp_sorted(sp, grid) if m != k else sp

    diff = (sq - sp).abs()
    return (diff ** p).mean()


def _interp_sorted(sorted_vals: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Linearly resample a sorted ``[n, k]`` sample onto ``len(grid)`` quantiles."""
    n = sorted_vals.shape[0]
    pos = grid * (n - 1)
    lo = pos.floor().long().clamp(0, n - 1)
    hi = pos.ceil().long().clamp(0, n - 1)
    w = (pos - lo.to(pos.dtype)).unsqueeze(1)
    return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w
