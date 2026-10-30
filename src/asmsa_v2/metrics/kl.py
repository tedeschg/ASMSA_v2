"""k-nearest-neighbour Kullback-Leibler divergence between two samples.

This is the metric v1 used to *score* hyperparameter trials: how closely the
aggregate posterior q(z) matches the prior p(z). It is the quantity the
adversarial game is nominally minimizing, so it is a far better model-selection
criterion than the reconstruction loss (which a model can improve while
abandoning the prior shape entirely).

Estimator from Perez-Cruz, "Kullback-Leibler divergence estimation of
continuous distributions", IEEE ISIT 2008 — ported verbatim in behaviour from
v1's ``asmsa.aae_hyper_model._KLdivergence`` so scores remain comparable
across the two versions.
"""

from __future__ import annotations

import numpy as np
import torch


def _as_2d_np(x) -> np.ndarray:
    """Convert a tensor or array-like to a float64 NumPy array of shape [n, d]."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.atleast_2d(np.asarray(x, dtype=np.float64))


def kl_divergence_knn(
    x,
    y,
    eps: float = 0.01,
    clip_negative: bool = True,
) -> float:
    """Estimate ``D(P || Q)`` from samples ``x ~ P`` and ``y ~ Q``.

    Parameters
    ----------
    x : ``[n, d]``
        Samples from the reference distribution P. In ASMSA's usage this is
        the prior sample.
    y : ``[m, d]``
        Samples from Q — the encoder's latent outputs.
    eps : float
        Approximation tolerance passed to the KD-tree query, as in v1.
    clip_negative : bool
        The estimator is only asymptotically non-negative and returns small
        negative values when the two samples are nearly identical. v1 clipped
        these to zero before aggregating; keep that for comparability.

    Returns
    -------
    float
        Estimated divergence in nats.
    """
    # Imported lazily so that importing this module does not require SciPy.
    from scipy.spatial import cKDTree as KDTree

    x = _as_2d_np(x)
    y = _as_2d_np(y)
    n, d = x.shape
    m, dy = y.shape
    if d != dy:
        raise ValueError(f"dimension mismatch: x has {d}, y has {dy}")
    if n < 2:
        raise ValueError("need at least 2 samples in x")
    if m < 1:
        raise ValueError("need at least 1 sample in y")

    xtree = KDTree(x)
    ytree = KDTree(y)

    # k=2 for x because the nearest neighbour of a point in its own sample is
    # itself; take the second.
    r = xtree.query(x, k=2, eps=eps, p=2)[0][:, 1]
    s = ytree.query(x, k=1, eps=eps, p=2)[0]

    # Guard against coincident points, which would otherwise give -inf/+inf.
    tiny = np.finfo(np.float64).tiny
    r = np.maximum(r, tiny)
    s = np.maximum(s, tiny)

    kl = float(-np.log(r / s).sum() * d / n + np.log(m / (n - 1.0)))
    if clip_negative and kl < 0.0:
        return 0.0
    return kl


def prior_match_score(
    prior,
    z,
    n_samples: int | None = None,
    seed: int | None = 0,
) -> float:
    """KL divergence between a :class:`~asmsa_v2.priors.base.Prior` and latents.

    Draws a prior sample of the same size as ``z`` (unless ``n_samples`` says
    otherwise) and returns ``kl_divergence_knn(prior_sample, z)``. Lower is
    better; this is the quantity minimized during hyperparameter tuning.
    """
    z_np = _as_2d_np(z)
    n = int(n_samples or z_np.shape[0])
    gen_state = None
    if seed is not None:
        gen_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
    try:
        p = prior.sample(n)
    finally:
        if gen_state is not None:
            torch.random.set_rng_state(gen_state)
    return kl_divergence_knn(p, z_np)
