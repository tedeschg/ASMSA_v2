"""VAMP-2 scoring of a latent trajectory.

A reconstruction loss tells you how much *variance* the latent captured. For
a collective variable driving enhanced sampling you instead want the latent
to capture the *slow* dynamics — the degrees of freedom that take a long time
to decorrelate, because those are the ones metadynamics must push along.

The variational approach for Markov processes (VAMP; Wu & Noe, J. Nonlinear
Sci. 30, 2020) gives a scalar score for exactly that, and VAMPnets (Mardt,
Pasquali, Wu & Noe, Nat. Commun. 9, 5 (2018)) made it the standard objective
for learning MD coordinates. The VAMP-2 score is variational: a higher score
provably means a better approximation of the slow subspace, so it can be
compared across models and used directly for model selection.

Convention used here: the input is mean-free, the trivial stationary
eigenvalue is therefore excluded from the singular values and added back as
the constant 1, so the score lies in ``[1, 1 + latent_dim]``.
"""

from __future__ import annotations

import numpy as np
import torch


def _as_np(x) -> np.ndarray:
    """Convert a tensor or array-like to a float64 NumPy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _sym_inv_sqrt(C: np.ndarray, epsilon: float) -> np.ndarray:
    """``C^{-1/2}`` for a symmetric PSD matrix, dropping near-null directions."""
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    keep = w > epsilon * max(w.max(), 1e-30)
    if not keep.any():
        raise np.linalg.LinAlgError(
            "covariance matrix is numerically zero; the latent has no variance"
        )
    w_inv_sqrt = np.zeros_like(w)
    w_inv_sqrt[keep] = w[keep] ** -0.5
    return (V * w_inv_sqrt) @ V.T


def koopman_singular_values(
    z,
    lag: int = 1,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Singular values of the mean-free Koopman matrix at ``lag``.

    ``z`` must be a single latent trajectory ``[n_frames, d]`` **in time
    order**; do not pass shuffled frames.
    """
    Z = _as_np(z)
    if Z.ndim != 2:
        raise ValueError(f"expected 2-D [n_frames, d], got {Z.shape}")
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    n = Z.shape[0]
    if n - lag < Z.shape[1] + 1:
        raise ValueError(
            f"too few frames ({n}) for lag={lag} and latent dim {Z.shape[1]}"
        )

    X = Z[:-lag]
    Y = Z[lag:]
    # Mean-free using the pooled mean so C00 and C11 estimate the same
    # stationary distribution; this is what removes the trivial eigenvalue.
    mu = 0.5 * (X.mean(axis=0) + Y.mean(axis=0))
    X = X - mu
    Y = Y - mu
    m = X.shape[0]

    C00 = X.T @ X / m
    C11 = Y.T @ Y / m
    C01 = X.T @ Y / m

    K = _sym_inv_sqrt(C00, epsilon) @ C01 @ _sym_inv_sqrt(C11, epsilon)
    return np.linalg.svd(K, compute_uv=False)


def vamp2_score(z, lag: int = 1, epsilon: float = 1e-10) -> float:
    """VAMP-2 score of a latent trajectory. Higher is better.

    Range is ``[1, 1 + d]`` where ``d`` is the latent dimension: 1 means the
    latent carries no resolvable slow process at this lag, ``1 + d`` means
    every latent coordinate tracks a perfectly deterministic slow process.
    """
    s = koopman_singular_values(z, lag=lag, epsilon=epsilon)
    # Numerical noise can push a singular value marginally above 1.
    s = np.clip(s, 0.0, 1.0)
    return float(1.0 + np.sum(s**2))


def implied_timescales(
    z,
    lag: int = 1,
    dt: float = 1.0,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Implied timescales ``-lag*dt / ln(sigma_i)`` from the Koopman spectrum.

    ``dt`` is the physical time between consecutive frames, so the result is
    in the same units. Plotting these against ``lag`` is the standard
    convergence check: the timescales should plateau once the lag exceeds the
    fast relaxation, and the plateau value is the one to trust.
    """
    s = koopman_singular_values(z, lag=lag, epsilon=epsilon)
    s = np.clip(s, 1e-12, 1.0 - 1e-12)
    return -lag * dt / np.log(s)


def vamp2_scan(z, lags, epsilon: float = 1e-10) -> dict[int, float]:
    """VAMP-2 score at several lags, skipping lags the trajectory is too short for."""
    out: dict[int, float] = {}
    for lag in lags:
        try:
            out[int(lag)] = vamp2_score(z, lag=int(lag), epsilon=epsilon)
        except (ValueError, np.linalg.LinAlgError):
            continue
    return out
