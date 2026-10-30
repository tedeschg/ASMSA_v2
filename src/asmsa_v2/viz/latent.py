"""2-D latent-space visualization helpers (scatter, 2-D hist, KDE)."""
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde


def _to_np(z) -> np.ndarray:
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    return np.asarray(z)


def scatter_latents(
    latents: Mapping[str, "torch.Tensor | np.ndarray"],
    ax=None,
    alpha: float = 0.4,
    s: float = 4.0,
):
    """Overlay per-molecule scatter plots on a single axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    for mol_id, z in latents.items():
        z = _to_np(z)
        ax.scatter(z[:, 0], z[:, 1], s=s, alpha=alpha, label=mol_id)
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.legend(fontsize=8, markerscale=2)
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def hist2d_latent(z, ax=None, bins: int = 60):
    z = _to_np(z)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.hist2d(z[:, 0], z[:, 1], bins=bins, cmap="viridis")
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def kde_latent(z, ax=None, grid: int = 100):
    z = _to_np(z)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    kde = gaussian_kde(z.T)
    xmin, ymin = z.min(axis=0) - 0.3
    xmax, ymax = z.max(axis=0) + 0.3
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, grid), np.linspace(ymin, ymax, grid))
    f = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contourf(xx, yy, f, levels=20, cmap="viridis")
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_aspect("equal", adjustable="datalim")
    return ax
