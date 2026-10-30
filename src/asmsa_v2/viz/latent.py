"""Static matplotlib / scipy visualizations of the AAE latent space.

For interactive hover + 3D structure linkage, see :mod:`asmsa_v2.viz.interactive`.
"""
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
    title: str | None = None,
):
    """Overlay per-molecule (or per-cluster) scatter plots on a single axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    for mol_id, z in latents.items():
        z = _to_np(z)
        ax.scatter(z[:, 0], z[:, 1], s=s, alpha=alpha, label=str(mol_id))
    ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]")
    ax.legend(fontsize=8, markerscale=2)
    ax.set_aspect("equal", adjustable="datalim")
    if title:
        ax.set_title(title)
    return ax


def hist2d_latent(z, ax=None, bins: int = 60, cmap: str = "viridis"):
    z = _to_np(z)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.hist2d(z[:, 0], z[:, 1], bins=bins, cmap=cmap)
    ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]")
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def kde_latent(z, ax=None, grid: int = 100, cmap: str = "viridis"):
    z = _to_np(z)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    kde = gaussian_kde(z.T)
    xmin, ymin = z.min(axis=0) - 0.3
    xmax, ymax = z.max(axis=0) + 0.3
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, grid), np.linspace(ymin, ymax, grid))
    f = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contourf(xx, yy, f, levels=20, cmap=cmap)
    ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]")
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def free_energy_surface(
    z,
    ax=None,
    grid: int = 120,
    kT: float = 1.0,
    max_FE: float | None = 8.0,
    cmap: str | None = None,
    overlay_points: bool = False,
    point_alpha: float = 0.03,
    weights=None,
):
    """Negative-log-density free-energy surface on the latent plane.

    ``FE(z) = -kT · log P(z)``, shifted so ``min(FE) == 0``. Regions above
    ``max_FE`` are left blank rather than clipped to the cap, so an
    unsampled area reads as unsampled instead of as a broad plateau.

    ``kT=1.0`` (the default) gives the surface in units of kT, which is the
    natural unit for reading basin depths: a 3 kT barrier is crossed
    routinely, a 10 kT barrier is not.

    Parameters
    ----------
    weights : array, optional
        Per-frame weights, e.g. from
        :func:`asmsa_v2.md.analysis.reweight_weights`, to turn a biased
        trajectory's latent density into an unbiased free-energy surface.
    """
    from .style import FES_CMAP

    cmap = cmap or FES_CMAP
    z = _to_np(z)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    w = None if weights is None else _to_np(weights).astype(float)
    kde = gaussian_kde(z.T, weights=w)
    pad = 0.3
    xmin, ymin = z.min(axis=0) - pad
    xmax, ymax = z.max(axis=0) + pad
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, grid), np.linspace(ymin, ymax, grid))
    p = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    with np.errstate(divide="ignore"):
        FE = -kT * np.log(p + 1e-30)
    FE -= FE.min()
    if max_FE is not None:
        # blank, not clipped: an unsampled region must not look like a plateau
        FE = np.where(FE > max_FE, np.nan, FE)

    cf = ax.contourf(xx, yy, FE, levels=20, cmap=cmap)
    ax.contour(xx, yy, FE, levels=10, colors="k", linewidths=0.3, alpha=0.4)
    cb = plt.colorbar(cf, ax=ax, fraction=0.045, pad=0.04)
    cb.set_label("FE / kT" if kT == 1.0 else f"FE (x {kT:g})")

    if overlay_points:
        ax.scatter(z[:, 0], z[:, 1], s=1, c="w", alpha=point_alpha)

    ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Latent free-energy surface")
    return ax


def latent_colored_by(
    z,
    values,
    ax=None,
    cmap: str = "viridis",
    s: float = 6.0,
    alpha: float = 0.7,
    cbar_label: str | None = None,
    title: str | None = None,
):
    """Scatter of latent points colored by an arbitrary per-frame scalar
    (e.g. ground-truth pose id, cluster id, time, an order parameter)."""
    z = _to_np(z); values = _to_np(values)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(z[:, 0], z[:, 1], c=values, cmap=cmap, s=s, alpha=alpha)
    cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.04)
    if cbar_label:
        cb.set_label(cbar_label)
    ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]")
    ax.set_aspect("equal", adjustable="datalim")
    if title:
        ax.set_title(title)
    return ax


def latent_trajectory(
    z,
    ax=None,
    arrow_every: int = 50,
    cmap: str = "viridis",
    linewidth: float = 0.6,
):
    """Draw the latent path as a time-colored line with occasional arrows,
    useful for seeing transitions between poses along an MD trajectory."""
    z = _to_np(z)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    t = np.arange(len(z))
    # colored line via LineCollection
    from matplotlib.collections import LineCollection
    pts = z.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, linewidth=linewidth, alpha=0.6)
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    ax.autoscale()

    if arrow_every > 0:
        for i in range(0, len(z) - 1, arrow_every):
            dx, dy = z[i + 1] - z[i]
            if dx == 0 and dy == 0:
                continue
            ax.annotate("", xy=z[i + 1], xytext=z[i],
                        arrowprops=dict(arrowstyle="->", color="k", alpha=0.5, lw=0.6))

    ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Latent trajectory (colored by time)")
    return ax
