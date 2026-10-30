"""Consistent plot styling across the notebooks.

Two things matter beyond looking tidy. The sequential colormap used for
free-energy surfaces must be perceptually uniform, or the eye invents basins
where the colormap merely changes gradient (``turbo`` and ``jet`` do exactly
this). And latent-space axes must share a scale, since the latent plane is
isotropic by construction and a stretched aspect ratio makes a round
distribution look elongated.
"""
from __future__ import annotations

# Perceptually uniform, dark = low free energy (populated), light = high.
FES_CMAP = "viridis_r"
# For continuous per-frame properties (Rg, RMSD, time).
PROPERTY_CMAP = "viridis"
# For discrete cluster ids.
CLUSTER_CMAP = "tab10"

_RC = {
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "image.cmap": PROPERTY_CMAP,
    "scatter.edgecolors": "none",
}


def use_asmsa_style() -> None:
    """Apply the ASMSA matplotlib defaults. Call once per notebook."""
    import matplotlib as mpl

    mpl.rcParams.update(_RC)


def latent_axes(ax, xlabel: str = "CV 1", ylabel: str = "CV 2"):
    """Label a latent-space axis and force an equal aspect ratio."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="datalim")
    return ax
