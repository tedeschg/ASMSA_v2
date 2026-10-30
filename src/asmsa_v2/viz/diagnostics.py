"""Diagnostic figures for the ASMSA pipeline.

These are the plots that answer "is this model usable?", bundled so a
notebook asks the question in one cell instead of ten.
"""
from __future__ import annotations

import numpy as np
import torch

from .style import CLUSTER_CMAP, PROPERTY_CMAP, latent_axes
from .latent import free_energy_surface, latent_colored_by, latent_trajectory


def _np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def feature_scale_report(X, blocks: dict[str, tuple[int, int]] | None = None,
                         axes=None):
    """Per-feature standard deviation, by feature block.

    This is the plot that justifies standardization: on a real complex the
    covalent-bond block has a standard deviation around 0.007 nm while the
    inter-component distances span nanometres, so an unweighted MSE sees only
    the latter. Run it on the *raw* features, before scaling.
    """
    import matplotlib.pyplot as plt

    X = _np(X)
    sd = X.std(axis=0)
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    ax0, ax1 = axes

    ax0.semilogy(np.maximum(sd, 1e-12), lw=0.8)
    ax0.set_xlabel("feature index")
    ax0.set_ylabel("std (raw units)")
    nz = sd[sd > 1e-12]
    ratio = nz.max() / nz.min() if nz.size else float("nan")
    ax0.set_title(f"per-feature spread — max/min = {ratio:.0f}x")
    if blocks:
        for name, (a, b) in blocks.items():
            ax0.axvspan(a, b, alpha=0.10)
            ax0.text((a + b) / 2, ax0.get_ylim()[1], name, ha="center",
                     va="top", fontsize=7, rotation=90)

    if blocks:
        total = float((sd ** 2).sum()) or 1.0
        names = list(blocks)
        shares = [float((sd[a:b] ** 2).sum()) / total * 100 for a, b in blocks.values()]
        ax1.barh(names, shares)
        for i, s in enumerate(shares):
            ax1.text(s, i, f" {s:.3f}%", va="center", fontsize=8)
        ax1.set_xlabel("share of total variance (%)")
        ax1.set_title("variance seen by an unweighted MSE")
        ax1.set_xlim(0, 105)
        ax1.grid(alpha=0.25, axis="x")
    else:
        ax1.hist(np.log10(np.maximum(sd, 1e-12)), bins=40)
        ax1.set_xlabel("log10(std)")
        ax1.set_title("distribution of feature scales")
    return axes


def training_curves(history: dict, ax=None, best_epoch: int | None = None):
    """Train/validation curves with the selected epoch marked."""
    import matplotlib.pyplot as plt

    panels = [("reconstruction", ("loss_recon", "val_loss_recon")),
              ("latent penalty", ("loss_latent", "val_loss_latent")),
              ("discriminator", ("loss_d",))]
    panels = [(t, ks) for t, ks in panels if any(k in history for k in ks)]
    if ax is None:
        _, ax = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.1),
                             squeeze=False)
        ax = ax[0]
    ax = np.atleast_1d(ax)
    for a, (title, keys) in zip(ax, panels):
        for k in keys:
            if k not in history:
                continue
            y = history[k]
            a.plot(range(1, len(y) + 1), y, lw=1.6, label=k,
                   ls="--" if k.startswith("val_") else "-")
        if all(v > 0 for k in keys if k in history for v in history[k]):
            a.set_yscale("log")
        if title == "discriminator" and "loss_d" in history:
            a.axhline(2 * np.log(2), color="k", ls=":", lw=1,
                      label="2·log2 (equilibrium)")
        a.set_xlabel("epoch")
        a.set_title(title)
        a.legend(fontsize=8)
        if best_epoch is not None:
            a.axvline(best_epoch + 1, color="k", ls=":", lw=1, alpha=0.6)
    return ax


def latent_overview(
    z,
    properties: dict[str, np.ndarray] | None = None,
    labels=None,
    kT_units: bool = True,
    max_fe: float = 8.0,
    in_time_order: bool = True,
):
    """One figure answering the usual questions about a trained latent space.

    Panels: free-energy surface, latent path in time, then one panel per
    supplied per-frame property (radius of gyration, RMSD, cluster id, ...).
    """
    import matplotlib.pyplot as plt

    z = _np(z)
    properties = dict(properties or {})
    if labels is not None:
        properties["cluster"] = _np(labels)

    n = 2 + len(properties)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.3 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    for a in flat:
        a.set_axis_off()

    a0 = flat[0]; a0.set_axis_on()
    free_energy_surface(z, ax=a0, kT=1.0 if kT_units else 1.0, max_FE=max_fe)
    latent_axes(a0)

    a1 = flat[1]; a1.set_axis_on()
    if in_time_order:
        latent_trajectory(z, ax=a1, arrow_every=max(1, len(z) // 20))
    else:
        a1.scatter(z[:, 0], z[:, 1], s=2, alpha=0.3)
        a1.set_title("latent scatter (frames not in time order)")
    latent_axes(a1)

    for a, (name, vals) in zip(flat[2:], properties.items()):
        a.set_axis_on()
        cmap = CLUSTER_CMAP if name == "cluster" else PROPERTY_CMAP
        latent_colored_by(z, vals, ax=a, cmap=cmap, cbar_label=name,
                          title=f"coloured by {name}", s=4, alpha=0.6)
        latent_axes(a)

    fig.suptitle("Latent space overview", fontsize=12)
    fig.tight_layout()
    return fig


def vamp_scan_plot(z, lags=None, dt: float = 1.0, ax=None):
    """VAMP-2 score and implied timescales against lag time.

    The standard convergence check for a kinetic model: the implied
    timescales should plateau once the lag exceeds the fast relaxation, and
    only the plateau value is physically meaningful.
    """
    import matplotlib.pyplot as plt

    from ..metrics.vamp import implied_timescales, vamp2_score

    z = _np(z)
    if lags is None:
        top = max(2, len(z) // 10)
        lags = np.unique(np.geomspace(1, top, 12).astype(int))
    lags = np.asarray([int(l) for l in lags if 1 <= l < len(z) - z.shape[1] - 1])

    scores, its = [], []
    for lag in lags:
        try:
            scores.append(vamp2_score(z, lag=int(lag)))
            its.append(implied_timescales(z, lag=int(lag), dt=dt))
        except Exception:
            scores.append(np.nan)
            its.append(np.full(z.shape[1], np.nan))
    its = np.asarray(its)

    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax = np.atleast_1d(ax)
    ax[0].plot(lags * dt, scores, "o-", lw=1.5)
    ax[0].axhline(1.0, color="k", ls=":", lw=1, label="1.0 (no slow process)")
    ax[0].axhline(1 + z.shape[1], color="k", ls="--", lw=1,
                  label=f"{1 + z.shape[1]} (theoretical max)")
    ax[0].set_xlabel(f"lag ({'frames' if dt == 1.0 else 'time'})")
    ax[0].set_ylabel("VAMP-2 score")
    ax[0].set_title("slow-dynamics content of the latent")
    ax[0].set_xscale("log")
    ax[0].legend(fontsize=8)

    if len(ax) > 1:
        for i in range(its.shape[1]):
            ax[1].plot(lags * dt, its[:, i], "o-", lw=1.5, label=f"CV {i + 1}")
        ax[1].plot(lags * dt, lags * dt, "k:", lw=1, label="lag (resolution limit)")
        ax[1].set_xscale("log"); ax[1].set_yscale("log")
        ax[1].set_xlabel(f"lag ({'frames' if dt == 1.0 else 'time'})")
        ax[1].set_ylabel("implied timescale")
        ax[1].set_title("implied timescales — look for a plateau")
        ax[1].legend(fontsize=8)
    return ax


def reconstruction_report(x, x_recon, blocks=None, axes=None):
    """Per-feature reconstruction error, and the worst-reconstructed features."""
    import matplotlib.pyplot as plt

    x = _np(x); r = _np(x_recon)
    err = ((r - x) ** 2).mean(axis=0)
    var = x.var(axis=0)
    rel = err / np.maximum(var, 1e-12)

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    a0, a1 = axes
    a0.plot(err, lw=0.8)
    a0.set_xlabel("feature index"); a0.set_ylabel("MSE")
    a0.set_title(f"per-feature reconstruction error (mean {err.mean():.4f})")
    if blocks:
        for name, (a, b) in blocks.items():
            a0.axvspan(a, b, alpha=0.10)
            a0.text((a + b) / 2, a0.get_ylim()[1], name, ha="center", va="top",
                    fontsize=7, rotation=90)

    a1.hist(np.clip(rel, 0, 2), bins=50)
    a1.axvline(1.0, color="crimson", ls="--", lw=1.2,
               label="1.0 = no better than the mean")
    a1.set_xlabel("MSE / feature variance")
    a1.set_ylabel("features")
    a1.set_title(f"{int((rel > 1).sum())} of {len(rel)} features unexplained")
    a1.legend(fontsize=8)
    return axes


def fes_plot(x, y, fe, ax=None, label: str = "FE (kJ/mol)", levels: int = 20,
             title: str = "Free-energy surface"):
    """Contour a free-energy surface produced by the MD analysis helpers."""
    import matplotlib.pyplot as plt

    from .style import FES_CMAP

    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 4.6))
    cf = ax.contourf(x, y, fe, levels=levels, cmap=FES_CMAP)
    ax.contour(x, y, fe, levels=min(levels, 12), colors="k", linewidths=0.3,
               alpha=0.45)
    cb = plt.colorbar(cf, ax=ax, fraction=0.045, pad=0.04)
    cb.set_label(label)
    ax.set_title(title)
    latent_axes(ax)
    return ax


def convergence_plot(fes_deviation=None, hill_heights=None, axes=None,
                     stride_label: str = "sum_hills block"):
    """Metadynamics convergence: FES drift and deposited hill height.

    Both should flatten. An FES deviation still falling, or hill heights still
    near their initial value, means the run needs to be extended — reading
    this wrong is the most common way a metadynamics study reports a free
    energy that is simply not converged.
    """
    import matplotlib.pyplot as plt

    n = sum(v is not None for v in (fes_deviation, hill_heights))
    if n == 0:
        raise ValueError("pass fes_deviation and/or hill_heights")
    if axes is None:
        _, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.1), squeeze=False)
        axes = axes[0]
    axes = np.atleast_1d(axes)
    i = 0
    if fes_deviation is not None:
        a = axes[i]; i += 1
        d = np.asarray(fes_deviation)
        a.plot(range(1, len(d) + 1), d, "o-", lw=1.5)
        a.set_xlabel(stride_label)
        a.set_ylabel("mean |FES - final| (kJ/mol)")
        a.set_title("FES convergence — should flatten near 0")
    if hill_heights is not None:
        a = axes[i]
        h = np.asarray(hill_heights)
        a.plot(h, lw=0.9)
        a.set_xlabel("hill index")
        a.set_ylabel("height (kJ/mol)")
        a.set_title("deposited hill height — should decay")
        a.set_yscale("log" if np.all(h > 0) else "linear")
    return axes
