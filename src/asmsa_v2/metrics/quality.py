"""A single bundled report on how good a trained latent space is.

Four questions matter, and no single number answers them:

reconstruction
    Did the autoencoder keep the information? (test MSE, and MSE relative to
    the variance of the standardized features, so 1.0 means "no better than
    predicting the mean".)
prior match
    Did the adversary/MMD actually shape the latent into the prior? (k-NN KL
    divergence — v1's model-selection criterion.)
slow dynamics
    Is the latent a *useful collective variable*, i.e. does it track the slow
    processes? (VAMP-2 score.)
usable spread
    Is the latent actually two-dimensional, or did it collapse onto a line?
    (participation ratio of the latent covariance eigenvalues.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .kl import prior_match_score
from .vamp import vamp2_score

_NAN = float("nan")


def _as_np(x) -> np.ndarray:
    """Convert a tensor or array-like to a NumPy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _grade(
    value: float,
    good: float,
    ok: float,
    *,
    lower_is_better: bool = True,
) -> str:
    """Map a value to ``"good"``, ``"ok"``, ``"poor"`` or ``"n/a"`` (non-finite)."""
    if not np.isfinite(value):
        return "n/a"
    if lower_is_better:
        return "good" if value < good else "ok" if value < ok else "poor"
    return "good" if value > good else "ok" if value > ok else "poor"


def effective_dimension(z) -> float:
    """Participation ratio of the latent covariance eigenvalues.

    ``(sum w)^2 / sum(w^2)`` over the eigenvalues of ``cov(z)``. Equals ``d``
    for an isotropic latent and 1 if every point lies on a single line, so it
    detects the classic failure where a 2-D latent collapses to 1-D.
    """
    Z = _as_np(z)
    if Z.ndim != 2 or Z.shape[0] < 2:
        return _NAN
    C = np.atleast_2d(np.cov(Z.T))
    w = np.linalg.eigvalsh(0.5 * (C + C.T))
    w = np.clip(w, 0.0, None)
    denom = float((w**2).sum())
    if denom <= 0:
        return _NAN
    return float(w.sum() ** 2 / denom)


@dataclass
class QualityReport:
    """Scores for one trained model. Display it in a notebook to get a table."""

    recon_mse: float = _NAN
    recon_relative: float = _NAN
    kl_prior: float = _NAN
    vamp2: float = _NAN
    vamp2_lag: int = 0
    effective_dim: float = _NAN
    latent_dim: int = 0
    n_frames: int = 0
    extra: dict[str, float] = field(default_factory=dict)

    # --------------------------------------------------------------- verdicts
    def _verdicts(self) -> list[tuple[str, str, str, str]]:
        """``(label, formatted value, verdict, what it means)`` rows."""
        rows: list[tuple[str, str, str, str]] = []

        r = self.recon_relative
        rows.append(
            (
                "reconstruction (relative MSE)",
                f"{r:.3f}" if np.isfinite(r) else "n/a",
                _grade(r, good=0.2, ok=0.5),
                "fraction of standardized feature variance left unexplained; "
                "1.0 = no better than predicting the mean",
            )
        )

        k = self.kl_prior
        rows.append(
            (
                "prior match (KL, nats)",
                f"{k:.3f}" if np.isfinite(k) else "n/a",
                _grade(k, good=0.3, ok=0.7),
                "k-NN KL between prior and latent samples; v1 treated < 0.5-0.7 "
                "as acceptable",
            )
        )

        v, vmax = self.vamp2, 1.0 + max(self.latent_dim, 1)
        frac = (
            (v - 1.0) / max(vmax - 1.0, 1e-12) if np.isfinite(v) else _NAN
        )
        rows.append(
            (
                f"slow dynamics (VAMP-2 @ lag {self.vamp2_lag})",
                f"{v:.3f} / {vmax:.0f}" if np.isfinite(v) else "n/a",
                _grade(frac, good=0.6, ok=0.3, lower_is_better=False),
                "how much of the slow subspace the latent resolves; higher is "
                "better, max is 1 + latent_dim",
            )
        )

        e, d = self.effective_dim, self.latent_dim
        rows.append(
            (
                "usable spread (effective dim)",
                f"{e:.2f} / {d}" if np.isfinite(e) else "n/a",
                _grade(e, good=0.8 * d, ok=0.5 * d, lower_is_better=False),
                "participation ratio of latent variance; well below latent_dim "
                "means the latent collapsed to a lower-dimensional manifold",
            )
        )
        return rows

    def summary(self) -> str:
        """Plain-text table of all scores and verdicts."""
        header = (
            f"QualityReport  ({self.n_frames} frames, "
            f"latent_dim={self.latent_dim})"
        )
        lines = [header]
        for label, value, verdict, _ in self._verdicts():
            lines.append(f"  {label:<38} {value:>14}   [{verdict}]")
        if self.extra:
            lines.append("  extra:")
            for k, v in self.extra.items():
                lines.append(f"    {k:<36} {v:>14.5g}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        colors = {
            "good": "#1a7f37",
            "ok": "#9a6700",
            "poor": "#cf222e",
            "n/a": "#57606a",
        }
        cell = "padding:3px 10px"
        right = f"{cell};text-align:right"
        left = f"{cell};text-align:left"

        rows = "".join(
            "<tr>"
            f"<td style='{cell}'>{label}</td>"
            f"<td style='{right};font-variant-numeric:tabular-nums'>"
            f"<b>{value}</b></td>"
            f"<td style='{cell};color:{colors[verdict]}'><b>{verdict}</b></td>"
            f"<td style='{cell};font-size:90%;color:#57606a'>{note}</td>"
            "</tr>"
            for label, value, verdict, note in self._verdicts()
        )
        extra = "".join(
            f"<tr><td style='{cell}'>{k}</td>"
            f"<td style='{right}'><b>{v:.5g}</b></td>"
            "<td colspan='2'></td></tr>"
            for k, v in self.extra.items()
        )
        return (
            "<table style='border-collapse:collapse;font-size:95%'>"
            "<caption style='text-align:left;padding-bottom:4px'>"
            f"<b>Latent quality</b> &mdash; {self.n_frames} frames, "
            f"latent_dim={self.latent_dim}</caption>"
            "<tr style='border-bottom:1px solid #d0d7de'>"
            f"<th style='{left}'>metric</th>"
            f"<th style='{right}'>value</th>"
            f"<th style='{left}'>verdict</th>"
            f"<th style='{left}'>meaning</th></tr>"
            f"{rows}{extra}</table>"
        )

    def __repr__(self) -> str:
        return self.summary()


def evaluate_latent(
    z,
    x=None,
    x_recon=None,
    prior=None,
    vamp_lag: int = 1,
    in_time_order: bool = True,
) -> QualityReport:
    """Build a :class:`QualityReport` from whatever is available.

    Parameters
    ----------
    z : ``[n, d]``
        Latent coordinates.
    x, x_recon : ``[n, f]``, optional
        Standardized input features and their reconstruction. Supply both to
        get the reconstruction scores.
    prior : Prior, optional
        Supply to get the KL prior-match score.
    vamp_lag : int
        Lag (in frames) for the VAMP-2 score.
    in_time_order : bool
        Set ``False`` if ``z`` rows are shuffled, in which case VAMP-2 is
        meaningless and is skipped rather than silently reported as ~1.
    """
    Z = _as_np(z)
    rep = QualityReport(
        latent_dim=int(Z.shape[1]) if Z.ndim == 2 else 0,
        n_frames=int(Z.shape[0]) if Z.ndim == 2 else 0,
        vamp2_lag=int(vamp_lag),
    )

    if x is not None and x_recon is not None:
        X = _as_np(x).astype(np.float64)
        R = _as_np(x_recon).astype(np.float64)
        if X.shape != R.shape:
            raise ValueError(
                f"x {X.shape} and x_recon {R.shape} must have the same shape"
            )
        rep.recon_mse = float(((R - X) ** 2).mean())
        var = float(X.var(axis=0).mean())
        rep.recon_relative = rep.recon_mse / var if var > 0 else _NAN

    if prior is not None:
        try:
            rep.kl_prior = prior_match_score(prior, Z)
        except Exception:
            rep.kl_prior = _NAN

    if in_time_order:
        try:
            rep.vamp2 = vamp2_score(Z, lag=int(vamp_lag))
        except Exception:
            rep.vamp2 = _NAN

    rep.effective_dim = effective_dimension(Z)
    return rep
