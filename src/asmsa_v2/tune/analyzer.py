"""Inspect a :class:`~asmsa_v2.tune.tuner.TuningResult`.

Replaces v1's ``TuningAnalyzer``, which depended on a TensorBoard
``SummaryWriter`` and on unpickling per-trial files from a directory chosen by
modification time. Everything here reads the single JSON a tuning run writes,
so results are portable and a notebook never has to guess which directory
held the latest run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .tuner import TuningResult


class TuningAnalyzer:
    """Plots and tables over a tuning run.

    Parameters
    ----------
    result : TuningResult or path
        A result object, or a path to the ``tuning.json`` written by
        :func:`asmsa_v2.tune.tune`.
    """

    def __init__(self, result: TuningResult | str | Path):
        if isinstance(result, (str, Path)):
            p = Path(result)
            if p.is_dir():
                p = p / "tuning.json"
            if not p.exists():
                raise FileNotFoundError(
                    f"no tuning results at {p}. Run asmsa_v2.tune.tune(out_dir=...) first."
                )
            result = TuningResult.load(p)
        self.result = result

    # -------------------------------------------------------------- tabular
    def best_params(self) -> dict:
        return self.result.best_params

    def table(self, n: int = 10):
        """Top-``n`` trials as a DataFrame."""
        df = self.result.to_dataframe()
        return df.head(n)

    def print_best(self, n: int = 5) -> None:
        for t in self.result.top(n):
            print(f"--- trial #{t.number}   score={t.score:.5f}   "
                  f"({t.duration_s:.0f}s)")
            for k, v in sorted(t.params.items()):
                print(f"      {k:<20} {v}")
            if t.metrics:
                print("      metrics: " + ", ".join(
                    f"{k}={v:.4g}" for k, v in sorted(t.metrics.items())))

    # ---------------------------------------------------------------- plots
    def plot_scores(self, ax=None):
        """Score per trial, with the running best overlaid.

        A running best that is still dropping at the end of the run means the
        budget was too small — the single most useful thing to know after
        tuning.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 3.2))
        ts = sorted(self.result.trials, key=lambda t: t.number)
        nums = [t.number for t in ts]
        scores = np.array([t.score for t in ts], dtype=float)
        finite = np.isfinite(scores)
        ax.scatter(np.array(nums)[finite], scores[finite], s=22, alpha=0.75,
                   label="trial", zorder=3)
        if finite.any():
            acc = np.minimum.accumulate if self.result.direction == "minimize" \
                else np.maximum.accumulate
            ax.plot(np.array(nums)[finite], acc(scores[finite]),
                    color="crimson", lw=1.8, label="running best", zorder=2)
        n_bad = int((~finite).sum())
        ax.set_xlabel("trial")
        ax.set_ylabel(f"{self.result.objective} score")
        ax.set_title(
            f"tuning progress ({self.result.direction})"
            + (f" — {n_bad} trial(s) failed/pruned" if n_bad else "")
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        return ax

    def plot_curves(self, n: int = 5, metric: str = "kl_prior", ax=None):
        """Per-epoch curves for the top-``n`` trials.

        v1's guidance applies: a good trial shows a KL curve that is smooth
        and settled beyond roughly the 20th epoch. A curve still oscillating
        at the end means the model never converged, whatever its final score.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 3.2))
        plotted = 0
        for t in self.result.top(n):
            y = t.kl_curve if metric == "kl_prior" and t.kl_curve else t.history.get(metric)
            if not y:
                continue
            ax.plot(range(1, len(y) + 1), y, lw=1.4,
                    label=f"#{t.number} ({t.score:.3f})")
            plotted += 1
        if plotted == 0:
            ax.text(0.5, 0.5, f"no {metric!r} recorded", ha="center", va="center",
                    transform=ax.transAxes, color="#57606a")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} for the top {plotted} trial(s)")
        ax.grid(alpha=0.25)
        if plotted:
            ax.legend(fontsize=8, title="trial (score)", title_fontsize=8)
        return ax

    def plot_param_effect(self, param: str, ax=None):
        """Score distribution grouped by the values of one hyperparameter."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 3.2))
        vals, scores = [], []
        for t in self.result.trials:
            if param in t.params and np.isfinite(t.score):
                vals.append(t.params[param])
                scores.append(t.score)
        if not vals:
            raise ValueError(f"{param!r} was not searched over in this run")

        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)
        if numeric and len(set(vals)) > 6:
            ax.scatter(vals, scores, s=22, alpha=0.75)
            ax.set_xlabel(param)
            if min(vals) > 0 and max(vals) / min(vals) > 50:
                ax.set_xscale("log")
        else:
            groups: dict = {}
            for v, s in zip(vals, scores):
                groups.setdefault(str(v), []).append(s)
            keys = sorted(groups, key=lambda k: np.median(groups[k]))
            ax.boxplot([groups[k] for k in keys], tick_labels=keys)
            ax.set_xlabel(param)
        ax.set_ylabel(f"{self.result.objective} score")
        ax.set_title(f"effect of {param}")
        ax.grid(alpha=0.25, axis="y")
        return ax

    def plot_overview(self, params: list[str] | None = None, max_params: int = 6):
        """One figure: progress, top curves, and the most-varied parameters."""
        import matplotlib.pyplot as plt

        if params is None:
            params = []
            for t in self.result.trials:
                for k, v in t.params.items():
                    if k not in params:
                        params.append(k)
            varied = []
            for k in params:
                seen = {str(t.params.get(k)) for t in self.result.trials if k in t.params}
                if len(seen) > 1:
                    varied.append(k)
            params = varied[:max_params]

        n_panels = 2 + len(params)
        ncols = 3
        nrows = (n_panels + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.1 * nrows),
                                 squeeze=False)
        flat = axes.ravel()
        for a in flat:
            a.set_axis_off()
        self.plot_scores(ax=_on(flat[0]))
        self.plot_curves(ax=_on(flat[1]))
        for a, p in zip(flat[2:], params):
            try:
                self.plot_param_effect(p, ax=_on(a))
            except ValueError:
                a.set_axis_off()
        fig.suptitle(
            f"Tuning overview — {len(self.result.trials)} trials, "
            f"objective {self.result.objective}", fontsize=12,
        )
        fig.tight_layout()
        return fig

    def _repr_html_(self) -> str:
        return self.result._repr_html_()

    def __repr__(self) -> str:
        return f"TuningAnalyzer({self.result.summary(5)})"


def _on(ax):
    ax.set_axis_on()
    return ax
