"""Hyperparameter tuning.

v1 scored each trial by the k-NN KL divergence between the prior and the
aggregate posterior, averaged over the tail of training and penalized by its
variance (``min(mean + var)`` over the last ``1 - threshold`` fraction of
epochs). That is a good criterion — it measures the thing the adversarial
game is actually supposed to achieve, and the variance term rejects models
whose latent never settles — so it is reproduced here exactly, alongside
newer alternatives (validation reconstruction, VAMP-2, and a composite).

Optuna replaces keras-tuner: it is framework-agnostic (v1's tuner was welded
to TensorFlow and could not survive the PyTorch rewrite), its TPE sampler
beats random search at equal budget, and its pruners abandon hopeless trials
early instead of running every one to the full epoch count. If Optuna is not
installed the module falls back to plain random search so the notebook still
runs.
"""
from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from ..metrics.kl import prior_match_score
from ..metrics.vamp import vamp2_score
from ..models.aae import AAE
from ..train.callbacks import Callback
from ..train.trainer import Trainer
from .space import SearchSpace, params_to_configs

OBJECTIVES = ("kl", "val_recon", "vamp2", "composite")
_MAXIMIZE = {"vamp2"}


class KLTracker(Callback):
    """Record the prior/posterior KL divergence after every epoch.

    This is the per-epoch series v1's tuner aggregated into a trial score.
    ``max_frames`` caps the cost of the KD-tree query, which is the dominant
    expense of tuning on long trajectories.
    """

    def __init__(self, tensors_by_mol, max_frames: int = 4000):
        self.tensors = {k: torch.as_tensor(v)[:max_frames] for k, v in tensors_by_mol.items()}
        self.kl: list[float] = []

    def on_epoch_metrics(self, trainer, epoch, logs):
        z = trainer.encode_all(self.tensors)
        vals = []
        for zz in z.values():
            try:
                vals.append(prior_match_score(trainer.prior, zz))
            except Exception:
                vals.append(float("nan"))
        kl = float(np.nanmean(vals)) if vals else float("nan")
        self.kl.append(kl)
        logs["kl_prior"] = kl

    def score(self, threshold: float = 0.25) -> float:
        """v1's aggregation: ``mean + var`` over the last ``1 - threshold`` of epochs."""
        if not self.kl:
            return float("inf")
        arr = np.asarray(self.kl[int(len(self.kl) * threshold):], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("inf")
        return float(arr.mean() + arr.var())


@dataclass
class Trial:
    """One evaluated hyperparameter setting."""

    number: int
    params: dict[str, Any]
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)
    kl_curve: list[float] = field(default_factory=list)
    duration_s: float = 0.0
    state: str = "complete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number, "params": self.params, "score": self.score,
            "metrics": self.metrics, "history": self.history,
            "kl_curve": self.kl_curve, "duration_s": self.duration_s,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trial":
        return cls(**d)


@dataclass
class TuningResult:
    """All trials from one tuning run, best first."""

    trials: list[Trial]
    objective: str
    direction: str
    space: dict[str, Any] = field(default_factory=dict)
    epochs: int = 0

    def __post_init__(self):
        rev = self.direction == "maximize"
        self.trials = sorted(
            self.trials,
            key=lambda t: (not math.isfinite(t.score),
                           -t.score if rev else t.score),
        )

    @property
    def best(self) -> Trial:
        if not self.trials:
            raise RuntimeError("no trials")
        return self.trials[0]

    @property
    def best_params(self) -> dict[str, Any]:
        return dict(self.best.params)

    def top(self, n: int = 10) -> list[Trial]:
        return self.trials[:n]

    def to_configs(self, trial: Trial | int | None = None, epochs: int | None = None, **kw):
        """``(ModelConfig, TrainConfig)`` for a trial — by default the best one."""
        t = self.best if trial is None else (
            self.trials[trial] if isinstance(trial, int) else trial
        )
        return params_to_configs(t.params, epochs=epochs or self.epochs, **kw)

    def to_dataframe(self):
        """Trials as a pandas DataFrame (one row per trial)."""
        import pandas as pd

        rows = []
        for t in self.trials:
            row = {"number": t.number, "score": t.score, "state": t.state,
                   "duration_s": round(t.duration_s, 1)}
            row.update(t.params)
            row.update({f"metric_{k}": v for k, v in t.metrics.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "objective": self.objective, "direction": self.direction,
            "space": self.space, "epochs": self.epochs,
            "trials": [t.to_dict() for t in self.trials],
        }, indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TuningResult":
        d = json.loads(Path(path).read_text())
        return cls(
            trials=[Trial.from_dict(t) for t in d["trials"]],
            objective=d["objective"], direction=d["direction"],
            space=d.get("space", {}), epochs=d.get("epochs", 0),
        )

    def summary(self, n: int = 10) -> str:
        lines = [
            f"TuningResult: {len(self.trials)} trials, objective={self.objective} "
            f"({self.direction}), {self.epochs} epochs/trial"
        ]
        for t in self.top(n):
            keys = ("latent_reg", "ae_neuron_seed", "ae_layers",
                    "disc_neuron_seed", "disc_layers", "batch_size",
                    "lr_ae", "lam_latent")
            ps = " ".join(
                f"{k}={t.params[k]:.4g}" if isinstance(t.params.get(k), float)
                else f"{k}={t.params[k]}"
                for k in keys if k in t.params
            )
            lines.append(f"  #{t.number:<3} score={t.score:>9.4f}  {ps}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        keys = [k for k in ("latent_reg", "lam_latent", "ae_neuron_seed", "ae_layers",
                            "disc_neuron_seed", "disc_layers", "batch_size", "lr_ae",
                            "lr_d", "activation", "recon_loss")
                if any(k in t.params for t in self.trials)]
        head = "".join(f"<th style='padding:2px 8px'>{k}</th>" for k in keys)
        rows = []
        for i, t in enumerate(self.top(10)):
            cells = "".join(
                "<td style='padding:2px 8px;text-align:right'>"
                + (f"{t.params[k]:.4g}" if isinstance(t.params.get(k), float)
                   else f"{t.params.get(k, '')}")
                + "</td>"
                for k in keys
            )
            bg = "background:#ddf4e4" if i == 0 else ""
            rows.append(
                f"<tr style='{bg}'><td style='padding:2px 8px'>#{t.number}</td>"
                f"<td style='padding:2px 8px;text-align:right'><b>{t.score:.4f}</b></td>"
                f"{cells}</tr>"
            )
        return (
            "<table style='border-collapse:collapse;font-size:90%'>"
            f"<caption style='text-align:left;padding-bottom:4px'><b>Tuning</b> &mdash; "
            f"{len(self.trials)} trials, objective <code>{self.objective}</code> "
            f"({self.direction}), {self.epochs} epochs/trial. Best row highlighted."
            "</caption>"
            f"<tr style='border-bottom:1px solid #d0d7de'><th style='padding:2px 8px'>trial</th>"
            f"<th style='padding:2px 8px'>score</th>{head}</tr>"
            + "".join(rows) + "</table>"
        )

    def __repr__(self) -> str:
        return self.summary()


def _score_trial(objective: str, trainer: Trainer, kl_cb: KLTracker | None,
                 val_tensors, kl_threshold: float, vamp_lag: int) -> tuple[float, dict]:
    metrics: dict[str, float] = {}

    if kl_cb is not None and kl_cb.kl:
        metrics["kl_tail"] = kl_cb.score(kl_threshold)
        metrics["kl_final"] = kl_cb.kl[-1]

    hist = trainer.history
    if "val_loss_recon" in hist and hist["val_loss_recon"]:
        metrics["val_recon"] = float(min(hist["val_loss_recon"]))
    elif "loss_recon" in hist and hist["loss_recon"]:
        metrics["val_recon"] = float(min(hist["loss_recon"]))

    if objective in ("vamp2", "composite"):
        z = trainer.encode_all(val_tensors)
        vals = []
        for zz in z.values():
            try:
                vals.append(vamp2_score(zz.numpy(), lag=vamp_lag))
            except Exception:
                pass
        if vals:
            metrics["vamp2"] = float(np.mean(vals))

    if objective == "kl":
        score = metrics.get("kl_tail", float("inf"))
    elif objective == "val_recon":
        score = metrics.get("val_recon", float("inf"))
    elif objective == "vamp2":
        score = metrics.get("vamp2", -float("inf"))
    else:  # composite: match the prior AND reconstruct AND be slow
        kl = metrics.get("kl_tail", float("inf"))
        rec = metrics.get("val_recon", float("inf"))
        v = metrics.get("vamp2", 1.0)
        d = max(trainer.model.latent_dim, 1)
        score = kl + rec - (v - 1.0) / d
    return float(score), metrics


def tune(
    train_tensors: dict[str, torch.Tensor],
    val_tensors: dict[str, torch.Tensor],
    prior_factory: Callable[[int], Any],
    space: SearchSpace | None = None,
    n_trials: int = 30,
    epochs: int = 40,
    objective: str = "kl",
    kl_threshold: float = 0.25,
    vamp_lag: int = 1,
    device: str | None = None,
    seed: int = 0,
    n_jobs: int = 1,
    prune: bool = True,
    out_dir: str | Path | None = None,
    progress: bool = True,
    extra_train_kwargs: dict | None = None,
) -> TuningResult:
    """Search hyperparameters and return the ranked trials.

    Parameters
    ----------
    train_tensors, val_tensors : mapping ``mol_id -> [n_frames, n_features]``
        **Standardized** features (fit the :class:`FeatureScaler` on the
        training split only, then transform both).
    prior_factory : callable ``latent_dim -> Prior``
        A factory rather than an instance, because ``latent_dim`` can itself
        be searched over.
    space : SearchSpace
        Defaults to :data:`asmsa_v2.tune.space.SMALL_SPACE`.
    objective : {"kl", "val_recon", "vamp2", "composite"}
        ``"kl"`` reproduces v1's criterion and is the default.
    kl_threshold : float
        Fraction of initial epochs discarded before aggregating the KL curve,
        as in v1 (0.25 = ignore the first quarter).
    vamp_lag : int
        Lag in frames for the VAMP-2 score. Required to be meaningful for
        the ``"vamp2"`` and ``"composite"`` objectives; ``val_tensors`` must
        then be in trajectory order.
    n_jobs : int
        Parallel trials. Each trial is small, so the useful range is roughly
        the core count divided by ``torch.get_num_threads()``.
    prune : bool
        Let Optuna abandon trials that are clearly behind (median pruner).
    out_dir : path, optional
        Where to write ``tuning.json``. Reload later with
        :meth:`TuningResult.load` — the v2 equivalent of v1's trial pickles.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; use one of {OBJECTIVES}")
    space = space or SearchSpace()
    direction = "maximize" if objective in _MAXIMIZE else "minimize"
    extra_train_kwargs = dict(extra_train_kwargs or {})

    from ..data.dataset import MultiMolDataset

    train_ds_cache = MultiMolDataset(train_tensors)
    val_ds_cache = MultiMolDataset(val_tensors)
    needs_kl = objective in ("kl", "composite")

    trials: list[Trial] = []

    def run_one(number: int, params: dict[str, Any], report=None) -> tuple[float, Trial]:
        t0 = time.time()
        model_cfg, train_cfg = params_to_configs(
            params, epochs=epochs, device=device, seed=seed, **extra_train_kwargs
        )
        bs = train_cfg.batch_size
        try:
            train_loader = train_ds_cache.loader(batch_size=bs)
            val_loader = val_ds_cache.loader(batch_size=bs, shuffle=False)
        except ValueError as exc:
            # batch_size larger than a split -> unusable trial, not a crash
            return (float("inf") if direction == "minimize" else -float("inf")), Trial(
                number=number, params=params,
                score=float("inf") if direction == "minimize" else -float("inf"),
                state=f"invalid: {exc}",
            )

        model = AAE(
            feature_dims=train_ds_cache.feature_dims,
            latent_dim=model_cfg.latent_dim,
            encoder_widths=model_cfg.encoder_widths,
            decoder_widths=model_cfg.decoder_widths,
            disc_widths=model_cfg.disc_widths,
            activation=model_cfg.activation,
            batchnorm=model_cfg.batchnorm,
            spectral_norm_disc=model_cfg.spectral_norm_disc,
        )
        prior = prior_factory(model_cfg.latent_dim)
        kl_cb = KLTracker(val_ds_cache.tensors()) if needs_kl else None
        cbs: list[Callback] = [kl_cb] if kl_cb else []
        if report is not None:
            cbs.append(_PruneReporter(report, objective, direction))

        train_cfg.restore_best = False  # scoring uses the whole curve
        trainer = Trainer(model, prior, train_cfg, callbacks=cbs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.fit(train_loader, val_loader)

        score, metrics = _score_trial(
            objective, trainer, kl_cb, val_ds_cache.tensors(), kl_threshold, vamp_lag
        )
        tr = Trial(
            number=number, params=params, score=score, metrics=metrics,
            history={k: list(map(float, v)) for k, v in trainer.history.items()},
            kl_curve=list(map(float, kl_cb.kl)) if kl_cb else [],
            duration_s=time.time() - t0,
        )
        return score, tr

    try:
        import optuna
        have_optuna = True
    except ImportError:
        have_optuna = False

    if have_optuna:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner = (optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=max(5, epochs // 4))
                  if prune else optuna.pruners.NopPruner())
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

        def objective_fn(trial):
            params = space.suggest(trial)
            try:
                score, tr = run_one(trial.number, params, report=trial)
            except optuna.TrialPruned:
                # Record the pruned trial rather than dropping it, so the
                # analyzer can report how much of the budget was reclaimed.
                trials.append(Trial(
                    number=trial.number, params=params,
                    score=float("inf") if direction == "minimize" else -float("inf"),
                    state="pruned",
                ))
                raise
            trials.append(tr)
            return score

        study.optimize(
            objective_fn, n_trials=n_trials, n_jobs=n_jobs,
            show_progress_bar=progress,
            catch=(RuntimeError, ValueError, torch.linalg.LinAlgError)
            if hasattr(torch.linalg, "LinAlgError") else (RuntimeError, ValueError),
        )
    else:
        warnings.warn(
            "optuna is not installed; falling back to random search "
            "(no TPE sampling, no pruning). `pip install asmsa_v2[tune]`.",
            stacklevel=2,
        )
        rng = np.random.default_rng(seed)
        it = range(n_trials)
        if progress:
            from tqdm.auto import tqdm
            it = tqdm(it, desc="random search", unit="trial")
        for i in it:
            _, tr = run_one(i, space.sample_random(rng))
            trials.append(tr)

    result = TuningResult(
        trials=trials, objective=objective, direction=direction,
        space=space.to_dict(), epochs=epochs,
    )
    if out_dir is not None:
        result.save(Path(out_dir) / "tuning.json")
    return result


class _PruneReporter(Callback):
    """Feed the running objective to Optuna so hopeless trials can be pruned."""

    def __init__(self, trial, objective: str, direction: str):
        self.trial = trial
        self.objective = objective
        self.direction = direction

    def on_epoch_end(self, trainer, epoch, logs):
        import optuna

        if self.objective == "kl":
            v = logs.get("kl_prior")
        elif self.objective == "val_recon":
            v = logs.get("val_loss_recon", logs.get("loss_recon"))
        else:
            v = logs.get("val_loss_recon", logs.get("loss_recon"))
            if v is not None and self.direction == "maximize":
                v = -v
        if v is None or not math.isfinite(v):
            return
        self.trial.report(float(v), epoch)
        if self.trial.should_prune():
            trainer.stop = True
            raise optuna.TrialPruned()
