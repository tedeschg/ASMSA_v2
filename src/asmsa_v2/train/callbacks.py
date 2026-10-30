"""Callback API used by :class:`asmsa_v2.train.trainer.Trainer`."""
from __future__ import annotations

import csv
import math
from pathlib import Path


class Callback:
    """Base class. Override whichever hooks you need; all are no-ops.

    Hook order within one epoch:

    1. :meth:`on_epoch_metrics` — *contribute* extra metrics by mutating
       ``logs``. Runs before model selection, so a metric added here can be
       used as ``TrainConfig.monitor`` and lands in ``trainer.history``.
    2. best-weight tracking and history recording (trainer-internal).
    3. :meth:`on_epoch_end` — *react* to the finished epoch (log, plot, stop).

    Compute a metric in step 1 and consume it in step 3.
    """

    def on_train_begin(self, trainer): pass
    def on_train_end(self, trainer): pass
    def on_epoch_begin(self, trainer, epoch): pass
    def on_epoch_metrics(self, trainer, epoch, logs): pass
    def on_epoch_end(self, trainer, epoch, logs): pass
    def on_batch_end(self, trainer, step, logs): pass


class CsvLogger(Callback):
    """Append per-epoch metrics to a CSV.

    Columns are discovered from the first epoch's logs, so the file adapts to
    whichever metrics the configured objective actually produces.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = None
        self._writer = None
        self._columns: list[str] | None = None

    def on_train_begin(self, trainer):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._columns = None

    def on_epoch_end(self, trainer, epoch, logs):
        if self._writer is None:
            return
        if self._columns is None:
            self._columns = sorted(logs.keys())
            self._writer.writerow(["epoch"] + self._columns)
        self._writer.writerow(
            [epoch] + [logs.get(c, math.nan) for c in self._columns]
        )
        self._fh.flush()

    def on_train_end(self, trainer):
        if self._fh:
            self._fh.close()
            self._fh = None


class EarlyStopping(Callback):
    """Stop when ``monitor`` has not improved for ``patience`` epochs.

    Defaults to the validation reconstruction loss. Monitoring a *training*
    loss cannot detect overfitting, and under an adversarial objective it is
    not even monotone, so a train metric is accepted but warned about.

    Best-weight restoration is handled by the trainer
    (``TrainConfig.restore_best``), which already snapshots on
    ``TrainConfig.monitor``; set the two monitors to the same key.
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        monitor: str = "val_loss_recon",
        mode: str = "min",
        verbose: bool = True,
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self._best = math.inf if mode == "min" else -math.inf
        self._bad = 0
        self._warned = False

    def _improved(self, v: float) -> bool:
        if self.mode == "min":
            return v < self._best - self.min_delta
        return v > self._best + self.min_delta

    def on_epoch_end(self, trainer, epoch, logs):
        if self.monitor not in logs:
            if not self._warned:
                import warnings
                warnings.warn(
                    f"EarlyStopping monitors {self.monitor!r} which is not in the "
                    f"epoch logs (available: {sorted(logs)}). Early stopping is "
                    f"inactive. Pass a val_loader to Trainer.fit() to get "
                    f"'val_*' metrics.",
                    stacklevel=2,
                )
                self._warned = True
            return
        if not self._warned and not self.monitor.startswith("val_"):
            import warnings
            warnings.warn(
                f"EarlyStopping is monitoring the training metric "
                f"{self.monitor!r}; it cannot detect overfitting. Prefer a "
                f"'val_*' metric.",
                stacklevel=2,
            )
            self._warned = True

        v = logs[self.monitor]
        if not math.isfinite(v):
            return
        if self._improved(v):
            self._best = v
            self._bad = 0
        else:
            self._bad += 1
            if self._bad >= self.patience:
                trainer.stop = True
                if self.verbose:
                    print(
                        f"EarlyStopping: {self.monitor} did not improve for "
                        f"{self.patience} epochs (best {self._best:.5g} at "
                        f"epoch {trainer.best_epoch}); stopping at epoch {epoch}."
                    )


class ModelCheckpoint(Callback):
    """Write a checkpoint each time ``monitor`` improves.

    Pairs with :meth:`Trainer.save_checkpoint`, so the saved file carries the
    optimizer state and config too and training can be resumed from it.
    """

    def __init__(
        self,
        path: str | Path,
        monitor: str = "val_loss_recon",
        mode: str = "min",
        save_last: bool = False,
        verbose: bool = False,
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.path = Path(path)
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.verbose = verbose
        self._best = math.inf if mode == "min" else -math.inf
        self.saved_epoch: int | None = None

    def on_epoch_end(self, trainer, epoch, logs):
        v = logs.get(self.monitor)
        if v is None or not math.isfinite(v):
            return
        better = v < self._best if self.mode == "min" else v > self._best
        if better:
            self._best = v
            self.saved_epoch = epoch
            trainer.save_checkpoint(self.path, monitor=self.monitor, monitor_value=v)
            if self.verbose:
                print(f"  checkpoint: epoch {epoch}, {self.monitor}={v:.5g} -> {self.path}")

    def on_train_end(self, trainer):
        if self.save_last:
            trainer.save_checkpoint(self.path.with_suffix(".last" + self.path.suffix))


class LiveLossPlot(Callback):
    """Live training curves for notebooks.

    Updates one figure in place via a display handle rather than calling
    ``clear_output``, so it does not fight with the tqdm progress bar or wipe
    earlier cell output. Train and validation curves are drawn on the same
    axes so divergence between them is visible immediately.
    """

    def __init__(self, figsize: tuple[float, float] = (11, 3.4), every: int = 1,
                 log_scale: bool = True):
        self.figsize = figsize
        self.every = max(1, int(every))
        self.log_scale = log_scale
        self.history: dict[str, list[float]] = {}
        self._handle = None
        self._fig = None
        try:
            import matplotlib.pyplot as plt  # noqa: F401
            from IPython.display import display  # noqa: F401
            self._enabled = True
        except Exception:
            self._enabled = False

    # metric name -> which panel it belongs in
    _PANELS = (
        ("reconstruction", ("loss_recon", "val_loss_recon")),
        ("latent penalty", ("loss_latent", "val_loss_latent")),
        ("discriminator", ("loss_d",)),
    )

    def on_epoch_end(self, trainer, epoch, logs):
        if not self._enabled:
            return
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        if (epoch + 1) % self.every and epoch + 1 != trainer.cfg.epochs:
            return

        import matplotlib.pyplot as plt
        from IPython.display import display

        panels = [(t, ks) for t, ks in self._PANELS
                  if any(k in self.history for k in ks)]
        if not panels:
            return
        if self._fig is not None:
            plt.close(self._fig)
        fig, axes = plt.subplots(1, len(panels), figsize=self.figsize, squeeze=False)
        self._fig = fig
        for ax, (title, keys) in zip(axes[0], panels):
            for k in keys:
                if k not in self.history:
                    continue
                y = self.history[k]
                ax.plot(range(1, len(y) + 1), y,
                        label=k, lw=1.6,
                        ls="--" if k.startswith("val_") else "-")
            if self.log_scale and all(
                v > 0 for k in keys if k in self.history for v in self.history[k]
            ):
                ax.set_yscale("log")
            ax.set_xlabel("epoch")
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
        if trainer.best_epoch is not None:
            for ax in axes[0]:
                ax.axvline(trainer.best_epoch + 1, color="k", ls=":", lw=1, alpha=0.6)
        fig.suptitle(
            f"epoch {epoch + 1}/{trainer.cfg.epochs}"
            + (f"   best {trainer.cfg.monitor}={trainer.best_score:.5g} "
               f"@ epoch {trainer.best_epoch + 1}"
               if trainer.best_score is not None and trainer.best_epoch is not None
               else ""),
            fontsize=10,
        )
        fig.tight_layout()
        if self._handle is None:
            self._handle = display(fig, display_id=True)
        else:
            self._handle.update(fig)
        plt.close(fig)

    def on_train_end(self, trainer):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None


class LatentSnapshot(Callback):
    """Record latent coordinates every ``every`` epochs.

    Lets you watch the latent space organize itself over training, which is
    the single most informative diagnostic for an adversarial autoencoder:
    a latent that keeps drifting or collapses to a line is obvious here long
    before it shows up in the loss curves.
    """

    def __init__(self, tensors_by_mol, every: int = 5, max_frames: int = 4000):
        self.tensors = {
            k: (v[:max_frames] if hasattr(v, "__getitem__") else v)
            for k, v in tensors_by_mol.items()
        }
        self.every = max(1, int(every))
        self.snapshots: list[tuple[int, dict]] = []

    def on_epoch_end(self, trainer, epoch, logs):
        if epoch % self.every and epoch + 1 != trainer.cfg.epochs:
            return
        z = trainer.encode_all(self.tensors)
        self.snapshots.append((epoch, {k: v.numpy() for k, v in z.items()}))

    def plot(self, ncols: int = 5, figsize_per: float = 2.4, mol_id: str | None = None):
        """Grid of the recorded latents, one panel per snapshot."""
        import matplotlib.pyplot as plt

        if not self.snapshots:
            raise RuntimeError("no snapshots recorded")
        key = mol_id or next(iter(self.snapshots[0][1]))
        n = len(self.snapshots)
        ncols = min(ncols, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(figsize_per * ncols, figsize_per * nrows),
                                 squeeze=False)
        for ax in axes.ravel():
            ax.set_axis_off()
        for ax, (epoch, zs) in zip(axes.ravel(), self.snapshots):
            z = zs[key]
            ax.set_axis_on()
            ax.scatter(z[:, 0], z[:, 1], s=1.5, alpha=0.3)
            ax.set_title(f"epoch {epoch + 1}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
        fig.suptitle(f"latent evolution — {key}", fontsize=11)
        fig.tight_layout()
        return fig
