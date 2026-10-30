"""Minimal callback API used by :class:`asmsa_v2.train.trainer.Trainer`."""
from __future__ import annotations

import csv
import math
from pathlib import Path


class Callback:
    def on_train_begin(self, trainer): pass
    def on_train_end(self, trainer): pass
    def on_epoch_begin(self, trainer, epoch): pass
    def on_epoch_end(self, trainer, epoch, logs): pass
    def on_batch_end(self, trainer, step, logs): pass


class CsvLogger(Callback):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = None
        self._writer = None

    def on_train_begin(self, trainer):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["epoch", "loss_recon", "loss_d", "loss_g"])

    def on_epoch_end(self, trainer, epoch, logs):
        self._writer.writerow([epoch,
                               logs.get("loss_recon", math.nan),
                               logs.get("loss_d", math.nan),
                               logs.get("loss_g", math.nan)])
        self._fh.flush()

    def on_train_end(self, trainer):
        if self._fh:
            self._fh.close()


class EarlyStopping(Callback):
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, monitor: str = "loss_recon"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self._best = math.inf
        self._bad = 0

    def on_epoch_end(self, trainer, epoch, logs):
        v = logs.get(self.monitor, math.inf)
        if v < self._best - self.min_delta:
            self._best = v
            self._bad = 0
        else:
            self._bad += 1
            if self._bad >= self.patience:
                trainer.stop = True


class LiveLossPlot(Callback):
    """Lightweight live loss plot for notebooks (matplotlib). No-op outside IPython."""

    def __init__(self):
        self.history = {"loss_recon": [], "loss_d": [], "loss_g": []}
        try:
            from IPython.display import display, clear_output  # noqa: F401
            import matplotlib.pyplot as plt  # noqa: F401
            self._enabled = True
        except Exception:
            self._enabled = False

    def on_epoch_end(self, trainer, epoch, logs):
        if not self._enabled:
            return
        for k in self.history:
            self.history[k].append(logs.get(k, float("nan")))
        from IPython.display import clear_output
        import matplotlib.pyplot as plt
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(6, 3))
        for k, v in self.history.items():
            ax.plot(v, label=k)
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
