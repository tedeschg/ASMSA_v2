"""AAE trainer.

Design notes
------------
**Latent regularization is pluggable.** The classic AAE matches the aggregate
posterior to the prior with a discriminator. That is one option
(``latent_reg="adversarial"``); the Wasserstein-autoencoder penalties
(``"mmd"``, ``"sliced"``) reach the same goal without a minimax game and are
usually easier to train for the low-dimensional latents used as collective
variables. See :mod:`asmsa_v2.losses.mmd` and :mod:`asmsa_v2.losses.sliced`.

**One encoder forward per batch.** The previous implementation ran the
encoder three times per batch (autoencoder step, discriminator step, generator
step). Besides the wasted compute, each pass updated the encoder's
BatchNorm running statistics, so the buffers used at evaluation time were an
average over three differently-normalized views of the same batch. Here the
discriminator reuses the detached latent from the single forward pass, and in
the default ``adv_mode="joint"`` the reconstruction and adversarial terms
share one backward pass through the encoder — the Wasserstein-autoencoder
formulation of ``recon + lambda * latent_penalty``.

**Validation is first class.** A validation loader drives early stopping and
best-weight restoration. Monitoring a *training* loss, as before, cannot
detect overfitting, and with an adversarial objective the training
reconstruction loss is not even monotone.
"""
from __future__ import annotations

import copy
import math
import warnings
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch.utils.data import DataLoader

from ..losses.adversarial import bce_disc_loss, bce_gen_loss, r1_penalty
from ..losses.cv1_bias import cv1_bias_loss
from ..losses.mmd import mmd_loss
from ..losses.recon import RECON_LOSSES
from ..losses.sliced import sliced_wasserstein_loss
from ..models.aae import AAE
from ..priors.base import Prior
from ..utils.config import TrainConfig, resolve_device
from ..utils.seeding import set_seed
from .callbacks import Callback


class Trainer:
    """Train an :class:`~asmsa_v2.models.aae.AAE`.

    Parameters
    ----------
    model : AAE
    prior : Prior
        Target distribution for the latent space.
    config : TrainConfig
    callbacks : iterable of Callback
    """

    def __init__(
        self,
        model: AAE,
        prior: Prior,
        config: TrainConfig | None = None,
        callbacks: Iterable[Callback] = (),
    ):
        self.model = model
        self.prior = prior
        self.cfg = config or TrainConfig()
        self.callbacks = list(callbacks)

        if self.cfg.seed is not None:
            set_seed(self.cfg.seed)

        if prior.latent_dim != model.latent_dim:
            raise ValueError(
                f"prior latent_dim={prior.latent_dim} does not match model "
                f"latent_dim={model.latent_dim}"
            )

        self.device = resolve_device(self.cfg.device)
        self.model.to(self.device)
        self.prior.to(self.device)

        betas = tuple(self.cfg.adam_betas)
        wd = self.cfg.weight_decay
        opt_cls = torch.optim.AdamW if wd > 0 else torch.optim.Adam
        opt_kw = {"betas": betas}
        if wd > 0:
            opt_kw["weight_decay"] = wd

        ae_params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        self.opt_ae = opt_cls(ae_params, lr=self.cfg.lr_ae, **opt_kw)
        self.opt_d = (
            opt_cls(self.model.discriminator.parameters(), lr=self.cfg.lr_d, **opt_kw)
            if self.cfg.uses_discriminator else None
        )
        # Separate encoder-only optimizer, used by adv_mode="alternating" so
        # the adversarial step's Adam moments stay separate from the
        # reconstruction step's.
        self.opt_g = (
            opt_cls(self.model.encoder.parameters(), lr=self.cfg.lr_g, **opt_kw)
            if (self.cfg.uses_discriminator and self.cfg.adv_mode == "alternating")
            else None
        )

        self._schedulers = []
        if self.cfg.lr_schedule == "cosine":
            for opt in filter(None, (self.opt_ae, self.opt_d, self.opt_g)):
                self._schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=max(1, self.cfg.epochs)
                    )
                )

        self.recon_fn = RECON_LOSSES[self.cfg.recon_loss]
        self.stop = False
        self.history: dict[str, list[float]] = {}
        self.best_epoch: int | None = None
        self.best_score: float | None = None
        self._best_state: dict | None = None
        self.epoch = 0

    # ------------------------------------------------------------- internals
    def _latent_penalty(self, z: torch.Tensor) -> torch.Tensor:
        """Penalty pushing the aggregate posterior towards the prior."""
        reg = self.cfg.latent_reg
        if reg == "none":
            return z.new_zeros(())
        if reg == "adversarial":
            return bce_gen_loss(self.model.discriminate(z))
        z_prior = self.prior.sample(z.shape[0]).to(z.device)
        if reg == "mmd":
            return mmd_loss(z, z_prior)
        return sliced_wasserstein_loss(z, z_prior)

    def _disc_step(self, z_detached: torch.Tensor) -> float:
        """One or more discriminator updates against a fixed latent batch."""
        assert self.opt_d is not None
        last = 0.0
        for _ in range(self.cfg.d_steps):
            self.opt_d.zero_grad(set_to_none=True)
            z_real = self.prior.sample(z_detached.shape[0]).to(self.device)
            need_r1 = self.cfg.r1_gamma > 0
            if need_r1:
                z_real = z_real.detach().requires_grad_(True)
            logits_real = self.model.discriminate(z_real)
            logits_fake = self.model.discriminate(z_detached)
            loss_d = bce_disc_loss(logits_real, logits_fake, self.cfg.label_smoothing)
            if need_r1:
                loss_d = loss_d + self.cfg.r1_gamma * r1_penalty(logits_real, z_real)
            loss_d.backward()
            if self.cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.discriminator.parameters(), self.cfg.grad_clip
                )
            self.opt_d.step()
            last = float(loss_d.detach())
        return last

    def _set_disc_requires_grad(self, flag: bool):
        if self.cfg.uses_discriminator:
            for p in self.model.discriminator.parameters():
                p.requires_grad_(flag)

    def _recon_target(self, x: torch.Tensor, extras: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """What the decoder should reproduce: the input, or the lagged frame."""
        if not self.cfg.time_lagged:
            return x
        if "x_lag" not in extras:
            raise ValueError(
                "time_lagged=True but the batch carries no 'x_lag' target. "
                "Build the dataset with asmsa_v2.data.make_time_lagged_dataset()."
            )
        return extras["x_lag"].to(self.device, non_blocking=True)

    @staticmethod
    def _unpack(batch):
        """Accept ``(x, mol_id)`` or ``(x, mol_id, extras)``."""
        if len(batch) == 2:
            return batch[0], batch[1], {}
        return batch[0], batch[1], (batch[2] or {})

    def _train_one_batch(self, batch) -> dict[str, float]:
        x, mol_id, extras = self._unpack(batch)
        x = x.to(self.device, non_blocking=True)
        target = self._recon_target(x, extras)
        if x.shape[0] < 2 and self.cfg.latent_reg != "none":
            # BatchNorm and the two-sample latent penalties both need >1 row.
            return {}

        # --- single encoder forward -----------------------------------------
        z = self.model.encode(x, mol_id)

        # --- discriminator update, on the detached latent --------------------
        loss_d = 0.0
        if self.cfg.uses_discriminator:
            self._set_disc_requires_grad(True)
            loss_d = self._disc_step(z.detach())

        logs: dict[str, float] = {}

        if self.cfg.adv_mode == "alternating":
            # v1-style: reconstruction and adversarial encoder updates are
            # separate optimizer steps with separate Adam state.
            self.opt_ae.zero_grad(set_to_none=True)
            recon = self.model.decode(z, mol_id)
            loss_recon = self.recon_fn(recon, target)
            loss_ae = loss_recon
            if self.cfg.with_cv1_bias and "cv1" in extras:
                loss_ae = loss_ae + self.cfg.lam_cv1 * cv1_bias_loss(
                    z, extras["cv1"].to(self.device)
                )
            loss_ae.backward()
            self._clip(self.model.encoder, self.model.decoder)
            self.opt_ae.step()

            assert self.opt_g is not None
            self._set_disc_requires_grad(False)
            self.opt_g.zero_grad(set_to_none=True)
            z_g = self.model.encode(x, mol_id)
            loss_lat = bce_gen_loss(self.model.discriminate(z_g))
            loss_lat.backward()
            self._clip(self.model.encoder)
            self.opt_g.step()
            self._set_disc_requires_grad(True)
        else:
            # Joint (WAE-style): one backward pass for recon + latent penalty.
            self._set_disc_requires_grad(False)
            self.opt_ae.zero_grad(set_to_none=True)
            recon = self.model.decode(z, mol_id)
            loss_recon = self.recon_fn(recon, target)
            loss_lat = self._latent_penalty(z)
            loss_ae = loss_recon + self.cfg.lam_latent * loss_lat
            if self.cfg.with_cv1_bias and "cv1" in extras:
                loss_cv1 = cv1_bias_loss(z, extras["cv1"].to(self.device))
                loss_ae = loss_ae + self.cfg.lam_cv1 * loss_cv1
                logs["loss_cv1"] = float(loss_cv1.detach())
            loss_ae.backward()
            self._clip(self.model.encoder, self.model.decoder)
            self.opt_ae.step()
            self._set_disc_requires_grad(True)

        logs.update({
            "loss_recon": float(loss_recon.detach()),
            "loss_latent": float(loss_lat.detach()),
            "loss_d": float(loss_d),
        })
        return logs

    def _clip(self, *modules):
        if not self.cfg.grad_clip:
            return
        params = [p for m in modules for p in m.parameters()]
        torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)

    # ------------------------------------------------------------ validation
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Validation losses over ``loader``, in eval mode, with no updates.

        Averages are weighted by the number of frames in each batch, so a
        final partial batch does not count as much as a full one. Build the
        loader with :meth:`MultiMolDataset.eval_loader` — a shuffling loader
        makes this metric depend on the batch partition and therefore
        non-reproducible, which is warned about once.
        """
        self._warn_if_val_loader_shuffles(loader)
        was_training = self.model.training
        self.model.eval()
        try:
            totals = {"val_loss_recon": 0.0, "val_loss_latent": 0.0}
            n_frames = 0
            for batch in loader:
                x, mol_id, extras = self._unpack(batch)
                x = x.to(self.device, non_blocking=True)
                target = self._recon_target(x, extras)
                z = self.model.encode(x, mol_id)
                recon = self.model.decode(z, mol_id)
                bs = int(x.shape[0])
                totals["val_loss_recon"] += float(self.recon_fn(recon, target)) * bs
                if self.cfg.latent_reg == "adversarial":
                    lat = bce_gen_loss(self.model.discriminate(z))
                elif self.cfg.latent_reg == "mmd":
                    lat = mmd_loss(z, self.prior.sample(z.shape[0]).to(self.device))
                elif self.cfg.latent_reg == "sliced":
                    lat = sliced_wasserstein_loss(
                        z, self.prior.sample(z.shape[0]).to(self.device)
                    )
                else:
                    lat = torch.zeros((), device=self.device)
                totals["val_loss_latent"] += float(lat) * bs
                n_frames += bs
            n_frames = max(n_frames, 1)
            return {k: v / n_frames for k, v in totals.items()}
        finally:
            if was_training:
                self.model.train()

    def _warn_if_val_loader_shuffles(self, loader: DataLoader):
        if getattr(self, "_val_shuffle_warned", False):
            return
        sampler = getattr(loader, "batch_sampler", None)
        if getattr(sampler, "shuffle", False) or getattr(sampler, "drop_last", False):
            self._val_shuffle_warned = True
            warnings.warn(
                "the validation loader shuffles and/or drops the last batch, so "
                "the validation metric is not reproducible between passes and "
                "some frames may be ignored. Build it with "
                "dataset.eval_loader(batch_size=...) instead.",
                stacklevel=3,
            )

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, list[float]]:
        """Train for ``cfg.epochs``, returning the metric history.

        ``val_loader`` is optional but strongly recommended: without it there
        is nothing to early-stop on and ``restore_best`` has no signal, so
        both are disabled and a warning is issued.
        """
        self._warn_batch_size_mismatch(loader)
        if val_loader is None and self.cfg.restore_best:
            warnings.warn(
                "no val_loader given: best-weight restoration and any "
                "EarlyStopping on a 'val_*' metric are inactive. Split your "
                "trajectory with asmsa_v2.data.split_frames() to get one.",
                stacklevel=2,
            )

        self.model.train()
        self.stop = False
        for cb in self.callbacks:
            cb.on_train_begin(self)

        step = 0
        for epoch in range(self.cfg.epochs):
            self.epoch = epoch
            if self.stop:
                break
            for cb in self.callbacks:
                cb.on_epoch_begin(self, epoch)

            sums: dict[str, float] = {}
            n_batches = 0
            iterator = loader
            if self.cfg.progress:
                from tqdm.auto import tqdm
                iterator = tqdm(
                    loader, desc=f"epoch {epoch + 1}/{self.cfg.epochs}",
                    leave=False, unit="batch",
                )

            for batch in iterator:
                batch_logs = self._train_one_batch(batch)
                if not batch_logs:
                    continue
                step += 1
                n_batches += 1
                for k, v in batch_logs.items():
                    sums[k] = sums.get(k, 0.0) + v
                for cb in self.callbacks:
                    cb.on_batch_end(self, step, batch_logs)
                if self.cfg.progress and step % max(1, self.cfg.log_every) == 0:
                    iterator.set_postfix({k: f"{v:.4f}" for k, v in batch_logs.items()})

            if n_batches == 0:
                raise RuntimeError(
                    "the training loader produced no usable batches; check that "
                    "batch_size is not larger than the number of frames"
                )
            logs = {k: v / n_batches for k, v in sums.items()}
            logs["lr_ae"] = float(self.opt_ae.param_groups[0]["lr"])

            if val_loader is not None:
                logs.update(self.evaluate(val_loader))

            for sch in self._schedulers:
                sch.step()
            self._apply_warmup(epoch)

            # Callbacks may contribute extra metrics (e.g. the prior-match KL
            # that v1 selected models on) before selection and history, so
            # such a metric can be used as cfg.monitor.
            for cb in self.callbacks:
                cb.on_epoch_metrics(self, epoch, logs)

            self._track_best(epoch, logs)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs)

        if self.cfg.restore_best and self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        for cb in self.callbacks:
            cb.on_train_end(self)
        self.model.eval()
        return self.history

    def _apply_warmup(self, epoch: int):
        if self.cfg.warmup_epochs <= 0 or epoch >= self.cfg.warmup_epochs:
            return
        scale = (epoch + 1) / self.cfg.warmup_epochs
        for opt, base in ((self.opt_ae, self.cfg.lr_ae),
                          (self.opt_d, self.cfg.lr_d),
                          (self.opt_g, self.cfg.lr_g)):
            if opt is None:
                continue
            for g in opt.param_groups:
                g["lr"] = base * scale

    def _track_best(self, epoch: int, logs: dict[str, float]):
        key = self.cfg.monitor
        if key not in logs:
            return
        v = logs[key]
        if not math.isfinite(v):
            return
        better = (
            self.best_score is None
            or (v < self.best_score if self.cfg.monitor_mode == "min" else v > self.best_score)
        )
        if better:
            self.best_score = v
            self.best_epoch = epoch
            if self.cfg.restore_best:
                self._best_state = copy.deepcopy(
                    {k: t.detach().cpu().clone() for k, t in self.model.state_dict().items()}
                )

    def _warn_batch_size_mismatch(self, loader: DataLoader):
        bs = getattr(getattr(loader, "batch_sampler", None), "batch_size", None)
        if bs is None:
            bs = getattr(loader, "batch_size", None)
        if bs is not None and int(bs) != int(self.cfg.batch_size):
            warnings.warn(
                f"TrainConfig.batch_size={self.cfg.batch_size} but the loader "
                f"uses {int(bs)}; the loader wins. Build the loader with "
                f"dataset.loader(batch_size=cfg.batch_size) to keep them in sync.",
                stacklevel=3,
            )

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def encode_all(
        self,
        tensors_by_mol: Mapping[str, torch.Tensor],
        chunk_size: int = 8192,
    ) -> dict[str, torch.Tensor]:
        """Encode full per-molecule tensors in eval mode, chunked.

        Chunking keeps peak memory bounded on long trajectories. The model's
        original train/eval mode is restored on exit, so calling this does not
        silently flip a model you had put in ``eval()``.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            out: dict[str, torch.Tensor] = {}
            for mol_id, t in tensors_by_mol.items():
                t = torch.as_tensor(t, dtype=torch.float32)
                parts = []
                for start in range(0, t.shape[0], max(1, chunk_size)):
                    chunk = t[start:start + chunk_size].to(self.device)
                    parts.append(self.model.encode(chunk, mol_id).cpu())
                out[mol_id] = torch.cat(parts, dim=0) if parts else torch.empty(0)
            return out
        finally:
            if was_training:
                self.model.train()

    @torch.no_grad()
    def reconstruct_all(
        self,
        tensors_by_mol: Mapping[str, torch.Tensor],
        chunk_size: int = 8192,
    ) -> dict[str, torch.Tensor]:
        """Encode then decode full per-molecule tensors, chunked, in eval mode."""
        was_training = self.model.training
        self.model.eval()
        try:
            out: dict[str, torch.Tensor] = {}
            for mol_id, t in tensors_by_mol.items():
                t = torch.as_tensor(t, dtype=torch.float32)
                parts = []
                for start in range(0, t.shape[0], max(1, chunk_size)):
                    chunk = t[start:start + chunk_size].to(self.device)
                    z = self.model.encode(chunk, mol_id)
                    parts.append(self.model.decode(z, mol_id).cpu())
                out[mol_id] = torch.cat(parts, dim=0) if parts else torch.empty(0)
            return out
        finally:
            if was_training:
                self.model.train()

    # ---------------------------------------------------------- checkpointing
    def save_checkpoint(self, path: str | Path, **extra) -> Path:
        """Save model weights, optimizer state, config and history."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "feature_dims": dict(self.model.feature_dims),
            "latent_dim": self.model.latent_dim,
            "opt_ae": self.opt_ae.state_dict(),
            "opt_d": self.opt_d.state_dict() if self.opt_d else None,
            "opt_g": self.opt_g.state_dict() if self.opt_g else None,
            "config": self.cfg.to_dict(),
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "epoch": self.epoch,
            **extra,
        }
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> dict:
        """Restore from :meth:`save_checkpoint`. Returns the raw payload."""
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"], strict=strict)
        if payload.get("opt_ae"):
            self.opt_ae.load_state_dict(payload["opt_ae"])
        if self.opt_d is not None and payload.get("opt_d"):
            self.opt_d.load_state_dict(payload["opt_d"])
        if self.opt_g is not None and payload.get("opt_g"):
            self.opt_g.load_state_dict(payload["opt_g"])
        self.history = payload.get("history", {})
        self.best_epoch = payload.get("best_epoch")
        self.best_score = payload.get("best_score")
        self.epoch = payload.get("epoch", 0)
        return payload
