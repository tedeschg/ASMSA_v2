"""AAE trainer: three optimizers (AE / discriminator / encoder-adversarial)
so Adam moments stay cleanly separated across the three objectives.
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..losses.adversarial import bce_disc_loss, bce_gen_loss
from ..losses.cv1_bias import cv1_bias_loss
from ..losses.density import density_matching_loss
from ..losses.recon import RECON_LOSSES
from ..models.aae import AAE
from ..priors.base import Prior
from ..utils.config import TrainConfig
from ..utils.seeding import set_seed
from .callbacks import Callback


class Trainer:
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

        self.device = torch.device(self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.prior.to(self.device)

        betas = tuple(self.cfg.adam_betas)
        self.opt_ae = torch.optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
            lr=self.cfg.lr_ae, betas=betas,
        )
        self.opt_d = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.cfg.lr_d, betas=betas,
        )
        self.opt_g = torch.optim.Adam(
            self.model.encoder.parameters(),
            lr=self.cfg.lr_g, betas=betas,
        )

        self.recon_fn = RECON_LOSSES[self.cfg.recon_loss]
        self.stop = False

    def _batch_targets(self, batch_raw, key):
        """Pull optional per-batch targets (density, cv1) off a tuple-formed batch."""
        if isinstance(batch_raw, (tuple, list)) and len(batch_raw) >= 3:
            # (x, mol_id, extras_dict)
            extras = batch_raw[2]
            if isinstance(extras, dict) and key in extras:
                return extras[key].to(self.device)
        return None

    def fit(self, loader: DataLoader) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {"loss_recon": [], "loss_d": [], "loss_g": []}

        for cb in self.callbacks:
            cb.on_train_begin(self)

        step = 0
        for epoch in range(self.cfg.epochs):
            if self.stop:
                break
            for cb in self.callbacks:
                cb.on_epoch_begin(self, epoch)

            epoch_losses = {"loss_recon": 0.0, "loss_d": 0.0, "loss_g": 0.0}
            n_batches = 0

            pbar = tqdm(loader, desc=f"epoch {epoch+1}/{self.cfg.epochs}", leave=False)
            for batch_raw in pbar:
                x, mol_id = batch_raw[0], batch_raw[1]
                x = x.to(self.device, non_blocking=True)

                # 1. AE step (recon + optional density / cv1 bias)
                self.opt_ae.zero_grad(set_to_none=True)
                z = self.model.encode(x, mol_id)
                recon = self.model.decode(z, mol_id)
                loss_recon = self.recon_fn(recon, x)
                loss_ae = loss_recon
                if self.cfg.with_density:
                    tgt = self._batch_targets(batch_raw, "density")
                    if tgt is not None:
                        loss_ae = loss_ae + self.cfg.lam_density * density_matching_loss(z, tgt, self.prior)
                if self.cfg.with_cv1_bias:
                    tgt = self._batch_targets(batch_raw, "cv1")
                    if tgt is not None:
                        loss_ae = loss_ae + self.cfg.lam_cv1 * cv1_bias_loss(z, tgt)
                loss_ae.backward()
                self.opt_ae.step()

                # 2. Discriminator step
                self.opt_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    z_fake = self.model.encode(x, mol_id)
                z_real = self.prior.sample(x.shape[0]).to(self.device)
                logits_real = self.model.discriminate(z_real)
                logits_fake = self.model.discriminate(z_fake)
                loss_d = bce_disc_loss(logits_real, logits_fake, self.cfg.label_smoothing)
                loss_d.backward()
                self.opt_d.step()

                # 3. Encoder adversarial step
                self.opt_g.zero_grad(set_to_none=True)
                z_g = self.model.encode(x, mol_id)
                logits_g = self.model.discriminate(z_g)
                loss_g = bce_gen_loss(logits_g)
                loss_g.backward()
                self.opt_g.step()

                step += 1
                epoch_losses["loss_recon"] += float(loss_recon.detach())
                epoch_losses["loss_d"] += float(loss_d.detach())
                epoch_losses["loss_g"] += float(loss_g.detach())
                n_batches += 1

                batch_logs = {"loss_recon": float(loss_recon.detach()),
                              "loss_d": float(loss_d.detach()),
                              "loss_g": float(loss_g.detach())}
                for cb in self.callbacks:
                    cb.on_batch_end(self, step, batch_logs)
                if step % max(1, self.cfg.log_every) == 0:
                    pbar.set_postfix({k: f"{v:.4f}" for k, v in batch_logs.items()})

            n_batches = max(n_batches, 1)
            logs = {k: v / n_batches for k, v in epoch_losses.items()}
            for k, v in logs.items():
                history[k].append(v)
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs)

        for cb in self.callbacks:
            cb.on_train_end(self)
        return history

    @torch.no_grad()
    def encode_all(self, tensors_by_mol: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode a full per-molecule tensor through the model (eval mode)."""
        self.model.eval()
        out: dict[str, torch.Tensor] = {}
        for mol_id, t in tensors_by_mol.items():
            t = torch.as_tensor(t, dtype=torch.float32).to(self.device)
            out[mol_id] = self.model.encode(t, mol_id).cpu()
        self.model.train()
        return out
