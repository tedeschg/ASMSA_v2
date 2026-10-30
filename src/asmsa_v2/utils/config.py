from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Any, Sequence


@dataclass
class ModelConfig:
    """Architecture of the AAE."""

    latent_dim: int = 2
    encoder_widths: Sequence[int] = (128, 64, 32)
    decoder_widths: Sequence[int] = (32, 64, 128)
    disc_widths: Sequence[int] = (64, 64)
    activation: str = "gelu"
    batchnorm: bool = True
    spectral_norm_disc: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, (list, tuple)) else v)
                for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_seed(cls, latent_dim: int = 2, ae_seed: int = 128, ae_layers: int = 3,
                  disc_seed: int = 64, disc_layers: int = 2, **kw) -> "ModelConfig":
        """Build widths the way v1 did: halve a seed width per layer.

        v1 parameterized architectures as ``(seed, n_layers)`` and derived
        ``[seed, seed/2, seed/4, ...]``. Hyperparameter ranges published for v1
        are expressed in those terms, so this keeps them usable.
        """
        enc = [max(1, int(ae_seed / (2 ** i))) for i in range(max(1, ae_layers))]
        disc = [max(1, int(disc_seed / (2 ** i))) for i in range(max(1, disc_layers))]
        return cls(latent_dim=latent_dim, encoder_widths=tuple(enc),
                   decoder_widths=tuple(reversed(enc)), disc_widths=tuple(disc), **kw)


@dataclass
class TrainConfig:
    """Optimization settings.

    Notes
    -----
    ``batch_size`` is used by :meth:`MultiMolDataset.loader` and by the
    trainer when it has to build a loader itself. If you construct the
    ``DataLoader`` by hand, the loader's own batch size wins — the trainer
    warns when the two disagree rather than letting the mismatch pass
    silently as it did before.
    """

    # --- schedule -----------------------------------------------------------
    epochs: int = 100
    batch_size: int = 128

    # --- optimizers ---------------------------------------------------------
    lr_ae: float = 2e-4
    lr_d: float = 1e-4
    lr_g: float = 2e-4                    # only used when adv_mode="alternating"
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = (0.5, 0.999)
    grad_clip: float | None = 5.0
    lr_schedule: str = "none"             # "none" | "cosine"
    warmup_epochs: int = 0

    # --- objective ----------------------------------------------------------
    recon_loss: str = "mse"               # key into asmsa_v2.losses.recon.RECON_LOSSES
    latent_reg: str = "adversarial"       # "adversarial" | "mmd" | "sliced" | "none"
    lam_latent: float = 1.0               # weight of the latent penalty
    adv_mode: str = "joint"               # "joint" | "alternating"
    label_smoothing: float = 0.0
    r1_gamma: float = 0.0                 # >0 enables the R1 gradient penalty
    d_steps: int = 1                      # critic updates per generator update

    # --- optional auxiliary objectives --------------------------------------
    with_cv1_bias: bool = False
    lam_cv1: float = 1.0
    time_lagged: bool = False             # reconstruct extras["x_lag"] instead of x

    # --- validation / stopping ---------------------------------------------
    monitor: str = "val_loss_recon"
    monitor_mode: str = "min"
    restore_best: bool = True

    # --- bookkeeping --------------------------------------------------------
    log_every: int = 50
    progress: bool = True
    device: str | None = None             # None => auto (cuda / mps / cpu)
    seed: int | None = None

    def __post_init__(self):
        if self.latent_reg not in ("adversarial", "mmd", "sliced", "none"):
            raise ValueError(
                f"unknown latent_reg {self.latent_reg!r}; use 'adversarial', "
                "'mmd', 'sliced' or 'none'"
            )
        if self.adv_mode not in ("joint", "alternating"):
            raise ValueError(
                f"unknown adv_mode {self.adv_mode!r}; use 'joint' or 'alternating'"
            )
        if self.monitor_mode not in ("min", "max"):
            raise ValueError(f"monitor_mode must be 'min' or 'max', got {self.monitor_mode!r}")
        if self.lr_schedule not in ("none", "cosine"):
            raise ValueError(f"unknown lr_schedule {self.lr_schedule!r}; use 'none' or 'cosine'")
        if self.adv_mode == "alternating" and self.latent_reg != "adversarial":
            raise ValueError(
                f"adv_mode='alternating' only makes sense with "
                f"latent_reg='adversarial', got {self.latent_reg!r}; "
                "non-adversarial penalties have no separate generator step"
            )
        if self.d_steps < 1:
            raise ValueError(f"d_steps must be >= 1, got {self.d_steps}")

    @property
    def uses_discriminator(self) -> bool:
        return self.latent_reg == "adversarial"

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        if "adam_betas" in kw and isinstance(kw["adam_betas"], list):
            kw["adam_betas"] = tuple(kw["adam_betas"])
        return cls(**kw)


def resolve_device(device: str | None = None) -> "object":
    """Pick a torch device: explicit choice, else cuda, else mps, else cpu."""
    import torch

    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
