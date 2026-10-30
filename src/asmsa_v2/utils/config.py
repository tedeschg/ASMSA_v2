from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class ModelConfig:
    latent_dim: int = 2
    encoder_widths: Sequence[int] = (128, 64, 32)
    decoder_widths: Sequence[int] = (32, 64, 128)
    disc_widths: Sequence[int] = (64, 64)
    activation: str = "gelu"
    batchnorm: bool = True
    spectral_norm_disc: bool = True


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr_ae: float = 2e-4
    lr_d: float = 1e-4
    lr_g: float = 2e-4
    adam_betas: tuple[float, float] = (0.5, 0.999)
    recon_loss: str = "mse"               # key into asmsa_v2.losses.recon.RECON_LOSSES
    label_smoothing: float = 0.0
    with_density: bool = False
    lam_density: float = 1.0
    with_cv1_bias: bool = False
    lam_cv1: float = 1.0
    log_every: int = 50
    device: str | None = None             # None => auto (cuda if available)
    seed: int | None = None
