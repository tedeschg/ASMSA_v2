"""Hyperparameter search space.

Expressed in the same terms v1 used — a *seed* width that is halved per layer,
plus a layer count — so hyperparameter ranges published for v1 transfer
directly, while also allowing the new v2 knobs (latent regularizer, its
weight, time lag) to be searched.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Sequence


@dataclass
class SearchSpace:
    """Candidate values per hyperparameter.

    A list means "search over these"; a bare scalar means "fix at this".
    ``(low, high)`` tuples given to ``lr_ae``/``lr_d``/``lam_latent`` are
    treated as continuous log-uniform ranges.
    """

    # architecture, v1-style (seed halved per layer)
    ae_neuron_seed: Sequence[int] | int = (32, 64, 96, 128)
    ae_layers: Sequence[int] | int = (2, 3, 4)
    disc_neuron_seed: Sequence[int] | int = (32, 64, 96, 128)
    disc_layers: Sequence[int] | int = (2, 3)
    activation: Sequence[str] | str = ("gelu", "silu")
    batchnorm: Sequence[bool] | bool = True
    latent_dim: Sequence[int] | int = 2

    # optimization
    lr_ae: Sequence[float] | float | tuple = (1e-4, 1e-3)
    lr_d: Sequence[float] | float | tuple = (5e-5, 5e-4)
    batch_size: Sequence[int] | int = (64, 128, 256)
    weight_decay: Sequence[float] | float = 0.0
    grad_clip: Sequence[float] | float | None = 5.0

    # objective
    latent_reg: Sequence[str] | str = ("adversarial", "mmd", "sliced")
    lam_latent: Sequence[float] | float | tuple = (0.1, 20.0)
    recon_loss: Sequence[str] | str = ("mse", "huber")
    label_smoothing: Sequence[float] | float = 0.0
    r1_gamma: Sequence[float] | float = 0.0

    _continuous: tuple[str, ...] = field(
        default=("lr_ae", "lr_d", "lam_latent"), repr=False
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_continuous", None)
        return {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in d.items()}

    def suggest(self, trial) -> dict[str, Any]:
        """Draw one hyperparameter dict from an Optuna trial."""
        out: dict[str, Any] = {}
        for name, spec in self.to_dict().items():
            if isinstance(spec, (list, tuple)):
                if name in self._continuous and len(spec) == 2 and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) for v in spec
                ):
                    lo, hi = float(spec[0]), float(spec[1])
                    out[name] = trial.suggest_float(name, lo, hi, log=lo > 0)
                else:
                    out[name] = trial.suggest_categorical(name, list(spec))
            else:
                out[name] = spec
        return out

    def sample_random(self, rng) -> dict[str, Any]:
        """Draw one hyperparameter dict without Optuna (fallback search)."""
        out: dict[str, Any] = {}
        for name, spec in self.to_dict().items():
            if isinstance(spec, (list, tuple)):
                if name in self._continuous and len(spec) == 2 and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) for v in spec
                ):
                    lo, hi = float(spec[0]), float(spec[1])
                    if lo > 0:
                        import math
                        out[name] = math.exp(rng.uniform(math.log(lo), math.log(hi)))
                    else:
                        out[name] = rng.uniform(lo, hi)
                else:
                    out[name] = list(spec)[rng.integers(len(spec))]
            else:
                out[name] = spec
        return out

    def size(self) -> int | float:
        """Number of discrete combinations, or ``inf`` if any axis is continuous."""
        total = 1
        for name, spec in self.to_dict().items():
            if isinstance(spec, (list, tuple)):
                if name in self._continuous and len(spec) == 2:
                    return float("inf")
                total *= max(1, len(spec))
        return total


# Ready-made spaces mirroring v1's small/medium presets.
SMALL_SPACE = SearchSpace(
    ae_neuron_seed=(32, 96),
    ae_layers=(2, 3),
    disc_neuron_seed=(32, 96),
    disc_layers=(2, 3),
    activation="gelu",
    batch_size=(64, 128),
    lr_ae=2e-4,
    lr_d=1e-4,
    latent_reg=("adversarial", "mmd"),
    lam_latent=(0.5, 10.0),
    recon_loss="mse",
)

MEDIUM_SPACE = SearchSpace(
    ae_neuron_seed=(32, 64, 96, 128),
    ae_layers=(2, 3, 4),
    disc_neuron_seed=(32, 64, 96, 128),
    disc_layers=(2, 3),
    activation=("gelu", "silu"),
    batch_size=(64, 128, 256),
    lr_ae=(1e-4, 1e-3),
    lr_d=(5e-5, 5e-4),
    latent_reg=("adversarial", "mmd", "sliced"),
    lam_latent=(0.1, 20.0),
    recon_loss=("mse", "huber"),
)


def params_to_configs(params: dict[str, Any], epochs: int, device: str | None = None,
                      seed: int | None = None, **train_overrides):
    """Turn a sampled hyperparameter dict into ``(ModelConfig, TrainConfig)``."""
    from ..utils.config import ModelConfig, TrainConfig

    model_cfg = ModelConfig.from_seed(
        latent_dim=int(params.get("latent_dim", 2)),
        ae_seed=int(params["ae_neuron_seed"]),
        ae_layers=int(params["ae_layers"]),
        disc_seed=int(params["disc_neuron_seed"]),
        disc_layers=int(params["disc_layers"]),
        activation=str(params.get("activation", "gelu")),
        batchnorm=bool(params.get("batchnorm", True)),
    )
    train_cfg = TrainConfig(
        epochs=epochs,
        batch_size=int(params.get("batch_size", 128)),
        lr_ae=float(params.get("lr_ae", 2e-4)),
        lr_d=float(params.get("lr_d", 1e-4)),
        lr_g=float(params.get("lr_ae", 2e-4)),
        weight_decay=float(params.get("weight_decay", 0.0)),
        grad_clip=params.get("grad_clip", 5.0),
        recon_loss=str(params.get("recon_loss", "mse")),
        latent_reg=str(params.get("latent_reg", "adversarial")),
        lam_latent=float(params.get("lam_latent", 1.0)),
        label_smoothing=float(params.get("label_smoothing", 0.0)),
        r1_gamma=float(params.get("r1_gamma", 0.0)),
        device=device,
        seed=seed,
        progress=False,
        **train_overrides,
    )
    return model_cfg, train_cfg
