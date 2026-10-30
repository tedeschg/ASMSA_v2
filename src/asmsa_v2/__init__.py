"""ASMSA v2 — PyTorch adversarial autoencoder for molecular dynamics.

Public API re-exported from submodules; see :mod:`asmsa_v2.models.aae.AAE` and
:mod:`asmsa_v2.train.trainer.Trainer` for the main entry points.
"""
from .models import AAE, Encoder, Decoder, Discriminator
from .priors import Prior, IsotropicNormalPrior, GMMPrior, gmm_on_circle, ImagePrior
from .data import MultiMolDataset, RoundRobinBatchSampler, multi_mol_collate, featurize_frames
from .train import Trainer, Callback, CsvLogger, EarlyStopping, LiveLossPlot
from .utils import ModelConfig, TrainConfig, set_seed

__version__ = "2.0.0a0"

__all__ = [
    "AAE", "Encoder", "Decoder", "Discriminator",
    "Prior", "IsotropicNormalPrior", "GMMPrior", "gmm_on_circle", "ImagePrior",
    "MultiMolDataset", "RoundRobinBatchSampler", "multi_mol_collate", "featurize_frames",
    "Trainer", "Callback", "CsvLogger", "EarlyStopping", "LiveLossPlot",
    "ModelConfig", "TrainConfig", "set_seed",
    "__version__",
]
