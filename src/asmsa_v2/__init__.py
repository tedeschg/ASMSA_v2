"""ASMSA v2 — PyTorch adversarial autoencoder for molecular dynamics.

Analysis and Sampling of Molecular Simulations by adversarial Autoencoders:
learn a low-dimensional latent space from an MD trajectory, then use its
coordinates as collective variables to drive enhanced sampling in PLUMED.

The pipeline, mirroring the notebooks in ``notebooks/``:

1. **setup** — describe the system once, in a :class:`Project`.
2. **prepare** — featurize into internal coordinates, split, standardize.
3. **tune** — search hyperparameters (:func:`asmsa_v2.tune.tune`).
4. **train** — fit the autoencoder (:class:`Trainer`) and score it
   (:func:`asmsa_v2.metrics.evaluate_latent`).
5. **export** — trace the CV to TorchScript and write ``plumed.dat``
   (:mod:`asmsa_v2.export`).
6. **md** — run biased MD and compute free energies (:mod:`asmsa_v2.md`).
7. **analyse** — cluster the latent into states/poses
   (:func:`cluster_latent`).

See :mod:`asmsa_v2.models.aae.AAE` and
:mod:`asmsa_v2.train.trainer.Trainer` for the modelling entry points.
"""
__version__ = "2.0.0"

from .models import AAE, Encoder, Decoder, Discriminator
from .priors import Prior, IsotropicNormalPrior, GMMPrior, gmm_on_circle, ImagePrior
from .data import (
    MultiMolDataset, RoundRobinBatchSampler, multi_mol_collate,
    make_time_lagged_dataset, featurize_frames,
    FeatureScaler, StandardizeModule,
    SplitIndices, split_frames, decorrelate_by_rmsd, lag_pairs,
)
from .train import (
    Trainer, Callback, CsvLogger, EarlyStopping, ModelCheckpoint,
    LiveLossPlot, LatentSnapshot,
)
from .utils import (
    ModelConfig, TrainConfig, resolve_device, set_seed,
    Project, PreflightReport,
)
from .featurize import (
    Molecule, ComplexMolecule, MoleculeModel, InterComponentDistances,
    NBDistancesSparse, NBDistancesDense,
    auto_features, ca_atoms, heavy_atoms, contact_atoms, high_variance_pairs,
    extract_dihedral_indices_local, extract_atom_indices,
)
from .analysis import ClusterResult, cluster_latent, save_cluster_representatives
from .metrics import (
    QualityReport, evaluate_latent, effective_dimension,
    kl_divergence_knn, prior_match_score,
    vamp2_score, vamp2_scan, implied_timescales,
)

__all__ = [
    "__version__",
    # models
    "AAE", "Encoder", "Decoder", "Discriminator",
    # priors
    "Prior", "IsotropicNormalPrior", "GMMPrior", "gmm_on_circle", "ImagePrior",
    # data
    "MultiMolDataset", "RoundRobinBatchSampler", "multi_mol_collate",
    "make_time_lagged_dataset", "featurize_frames",
    "FeatureScaler", "StandardizeModule",
    "SplitIndices", "split_frames", "decorrelate_by_rmsd", "lag_pairs",
    # training
    "Trainer", "Callback", "CsvLogger", "EarlyStopping", "ModelCheckpoint",
    "LiveLossPlot", "LatentSnapshot",
    # config
    "ModelConfig", "TrainConfig", "resolve_device", "set_seed",
    "Project", "PreflightReport",
    # featurization
    "Molecule", "ComplexMolecule", "MoleculeModel", "InterComponentDistances",
    "NBDistancesSparse", "NBDistancesDense",
    "auto_features", "ca_atoms", "heavy_atoms", "contact_atoms",
    "high_variance_pairs",
    "extract_dihedral_indices_local", "extract_atom_indices",
    # analysis
    "ClusterResult", "cluster_latent", "save_cluster_representatives",
    # metrics
    "QualityReport", "evaluate_latent", "effective_dimension",
    "kl_divergence_knn", "prior_match_score",
    "vamp2_score", "vamp2_scan", "implied_timescales",
]
