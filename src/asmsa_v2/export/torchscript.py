"""TorchScript export of the trained collective variable.

The whole point of ASMSA is to hand the learned latent coordinates to PLUMED
as collective variables, which means exporting a single self-contained
TorchScript module that maps **atom positions** to **CV values**. That module
has to contain the entire chain:

    positions -> internal coordinates -> standardization -> encoder -> z

Dropping any link breaks the simulation silently. In particular the
standardization must travel *inside* the exported model, because PLUMED only
ever hands it raw coordinates and has no idea the features were scaled during
training.

Two variants are produced:

:class:`BatchCVModel`
    ``[n_frames, n_atoms, 3] -> [n_frames, latent_dim]``, for analysing a
    trajectory in Python.
:class:`PlumedCVModel`
    ``[1, n_atoms * 3] -> [1, latent_dim]``, PLUMED's single-configuration
    signature passed through POSITION components to ``PYTORCH_MODEL``.

Everything is traced in ``eval()`` mode. This is not cosmetic: the encoder
contains BatchNorm, and tracing it in training mode would bake in
batch-statistics normalization, which for PLUMED's batch of one gives a
degenerate (zero-variance) output and a meaningless CV.
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..data.normalize import FeatureScaler, StandardizeModule
from ..featurize.mol_model import MoleculeModel
from ..models.aae import AAE


class SingleMolEncoder(nn.Module):
    """Encoder specialized to one ``mol_id``, with the string argument removed.

    The multi-molecule encoder takes ``(x, mol_id)``; a traced module cannot
    take a string, so the relevant per-molecule adapter is bound here.
    """

    def __init__(self, model: AAE, mol_id: str):
        super().__init__()
        if mol_id not in model.encoder.adapter_in.adapters:
            raise KeyError(
                f"unknown mol_id {mol_id!r}; model knows "
                f"{list(model.encoder.adapter_in.adapters)}"
            )
        self.adapter = model.encoder.adapter_in.adapters[mol_id]
        self.trunk = model.encoder.trunk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(self.adapter(x))


class SingleMolDecoder(nn.Module):
    """Decoder specialized to one ``mol_id``: ``[n, latent] -> [n, n_features]``."""

    def __init__(self, model: AAE, mol_id: str):
        super().__init__()
        if mol_id not in model.decoder.adapter_out.adapters:
            raise KeyError(
                f"unknown mol_id {mol_id!r}; model knows "
                f"{list(model.decoder.adapter_out.adapters)}"
            )
        self.trunk = model.decoder.trunk
        self.adapter = model.decoder.adapter_out.adapters[mol_id]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.trunk(z))


class BatchCVModel(nn.Module):
    """``[n_frames, n_atoms, 3] -> [n_frames, latent_dim]``."""

    def __init__(
        self,
        mol_model: MoleculeModel,
        standardize: StandardizeModule,
        encoder: nn.Module,
    ):
        super().__init__()
        self.mol_model = mol_model
        self.standardize = standardize
        self.encoder = encoder

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # mol_model wants [n_atoms, 3, n_frames]
        feats = self.mol_model(xyz.permute(1, 2, 0))  # [n_features, n_frames]
        feats = feats.transpose(0, 1)  # [n_frames, n_features]
        return self.encoder(self.standardize(feats))


class PlumedCVModel(nn.Module):
    """``[1, n_atoms * 3] -> [1, latent_dim]``, PLUMED's per-step signature.

    PLUMED passes the x/y/z components of mapped atoms as scalar arguments in
    nanometres, matching GROMACS' and mdtraj's units, so no unit conversion is
    applied.
    """

    def __init__(
        self,
        mol_model: MoleculeModel,
        standardize: StandardizeModule,
        encoder: nn.Module,
    ):
        super().__init__()
        self.mol_model = mol_model
        self.standardize = standardize
        self.encoder = encoder

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        xyz = positions.reshape(-1, 3, 1)  # [n_atoms, 3, 1]
        feats = self.mol_model(xyz)  # [n_features, 1]
        feats = feats.transpose(0, 1)  # [1, n_features]
        z = self.encoder(self.standardize(feats))  # [1, latent_dim]
        return z


def _freeze_for_export(*modules: nn.Module) -> None:
    """Eval mode, no grads, on CPU.

    CPU is not a convenience: PLUMED loads the TorchScript model inside the
    GROMACS process and runs it on the host, so a model whose weights were
    saved on CUDA or MPS would fail to load there. Exporting from CPU also
    makes the artifact portable between machines.
    """
    for m in modules:
        m.eval()
        m.to("cpu")
        for p in m.parameters():
            p.requires_grad_(False)


def build_cv_models(
    model: AAE,
    mol_model: MoleculeModel,
    scaler: FeatureScaler,
    mol_id: str = "complex",
) -> tuple[BatchCVModel, PlumedCVModel]:
    """Assemble the two un-traced CV modules (eval mode, grads off, CPU).

    The model and featurizer are deep-copied, so exporting mid-training
    neither moves the live model off its accelerator nor flips it out of
    training mode.
    """
    n_feat_model = model.feature_dims.get(mol_id)
    if n_feat_model is not None and scaler.n_features != int(n_feat_model):
        raise ValueError(
            f"scaler was fitted on {scaler.n_features} features but the model's "
            f"{mol_id!r} adapter expects {int(n_feat_model)}. The scaler and the "
            "model must come from the same prepare/train run."
        )
    model = copy.deepcopy(model)
    mol_model = copy.deepcopy(mol_model)
    standardize = scaler.as_module()
    enc = SingleMolEncoder(model, mol_id)
    batch = BatchCVModel(mol_model, standardize, enc)
    plumed = PlumedCVModel(mol_model, standardize, enc)
    _freeze_for_export(batch, plumed, mol_model, standardize, enc)
    return batch, plumed


def _trace_and_freeze(module: nn.Module, example) -> torch.jit.ScriptModule:
    """Trace to TorchScript and freeze, in eval mode.

    TorchScript specifically, not ``torch.compile``: PLUMED's pytorch module
    loads the file with the C++ ``torch::jit::load``, which only understands a
    serialized ScriptModule. Recent PyTorch marks ``torch.jit.*`` as
    deprecated in favour of ``torch.compile``/``torch.export``, but that path
    produces artifacts PLUMED cannot read, so the warning is suppressed
    deliberately rather than worked around.

    Freezing inlines the eval-mode BatchNorm statistics as constants, so the
    saved model cannot be coaxed into batch-statistics behaviour no matter how
    the consumer loads it. It is best-effort: if a future release removes it,
    the traced module is still correct because it was traced in eval mode.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        traced = torch.jit.trace(module, example)
        try:
            traced = torch.jit.freeze(traced)
        except Exception:
            # Best-effort (see docstring): the eval-mode trace is already correct.
            pass
    return traced


def export_cv_models(
    model: AAE,
    mol_model: MoleculeModel,
    scaler: FeatureScaler,
    n_atoms: int,
    out_dir: str | Path,
    mol_id: str = "complex",
    name: str = "cv",
    example_xyz: np.ndarray | torch.Tensor | None = None,
    check_tolerance: float = 1e-4,
) -> dict[str, Path]:
    """Trace and save the batch and PLUMED CV models, then verify them.

    Parameters
    ----------
    model : AAE
        Trained model (any mode; it is switched to eval for the export).
    mol_model : MoleculeModel
        The *same* feature model used to produce the training features.
    scaler : FeatureScaler
        The *same* scaler fitted on the training split.
    n_atoms : int
        Atom count of the system the CV is defined on.
    out_dir : path
    name : str
        File stem; files are ``{name}-batch.pt``, ``{name}-plumed.pt``,
        ``{name}-encoder.pt``, ``{name}-decoder.pt``.
    example_xyz : ``[n_frames, n_atoms, 3]``, optional
        Real frames to verify against. Random coordinates are used otherwise,
        which still catches shape and mode errors but is a weaker check of
        numerical agreement, so passing real frames is recommended.
    check_tolerance : float
        Maximum allowed absolute difference between the traced and eager
        outputs. Raises if exceeded rather than saving a silently wrong model.

    Returns
    -------
    dict
        Mapping of artifact name to written path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_m, plumed_m = build_cv_models(model, mol_model, scaler, mol_id=mol_id)

    if example_xyz is None:
        ex = torch.randn(4, n_atoms, 3, dtype=torch.float32, device="cpu") * 0.5
    else:
        ex = torch.as_tensor(np.asarray(example_xyz), dtype=torch.float32).cpu()
        if ex.ndim != 3 or ex.shape[1] != n_atoms or ex.shape[2] != 3:
            raise ValueError(
                f"example_xyz must be [n_frames, {n_atoms}, 3], "
                f"got {tuple(ex.shape)}"
            )
        ex = ex[:8]

    paths: dict[str, Path] = {}
    with torch.no_grad():
        traced_batch = _trace_and_freeze(batch_m, ex)
        paths["batch"] = out_dir / f"{name}-batch.pt"
        traced_batch.save(str(paths["batch"]))

        flat = ex[0].reshape(1, -1).contiguous()
        traced_plumed = _trace_and_freeze(plumed_m, flat)
        paths["plumed"] = out_dir / f"{name}-plumed.pt"
        traced_plumed.save(str(paths["plumed"]))

        enc = SingleMolEncoder(copy.deepcopy(model), mol_id)
        _freeze_for_export(enc)
        n_feat = scaler.n_features
        traced_enc = _trace_and_freeze(enc, torch.randn(4, n_feat))
        paths["encoder"] = out_dir / f"{name}-encoder.pt"
        traced_enc.save(str(paths["encoder"]))

        dec = SingleMolDecoder(copy.deepcopy(model), mol_id)
        _freeze_for_export(dec)
        traced_dec = _trace_and_freeze(dec, torch.randn(4, model.latent_dim))
        paths["decoder"] = out_dir / f"{name}-decoder.pt"
        traced_dec.save(str(paths["decoder"]))

    report = verify_cv_models(
        paths["batch"], paths["plumed"], batch_m, ex, tolerance=check_tolerance
    )
    check_path = out_dir / f"{name}-export-check.txt"
    check_path.write_text(report)
    paths["check"] = check_path
    return paths


def verify_cv_models(
    batch_path: str | Path,
    plumed_path: str | Path,
    eager_batch: BatchCVModel,
    example_xyz: torch.Tensor,
    tolerance: float = 1e-4,
) -> str:
    """Check that the saved models reproduce the eager model and each other.

    Three things are checked, each of which has silently broken real
    simulations before:

    1. traced batch model == eager model (tracing captured the right graph),
    2. PLUMED model == batch model frame by frame (the reshape from a flat
       position vector matches the ``[n_atoms, 3, n_frames]`` layout),
    3. outputs are finite.

    Raises ``AssertionError`` if any check fails, so a broken export cannot
    reach a 500 ns simulation.
    """
    lines: list[str] = []
    with torch.no_grad():
        ref = eager_batch(example_xyz)

        tb = torch.jit.load(str(batch_path))
        tb.eval()
        got = tb(example_xyz)
        err_batch = float((got - ref).abs().max())
        lines.append(f"traced batch vs eager      : max |diff| = {err_batch:.3e}")

        tp = torch.jit.load(str(plumed_path))
        tp.eval()
        per_frame = torch.stack(
            [
                tp(example_xyz[i].reshape(1, -1).contiguous()).squeeze(0)
                for i in range(example_xyz.shape[0])
            ]
        )
        err_plumed = float((per_frame - ref).abs().max())
        lines.append(f"plumed (per frame) vs batch: max |diff| = {err_plumed:.3e}")
        lines.append(f"output shape (batch)       : {tuple(got.shape)}")
        lines.append(f"output shape (plumed)      : (1, {per_frame.shape[1]}) per frame")
        lines.append(f"finite outputs             : {bool(torch.isfinite(got).all())}")

    report = "\n".join(lines)
    if not torch.isfinite(got).all():
        raise AssertionError(
            "exported CV model produced non-finite output\n" + report
        )
    if err_batch > tolerance:
        raise AssertionError(
            f"traced batch model disagrees with the eager model by {err_batch:.3e} "
            f"(> {tolerance:.0e}).\n{report}"
        )
    if err_plumed > tolerance:
        raise AssertionError(
            f"PLUMED model disagrees with the batch model by {err_plumed:.3e} "
            f"(> {tolerance:.0e}). The two disagree only if the flat-position "
            "reshape does not match the batch layout, which would make the "
            f"simulation bias a different quantity than you analysed.\n{report}"
        )
    return report


def export_features_model(
    mol_model: MoleculeModel,
    n_atoms: int,
    path: str | Path,
) -> Path:
    """Trace and save just the featurizer (v1's ``features.pt``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mol_model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(mol_model, torch.randn(n_atoms, 3, 1))
    traced.save(str(path))
    return path
