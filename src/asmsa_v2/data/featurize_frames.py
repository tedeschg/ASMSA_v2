"""Convenience: run a built :class:`MoleculeModel` over a trajectory of
atom coordinates and return a ``[n_frames, n_features]`` tensor ready for the
AAE.
"""
import numpy as np
import torch

from ..featurize.mol_model import MoleculeModel


def featurize_frames(geoms: np.ndarray | torch.Tensor, mol_model: MoleculeModel) -> torch.Tensor:
    """``geoms`` has shape ``[n_frames, n_atoms, 3]`` (as returned by mdtraj's
    ``.xyz``). Returns ``[n_frames, n_features]`` on CPU.
    """
    g = torch.as_tensor(np.asarray(geoms), dtype=torch.float32)
    if g.ndim != 3 or g.shape[2] != 3:
        raise ValueError(f"expected geoms of shape [n_frames, n_atoms, 3], got {tuple(g.shape)}")
    # mol_model expects [n_atoms, 3, n_conformations]
    g = g.permute(1, 2, 0).contiguous()
    with torch.no_grad():
        feats = mol_model(g)  # [n_features, n_frames]
    return feats.T.contiguous()
