"""Run a built :class:`MoleculeModel` over a trajectory and return a
``[n_frames, n_features]`` tensor ready for the AAE.

Featurization is chunked over frames. A dense inter-component feature map on
a real complex easily reaches 10^4 features, so a 10^5-frame trajectory is
~4 GB in float32 for the output alone and several times that for the
intermediate per-feature-map tensors. Chunking bounds the intermediates and
lets the result stream into a preallocated buffer.
"""
from __future__ import annotations

import numpy as np
import torch

from ..featurize.mol_model import MoleculeModel


def featurize_frames(
    geoms: np.ndarray | torch.Tensor,
    mol_model: MoleculeModel,
    chunk_size: int = 2048,
    device: str | torch.device | None = None,
    progress: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Featurize a trajectory.

    Parameters
    ----------
    geoms : array ``[n_frames, n_atoms, 3]``
        As returned by mdtraj's ``traj.xyz`` (nanometres).
    mol_model : MoleculeModel
        Built feature model, e.g. ``molecule.get_model()``.
    chunk_size : int
        Frames featurized per pass. Lower it if you hit memory pressure on a
        very wide feature set.
    device : str or torch.device, optional
        Device to run the featurization on. The result is always returned on
        CPU. A GPU gives a large speedup for dense distance maps.
    progress : bool
        Show a tqdm bar — worth enabling for long trajectories, since the
        featurization of a real complex can take minutes.

    Returns
    -------
    torch.Tensor
        ``[n_frames, n_features]`` on CPU.
    """
    g_all = torch.as_tensor(np.asarray(geoms), dtype=dtype)
    if g_all.ndim != 3 or g_all.shape[2] != 3:
        raise ValueError(
            f"expected geoms of shape [n_frames, n_atoms, 3], got {tuple(g_all.shape)}"
        )
    n_frames, n_atoms = g_all.shape[0], g_all.shape[1]
    if n_frames == 0:
        raise ValueError("geoms contains no frames")
    expected = getattr(mol_model, "n_atoms", None)
    if expected is not None and int(expected) != n_atoms:
        raise ValueError(
            f"mol_model was built for {int(expected)} atoms but geoms has "
            f"{n_atoms}. The trajectory and the topology/PDB used to build "
            f"the Molecule must describe the same atom set, in the same order."
        )

    dev = torch.device(device) if device is not None else torch.device("cpu")
    mol_model = mol_model.to(dev)
    chunk_size = max(1, int(chunk_size))

    rng = range(0, n_frames, chunk_size)
    if progress:
        from tqdm.auto import tqdm
        rng = tqdm(rng, desc="featurize", unit="chunk",
                   total=(n_frames + chunk_size - 1) // chunk_size)

    out: torch.Tensor | None = None
    with torch.no_grad():
        for start in rng:
            stop = min(start + chunk_size, n_frames)
            # mol_model expects [n_atoms, 3, n_conformations]
            chunk = g_all[start:stop].permute(1, 2, 0).contiguous().to(dev)
            feats = mol_model(chunk)          # [n_features, n_chunk]
            feats = feats.T.contiguous().to("cpu")
            if out is None:
                out = torch.empty((n_frames, feats.shape[1]), dtype=dtype)
            out[start:stop] = feats

    assert out is not None
    if not torch.isfinite(out).all():
        n_bad = int((~torch.isfinite(out)).any(dim=1).sum())
        raise ValueError(
            f"featurization produced non-finite values in {n_bad} frame(s). "
            "Common causes: duplicate atom coordinates (zero-length bond or "
            "a degenerate angle/dihedral), or a trajectory/topology atom-order "
            "mismatch."
        )
    return out
