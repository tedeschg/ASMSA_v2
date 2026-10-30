import numpy as np
import pytest
import torch

from asmsa_v2 import InterComponentDistances
from asmsa_v2.featurize.mol_model import MoleculeModel, BondsModel


def _complex_geoms(n_atoms=12, n_frames=3, protein_center=(0., 0., 0.),
                   ligand_center=(5., 0., 0.)):
    rng = np.random.default_rng(0)
    coords = np.zeros((n_atoms, 3, n_frames), dtype=np.float32)
    # first 8 atoms = "protein" clustered near protein_center
    coords[:8] = (np.array(protein_center)[None, :, None]
                  + 0.3 * rng.standard_normal((8, 3, n_frames)).astype(np.float32))
    # last 4 = "ligand" clustered near ligand_center
    coords[8:] = (np.array(ligand_center)[None, :, None]
                  + 0.3 * rng.standard_normal((4, 3, n_frames)).astype(np.float32))
    return torch.tensor(coords)


def test_dense_pair_count():
    fm = InterComponentDistances(n_atoms=12, atoms_a=range(8), atoms_b=range(8, 12),
                                 mode="dense")
    assert fm.bonds.shape == (8 * 4, 2)
    g = _complex_geoms()
    d = fm(g)
    assert d.shape == (32, 3)
    # all distances should be positive
    assert torch.all(d > 0)


def test_sparse_pair_count():
    fm = InterComponentDistances(n_atoms=12, atoms_a=range(8), atoms_b=range(8, 12),
                                 mode="sparse", k=10, seed=42)
    assert fm.bonds.shape == (10, 2)
    g = _complex_geoms()
    d = fm(g)
    assert d.shape == (10, 3)


def test_sparse_unique_pairs():
    fm = InterComponentDistances(n_atoms=12, atoms_a=range(8), atoms_b=range(8, 12),
                                 mode="sparse", k=20, seed=42)
    pairs = {(int(a), int(b)) for a, b in fm.bonds}
    assert len(pairs) == 20


def test_overlap_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        InterComponentDistances(n_atoms=12, atoms_a=[1, 2, 3], atoms_b=[3, 4, 5])


def test_distances_reflect_component_separation():
    """Moving the ligand farther should increase every inter-component distance."""
    fm = InterComponentDistances(n_atoms=12, atoms_a=range(8), atoms_b=range(8, 12),
                                 mode="dense")
    close = fm(_complex_geoms(ligand_center=(5., 0., 0.)))
    far = fm(_complex_geoms(ligand_center=(50., 0., 0.)))
    assert torch.all(far.mean(dim=1) > close.mean(dim=1))


def test_integrates_into_moleculemodel():
    """InterComponentDistances must plug into MoleculeModel as a feature map."""
    protein_bonds = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    inter = InterComponentDistances(n_atoms=12, atoms_a=range(8), atoms_b=range(8, 12),
                                    mode="dense")
    mm = MoleculeModel(n_atoms=12, bonds=protein_bonds, feature_maps=[inter])
    g = _complex_geoms()
    out = mm(g)
    assert out.shape == (3 + 32, 3)
