import numpy as np
import pytest
import torch

from asmsa_v2 import ComplexMolecule


def _complex_geoms(n_atoms=12, n_frames=4):
    rng = np.random.default_rng(0)
    coords = np.zeros((n_atoms, 3, n_frames), dtype=np.float32)
    coords[:8] = 0.3 * rng.standard_normal((8, 3, n_frames)).astype(np.float32)
    coords[8:] = np.array([5., 0., 0.])[None, :, None] + \
                 0.3 * rng.standard_normal((4, 3, n_frames)).astype(np.float32)
    return torch.tensor(coords)


def test_two_components_dense():
    protein_bonds = np.array([[0, 1], [1, 2]], dtype=np.int32)
    cm = ComplexMolecule(
        n_atoms=12,
        components={"protein": range(8), "ligand": range(8, 12)},
        bonds=protein_bonds,
        inter_mode="dense",
    )
    feats = cm.intcoord(_complex_geoms())
    # 2 bonds + 8*4 = 32 inter distances
    assert feats.shape == (2 + 32, 4)


def test_two_components_sparse():
    cm = ComplexMolecule(
        n_atoms=12,
        components={"protein": range(8), "ligand": range(8, 12)},
        bonds=np.array([[0, 1]], dtype=np.int32),
        inter_mode="sparse", inter_k=12, inter_seed=1,
    )
    feats = cm.intcoord(_complex_geoms())
    assert feats.shape == (1 + 12, 4)


def test_inter_subsets_restrict_pairs():
    cm = ComplexMolecule(
        n_atoms=12,
        components={"protein": range(8), "ligand": range(8, 12)},
        bonds=np.array([[0, 1]], dtype=np.int32),
        inter_subsets={"protein": [0, 1], "ligand": [8, 9, 10]},  # 2 × 3 = 6 pairs
        inter_mode="dense",
    )
    feats = cm.intcoord(_complex_geoms())
    assert feats.shape == (1 + 6, 4)


def test_three_components_generates_all_pairs():
    cm = ComplexMolecule(
        n_atoms=15,
        components={
            "A": range(5),
            "B": range(5, 10),
            "C": range(10, 15),
        },
        bonds=np.array([[0, 1]], dtype=np.int32),
        inter_mode="dense",
    )
    # 3 pairs of components × (5*5) = 75 inter distances
    assert sum(cm._n_inter_pairs) == 75


def test_overlapping_components_rejected():
    with pytest.raises(ValueError, match="overlaps"):
        ComplexMolecule(
            n_atoms=12,
            components={"a": range(0, 7), "b": range(6, 12)},  # overlap at 6
            bonds=np.array([[0, 1]], dtype=np.int32),
        )


def test_single_component_rejected():
    with pytest.raises(ValueError, match="at least two"):
        ComplexMolecule(
            n_atoms=12,
            components={"only": range(12)},
            bonds=np.array([[0, 1]], dtype=np.int32),
        )
