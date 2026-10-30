import numpy as np
import pytest

mdtraj = pytest.importorskip("mdtraj")

from asmsa_v2 import ca_atoms, heavy_atoms, contact_atoms, high_variance_pairs


def _make_toy_complex_traj(n_frames=20, seed=0):
    """2-residue 'protein' (N, CA, C) + 2-atom 'ligand' (H1, C1)."""
    import mdtraj as md

    top = md.Topology()
    chain_p = top.add_chain()
    for _ in range(2):
        res = top.add_residue("ALA", chain_p)
        top.add_atom("N", md.element.nitrogen, res)
        top.add_atom("CA", md.element.carbon, res)
        top.add_atom("C", md.element.carbon, res)
    chain_l = top.add_chain()
    res_l = top.add_residue("LIG", chain_l)
    top.add_atom("H1", md.element.hydrogen, res_l)
    top.add_atom("C1", md.element.carbon, res_l)

    rng = np.random.default_rng(seed)
    xyz = np.zeros((n_frames, top.n_atoms, 3), dtype=np.float32)
    # protein atoms 0..5 around origin
    for i in range(6):
        xyz[:, i] = 0.1 * rng.standard_normal((n_frames, 3)).astype(np.float32)
    # ligand atoms 6, 7 close to protein (binding pose)
    xyz[:, 6] = np.array([0.3, 0.0, 0.0], dtype=np.float32) + \
                0.02 * rng.standard_normal((n_frames, 3)).astype(np.float32)
    xyz[:, 7] = np.array([0.4, 0.0, 0.0], dtype=np.float32) + \
                0.02 * rng.standard_normal((n_frames, 3)).astype(np.float32)
    return md.Trajectory(xyz, top)


def test_ca_atoms():
    traj = _make_toy_complex_traj()
    cas = ca_atoms(traj, range(6))                # protein component
    # Our protein has 2 CAs at atom indices 1 and 4
    assert sorted(cas.tolist()) == [1, 4]


def test_heavy_atoms_excludes_hydrogen():
    traj = _make_toy_complex_traj()
    heavy = heavy_atoms(traj, [6, 7])             # ligand: H1 at 6, C1 at 7
    assert heavy.tolist() == [7]


def test_contact_atoms_picks_the_nearby_pair():
    traj = _make_toy_complex_traj()
    a_hit, b_hit = contact_atoms(
        traj, atoms_a=list(range(6)), atoms_b=[6, 7],
        cutoff_nm=0.5, min_fraction=0.5,
    )
    # ligand atoms are both within 0.5 nm of several protein atoms
    assert len(a_hit) > 0
    assert set(b_hit.tolist()).issubset({6, 7})


def test_high_variance_pairs_returns_k():
    traj = _make_toy_complex_traj(n_frames=30)
    pairs = high_variance_pairs(traj, list(range(6)), [6, 7], k=5)
    assert pairs.shape == (5, 2)
    # pairs mix the two components
    for a, b in pairs:
        assert a < 6 and b in (6, 7)
