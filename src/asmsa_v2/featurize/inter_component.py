"""Inter-component distance features for molecular complexes.

For a protein-ligand (or protein-protein) complex, intra-component bonds /
angles / dihedrals describe each component's internal geometry, but the
binding pose is encoded by **relative** geometry between the two components.
``InterComponentDistances`` computes a chosen set of atom-atom distances
across the two components and plugs into :class:`MoleculeModel` as a
feature map.
"""
from typing import Sequence

import numpy as np

from .mol_model import BondsModel


class InterComponentDistances(BondsModel):
    """Distances between atoms in two disjoint components of the same system.

    Parameters
    ----------
    n_atoms : int
        Total number of atoms in the parent system (not just the two components).
    atoms_a, atoms_b : sequence of int
        Atom indices (into the parent system) belonging to each component.
    mode : {"dense", "sparse"}
        ``"dense"`` → every pair in ``atoms_a × atoms_b``.
        ``"sparse"`` → ``k`` random pairs (reproducible via ``seed``).
    k : int, optional
        Number of pairs when ``mode="sparse"``.
    seed : int, optional
        RNG seed for sparse pair sampling.
    """

    def __init__(
        self,
        n_atoms: int,
        atoms_a: Sequence[int],
        atoms_b: Sequence[int],
        mode: str = "dense",
        k: int | None = None,
        seed: int | None = 0,
    ):
        atoms_a = np.asarray(list(atoms_a), dtype=np.int64)
        atoms_b = np.asarray(list(atoms_b), dtype=np.int64)
        if atoms_a.size == 0 or atoms_b.size == 0:
            raise ValueError("both components must contain at least one atom")
        if set(atoms_a.tolist()) & set(atoms_b.tolist()):
            raise ValueError("component atom sets must be disjoint")

        if mode == "dense":
            aa, bb = np.meshgrid(atoms_a, atoms_b, indexing="ij")
            pairs = np.stack([aa.ravel(), bb.ravel()], axis=1)
        elif mode == "sparse":
            if k is None:
                raise ValueError("sparse mode requires k")
            rng = np.random.default_rng(seed)
            total = atoms_a.size * atoms_b.size
            k = int(min(k, total))
            flat = rng.choice(total, size=k, replace=False)
            i = flat // atoms_b.size
            j = flat % atoms_b.size
            pairs = np.stack([atoms_a[i], atoms_b[j]], axis=1)
        else:
            raise ValueError(f"unknown mode {mode!r}; use 'dense' or 'sparse'")

        super().__init__(n_atoms, pairs.astype(np.int32))
        self.atoms_a = atoms_a
        self.atoms_b = atoms_b
        self.mode = mode
