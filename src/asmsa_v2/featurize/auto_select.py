"""Automatic selection of atom subsets for inter-component distance features.

Choosing which atoms contribute inter-component distances is a modeling
decision. Common sensible defaults:

- **CA × heavy atoms** for a protein-ligand complex (collapses backbone
  flexibility to one atom per residue; keeps all meaningful ligand motion).
- **Contact-based**: atoms whose minimum inter-component distance across the
  trajectory falls below a cutoff for some fraction of frames. Picks the
  atoms the binding interface actually touches.
- **Variance-based**: atom pairs whose distance varies most along the
  trajectory — captures the dynamically relevant inter-component geometry.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def ca_atoms(traj, component_atoms: Sequence[int]) -> np.ndarray:
    """Return the CA atoms that fall inside ``component_atoms``.

    ``traj`` is an mdtraj Trajectory. Falls back to the full component list
    if no CAs are present (e.g. for a ligand).
    """
    comp = np.asarray(list(component_atoms), dtype=np.int64)
    try:
        all_ca = traj.top.select("name CA")
    except Exception:
        return comp
    out = np.intersect1d(comp, all_ca)
    return out if out.size > 0 else comp


def heavy_atoms(traj, component_atoms: Sequence[int]) -> np.ndarray:
    """Return the heavy (non-H) atoms inside ``component_atoms``."""
    comp = np.asarray(list(component_atoms), dtype=np.int64)
    try:
        heavy = traj.top.select("not element H")
    except Exception:
        return comp
    out = np.intersect1d(comp, heavy)
    return out if out.size > 0 else comp


def contact_atoms(
    traj,
    atoms_a: Sequence[int],
    atoms_b: Sequence[int],
    cutoff_nm: float = 0.5,
    min_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the subset of ``atoms_a`` / ``atoms_b`` that participate in
    contacts (distance < ``cutoff_nm``) in at least ``min_fraction`` of frames.
    """
    import mdtraj as md
    a = np.asarray(list(atoms_a), dtype=np.int64)
    b = np.asarray(list(atoms_b), dtype=np.int64)
    pairs = np.stack(np.meshgrid(a, b, indexing="ij"), axis=-1).reshape(-1, 2)
    d = md.compute_distances(traj, pairs)  # [n_frames, n_pairs]
    close = (d < cutoff_nm).mean(axis=0)   # [n_pairs] fraction of frames close
    contacting = close >= min_fraction
    if not contacting.any():
        return a, b
    hit_a = np.unique(pairs[contacting, 0])
    hit_b = np.unique(pairs[contacting, 1])
    return hit_a, hit_b


def high_variance_pairs(
    traj,
    atoms_a: Sequence[int],
    atoms_b: Sequence[int],
    k: int = 100,
) -> np.ndarray:
    """Return the ``k`` atom-atom pairs between the two components whose
    distance has the highest variance across the trajectory. Each row is
    ``(index_a, index_b)``. Picks the most informative inter-features.
    """
    import mdtraj as md
    a = np.asarray(list(atoms_a), dtype=np.int64)
    b = np.asarray(list(atoms_b), dtype=np.int64)
    pairs = np.stack(np.meshgrid(a, b, indexing="ij"), axis=-1).reshape(-1, 2)
    d = md.compute_distances(traj, pairs)   # [n_frames, n_pairs]
    var = d.var(axis=0)
    k = min(k, var.size)
    top = np.argpartition(var, -k)[-k:]
    return pairs[top]


def auto_features(
    traj,
    components: Mapping[str, Sequence[int]],
    inter_strategy: str = "ca_heavy",
    contact_cutoff_nm: float = 0.5,
    contact_min_fraction: float = 0.1,
    diheds: Sequence | None = None,
    extract_dihedrals: bool = False,
    pdb_for_dihedrals: str | None = None,
    **complex_kwargs,
):
    """One-liner that returns a ready-to-use :class:`ComplexMolecule`.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        Used both for shape info and for picking inter-component atom subsets.
    components : mapping name -> atom-index sequence
        Must contain at least two entries (e.g. ``{"protein": ..., "ligand": ...}``).
    inter_strategy : one of
        - ``"ca_heavy"``: CA atoms (component A) × heavy atoms (component B)
          per unordered component pair. Good default for protein-ligand.
        - ``"contact"``: only atoms that make contact (< ``contact_cutoff_nm``
          in ≥ ``contact_min_fraction`` of frames).
        - ``"all"``: every atom of each component (quadratic cost).
    diheds : optional sequence
        Dihedral quadruples passed through to :class:`ComplexMolecule`.
    extract_dihedrals : bool
        If True, auto-extract backbone + sidechain dihedrals from
        ``pdb_for_dihedrals`` using
        :func:`asmsa_v2.featurize.helpers.extract_dihedral_indices_local`.
    pdb_for_dihedrals : str
        PDB path for the auto-dihedral extraction (needs the whole-complex PDB).
    **complex_kwargs
        Forwarded to :class:`ComplexMolecule`.
    """
    from .complex import ComplexMolecule
    from .helpers import extract_dihedral_indices_local

    if extract_dihedrals:
        if pdb_for_dihedrals is None:
            raise ValueError("extract_dihedrals=True requires pdb_for_dihedrals")
        diheds = extract_dihedral_indices_local(pdb_for_dihedrals)

    inter_subsets: dict[str, Sequence[int]] = {}
    names = list(components.keys())
    if inter_strategy == "all":
        inter_subsets = {n: list(components[n]) for n in names}
    elif inter_strategy == "ca_heavy":
        # Protein-side gets CAs; everything else gets heavy atoms.
        # Heuristic: pick the component with the most CA atoms as the protein.
        ca_counts = {n: len(ca_atoms(traj, components[n])) for n in names}
        protein_name = max(ca_counts, key=ca_counts.get) if max(ca_counts.values()) > 0 else None
        for n in names:
            if n == protein_name:
                inter_subsets[n] = ca_atoms(traj, components[n]).tolist()
            else:
                inter_subsets[n] = heavy_atoms(traj, components[n]).tolist()
    elif inter_strategy == "contact":
        if len(names) != 2:
            raise ValueError("'contact' strategy currently only supports exactly 2 components")
        a_name, b_name = names
        hit_a, hit_b = contact_atoms(
            traj, components[a_name], components[b_name],
            cutoff_nm=contact_cutoff_nm, min_fraction=contact_min_fraction,
        )
        inter_subsets[a_name] = hit_a.tolist()
        inter_subsets[b_name] = hit_b.tolist()
    else:
        raise ValueError(f"unknown inter_strategy {inter_strategy!r}; "
                         "use 'ca_heavy', 'contact', or 'all'")

    return ComplexMolecule(
        components=components,
        inter_subsets=inter_subsets,
        diheds=diheds,
        n_atoms=traj.n_atoms,
        **complex_kwargs,
    )
