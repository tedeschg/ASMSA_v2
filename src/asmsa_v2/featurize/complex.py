"""Thin convenience wrapper around :class:`Molecule` for protein-ligand
(or protein-protein) complexes.

The underlying :class:`Molecule` already handles arbitrary atom-indexed
systems — a complex is just one system with its two components concatenated
into a single PDB. ``ComplexMolecule`` adds the one piece that's missing in
the base class: automatic generation of **inter-component distance features**
that encode the binding pose.
"""
from typing import Mapping, Sequence

import numpy as np

from .inter_component import InterComponentDistances
from .molecule import Molecule


class ComplexMolecule(Molecule):
    """A :class:`Molecule` with inter-component distance feature maps attached.

    Parameters
    ----------
    components : mapping ``name -> atom-index sequence``
        Must contain at least two disjoint components (e.g. ``"protein"`` and
        ``"ligand"``). Atom indices refer to the single combined system.
    inter_mode : {"dense", "sparse"}
        How to populate inter-component distances. ``"dense"`` emits every
        cross-component pair (quadratic in subset size); ``"sparse"`` samples
        ``inter_k`` pairs per component pair.
    inter_subsets : optional mapping ``name -> atom-index sequence``
        Per-component subset to use *for the inter-component distances only*
        (intra features still see all component atoms via ``bonds``/``angles``
        /``diheds``). Defaults to the full ``components[name]``. Typical
        choices: CA atoms for a protein, heavy atoms for a ligand.
    inter_k : int, optional
        Number of pairs per component pair when ``inter_mode="sparse"``.
    inter_seed : int, optional
        RNG seed for sparse pair sampling.
    extra_fms : list, optional
        Additional feature maps appended after the inter-component ones.
    **molecule_kwargs
        Forwarded verbatim to :class:`Molecule` (``pdb``, ``n_atoms``,
        ``diheds``, ``bonds``, ``angles``, ``top``, ``ff``, ``ndx``).

    Notes
    -----
    If more than two components are given, inter-component distances are
    generated for every unordered pair.
    """

    def __init__(
        self,
        components: Mapping[str, Sequence[int]],
        inter_mode: str = "dense",
        inter_subsets: Mapping[str, Sequence[int]] | None = None,
        inter_k: int | None = None,
        inter_seed: int | None = 0,
        extra_fms: list | None = None,
        **molecule_kwargs,
    ):
        if len(components) < 2:
            raise ValueError("ComplexMolecule needs at least two components")

        comp_arrays = {name: np.asarray(list(idx), dtype=np.int64)
                       for name, idx in components.items()}
        seen: set[int] = set()
        for name, idx in comp_arrays.items():
            overlap = seen & set(idx.tolist())
            if overlap:
                raise ValueError(f"component {name!r} overlaps existing atoms: {sorted(overlap)[:5]}")
            seen.update(idx.tolist())

        n_atoms = molecule_kwargs.get("n_atoms")
        if n_atoms is None:
            n_atoms = int(max(a.max() for a in comp_arrays.values())) + 1
            molecule_kwargs["n_atoms"] = n_atoms

        subsets = {name: np.asarray(list((inter_subsets or {}).get(name, idx)), dtype=np.int64)
                   for name, idx in comp_arrays.items()}

        inter_fms = []
        names = list(comp_arrays.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                inter_fms.append(
                    InterComponentDistances(
                        n_atoms=n_atoms,
                        atoms_a=subsets[a], atoms_b=subsets[b],
                        mode=inter_mode, k=inter_k, seed=inter_seed,
                    )
                )

        fms = inter_fms + list(extra_fms or [])
        existing = molecule_kwargs.pop("fms", None)
        if existing:
            fms = fms + list(existing)

        super().__init__(fms=fms, **molecule_kwargs)
        self.components = comp_arrays
        self.inter_subsets = subsets
        self._n_inter_pairs = [fm.bonds.shape[0] for fm in inter_fms]
