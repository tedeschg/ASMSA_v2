"""PLUMED input generation for metadynamics on the learned CVs.

The hard part is not the ``plumed.dat`` text, it is the **atom numbering**.

ASMSA trains on atoms numbered as in the input PDB. The simulation, however,
runs on a topology built by ``gmx pdb2gmx``, which adds hydrogens and
reorders atoms within residues (notably it moves the backbone carboxyl group
to the end of each residue). PLUMED's ``POSITION`` actions must therefore
enumerate atoms **in the order the featurizer expects** (PDB order) while
**referring to the indices the simulation uses** (topology order).

Get this wrong and nothing crashes: the simulation runs happily and biases a
scrambled, meaningless coordinate. v1 derived the mapping from the sort order
of a GROMACS index file and warned the user to "check the results manually".
Here the mapping is instead resolved by matching
``(chain, residue number, residue name, atom name)`` between the two files,
which is deterministic, reports exactly which atoms failed to match, and
refuses to emit input it cannot justify.

The standard PLUMED ``PYTORCH_MODEL`` action accepts scalar ``ARG`` inputs,
so mapped atom positions are passed as ordered x/y/z components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# GROMACS force fields rename protonation-state variants; map them back to the
# canonical PDB residue name so matching is not defeated by the rename.
_RESIDUE_ALIASES = {
    "HISE": "HIS",
    "HISD": "HIS",
    "HISH": "HIS",
    "HIS1": "HIS",
    "HIS2": "HIS",
    "HSE": "HIS",
    "HSD": "HIS",
    "HSP": "HIS",
    "LYSH": "LYS",
    "LSN": "LYS",
    "ASPH": "ASP",
    "ASH": "ASP",
    "GLUH": "GLU",
    "GLH": "GLU",
    "CYS2": "CYS",
    "CYX": "CYS",
    "CYM": "CYS",
    "ARGN": "ARG",
}

# Atom-name variants for the same physical atom.
_ATOM_ALIASES = {
    "OC1": "O",
    "OT1": "O",
    "O1": "O",
    "OC2": "OXT",
    "OT2": "OXT",
    "O2": "OXT",
    "CD": "CD1",  # ILE
    "OD": "OD1",
    "OE": "OE1",
}


def _canon_res(name: str) -> str:
    """Canonical (PDB) residue name for a possibly force-field-renamed one."""
    n = name.strip().upper()
    return _RESIDUE_ALIASES.get(n, n)


def _canon_atom(name: str) -> str:
    """Canonical atom name for a possibly renamed one."""
    n = name.strip().upper()
    return _ATOM_ALIASES.get(n, n)


def _atom_key(atom, use_chain: bool = True) -> tuple[int, int, str, str]:
    """Identity key ``(chain, residue number, residue name, atom name)``."""
    res = atom.residue
    chain = res.chain.index if use_chain else 0
    return (chain, res.resSeq, _canon_res(res.name), _canon_atom(atom.name))


def _format_floats(values) -> str:
    """Comma-separated, 4-decimal rendering of an array, as PLUMED expects."""
    return ",".join(f"{v:.4f}" for v in np.asarray(values).ravel())


def _position_inputs(atom_map: PlumedAtomMap, label: str) -> tuple[list[str], str]:
    position_lines = []
    arguments = []
    for i, atom_index in enumerate(atom_map.atoms_1based):
        position_label = f"{label}_position_{i}"
        position_lines.append(
            f"{position_label}: POSITION ATOM={int(atom_index)} NOPBC"
        )
        arguments.extend(
            f"{position_label}.{axis}" for axis in ("x", "y", "z")
        )
    return position_lines, ",".join(arguments)


@dataclass
class PlumedAtomMap:
    """Resolved PDB-order to topology-index mapping.

    Attributes
    ----------
    atoms_1based : ``[n_atoms]``
        Element ``k`` is the 1-based topology index used by the POSITION
        action for the atom that is ``k``-th in featurizer (PDB) order.
    n_topology_atoms : int
        Total atoms in the topology file, used for ``WHOLEMOLECULES``.
    n_pdb_atoms : int
        Number of featurized atoms taken from the PDB.
    unmatched : list of str
        Human-readable description of any PDB atom with no topology partner.
    used_chain_matching : bool
        Whether chain indices were part of the matching key.
    """

    atoms_1based: np.ndarray
    n_topology_atoms: int
    n_pdb_atoms: int
    unmatched: list[str] = field(default_factory=list)
    used_chain_matching: bool = True

    @property
    def ok(self) -> bool:
        """True if every featurized atom was matched."""
        return not self.unmatched

    def atoms_string(self) -> str:
        """Return the comma-separated topology indices for mapped atoms."""
        return ",".join(str(int(i)) for i in self.atoms_1based)

    def report(self) -> str:
        """Plain-text summary of the mapping, including any unmatched atoms."""
        identity = bool(
            np.array_equal(
                self.atoms_1based, np.arange(1, self.n_pdb_atoms + 1)
            )
        )
        first = ", ".join(map(str, self.atoms_1based[:12].astype(int)))
        more = " ..." if self.n_pdb_atoms > 12 else ""
        lines = [
            f"PlumedAtomMap: {self.n_pdb_atoms} featurized atoms -> "
            f"topology with {self.n_topology_atoms} atoms",
            f"  chain-aware matching: {self.used_chain_matching}",
            f"  identity mapping    : {identity}"
            "  (False is normal: pdb2gmx reorders atoms)",
            f"  first 12 indices    : {first}{more}",
        ]
        if self.unmatched:
            lines.append(f"  UNMATCHED ({len(self.unmatched)}):")
            lines.extend(f"    {u}" for u in self.unmatched[:10])
            if len(self.unmatched) > 10:
                lines.append(f"    ... and {len(self.unmatched) - 10} more")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        colour = "#1a7f37" if self.ok else "#cf222e"
        status = "resolved" if self.ok else f"{len(self.unmatched)} UNMATCHED"
        cell = "padding:2px 8px"
        first = ",".join(map(str, self.atoms_1based[:12].astype(int)))
        more = "..." if self.n_pdb_atoms > 12 else ""
        return (
            "<table style='border-collapse:collapse;font-size:92%'>"
            "<caption style='text-align:left'><b>PLUMED atom map</b> &mdash; "
            f"<span style='color:{colour}'><b>{status}</b></span></caption>"
            f"<tr><td style='{cell}'>featurized atoms</td>"
            f"<td style='{cell}'><b>{self.n_pdb_atoms}</b></td></tr>"
            f"<tr><td style='{cell}'>topology atoms</td>"
            f"<td style='{cell}'><b>{self.n_topology_atoms}</b></td></tr>"
            f"<tr><td style='{cell}'>first indices</td>"
            f"<td style='{cell}'><code>{first}{more}</code></td></tr>"
            "</table>"
        )

    def __repr__(self) -> str:
        return self.report()


def map_pdb_to_topology(
    pdb: str | Path,
    topology: str | Path,
    featurized_atoms: np.ndarray | None = None,
    strict: bool = True,
) -> PlumedAtomMap:
    """Resolve the PDB-order to topology-index mapping by atom identity.

    Parameters
    ----------
    pdb : path
        The conformation the model was featurized against (no hydrogens, in
        ASMSA's usual protocol).
    topology : path
        A ``.gro`` or ``.pdb`` describing the simulated system — the output of
        ``gmx pdb2gmx``, *including* hydrogens.
    featurized_atoms : array, optional
        Indices (into the PDB) of the atoms the feature model actually uses.
        Defaults to every PDB atom. Pass this when the molecule was built on
        a subset.
    strict : bool
        Raise on any unmatched atom. With ``False`` the map is returned with
        ``unmatched`` populated so you can inspect it — but do not run a
        simulation from it.

    Returns
    -------
    PlumedAtomMap
    """
    import mdtraj as md

    pdb_t = md.load(str(pdb))
    top_t = md.load(str(topology))

    pdb_atoms = list(pdb_t.topology.atoms)
    if featurized_atoms is not None:
        sel = np.asarray(featurized_atoms, dtype=np.int64)
        pdb_atoms = [pdb_atoms[i] for i in sel]

    n_pdb_chains = pdb_t.topology.n_chains
    n_top_chains = top_t.topology.n_chains
    use_chain = n_pdb_chains == n_top_chains and n_pdb_chains > 1

    def build_index(use_chain_flag: bool) -> dict[tuple, int]:
        idx: dict[tuple, int] = {}
        collisions = set()
        for a in top_t.topology.atoms:
            k = _atom_key(a, use_chain_flag)
            if k in idx:
                collisions.add(k)
            else:
                idx[k] = a.index
        # Ambiguous keys are dropped so they surface as "unmatched" rather
        # than silently mapping to the wrong atom.
        for k in collisions:
            idx.pop(k, None)
        return idx

    top_index = build_index(use_chain)
    fallback_index: dict[tuple, int] | None = None  # built lazily, at most once
    topology_name = Path(topology).name

    atoms: list[int] = []
    unmatched: list[str] = []
    for a in pdb_atoms:
        k = _atom_key(a, use_chain)
        j = top_index.get(k)
        if j is None and use_chain:
            # retry ignoring chains, in case the two files split chains differently
            if fallback_index is None:
                fallback_index = build_index(False)
            j = fallback_index.get(_atom_key(a, False))
        if j is None:
            res = a.residue
            unmatched.append(
                f"PDB atom {a.index}: {res.name}{res.resSeq}:{a.name} "
                f"(chain {res.chain.index}) has no match in {topology_name}"
            )
            atoms.append(0)
        else:
            atoms.append(j + 1)  # PLUMED is 1-based

    result = PlumedAtomMap(
        atoms_1based=np.asarray(atoms, dtype=np.int64),
        n_topology_atoms=int(top_t.n_atoms),
        n_pdb_atoms=len(pdb_atoms),
        unmatched=unmatched,
        used_chain_matching=use_chain,
    )
    if strict and not result.ok:
        raise ValueError(
            "could not map every featurized atom onto the simulation topology; "
            "PLUMED input would bias a scrambled coordinate.\n"
            + result.report()
            + "\n\nCommon causes: the PDB and the topology describe different "
            "residues or protonation states, a residue was renamed by the force "
            "field in a way not covered by the alias table, or the PDB contains "
            "hydrogens. Pass strict=False to inspect the mapping."
        )
    return result


@dataclass
class CVGrid:
    """Metadynamics grid and hill width derived from observed CV values.

    PLUMED pre-allocates a grid for the accumulated bias, so its extent must
    be fixed in advance. Too narrow and the simulation escapes the grid and
    crashes; too wide costs memory and some performance. The observed range
    is therefore padded generously.
    """

    cv_min: np.ndarray
    cv_max: np.ndarray
    grid_min: np.ndarray
    grid_max: np.ndarray
    sigma: np.ndarray
    margin: float
    sigma_divisor: float

    @property
    def grid_min_str(self) -> str:
        return _format_floats(self.grid_min)

    @property
    def grid_max_str(self) -> str:
        return _format_floats(self.grid_max)

    @property
    def sigma_str(self) -> str:
        return _format_floats(self.sigma)

    def report(self) -> str:
        """Plain-text summary of the observed range, grid and hill width."""
        lines = [
            f"CVGrid (margin={self.margin}x range, "
            f"sigma=range/{self.sigma_divisor:g})"
        ]
        for i in range(len(self.cv_min)):
            lines.append(
                f"  CV{i}: observed [{self.cv_min[i]:+.3f}, {self.cv_max[i]:+.3f}]  "
                f"grid [{self.grid_min[i]:+.3f}, {self.grid_max[i]:+.3f}]  "
                f"sigma {self.sigma[i]:.4f}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.report()


def cv_grid(
    z,
    margin: float = 3.0,
    sigma_divisor: float = 20.0,
    min_sigma: float = 1e-3,
) -> CVGrid:
    """Grid extent and Gaussian width for METAD, from observed CV values.

    Parameters
    ----------
    z : ``[n_frames, n_cv]``
        CV values over a reference trajectory (the test split, or the whole
        trajectory).
    margin : float
        Grid padding on each side, as a multiple of the observed range. v1
        used 3.0; metadynamics deliberately pushes the system *outside* the
        sampled region, so generous padding is the safe choice.
    sigma_divisor : float
        Hill width = observed range / this. v1 used 20.
    """
    import torch

    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"expected 2-D [n_frames, n_cv], got {z.shape}")
    if not np.isfinite(z).all():
        raise ValueError("CV values contain non-finite entries")

    lo = z.min(axis=0)
    hi = z.max(axis=0)
    rng = hi - lo
    if np.any(rng <= 0):
        raise ValueError(
            f"a CV has zero range ({rng}); the latent collapsed and cannot "
            "drive metadynamics. Retrain, checking the 'usable spread' metric."
        )
    sigma = np.maximum(rng / float(sigma_divisor), min_sigma)
    return CVGrid(
        cv_min=lo,
        cv_max=hi,
        grid_min=lo - rng * margin,
        grid_max=hi + rng * margin,
        sigma=sigma,
        margin=float(margin),
        sigma_divisor=float(sigma_divisor),
    )


def write_plumed_dat(
    path: str | Path,
    model_file: str,
    atom_map: PlumedAtomMap,
    grid: CVGrid,
    n_cv: int = 2,
    *,
    temperature: float = 300.0,
    pace: int = 1000,
    height: float = 1.0,
    biasfactor: float = 15.0,
    stride_print: int = 100,
    hills_file: str = "HILLS",
    colvar_file: str = "COLVAR",
    grid_bin: int | None = None,
    whole_molecules: bool = True,
    whole_molecules_atoms: int | None = None,
    restart: bool = True,
    calc_rct: bool = True,
    label: str = "model",
) -> Path:
    """Write a well-tempered metadynamics ``plumed.dat`` on the learned CVs.

    Parameters
    ----------
    model_file : str
        Filename of the traced PLUMED CV model, *as seen from the simulation's
        working directory* (so a bare filename, not an absolute path, if you
        copy it next to the ``.tpr``).
    atom_map : PlumedAtomMap
        Must be fully resolved; a map with unmatched atoms is refused.
    grid : CVGrid
    temperature : float
        Written as ``TEMP``. Well-tempered metadynamics needs the temperature
        to interpret ``BIASFACTOR``; stating it explicitly avoids depending on
        what the MD engine reports.
    biasfactor : float
        Well-tempered bias factor. ``1`` would mean plain (non-tempered)
        metadynamics, which does not converge.
    calc_rct : bool
        Add ``CALC_RCT``, which makes PLUMED accumulate the time-dependent
        offset c(t) needed to reweight the biased trajectory back to
        unbiased free energies. Cheap, and you cannot add it retroactively.
    restart : bool
        Emit ``RESTART`` so an interrupted run resumes by reading the existing
        ``HILLS``. Requires the file to exist — create it with ``touch HILLS``
        before the first run, as v1's notebook did.
    """
    if not atom_map.ok:
        raise ValueError(
            "refusing to write plumed.dat from an unresolved atom map:\n"
            + atom_map.report()
        )
    if n_cv < 1:
        raise ValueError("n_cv must be >= 1")
    if len(grid.sigma) < n_cv:
        raise ValueError(f"grid describes {len(grid.sigma)} CVs but n_cv={n_cv}")
    if biasfactor <= 1.0:
        raise ValueError(
            "biasfactor must be > 1 for well-tempered metadynamics, "
            f"got {biasfactor}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    args = ",".join(f"{label}.node-{i}" for i in range(n_cv))
    position_lines, position_args = _position_inputs(atom_map, label)
    lines: list[str] = [
        "# Generated by asmsa_v2.export.plumed.write_plumed_dat",
        "# CVs are the latent coordinates of the trained ASMSA autoencoder.",
        "# POSITION actions follow featurizer (PDB) order and use topology indices.",
        "# See asmsa_v2.export.plumed.map_pdb_to_topology.",
        "",
    ]
    if restart:
        lines += [
            "# Resume from an existing HILLS file (create it with `touch HILLS`",
            "# before the first run).",
            "RESTART",
            "",
        ]
    if whole_molecules:
        n_whole = whole_molecules_atoms or atom_map.n_topology_atoms
        lines += [
            "# Reassemble molecules broken by periodic boundary conditions",
            "# before the CV is evaluated.",
            f"WHOLEMOLECULES ENTITY0=1-{int(n_whole)}",
            "",
        ]

    lines += [
        *position_lines,
        f"{label}: PYTORCH_MODEL FILE={model_file} ARG={position_args}",
        "",
    ]

    metad = [
        f"ARG={args}",
        f"PACE={pace}",
        f"HEIGHT={height}",
        f"BIASFACTOR={biasfactor}",
        f"TEMP={temperature}",
        f"SIGMA={_format_floats(grid.sigma[:n_cv])}",
        f"GRID_MIN={_format_floats(grid.grid_min[:n_cv])}",
        f"GRID_MAX={_format_floats(grid.grid_max[:n_cv])}",
    ]
    if grid_bin:
        metad.append(f"GRID_BIN={','.join([str(int(grid_bin))] * n_cv)}")
    if calc_rct:
        metad.append("CALC_RCT")
    metad.append(f"FILE={hills_file}")

    print_args = f"{args},metad.bias" + (",metad.rct" if calc_rct else "")
    lines += [
        "metad: METAD ...",
        *[f"  {m}" for m in metad],
        "...",
        "",
        f"PRINT FILE={colvar_file} ARG={print_args} STRIDE={stride_print}",
        "",
    ]

    path.write_text("\n".join(lines))
    return path


def write_plumed_driver_dat(
    path: str | Path,
    model_file: str,
    atom_map: PlumedAtomMap,
    n_cv: int = 2,
    colvar_file: str = "COLVAR.driver",
    stride: int = 1,
    label: str = "model",
) -> Path:
    """Write a ``plumed.dat`` that only *evaluates* the CVs, with no bias.

    Useful with ``plumed driver`` to recompute CV values on any trajectory --
    for example to confirm that the CVs PLUMED computes during the simulation
    match the ones Python computes from the same frames.
    """
    if not atom_map.ok:
        raise ValueError("refusing to write input from an unresolved atom map")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    args = ",".join(f"{label}.node-{i}" for i in range(n_cv))
    position_lines, position_args = _position_inputs(atom_map, label)
    lines = [
        "# CV evaluation only (no bias) -- for `plumed driver`",
        f"WHOLEMOLECULES ENTITY0=1-{atom_map.n_topology_atoms}",
        *position_lines,
        f"{label}: PYTORCH_MODEL FILE={model_file} ARG={position_args}",
        f"PRINT FILE={colvar_file} ARG={args} STRIDE={stride}",
        "",
    ]
    path.write_text("\n".join(lines))
    return path
