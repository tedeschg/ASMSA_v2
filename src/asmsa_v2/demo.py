"""A synthetic protein-ligand demo system.

The notebook pipeline is only useful if it can be run before you have data.
This module writes a small but *structurally valid* PDB + XTC pair that
exercises every stage: the "protein" is a peptide chain with a slow breathing
mode, and the "ligand" occupies two distinct binding poses with occasional
transitions between them. So there is

- a genuine slow degree of freedom for VAMP-2 to find,
- two well-separated states for the latent space to resolve and the pose
  clustering to recover,
- feature blocks on very different scales (stiff covalent bonds versus
  nanometre-scale inter-component distances), which is what makes
  standardization necessary.

It also writes a mock ``topology.gro`` with hydrogens added and atoms
reordered the way ``gmx pdb2gmx`` would, so the PLUMED atom-mapping stage can
be exercised without GROMACS installed.

This is a *geometric* mock, not a physical simulation: bond lengths and
angles are plausible but the trajectory is not an integration of any force
field. Use it to learn and test the pipeline, never to draw conclusions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Backbone + CB, in the order a PDB file lists them.
_RESIDUE_ATOMS = ["N", "CA", "C", "O", "CB"]
_LIGAND_ATOMS = ["C1", "C2", "C3", "O1", "N1"]

# How pdb2gmx reorders a residue: hydrogens inserted, backbone carboxyl last.
_GMX_RESIDUE_ATOMS = ["N", "H", "CA", "HA", "CB", "HB1", "HB2", "HB3", "C", "O"]
_GMX_LIGAND_ATOMS = ["C1", "H1", "C2", "H2", "C3", "H3", "O1", "N1", "HN1"]


@dataclass
class DemoSystem:
    """Paths and metadata for a generated demo system."""

    conf: Path
    traj: Path
    topology_gro: Path
    n_residues: int
    n_frames: int
    n_atoms: int
    protein_atoms: np.ndarray
    ligand_atoms: np.ndarray
    true_pose: np.ndarray            # ground-truth pose id per frame
    breathing: np.ndarray            # ground-truth slow collective coordinate

    def summary(self) -> str:
        return (
            f"DemoSystem: {self.n_residues} residues + {len(self.ligand_atoms)}-atom "
            f"ligand = {self.n_atoms} atoms, {self.n_frames} frames\n"
            f"  conformation : {self.conf}\n"
            f"  trajectory   : {self.traj}\n"
            f"  mock topology: {self.topology_gro}\n"
            f"  ground truth : 2 binding poses "
            f"({int((self.true_pose == 0).sum())} / "
            f"{int((self.true_pose == 1).sum())} frames) + 1 slow breathing mode"
        )

    def __repr__(self) -> str:
        return self.summary()


def _build_topology(n_residues: int):
    import mdtraj as md

    top = md.Topology()
    elements = {"N": "N", "CA": "C", "C": "C", "O": "O", "CB": "C",
                "C1": "C", "C2": "C", "C3": "C", "O1": "O", "N1": "N"}
    chain = top.add_chain()
    for r in range(n_residues):
        res = top.add_residue("ALA", chain, resSeq=r + 1)
        for name in _RESIDUE_ATOMS:
            top.add_atom(name, md.element.get_by_symbol(elements[name]), res)
    lig_chain = top.add_chain()
    lig_res = top.add_residue("LIG", lig_chain, resSeq=n_residues + 1)
    for name in _LIGAND_ATOMS:
        top.add_atom(name, md.element.get_by_symbol(elements[name]), lig_res)
    return top


def _write_mock_gro(path: Path, n_residues: int) -> None:
    """A ``.gro`` mimicking pdb2gmx output: hydrogens added, atoms reordered."""
    rows = []
    for r in range(n_residues):
        for name in _GMX_RESIDUE_ATOMS:
            rows.append((r + 1, "ALA", name))
    for name in _GMX_LIGAND_ATOMS:
        rows.append((n_residues + 1, "LIG", name))
    with open(path, "w") as fh:
        fh.write("mock pdb2gmx output (asmsa_v2.demo)\n")
        fh.write("%5d\n" % len(rows))
        for i, (rs, rn, an) in enumerate(rows, 1):
            fh.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"
                     % (rs, rn, an, i, i * 0.1, i * 0.05, i * 0.02))
        fh.write("   6.00000   6.00000   6.00000\n")


def make_demo_complex(
    out_dir: str | Path,
    n_residues: int = 12,
    n_frames: int = 3000,
    seed: int = 0,
    conf_name: str = "complex.pdb",
    traj_name: str = "complex.xtc",
    gro_name: str = "topology.gro",
    overwrite: bool = False,
) -> DemoSystem:
    """Write a synthetic protein-ligand complex trajectory.

    Parameters
    ----------
    out_dir : path
    n_residues : int
        Length of the peptide "protein".
    n_frames : int
        Trajectory length. Frames are in time order and temporally
        correlated, as a real trajectory is, so the time-lagged and VAMP-2
        machinery is meaningful on it.
    seed : int
    overwrite : bool
        Regenerate even if the files already exist.

    Returns
    -------
    DemoSystem
    """
    import mdtraj as md

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conf = out_dir / conf_name
    traj_path = out_dir / traj_name
    gro = out_dir / gro_name

    rng = np.random.default_rng(seed)
    top = _build_topology(n_residues)
    n_prot = n_residues * len(_RESIDUE_ATOMS)
    n_lig = len(_LIGAND_ATOMS)
    n_atoms = n_prot + n_lig

    protein_atoms = np.arange(n_prot)
    ligand_atoms = np.arange(n_prot, n_atoms)

    # ---- ground-truth slow coordinates -------------------------------------
    # A breathing mode: an AR(1) process with a long correlation time, which
    # modulates how extended the peptide is.
    breathing = np.zeros(n_frames)
    for i in range(1, n_frames):
        breathing[i] = 0.995 * breathing[i - 1] + 0.1 * rng.standard_normal()
    breathing /= max(np.std(breathing), 1e-9)

    # A two-state pose process: rare switches, so both states are well sampled
    # and the transitions are visible as a path in the latent space.
    pose = np.zeros(n_frames, dtype=int)
    p_switch = 4.0 / n_frames * 6
    for i in range(1, n_frames):
        pose[i] = 1 - pose[i - 1] if rng.random() < p_switch else pose[i - 1]

    # ---- build coordinates -------------------------------------------------
    xyz = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)

    # Reference peptide: a gentle helix-like arc along x.
    t = np.arange(n_residues)
    ca_ref = np.stack([t * 0.38,
                       0.6 * np.sin(t * 0.9),
                       0.6 * np.cos(t * 0.9)], axis=1)

    for f in range(n_frames):
        # breathing scales the end-to-end extension by +-8%
        scale = 1.0 + 0.08 * breathing[f]
        ca = ca_ref * np.array([scale, 1.0, 1.0])
        ca = ca + 0.015 * rng.standard_normal(ca.shape)

        for r in range(n_residues):
            base = r * len(_RESIDUE_ATOMS)
            c_alpha = ca[r]
            # local frame from the chain direction
            nxt = ca[min(r + 1, n_residues - 1)]
            prv = ca[max(r - 1, 0)]
            d = nxt - prv
            nd = np.linalg.norm(d)
            d = d / nd if nd > 1e-6 else np.array([1.0, 0.0, 0.0])
            perp = np.cross(d, [0.0, 0.0, 1.0])
            np_ = np.linalg.norm(perp)
            perp = perp / np_ if np_ > 1e-6 else np.array([0.0, 1.0, 0.0])
            perp2 = np.cross(d, perp)
            j = 0.004 * rng.standard_normal(3)
            # plausible bond geometry: N-CA 0.146, CA-C 0.152, C=O 0.123,
            # CA-CB 0.153 nm
            xyz[f, base + 0] = c_alpha - 0.146 * d + j           # N
            xyz[f, base + 1] = c_alpha + j                        # CA
            xyz[f, base + 2] = c_alpha + 0.152 * d + j            # C
            xyz[f, base + 3] = c_alpha + 0.152 * d + 0.123 * perp + j   # O
            xyz[f, base + 4] = c_alpha + 0.153 * perp2 + j        # CB

        # Ligand: a rigid 5-atom cluster sitting in one of two pockets, close
        # to different parts of the peptide.
        pocket = (np.array([0.35, 1.05, 0.55]) if pose[f] == 0
                  else np.array([0.80, -0.95, -0.50]))
        center = ca[n_residues // 2] * np.array([1.0, 0.0, 0.0]) + pocket
        center = center + 0.05 * rng.standard_normal(3)
        local = np.array([[0.0, 0.0, 0.0],
                          [0.150, 0.0, 0.0],
                          [0.075, 0.130, 0.0],
                          [0.225, 0.130, 0.0],
                          [-0.140, 0.060, 0.0]])
        # a small random rigid rotation per frame
        ang = rng.normal(0, 0.25, 3)
        Rx = _rot_x(ang[0]); Ry = _rot_y(ang[1]); Rz = _rot_z(ang[2])
        local = local @ (Rx @ Ry @ Rz).T
        xyz[f, n_prot:] = center + local + 0.01 * rng.standard_normal((n_lig, 3))

    traj = md.Trajectory(xyz, top)
    traj.time = np.arange(n_frames, dtype=np.float64) * 10.0   # 10 ps/frame

    if overwrite or not conf.exists():
        traj[0].save_pdb(str(conf))
    if overwrite or not traj_path.exists():
        traj.save_xtc(str(traj_path))
    if overwrite or not gro.exists():
        _write_mock_gro(gro, n_residues)

    return DemoSystem(
        conf=conf, traj=traj_path, topology_gro=gro,
        n_residues=n_residues, n_frames=n_frames, n_atoms=n_atoms,
        protein_atoms=protein_atoms, ligand_atoms=ligand_atoms,
        true_pose=pose, breathing=breathing,
    )


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
