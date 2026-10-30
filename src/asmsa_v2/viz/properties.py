"""Per-frame structural properties for colouring the latent space.

Colouring the latent plane by an interpretable observable is how you find out
*what the model learned*. v1 computed radius of gyration and RMSD inline in
two notebooks; collecting them here means the complex-specific ones
(ligand-protein distance, contact count, ligand RMSD after aligning on the
protein) are available too, which are what actually distinguish binding poses.
"""
from __future__ import annotations

import numpy as np


def radius_of_gyration(traj) -> np.ndarray:
    import mdtraj as md
    return md.compute_rg(traj)


def rmsd_to(traj, reference=None, frame: int = 0, atom_indices=None) -> np.ndarray:
    """RMSD of every frame to a reference structure (nm)."""
    import mdtraj as md
    ref = reference if reference is not None else traj
    return md.rmsd(traj, ref, frame=frame, atom_indices=atom_indices)


def component_distance(traj, atoms_a, atoms_b) -> np.ndarray:
    """Distance between the centres of mass of two atom groups (nm).

    For a protein-ligand complex this is the most direct read on whether the
    ligand is bound at all.
    """
    import mdtraj as md
    a = np.asarray(atoms_a, dtype=np.int64)
    b = np.asarray(atoms_b, dtype=np.int64)
    com_a = md.compute_center_of_mass(traj.atom_slice(a))
    com_b = md.compute_center_of_mass(traj.atom_slice(b))
    return np.linalg.norm(com_a - com_b, axis=1)


def contact_count(traj, atoms_a, atoms_b, cutoff_nm: float = 0.45) -> np.ndarray:
    """Number of inter-component atom pairs closer than ``cutoff_nm`` per frame.

    Memory grows as ``len(a) * len(b)``; restrict to heavy atoms (and for a
    protein, to interface residues) on large systems.
    """
    import mdtraj as md
    a = np.asarray(atoms_a, dtype=np.int64)
    b = np.asarray(atoms_b, dtype=np.int64)
    pairs = np.stack(np.meshgrid(a, b, indexing="ij"), axis=-1).reshape(-1, 2)
    d = md.compute_distances(traj, pairs)
    return (d < cutoff_nm).sum(axis=1).astype(float)


def ligand_rmsd_after_protein_alignment(
    traj, protein_atoms, ligand_atoms, frame: int = 0
) -> np.ndarray:
    """Ligand RMSD once each frame is superposed on the protein (nm).

    This is *the* pose metric: it separates "the ligand moved" from "the whole
    complex tumbled", which a plain RMSD cannot do. Two frames with the same
    value sit in the same binding pose.
    """
    import mdtraj as md
    protein_atoms = np.asarray(protein_atoms, dtype=np.int64)
    ligand_atoms = np.asarray(ligand_atoms, dtype=np.int64)
    aligned = traj[:]
    aligned.superpose(traj, frame=frame, atom_indices=protein_atoms)
    ref = aligned[frame]
    lig = aligned.xyz[:, ligand_atoms, :]
    ref_lig = ref.xyz[0, ligand_atoms, :]
    return np.sqrt(((lig - ref_lig) ** 2).sum(axis=2).mean(axis=1))


def standard_properties(
    traj,
    components: dict[str, np.ndarray] | None = None,
    reference=None,
) -> dict[str, np.ndarray]:
    """A sensible default set of per-frame properties for this system.

    Always includes radius of gyration, RMSD to the reference and frame
    index. For a two-component system it adds the inter-component centre-of-
    mass distance, the contact count, and the pose-defining ligand RMSD.
    """
    out: dict[str, np.ndarray] = {
        "Rg (nm)": radius_of_gyration(traj),
        "RMSD (nm)": rmsd_to(traj, reference=reference, frame=0),
        "frame": np.arange(traj.n_frames, dtype=float),
    }
    if components and len(components) >= 2:
        names = list(components)
        a_name, b_name = names[0], names[1]
        a, b = np.asarray(components[a_name]), np.asarray(components[b_name])
        # treat the larger component as the "protein" for alignment
        if a.size < b.size:
            a_name, b_name, a, b = b_name, a_name, b, a
        try:
            out[f"{a_name}-{b_name} COM dist (nm)"] = component_distance(traj, a, b)
        except Exception:
            pass
        try:
            heavy = set(traj.top.select("element != H").tolist())
            a_h = np.array([i for i in a if i in heavy] or a)
            b_h = np.array([i for i in b if i in heavy] or b)
            if a_h.size * b_h.size <= 4_000_000:
                out["contacts (<0.45 nm)"] = contact_count(traj, a_h, b_h)
        except Exception:
            pass
        try:
            out[f"{b_name} RMSD, {a_name}-aligned (nm)"] = \
                ligand_rmsd_after_protein_alignment(traj, a, b)
        except Exception:
            pass
    return out
