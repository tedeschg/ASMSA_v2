"""Export cluster representatives to disk."""
from __future__ import annotations

from pathlib import Path

from .clustering import ClusterResult


def save_cluster_representatives(
    result: ClusterResult,
    traj,
    out_dir: str | Path,
    prefix: str = "cluster",
) -> dict[int, Path]:
    """Save each cluster's representative frame as a PDB in ``out_dir``.

    Returns a dict mapping cluster id → saved file path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[int, Path] = {}
    for cid, frame in result.representatives.items():
        path = out_dir / f"{prefix}_{cid:02d}_frame{frame:06d}.pdb"
        traj[int(frame)].save_pdb(str(path))
        out[cid] = path
    return out
