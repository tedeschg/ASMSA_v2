"""py3Dmol-based 3D structure rendering for individual trajectory frames.

py3Dmol is an optional dependency: ``pip install py3Dmol``. If unavailable,
these functions raise a clear ImportError with the install hint.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Sequence

try:
    import py3Dmol
    _P3D_AVAILABLE = True
except ImportError:
    py3Dmol = None
    _P3D_AVAILABLE = False


def _require_p3d():
    if not _P3D_AVAILABLE:
        raise ImportError(
            "py3Dmol is not installed. Install with `pip install py3Dmol` to use "
            "ASMSA v2's 3D structure visualizations."
        )


def _frame_pdb(traj, frame: int) -> str:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "frame.pdb"
        traj[frame].save_pdb(str(path))
        return path.read_text()


def view_frame(
    traj,
    frame: int,
    width: int = 400,
    height: int = 400,
    style: str | dict = "cartoon_sticks",
    component_colors: dict[str, str] | None = None,
    component_atoms: dict[str, Sequence[int]] | None = None,
):
    """Render a single trajectory frame with py3Dmol.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        A trajectory whose ``traj[frame]`` can be sliced to one frame.
    frame : int
        Frame index to render.
    style : str or dict
        Preset: ``"cartoon"`` (protein ribbon), ``"sticks"`` (all heavy atoms),
        ``"cartoon_sticks"`` (ribbon for protein + sticks for the rest).
        Pass a py3Dmol style dict for full control.
    component_colors : optional, maps component name -> color (e.g. {'protein': 'cyan', 'ligand': 'magenta'}).
        Requires ``component_atoms`` to be set.
    component_atoms : optional, maps component name -> list of atom indices.
    """
    _require_p3d()
    pdb = _frame_pdb(traj, frame)

    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb, "pdb")

    if component_colors and component_atoms:
        view.setStyle({"model": -1}, {"stick": {"colorscheme": "default", "opacity": 0.0}})
        for name, color in component_colors.items():
            idxs = list(component_atoms[name])
            serials = [i + 1 for i in idxs]  # py3Dmol uses 1-based PDB serials
            if isinstance(style, str) and "cartoon" in style and name.lower() == "protein":
                view.addStyle({"serial": serials}, {"cartoon": {"color": color}})
            view.addStyle({"serial": serials}, {"stick": {"color": color, "radius": 0.15}})
    else:
        if style == "cartoon":
            view.setStyle({}, {"cartoon": {"color": "spectrum"}})
        elif style == "sticks":
            view.setStyle({}, {"stick": {}})
        elif style == "cartoon_sticks":
            view.setStyle({}, {"cartoon": {"color": "white"}})
            view.addStyle({"hetflag": True}, {"stick": {"colorscheme": "greenCarbon"}})
        elif isinstance(style, dict):
            view.setStyle({}, style)
        else:
            raise ValueError(f"unknown style {style!r}")

    view.zoomTo()
    return view


def view_frames_grid(
    traj,
    frames: Sequence[int],
    titles: Sequence[str] | None = None,
    width: int = 300,
    height: int = 300,
    **style_kwargs,
):
    """Render multiple frames side-by-side in a py3Dmol grid viewer.

    Useful for showing one representative frame per latent cluster.
    """
    _require_p3d()
    n = len(frames)
    if n == 0:
        raise ValueError("`frames` must have at least one index")
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    view = py3Dmol.view(width=width * cols, height=height * rows,
                        viewergrid=(rows, cols), linked=False)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        view.addModel(_frame_pdb(traj, f), "pdb", viewer=(r, c))
        view.setStyle({"model": -1}, {"stick": {}}, viewer=(r, c))
        view.zoomTo(viewer=(r, c))
    return view
