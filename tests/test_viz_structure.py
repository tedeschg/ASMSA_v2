from pathlib import Path

from asmsa_v2.viz import structure


class _Frame:
    def save_pdb(self, filename):
        Path(filename).write_text("PDB frame")


class _Trajectory:
    def __getitem__(self, frame):
        return _Frame()


class _Viewer:
    def __init__(self):
        self.models = []

    def addModel(self, pdb, *args, **kwargs):
        self.models.append(pdb)

    def setStyle(self, *args, **kwargs):
        pass

    def addStyle(self, *args, **kwargs):
        pass

    def zoomTo(self, *args, **kwargs):
        pass


class _Py3Dmol:
    def __init__(self):
        self.viewer = _Viewer()

    def view(self, **kwargs):
        return self.viewer


def test_view_frame_saves_pdb_to_path(monkeypatch):
    fake_py3dmol = _Py3Dmol()
    monkeypatch.setattr(structure, "_P3D_AVAILABLE", True)
    monkeypatch.setattr(structure, "py3Dmol", fake_py3dmol)

    structure.view_frame(_Trajectory(), 0, style="sticks")

    assert fake_py3dmol.viewer.models == ["PDB frame"]


def test_view_frames_grid_saves_pdb_to_path(monkeypatch):
    fake_py3dmol = _Py3Dmol()
    monkeypatch.setattr(structure, "_P3D_AVAILABLE", True)
    monkeypatch.setattr(structure, "py3Dmol", fake_py3dmol)

    structure.view_frames_grid(_Trajectory(), [0, 1])

    assert fake_py3dmol.viewer.models == ["PDB frame", "PDB frame"]
