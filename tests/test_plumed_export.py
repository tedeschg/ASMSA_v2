import numpy as np
import torch
from torch import nn

from asmsa_v2.export.plumed import (
    PlumedAtomMap,
    cv_grid,
    write_plumed_dat,
    write_plumed_driver_dat,
)
from asmsa_v2.export.torchscript import PlumedCVModel


def _atom_map():
    return PlumedAtomMap(
        atoms_1based=np.array([3, 1]),
        n_topology_atoms=4,
        n_pdb_atoms=2,
    )


def test_write_plumed_dat_feeds_mapped_positions_to_torch_model(tmp_path):
    path = write_plumed_dat(
        tmp_path / "plumed.dat",
        model_file="cv-plumed.pt",
        atom_map=_atom_map(),
        grid=cv_grid(np.array([[-1.0, -1.0], [1.0, 1.0]])),
        restart=False,
        calc_rct=False,
    )

    text = path.read_text()
    assert "model_position_0: POSITION ATOM=3" in text
    assert "model_position_1: POSITION ATOM=1" in text
    assert (
        "model: PYTORCH_MODEL FILE=cv-plumed.pt "
        "ARG=model_position_0.x,model_position_0.y,model_position_0.z,"
        "model_position_1.x,model_position_1.y,model_position_1.z"
    ) in text
    assert "PYTORCH_MODEL_CV" not in text


def test_write_plumed_driver_dat_uses_standard_torch_model(tmp_path):
    path = write_plumed_driver_dat(
        tmp_path / "driver.dat",
        model_file="cv-plumed.pt",
        atom_map=_atom_map(),
    )

    text = path.read_text()
    assert "model_position_0: POSITION ATOM=3" in text
    assert "model_position_1: POSITION ATOM=1" in text
    assert "model: PYTORCH_MODEL FILE=cv-plumed.pt ARG=model_position_0.x," in text
    assert "PYTORCH_MODEL_CV" not in text


def test_plumed_cv_model_preserves_single_sample_batch_dimension():
    class FlattenPositions(nn.Module):
        def forward(self, xyz):
            return xyz.reshape(-1, 1)

    model = PlumedCVModel(
        FlattenPositions(),
        nn.Identity(),
        nn.Linear(6, 2),
    )

    positions = torch.arange(6, dtype=torch.float32).reshape(1, -1)
    traced = torch.jit.trace(model, positions)
    result = traced(positions)

    assert result.shape == (1, 2)