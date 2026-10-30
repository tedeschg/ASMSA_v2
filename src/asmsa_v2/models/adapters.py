"""Per-molecule input / output adapters that project between the
molecule-specific feature-vector length and the shared trunk width.
"""
from typing import Mapping

import torch
import torch.nn as nn


class PerMolInputAdapter(nn.Module):
    def __init__(self, feature_dims: Mapping[str, int], hidden: int):
        super().__init__()
        self.hidden = hidden
        self.adapters = nn.ModuleDict({
            mol_id: nn.Linear(int(dim), hidden) for mol_id, dim in feature_dims.items()
        })

    def forward(self, x: torch.Tensor, mol_id: str) -> torch.Tensor:
        if mol_id not in self.adapters:
            raise KeyError(f"unknown mol_id {mol_id!r}; known: {list(self.adapters)}")
        return self.adapters[mol_id](x)


class PerMolOutputAdapter(nn.Module):
    def __init__(self, feature_dims: Mapping[str, int], hidden: int):
        super().__init__()
        self.hidden = hidden
        self.adapters = nn.ModuleDict({
            mol_id: nn.Linear(hidden, int(dim)) for mol_id, dim in feature_dims.items()
        })

    def forward(self, h: torch.Tensor, mol_id: str) -> torch.Tensor:
        if mol_id not in self.adapters:
            raise KeyError(f"unknown mol_id {mol_id!r}; known: {list(self.adapters)}")
        return self.adapters[mol_id](h)
