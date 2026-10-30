from typing import Mapping, Sequence

import torch
import torch.nn as nn

from .adapters import PerMolOutputAdapter
from .encoder import _activation


class Decoder(nn.Module):
    def __init__(
        self,
        feature_dims: Mapping[str, int],
        latent_dim: int,
        trunk_widths: Sequence[int] = (32, 64, 128),
        activation: str = "gelu",
        batchnorm: bool = True,
    ):
        super().__init__()
        self.feature_dims = dict(feature_dims)
        self.latent_dim = latent_dim

        layers: list[nn.Module] = [nn.Linear(latent_dim, trunk_widths[0])]
        if batchnorm:
            layers.append(nn.BatchNorm1d(trunk_widths[0]))
        layers.append(_activation(activation))
        for in_w, out_w in zip(trunk_widths[:-1], trunk_widths[1:]):
            layers.append(nn.Linear(in_w, out_w))
            if batchnorm:
                layers.append(nn.BatchNorm1d(out_w))
            layers.append(_activation(activation))
        self.trunk = nn.Sequential(*layers)

        self.adapter_out = PerMolOutputAdapter(feature_dims, hidden=trunk_widths[-1])

    def forward(self, z: torch.Tensor, mol_id: str) -> torch.Tensor:
        h = self.trunk(z)
        return self.adapter_out(h, mol_id)
