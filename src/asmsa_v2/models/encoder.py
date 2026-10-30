from typing import Mapping, Sequence

import torch
import torch.nn as nn

from .adapters import PerMolInputAdapter


def _activation(name: str) -> nn.Module:
    name = name.lower()
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU(0.2)}[name]


class Encoder(nn.Module):
    def __init__(
        self,
        feature_dims: Mapping[str, int],
        latent_dim: int,
        trunk_widths: Sequence[int] = (128, 64, 32),
        activation: str = "gelu",
        batchnorm: bool = True,
    ):
        super().__init__()
        self.feature_dims = dict(feature_dims)
        self.latent_dim = latent_dim

        self.adapter_in = PerMolInputAdapter(feature_dims, hidden=trunk_widths[0])

        layers: list[nn.Module] = []
        if batchnorm:
            layers.append(nn.BatchNorm1d(trunk_widths[0]))
        layers.append(_activation(activation))
        for in_w, out_w in zip(trunk_widths[:-1], trunk_widths[1:]):
            layers.append(nn.Linear(in_w, out_w))
            if batchnorm:
                layers.append(nn.BatchNorm1d(out_w))
            layers.append(_activation(activation))
        layers.append(nn.Linear(trunk_widths[-1], latent_dim))
        self.trunk = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mol_id: str) -> torch.Tensor:
        h = self.adapter_in(x, mol_id)
        return self.trunk(h)
