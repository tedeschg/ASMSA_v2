from typing import Sequence

import torch
import torch.nn as nn

try:  # current API (torch >= 1.12); the torch.nn.utils.spectral_norm hook is legacy
    from torch.nn.utils.parametrizations import spectral_norm
except ImportError:  # pragma: no cover - very old torch
    from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    """Molecule-agnostic latent-space critic.

    Spectral-normalized ``Linear`` layers + ``LeakyReLU`` + raw-logit output.
    Pair with ``bce_disc_loss`` / ``bce_gen_loss`` from ``asmsa_v2.losses``.
    """

    def __init__(
        self,
        latent_dim: int,
        widths: Sequence[int] = (64, 64),
        spectral_norm_enabled: bool = True,
        leaky_slope: float = 0.2,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        def _lin(i, o):
            layer = nn.Linear(i, o)
            return spectral_norm(layer) if spectral_norm_enabled else layer

        layers: list[nn.Module] = []
        prev = latent_dim
        for w in widths:
            layers.append(_lin(prev, w))
            layers.append(nn.LeakyReLU(leaky_slope))
            prev = w
        layers.append(_lin(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
