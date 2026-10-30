from typing import Mapping, Sequence

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .discriminator import Discriminator


class AAE(nn.Module):
    """Multi-molecule adversarial autoencoder.

    Each ``mol_id`` has its own input/output adapter projecting between the
    molecule's feature-vector length ``D_mol`` and a shared trunk width. The
    discriminator is molecule-agnostic — it sees only ``z`` — so the latent
    space is forced to be shared across molecules.
    """

    def __init__(
        self,
        feature_dims: Mapping[str, int],
        latent_dim: int = 2,
        encoder_widths: Sequence[int] = (128, 64, 32),
        decoder_widths: Sequence[int] = (32, 64, 128),
        disc_widths: Sequence[int] = (64, 64),
        activation: str = "gelu",
        batchnorm: bool = True,
        spectral_norm_disc: bool = True,
    ):
        super().__init__()
        self.feature_dims = dict(feature_dims)
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            feature_dims, latent_dim, trunk_widths=encoder_widths,
            activation=activation, batchnorm=batchnorm,
        )
        self.decoder = Decoder(
            feature_dims, latent_dim, trunk_widths=decoder_widths,
            activation=activation, batchnorm=batchnorm,
        )
        self.discriminator = Discriminator(
            latent_dim, widths=disc_widths, spectral_norm_enabled=spectral_norm_disc,
        )

    def encode(self, x: torch.Tensor, mol_id: str) -> torch.Tensor:
        return self.encoder(x, mol_id)

    def decode(self, z: torch.Tensor, mol_id: str) -> torch.Tensor:
        return self.decoder(z, mol_id)

    def discriminate(self, z: torch.Tensor) -> torch.Tensor:
        return self.discriminator(z)

    def forward(self, x: torch.Tensor, mol_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, mol_id)
        x_hat = self.decode(z, mol_id)
        return x_hat, z
