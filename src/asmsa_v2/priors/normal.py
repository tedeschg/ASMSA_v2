import torch
import torch.distributions as D

from .base import Prior


class IsotropicNormalPrior(Prior):
    def __init__(self, latent_dim: int, scale: float = 1.0):
        super().__init__(latent_dim)
        self.scale = scale
        self._device = torch.device("cpu")
        self._build()

    def _build(self):
        loc = torch.zeros(self.latent_dim, device=self._device)
        scale = torch.full((self.latent_dim,), self.scale, device=self._device)
        self.dist = D.Independent(D.Normal(loc, scale), 1)

    def to(self, device):
        self._device = torch.device(device)
        self._build()
        return self

    def sample(self, n: int) -> torch.Tensor:
        return self.dist.sample((n,))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(z)
