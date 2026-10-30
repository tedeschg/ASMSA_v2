import math

import torch
import torch.distributions as D

from .base import Prior


class GMMPrior(Prior):
    """Gaussian mixture prior with arbitrary component means / scales / weights."""

    def __init__(self, means, scales, weights=None):
        means = torch.as_tensor(means, dtype=torch.float32)
        scales = torch.as_tensor(scales, dtype=torch.float32)
        assert means.ndim == 2 and scales.ndim == 2 and means.shape == scales.shape
        latent_dim = means.shape[1]
        super().__init__(latent_dim)
        K = means.shape[0]
        if weights is None:
            weights = torch.full((K,), 1.0 / K)
        else:
            weights = torch.as_tensor(weights, dtype=torch.float32)
            weights = weights / weights.sum()
        self._means = means
        self._scales = scales
        self._weights = weights
        self._device = torch.device("cpu")
        self._build()

    def _build(self):
        mix = D.Categorical(probs=self._weights.to(self._device))
        comp = D.Independent(
            D.Normal(self._means.to(self._device), self._scales.to(self._device)), 1
        )
        self.dist = D.MixtureSameFamily(mix, comp)

    def to(self, device):
        self._device = torch.device(device)
        self._build()
        return self

    def sample(self, n: int) -> torch.Tensor:
        return self.dist.sample((n,))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(z)


def gmm_on_circle(K: int = 10, radius: float = 1.5, scale_std: float = 0.15) -> GMMPrior:
    angles = torch.linspace(0, 2 * math.pi, K + 1)[:-1]
    means = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
    scales = torch.full((K, 2), scale_std)
    return GMMPrior(means=means, scales=scales)
