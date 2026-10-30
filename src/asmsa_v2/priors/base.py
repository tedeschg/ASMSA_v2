from abc import ABC, abstractmethod

import torch


class Prior(ABC):
    """Abstract prior over the latent space.

    Subclasses must implement ``sample(n)`` returning ``[n, latent_dim]`` and
    ``log_prob(z)`` returning ``[n]`` (unnormalized log-density is fine for
    density-matching losses, but normalized is preferred).
    """

    latent_dim: int

    def __init__(self, latent_dim: int):
        self.latent_dim = latent_dim

    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:  # [n, latent_dim]
        ...

    @abstractmethod
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:  # [n]
        ...

    def to(self, device):
        return self
