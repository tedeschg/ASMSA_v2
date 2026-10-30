import numpy as np
import torch
from PIL import Image

from .base import Prior


class ImagePrior(Prior):
    """2-D prior sampled via inverse-CDF from a grayscale image.

    Pixel intensities form a probability density on ``[0, 1] x [0, 1]``.
    ``log_prob`` returns the log of the (normalized) pixel density.
    """

    def __init__(self, image_path: str):
        super().__init__(latent_dim=2)
        img = np.array(Image.open(image_path), dtype=np.float64)
        if img.ndim == 3:
            img = img.sum(axis=2)
        img = img / img.sum()
        self._H, self._W = img.shape
        self._density = torch.tensor(img, dtype=torch.float32)
        self._flat_cdf = torch.tensor(np.cumsum(img.flatten()), dtype=torch.float32)
        self._device = torch.device("cpu")

    def to(self, device):
        self._device = torch.device(device)
        self._density = self._density.to(self._device)
        self._flat_cdf = self._flat_cdf.to(self._device)
        return self

    def sample(self, n: int) -> torch.Tensor:
        u = torch.rand(n, device=self._device)
        idx = torch.searchsorted(self._flat_cdf, u)
        idx = idx.clamp(max=self._H * self._W - 1)
        x = (idx % self._W).float() + torch.rand(n, device=self._device)
        y = (idx // self._W).float() + torch.rand(n, device=self._device)
        x = x / self._W
        y = 1.0 - y / self._H
        return torch.stack([x, y], dim=-1)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        x = (z[:, 0] * self._W).long().clamp(0, self._W - 1)
        y = ((1.0 - z[:, 1]) * self._H).long().clamp(0, self._H - 1)
        p = self._density[y, x].clamp_min(1e-12)
        return torch.log(p)
