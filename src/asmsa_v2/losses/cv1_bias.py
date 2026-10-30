"""CV1 bias: force the first latent dimension to match a user-supplied CV."""
import torch
import torch.nn.functional as F


def cv1_bias_loss(z: torch.Tensor, cv1_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(z[:, 0], cv1_target)
