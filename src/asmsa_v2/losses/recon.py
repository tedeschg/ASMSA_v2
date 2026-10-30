import torch
import torch.nn.functional as F


def mse_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon, target)


def huber_loss(recon: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.huber_loss(recon, target, delta=delta)


RECON_LOSSES = {"mse": mse_loss, "huber": huber_loss}
