import torch
import torch.nn.functional as F

from asmsa_v2.losses.adversarial import bce_disc_loss, bce_gen_loss


def test_bce_disc_matches_torch_bce():
    torch.manual_seed(0)
    logits_real = torch.randn(32, 1)
    logits_fake = torch.randn(32, 1)
    out = bce_disc_loss(logits_real, logits_fake)
    expected = (
        F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
        + F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
    )
    assert torch.isclose(out, expected, atol=1e-6)


def test_bce_gen_matches_torch_bce():
    torch.manual_seed(0)
    logits_fake = torch.randn(32, 1)
    out = bce_gen_loss(logits_fake)
    expected = F.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))
    assert torch.isclose(out, expected, atol=1e-6)


def test_label_smoothing_applied():
    torch.manual_seed(0)
    logits_real = torch.randn(16, 1) + 2.0  # pushed toward "real" side
    logits_fake = torch.randn(16, 1) - 2.0
    smooth = bce_disc_loss(logits_real, logits_fake, label_smoothing=0.1)
    sharp = bce_disc_loss(logits_real, logits_fake, label_smoothing=0.0)
    assert not torch.isclose(smooth, sharp)
