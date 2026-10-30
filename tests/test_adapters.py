import torch

from asmsa_v2 import AAE


def test_round_trip_two_molecules():
    torch.manual_seed(0)
    D_a, D_b = 20, 35
    model = AAE(feature_dims={"molA": D_a, "molB": D_b}, latent_dim=2,
                encoder_widths=(32, 16), decoder_widths=(16, 32),
                disc_widths=(16,))
    model.eval()
    x_a = torch.randn(8, D_a)
    x_b = torch.randn(8, D_b)
    xa_hat, za = model(x_a, "molA")
    xb_hat, zb = model(x_b, "molB")
    assert xa_hat.shape == x_a.shape
    assert xb_hat.shape == x_b.shape
    assert za.shape == (8, 2)
    assert zb.shape == (8, 2)


def test_unknown_mol_id_raises():
    model = AAE(feature_dims={"molA": 10}, latent_dim=2,
                encoder_widths=(16, 8), decoder_widths=(8, 16), disc_widths=(8,))
    model.eval()
    try:
        model(torch.randn(4, 10), "nonexistent")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown mol_id")
