import torch

from asmsa_v2 import IsotropicNormalPrior, GMMPrior, gmm_on_circle


def test_normal_prior_shapes():
    p = IsotropicNormalPrior(latent_dim=3)
    z = p.sample(64)
    assert z.shape == (64, 3)
    lp = p.log_prob(z)
    assert lp.shape == (64,)


def test_gmm_prior_shapes():
    p = GMMPrior(means=torch.tensor([[0., 0.], [2., 2.]]),
                 scales=torch.tensor([[0.3, 0.3], [0.3, 0.3]]))
    z = p.sample(128)
    assert z.shape == (128, 2)
    lp = p.log_prob(z)
    assert lp.shape == (128,)


def test_gmm_on_circle():
    p = gmm_on_circle(K=8, radius=2.0, scale_std=0.2)
    z = p.sample(1000)
    assert z.shape == (1000, 2)
    # points should sit near radius 2 on average
    r = torch.linalg.norm(z, dim=1)
    assert 1.5 < r.mean() < 2.5
