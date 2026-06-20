import torch
from des.kernels.common import z_eff, fires_this_tick, binom

DEV = torch.device("cpu")

def test_z_eff_saturates_to_zmax():
    z = torch.tensor([0.0, 1.0, 1e9], device=DEV)
    out = z_eff(z, z_max=8.0)
    assert out[0].item() == 0.0
    assert abs(out[1].item() - 8.0 * 1.0 / 9.0) < 1e-6
    assert abs(out[2].item() - 8.0) < 1e-2

def test_z_eff_tangent_at_zero():
    z = torch.tensor([0.01], device=DEV)
    assert abs(z_eff(z, 8.0).item() - 0.01) < 1e-3

def test_fires_this_tick():
    birth = torch.tensor([0, 0, 1], device=DEV)
    period = torch.tensor([4, 3, 4], device=DEV)
    fired = fires_this_tick(birth, period, T=4)
    assert fired.tolist() == [True, False, False]

def test_fires_never_when_period_nonpositive():
    birth = torch.tensor([0], device=DEV)
    period = torch.tensor([0], device=DEV)
    assert fires_this_tick(birth, period, T=0).item() is False

def test_binom_bounds_and_determinism():
    g1 = torch.Generator(device=DEV); g1.manual_seed(123)
    g2 = torch.Generator(device=DEV); g2.manual_seed(123)
    n = torch.full((1000,), 10, dtype=torch.int32, device=DEV)
    p = torch.full((1000,), 0.3, device=DEV)
    a = binom(n, p, g1)
    b = binom(n, p, g2)
    assert torch.equal(a, b)
    assert (a >= 0).all() and (a <= 10).all()
    assert abs(a.float().mean().item() - 3.0) < 0.5
