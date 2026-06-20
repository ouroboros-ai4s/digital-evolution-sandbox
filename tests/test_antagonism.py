# tests/test_antagonism.py
import torch
from des.kernels.antagonism import phase1_antagonism
from des.phenotype_cache import StrainTable

DEV = torch.device("cpu")


def _setup():
    t = StrainTable()
    pred = t.get_or_mint(("BroadSweep",))      # z=0.4, prey F∪Z
    prey = t.get_or_mint(("F4Nr4",))           # F feature, no z
    phe = t.phenotype_arrays(DEV)
    return t, pred, prey, phe


def test_same_strain_immunity_no_self_annihilation():
    t, pred, prey, phe = _setup()
    sid = torch.zeros((1, 1, 4), dtype=torch.int32)
    cnt = torch.zeros((1, 1, 4), dtype=torch.int32)
    sid[0, 0, 0] = pred; cnt[0, 0, 0] = 50
    sid[0, 0, 1] = pred; cnt[0, 0, 1] = 50         # identical strain
    birth = torch.zeros((1, 1, 4), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    out = phase1_antagonism(sid, cnt, phe, birth, T=5, z_max=8.0, generator=g)
    assert out[0, 0, 0].item() == 50 and out[0, 0, 1].item() == 50


def test_predator_kills_prey_with_self_loss():
    t, pred, prey, phe = _setup()
    sid = torch.zeros((1, 1, 4), dtype=torch.int32)
    cnt = torch.zeros((1, 1, 4), dtype=torch.int32)
    sid[0, 0, 0] = pred; cnt[0, 0, 0] = 100
    sid[0, 0, 1] = prey; cnt[0, 0, 1] = 100
    birth = torch.zeros((1, 1, 4), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    out = phase1_antagonism(sid, cnt, phe, birth, T=5, z_max=8.0, generator=g)
    assert out[0, 0, 1].item() < 100
    assert out[0, 0, 0].item() < 100
    assert out[0, 0, 1].item() == 100 - round(100 * (8 * 0.4 / 8.4))


def test_no_hit_without_matching_feature():
    t, pred, prey, phe = _setup()
    sid = torch.zeros((1, 1, 4), dtype=torch.int32)
    cnt = torch.zeros((1, 1, 4), dtype=torch.int32)
    sid[0, 0, 0] = prey; cnt[0, 0, 0] = 50
    sid[0, 0, 1] = prey; cnt[0, 0, 1] = 50
    birth = torch.zeros((1, 1, 4), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    out = phase1_antagonism(sid, cnt, phe, birth, T=5, z_max=8.0, generator=g)
    assert out[0, 0, 0].item() == 50 and out[0, 0, 1].item() == 50
