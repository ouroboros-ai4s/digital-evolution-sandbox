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
    fac = torch.zeros((1, 1, 4), dtype=torch.int8)
    sid[0, 0, 0] = pred; cnt[0, 0, 0] = 50
    sid[0, 0, 1] = pred; cnt[0, 0, 1] = 50         # identical strain, same faction (0)
    birth = torch.zeros((1, 1, 4), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    out = phase1_antagonism(sid, cnt, fac, phe, birth, T=5, z_max=8.0, generator=g)
    assert out[0, 0, 0].item() == 50 and out[0, 0, 1].item() == 50


def test_predator_kills_prey_with_self_loss():
    t, pred, prey, phe = _setup()
    sid = torch.zeros((1, 1, 4), dtype=torch.int32)
    cnt = torch.zeros((1, 1, 4), dtype=torch.int32)
    fac = torch.zeros((1, 1, 4), dtype=torch.int8)
    fac[0, 0, 0] = 0; fac[0, 0, 1] = 1     # different factions: faction gate passes
    sid[0, 0, 0] = pred; cnt[0, 0, 0] = 100
    sid[0, 0, 1] = prey; cnt[0, 0, 1] = 100
    birth = torch.zeros((1, 1, 4), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    out = phase1_antagonism(sid, cnt, fac, phe, birth, T=5, z_max=8.0, generator=g)
    assert out[0, 0, 1].item() < 100
    assert out[0, 0, 0].item() < 100
    assert out[0, 0, 1].item() == 100 - round(100 * (8 * 0.4 / 8.4))


def test_no_hit_without_matching_feature():
    t, pred, prey, phe = _setup()
    sid = torch.zeros((1, 1, 4), dtype=torch.int32)
    cnt = torch.zeros((1, 1, 4), dtype=torch.int32)
    fac = torch.zeros((1, 1, 4), dtype=torch.int8)
    fac[0, 0, 0] = 0; fac[0, 0, 1] = 1     # different factions: faction gate passes
    sid[0, 0, 0] = prey; cnt[0, 0, 0] = 50
    sid[0, 0, 1] = prey; cnt[0, 0, 1] = 50
    birth = torch.zeros((1, 1, 4), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    out = phase1_antagonism(sid, cnt, fac, phe, birth, T=5, z_max=8.0, generator=g)
    assert out[0, 0, 0].item() == 50 and out[0, 0, 1].item() == 50


def test_same_faction_immune_even_when_different_strains():
    # two DIFFERENT strains (predator + its valid prey) on the SAME faction:
    # must NOT fight (faction gate), even though strain-targeting would allow it.
    t, pred, prey, phe = _setup()
    sid = torch.zeros((1, 1, 4), dtype=torch.int32)
    cnt = torch.zeros((1, 1, 4), dtype=torch.int32)
    fac = torch.zeros((1, 1, 4), dtype=torch.int8)
    sid[0, 0, 0] = pred; cnt[0, 0, 0] = 100; fac[0, 0, 0] = 1
    sid[0, 0, 1] = prey; cnt[0, 0, 1] = 100; fac[0, 0, 1] = 1   # same faction
    birth = torch.zeros((1, 1, 4), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    out = phase1_antagonism(sid, cnt, fac, phe, birth, T=5, z_max=8.0, generator=g)
    assert out[0, 0, 0].item() == 100 and out[0, 0, 1].item() == 100   # no kills


def test_cross_faction_fights_even_when_same_strain():
    # SAME strain (a self-targeting predator) on DIFFERENT factions: must fight.
    # BroadSweep preys on families F,Z; BroadSweep is itself Z-family -> self-prey.
    t = StrainTable()
    pred = t.get_or_mint(("BroadSweep",))
    phe = t.phenotype_arrays(DEV)
    sid = torch.zeros((1, 1, 4), dtype=torch.int32)
    cnt = torch.zeros((1, 1, 4), dtype=torch.int32)
    fac = torch.zeros((1, 1, 4), dtype=torch.int8)
    sid[0, 0, 0] = pred; cnt[0, 0, 0] = 100; fac[0, 0, 0] = 0
    sid[0, 0, 1] = pred; cnt[0, 0, 1] = 100; fac[0, 0, 1] = 3   # different faction, same strain
    birth = torch.zeros((1, 1, 4), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    out = phase1_antagonism(sid, cnt, fac, phe, birth, T=5, z_max=8.0, generator=g)
    assert out[0, 0, 0].item() < 100 and out[0, 0, 1].item() < 100   # both took losses
