# tests/test_world.py
import torch
from des.world import World, init_bb0
from des.phenotype_cache import StrainTable
from des.registry import BB0_TEMPLATE

DEV = torch.device("cpu")

def test_world_allocates_empty():
    w = World(8, 8, 16, DEV)
    assert w.strain_id.shape == (8, 8, 16)
    assert w.count.shape == (8, 8, 16)
    assert w.count.sum().item() == 0
    assert w.strain_id.dtype == torch.int32

def test_snapshot_is_independent_copy():
    w = World(4, 4, 8, DEV)
    w.count[0, 0, 0] = 5
    sid, cnt, fac = w.snapshot()
    w.count[0, 0, 0] = 99
    assert cnt[0, 0, 0].item() == 5      # snapshot unaffected

def test_init_bb0_fills_every_cell_with_one_strain():
    t = StrainTable()
    w = init_bb0(8, 8, 16, DEV, t, fill_per_cell=10)
    bb0 = t.get_or_mint(BB0_TEMPLATE["layout"])
    assert (w.count[:, :, 0] == 10).all()
    assert (w.strain_id[:, :, 0] == bb0).all()
    assert w.occupancy().min().item() == 10
    assert w.distinct_per_cell().max().item() == 1   # one strain everywhere

from des.world import init_factions

def test_world_has_faction_tensor():
    w = World(8, 8, 16, DEV)
    assert w.faction.shape == (8, 8, 16)
    assert w.faction.dtype == torch.int8
    assert w.faction.sum().item() == 0          # zero-init

def test_snapshot_returns_faction():
    w = World(4, 4, 8, DEV)
    w.count[0, 0, 0] = 5
    w.faction[0, 0, 0] = 2
    sid, cnt, fac = w.snapshot()                 # now a 3-tuple
    assert fac[0, 0, 0].item() == 2
    w.faction[0, 0, 0] = 9
    assert fac[0, 0, 0].item() == 2              # snapshot is an independent copy

def test_init_factions_seeds_four_quadrant_centers():
    t = StrainTable()
    H = W = 16; K = 32
    w = init_factions(H, W, K, DEV, t, fill_per_cell=10, n_fac=4)
    centers = [(H//4, W//4), (H//4, 3*W//4), (3*H//4, W//4), (3*H//4, 3*W//4)]
    # exactly 4 non-empty cells
    assert int((w.count.sum(dim=-1) > 0).sum()) == 4
    bb0 = t.get_or_mint(BB0_TEMPLATE["layout"])
    seen_factions = set()
    for (cy, cx) in centers:
        assert int(w.count[cy, cx, 0]) == 10
        assert int(w.strain_id[cy, cx, 0]) == bb0     # all four are the same BB0 sequence
        seen_factions.add(int(w.faction[cy, cx, 0]))
    assert seen_factions == {0, 1, 2, 3}              # one faction per center

def test_init_factions_d4_symmetric_orbit():
    # the four centers are the D4 orbit of one point. Concretely: reflecting the
    # center set about the mid-lines (rows about H/2, cols about W/2) must permute
    # the set onto itself — that is the no-faction-gets-a-geometric-edge guarantee.
    H = W = 16
    centers = {(H//4, W//4), (H//4, 3*W//4), (3*H//4, W//4), (3*H//4, 3*W//4)}
    refl_x = {(cy, (W - cx) % W) for (cy, cx) in centers}
    refl_y = {((H - cy) % H, cx) for (cy, cx) in centers}
    assert refl_x == centers and refl_y == centers
