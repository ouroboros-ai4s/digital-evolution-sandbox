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
    sid, cnt = w.snapshot()
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
