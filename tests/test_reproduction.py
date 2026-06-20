# tests/test_reproduction.py
import torch
from des.world import World, init_bb0
from des.kernels.reproduction import phase2_reproduce, ArrivalBuffer
from des.phenotype_cache import StrainTable
from des.registry import BB0_TEMPLATE

DEV = torch.device("cpu")

def test_arrival_buffer_accumulates():
    buf = ArrivalBuffer(DEV)
    buf.add(torch.tensor([0]), torch.tensor([1]), torch.tensor([7]), torch.tensor([3]))
    ty, tx, sid, cnt = buf.tensors()
    assert ty.tolist() == [0] and tx.tolist() == [1]
    assert sid.tolist() == [7] and cnt.tolist() == [3]

def test_reproduction_scatters_to_neighbor():
    t = StrainTable()
    # a pure F4Nr4 world (dir = 4 von-Neumann neighbors, f=0.5, period 5)
    fid = t.get_or_mint(("F4Nr4",))
    w = World(4, 4, 64, DEV)
    w.strain_id[:, :, 0] = fid
    w.count[:, :, 0] = 100
    phe = t.phenotype_arrays(DEV)
    birth = torch.zeros((4, 4, 64), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    buf, live = phase2_reproduce(w, w.strain_id.clone(), w.count.clone(),
                                 phe, t, birth, T=5, generator=g)
    ty, tx, sid, cnt = buf.tensors()
    assert cnt.numel() > 0                      # something scattered
    assert (cnt > 0).all()
    # v1 mutation is rare and the seed is fixed → arrivals are dominantly the
    # parent strain. Assert the parent actually dominates (a real, falsifiable
    # check — not a tautology).
    assert cnt[sid == fid].sum() >= cnt[sid != fid].sum()

def test_no_reproduction_when_not_firing():
    t = StrainTable()
    fid = t.get_or_mint(("F4Nr4",))             # period 5
    w = World(4, 4, 64, DEV)
    w.strain_id[:, :, 0] = fid; w.count[:, :, 0] = 100
    phe = t.phenotype_arrays(DEV)
    birth = torch.zeros((4, 4, 64), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    buf, live = phase2_reproduce(w, w.strain_id.clone(), w.count.clone(),
                                 phe, t, birth, T=1, generator=g)   # 1%5≠0
    ty, tx, sid, cnt = buf.tensors()
    assert cnt.numel() == 0                     # nobody fired
