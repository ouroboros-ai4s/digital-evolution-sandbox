# tests/test_reproduction.py
import torch
from des.world import World, init_bb0
from des.kernels.reproduction import phase2_reproduce, ArrivalBuffer
from des.phenotype_cache import StrainTable
from des.registry import BB0_TEMPLATE

DEV = torch.device("cpu")

def test_arrival_buffer_accumulates():
    buf = ArrivalBuffer(DEV)
    buf.add(torch.tensor([0]), torch.tensor([1]), torch.tensor([7]),
            torch.tensor([3]), torch.tensor([2], dtype=torch.int8))
    ty, tx, sid, cnt, fac = buf.tensors()
    assert ty.tolist() == [0] and tx.tolist() == [1]
    assert sid.tolist() == [7] and cnt.tolist() == [3]
    assert fac.tolist() == [2]

def test_offspring_land_on_neighbors_not_source():
    # single occupied cell, pure F4Nr4 (4 neighbors). Offspring must arrive at the
    # 4 von-Neumann neighbors of the source, NOT back on the source cell.
    t = StrainTable()
    fid = t.get_or_mint(("F4Nr4",))
    w = World(5, 5, 64, DEV)
    w.strain_id[2, 2, 0] = fid; w.count[2, 2, 0] = 200; w.faction[2, 2, 0] = 1
    phe = t.phenotype_arrays(DEV)
    birth = torch.zeros((5, 5, 64), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    buf, live = phase2_reproduce(w, w.strain_id.clone(), w.count.clone(),
                                 w.faction.clone(), phe, t, birth, T=5, generator=g)
    ty, tx, sid, cnt, fac = buf.tensors()
    arrived = {(int(y), int(x)) for y, x, c in zip(ty, tx, cnt) if c > 0}
    neighbors = {(1, 2), (3, 2), (2, 1), (2, 3)}
    assert arrived <= neighbors, f"offspring landed off-neighbor: {arrived - neighbors}"
    assert (2, 2) not in arrived, "B4 regression: offspring deposited back on source cell"
    assert arrived, "no offspring scattered at all"

def test_offspring_inherit_parent_faction():
    t = StrainTable()
    fid = t.get_or_mint(("F4Nr4",))
    w = World(5, 5, 64, DEV)
    w.strain_id[2, 2, 0] = fid; w.count[2, 2, 0] = 200; w.faction[2, 2, 0] = 3
    phe = t.phenotype_arrays(DEV)
    birth = torch.zeros((5, 5, 64), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(1)
    buf, live = phase2_reproduce(w, w.strain_id.clone(), w.count.clone(),
                                 w.faction.clone(), phe, t, birth, T=5, generator=g)
    ty, tx, sid, cnt, fac = buf.tensors()
    assert (fac[cnt > 0] == 3).all(), "offspring did not inherit parent faction 3"

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
                                 w.faction.clone(), phe, t, birth, T=5, generator=g)
    ty, tx, sid, cnt, fac = buf.tensors()
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
    # F4Nr4's only F-period is 5, so repro_period=5 under both the pre-Task-7 min-rule
    # and Task 7's per-phase clock → 1 % 5 != 0, nobody fires. No Task-7 dependency here.
    buf, live = phase2_reproduce(w, w.strain_id.clone(), w.count.clone(),
                                 w.faction.clone(), phe, t, birth, T=1, generator=g)   # 1%5≠0
    ty, tx, sid, cnt, fac = buf.tensors()
    assert cnt.numel() == 0                     # nobody fired


# ── Fix-wave-1 tests ────────────────────────────────────────────────────────

from des.kernels.reproduction import _mutate_sequence
from des.registry import BB0_TEMPLATE, phenotype as _phenotype

_MUTABLE_SLOTS  = {0, 2, 3, 9, 10, 13}
_LOCKED_SLOTS   = {1, 5, 7}            # F4Nr1 / BroadSweep / P_base


def test_mutation_only_hits_mutable_slots():
    """Mutated positions must be a strict subset of the 6 design-mutable slots;
    backbone-locked positions {1,5,7} must NEVER change."""
    seq = BB0_TEMPLATE["layout"]
    mask = BB0_TEMPLATE["mutable"]
    spec = _phenotype(seq).spectrum
    changed = set()
    for seed in range(2000):
        g = torch.Generator(device=torch.device("cpu"))
        g.manual_seed(seed)
        mutated = _mutate_sequence(seq, mask, spec, g)
        for i, (a, b) in enumerate(zip(seq, mutated)):
            if a != b:
                changed.add(i)
    assert changed.issubset(_MUTABLE_SLOTS), (
        f"mutation hit non-mutable positions: {changed - _MUTABLE_SLOTS}"
    )
    assert _LOCKED_SLOTS.isdisjoint(changed), (
        f"mutation hit backbone-locked positions: {_LOCKED_SLOTS & changed}"
    )


def test_mutation_spreads_across_mutable_slots():
    """Over many draws, MORE THAN ONE distinct mutable slot must vary.
    This FAILS under the old idxs[0] code (always position 0)."""
    seq = BB0_TEMPLATE["layout"]
    mask = BB0_TEMPLATE["mutable"]
    spec = _phenotype(seq).spectrum
    changed = set()
    for seed in range(2000):
        g = torch.Generator(device=torch.device("cpu"))
        g.manual_seed(seed)
        mutated = _mutate_sequence(seq, mask, spec, g)
        for i, (a, b) in enumerate(zip(seq, mutated)):
            if a != b:
                changed.add(i)
    assert len(changed) >= 2, (
        f"mutation confined to positions {changed}; expected spread across ≥2 mutable slots"
    )


def test_mutation_deterministic():
    """Same seed → identical output; different seed differs at least once."""
    seq = BB0_TEMPLATE["layout"]
    mask = BB0_TEMPLATE["mutable"]
    spec = _phenotype(seq).spectrum

    results_a, results_b, results_c = [], [], []
    for seed in range(200):
        g1 = torch.Generator(device=torch.device("cpu")); g1.manual_seed(seed)
        g2 = torch.Generator(device=torch.device("cpu")); g2.manual_seed(seed)
        g3 = torch.Generator(device=torch.device("cpu")); g3.manual_seed(seed + 9999)
        results_a.append(_mutate_sequence(seq, mask, spec, g1))
        results_b.append(_mutate_sequence(seq, mask, spec, g2))
        results_c.append(_mutate_sequence(seq, mask, spec, g3))

    # same seed → identical
    assert results_a == results_b, "same-seed generators produced different sequences"
    # different seed → at least one difference
    assert results_a != results_c, "different-seed generators never differed (unexpected)"
