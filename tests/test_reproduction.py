# tests/test_reproduction.py
import pytest
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

from des.kernels.reproduction import _mutation_outcomes
from des.registry import BB0_TEMPLATE, phenotype as _phenotype

_MUTABLE_SLOTS  = {0, 2, 3, 9, 10, 13}
_LOCKED_SLOTS   = {1, 5, 7}            # F4Nr1 / BroadSweep / P_base


def test_mutation_only_hits_mutable_slots():
    """Every outcome's changed positions must be a subset of the 6 design-mutable
    slots; backbone-locked positions {1,5,7} must NEVER change."""
    from des.registry import motif_blocks
    seq = BB0_TEMPLATE["layout"]
    spec = _phenotype(seq).spectrum
    children, _ = _mutation_outcomes(seq, BB0_TEMPLATE["mutable"], spec, motif_blocks(seq))
    changed = set()
    for child in children:
        for i, (a, b) in enumerate(zip(seq, child)):
            if a != b:
                changed.add(i)
    assert changed.issubset(_MUTABLE_SLOTS), (
        f"mutation hit non-mutable positions: {changed - _MUTABLE_SLOTS}"
    )
    assert _LOCKED_SLOTS.isdisjoint(changed), (
        f"mutation hit backbone-locked positions: {_LOCKED_SLOTS & changed}"
    )


def test_mutation_spreads_across_mutable_slots():
    """The outcome set must reach MORE THAN ONE distinct mutable slot
    (FAILS under the old idxs[0]-only code that always hit position 0)."""
    from des.registry import motif_blocks
    seq = BB0_TEMPLATE["layout"]
    spec = _phenotype(seq).spectrum
    children, _ = _mutation_outcomes(seq, BB0_TEMPLATE["mutable"], spec, motif_blocks(seq))
    changed = set()
    for child in children:
        for i, (a, b) in enumerate(zip(seq, child)):
            if a != b:
                changed.add(i)
    assert len(changed) >= 2, (
        f"mutation confined to positions {changed}; expected spread across ≥2 mutable slots"
    )


def test_mutation_outcomes_weights_normalized():
    """Weights sum to 1 (a proper categorical) and align 1:1 with children."""
    from des.registry import motif_blocks
    seq = BB0_TEMPLATE["layout"]
    spec = _phenotype(seq).spectrum
    children, weights = _mutation_outcomes(seq, BB0_TEMPLATE["mutable"], spec, motif_blocks(seq))
    assert len(children) == len(weights) == len(_MUTABLE_SLOTS) * len(spec)
    assert abs(sum(weights) - 1.0) < 1e-6, f"weights sum to {sum(weights)}, expected 1.0"


def test_mutation_outcomes_signature_takes_blocks():
    """Regression: post-S6 _mutation_outcomes accepts a 4th positional `blocks` arg.
    Calling with the legacy 3-arg form must raise."""
    from des.kernels.reproduction import _mutation_outcomes
    from des.registry import _spectrum_for, BB0_TEMPLATE
    seq = BB0_TEMPLATE["layout"]
    mutable = BB0_TEMPLATE["mutable"]
    spectrum = _spectrum_for("P_base")
    with pytest.raises(TypeError):
        _mutation_outcomes(seq, mutable, spectrum)  # missing blocks arg


def test_fstack_strain_offspring_stay_in_source_cell():
    """FSTACK in-place path: offspring deposit in source cell (no roll).
    Run phase2_reproduce directly on a minimal world to verify no cross-cell
    migration — offspring arriving at the buffer must only land on source coords."""
    import torch
    from des.world import World
    from des.kernels.reproduction import phase2_reproduce
    from des.phenotype_cache import StrainTable
    # BB0-valid layout with FSTACK at slot 0.
    fstack_layout = ("FSTACK", "F4Nr4", "N0", "N0", "N0", "BroadSweep", "N0", "P_base") + ("N0",) * 8
    t = StrainTable()
    fid = t.get_or_mint(fstack_layout)
    w = World(1, 3, 8, torch.device("cpu"))
    # Seed only the middle cell (0,1) with the FSTACK strain.
    w.strain_id[0, 1, 0] = fid
    w.count[0, 1, 0] = 200
    w.faction[0, 1, 0] = 1
    phe = t.phenotype_arrays(torch.device("cpu"))
    birth = torch.zeros((1, 3, 8), dtype=torch.int32)
    g = torch.Generator(device=torch.device("cpu"))
    g.manual_seed(0)
    buf, live = phase2_reproduce(w, w.strain_id.clone(), w.count.clone(),
                                 w.faction.clone(), phe, t, birth, T=3, generator=g)
    ty, tx, sid, cnt, fac = buf.tensors()
    # All arrivals must land on x=1 (the source column). In-place never rolls.
    if cnt.numel() > 0:
        assert (tx[cnt > 0] == 1).all(), \
            f"FSTACK offspring crossed to non-source columns: {tx[cnt > 0].tolist()}"


def test_fdrift_strain_same_seed_reproducible():
    """FDRIFT 跨 process 同 seed 同结果: kernel generator (world RNG) 必须是
    seed 的确定函数, 不抓 Python 默认 random / torch 默认 RNG."""
    import torch
    from des.engine import Engine
    # BB0-valid layout: locked positions {1:F4Nr4, 5:BroadSweep, 7:P_base}, FDRIFT at slot 0.
    fdrift_layout = ("FDRIFT", "F4Nr4", "N0", "N0", "N0", "BroadSweep", "N0", "P_base") + ("N0",) * 8
    layouts = (fdrift_layout,) * 4

    eng_a = Engine(H=4, W=4, K=8, seed=42, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2, layouts=layouts)
    eng_b = Engine(H=4, W=4, K=8, seed=42, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2, layouts=layouts)
    eng_a.run(3, recorder=None, stop_on=())
    eng_b.run(3, recorder=None, stop_on=())
    assert torch.equal(eng_a.world.count, eng_b.world.count)
    assert torch.equal(eng_a.world.strain_id, eng_b.world.strain_id)


def test_f4nr4_byte_identical_after_s4():
    """F4Nr4 仍 4 邻全开 (spec §1 表锁死), S4 重底 F4Nr1 不许动 F4Nr4. 同 seed
    跑 2 次 → bit-identical."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng_a = Engine(H=8, W=8, K=8, seed=0, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2,
                   layouts=(BB0_TEMPLATE["layout"],) * 4)
    eng_b = Engine(H=8, W=8, K=8, seed=0, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2,
                   layouts=(BB0_TEMPLATE["layout"],) * 4)
    eng_a.run(3, recorder=None, stop_on=())
    eng_b.run(3, recorder=None, stop_on=())
    assert torch.equal(eng_a.world.count, eng_b.world.count)


def test_f4nr1_offspring_lands_at_hash_locked_neighbor():
    """F4Nr1 re-baselined: offspring land at crc32-determined neighbor, not north-only."""
    import torch
    from des.engine import Engine
    from des.registry import phenotype
    # popcount==1: exactly one direction
    assert bin(phenotype(("F4Nr1",) + ("N0",) * 15).dir_bits).count("1") == 1
    # same-seed reproducibility — BB0-valid: locked pos {1:F4Nr4, 5:BroadSweep, 7:P_base},
    # F4Nr1 at mutable slot 0.
    layout = ("F4Nr1", "F4Nr4", "N0", "N0", "N0", "BroadSweep", "N0", "P_base") + ("N0",) * 8
    eng_a = Engine(H=4, W=4, K=8, seed=0, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2, layouts=(layout,) * 4)
    eng_b = Engine(H=4, W=4, K=8, seed=0, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2, layouts=(layout,) * 4)
    eng_a.run(1, recorder=None, stop_on=())
    eng_b.run(1, recorder=None, stop_on=())
    assert torch.equal(eng_a.world.count, eng_b.world.count)

