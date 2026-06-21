# Spec §8 acceptance criteria as runnable assertions (small grid, CI-fast).
from __future__ import annotations
import inspect
import torch
from des.engine import Engine
from des.world import init_factions
from des.phenotype_cache import StrainTable

DEV = torch.device("cpu")


def test_acc1_seeding_exactly_four_cells():
    t = StrainTable()
    w = init_factions(32, 32, 64, DEV, t, fill_per_cell=20, n_fac=4)
    assert int((w.count.sum(dim=-1) > 0).sum()) == 4
    facs = w.faction[w.count > 0]
    assert set(facs.tolist()) == {0, 1, 2, 3}

def test_acc2_expansion_is_monotone_then_grows():
    # occupancy (number of non-empty cells) strictly grows over early ticks: from 4 up.
    e = Engine(H=32, W=32, K=64, seed=0, device=DEV, fill_per_cell=20)
    occ0 = int((e.world.count.sum(dim=-1) > 0).sum())
    assert occ0 == 4
    # repro_period=5 -> first expansion lands at T=5. Run 12 ticks: must exceed 4.
    e.run(12, stop_on=())
    occ1 = int((e.world.count.sum(dim=-1) > 0).sum())
    assert occ1 > occ0, f"world did not expand from 4 (got {occ1})"

def test_acc3_cross_faction_contact_causes_losses():
    # tiny grid so fronts meet fast; after enough ticks some cell holds >1 faction
    # and total count is not just monotonically growing (antagonism removes some).
    e = Engine(H=8, W=8, K=64, seed=1, device=DEV, fill_per_cell=20)
    e.run(30, stop_on=())
    # at least one cell now holds two different factions (contact happened)
    fac = e.world.faction; cnt = e.world.count
    multi = 0
    for y in range(8):
        for x in range(8):
            live = cnt[y, x] > 0
            if live.any() and torch.unique(fac[y, x][live]).numel() > 1:
                multi += 1
    assert multi > 0, "no cross-faction contact after 30 ticks on an 8x8 grid"

def test_acc4_same_faction_never_fights():
    # a whole grid of ONE faction (all same faction, mixed strains) -> antagonism
    # removes nobody, ever. Single-faction field => no kills.
    from des.kernels.antagonism import phase1_antagonism
    t = StrainTable()
    pred = t.get_or_mint(("BroadSweep",))
    prey = t.get_or_mint(("F4Nr4",))
    phe = t.phenotype_arrays(DEV)
    sid = torch.zeros((2, 2, 4), dtype=torch.int32)
    cnt = torch.zeros((2, 2, 4), dtype=torch.int32)
    fac = torch.zeros((2, 2, 4), dtype=torch.int8)   # ALL faction 0
    sid[..., 0] = pred; cnt[..., 0] = 50
    sid[..., 1] = prey; cnt[..., 1] = 50
    birth = torch.zeros((2, 2, 4), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    out = phase1_antagonism(sid, cnt, fac, phe, birth, T=5, z_max=8.0, generator=g)
    assert torch.equal(out, cnt), "same-faction field suffered antagonism losses"

def test_acc5_phenotype_path_has_no_faction_index():
    # red-line static guard: the phenotype function signature takes only `sequence`,
    # and the phenotype_arrays dict keys never include a faction axis.
    from des.registry import phenotype
    assert list(inspect.signature(phenotype).parameters) == ["sequence"]
    t = StrainTable(); t.get_or_mint(("F4Nr4",))
    arr = t.phenotype_arrays(DEV)
    assert "faction" not in arr and "fac" not in arr

def test_acc6_faction_blind_cross_seed_winrate():
    # symmetry-group sneak-goods self-check (spec §7 theorem, weak CI form): across
    # seeds, no faction should systematically dominate the EARLY expansion. Measure
    # occupied-cell share per faction at a fixed early tick; each faction's mean share
    # ~1/4. This is a coarse CI guard (the real 1/4 binomial CI check is the batch run).
    shares = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    SEEDS = 8
    for seed in range(SEEDS):
        e = Engine(H=24, W=24, K=64, seed=seed, device=DEV, fill_per_cell=20)
        e.run(10, stop_on=())
        cnt = e.world.count; fac = e.world.faction
        live = cnt > 0
        facs = fac[live]
        total = facs.numel()
        if total == 0:
            continue
        for fk in shares:
            shares[fk] += float((facs == fk).sum()) / total
    means = {k: v / SEEDS for k, v in shares.items()}
    # symmetric seeding -> each faction's mean occupied share within [0.15, 0.35] of 0.25
    for k, m in means.items():
        assert 0.15 < m < 0.35, f"faction {k} mean share {m:.3f} off symmetric 0.25: {means}"


def test_acc7_mutation_supply_scales_with_population():
    # design L243: n*p(x) individuals each mutate INDEPENDENTLY, so new-sequence supply
    # scales with population N (individuals/cell), NOT with strain count. The old code
    # minted <=1 child per parent per tick regardless of N -> this test fails under it.
    # Metric: distinct strains actually PRESENT (count>0) at a fixed early tick, averaged
    # over seeds. More individuals/cell -> more independent draws -> more of the bounded
    # neighbor set realized. Pre-saturation, this rises monotonically with fill.
    def mean_present(fill, seeds=5, T=6):
        vals = []
        for s in range(seeds):
            e = Engine(H=48, W=48, K=64, seed=s, device=DEV, fill_per_cell=fill)
            e.run(T, stop_on=())
            vals.append(e.distinct_strains())
        return sum(vals) / len(vals)

    lo, hi = mean_present(2), mean_present(48)
    assert hi > lo, (
        f"mutation supply did not scale with population: fill=2 -> {lo:.1f} present "
        f"strains, fill=48 -> {hi:.1f} (expected hi > lo per N*mu, design L243)"
    )

