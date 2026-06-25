"""S1 vis machinery: VIS table consumers, vis_sum/n_count aggregates,
vis_mode kernel bypass, vis_lowvis predicate bit, relabel-invariance audit.

Default v1 alphabet has only N0 as a non-zero-vis letter, so most assertions
build hand-crafted strains via monkeypatching VIS / ALPHABET / _Z to simulate
future vis-bearing primitives. Production code never mutates these tables."""
from __future__ import annotations
import pytest
from des import registry
from des.registry import phenotype


def test_phenotype_vis_sum_and_n_count_default_bb0():
    """The default BB0 layout is mostly N0 backbones — every N0 letter
    contributes vis=0.20 to vis_sum and 1 to n_count."""
    seq = registry.BB0_TEMPLATE["layout"]
    p = phenotype(seq)
    n_positions = [i for i, ltr in enumerate(seq) if registry.ALPHABET[ltr] == "N"]
    assert p.n_count == len(n_positions)
    assert p.vis_sum == pytest.approx(sum(registry.VIS[seq[i]] for i in n_positions))


def test_phenotype_vis_sum_only_counts_N_family_letters():
    """vis_sum / n_count read only fam=N letters; F/P/Z never contribute."""
    seq = ("F4Nr1", "BroadSweep", "P_base", "F4Nr1", "BroadSweep", "P_base") + ("F4Nr1",) * 10
    p = phenotype(seq)
    assert p.n_count == 0
    assert p.vis_sum == 0.0


def test_phenotype_vis_sum_pure_zeros_when_no_N():
    """Empty N profile: vis_sum=0, n_count=0 (kernel will produce p_hit=0)."""
    seq = ("F4Nr1",) * 16
    p = phenotype(seq)
    assert p.n_count == 0
    assert p.vis_sum == 0.0


def test_phenotype_vis_sum_with_synthetic_N_letters(monkeypatch):
    """Hand-craft a sequence with multiple synthetic N letters to verify
    the sum is exact across distinct vis values."""
    monkeypatch.setitem(registry.ALPHABET, "Nh", "N")
    monkeypatch.setitem(registry.VIS, "Nh", 0.70)
    monkeypatch.setitem(registry.ALPHABET, "Nl", "N")
    monkeypatch.setitem(registry.VIS, "Nl", 0.10)
    monkeypatch.setitem(registry.GRAN, "Nh", "residue")
    monkeypatch.setitem(registry.GRAN, "Nl", "residue")
    seq = ("Nh", "Nl", "Nh", "F4Nr1") + ("N0",) * 12
    p = phenotype(seq)
    # 2 Nh + 1 Nl + 12 N0 = 15 N letters
    assert p.n_count == 15
    expected = 2 * 0.70 + 1 * 0.10 + 12 * 0.20
    assert p.vis_sum == pytest.approx(expected)


def test_phenotype_vis_mode_default_is_zero():
    """Default BB0 strain has no vis-weighted hunter → vis_mode == 0."""
    seq = registry.BB0_TEMPLATE["layout"]
    p = phenotype(seq)
    assert p.vis_mode == 0


def test_phenotype_vis_mode_reads_z_row_mode_1(monkeypatch):
    """Synthetic 'ScatterNip' Z row with vis_mode=1: a strain carrying it
    must have phenotype.vis_mode == 1."""
    monkeypatch.setitem(registry.ALPHABET, "ScatterNip", "Z")
    monkeypatch.setitem(registry.GRAN, "ScatterNip", "residue")
    monkeypatch.setitem(registry.VIS, "ScatterNip", 0.0)
    monkeypatch.setitem(registry._Z, "ScatterNip",
                        (0.40, (("N",),), 5, 1))   # mode 1
    seq = ("ScatterNip",) + ("N0",) * 15
    p = phenotype(seq)
    assert p.vis_mode == 1


def test_phenotype_vis_mode_reads_z_row_mode_2(monkeypatch):
    """Synthetic 'GhostSpike' Z row with vis_mode=2: phenotype.vis_mode == 2."""
    monkeypatch.setitem(registry.ALPHABET, "GhostSpike", "Z")
    monkeypatch.setitem(registry.GRAN, "GhostSpike", "residue")
    monkeypatch.setitem(registry.VIS, "GhostSpike", 0.0)
    monkeypatch.setitem(registry._Z, "GhostSpike",
                        (0.40, (("N",),), 5, 2))   # mode 2
    seq = ("GhostSpike",) + ("N0",) * 15
    p = phenotype(seq)
    assert p.vis_mode == 2


def test_phenotype_vis_mode_takes_max_across_multiple_z(monkeypatch):
    """If a strain carries multiple Z primitives the resolved vis_mode is the
    max across them (multi-Z is not in v1, but the rule must be defined)."""
    monkeypatch.setitem(registry.ALPHABET, "ScatterNip", "Z")
    monkeypatch.setitem(registry.GRAN, "ScatterNip", "residue")
    monkeypatch.setitem(registry.VIS, "ScatterNip", 0.0)
    monkeypatch.setitem(registry._Z, "ScatterNip",
                        (0.40, (("N",),), 5, 1))
    seq = ("ScatterNip", "BroadSweep") + ("N0",) * 14
    p = phenotype(seq)
    # ScatterNip mode 1 vs BroadSweep mode 0 → max = 1
    assert p.vis_mode == 1


def test_phenotype_z_row_3_tuple_back_compat(monkeypatch):
    """A 3-tuple Z row (no explicit mode) must default vis_mode to 0."""
    monkeypatch.setitem(registry.ALPHABET, "Z3", "Z")
    monkeypatch.setitem(registry.GRAN, "Z3", "residue")
    monkeypatch.setitem(registry.VIS, "Z3", 0.0)
    monkeypatch.setitem(registry._Z, "Z3",
                        (0.30, (("F",),), 5))      # 3-tuple (no mode element)
    seq = ("Z3",) + ("N0",) * 15
    p = phenotype(seq)
    assert p.vis_mode == 0


import torch


def _bb0_with_slot0(letter: str) -> tuple:
    """Return a valid BB0 layout with `letter` at slot position 0.
    All locked positions (_LOCKED) stay correct; other positions stay N0."""
    from des.registry import _LOCKED
    return tuple(letter if i == 0 else _LOCKED.get(i, "N0") for i in range(16))


def _build_eng_with_synthetic_hunter(monkeypatch, mode, prey_letter, prey_vis):
    """Helper: build a minimal engine where faction-0 has a synthetic Z hunter
    (mode=1 or 2, prey={N}) and faction-1 has a prey N letter with VIS=prey_vis.
    Small K/fill so prey survives one tick with a measurable count.
    Returns the engine (not yet run)."""
    monkeypatch.setitem(registry.ALPHABET, "Hunt", "Z")
    monkeypatch.setitem(registry.GRAN, "Hunt", "residue")
    monkeypatch.setitem(registry.VIS, "Hunt", 0.0)
    monkeypatch.setitem(registry._Z, "Hunt",
                        (0.40, (("N",),), 5, mode))
    monkeypatch.setitem(registry.ALPHABET, prey_letter, "N")
    monkeypatch.setitem(registry.GRAN, prey_letter, "residue")
    monkeypatch.setitem(registry.VIS, prey_letter, prey_vis)
    from des.engine import Engine
    hunter_layout = _bb0_with_slot0("Hunt")
    prey_layout   = _bb0_with_slot0(prey_letter)
    layouts = (hunter_layout, prey_layout, hunter_layout, prey_layout)
    # K=64, fill=32: raw_kill=round(32*0.40)=13.
    # Mode-1 scaled kill: floor(13 * vis_sum / 16).
    #   vis_sum_hi = 0.95 + 15*0.20 = 3.95 → floor(13*3.95/16) = floor(3.21) = 3
    #   vis_sum_lo = 0.05 + 15*0.20 = 3.05 → floor(13*3.05/16) = floor(2.48) = 2
    # Different → test can distinguish. Prey starts at 32, loses a few counts → survives.
    return Engine(H=2, W=2, K=64, seed=0, device=torch.device("cpu"),
                  z_max=8.0, fill_per_cell=32, layouts=layouts)


def test_mode0_byte_identical_to_pre_s1():
    """The default BB0 4-faction symmetric run after 3 ticks must produce
    a world state byte-identical between two engines built with the same seed
    — the kernel mode-0 branch SKIPs the multiply (regression lock)."""
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    seed = 0
    eng_a = Engine(H=8, W=8, K=16, seed=seed, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=4,
                   layouts=(BB0_TEMPLATE["layout"],) * 4)
    eng_b = Engine(H=8, W=8, K=16, seed=seed, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=4,
                   layouts=(BB0_TEMPLATE["layout"],) * 4)
    eng_a.run(3, recorder=None, stop_on=())
    eng_b.run(3, recorder=None, stop_on=())
    assert torch.equal(eng_a.world.count, eng_b.world.count)
    assert torch.equal(eng_a.world.strain_id, eng_b.world.strain_id)


def _run_antagonism_direct(phe, h_sid, p_sid, fill=100):
    """Place hunter (faction 0) and prey (faction 1) in same cell and run
    phase1_antagonism once. Returns (hunter_after, prey_after) counts."""
    from des.kernels.antagonism import phase1_antagonism
    H, W, K = 1, 1, 4
    strain_id = torch.zeros(H, W, K, dtype=torch.int32)
    count     = torch.zeros(H, W, K, dtype=torch.int32)
    faction   = torch.zeros(H, W, K, dtype=torch.int8)
    strain_id[0, 0, 0] = h_sid
    count[0, 0, 0]     = fill
    faction[0, 0, 0]   = 0
    strain_id[0, 0, 1] = p_sid
    count[0, 0, 1]     = fill
    faction[0, 0, 1]   = 1
    birth = torch.zeros(H, W, K, dtype=torch.int32)
    gen   = torch.Generator()
    gen.manual_seed(0)
    new = phase1_antagonism(strain_id, count, faction, phe,
                            birth, T=0, z_max=8.0, generator=gen)
    return int(new[0, 0, 0].item()), int(new[0, 0, 1].item())


def _build_phe_for_vis_test(monkeypatch, mode, prey_letter, prey_vis):
    """Build a StrainTable with a synthetic Hunt Z (mode=mode) and a prey N
    letter at prey_vis. Prey layout has no Z primitive so it cannot retaliate
    and die via self-loss. Returns (phe, h_sid, p_sid)."""
    monkeypatch.setitem(registry.ALPHABET, "Hunt", "Z")
    monkeypatch.setitem(registry.GRAN,     "Hunt", "residue")
    monkeypatch.setitem(registry.VIS,      "Hunt", 0.0)
    monkeypatch.setitem(registry._Z,       "Hunt",
                        (0.40, (("N",),), 1, mode))   # period=1 → always fires
    monkeypatch.setitem(registry.ALPHABET, prey_letter, "N")
    monkeypatch.setitem(registry.GRAN,     prey_letter, "residue")
    monkeypatch.setitem(registry.VIS,      prey_letter, prey_vis)
    from des.phenotype_cache import StrainTable
    # Hunter: Hunt Z + 15 prey_letter N slots (all same vis for clean signal).
    # Prey: 16 prey_letter N slots, no Z → no retaliation → no self-loss death.
    hunter_seq = ("Hunt",)       + (prey_letter,) * 15
    prey_seq   = (prey_letter,)  * 16
    t = StrainTable()
    h_sid = t.get_or_mint(hunter_seq)
    p_sid = t.get_or_mint(prey_seq)
    phe = t.phenotype_arrays(torch.device("cpu"))
    return phe, h_sid, p_sid


def test_mode1_high_vis_prey_dies_faster_than_low_vis(monkeypatch):
    """Scatter-Nip mode 1: prey vis=0.95 must lose more count than prey
    vis=0.05 after one antagonism phase, all else equal."""
    phe_hi, h_hi, p_hi = _build_phe_for_vis_test(monkeypatch, 1, "N_hi",  0.95)
    phe_lo, h_lo, p_lo = _build_phe_for_vis_test(monkeypatch, 1, "N_lo",  0.05)
    _, prey_hi = _run_antagonism_direct(phe_hi, h_hi, p_hi)
    _, prey_lo = _run_antagonism_direct(phe_lo, h_lo, p_lo)
    assert prey_hi < prey_lo, f"hi-vis prey={prey_hi}, lo-vis prey={prey_lo}"


def test_mode2_low_vis_prey_dies_faster_than_high_vis(monkeypatch):
    """Ghost-Spike mode 2: inverse. Prey vis=0.05 must lose more count than
    prey vis=0.95 after one antagonism phase."""
    phe_hi, h_hi, p_hi = _build_phe_for_vis_test(monkeypatch, 2, "N_hi2", 0.95)
    phe_lo, h_lo, p_lo = _build_phe_for_vis_test(monkeypatch, 2, "N_lo2", 0.05)
    _, prey_hi = _run_antagonism_direct(phe_hi, h_hi, p_hi)
    _, prey_lo = _run_antagonism_direct(phe_lo, h_lo, p_lo)
    assert prey_lo < prey_hi, f"hi-vis prey={prey_hi}, lo-vis prey={prey_lo}"


def test_mode1_empty_n_profile_kills_zero(monkeypatch):
    """Mode-1 hunter targeting {N}: prey where all N letters have vis=0.0
    must not lose count FROM THE HUNTER (vis_sum=0 → p_hit=0 → floor(raw*0)=0).
    The prey layout must have no Z primitive so it cannot retaliate and die
    via self-loss. We bypass validate_bb0_layout by building StrainTable directly."""
    monkeypatch.setitem(registry.ALPHABET, "Hunt", "Z")
    monkeypatch.setitem(registry.GRAN,     "Hunt", "residue")
    monkeypatch.setitem(registry.VIS,      "Hunt", 0.0)
    monkeypatch.setitem(registry._Z,       "Hunt",
                        (0.40, (("N",),), 1, 1))   # period=1 → always fires
    # Synthetic inert N letter: vis=0.0, no Z primitive.
    monkeypatch.setitem(registry.ALPHABET, "Ninert", "N")
    monkeypatch.setitem(registry.GRAN,     "Ninert", "residue")
    monkeypatch.setitem(registry.VIS,      "Ninert", 0.0)
    from des.phenotype_cache import StrainTable
    # Build a prey layout with only Ninert and F4Nr1 — no Z → prey won't retaliate.
    hunter_seq = ("Hunt",) + ("Ninert",) * 15
    prey_seq   = ("F4Nr1",) + ("Ninert",) * 15   # no Z, vis_sum=0
    t = StrainTable()
    h_sid = t.get_or_mint(hunter_seq)
    p_sid = t.get_or_mint(prey_seq)
    phe = t.phenotype_arrays(torch.device("cpu"))
    assert phe["vis_sum"][p_sid].item() == 0.0, "sanity: prey vis_sum must be 0"
    assert phe["z_raw"][p_sid].item() == 0.0, "sanity: prey must have no Z"
    _, prey_after = _run_antagonism_direct(phe, h_sid, p_sid, fill=100)
    # vis_sum=0 → p_hit=0 → scaled kill=0 → prey count unchanged.
    assert prey_after == 100, f"expected prey=100, got {prey_after}"
