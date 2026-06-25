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
