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
