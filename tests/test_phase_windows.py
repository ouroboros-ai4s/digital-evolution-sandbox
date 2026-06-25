# tests/test_phase_windows.py
"""S5 phase-window f primitives + kernel where-on-window branch."""
from __future__ import annotations
import pytest


def test_phenotype_has_f_hi_field():
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert hasattr(p, "f_hi")
    assert isinstance(p.f_hi, float)


def test_phenotype_has_f_lo_field():
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert hasattr(p, "f_lo")
    assert isinstance(p.f_lo, float)


def test_phenotype_has_burst_w_default_one():
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert hasattr(p, "burst_w")
    assert isinstance(p.burst_w, int)
    assert p.burst_w >= 1


def test_phenotype_has_burst_k_default_one():
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert hasattr(p, "burst_k")
    assert isinstance(p.burst_k, int)
    assert p.burst_k >= 1


def test_phenotype_is_still_frozen_after_s5_fields():
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    with pytest.raises(Exception):
        p.f_hi = 0.99


def test_existing_F_rows_static_default_means_f_hi_eq_f_lo_eq_f():
    from des.registry import phenotype
    p_f4nr1 = phenotype(("F4Nr1",) + ("N0",) * 15)
    assert p_f4nr1.f_hi == p_f4nr1.f
    assert p_f4nr1.f_lo == p_f4nr1.f
    assert p_f4nr1.burst_w == 1
    assert p_f4nr1.burst_k == 1
    p_f4nr4 = phenotype(("F4Nr4",) + ("N0",) * 15)
    assert p_f4nr4.f_hi == p_f4nr4.f
    assert p_f4nr4.f_lo == p_f4nr4.f
    assert p_f4nr4.burst_w == 1
    assert p_f4nr4.burst_k == 1


def test_default_bb0_layout_phenotype_static_default():
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert p.f_hi == p.f
    assert p.f_lo == p.f
    assert p.burst_w == 1
    assert p.burst_k == 1


def test_multi_F_static_strain_stacks_f_via_one_minus_prod():
    from des.registry import phenotype
    seq = ("F4Nr4", "F4Nr1") + ("N0",) * 14
    p = phenotype(seq)
    expected_f = 1 - (1 - 0.50) * (1 - 0.30)
    assert abs(p.f - expected_f) < 1e-9
    assert p.f_hi == p.f
    assert abs(p.f_lo - expected_f) < 1e-9   # static: f_lo == f
    assert p.burst_w == 1
    assert p.burst_k == 1


def test_F_row_is_7_tuple_for_existing_letters():
    from des.registry import _F
    for letter in ("F4Nr1", "F4Nr4"):
        row = _F[letter]
        assert len(row) == 7, f"{letter}: expected 7-tuple, got len={len(row)}"
        f_val, dirs, p_leave, period, f_lo, burst_w, burst_k = row
        assert f_lo == f_val
        assert burst_w == 1
        assert burst_k == 1


def test_phenotype_f_field_is_alias_of_f_hi():
    from des.registry import phenotype
    for seq in (("F4Nr1",) + ("N0",)*15, ("F4Nr4",) + ("N0",)*15,
                ("F4Nr4", "F4Nr1") + ("N0",)*14):
        p = phenotype(seq)
        assert p.f == p.f_hi, f"seq={seq!r}: f {p.f} != f_hi {p.f_hi}"
