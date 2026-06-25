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
