# tests/test_registry.py
import pytest
from des.registry import (ALPHABET, FEATURE_BIT, affinity, phenotype,
                          BB0_TEMPLATE, MU)
from des.types import PhaseType

def test_affinity_tiers():
    assert affinity("F", "F") == 0.70
    assert affinity("N", "F") == 0.25      # adjacent ranks 0,1
    assert affinity("N", "Z") == 0.05      # cross ranks 0,3

def test_phenotype_pure_function_of_sequence():
    seq = ("F4Nr1", "N0", "BroadSweep")
    a = phenotype(seq)
    b = phenotype(seq)
    assert a == b                          # deterministic
    assert a.f == pytest.approx(0.30)      # single F letter
    assert a.z_raw == pytest.approx(0.40)  # single Z letter
    assert a.p_x == pytest.approx(MU)      # no P letter → floor μ

def test_phenotype_f_composition():
    # two F letters compose 1 - (1-0.3)(1-0.5) = 0.65
    p = phenotype(("F4Nr1", "F4Nr4"))
    assert p.f == pytest.approx(1 - 0.7 * 0.5)

def test_phenotype_p_floor():
    # no P letter still yields baseline μ
    assert phenotype(("N0",)).p_x == pytest.approx(MU)
    # a P letter raises above μ
    assert phenotype(("P_hotspot",)).p_x > MU

def test_feature_mask_is_or_of_letter_bits():
    p = phenotype(("F4Nr1", "BroadSweep"))
    assert p.feature_mask == (FEATURE_BIT["F4Nr1"] | FEATURE_BIT["BroadSweep"])

def test_broadsweep_prey_targets_F_and_Z_families():
    p = phenotype(("BroadSweep",))
    # prey_mask must include at least one F-family and one Z-family letter bit
    f_bits = FEATURE_BIT["F4Nr1"] | FEATURE_BIT["F4Nr4"]
    z_bits = FEATURE_BIT["BroadSweep"]
    assert p.prey_mask & f_bits
    assert p.prey_mask & z_bits

def test_bb0_template_shape():
    assert len(BB0_TEMPLATE["layout"]) == 16
    assert len(BB0_TEMPLATE["mutable"]) == 16
    assert sum(BB0_TEMPLATE["mutable"]) == 6           # 6 slots
    assert BB0_TEMPLATE["fold"] == (frozenset({0,2,3,4}), frozenset({9,13,15}))
    # locked functional primitives present at design positions (0-indexed)
    assert BB0_TEMPLATE["layout"][1] == "F4Nr1"
    assert BB0_TEMPLATE["layout"][5] == "BroadSweep"
    assert BB0_TEMPLATE["layout"][7] == "P_base"
