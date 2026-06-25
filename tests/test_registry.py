# tests/test_registry.py
import pytest
from des.registry import (ALPHABET, FEATURE_BIT, affinity, phenotype,
                          BB0_TEMPLATE, MU, _spectrum_for)
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
    assert BB0_TEMPLATE["layout"][1] == "F4Nr4"   # B3 fix: 4-dir expansion (was F4Nr1, north-only)
    assert BB0_TEMPLATE["layout"][5] == "BroadSweep"
    assert BB0_TEMPLATE["layout"][7] == "P_base"


# ---------------------------------------------------------------------------
# Fix-wave-1 tests
# ---------------------------------------------------------------------------

def test_pmax_named_constant():
    """P_MAX must be promoted to a named module constant at 0.08."""
    from des import registry
    assert registry.P_MAX == 0.08
    # Cap inertness check: P_hotspot contributes p_add=0.05, so
    # MU + p_add = 0.01 + 0.05 = 0.06 < P_MAX=0.08 → cap never fires in v1.
    # The compound p_x for a single P_hotspot must equal max(MU, 1-(1-0.06)) = 0.06.
    p = phenotype(("P_hotspot",))
    expected_p_x = max(MU, 1 - (1 - min(registry.P_MAX, MU + 0.05)))
    assert p.p_x == pytest.approx(expected_p_x)


def test_dominant_p_is_position_independent():
    """Two sequences that are the same multiset but differ in slot placement
    must produce identical spectra — last-seen order must NOT matter (red test)."""
    # BB0-style 16-letter sequences: locked P_base at index 7, P_hotspot at either
    # low slot (index 3) or high slot (index 13), everything else N0.
    # Under last-seen the HIGH-index version always uses P_hotspot (last seen) and
    # the LOW-index version might use P_base (last seen after P_hotspot).
    # Under max-p_add both must resolve to P_hotspot (higher p_add=0.05 > 0.0).
    seq_low = tuple(
        "P_hotspot" if i == 3
        else "P_base" if i == 7
        else "N0"
        for i in range(16)
    )
    seq_high = tuple(
        "P_base" if i == 7
        else "P_hotspot" if i == 13
        else "N0"
        for i in range(16)
    )
    ph_low = phenotype(seq_low)
    ph_high = phenotype(seq_high)
    assert ph_low.spectrum == ph_high.spectrum, (
        f"spectrum differs by slot position: low={ph_low.spectrum} high={ph_high.spectrum}"
    )


def test_dominant_p_picks_higher_padd():
    """A sequence containing both P_base (p_add=0.0) and P_hotspot (p_add=0.05)
    must resolve to P_hotspot's spectrum, not P_base's."""
    p = phenotype(("P_base", "P_hotspot"))
    assert p.spectrum == _spectrum_for("P_hotspot")
    assert p.spectrum != _spectrum_for("P_base")


# ---------------------------------------------------------------------------
# Task-5 tests: dir_bits + per-phase periods
# ---------------------------------------------------------------------------

from des.registry import ALL_DIRECTIONS

def test_dir_bits_match_directions():
    # F4Nr1 = north only -> exactly the bit for (-1,0)
    north_bit = 1 << ALL_DIRECTIONS.index((-1, 0))
    assert phenotype(("F4Nr1",)).dir_bits == north_bit
    # F4Nr4 = all four -> all four bits
    assert phenotype(("F4Nr4",)).dir_bits == (1 << len(ALL_DIRECTIONS)) - 1
    # no F letter -> no directions -> 0
    assert phenotype(("N0",)).dir_bits == 0

def test_per_phase_periods_split():
    # BB0 has F4Nr4 (F, period 5), BroadSweep (Z, period 5), P_base (P, period 1).
    # OLD period = min(5,5,1) = 1.  NEW: repro_period = 5 (F only), anta_period = 5 (Z only).
    ph = phenotype(BB0_TEMPLATE["layout"])
    assert ph.period == 1            # old min-over-all unchanged (back-compat)
    assert ph.repro_period == 5      # F-primitive period
    assert ph.anta_period == 5       # Z-primitive period
    # a strain with no F letter -> repro_period defaults to 1
    assert phenotype(("BroadSweep",)).repro_period == 1
    # a strain with no Z letter -> anta_period defaults to 1
    assert phenotype(("F4Nr4",)).anta_period == 1


# ---------------------------------------------------------------------------
# S6 Task 1: GRAN / MOTIF_LEN tables
# ---------------------------------------------------------------------------

def test_gran_covers_every_alphabet_letter():
    """GRAN must have one entry per letter in ALPHABET, value in {residue, motif}."""
    from des.registry import GRAN, ALPHABET
    assert set(GRAN.keys()) == set(ALPHABET.keys())
    for letter, gran in GRAN.items():
        assert gran in ("residue", "motif"), f"{letter}: bad gran {gran!r}"


def test_v1_alphabet_is_all_residue_motif_len_empty():
    """v1 has no motif primitives yet — every letter is residue, MOTIF_LEN is empty."""
    from des.registry import GRAN, MOTIF_LEN
    for letter, gran in GRAN.items():
        assert gran == "residue", f"{letter}: v1 must be residue, got {gran!r}"
    assert MOTIF_LEN == {}, f"v1 has no motif primitives, got MOTIF_LEN={MOTIF_LEN!r}"
