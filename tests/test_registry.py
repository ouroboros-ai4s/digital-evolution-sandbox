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

def test_feature_mask_is_or_of_family_predicate_bits():
    """Post-S6: feature_mask is OR of family_<X> predicate bits, not letter bits."""
    from des.registry import PREDICATE_BIT
    p = phenotype(("F4Nr1", "BroadSweep"))
    assert p.feature_mask & PREDICATE_BIT["family_F"]
    assert p.feature_mask & PREDICATE_BIT["family_Z"]
    # no N letter, no P letter
    assert not (p.feature_mask & PREDICATE_BIT["family_N"])
    assert not (p.feature_mask & PREDICATE_BIT["family_P"])

def test_broadsweep_prey_targets_F_and_Z_families():
    """Post-S6: prey_mask is OR of family_<X> predicate bits selected by clauses."""
    from des.registry import PREDICATE_BIT
    p = phenotype(("BroadSweep",))
    assert p.prey_mask & PREDICATE_BIT["family_F"]
    assert p.prey_mask & PREDICATE_BIT["family_Z"]
    # BroadSweep's clauses select F and Z families only.
    assert not (p.prey_mask & PREDICATE_BIT["family_P"])
    assert not (p.prey_mask & PREDICATE_BIT["family_N"])

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
    """S4 重底定: F4Nr1 现在是 hash-locked 1-of-4 (而非 v1 占位 ((-1,0),)).
    断言 dir_bits 的 popcount==1, 但**不锁哪个 bit** (由 crc32 决定).
    F4Nr4 仍是 4 邻全开, 不变."""
    # F4Nr1: 1 bit set (hash-locked 单方向, 跨 strain 看似随机)
    db_f4nr1 = phenotype(("F4Nr1",)).dir_bits
    assert bin(db_f4nr1).count("1") == 1, (
        f"F4Nr1 must hash-lock to exactly 1 direction, got dir_bits={db_f4nr1:04b}")
    # F4Nr4 = all four -> all four bits (S4 不动)
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
    """S4 adds motif primitives FCLUMP / FFRONT — this test is updated to reflect
    that only non-motif letters are residue, and MOTIF_LEN covers exactly the
    motif letters in GRAN."""
    from des.registry import GRAN, MOTIF_LEN
    motif_letters = {l for l, g in GRAN.items() if g == "motif"}
    residue_letters = {l for l, g in GRAN.items() if g == "residue"}
    # Every motif letter must appear in MOTIF_LEN; residue letters must NOT.
    for letter in motif_letters:
        assert letter in MOTIF_LEN, f"{letter}: motif letter missing from MOTIF_LEN"
    for letter in residue_letters:
        assert letter not in MOTIF_LEN, f"{letter}: residue letter must not be in MOTIF_LEN"


# ---------------------------------------------------------------------------
# S1 Task 1: VIS registry table
# ---------------------------------------------------------------------------

def test_vis_covers_every_alphabet_letter():
    """Every letter in ALPHABET must have a VIS value (default 0.0 for non-N)."""
    from des.registry import VIS, ALPHABET
    for letter in ALPHABET:
        assert letter in VIS, f"{letter}: missing from VIS table"


def test_vis_values_in_unit_interval():
    """All VIS values must be in [0.0, 1.0] (spec §5)."""
    from des.registry import VIS
    for letter, v in VIS.items():
        assert 0.0 <= v <= 1.0, f"{letter}: vis {v} outside [0,1]"


def test_vis_n0_roster_value_is_0p20():
    """Spec §1: N0 vis = 0.20 (roster value, also drives vis_lowvis bit)."""
    from des.registry import VIS
    assert VIS["N0"] == 0.20


def test_vis_v1_non_N_letters_are_zero():
    """v1 non-N letters carry vis=0.0 (the only output of N primitives is vis;
    F/P/Z primitives never produce vis)."""
    from des.registry import VIS
    for letter in ("F4Nr1", "F4Nr4", "P_base", "P_hotspot", "BroadSweep"):
        assert VIS[letter] == 0.0, f"{letter}: non-N letter must have vis=0.0"


# ---------------------------------------------------------------------------
# S2 Task 1: P pool expansion (10 new P primitives)
# ---------------------------------------------------------------------------

_S2_NEW_P = (
    ("P_aic",            0.03, 3),
    ("P_ep",             0.04, 3),
    ("P_fscan",          0.02, 5),
    ("P_zscan",          0.02, 5),
    ("P_entropy_brake",  0.01, 7),
    ("P_loopswap_lite",  0.03, 4),
    ("P_neutral_sink",   0.02, 5),
    ("P_slow_drift",     0.0,  9),
    ("P_burst_lite",     0.07, 2),
    ("P_balanced",       0.04, 3),
)


def test_s2_new_P_letters_present_in_alphabet_with_family_P():
    """每个新 P 字母在 ALPHABET 里, family 全是 'P'."""
    from des.registry import ALPHABET
    for letter, _, _ in _S2_NEW_P:
        assert letter in ALPHABET, f"{letter}: missing from ALPHABET"
        assert ALPHABET[letter] == "P", f"{letter}: family must be 'P', got {ALPHABET[letter]!r}"


def test_s2_new_P_letters_have_gran_residue():
    """所有新 P 字母 gran='residue'."""
    from des.registry import GRAN
    for letter, _, _ in _S2_NEW_P:
        assert GRAN.get(letter) == "residue", f"{letter}: gran must be 'residue', got {GRAN.get(letter)!r}"


def test_s2_new_P_letters_have_exact_p_add_and_period():
    """每行 (p_add, period) 与 spec §1 表 verbatim 一致."""
    from des.registry import _P
    for letter, p_add, period in _S2_NEW_P:
        assert letter in _P, f"{letter}: missing from _P"
        assert _P[letter] == (p_add, period), (
            f"{letter}: expected (p_add={p_add}, period={period}), got {_P[letter]!r}")


def test_s2_existing_P_rows_unchanged():
    """既有 P_base / P_hotspot 两行 (p_add, period) 不变."""
    from des.registry import _P
    assert _P["P_base"] == (0.0, 1)
    assert _P["P_hotspot"] == (0.05, 3)


def test_existing_F_rows_are_7_tuple_post_s5():
    from des.registry import _F
    f4nr1 = _F["F4Nr1"]
    assert len(f4nr1) == 7
    assert f4nr1[0] == 0.30
    assert f4nr1[2] == 0.05
    assert f4nr1[3] == 4
    assert f4nr1[4] == 0.30
    assert f4nr1[5] == 1
    assert f4nr1[6] == 1
    f4nr4 = _F["F4Nr4"]
    assert len(f4nr4) == 7
    assert f4nr4[0] == 0.50
    assert f4nr4[4] == 0.50
    assert f4nr4[5] == 1
    assert f4nr4[6] == 1


def test_s5_alphabet_contains_FBURST_and_F_NOVA():
    from des.registry import ALPHABET
    assert "FBURST" in ALPHABET
    assert "F_NOVA" in ALPHABET
    assert ALPHABET["FBURST"] == "F"
    assert ALPHABET["F_NOVA"] == "F"


def test_s5_gran_covers_FBURST_and_F_NOVA():
    from des.registry import GRAN, ALPHABET
    for letter in ALPHABET:
        assert letter in GRAN, f"{letter} missing from GRAN"


# ---------------------------------------------------------------------------
# S3 Task 1: _Z_PREY_CARD module-load derivation
# ---------------------------------------------------------------------------

def test_s3_z_prey_card_module_level_derivation():
    """_Z_PREY_CARD 是 module-load 派生 (而非运行时算), key 与 _Z 同."""
    from des.registry import _Z, _Z_PREY_CARD
    assert set(_Z_PREY_CARD) == set(_Z)
    assert _Z_PREY_CARD["BroadSweep"] == 2


# ---------------------------------------------------------------------------
# S7 Task 1: SLOTS_PER_EVENT registry table
# ---------------------------------------------------------------------------

def test_s7_slots_per_event_covers_every_alphabet_letter():
    """S7: 跨 file 同款覆盖断言 (sibling to S6 test_gran_covers_every_alphabet_letter)."""
    from des.registry import SLOTS_PER_EVENT, ALPHABET
    assert set(SLOTS_PER_EVENT.keys()) == set(ALPHABET.keys())
    for letter, n in SLOTS_PER_EVENT.items():
        if letter == "P_cascade":
            assert n == 2, f"P_cascade must have slots_per_event=2 (S8)"
        else:
            assert n == 1, f"{letter}: non-P_cascade letter must have slots_per_event=1"


# ---------------------------------------------------------------------------
# S8 Task 2: A-pool rows merged into registry tables
# ---------------------------------------------------------------------------

def test_s8_a_pool_all_24_letters_in_alphabet():
    """All 24 A-pool letters merged into ALPHABET with correct family values."""
    from des.registry import ALPHABET
    from des._a_pool import A_FAMILY
    for letter, fam in A_FAMILY.items():
        assert letter in ALPHABET, f"{letter!r} missing from ALPHABET"
        assert ALPHABET[letter] == fam, (
            f"{letter!r}: ALPHABET={ALPHABET[letter]!r}, expected {fam!r}")


def test_s8_a_pool_gran_and_motif_len_merged():
    """GRAN and MOTIF_LEN updated for all 24 A letters."""
    from des.registry import GRAN, MOTIF_LEN
    from des._a_pool import A_GRAN, A_MOTIF_LEN
    for letter, gran in A_GRAN.items():
        assert GRAN.get(letter) == gran, f"{letter}: GRAN mismatch"
    for letter, ml in A_MOTIF_LEN.items():
        assert MOTIF_LEN.get(letter) == ml, f"{letter}: MOTIF_LEN mismatch"


def test_s8_a_pool_f_z_p_tables_updated():
    """_F, _Z, _P extended with all A-pool rows."""
    from des.registry import _F, _Z, _P
    from des._a_pool import A_F, A_Z, A_P
    for letter in A_F:
        assert letter in _F, f"{letter!r} missing from _F"
    for letter in A_Z:
        assert letter in _Z, f"{letter!r} missing from _Z"
    for letter in A_P:
        assert letter in _P, f"{letter!r} missing from _P"


def test_s8_p_cascade_slots_per_event_is_2():
    """P_cascade is the only letter with SLOTS_PER_EVENT=2 after S8 merge."""
    from des.registry import SLOTS_PER_EVENT
    assert SLOTS_PER_EVENT["P_cascade"] == 2
    # Spot-check a few others stay 1
    assert SLOTS_PER_EVENT["P_stutter"] == 1
    assert SLOTS_PER_EVENT["Apex Bloom"] == 1


def test_s8_slots_per_event_covers_all_alphabet_letters():
    """SLOTS_PER_EVENT key-set == ALPHABET key-set after A merge."""
    from des.registry import SLOTS_PER_EVENT, ALPHABET
    assert set(SLOTS_PER_EVENT.keys()) == set(ALPHABET.keys()), (
        f"missing={set(ALPHABET.keys()) - set(SLOTS_PER_EVENT.keys())}, "
        f"extra={set(SLOTS_PER_EVENT.keys()) - set(ALPHABET.keys())}")


def test_s8_a_pool_extreme_value_bounds_assert_at_module_load():
    """A-pool extreme values are within spec bounds (module loads without AssertionError)."""
    from des.registry import _F, _Z, _P
    from des._a_pool import A_FAMILY
    a_f = {k for k, v in A_FAMILY.items() if v == "F"}
    a_z = {k for k, v in A_FAMILY.items() if v == "Z"}
    a_p = {k for k, v in A_FAMILY.items() if v == "P"}
    for letter in a_f:
        assert _F[letter][0] <= 0.85, f"{letter}: f={_F[letter][0]} exceeds 0.85"
    for letter in a_z:
        assert _Z[letter][0] <= 1.50, f"{letter}: z={_Z[letter][0]} exceeds 1.50"
    for letter in a_p:
        assert _P[letter][0] <= 0.34, f"{letter}: p_add={_P[letter][0]} exceeds 0.34"
