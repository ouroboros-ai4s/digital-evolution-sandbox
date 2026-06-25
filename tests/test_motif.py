# tests/test_motif.py
"""S6 motif machinery: block decomposition, gran-matched mutation, n_locked,
predicate-bit feature/prey masks, antagonism-match invariance.

Default v1 alphabet is all-residue, so most assertions here build hand-crafted
layouts via monkeypatching GRAN/MOTIF_LEN to simulate a future motif primitive.
This is the only test file allowed to mutate registry tables — production code
never does."""
from __future__ import annotations
import pytest
from des import registry
from des.registry import motif_blocks


def test_all_residue_layout_yields_16_singletons():
    layout = ("N0",) * 16
    blocks = motif_blocks(layout)
    assert len(blocks) == 16
    for i, (s, e, ltr) in enumerate(blocks):
        assert (s, e, ltr) == (i, i + 1, "N0")


def test_default_bb0_layout_yields_16_singletons():
    """The default BB0 backbone is all-residue → motif_blocks must agree
    with the trivial decomposition."""
    blocks = motif_blocks(registry.BB0_TEMPLATE["layout"])
    assert len(blocks) == 16
    for i, (s, e, _) in enumerate(blocks):
        assert (s, e) == (i, i + 1)


def test_motif_block_groups_consecutive_repeated_motif_letter(monkeypatch):
    """Hand-craft a 16-position layout with a length-3 motif letter M3 and
    verify motif_blocks groups the 3 consecutive Ms into one block."""
    monkeypatch.setitem(registry.GRAN, "M3", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M3", 3)
    monkeypatch.setitem(registry.ALPHABET, "M3", "F")
    layout = ("N0", "M3", "M3", "M3", "N0", "BroadSweep", "N0", "P_base",
              "N0", "N0", "N0", "N0", "N0", "N0", "N0", "N0")
    blocks = motif_blocks(layout)
    # 1 N0 + 1 M3 (length 3) + 1 N0 + 1 BroadSweep + 1 N0 + 1 P_base + 8 N0 = 14
    assert len(blocks) == 14
    assert (1, 4, "M3") in blocks
    # all other positions are singletons
    for s, e, ltr in blocks:
        if ltr != "M3":
            assert e - s == 1


def test_two_separated_motif_blocks_of_same_letter_stay_separated(monkeypatch):
    """Two non-contiguous runs of the same motif letter are TWO blocks (the
    separator breaks the run)."""
    monkeypatch.setitem(registry.GRAN, "M2", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M2", 2)
    monkeypatch.setitem(registry.ALPHABET, "M2", "Z")
    layout = ("M2", "M2", "N0", "M2", "M2") + ("N0",) * 11
    blocks = motif_blocks(layout)
    m2_blocks = [b for b in blocks if b[2] == "M2"]
    assert len(m2_blocks) == 2
    assert (0, 2, "M2") in m2_blocks
    assert (3, 5, "M2") in m2_blocks


def test_spectrum_residue_path_byte_identical_to_legacy():
    """Default v1 alphabet is all-residue: for letters whose SPECTRUM_SHAPE is
    the default (1.0, None, 0.0), the spectrum must be byte-identical to the
    legacy plain-affinity formula (gran-match + renorm, no shape knobs).
    S2 letters with non-default shape (power!=1, mask!=None, or mix!=0) are
    intentional design changes and are excluded from this regression lock."""
    from des.registry import _spectrum_for, ALPHABET, SPECTRUM_SHAPE, GRAN, MOTIF_LEN
    from des.registry import affinity, ALPHABET as A
    _DEFAULT_SHAPE = (1.0, None, 0.0)

    def legacy(letter):
        src_fam = A[letter]
        src_gran = GRAN[letter]
        src_len = MOTIF_LEN.get(letter)
        weights = {}
        for t in A:
            if t == letter:
                continue
            if GRAN[t] != src_gran:
                continue
            if src_gran == "motif" and MOTIF_LEN[t] != src_len:
                continue
            weights[t] = affinity(src_fam, A[t])
        tot = sum(weights.values())
        if tot == 0:
            return ()
        return tuple((t, w / tot) for t, w in sorted(weights.items()))

    for letter in ALPHABET:
        if SPECTRUM_SHAPE.get(letter, _DEFAULT_SHAPE) != _DEFAULT_SHAPE:
            continue  # shaped P letter: divergence from legacy is by design (S2)
        assert _spectrum_for(letter) == legacy(letter), \
            f"residue spectrum changed for {letter}"


def test_spectrum_motif_excludes_cross_gran_targets(monkeypatch):
    """A motif source letter must not produce residue targets in its spectrum."""
    monkeypatch.setitem(registry.GRAN, "M2", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M2", 2)
    monkeypatch.setitem(registry.ALPHABET, "M2", "F")
    monkeypatch.setitem(registry.GRAN, "M2b", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M2b", 2)
    monkeypatch.setitem(registry.ALPHABET, "M2b", "Z")
    monkeypatch.setitem(registry.GRAN, "M3", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M3", 3)
    monkeypatch.setitem(registry.ALPHABET, "M3", "F")
    spec = registry._spectrum_for("M2")
    targets = {t for t, _ in spec}
    # No residue letters in the M2 spectrum.
    assert "F4Nr1" not in targets and "P_base" not in targets and "N0" not in targets
    # M3 excluded by equal-length predicate (different MOTIF_LEN).
    assert "M3" not in targets
    # M2b survives (same gran, same length, different family).
    assert "M2b" in targets


def test_spectrum_motif_renormalizes_to_unit_sum(monkeypatch):
    monkeypatch.setitem(registry.GRAN, "M2", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M2", 2)
    monkeypatch.setitem(registry.ALPHABET, "M2", "F")
    monkeypatch.setitem(registry.GRAN, "M2b", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M2b", 2)
    monkeypatch.setitem(registry.ALPHABET, "M2b", "Z")
    monkeypatch.setitem(registry.GRAN, "M2c", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M2c", 2)
    monkeypatch.setitem(registry.ALPHABET, "M2c", "P")
    spec = registry._spectrum_for("M2")
    total = sum(q for _, q in spec)
    assert abs(total - 1.0) < 1e-12, f"spectrum did not renormalize: total={total}"


def test_spectrum_empty_when_no_compatible_target(monkeypatch):
    """If the gran-matched + equal-length filter leaves zero candidates,
    _spectrum_for must return ()."""
    monkeypatch.setitem(registry.GRAN, "M_lonely", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M_lonely", 7)
    monkeypatch.setitem(registry.ALPHABET, "M_lonely", "F")
    spec = registry._spectrum_for("M_lonely")
    assert spec == ()


def test_mutation_outcomes_residue_only_path_byte_identical(monkeypatch):
    """The legacy single-slot overwrite must still produce identical
    (children, weights) for an all-residue parent — regression lock."""
    from des.kernels.reproduction import _mutation_outcomes
    from des.registry import _spectrum_for, BB0_TEMPLATE, motif_blocks
    seq = BB0_TEMPLATE["layout"]
    mutable = BB0_TEMPLATE["mutable"]
    # use the BB0 dominant_p ("P_base") spectrum to get a real one
    spectrum = _spectrum_for("P_base")
    blocks = motif_blocks(seq)
    children, weights = _mutation_outcomes(seq, mutable, spectrum, blocks)
    # all-residue path: outcomes count == |slots| * |spectrum|
    n_slots = sum(mutable)
    assert len(children) == n_slots * len(spectrum)
    assert abs(sum(weights) - 1.0) < 1e-9
    # every child differs from seq in exactly ONE position (singleton overwrite)
    for child in children:
        diff = sum(1 for a, b in zip(seq, child) if a != b)
        assert diff in (0, 1)  # 0 = self-loop; 1 = single residue swap


def test_mutation_outcomes_motif_slot_overwrites_whole_block(monkeypatch):
    """Hand-craft a parent containing a length-3 motif and pick a mutable
    slot inside it. The outcome must overwrite all 3 positions of the
    motif with the target letter, not just one."""
    monkeypatch.setitem(registry.GRAN, "M3a", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M3a", 3)
    monkeypatch.setitem(registry.ALPHABET, "M3a", "F")
    monkeypatch.setitem(registry.GRAN, "M3b", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M3b", 3)
    monkeypatch.setitem(registry.ALPHABET, "M3b", "Z")
    from des.kernels.reproduction import _mutation_outcomes
    from des.registry import motif_blocks
    # parent: M3a M3a M3a at indices 0,1,2; all 3 positions mutable
    seq = ("M3a", "M3a", "M3a") + ("N0",) * 13
    mutable = (True, True, True) + (False,) * 13
    spectrum = (("M3b", 1.0),)              # single target, equal length
    blocks = motif_blocks(seq)
    children, weights = _mutation_outcomes(seq, mutable, spectrum, blocks)
    # 3 mutable slots × 1 target = 3 outcomes, all identical (whole-block overwrite)
    assert len(children) == 3
    # every outcome replaces positions 0..2 with M3b
    expected = ("M3b", "M3b", "M3b") + ("N0",) * 13
    for child in children:
        assert child == expected
    # weights uniformly 1/3
    assert all(abs(w - 1 / 3) < 1e-9 for w in weights)


def test_mutation_outcomes_motif_outcome_preserves_layout_length(monkeypatch):
    """Length-fixed invariant: every outcome layout MUST be exactly 16 positions."""
    monkeypatch.setitem(registry.GRAN, "M2x", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M2x", 2)
    monkeypatch.setitem(registry.ALPHABET, "M2x", "F")
    monkeypatch.setitem(registry.GRAN, "M2y", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M2y", 2)
    monkeypatch.setitem(registry.ALPHABET, "M2y", "Z")
    from des.kernels.reproduction import _mutation_outcomes
    from des.registry import motif_blocks
    seq = ("M2x", "M2x") + ("N0",) * 14
    mutable = (True, True) + (False,) * 14
    spectrum = (("M2y", 1.0),)
    children, _ = _mutation_outcomes(seq, mutable, spectrum, motif_blocks(seq))
    for child in children:
        assert len(child) == 16


def test_n_locked_default_bb0_FPZ_counts_1_each():
    """Default BB0 has F4Nr4 at index 1, BroadSweep at index 5, P_base at index 7
    — three residue letters at locked positions → F:1 P:1 Z:1."""
    from des.registry import n_locked, BB0_TEMPLATE
    layout = BB0_TEMPLATE["layout"]
    assert n_locked(layout, "F") == 1
    assert n_locked(layout, "P") == 1
    assert n_locked(layout, "Z") == 1


def test_n_locked_rejects_N_channel():
    """N never counts (spec §3.4)."""
    from des.registry import n_locked, BB0_TEMPLATE
    with pytest.raises(ValueError):
        n_locked(BB0_TEMPLATE["layout"], "N")


def test_n_locked_rejects_unknown_channel():
    from des.registry import n_locked, BB0_TEMPLATE
    with pytest.raises(ValueError):
        n_locked(BB0_TEMPLATE["layout"], "X")


def test_n_locked_counts_motif_block_as_one(monkeypatch):
    """A locked motif of family F (length 3) counts as 1 F-block, not 3."""
    monkeypatch.setitem(registry.GRAN, "MF3", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "MF3", 3)
    monkeypatch.setitem(registry.ALPHABET, "MF3", "F")
    # Patch _LOCKED to require positions 0,1,2 to be the MF3 motif so the block
    # lies entirely inside locked positions. Original _LOCKED is restored by
    # monkeypatch's tearDown.
    monkeypatch.setattr(registry, "_LOCKED", {0: "MF3", 1: "MF3", 2: "MF3",
                                              5: "BroadSweep", 7: "P_base"})
    layout = ("MF3", "MF3", "MF3", "N0", "N0", "BroadSweep",
              "N0", "P_base") + ("N0",) * 8
    assert registry.n_locked(layout, "F") == 1
    assert registry.n_locked(layout, "P") == 1
    assert registry.n_locked(layout, "Z") == 1


def test_n_locked_excludes_block_partially_outside_locked_set(monkeypatch):
    """A motif block that straddles a locked position and a non-locked position
    is NOT a locked block — n_locked counts only blocks that lie entirely
    inside _LOCKED."""
    monkeypatch.setitem(registry.GRAN, "MF2", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "MF2", 2)
    monkeypatch.setitem(registry.ALPHABET, "MF2", "F")
    # _LOCKED only contains position 1 (not 2). The MF2 block spans 1..3 → not all locked.
    monkeypatch.setattr(registry, "_LOCKED", {1: "MF2", 5: "BroadSweep", 7: "P_base"})
    layout = ("N0", "MF2", "MF2", "N0", "N0", "BroadSweep",
              "N0", "P_base") + ("N0",) * 8
    # The single MF2 block (1, 3, 'MF2') is not fully inside _LOCKED={1,5,7}
    # → it does NOT contribute to n_locked("F"); the count is 0 for F.
    assert registry.n_locked(layout, "F") == 0
    assert registry.n_locked(layout, "Z") == 1
    assert registry.n_locked(layout, "P") == 1


def test_predicate_bits_present_and_distinct():
    """11 S6 predicates + 4 reserved (S1/S3) = 15 names; bit indices distinct."""
    from des.registry import PREDICATE_BITS
    expected_names = {
        "family_N", "family_F", "family_P", "family_Z",
        "motif_F", "motif_P", "motif_Z", "motif_N",
        "motif3_F", "motif3_P", "motif3_Z",
        "vis_lowvis", "thr_crest", "thr_hotspot", "thr_mirror",
    }
    assert set(PREDICATE_BITS.keys()) == expected_names
    indices = list(PREDICATE_BITS.values())
    assert len(indices) == len(set(indices)), "duplicate bit indices"
    for idx in indices:
        assert 0 <= idx < 63


def test_predicate_bit_is_shift_of_predicate_bits():
    from des.registry import PREDICATE_BITS, PREDICATE_BIT
    for name, idx in PREDICATE_BITS.items():
        assert PREDICATE_BIT[name] == 1 << idx


def test_predicate_vocabulary_fits_int64():
    """Module-level assertion: highest bit must fit signed int64."""
    from des.registry import PREDICATE_BITS
    assert max(PREDICATE_BITS.values()) < 63


def test_feature_mask_of_sets_family_bit_per_letter():
    """A v1 all-residue sequence sets only family_* bits + no motif_* / motif3_* bits."""
    from des.registry import feature_mask_of, PREDICATE_BIT
    seq = ("F4Nr1", "N0", "BroadSweep")
    m = feature_mask_of(seq)
    assert m & PREDICATE_BIT["family_F"]
    assert m & PREDICATE_BIT["family_N"]
    assert m & PREDICATE_BIT["family_Z"]
    # no P letter → no family_P bit set
    assert not (m & PREDICATE_BIT["family_P"])
    # no motif blocks → motif_* / motif3_* all clear
    for k in ("motif_F", "motif_P", "motif_Z", "motif_N",
              "motif3_F", "motif3_P", "motif3_Z"):
        assert not (m & PREDICATE_BIT[k]), f"unexpected {k} bit set on residue-only seq"


def test_feature_mask_of_sets_motif_and_motif3_bits(monkeypatch):
    """A length-3 motif of family F sets family_F + motif_F + motif3_F."""
    monkeypatch.setitem(registry.GRAN, "MF3", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "MF3", 3)
    monkeypatch.setitem(registry.ALPHABET, "MF3", "F")
    seq = ("MF3", "MF3", "MF3", "N0", "N0", "BroadSweep", "N0", "P_base") + ("N0",) * 8
    from des.registry import feature_mask_of, PREDICATE_BIT
    m = feature_mask_of(seq)
    assert m & PREDICATE_BIT["family_F"]
    assert m & PREDICATE_BIT["motif_F"]
    assert m & PREDICATE_BIT["motif3_F"]


def test_feature_mask_of_length2_motif_no_motif3_bit(monkeypatch):
    """A length-2 motif of family F sets motif_F but NOT motif3_F."""
    monkeypatch.setitem(registry.GRAN, "MF2", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "MF2", 2)
    monkeypatch.setitem(registry.ALPHABET, "MF2", "F")
    seq = ("MF2", "MF2") + ("N0",) * 14
    from des.registry import feature_mask_of, PREDICATE_BIT
    m = feature_mask_of(seq)
    assert m & PREDICATE_BIT["motif_F"]
    assert not (m & PREDICATE_BIT["motif3_F"])


def test_prey_mask_for_clauses_family_only_singletons():
    """v1 prey clauses are single-element family tuples; prey_mask = OR of family_* bits."""
    from des.registry import prey_mask_for_clauses, PREDICATE_BIT
    pm = prey_mask_for_clauses((("F",), ("Z",)))
    assert pm == (PREDICATE_BIT["family_F"] | PREDICATE_BIT["family_Z"])


def test_prey_mask_for_clauses_motif_clause():
    """A clause ('F', 'motif') targets motif_F bit only, not family_F."""
    from des.registry import prey_mask_for_clauses, PREDICATE_BIT
    pm = prey_mask_for_clauses((("F", "motif"),))
    assert pm == PREDICATE_BIT["motif_F"]


def test_prey_mask_for_clauses_motif3_clause():
    """A clause ('Z', 'motif', 'len>=3') targets motif3_Z bit only."""
    from des.registry import prey_mask_for_clauses, PREDICATE_BIT
    pm = prey_mask_for_clauses((("Z", "motif", "len>=3"),))
    assert pm == PREDICATE_BIT["motif3_Z"]


def test_antagonism_match_invariant_under_predicate_rewire():
    """The kernel match expression (prey_mask[i] & feature_mask[j]) != 0 must
    still pick the same (attacker, prey) pairs on the v1 alphabet — only what
    each bit means changes, not the match outcome."""
    from des.registry import phenotype
    # BroadSweep preys on F-family and Z-family. Build phenotypes for an
    # F-only prey, Z-only prey, P-only prey, N-only prey, and BroadSweep itself.
    p_bs   = phenotype(("BroadSweep",))
    p_f    = phenotype(("F4Nr1",))
    # v1 has only one Z-family primitive (BroadSweep), so Z-prey is unavoidably the same.
    # The test still verifies the predicate-bit encoding preserves the match relation.
    p_z    = phenotype(("BroadSweep",))  # same as p_bs — only Z letter in v1
    p_p    = phenotype(("P_base",))
    p_n    = phenotype(("N0",))
    # BroadSweep attacks F-prey and Z-prey, not P, not N
    assert (p_bs.prey_mask & p_f.feature_mask) != 0
    assert (p_bs.prey_mask & p_z.feature_mask) != 0
    assert (p_bs.prey_mask & p_p.feature_mask) == 0
    assert (p_bs.prey_mask & p_n.feature_mask) == 0


def test_validate_bb0_layout_all_residue_unchanged():
    """No motif letter present → validate behaves identically to pre-S6."""
    from des.registry import validate_bb0_layout, BB0_TEMPLATE
    validate_bb0_layout(BB0_TEMPLATE["layout"])   # must not raise


def test_validate_bb0_layout_broken_motif_span_raises(monkeypatch):
    """A length-3 motif placed at positions 0,1 (only 2 copies) must raise."""
    monkeypatch.setitem(registry.GRAN, "MF3", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "MF3", 3)
    monkeypatch.setitem(registry.ALPHABET, "MF3", "F")
    # Allow MF3 at the locked positions for this test by patching _LOCKED.
    monkeypatch.setattr(registry, "_LOCKED", {0: "MF3", 1: "MF3",
                                              5: "BroadSweep", 7: "P_base"})
    # Only 2 copies of MF3, but MOTIF_LEN is 3 → broken span.
    bad = ("MF3", "MF3", "N0", "N0", "N0", "BroadSweep",
           "N0", "P_base") + ("N0",) * 8
    with pytest.raises(ValueError, match="motif"):
        registry.validate_bb0_layout(bad)


def test_validate_bb0_layout_motif_correct_span_ok(monkeypatch):
    monkeypatch.setitem(registry.GRAN, "MF3", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "MF3", 3)
    monkeypatch.setitem(registry.ALPHABET, "MF3", "F")
    monkeypatch.setattr(registry, "_LOCKED", {0: "MF3", 1: "MF3", 2: "MF3",
                                              5: "BroadSweep", 7: "P_base"})
    good = ("MF3", "MF3", "MF3", "N0", "N0", "BroadSweep",
            "N0", "P_base") + ("N0",) * 8
    registry.validate_bb0_layout(good)   # must not raise


def test_relabel_invariance_motif_n_locked_feature_mask(monkeypatch):
    """Shuffle f/z/p magnitudes across letters; fix structural columns
    (gran/family/MOTIF_LEN). motif_blocks / n_locked / feature_mask must
    be byte-identical because they read structure, not magnitude.

    This is the §6 relabel-invariance audit translated into a single test."""
    from des.registry import motif_blocks, n_locked, feature_mask_of, BB0_TEMPLATE
    layout = BB0_TEMPLATE["layout"]
    pre_blocks = motif_blocks(layout)
    pre_n = (n_locked(layout, "F"), n_locked(layout, "P"), n_locked(layout, "Z"))
    pre_mask = feature_mask_of(layout)
    # mutate _F / _Z / _P magnitudes (NOT gran / NOT family / NOT MOTIF_LEN)
    monkeypatch.setitem(registry._F, "F4Nr1",
                        (0.95, ((1, 0),), 0.99, 99))                # change f, p_leave, period
    monkeypatch.setitem(registry._F, "F4Nr4",
                        (0.01, ((-1, 0),), 0.01, 1))
    monkeypatch.setitem(registry._Z, "BroadSweep",
                        (0.99, (("F",), ("Z",)), 99))                # change z, period
    monkeypatch.setitem(registry._P, "P_hotspot", (0.0, 99))
    monkeypatch.setitem(registry._P, "P_base", (0.05, 1))
    # structural readouts MUST be unchanged
    assert motif_blocks(layout) == pre_blocks
    assert (n_locked(layout, "F"), n_locked(layout, "P"), n_locked(layout, "Z")) == pre_n
    assert feature_mask_of(layout) == pre_mask
