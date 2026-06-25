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
    """Default v1 alphabet is all-residue: the spectrum for every letter must
    survive the gran-match filter exactly as before (regression lock)."""
    from des.registry import _spectrum_for, ALPHABET
    # Reproduce the legacy formula directly to compare.
    from des.registry import affinity, ALPHABET as A
    def legacy(letter):
        src_fam = A[letter]
        weights = {t: affinity(src_fam, A[t]) for t in A if t != letter}
        tot = sum(weights.values())
        if tot == 0:
            return ()
        return tuple((t, w / tot) for t, w in sorted(weights.items()))
    for letter in ALPHABET:
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


def test_mutation_outcomes_motif_outcome_preserves_layout_length():
    """Length-fixed invariant: every outcome layout MUST be exactly 16 positions."""
    import des.registry as reg
    reg.GRAN["M2x"] = "motif"; reg.MOTIF_LEN["M2x"] = 2; reg.ALPHABET["M2x"] = "F"
    reg.GRAN["M2y"] = "motif"; reg.MOTIF_LEN["M2y"] = 2; reg.ALPHABET["M2y"] = "Z"
    try:
        from des.kernels.reproduction import _mutation_outcomes
        from des.registry import motif_blocks
        seq = ("M2x", "M2x") + ("N0",) * 14
        mutable = (True, True) + (False,) * 14
        spectrum = (("M2y", 1.0),)
        children, _ = _mutation_outcomes(seq, mutable, spectrum, motif_blocks(seq))
        for child in children:
            assert len(child) == 16
    finally:
        del reg.GRAN["M2x"]; del reg.MOTIF_LEN["M2x"]; del reg.ALPHABET["M2x"]
        del reg.GRAN["M2y"]; del reg.MOTIF_LEN["M2y"]; del reg.ALPHABET["M2y"]
