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
