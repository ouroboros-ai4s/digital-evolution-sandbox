"""S8 A pool extremes: 24 letters across F/P/Z families (no rank-4),
de-gated reachability, multi-P spectrum blend.

This file is the S8 owner test file."""
from __future__ import annotations
import pytest


_A_FAMILY_EXPECTED = {
    "Apex Bloom": "F", "Ember Drip": "F", "Bastion Pile": "F",
    "Apex Fang": "Z", "Pan Sweep": "Z",
    "Hotspot Amp": "P", "Sink Cascade": "P", "Glacial Drift": "P",
    "F_NOVA": "F", "F_TRICKLE": "F", "F_SCATTER": "F",
    "Predator Lock": "Z", "Void Bite": "Z",
    "P_cascade": "P", "P_crossclan_surge": "P", "P_frozen": "P",
    "F8Ar1": "F", "Lance Front": "F",
    "Ambush Venom": "Z", "Sweep Surge": "Z", "Nip Whisper": "Z", "Coil Null": "Z",
    "P_zscan_invert": "P", "P_stutter": "P",
}


def test_a_pool_module_exposes_24_letters_with_family_F_P_Z():
    """24 A pool letters exist; family in {F, P, Z}; no rank-4."""
    from des._a_pool import A_FAMILY
    assert set(A_FAMILY.keys()) == set(_A_FAMILY_EXPECTED.keys())
    assert len(A_FAMILY) == 24, f"expected 24 A letters, got {len(A_FAMILY)}"
    for letter, fam in A_FAMILY.items():
        assert fam in ("F", "P", "Z"), f"{letter}: family {fam!r} must be F/P/Z (no rank-4)"
        assert fam == _A_FAMILY_EXPECTED[letter], (
            f"{letter}: expected family {_A_FAMILY_EXPECTED[letter]!r}, got {fam!r}")


def test_a_pool_no_rank_4_letter_in_alphabet_value_set():
    """spec §2: A is family F/P/Z extreme values, FAMILY_RANK stays {N,F,P,Z}."""
    from des._a_pool import A_FAMILY
    assert "A" not in set(A_FAMILY.values())
    assert "S" not in set(A_FAMILY.values())
