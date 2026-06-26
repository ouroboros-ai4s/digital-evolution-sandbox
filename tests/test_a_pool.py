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


# --- Task 3 surface: per-subpool verbatim audit -------------------------------

from des.types import IN_PLACE_DIR as _IN_PLACE_DIR

_SUB_YI1_F = (
    ("Apex Bloom",   (0.85, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.20, 4, 0.85, 1, 1)),
    ("Ember Drip",   (0.05, "hash:ember",                       0.04, 9, 0.05, 1, 1)),
    ("Bastion Pile", (0.85, (_IN_PLACE_DIR,),                   0.00, 3, 0.85, 1, 1)),
)
_SUB_YI1_Z = (
    ("Apex Fang", (1.50, (("Z", "generalist"),), 9, 0)),
    ("Pan Sweep", (0.50, (("F",), ("Z",), ("P",)), 6, 0)),
)
_SUB_YI1_P = (
    ("Hotspot Amp",   (0.30, 3)),
    ("Sink Cascade",  (0.25, 3)),
    ("Glacial Drift", (0.0,  12)),
)


def test_yi1_F_rows_verbatim():
    """乙1 F 类 3 行 (Apex Bloom / Ember Drip / Bastion Pile) verbatim."""
    from des.registry import _F
    for letter, expected in _SUB_YI1_F:
        row = _F[letter]
        assert len(row) == 7, f"{letter}: _F row must be 7-tuple (S5 shape)"
        assert row[0] == expected[0], f"{letter}: f mismatch"
        assert row[1] == expected[1], f"{letter}: dirs mismatch"
        assert row[2] == expected[2], f"{letter}: p_leave mismatch"
        assert row[3] == expected[3], f"{letter}: period mismatch"
        assert row[4] == expected[4], f"{letter}: f_lo mismatch"
        assert row[5] == expected[5], f"{letter}: burst_w mismatch"
        assert row[6] == expected[6], f"{letter}: burst_k mismatch"


def test_yi1_Z_rows_verbatim():
    """乙1 Z 类 2 行 (Apex Fang / Pan Sweep) verbatim."""
    from des.registry import _Z
    for letter, expected in _SUB_YI1_Z:
        assert _Z[letter] == expected, f"{letter}: _Z={_Z[letter]!r}, expected {expected!r}"


def test_yi1_P_rows_verbatim():
    """乙1 P 类 3 行 (Hotspot Amp / Sink Cascade / Glacial Drift) verbatim."""
    from des.registry import _P
    for letter, expected in _SUB_YI1_P:
        assert _P[letter] == expected, f"{letter}: _P={_P[letter]!r}, expected {expected!r}"


def test_yi2_F_rows_verbatim_with_F_NOVA_windowed():
    """乙2 F 类 3 行 (F_NOVA windowed / F_TRICKLE / F_SCATTER) verbatim."""
    from des.registry import _F
    expected = {
        "F_NOVA":    (0.85, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.50, 2, 0.05, 20, 1),
        "F_TRICKLE": (0.02, "hash:trickle",                     0.02, 8, 0.02, 1, 1),
        "F_SCATTER": (0.12, "hash:scatter3",                    0.60, 3, 0.12, 1, 1),
    }
    for letter, row in expected.items():
        assert _F[letter] == row, f"{letter}: _F={_F[letter]!r}, expected {row!r}"


def test_yi2_Z_rows_predator_lock_motif_void_bite_vis_weighted():
    """乙2 Z 类 2 行 (Predator Lock motif+len>=3 / Void Bite vis_weighted=1) verbatim.
    NOTE: vis_mode stored as int (0=uniform, 1=vis_weighted)."""
    from des.registry import _Z, GRAN, MOTIF_LEN
    assert _Z["Predator Lock"] == (1.45, (("Z", "motif", "len>=3"),), 9, 0)
    assert GRAN["Predator Lock"] == "motif"
    assert MOTIF_LEN["Predator Lock"] == 3
    assert _Z["Void Bite"] == (0.95, (("N", "lowvis"),), 5, 1)


def test_yi2_P_rows_cascade_crossclan_frozen_verbatim():
    """乙2 P 类 3 行 (P_cascade / P_crossclan_surge / P_frozen) verbatim."""
    from des.registry import _P, SLOTS_PER_EVENT
    assert _P["P_cascade"]         == (0.28, 2)
    assert _P["P_crossclan_surge"] == (0.20, 4)
    assert _P["P_frozen"]          == (0.0,  8)
    assert SLOTS_PER_EVENT["P_cascade"] == 2
    assert SLOTS_PER_EVENT["P_crossclan_surge"] == 1
    assert SLOTS_PER_EVENT["P_frozen"] == 1


def test_jia_F_rows_F8Ar1_random_LanceFront_hash():
    """甲 F 类 2 行 (F8Ar1 rand:1of4 / Lance Front hash:lance) verbatim."""
    from des.registry import _F
    assert _F["F8Ar1"]       == (0.25, "rand:1of4",  0.10, 2, 0.25, 1, 1)
    assert _F["Lance Front"] == (0.80, "hash:lance", 0.30, 4, 0.80, 1, 1)


def test_jia_Z_rows_ambush_sweep_nip_coil_verbatim():
    """甲 Z 类 4 行 — vis_mode as int (0=uniform, 1=vis_weighted)."""
    from des.registry import _Z
    assert _Z["Ambush Venom"] == (1.30, (("F", "motif"),),       7, 0)
    assert _Z["Sweep Surge"]  == (0.45, (("F",), ("P",)),        3, 0)
    assert _Z["Nip Whisper"]  == (0.15, (("N", "lowvis"),),      3, 1)
    assert _Z["Coil Null"]    == (0.20, (("Z",),),               8, 0)


def test_jia_P_rows_zscan_invert_stutter_verbatim():
    """甲 P 類 2 行 (P_zscan_invert F-only / P_stutter aff^4) verbatim."""
    from des.registry import _P
    assert _P["P_zscan_invert"] == (0.10, 4)
    assert _P["P_stutter"]      == (0.32, 2)


# --- De-gate audit (spec §2 + §3): n_locked overwrite gate is RETIRED ---------

def test_n_locked_is_advisory_only_not_wired_to_mutation_path():
    """spec §2: n_locked gate retired 2026-06-24. n_locked must NOT appear in
    src/des/kernels/reproduction.py — it may exist as advisory readout elsewhere."""
    import inspect
    from des.kernels import reproduction as repro_mod
    src = inspect.getsource(repro_mod)
    assert "n_locked" not in src, (
        "n_locked must NOT appear in src/des/kernels/reproduction.py — "
        "the gate was retired 2026-06-24 (spec §2). Found references.")


def test_a_letters_reachable_via_within_family_affinity_spectrum():
    """Apex Bloom (family=F) reachable from F4Nr1 spectrum (within-family,
    residue gran-match, aff=0.70). _spectrum_for('F4Nr1') must include it."""
    from des.registry import _spectrum_for
    spec = dict(_spectrum_for("F4Nr1"))
    assert "Apex Bloom" in spec, (
        "Apex Bloom must be reachable from F4Nr1 via within-family aff=0.70 "
        "spectrum (de-gate: A is a normal same-family target)")
    assert spec["Apex Bloom"] > 0.0


def test_p_letters_reachable_via_within_family_affinity_spectrum():
    """Hotspot Amp (family=P) reachable from P_base spectrum (within-family)."""
    from des.registry import _spectrum_for
    spec = dict(_spectrum_for("P_base"))
    assert "Hotspot Amp" in spec
    assert spec["Hotspot Amp"] > 0.0


# --- Relabel-invariance (spec §6) ---------------------------------------------

def test_a_family_is_structural_under_magnitude_relabel(monkeypatch):
    """spec §6: A family/gran are structural; relabelling _F/_Z/_P magnitudes
    must not change ALPHABET or GRAN."""
    from des import registry
    from des._a_pool import A_FAMILY, A_GRAN
    monkeypatch.setitem(registry._F, "Apex Bloom",
                        (0.50, ((-1, 0),), 0.99, 99, 0.50, 1, 1))
    monkeypatch.setitem(registry._Z, "Apex Fang",
                        (0.99, (("Z", "generalist"),), 99, 0))
    monkeypatch.setitem(registry._P, "Hotspot Amp", (0.01, 99))
    for letter, fam in A_FAMILY.items():
        assert registry.ALPHABET[letter] == fam
        assert registry.GRAN[letter] == A_GRAN[letter]


# --- Task 4 surface: SPECTRUM_SHAPE merge + value-domain extension ------------

def test_a_pool_shape_rows_merged():
    """8 A_SHAPE rows merged into SPECTRUM_SHAPE, verbatim from _a_pool.A_SHAPE."""
    from des.registry import SPECTRUM_SHAPE
    from des._a_pool import A_SHAPE
    for letter, row in A_SHAPE.items():
        assert letter in SPECTRUM_SHAPE, f"{letter}: missing from SPECTRUM_SHAPE"
        assert SPECTRUM_SHAPE[letter] == row, (
            f"{letter}: SPECTRUM_SHAPE={SPECTRUM_SHAPE[letter]!r}, A_SHAPE={row!r}")


def test_p_frozen_shape_power_4():
    """P_frozen aff^4 sharpening."""
    from des.registry import SPECTRUM_SHAPE
    assert SPECTRUM_SHAPE["P_frozen"] == (4.0, None, 0.0)


def test_p_stutter_shape_power_4():
    """P_stutter aff^4 sharpening."""
    from des.registry import SPECTRUM_SHAPE
    assert SPECTRUM_SHAPE["P_stutter"] == (4.0, None, 0.0)


def test_p_crossclan_surge_shape_cross_mask():
    """P_crossclan_surge cross-family jump."""
    from des.registry import SPECTRUM_SHAPE
    assert SPECTRUM_SHAPE["P_crossclan_surge"] == (1.0, "cross", 0.0)


def test_spectrum_shape_power_domain_includes_4():
    """S8 extension: power ∈ {1, 2, 3, 4}."""
    from des.registry import SPECTRUM_SHAPE
    powers = {row[0] for row in SPECTRUM_SHAPE.values()}
    assert 4.0 in powers, "power=4.0 must be present for P_frozen / P_stutter"
    for letter, (power, _, _) in SPECTRUM_SHAPE.items():
        assert power in (1.0, 2.0, 3.0, 4.0), f"{letter}: bad power {power!r}"


def test_spectrum_shape_family_mask_domain_includes_cross():
    """S8 extension: family_mask ∈ {None, F, Z, N, adjacent, cross}."""
    from des.registry import SPECTRUM_SHAPE
    masks = {row[1] for row in SPECTRUM_SHAPE.values()}
    assert "cross" in masks, "family_mask='cross' must be present for P_crossclan_surge"
    for letter, (_, mask, _) in SPECTRUM_SHAPE.items():
        assert mask in (None, "F", "Z", "N", "adjacent", "cross"), (
            f"{letter}: bad family_mask {mask!r}")
