# tests/test_spectrum_shape.py
"""S2 shaped mutation spectrum: SPECTRUM_SHAPE table + _spectrum_for body extension.

Default v1 alphabet pre-S2 had 6 letters and only P_base / P_hotspot in _P;
S2 grows _P to 12 rows. SPECTRUM_SHAPE is co-extensive with _P — every P key
must have a shape row."""
from __future__ import annotations
import pytest
from des import registry


# --- Task 2 surface: SPECTRUM_SHAPE table -----------------------------------

_S2_EXPECTED_SHAPE = {
    "P_base":           (1.0, None,       0.0),
    "P_hotspot":        (1.0, None,       0.0),
    "P_aic":            (2.0, None,       0.0),
    "P_ep":             (1.0, None,       0.5),
    "P_fscan":          (1.0, "F",        0.0),
    "P_zscan":          (1.0, "Z",        0.0),
    "P_entropy_brake":  (3.0, None,       0.0),
    "P_loopswap_lite":  (1.0, "adjacent", 0.0),
    "P_neutral_sink":   (1.0, "N",        0.0),
    "P_slow_drift":     (1.0, None,       0.0),
    "P_burst_lite":     (1.0, None,       0.0),
    "P_balanced":       (1.0, None,       0.0),
}


def test_spectrum_shape_covers_every_P_letter():
    """SPECTRUM_SHAPE 必须覆盖全部 _P 行,key 集合相等."""
    from des.registry import SPECTRUM_SHAPE, _P
    assert set(SPECTRUM_SHAPE.keys()) == set(_P.keys())


def test_spectrum_shape_values_match_roster_verbatim():
    """每条 (power, family_mask, flatten_mix) 与 spec §1 表一致."""
    from des.registry import SPECTRUM_SHAPE
    for letter, expected in _S2_EXPECTED_SHAPE.items():
        assert SPECTRUM_SHAPE[letter] == expected, (
            f"{letter}: expected {expected!r}, got {SPECTRUM_SHAPE[letter]!r}")


def test_spectrum_shape_power_in_legal_set():
    """power ∈ {1.0, 2.0, 3.0}."""
    from des.registry import SPECTRUM_SHAPE
    for letter, (power, _, _) in SPECTRUM_SHAPE.items():
        assert power in (1.0, 2.0, 3.0), f"{letter}: bad power {power!r}"


def test_spectrum_shape_family_mask_in_legal_set():
    """family_mask ∈ {None, 'F', 'Z', 'N', 'adjacent'}."""
    from des.registry import SPECTRUM_SHAPE
    for letter, (_, mask, _) in SPECTRUM_SHAPE.items():
        assert mask in (None, "F", "Z", "N", "adjacent"), (
            f"{letter}: bad family_mask {mask!r}")


def test_spectrum_shape_flatten_mix_in_unit_interval():
    """flatten_mix ∈ [0.0, 1.0]."""
    from des.registry import SPECTRUM_SHAPE
    for letter, (_, _, mix) in SPECTRUM_SHAPE.items():
        assert 0.0 <= mix <= 1.0, f"{letter}: flatten_mix {mix!r} outside [0,1]"
