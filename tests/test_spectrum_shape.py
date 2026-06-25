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


# --- Task 3 surface: _spectrum_for body ------------------------------------

def _spectrum_dict(letter):
    """Helper: convert _spectrum_for (target, q) sequence to dict."""
    from des.registry import _spectrum_for
    return dict(_spectrum_for(letter))


def test_p_aic_is_sharper_than_p_base():
    """power=2 sharpen: P_aic total P-family mass > P_base total P-family mass."""
    base = _spectrum_dict("P_base")
    aic  = _spectrum_dict("P_aic")
    # Sum mass over all P-family targets present in each spectrum independently.
    base_mass = sum(q for t, q in base.items() if registry.ALPHABET[t] == "P")
    aic_mass  = sum(q for t, q in aic.items()  if registry.ALPHABET[t] == "P")
    assert base_mass > 0, "no P-family target in P_base spectrum"
    assert aic_mass > base_mass + 1e-9, (
        f"P_aic mass on P-family ({aic_mass:.4f}) must exceed P_base ({base_mass:.4f})")


def test_p_entropy_brake_is_sharper_than_p_aic():
    """power=3 more concentrated than power=2."""
    aic   = _spectrum_dict("P_aic")
    brake = _spectrum_dict("P_entropy_brake")
    aic_mass   = sum(q for t, q in aic.items()   if registry.ALPHABET[t] == "P")
    brake_mass = sum(q for t, q in brake.items() if registry.ALPHABET[t] == "P")
    assert brake_mass > aic_mass + 1e-9, (
        f"P_entropy_brake mass ({brake_mass:.4f}) must exceed P_aic ({aic_mass:.4f})")


def test_p_ep_is_flatter_than_p_base_on_dominant_target():
    """flatten_mix=0.5: P_ep P-family mass < P_base P-family mass."""
    base = _spectrum_dict("P_base")
    ep   = _spectrum_dict("P_ep")
    same_family_targets = [t for t in base if registry.ALPHABET[t] == "P"]
    base_mass = sum(base[t] for t in same_family_targets)
    ep_mass   = sum(ep[t]   for t in same_family_targets if t in ep)
    assert ep_mass < base_mass - 1e-9, (
        f"P_ep mass on P-family ({ep_mass:.4f}) must be < P_base ({base_mass:.4f})")


def test_p_fscan_mass_only_on_F_targets():
    """family_mask='F': all mass on F-family targets only."""
    spec = _spectrum_dict("P_fscan")
    non_f_mass = sum(q for t, q in spec.items() if registry.ALPHABET[t] != "F")
    f_targets = [t for t in spec if registry.ALPHABET[t] == "F"]
    assert f_targets, "P_fscan should keep at least one F target in 16-letter alphabet"
    assert non_f_mass == 0.0, f"P_fscan leaked mass to non-F targets: {non_f_mass}"
    assert abs(sum(spec[t] for t in f_targets) - 1.0) < 1e-9


def test_p_zscan_mass_only_on_Z_targets():
    """family_mask='Z': all mass on Z-family targets only."""
    spec = _spectrum_dict("P_zscan")
    non_z = sum(q for t, q in spec.items() if registry.ALPHABET[t] != "Z")
    assert non_z == 0.0
    z_targets = [t for t in spec if registry.ALPHABET[t] == "Z"]
    if z_targets:
        assert abs(sum(spec[t] for t in z_targets) - 1.0) < 1e-9


def test_p_neutral_sink_mass_only_on_N_targets():
    """family_mask='N': all mass on N-family targets only."""
    spec = _spectrum_dict("P_neutral_sink")
    non_n = sum(q for t, q in spec.items() if registry.ALPHABET[t] != "N")
    assert non_n == 0.0
    n_targets = [t for t in spec if registry.ALPHABET[t] == "N"]
    if n_targets:
        assert abs(sum(spec[t] for t in n_targets) - 1.0) < 1e-9


def test_p_loopswap_lite_mass_only_on_adjacent_rank():
    """family_mask='adjacent': mass only on |Δrank|=1 targets."""
    from des.types import FAMILY_RANK
    src_rank = FAMILY_RANK["P"]
    spec = _spectrum_dict("P_loopswap_lite")
    bad = [t for t, q in spec.items()
           if q > 0 and abs(FAMILY_RANK[registry.ALPHABET[t]] - src_rank) != 1]
    assert not bad, f"P_loopswap_lite leaked mass to non-adjacent targets: {bad}"


def test_burst_lite_slow_drift_balanced_share_p_base_shape():
    """P_burst_lite / P_slow_drift / P_balanced share P_base shape (same knobs).
    Each spectrum excludes itself, so the target key sets differ by one entry —
    compare by dropping each letter's self-key then checking weights are equal."""
    base = _spectrum_dict("P_base")
    for letter in ("P_burst_lite", "P_slow_drift", "P_balanced"):
        spec = _spectrum_dict(letter)
        # Remove the self-exclusion key from each side for comparison.
        base_stripped = {t: q for t, q in base.items() if t != letter}
        spec_stripped = {t: q for t, q in spec.items() if t != "P_base"}
        assert set(base_stripped) == set(spec_stripped), (
            f"{letter}: target key sets differ from P_base after stripping self-keys")
        for t in base_stripped:
            assert abs(base_stripped[t] - spec_stripped[t]) < 1e-9, (
                f"{letter} weight for {t!r}: {spec_stripped[t]:.6f} != P_base {base_stripped[t]:.6f}")


def test_spectrum_normalizes_to_unit_sum_for_every_P_letter():
    """Every P row normalizes to Σq=1 (or returns empty ())."""
    from des.registry import _spectrum_for, _P
    for letter in _P:
        spec = _spectrum_for(letter)
        if spec == ():
            continue
        total = sum(q for _, q in spec)
        assert abs(total - 1.0) < 1e-9, f"{letter}: Σq={total} != 1"


def test_p_fscan_returns_empty_when_no_F_letter_in_alphabet(monkeypatch):
    """family_mask='F' + no F in alphabet → tot=0 → returns ()."""
    minimal = {k: v for k, v in registry.ALPHABET.items() if v not in ("F", "Z")}
    monkeypatch.setattr(registry, "ALPHABET", minimal)
    minimal_gran = {k: v for k, v in registry.GRAN.items() if k in minimal}
    monkeypatch.setattr(registry, "GRAN", minimal_gran)
    from des.registry import _spectrum_for
    assert _spectrum_for("P_fscan") == ()


def test_p_ep_flatten_uses_full_alphabet_size_in_denominator():
    """flatten 1/(|A|-1) uses len(ALPHABET)=16, not a hardcoded constant."""
    from des.registry import _spectrum_for, ALPHABET, SPECTRUM_SHAPE, affinity, GRAN
    A = len(ALPHABET)
    src_fam = ALPHABET["P_ep"]
    power, mask, mix = SPECTRUM_SHAPE["P_ep"]
    raw = {}
    for t in ALPHABET:
        if t == "P_ep":
            continue
        if GRAN[t] != "residue":
            continue
        w = affinity(src_fam, ALPHABET[t]) ** power
        w = (1 - mix) * w + mix * (1.0 / (A - 1))
        raw[t] = w
    tot = sum(raw.values())
    expected = {t: raw[t] / tot for t in raw}
    actual = dict(_spectrum_for("P_ep"))
    assert set(actual) == set(expected)
    for t in expected:
        assert abs(actual[t] - expected[t]) < 1e-9, (
            f"P_ep target {t}: expected {expected[t]:.6f}, got {actual[t]:.6f}")
