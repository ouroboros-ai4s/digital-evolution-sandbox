# tests/test_multi_p_blend.py
"""S8 multi-P spectrum blend: blend_p_spectra(pairs) -> spectrum.

Implements spec §4.1: spectrum(t) = Σ_i p_add_i · q_i(t) / Σ_i p_add_i,
or — when Σ p_add_i == 0 — equal-weight degenerate path Σ q_i / N_p.

Single-letter path is the dominant_p identity: blend([(p, q)]) == q
byte-equal. Default BB0 strain has dominant_p='P_base' → single-letter
path → byte-identical pre-S8."""
from __future__ import annotations
import pytest


def test_blend_empty_pairs_returns_empty():
    """No P letter ⇒ spectrum = ()."""
    from des.registry import blend_p_spectra
    assert blend_p_spectra(()) == ()


def test_blend_single_letter_is_identity():
    """Σ has one term ⇒ blend == q itself, byte-equal."""
    from des.registry import blend_p_spectra
    q = (("F4Nr1", 0.6), ("F4Nr4", 0.4))
    assert blend_p_spectra(((0.05, q),)) == q
    assert blend_p_spectra(((0.0, q),))  == q
    assert blend_p_spectra(((0.30, q),)) == q


def test_blend_two_letters_weighted_per_target():
    """spectrum(t) = (p1·q1(t) + p2·q2(t)) / (p1 + p2).
    p1=0.06 q1={(A:0.7,B:0.3)}, p2=0.04 q2={(A:0.2,B:0.8)}.
    A = (0.042+0.008)/0.10 = 0.50; B = (0.018+0.032)/0.10 = 0.50."""
    from des.registry import blend_p_spectra
    q1 = (("A", 0.7), ("B", 0.3))
    q2 = (("A", 0.2), ("B", 0.8))
    blended = dict(blend_p_spectra(((0.06, q1), (0.04, q2))))
    assert abs(blended["A"] - 0.50) < 1e-9
    assert abs(blended["B"] - 0.50) < 1e-9
    assert abs(sum(blended.values()) - 1.0) < 1e-9


def test_blend_two_letters_weighted_asymmetric():
    """Heavier letter dominates the blend."""
    from des.registry import blend_p_spectra
    q1 = (("A", 1.0), ("B", 0.0))
    q2 = (("A", 0.0), ("B", 1.0))
    blended = dict(blend_p_spectra(((0.10, q1), (0.02, q2))))
    assert abs(blended["A"] - 10/12) < 1e-9
    assert abs(blended["B"] - 2/12) < 1e-9


def test_blend_target_only_in_one_letter():
    """Target appearing in only one letter still gets that letter's weighted contribution."""
    from des.registry import blend_p_spectra
    q1 = (("A", 0.6), ("B", 0.4))
    q2 = (("A", 0.5), ("C", 0.5))
    blended = dict(blend_p_spectra(((0.06, q1), (0.04, q2))))
    assert abs(blended["B"] - 0.24) < 1e-9
    assert abs(blended["C"] - 0.20) < 1e-9
    assert abs(blended["A"] - 0.56) < 1e-9
    assert abs(sum(blended.values()) - 1.0) < 1e-9


def test_blend_sum_p_add_zero_uses_equal_weight_degenerate_path():
    """Σ p_add == 0 ⇒ equal-weight average (avoid div-by-zero)."""
    from des.registry import blend_p_spectra
    q1 = (("A", 0.8), ("B", 0.2))
    q2 = (("A", 0.4), ("B", 0.6))
    blended = dict(blend_p_spectra(((0.0, q1), (0.0, q2))))
    assert abs(blended["A"] - 0.6) < 1e-9
    assert abs(blended["B"] - 0.4) < 1e-9
    assert abs(sum(blended.values()) - 1.0) < 1e-9


def test_blend_single_letter_with_p_add_zero_still_identity():
    """Σ=0 + single letter ⇒ single-letter identity path (equal-weight of 1 = q)."""
    from des.registry import blend_p_spectra
    q = (("A", 0.7), ("B", 0.3))
    assert blend_p_spectra(((0.0, q),)) == q


def test_blend_output_sorted_by_target_name():
    """Output is sorted by target name (same canonical order as _spectrum_for)."""
    from des.registry import blend_p_spectra
    q1 = (("Apple", 0.5), ("Zebra", 0.5))
    q2 = (("Banana", 1.0),)
    blended = blend_p_spectra(((0.05, q1), (0.03, q2)))
    names = [t for t, _ in blended]
    assert names == sorted(names)


def test_blend_handles_empty_q_in_one_letter():
    """One letter's q is () — skip its contribution and renormalize survivors to Σ=1."""
    from des.registry import blend_p_spectra
    q1 = (("A", 0.6), ("B", 0.4))
    q2 = ()
    blended = blend_p_spectra(((0.05, q1), (0.03, q2)))
    blended_d = dict(blended)
    # q2 contributes nothing; result is essentially q1 renormalized (already Σ=1)
    assert abs(blended_d["A"] - 0.6) < 1e-9
    assert abs(blended_d["B"] - 0.4) < 1e-9
    assert abs(sum(blended_d.values()) - 1.0) < 1e-9
