"""S3 rich-prey predicates: thr_crest / thr_hotspot / thr_mirror feature bits
+ 4 new prey-clause tags ("f_hi" / "p_hi" / "generalist" / "lowvis").

This file is the S3 owner test file (sibling: tests/test_motif.py /
test_vis.py / test_spectrum_shape.py / test_direction_kinds.py /
test_phase_windows.py / test_hash_dirs.py). vis_lowvis bit (S6 reserved,
S1 owner-filled) is tested in tests/test_vis.py — S3 only audits it
in Task 5 to make sure S1's behavior is still live."""
from __future__ import annotations
import pytest
from des import registry


def test_z_prey_card_exists_and_covers_every_Z_row():
    """_Z_PREY_CARD 必须覆盖 _Z 的全部 key, 与 _Z.keys() 同集合."""
    from des.registry import _Z, _Z_PREY_CARD
    assert set(_Z_PREY_CARD.keys()) == set(_Z.keys()), (
        f"_Z_PREY_CARD keys {set(_Z_PREY_CARD)} != _Z keys {set(_Z)}")


def test_z_prey_card_broadsweep_is_two():
    """BroadSweep prey_clauses = (("F",), ("Z",)) → cardinality 2."""
    from des.registry import _Z_PREY_CARD
    assert _Z_PREY_CARD["BroadSweep"] == 2


def test_z_prey_card_values_match_len_of_prey_clauses():
    """每行 _Z_PREY_CARD[name] 必须 == len(_Z[name][1])."""
    from des.registry import _Z, _Z_PREY_CARD
    for name, row in _Z.items():
        clauses = row[1]
        assert _Z_PREY_CARD[name] == len(clauses), (
            f"{name}: card={_Z_PREY_CARD[name]} but len(prey_clauses)={len(clauses)}")


# ---------------------------------------------------------------------------
# Task 2: feature_mask_of 阈值 bit
# ---------------------------------------------------------------------------

def test_thr_crest_hits_on_f4nr4_at_boundary_0p50():
    """F4Nr4 f=0.50 触发 thr_crest (闭区间 ≥0.5)."""
    from des.registry import feature_mask_of, PREDICATE_BIT
    seq = ("F4Nr4",) + ("N0",) * 15
    m = feature_mask_of(seq)
    assert m & PREDICATE_BIT["thr_crest"], (
        "F4Nr4 (f=0.50) must SET thr_crest bit (boundary >= 0.5)")


def test_thr_crest_misses_on_f4nr1_below_threshold():
    """F4Nr1 f=0.30 < 0.5 → thr_crest CLEAR (假设 strain 不含其他 F)."""
    from des.registry import feature_mask_of, PREDICATE_BIT
    seq = ("F4Nr1",) + ("N0",) * 15
    m = feature_mask_of(seq)
    assert not (m & PREDICATE_BIT["thr_crest"]), (
        "F4Nr1 (f=0.30) must NOT set thr_crest bit")


def test_thr_crest_set_if_any_F_letter_meets_threshold():
    """同 strain 多 F letter, 一个命中即置位 (OR 语义)."""
    from des.registry import feature_mask_of, PREDICATE_BIT
    # F4Nr1 + F4Nr4 共存, F4Nr4 命中阈值
    seq = ("F4Nr1", "F4Nr4") + ("N0",) * 14
    m = feature_mask_of(seq)
    assert m & PREDICATE_BIT["thr_crest"]


def test_thr_hotspot_hits_on_p_hotspot_at_boundary_0p05():
    """P_hotspot p_add=0.05 触发 thr_hotspot (闭区间 ≥0.05)."""
    from des.registry import feature_mask_of, PREDICATE_BIT
    seq = ("P_hotspot",) + ("N0",) * 15
    m = feature_mask_of(seq)
    assert m & PREDICATE_BIT["thr_hotspot"], (
        "P_hotspot (p_add=0.05) must SET thr_hotspot bit (boundary >= 0.05)")


def test_thr_hotspot_misses_on_p_base_below_threshold():
    """P_base p_add=0.0 < 0.05 → thr_hotspot CLEAR."""
    from des.registry import feature_mask_of, PREDICATE_BIT
    seq = ("P_base",) + ("N0",) * 15
    m = feature_mask_of(seq)
    assert not (m & PREDICATE_BIT["thr_hotspot"]), (
        "P_base (p_add=0.0) must NOT set thr_hotspot bit")


def test_thr_hotspot_misses_on_no_P_letter():
    """不含 P letter → thr_hotspot CLEAR (any over empty 是 False)."""
    from des.registry import feature_mask_of, PREDICATE_BIT
    seq = ("F4Nr4",) + ("N0",) * 15
    m = feature_mask_of(seq)
    assert not (m & PREDICATE_BIT["thr_hotspot"])


def test_thr_mirror_hits_on_broadsweep_z040_with_two_prey():
    """BroadSweep z=0.40 ≤ 0.45 且 |prey|=2 → thr_mirror SET."""
    from des.registry import feature_mask_of, PREDICATE_BIT
    seq = ("BroadSweep",) + ("N0",) * 15
    m = feature_mask_of(seq)
    assert m & PREDICATE_BIT["thr_mirror"], (
        "BroadSweep (z=0.40, |prey|=2) must SET thr_mirror bit")


def test_thr_mirror_boundary_z_at_0p45_with_two_prey(monkeypatch):
    """合成 'SweepSurge' z=0.45 (boundary ≤0.45 闭) 且 |prey|=2 → 命中."""
    monkeypatch.setitem(registry.ALPHABET, "SweepSurge", "Z")
    monkeypatch.setitem(registry.GRAN, "SweepSurge", "residue")
    monkeypatch.setitem(registry.VIS, "SweepSurge", 0.0)
    monkeypatch.setitem(registry._Z, "SweepSurge",
                        (0.45, (("F",), ("P",)), 5, 0))   # z=0.45, 2 prey
    # 重新派生 _Z_PREY_CARD (因为 monkeypatch 不会自动跑 module-level 派生)
    monkeypatch.setitem(registry._Z_PREY_CARD, "SweepSurge", 2)
    from des.registry import feature_mask_of, PREDICATE_BIT
    seq = ("SweepSurge",) + ("N0",) * 15
    m = feature_mask_of(seq)
    assert m & PREDICATE_BIT["thr_mirror"], (
        "SweepSurge (z=0.45 boundary, |prey|=2) must SET thr_mirror bit")


def test_thr_mirror_misses_on_high_z(monkeypatch):
    """合成 'AttritionBite' z=0.55 > 0.45 → thr_mirror CLEAR."""
    monkeypatch.setitem(registry.ALPHABET, "AttritionBite", "Z")
    monkeypatch.setitem(registry.GRAN, "AttritionBite", "residue")
    monkeypatch.setitem(registry.VIS, "AttritionBite", 0.0)
    monkeypatch.setitem(registry._Z, "AttritionBite",
                        (0.55, (("F",), ("Z",)), 5, 0))
    monkeypatch.setitem(registry._Z_PREY_CARD, "AttritionBite", 2)
    from des.registry import feature_mask_of, PREDICATE_BIT
    seq = ("AttritionBite",) + ("N0",) * 15
    m = feature_mask_of(seq)
    assert not (m & PREDICATE_BIT["thr_mirror"]), (
        "AttritionBite (z=0.55 > 0.45) must NOT set thr_mirror bit")


def test_thr_mirror_misses_on_specialist_single_prey(monkeypatch):
    """合成 z=0.40 但 |prey|=1 (单 prey 的 specialist) → thr_mirror CLEAR."""
    monkeypatch.setitem(registry.ALPHABET, "Specialist", "Z")
    monkeypatch.setitem(registry.GRAN, "Specialist", "residue")
    monkeypatch.setitem(registry.VIS, "Specialist", 0.0)
    monkeypatch.setitem(registry._Z, "Specialist",
                        (0.40, (("F",),), 5, 0))   # 仅 1 prey clause
    monkeypatch.setitem(registry._Z_PREY_CARD, "Specialist", 1)
    from des.registry import feature_mask_of, PREDICATE_BIT
    seq = ("Specialist",) + ("N0",) * 15
    m = feature_mask_of(seq)
    assert not (m & PREDICATE_BIT["thr_mirror"]), (
        "Specialist (z=0.40 OK, but |prey|=1 < 2) must NOT set thr_mirror bit")


def test_default_bb0_strain_thr_bits_per_global_constraints():
    """默认 BB0 layout: F4Nr4 (f=0.50) → thr_crest SET; P_base (0.0) → thr_hotspot CLEAR;
    BroadSweep (z=0.40, |prey|=2) → thr_mirror SET."""
    from des.registry import feature_mask_of, PREDICATE_BIT, BB0_TEMPLATE
    m = feature_mask_of(BB0_TEMPLATE["layout"])
    assert m & PREDICATE_BIT["thr_crest"]
    assert not (m & PREDICATE_BIT["thr_hotspot"])
    assert m & PREDICATE_BIT["thr_mirror"]


def test_thr_bits_pure_function_of_sequence_under_fixed_registry():
    """spec §6 relabel-invariance flavor: registry 不动时, 同 sequence 调两次
    feature_mask_of, 阈值 bit 必须一致 (per-letter 是 sequence 的纯函数)."""
    from des.registry import feature_mask_of, PREDICATE_BIT
    seq = ("F4Nr4", "P_hotspot", "BroadSweep", "F4Nr1") + ("N0",) * 12
    a = feature_mask_of(seq)
    b = feature_mask_of(seq)
    assert a == b
    # 三阈值 bit 与 vis_lowvis 都置位 (F4Nr4 + P_hotspot + BroadSweep + N0)
    assert a & PREDICATE_BIT["thr_crest"]
    assert a & PREDICATE_BIT["thr_hotspot"]
    assert a & PREDICATE_BIT["thr_mirror"]
    assert a & PREDICATE_BIT["vis_lowvis"]
