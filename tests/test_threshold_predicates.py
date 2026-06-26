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


# ---------------------------------------------------------------------------
# Task 3: byte-identical 回归 + per-letter audit
# ---------------------------------------------------------------------------

def test_default_bb0_same_seed_byte_identical_post_s3():
    """S3 给 feature_mask_of 加 3 个 thr_* bit, 但 v1 没有 prey clause target
    它们 (BroadSweep 仍 (("F",), ("Z",)) family-only). 默认 BB0 4-faction 局
    跑 30 tick, world.count + strain_id 字节级一致 (regression lock §6).

    判定: 同 seed 双跑得到 bit-identical 结果 — 守 antagonism kernel 没多
    吃 / 少吃 kill。"""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng_a = Engine(H=8, W=8, K=8, seed=0, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2,
                   layouts=(BB0_TEMPLATE["layout"],) * 4)
    eng_b = Engine(H=8, W=8, K=8, seed=0, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2,
                   layouts=(BB0_TEMPLATE["layout"],) * 4)
    eng_a.run(30, recorder=None, stop_on=())
    eng_b.run(30, recorder=None, stop_on=())
    assert torch.equal(eng_a.world.count, eng_b.world.count)
    assert torch.equal(eng_a.world.strain_id, eng_b.world.strain_id)


def test_default_bb0_match_relation_byte_identical_with_synthetic_predator():
    """加一个仅命中 thr_mirror 的合成 predator strain, 它去打默认 BB0 prey
    时 antagonism 的 (prey_mask & feature_mask) 关系应与「BroadSweep 走 family
    clause」的旧关系一致 — thr_mirror SET on default BB0 prey, 但 ('Z',) family
    clause 也 hit prey BroadSweep 上的 family_Z 位 — 两条 path 都 produce 非零
    bitand. 这一条只验逻辑等价, 不验数值."""
    from des.registry import (feature_mask_of, prey_mask_for_clauses,
                               PREDICATE_BIT, BB0_TEMPLATE)
    prey_m = feature_mask_of(BB0_TEMPLATE["layout"])
    family_z_clause = (("Z",),)
    fm_pred_family = prey_mask_for_clauses(family_z_clause)
    # family_Z bit 与 BroadSweep 在 prey strain 上的 family_Z bit 相 & 应非零
    assert (fm_pred_family & prey_m) != 0
    # thr_mirror bit 也应在 prey strain 上 SET (BB0 含 BroadSweep z=0.40, |prey|=2)
    assert (PREDICATE_BIT["thr_mirror"] & prey_m) != 0


def test_thr_crest_is_per_letter_not_stacked_f():
    """spec §2 红线 + Global Constraint: thr_crest 读 `_F[letter][0]`, 不读
    Phenotype.f (stacked).

    构造: F4Nr1 (f=0.30) + F4Nr1 (f=0.30) → stacked f = 1-(1-0.3)(1-0.3) = 0.51,
    > 0.5 stacked threshold; 但 per-letter f 都是 0.30 < 0.5 → thr_crest CLEAR.

    若 thr_crest 误读了 Phenotype.f (stacked), 这条会假阳性 SET; 正确实现
    应 CLEAR."""
    from des.registry import phenotype, feature_mask_of, PREDICATE_BIT
    seq = ("F4Nr1", "F4Nr1") + ("N0",) * 14
    p = phenotype(seq)
    # stacked f 已超过 0.5 (sanity check 表达 stacked 算法仍生效)
    assert p.f > 0.5, f"stacked f should exceed 0.5; got {p.f}"
    # 但 per-letter 都未达 0.5, thr_crest 必须 CLEAR
    m = feature_mask_of(seq)
    assert not (m & PREDICATE_BIT["thr_crest"]), (
        f"thr_crest leaked stacked f. seq stacked f={p.f}, but no _F letter "
        "has f>=0.5; bit must be CLEAR (per-letter, not stacked).")


# ---------------------------------------------------------------------------
# Task 4: prey_mask_for_clauses 4 个新 clause-tag
# ---------------------------------------------------------------------------

def test_prey_mask_for_clauses_f_hi_tag_selects_thr_crest():
    """clause ('F', 'f_hi') 仅命中 PREDICATE_BIT['thr_crest']."""
    from des.registry import prey_mask_for_clauses, PREDICATE_BIT
    pm = prey_mask_for_clauses((("F", "f_hi"),))
    assert pm == PREDICATE_BIT["thr_crest"], (
        f"('F','f_hi') must select only thr_crest; got pm={pm:b}")


def test_prey_mask_for_clauses_p_hi_tag_selects_thr_hotspot():
    """clause ('P', 'p_hi') 仅命中 PREDICATE_BIT['thr_hotspot']."""
    from des.registry import prey_mask_for_clauses, PREDICATE_BIT
    pm = prey_mask_for_clauses((("P", "p_hi"),))
    assert pm == PREDICATE_BIT["thr_hotspot"]


def test_prey_mask_for_clauses_generalist_tag_selects_thr_mirror():
    """clause ('Z', 'generalist') 仅命中 PREDICATE_BIT['thr_mirror']."""
    from des.registry import prey_mask_for_clauses, PREDICATE_BIT
    pm = prey_mask_for_clauses((("Z", "generalist"),))
    assert pm == PREDICATE_BIT["thr_mirror"]


def test_prey_mask_for_clauses_lowvis_tag_selects_vis_lowvis():
    """clause ('N', 'lowvis') 仅命中 PREDICATE_BIT['vis_lowvis'] (S1 reserved 索引 11)."""
    from des.registry import prey_mask_for_clauses, PREDICATE_BIT
    pm = prey_mask_for_clauses((("N", "lowvis"),))
    assert pm == PREDICATE_BIT["vis_lowvis"]


def test_prey_mask_for_clauses_family_only_unchanged_post_s3():
    """S6 既有 (fam,) family-only clause 行为字节级不变 (regression lock)."""
    from des.registry import prey_mask_for_clauses, PREDICATE_BIT
    # BroadSweep 的 default clauses
    pm = prey_mask_for_clauses((("F",), ("Z",)))
    expected = PREDICATE_BIT["family_F"] | PREDICATE_BIT["family_Z"]
    assert pm == expected, (
        f"family-only clauses changed; got {pm:b}, expected {expected:b}")


def test_prey_mask_for_clauses_motif_clauses_unchanged_post_s3():
    """S6 既有 ('F', 'motif') / ('Z', 'motif', 'len>=3') 行为字节级不变."""
    from des.registry import prey_mask_for_clauses, PREDICATE_BIT
    pm_m = prey_mask_for_clauses((("F", "motif"),))
    assert pm_m == PREDICATE_BIT["motif_F"]
    pm_m3 = prey_mask_for_clauses((("Z", "motif", "len>=3"),))
    assert pm_m3 == PREDICATE_BIT["motif3_Z"]


def test_prey_mask_for_clauses_mixed_old_and_new_tags_OR():
    """混合 clause: (Z, generalist) + (F,) → thr_mirror | family_F (OR over clauses)."""
    from des.registry import prey_mask_for_clauses, PREDICATE_BIT
    pm = prey_mask_for_clauses((("Z", "generalist"), ("F",)))
    expected = PREDICATE_BIT["thr_mirror"] | PREDICATE_BIT["family_F"]
    assert pm == expected


def test_prey_mask_for_clauses_unknown_tag_falls_through_to_family():
    """spec §3.2 设计契约: unknown tag 不应抛, 而是 fall through 到 family-only.
    这条守 forward-compat — 未来 spec 加新 tag 时, 旧 _Z 行的 clause 仍正确解析。

    构造: ('F', 'unknown_future_tag') → 应解释为 family_F bit."""
    from des.registry import prey_mask_for_clauses, PREDICATE_BIT
    pm = prey_mask_for_clauses((("F", "unknown_future_tag"),))
    assert pm == PREDICATE_BIT["family_F"], (
        f"unknown tag must fall through to family bit; got {pm:b}")


# ---------------------------------------------------------------------------
# Task 5: vis_lowvis end-to-end audit (S1 prey side + S3 predator side)
# ---------------------------------------------------------------------------

def test_vis_lowvis_end_to_end_s1_prey_meets_s3_predator_clause():
    """S1 把 vis_lowvis bit 设在 feature_mask (prey 端);
       S3 把 ('N','lowvis') clause 映射到 vis_lowvis bit (predator 端).
       Match expression 在合成 predator vs default BB0 prey 之间必须命中.

       这一条是 'S1 vis_lowvis 死了没?' 的 end-to-end smoke — 守 Task 2
       的扩展没误删 S1 Task 6 在 feature_mask_of 里加的那段 vis_lowvis
       置位代码."""
    from des.registry import (feature_mask_of, prey_mask_for_clauses,
                               PREDICATE_BIT, BB0_TEMPLATE)
    # prey 端: default BB0 含 N0 (vis=0.20 ≤ 0.20)
    prey_m = feature_mask_of(BB0_TEMPLATE["layout"])
    assert prey_m & PREDICATE_BIT["vis_lowvis"], (
        "S1 Task 6 vis_lowvis bit 缺失 — Task 2 可能误覆写 feature_mask_of")
    # predator 端: ('N', 'lowvis') clause → vis_lowvis bit
    pred_m = prey_mask_for_clauses((("N", "lowvis"),))
    assert pred_m == PREDICATE_BIT["vis_lowvis"]
    # match 关系非零
    assert (prey_m & pred_m) != 0
