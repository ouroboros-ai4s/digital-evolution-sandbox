# tests/test_multi_slot.py
"""S7 multi-slot mutation: SLOTS_PER_EVENT table, Phenotype.slots_per_event field,
_mutation_outcomes N-slot joint enumeration.

This file is the S7 owner test file (sibling: tests/test_motif.py /
test_spectrum_shape.py / test_phase_windows.py / test_vis.py /
test_direction_kinds.py). Covers registry table coverage, phenotype field,
N=1 byte-identical regression, N>=2 joint-enumeration weights, N>=2 distinct
slots, N clamp on #mutable<N, relabel-invariance, same-sequence merge."""
from __future__ import annotations
import pytest


# --- Task 1 surface: SLOTS_PER_EVENT table -----------------------------------

def test_slots_per_event_covers_every_alphabet_letter():
    """SLOTS_PER_EVENT 必须覆盖全部 ALPHABET letter, key 集合相等."""
    from des.registry import SLOTS_PER_EVENT, ALPHABET
    assert set(SLOTS_PER_EVENT.keys()) == set(ALPHABET.keys()), (
        f"missing={set(ALPHABET) - set(SLOTS_PER_EVENT)}, "
        f"extra={set(SLOTS_PER_EVENT) - set(ALPHABET)}")


def test_slots_per_event_v1_all_one():
    """S7 落地时 (S0..S6 都已绿), 全部 active letter slots_per_event=1
    (spec §2 红线 3: P_cascade 是唯一 slots=2 的 roster 行, S8 才铸)."""
    from des.registry import SLOTS_PER_EVENT
    for letter, n in SLOTS_PER_EVENT.items():
        assert n == 1, f"{letter}: expected slots_per_event=1 in S7, got {n!r}"


def test_slots_per_event_value_in_legal_domain():
    """value ∈ {1, 2} — module-load assert 守门 (P_cascade roster L230 = 2;
    更高 N 留给 future-spec 显式放宽时 bump assert)."""
    from des.registry import SLOTS_PER_EVENT
    for letter, n in SLOTS_PER_EVENT.items():
        assert isinstance(n, int), f"{letter}: not int, got {type(n).__name__}"
        assert n in (1, 2), f"{letter}: slots_per_event {n!r} not in {{1, 2}}"


# --- Task 2 surface: Phenotype.slots_per_event field -----------------------

def test_phenotype_has_slots_per_event_field():
    """Phenotype.slots_per_event 字段存在, int 类型."""
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert hasattr(p, "slots_per_event")
    assert isinstance(p.slots_per_event, int)


def test_phenotype_default_bb0_slots_per_event_is_1():
    """默认 BB0 layout 的 dominant_p='P_base', SLOTS_PER_EVENT['P_base']=1
    → phenotype.slots_per_event = 1 (静态默认)."""
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert p.slots_per_event == 1


def test_phenotype_no_P_letter_slots_per_event_is_1():
    """sequence 没有 P letter → dominant_p is None → slots_per_event=1 (default)."""
    from des.registry import phenotype
    seq = ("F4Nr1",) + ("N0",) * 15
    p = phenotype(seq)
    assert p.slots_per_event == 1


def test_phenotype_is_still_frozen_after_s7_field():
    """加字段后 dataclass 仍 frozen, 不可改 (守不可变契约 + S4/S5 同纪律)."""
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    with pytest.raises(Exception):
        p.slots_per_event = 99      # FrozenInstanceError under @dataclass(frozen=True)


def test_phenotype_synthetic_P_with_slots_2_propagates(monkeypatch):
    """合成 letter P_cascade 进 ALPHABET / GRAN / _P / SPECTRUM_SHAPE / SLOTS_PER_EVENT
    全表, 让它成为 dominant_p (highest p_add); phenotype.slots_per_event 应 = 2.
    (S7 不铸 P_cascade — S8 owns; 此测用 monkeypatch 走 N=2 路径, sibling
    pattern 与 S6 test_motif.py 用合成 'M3' 同款.)"""
    from des import registry
    monkeypatch.setitem(registry.ALPHABET, "P_cascade", "P")
    monkeypatch.setitem(registry.GRAN,     "P_cascade", "residue")
    # p_add=0.28 高于既有所有 P 行 → dominant
    monkeypatch.setitem(registry._P, "P_cascade", (0.28, 2))
    monkeypatch.setitem(registry.SPECTRUM_SHAPE, "P_cascade", (1.0, None, 0.0))
    monkeypatch.setitem(registry.SLOTS_PER_EVENT, "P_cascade", 2)
    seq = ("P_cascade",) + ("N0",) * 15
    p = registry.phenotype(seq)
    assert p.slots_per_event == 2, (
        f"dominant_p='P_cascade' should propagate slots=2, got {p.slots_per_event}")
