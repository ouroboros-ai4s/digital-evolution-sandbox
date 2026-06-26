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
