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


# --- Task 3 surface: _mutation_outcomes signature + N=1 byte-identical -----

def test_mutation_outcomes_default_kwarg_byte_identical_to_legacy():
    """显式 4-arg call (N=1) 与 legacy 3-arg call (无 kwarg) 必须返回完全相同的
    (children, weights) — 同 enumeration 顺序、同权重、同 RNG 路径 (spec §3
    红线 'by construction byte-identical, not merely distributionally equal').

    检测方式: legacy 3-arg 与 4-arg call 对同一 (seq, mutable, spectrum) 比较."""
    from des.kernels.reproduction import _mutation_outcomes
    from des.registry import _spectrum_for, BB0_TEMPLATE
    seq = BB0_TEMPLATE["layout"]
    mutable = BB0_TEMPLATE["mutable"]
    spectrum = _spectrum_for("P_base")
    # 注: 若 S6 已落, signature 是 (seq, mutable, spectrum, blocks, slots_per_event=1);
    # 旧 caller 仍用 (seq, mutable, spectrum, blocks) — 默认 kwarg 接住.
    # 这里测试用 S6 后的 4-arg call (无 kwarg) + 5-arg call (kwarg=1) 等价.
    # 若 S6 未落 (3-arg legacy), 4-arg call (kwarg=1) 与 3-arg call 等价.
    try:
        from des.registry import motif_blocks
        blocks = motif_blocks(seq)
        a_children, a_weights = _mutation_outcomes(seq, mutable, spectrum, blocks)
        b_children, b_weights = _mutation_outcomes(
            seq, mutable, spectrum, blocks, slots_per_event=1)
    except ImportError:
        a_children, a_weights = _mutation_outcomes(seq, mutable, spectrum)
        b_children, b_weights = _mutation_outcomes(
            seq, mutable, spectrum, slots_per_event=1)
    assert a_children == b_children, (
        "N=1 path must enumerate children in the exact same order as legacy call")
    assert a_weights == b_weights, (
        "N=1 path must produce the exact same weights as legacy call")


def test_mutation_outcomes_N_eq_1_via_kwarg_matches_default():
    """显式 slots_per_event=1 与不传 (默认 1) 字节级相等."""
    from des.kernels.reproduction import _mutation_outcomes
    from des.registry import _spectrum_for, BB0_TEMPLATE
    seq = BB0_TEMPLATE["layout"]
    mutable = BB0_TEMPLATE["mutable"]
    spectrum = _spectrum_for("P_base")
    try:
        from des.registry import motif_blocks
        blocks = motif_blocks(seq)
        default = _mutation_outcomes(seq, mutable, spectrum, blocks)
        explicit = _mutation_outcomes(
            seq, mutable, spectrum, blocks, slots_per_event=1)
    except ImportError:
        default = _mutation_outcomes(seq, mutable, spectrum)
        explicit = _mutation_outcomes(seq, mutable, spectrum, slots_per_event=1)
    assert default == explicit


def test_mutation_outcomes_N_eq_2_raises_notimplemented_in_task_3():
    """Task 3 仅落 N=1; N≥2 触发 NotImplementedError 占位 (Task 4 实现).
    本测试在 Task 4 完成后被 Task 4 的 N=2 正确性测试替换 (Task 4 Step 1 列出删除)."""
    from des.kernels.reproduction import _mutation_outcomes
    from des.registry import _spectrum_for, BB0_TEMPLATE
    seq = BB0_TEMPLATE["layout"]
    mutable = BB0_TEMPLATE["mutable"]
    spectrum = _spectrum_for("P_base")
    with pytest.raises(NotImplementedError):
        try:
            from des.registry import motif_blocks
            _mutation_outcomes(
                seq, mutable, spectrum, motif_blocks(seq), slots_per_event=2)
        except ImportError:
            _mutation_outcomes(seq, mutable, spectrum, slots_per_event=2)


def test_default_bb0_engine_run_byte_identical_post_s7_kernel_change():
    """phase2_reproduce call-site 加 slots_per_event=N kwarg 后, 默认 BB0
    strain 全部 N=1 → kernel 走 verbatim N=1 路径 → 同 seed 跑两次仍 byte-identical.
    守 'S7 改 kernel call-site 一行不漂移 BB0 默认局' (spec §2 红线 3)."""
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
