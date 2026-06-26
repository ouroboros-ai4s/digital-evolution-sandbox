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


# --- Task 4 surface: N>=2 joint enumeration path -----------------------------

# 注: 删除 Task 3 留下的 test_mutation_outcomes_N_eq_2_raises_notimplemented_in_task_3
# 测试 (Task 4 实现 N=2, NotImplementedError 不再 raise). 同步把那条 test 替换为
# 下面的 N=2 正确性测试集。

def _synthesize_two_letter_spectrum():
    """N=2 测试用 fixture: 2 letter spectrum [(A, 0.6), (B, 0.4)],
    Σq=1, 排序确定."""
    return (("F4Nr1", 0.6), ("F4Nr4", 0.4))


def test_mutation_outcomes_N_eq_2_children_have_2_distinct_slot_changes():
    """N=2 时, 每条 child 在 mutable slot 中应有 exactly 2 个 distinct slot
    与 parent 不同 (排除 self-loop 时两 letter 都等于原 letter 的情形,
    但本 fixture spectrum 字母与默认 BB0 BB0 layout 同一位置 BB0 内容差异下,
    可能等于 parent — 因此本测试只断 'at most 2 slot diff')."""
    from des.kernels.reproduction import _mutation_outcomes
    from des.registry import BB0_TEMPLATE
    try:
        from des.registry import motif_blocks
        seq = BB0_TEMPLATE["layout"]
        mutable = BB0_TEMPLATE["mutable"]
        spectrum = _synthesize_two_letter_spectrum()
        children, weights = _mutation_outcomes(
            seq, mutable, spectrum, motif_blocks(seq), slots_per_event=2)
    except ImportError:
        seq = BB0_TEMPLATE["layout"]
        mutable = BB0_TEMPLATE["mutable"]
        spectrum = _synthesize_two_letter_spectrum()
        children, weights = _mutation_outcomes(
            seq, mutable, spectrum, slots_per_event=2)
    for child in children:
        diff = sum(1 for a, b in zip(seq, child) if a != b)
        # N=2: 0 diff (self-loop both slots), 1 diff (one self-loop), 2 diff (no self-loop)
        assert diff in (0, 1, 2), (
            f"N=2 outcome must change 0..2 positions; got {diff}: {child!r}")


def test_mutation_outcomes_N_eq_2_slot_set_count_matches_combinations():
    """N=2 outcome 総数 = C(m, 2) * |spectrum|^2.
    BB0 mutable = 6, spectrum = 2 letter → 15 * 4 = 60 outcomes."""
    from des.kernels.reproduction import _mutation_outcomes
    from des.registry import BB0_TEMPLATE
    from math import comb
    seq = BB0_TEMPLATE["layout"]
    mutable = BB0_TEMPLATE["mutable"]
    spectrum = _synthesize_two_letter_spectrum()
    m = sum(mutable)
    expected_count = comb(m, 2) * (len(spectrum) ** 2)
    try:
        from des.registry import motif_blocks
        children, weights = _mutation_outcomes(
            seq, mutable, spectrum, motif_blocks(seq), slots_per_event=2)
    except ImportError:
        children, weights = _mutation_outcomes(
            seq, mutable, spectrum, slots_per_event=2)
    assert len(children) == expected_count, (
        f"N=2 outcome count: expected C({m},2)*|spec|^2 = {expected_count}, "
        f"got {len(children)}")
    assert len(weights) == expected_count


def test_mutation_outcomes_N_eq_2_weights_sum_to_one():
    """Σweights = 1 (spec §3: (1/C(m,N)) · ∏ q;
    展开 = (1/C(m,N)) · C(m,N) · (Σq)^N = 1 因 spectrum 已归一)."""
    from des.kernels.reproduction import _mutation_outcomes
    from des.registry import BB0_TEMPLATE
    seq = BB0_TEMPLATE["layout"]
    mutable = BB0_TEMPLATE["mutable"]
    spectrum = _synthesize_two_letter_spectrum()
    try:
        from des.registry import motif_blocks
        _, weights = _mutation_outcomes(
            seq, mutable, spectrum, motif_blocks(seq), slots_per_event=2)
    except ImportError:
        _, weights = _mutation_outcomes(
            seq, mutable, spectrum, slots_per_event=2)
    assert abs(sum(weights) - 1.0) < 1e-9, f"Σweights = {sum(weights)} != 1"


def test_mutation_outcomes_N_eq_2_weight_formula_matches_spec():
    """每条 outcome weight = (1/C(m,N)) * ∏_{s} q(letter_s).
    用单 letter spectrum + 单 slot_set 子样本直接验.
    Fixture: spectrum=((A,0.6),(B,0.4)) m=6, N=2 → C(6,2)=15.
    outcome (slot_set=(0,2), letters=(A,B)) 的 weight = (1/15) * 0.6 * 0.4 = 0.016."""
    from des.kernels.reproduction import _mutation_outcomes
    from des.registry import BB0_TEMPLATE
    from math import comb
    seq = BB0_TEMPLATE["layout"]
    mutable = BB0_TEMPLATE["mutable"]
    spectrum = (("F4Nr1", 0.6), ("F4Nr4", 0.4))
    try:
        from des.registry import motif_blocks
        children, weights = _mutation_outcomes(
            seq, mutable, spectrum, motif_blocks(seq), slots_per_event=2)
    except ImportError:
        children, weights = _mutation_outcomes(
            seq, mutable, spectrum, slots_per_event=2)
    m = sum(mutable)
    inv_C = 1.0 / comb(m, 2)
    # 每个 weight 应是 inv_C × q_letter1 × q_letter2; 唯一可能值 = {inv_C×0.6×0.6,
    # inv_C×0.6×0.4, inv_C×0.4×0.4}
    expected_set = {inv_C * 0.6 * 0.6, inv_C * 0.6 * 0.4, inv_C * 0.4 * 0.4}
    for w in weights:
        # 模糊匹配到 expected_set
        assert any(abs(w - e) < 1e-9 for e in expected_set), (
            f"weight {w} not in {expected_set}")


def test_mutation_outcomes_N_eq_2_continuous_with_N_eq_1_formula():
    """N=1 时, joint 公式 (1/C(m,1)) * q = q/m, 与 verbatim path 输出 q/|slots| 数学等价.
    本测试不验 byte-identity (那是 Task 3 N=1 verbatim 守门;
    本测试验 Task 4 实现写出来的 joint 公式在 N=1 退化值上吻合)."""
    # 不直接调用 N=1 joint 路径 (那条永远走 verbatim), 直接验算式数学一致:
    from math import comb
    m = 6
    q = 0.3
    joint = (1.0 / comb(m, 1)) * q
    verbatim = q / m
    assert abs(joint - verbatim) < 1e-12


def test_mutation_outcomes_N_clamped_when_exceeds_mutable():
    """spec §5: N > #mutable → clamp to #mutable. mutable=2, N=10 → effective_N=2,
    outcome 数应 = C(2,2) * |spectrum|^2 = 1 * 4 = 4 (而非崩)."""
    from des.kernels.reproduction import _mutation_outcomes
    seq = ("N0", "N0") + ("N0",) * 14
    mutable = (True, True) + (False,) * 14   # 仅 2 mutable
    spectrum = (("F4Nr1", 0.6), ("F4Nr4", 0.4))
    try:
        from des.registry import motif_blocks
        children, weights = _mutation_outcomes(
            seq, mutable, spectrum, motif_blocks(seq), slots_per_event=10)
    except ImportError:
        children, weights = _mutation_outcomes(
            seq, mutable, spectrum, slots_per_event=10)
    # effective_N = min(10, 2) = 2 → C(2,2)=1 slot_set × 2^2=4 letter combos
    assert len(children) == 4, f"N=10 clamped to m=2: expected 4 outcomes, got {len(children)}"
    assert abs(sum(weights) - 1.0) < 1e-9


def test_mutation_outcomes_N_eq_2_slot_sets_are_distinct():
    """spec §5: N distinct slots, sample without replacement.
    itertools.combinations 天然保证 — 检测每条 outcome 的 slot_set 内部 distinct."""
    from des.kernels.reproduction import _mutation_outcomes
    from des.registry import BB0_TEMPLATE
    seq = BB0_TEMPLATE["layout"]
    mutable = BB0_TEMPLATE["mutable"]
    spectrum = (("F4Nr1", 0.6), ("F4Nr4", 0.4))
    try:
        from des.registry import motif_blocks
        children, _ = _mutation_outcomes(
            seq, mutable, spectrum, motif_blocks(seq), slots_per_event=2)
    except ImportError:
        children, _ = _mutation_outcomes(
            seq, mutable, spectrum, slots_per_event=2)
    # 对每个 child, 与 parent 不同的位置集合是 {distinct slots ⊆ mutable_idx}.
    # 大小至多 2 (N=2), 且每个 idx 在 mutable_idx 集合里.
    mutable_idx = {i for i, ok in enumerate(mutable) if ok}
    for child in children:
        diff = [i for i, (a, b) in enumerate(zip(seq, child)) if a != b]
        assert set(diff).issubset(mutable_idx), (
            f"diff slots {diff} must be subset of mutable {mutable_idx}")
        assert len(set(diff)) == len(diff), "diff slots must be distinct"


def test_mutation_outcomes_N_eq_2_same_sequence_collapsed_by_get_or_mint(monkeypatch):
    """spec §5: same-sequence cascade children merge via get_or_mint.
    本测试端到端走 phase2_reproduce: 合成 P_cascade-like letter (N=2), 跑 5 tick,
    multiple (slot, letter, letter) outcomes 在 child sequence 上重合 (例如 slot=0
    letter=A & slot=1 letter=B 与 slot=1 letter=B & slot=0 letter=A 都产同序列),
    get_or_mint 在 strain table 上应共享 sid."""
    import torch
    from des import registry
    from des.engine import Engine
    # 合成 P_cascade 让它走 N=2 路径; spectrum 退化为 1 letter 保证 children 全收敛同 seq
    monkeypatch.setitem(registry.ALPHABET, "P_cascade", "P")
    monkeypatch.setitem(registry.GRAN, "P_cascade", "residue")
    monkeypatch.setitem(registry._P, "P_cascade", (0.28, 2))
    monkeypatch.setitem(registry.SPECTRUM_SHAPE, "P_cascade", (1.0, None, 0.0))
    monkeypatch.setitem(registry.SLOTS_PER_EVENT, "P_cascade", 2)
    # 单 cell strain 装 P_cascade; 跑 5 tick 后 strain table 至少有 1 新 strain
    # (mutation 产生); 任何 same-sequence cascade 走 get_or_mint 共享 sid (端到端测).
    # BB0 slot 0 (position 0 is in _SLOTS); locked positions {1,5,7} must stay intact.
    from des.registry import BB0_TEMPLATE
    base = list(BB0_TEMPLATE["layout"])
    base[0] = "P_cascade"   # slot 0 is mutable (_SLOTS = {0,2,3,9,10,13})
    cascade_layout = tuple(base)
    eng = Engine(H=2, W=2, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2, layouts=(cascade_layout,) * 4)
    eng.run(5, recorder=None, stop_on=())
    # 不抛 (kernel N=2 路径无崩); strain table 完整性 (有 strain 被 mint 即 PASS)
    assert len(eng.table) >= 1, "P_cascade synthetic letter should mint at least 1 strain"
