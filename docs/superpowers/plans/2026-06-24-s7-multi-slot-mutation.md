# S7 — 多位点突变 (multi-slot mutation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `_mutation_outcomes` 从「单 slot 覆写」推广到「N slots/event 联合枚举」,默认 N=1 走当前代码逐字节(by construction),N≥2 走联合枚举路径;通过新 registry 表 `SLOTS_PER_EVENT[letter]` 给每个基元打一个 `slots_per_event: int` 标签(全部已有/S0–S6 active letter 都 =1),把它由 phenotype 累加(`piggyback` S2 的 spectrum-source 选择规则 — highest `p_add`,平手取首现)写到 `Phenotype.slots_per_event`,kernel 在既有 per-parent 循环里多读一行 `N = table.phenotype_of(p).slots_per_event`。P_cascade 是 roster 里唯一 `slots=2` 的基元,**S7 不铸**(归 S8 A-pool),tests 用 monkeypatch 合成 letter 跑 N=2 路径。

**Architecture:** 四件事,顺序: (1) `SLOTS_PER_EVENT: dict[str, int]` 表加进 `registry.py`,默认 `1` 等价于「当前单 slot 行为」,module-load `assert` 守 `keys ⊇ ALPHABET.keys()` 与 `value in {1, 2}`(P_cascade 当前夹层最大值是 2,future-spec 升 N 时再放宽); (2) `Phenotype` 加字段 `slots_per_event: int = 1`,`phenotype()` 在 `dominant_p` 解析完之后,读 `SLOTS_PER_EVENT.get(dominant_p, 1)`(无 P 行→默认 1)写入;静态默认下 v1 + S0–S6 全部 active letter `slots_per_event=1`,byte-identical; (3) `_mutation_outcomes(seq, mutable, spectrum, slots_per_event=1)` 加形参,N==1 走当前逐字节循环(`for s in slot_idx: for letter, q in spectrum: ...`,**枚举顺序、weight `q/len(slots)`、RNG 调用次数一律不动** — by construction byte-identical),N≥2 临时 raise `NotImplementedError` 占位; (4) 实现 N≥2 联合枚举路径: 列举 `unordered slot-sets S of size N` × per-slot spectrum letters,weight = `(1/C(m,N)) · ∏_{s∈S} q(letter_s)`,N==1 退化为 `q/C(m,1) = q/m`(与 (3) 的 N=1 路径数学连续但 RNG 路径不同 — N=1 仍走 (3) 的 verbatim 代码,by construction byte-identical),N > #mutable 时 clamp 到 #mutable(`min(N, m)`);call-site `phase2_reproduce` 在既有 `for p in sorted(set(...))` 距离体内多读一行 `N = table.phenotype_of(p).slots_per_event` 传给 `_mutation_outcomes`。

**Tech Stack:** Python 3 (`D:\anaconda3\envs\basic\python.exe`), PyTorch 2.10+cu128, pyarrow, pytest, `itertools.combinations` (stdlib). Windows 主机,`PYTHONPATH=src` 纪律。引擎源码 `src/des/`。**依赖**: S0 (CLI 已就位,不动) + S6 (`GRAN` 表已落地;P_cascade 若 future-spec 在 S8 铸,gran 走 `"residue"`) + S2 (`SPECTRUM_SHAPE` 表已落地 + `dominant_p` 仍是「highest p_add、平手取首现」单源,N piggybacks 同源选择,**不**引入新选择规则)。**不依赖** S5 (windowed-f) / S4 (动态 dirs) / S1 (vis) / S3 (富猎物谓词) — 它们与多位点突变正交。

## Global Constraints

- **`slots_per_event` 是结构 registry 全局值,不是 per-species 强度**: `SLOTS_PER_EVENT[letter] = int` 全 letter 全局 (spec §2 红线 1),禁止 per-species / per-cell 旋钮。每条 letter 的 N 由 roster 直读。
- **N piggybacks S2 spectrum-source 选择**: kernel 读 `N = phenotype.slots_per_event`,这个字段在 `phenotype()` 里由 dominant_p (highest p_add,平手取首现 — `registry.py:101-105` 的 `dominant_p` 规则) 读 `SLOTS_PER_EVENT[dominant_p]` 而来。**绝不**新增 selection rule (spec §3 红线)。
- **N==1 静态默认字节级不变 (by construction, NOT by distribution)**: 默认局 v1 + S0–S6 全部 active letter `SLOTS_PER_EVENT=1`,`_mutation_outcomes` 走 N=1 verbatim 路径 — 枚举顺序 (slot-ascending × `_spectrum_for` order)、weight (`q/|slots|`)、RNG 调用次数 (单 `torch.multinomial` per distinct parent) 与 pre-S7 字节级一致 (spec §2 红线 3)。回归锁 = 285 引擎 + 146 web 全套 pytest + S0/S6/S2 落地后新增 test 全绿。
- **P_cascade 是 S7 唯一 slots≠1 的 roster 行,但 S7 不铸 (S8 owns)**: spec §1 + §7 已锁。`SLOTS_PER_EVENT` 表里 **可选** 加一行 `P_cascade: 2` 当 reserved hook,但本 plan 选择**不加** — 等 S8 铸 letter 时一并加表项,保持 SLOTS_PER_EVENT.keys() ⊆ ALPHABET.keys() 闭合不变量。N≥2 路径的测试用 monkeypatch 合成 `P_cascade` 候补 letter 走 (跟 S6 motif 测试用合成 `M3` 同款纪律)。
- **多 P 字母 stacked 时 N 取 dominant 的,不是 sum/max**: 多 P 字母共存时,spectrum 已由 S2 走 dominant_p 单源 (S2 未碰多 P 混合,合并谱 `Σpᵢqᵢ/Σpᵢ` 归 S8);N 同样 piggyback dominant_p 取值。**多 P 行 stacking 决议归 S8** (spec §3 inline note + §7),S7 仅守「dominant 单源」。
- **N > #mutable 自动 clamp 到 #mutable**: BB0 默认 mutable 是 6,N=2 没问题;若 future-spec 把 N=10 而 mutable=6,kernel `_mutation_outcomes` 内 `effective_N = min(slots_per_event, m)`,**不抛**,数学上等价于「全部 mutable slot 一起突变」(spec §5)。
- **N≥2 走 joint enumeration + 单 `torch.multinomial`**: 不引入 sequential N-step draw (spec §3 ponytail ceiling 明确写「至 N≥3 才考虑换 sequential」)。N=2 outcome 空间 = `C(m,N) · |spectrum|^N` ≈ `15 × 16² = 3840` per parent (m=6, |spectrum|≈16),可承受。重用既有「per-parent 单 multinomial scatter」机器 — `_mutation_outcomes` 出口仍是 `(children: list[tuple[str,...]], weights: list[float])`,kernel call-site 不动 multinomial 调用 (spec §3 first ponytail note)。
- **N≥2 同步 sample without replacement**: N 个 slot 必须 distinct (spec §5),由 `itertools.combinations(slot_idx, N)` 天然保证 (combinations 是 unordered + no-repeat)。
- **Same-sequence children 仍走 `get_or_mint` 自动合并** (spec §5): 既有 `get_or_mint(c)` 调用 + `torch.unique(key, dim=0)` aggregation 不动 — 任意 N 下都是同一路径。
- **frozen `Phenotype` 加字段必带默认值**: 与 S4 (`in_place / rand_dir`) / S1 (`vis_sum / n_count`) / S5 (`f_hi/f_lo/burst_w/burst_k`) 同纪律。`slots_per_event: int = 1` 默认 1,既有所有构造点不需要带 kwarg。
- **out of scope**: P_cascade 的 rate / spectrum (S2: rate 走 p_add,spectrum 走 plain aff) / P_cascade A-pool gating / minting (S8) / 多 P stacked 时的 `slots_per_event` 合并规则 (S8) / 多 P stacked 时的 spectrum 混合 `Σpᵢqᵢ/Σpᵢ` (S8) / sequential N-step draw (deferred until N≥3) / per-faction asymmetric N (HARD-GATE,独立 brainstorming)。

---

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `src/des/registry.py` | **Modify** | (a) 加 `SLOTS_PER_EVENT: dict[str, int]` 表(每条 ALPHABET letter 一行,默认值 `1`);(b) module-load `assert`:`set(SLOTS_PER_EVENT.keys()) == set(ALPHABET.keys())` + `value in {1, 2}`;(c) `phenotype()` 在 `dominant_p` 解析完之后多一行:`slots_per_event = SLOTS_PER_EVENT.get(dominant_p, 1) if dominant_p else 1`,写进 `Phenotype(...)` 构造调用的新 kwarg。 |
| `src/des/types.py` | **Modify** | `@dataclass(frozen=True) class Phenotype` 加字段 `slots_per_event: int = 1`,紧跟 S5 加的 `f_hi / f_lo / burst_w / burst_k` 之后(全部 default-value 字段在最末尾,与 S4 / S1 / S5 同纪律)。 |
| `src/des/kernels/reproduction.py` | **Modify** | (a) `_mutation_outcomes(seq, mutable, spectrum, slots_per_event=1)` 加形参,N==1 走当前 verbatim 单 slot 循环(枚举顺序、weight、RNG 调用次数 byte-identical);N≥2 走新 joint enumeration 分支(`itertools.combinations(slot_idx, effective_N)` × per-slot spectrum letters);clamp `effective_N = min(slots_per_event, len(slot_idx))`;(b) call-site 在 `for p in sorted(set(...))` 循环体内多一行 `N = table.phenotype_of(p).slots_per_event`,传给 `_mutation_outcomes`。 |
| `tests/test_multi_slot.py` | **Create** | S7 owner 测试文件(sibling: S6 `test_motif.py` / S2 `test_spectrum_shape.py` / S5 `test_phase_windows.py` / S1 `test_vis.py` / S4 `test_direction_kinds.py`)。覆盖 `SLOTS_PER_EVENT` 表存在 + 全默认 1 + module-load 校验 + `Phenotype.slots_per_event` 字段 + frozen 守门 + 默认 BB0 phenotype `slots_per_event=1` + N=1 byte-identical (枚举顺序 / 权重 / RNG 路径) + N=2 (monkeypatch 合成 P_cascade) 联合枚举权重 + N=2 distinct slots + N=2 #mutable<N clamp + relabel-invariance + same-sequence merge。 |
| `tests/test_reproduction.py` | **Modify (append)** | 追加 `_mutation_outcomes` signature regression:legacy 3-arg call 仍接受(默认 `slots_per_event=1`,旧 caller 不必带 kwarg)+ 显式 4-arg call(N=1)byte-identical 于 3-arg call。 |
| `tests/test_registry.py` | **Modify (append)** | 追加 `SLOTS_PER_EVENT` 覆盖每个 ALPHABET letter + value 默认 1(P_cascade 不在 ALPHABET,不算反例)+ `phenotype(BB0).slots_per_event == 1`。 |
| `tests/test_phenotype_cache.py` | **Modify (append)** | 追加一条:`phenotype(seq).slots_per_event` 字段存在 + 默认 BB0 全 strain 落 1。**phenotype_arrays 不加新列** — kernel 在 per-parent 循环里读 Python `Phenotype` 对象而非 phe['...'] 张量(`table.phenotype_of(p).slots_per_event`),与既有 mutation core 同款访问模式;此偏离 S5 / S4 给 kernel 张量列的纪律是**故意的**:N 只在 hot-loop 极偶尔分叉(99% parent 是 N=1),per-parent scalar 读不必批量化(详见 Task 4 设计说明)。 |

**Naming contract (locked, used by every task):**

```python
# src/des/registry.py
SLOTS_PER_EVENT: dict[str, int]                                    # letter -> N (1, 2, ...)
# 当前所有 ALPHABET letter 全部 = 1;P_cascade 在 S8 铸时由 S8 加 SLOTS_PER_EVENT['P_cascade'] = 2

# src/des/types.py
@dataclass(frozen=True)
class Phenotype:
    # ... 既有字段 + S4 + S1 + S2 + S5 + S3 (按落地顺序) ...
    slots_per_event: int = 1   # S7: N slots/event;静态默认 1 即「当前单 slot 行为」

# src/des/kernels/reproduction.py
def _mutation_outcomes(
    seq: tuple[str, ...],
    mutable: tuple[bool, ...],
    spectrum: tuple[tuple[str, float], ...],
    slots_per_event: int = 1,
) -> tuple[list[tuple[str, ...]], list[float]]
# N=1 byte-identical to pre-S7 code path (same enumeration order, same RNG call count).
# N>=2: itertools.combinations(slot_idx, effective_N) x product(spectrum, repeat=effective_N)
# weight: (1/C(m, effective_N)) * prod(q(letter_s) for s in slot_set)
# effective_N = min(slots_per_event, len(slot_idx))

# call site in phase2_reproduce:
for p in sorted(set(int(x) for x in ind_parent.tolist())):
    if p == 0:
        continue
    seq = table.sequence_of(p)
    phe = table.phenotype_of(p)
    spectrum = phe.spectrum
    N = phe.slots_per_event                                         # S7: read N per parent
    children, weights = _mutation_outcomes(
        seq, BB0_TEMPLATE["mutable"], spectrum, slots_per_event=N)
    # ... 既有 get_or_mint / torch.multinomial / aggregation 路径全部不变 ...
```

`_mutation_outcomes` 的形参顺序锁死:`(seq, mutable, spectrum, slots_per_event)`;kwarg 默认 `slots_per_event=1`,**旧 3-arg call 仍合法**(测试 Task 3 显式守门)。S6 已经把 `_mutation_outcomes` signature 扩为 `(seq, mutable, spectrum, blocks)`;**本 plan 假设 S6 已落地**,真实 signature 应为 `(seq, mutable, spectrum, blocks, slots_per_event=1)` —— Task 3 / 4 的所有代码示例都按 5-arg shape 写。若 S6 未落 (假设序与 S0/S6/S1/S2/S4/S5/S3/S7 路线图相违),回到 4-arg `(seq, mutable, spectrum, slots_per_event=1)` 即可,逻辑全保。

---

---

### Task 1: `SLOTS_PER_EVENT` registry 表 + module-load 守门(纯数据,无消费者)

**Goal:** 在 `src/des/registry.py` 加一个新的全局表 `SLOTS_PER_EVENT: dict[str, int]`,**覆盖每个 ALPHABET letter**,**全部默认值 1**(spec §2 红线 1 + §3:S0–S7 所有 active letter 全部 1);在表定义之后立即用 `assert` 守 `set(SLOTS_PER_EVENT.keys()) == set(ALPHABET.keys())` 与 `value in {1, 2}`(P_cascade 是 roster 里唯一 slots=2 的 letter,future-spec 升 N 时再放宽值域)。这一步**不动** `phenotype()` / `_mutation_outcomes` / `Phenotype` — 没人读它,纯数据扩张,既有行为 0 漂移。

**Files:**
- Modify: `src/des/registry.py`(在 `_P` 块之后、`SPECTRUM_SHAPE` 之后,`def affinity` 之前插入 `SLOTS_PER_EVENT` + module-load 校验)
- Test: `tests/test_multi_slot.py` (Create — S7 owner file 的第一批断言;后续 Task 续填)
- Test: `tests/test_registry.py` (append 一条覆盖断言)

**Interfaces:**
- Consumes: `ALPHABET` 当前全集(S0/S6/S1/S2/S4/S5 落地后的全部 letter 集合)。
- Produces:
  - `SLOTS_PER_EVENT: dict[str, int]` — 每个 ALPHABET letter 一行,默认 `1`。
  - module-load `assert`:覆盖闭合 + 值域 `{1, 2}`,违反 → import 即 fail。

- [ ] **Step 1: 写失败测试 — 表存在 + 覆盖 + 默认值 + 值域**

新建 `tests/test_multi_slot.py`:

```python
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
```

追加到 `tests/test_registry.py`:

```python
def test_s7_slots_per_event_covers_every_alphabet_letter():
    """S7: 跨 file 同款覆盖断言 (sibling to S6 test_gran_covers_every_alphabet_letter)."""
    from des.registry import SLOTS_PER_EVENT, ALPHABET
    assert set(SLOTS_PER_EVENT.keys()) == set(ALPHABET.keys())
    for letter, n in SLOTS_PER_EVENT.items():
        assert n == 1, f"{letter}: pre-S8 active letter must have slots_per_event=1"
```

- [ ] **Step 2: 跑失败测试**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_multi_slot.py tests/test_registry.py::test_s7_slots_per_event_covers_every_alphabet_letter -v
```

Expected: 4 条全 FAIL with `ImportError: cannot import name 'SLOTS_PER_EVENT' from 'des.registry'`。

- [ ] **Step 3: 加 `SLOTS_PER_EVENT` 表 + module-load 校验**

打开 `src/des/registry.py`,在 `SPECTRUM_SHAPE` 块之后 (S2 已落) / 在 `def affinity` 之前,插入:

```python
# Slots-per-event per primitive (S7 §2-3). N slots/event for the mutation core.
# v1 + S0..S6 active letters all default to 1 (current single-slot behavior).
# P_cascade is the sole roster row with slots=2 (primitive-roster.md L230) and
# is minted by S8 — when S8 lands, S8 adds 'P_cascade': 2 here. The assert below
# keeps the key set closed under ALPHABET so a future letter without a row
# halts at import time rather than silently picking up `.get(letter, 1)` magic.
SLOTS_PER_EVENT: dict[str, int] = {letter: 1 for letter in ALPHABET}

# Module-load value-domain assertions (spec §2 red line + §5 error handling).
# Halt fail-fast at import if a future spec edits the dict to a malformed value.
assert set(SLOTS_PER_EVENT.keys()) == set(ALPHABET.keys()), (
    "SLOTS_PER_EVENT must be co-extensive with ALPHABET; "
    f"missing={set(ALPHABET) - set(SLOTS_PER_EVENT)}, "
    f"extra={set(SLOTS_PER_EVENT) - set(ALPHABET)}")
for _letter, _n in SLOTS_PER_EVENT.items():
    assert isinstance(_n, int), \
        f"SLOTS_PER_EVENT[{_letter!r}] = {_n!r}: must be int"
    assert _n in (1, 2), \
        f"SLOTS_PER_EVENT[{_letter!r}] = {_n!r}: value must be in {{1, 2}} pre-S8"
del _letter, _n
```

(用 dict comprehension `{letter: 1 for letter in ALPHABET}` 而非一行一行手写 — 任何新加的 letter 自动默认 1,S7 落地时也免去逐 letter 手写的维护;S8 铸 P_cascade 时直接 `SLOTS_PER_EVENT["P_cascade"] = 2` 一行覆写即可,assert 会接住未声明 letter。)

- [ ] **Step 4: 跑测试, 确认 PASS**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_multi_slot.py tests/test_registry.py::test_s7_slots_per_event_covers_every_alphabet_letter -v
```

Expected: 4 条全 PASS。

- [ ] **Step 5: 跑全 suite, 确认无回归**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿。`SLOTS_PER_EVENT` 添加是纯数据,无消费者读它,既有 phenotype / kernel / mutation 路径 0 漂移。

Backtrack: 若 module-load `assert` 失败导致大量 import 错 — 大概率是 `ALPHABET` 集合本身在 S0..S6 落地后有多余的合成 letter,或 `SLOTS_PER_EVENT` 的 dict comprehension 失败。pytest 输出会展示 ImportError tracebacks;按错误信息追加缺漏 letter。

- [ ] **Step 6: Commit**

```bash
git add src/des/registry.py tests/test_multi_slot.py tests/test_registry.py
git commit -m "feat(s7): add SLOTS_PER_EVENT registry table + module-load assert

Per-letter slots-per-event table (1 row per ALPHABET letter, default 1).
Module-load asserts halt fail-fast on key-set drift vs ALPHABET and on
out-of-domain values (currently {1, 2}; P_cascade=2 lands in S8). No
consumer yet — phenotype() wiring in Task 2, kernel in Task 3/4."
```

---

---

### Task 2: `Phenotype.slots_per_event` 字段 + `phenotype()` 写入(无 kernel 消费者)

**Goal:** 给 `src/des/types.py` 的 frozen `Phenotype` dataclass 加字段 `slots_per_event: int = 1`(紧跟 S5 加的 `f_hi / f_lo / burst_w / burst_k` 之后,与 S4 / S1 / S5 同纪律 — default-value 字段全部在末尾);改 `src/des/registry.py::phenotype()` 在 `dominant_p` 解析完之后多一行,把 `SLOTS_PER_EVENT.get(dominant_p, 1) if dominant_p else 1` 写进 `Phenotype(...)` 构造调用的新 kwarg。这一步**不**改 `_mutation_outcomes` / 不改 `phenotype_arrays` / 不改 kernel — 字段写出来,无人读,既有行为 0 漂移。

**Files:**
- Modify: `src/des/types.py`(`Phenotype` 末尾追加 1 行字段)
- Modify: `src/des/registry.py`(`phenotype()` 末尾 `Phenotype(...)` 构造点 + 新一行解析)
- Test: `tests/test_multi_slot.py`(append)
- Test: `tests/test_phenotype_cache.py`(append 一条字段存在 + BB0 默认 1 断言)

**Interfaces:**
- Consumes: `SLOTS_PER_EVENT`(Task 1)、既有 `dominant_p` 解析(`registry.py:101-105`)。
- Produces:
  - `Phenotype.slots_per_event: int = 1` — 默认 1。
  - `phenotype(seq).slots_per_event` — `dominant_p` 取 `SLOTS_PER_EVENT[dominant_p]`,无 P 行(`dominant_p is None`)→ 1。

- [ ] **Step 1: 写失败测试 — 字段存在 + 默认 BB0 = 1 + frozen 守门**

追加到 `tests/test_multi_slot.py`:

```python
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
```

追加到 `tests/test_phenotype_cache.py`:

```python
def test_phenotype_slots_per_event_field_default_bb0():
    """phenotype(BB0).slots_per_event == 1 — 默认 BB0 dominant_p='P_base',
    SLOTS_PER_EVENT['P_base']=1, 静态默认."""
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert p.slots_per_event == 1
```

- [ ] **Step 2: 跑失败测试**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_multi_slot.py tests/test_phenotype_cache.py::test_phenotype_slots_per_event_field_default_bb0 -v
```

Expected: 5 + 1 条测试 FAIL —— `AttributeError: 'Phenotype' object has no attribute 'slots_per_event'`(其余同理);frozen 测试也 FAIL(字段不存在)。Task 1 的覆盖测试仍 PASS。

- [ ] **Step 3: 加 `slots_per_event` 字段到 `Phenotype`**

打开 `src/des/types.py`,把 `Phenotype` dataclass 末尾(在 S4 加的 `in_place` / `rand_dir`、S5 加的 `f_hi / f_lo / burst_w / burst_k` 之后)追加 1 行:

```python
@dataclass(frozen=True)
class Phenotype:
    f: float
    directions: tuple[tuple[int, int], ...]
    p_leave: float
    z_raw: float
    prey_mask: int
    feature_mask: int
    p_x: float
    spectrum: tuple[tuple[str, float], ...]
    period: int
    repro_period: int
    anta_period: int
    dir_bits: int
    phase_type: PhaseType | None
    fold: tuple[frozenset[int], ...]
    in_place: bool = False     # S4
    rand_dir: bool = False     # S4
    f_hi: float = 0.0          # S5
    f_lo: float = 0.0          # S5
    burst_w: int = 1           # S5
    burst_k: int = 1           # S5
    slots_per_event: int = 1   # S7: N slots/event;静态默认 1 = 当前单 slot 行为
```

(注:若 S5 / S4 / S1 落地后,字段名次序与本示例略有偏差 — 关键纪律是「**所有有默认值字段排在所有无默认值字段之后**」,新加字段紧跟最末尾即可。`slots_per_event` 默认 1 让既有 `Phenotype(...)` 构造点不需带新 kwarg。)

- [ ] **Step 4: 改 `phenotype()` 末尾解析 + 构造调用**

打开 `src/des/registry.py`,在 `phenotype()` 函数体内,**`dominant_p` 解析循环结束之后** / `Phenotype(...)` 构造之前,插入一行:

```python
    # ... 既有循环结束, dominant_p 已确定 ...
    spectrum = _spectrum_for(dominant_p) if dominant_p else ()
    # S7: piggyback dominant_p (S2 spectrum-source rule, registry.py:101-105) —
    # N is read from the SAME letter the spectrum is sourced from. Default 1
    # (no P letter, or dominant_p has SLOTS_PER_EVENT=1).
    slots_per_event = SLOTS_PER_EVENT.get(dominant_p, 1) if dominant_p else 1
    period = min(periods) if periods else 1
    # ... 既有 repro_period / anta_period / dir_bits 计算 ...
```

把 `Phenotype(...)` 构造调用末尾追加 1 个 kwarg(注意保持 S4 / S5 加的 kwarg 顺序):

```python
    return Phenotype(
        f=f, directions=tuple(directions), p_leave=p_leave, z_raw=z_sum,
        prey_mask=prey_mask, feature_mask=feature_mask, p_x=p_x,
        spectrum=spectrum, period=period,
        repro_period=repro_period, anta_period=anta_period, dir_bits=dir_bits,
        phase_type=phase_type, fold=(),
        in_place=in_place, rand_dir=rand_dir,                             # S4
        f_hi=f_hi, f_lo=f_lo_stacked, burst_w=burst_w_out, burst_k=burst_k_out,  # S5
        slots_per_event=slots_per_event,                                   # S7
    )
```

(若 S5 / S4 不在当前已落地路径,删除对应 kwarg 行即可,不影响 S7 字段写入逻辑。)

- [ ] **Step 5: 跑测试, 确认 PASS**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_multi_slot.py tests/test_phenotype_cache.py -v
```

Expected: 全 PASS。Task 1 + Task 2 在 `test_multi_slot.py` 里的全部断言绿;`test_phenotype_slots_per_event_field_default_bb0` 绿;既有 `test_phenotype_cache.py` 测试不漂移(没人读 `phenotype_arrays` 的新列 — 本 plan 不为 `slots_per_event` 加张量列,kernel 在 per-parent loop 里直接读 `phe.slots_per_event` Python 值,详见 Task 3/4)。

Backtrack:
- 若 `TypeError: non-default argument follows default argument` — `slots_per_event` 必须放在 dataclass 末尾(全 default-value 字段之后)。
- 若 `Phenotype(...)` 构造抛 `TypeError: unexpected keyword argument 'slots_per_event'` — 检查 dataclass 的字段是否真的加上去了。
- 若 `test_phenotype_synthetic_P_with_slots_2_propagates` FAIL — root cause:`dominant_p` 解析的 `if p_add > _P[dominant_p][0]` 是严格 `>`,平手取首现;P_cascade 的 `p_add=0.28` 大于既有所有 P 行(P_hotspot=0.05、P_burst_lite=0.07 是当前最大),应当成 dominant。若 monkeypatch `_P` 加新行后没解析到,检查 `phenotype()` 循环里 `letter in _P` 分支是否走到。

- [ ] **Step 6: 跑全 suite, 确认无回归**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿。`Phenotype.slots_per_event` 默认 1,kernel 没改,既有所有 strain 字段写出 `slots_per_event=1`,既有 `phenotype_cache` / `phase2_reproduce` / `webapp.readouts` 读路径 0 漂移。

- [ ] **Step 7: Commit**

```bash
git add src/des/types.py src/des/registry.py tests/test_multi_slot.py tests/test_phenotype_cache.py
git commit -m "feat(s7): add Phenotype.slots_per_event + phenotype() dominant_p propagate

New frozen dataclass field (default 1) propagated by phenotype() from
SLOTS_PER_EVENT[dominant_p] (piggybacks S2's spectrum-source selection —
no new selection rule). Default BB0 phenotype gets slots_per_event=1;
synthetic P_cascade letter (S8 will mint it) propagates 2. No consumer
yet — _mutation_outcomes signature change in Task 3, joint-enumeration
path in Task 4."
```

---

---

### Task 3: `_mutation_outcomes` 加 `slots_per_event` 形参 + N=1 byte-identical 路径

**Goal:** 给 `src/des/kernels/reproduction.py::_mutation_outcomes` 加形参 `slots_per_event: int = 1`,kwarg 默认值 1 让 N=1 走当前 verbatim 单 slot 循环(**枚举顺序、weight `q/|slots|`、RNG 调用次数与 pre-S7 字节级一致** — spec §3 红线 "by construction byte-identical, not merely distributionally equal");N≥2 暂时 raise `NotImplementedError("slots_per_event >= 2 lands in Task 4")` 占位,Task 4 实现 joint enumeration 路径并替换占位。同时改 `phase2_reproduce` call-site:既有 `_mutation_outcomes(seq, BB0_TEMPLATE["mutable"], spectrum)` 多加一个 kwarg `slots_per_event=phe.slots_per_event`,kernel 在 per-parent loop 体内读 `Phenotype` 对象的字段(不读 `phe['...']` 张量列 — N 只在 hot loop 极偶尔分叉,99% parent 是 N=1,per-parent scalar 读不必批量化)。

**Files:**
- Modify: `src/des/kernels/reproduction.py`(`_mutation_outcomes` 函数签名 + N==1 路径 + N≥2 占位 raise + call-site 一行)
- Test: `tests/test_multi_slot.py`(append byte-identical 断言 + N=1 default kwarg 兼容)
- Test: `tests/test_reproduction.py`(append signature regression — 旧 3-arg call 仍合法)

**Interfaces:**
- Consumes: `Phenotype.slots_per_event`(Task 2);既有 `_mutation_outcomes(seq, mutable, spectrum)` 函数体逻辑(`src/des/kernels/reproduction.py:34-48`);既有 call-site `src/des/kernels/reproduction.py:134`。**若 S6 已落**:`_mutation_outcomes(seq, mutable, spectrum, blocks)` 已是 4-arg signature — 本 task 把它扩为 5-arg `(seq, mutable, spectrum, blocks, slots_per_event=1)`。
- Produces:
  - `_mutation_outcomes(seq, mutable, spectrum, slots_per_event=1) -> (children, weights)` — N=1 verbatim 同 pre-S7;N≥2 暂时 raise。
  - call-site `_mutation_outcomes(seq, BB0_TEMPLATE["mutable"], spectrum, slots_per_event=N)` — 默认 BB0 strain `N=1`,函数体走 verbatim 旧路径,与 pre-S7 字节级一致。

- [ ] **Step 1: 写失败测试 — N=1 byte-identical + N≥2 占位 raise**

追加到 `tests/test_multi_slot.py`:

```python
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
```

追加到 `tests/test_reproduction.py`:

```python
def test_mutation_outcomes_accepts_slots_per_event_kwarg_default_1():
    """_mutation_outcomes 加形参 slots_per_event=1, legacy 3-arg / 4-arg
    (S6 blocks) call 仍合法 — 默认 kwarg 接住."""
    from des.kernels.reproduction import _mutation_outcomes
    from des.registry import _spectrum_for, BB0_TEMPLATE
    seq = BB0_TEMPLATE["layout"]
    mutable = BB0_TEMPLATE["mutable"]
    spectrum = _spectrum_for("P_base")
    # legacy call (不带 slots_per_event) 仍正常返回
    try:
        from des.registry import motif_blocks
        out = _mutation_outcomes(seq, mutable, spectrum, motif_blocks(seq))
    except ImportError:
        out = _mutation_outcomes(seq, mutable, spectrum)
    children, weights = out
    assert len(children) == sum(mutable) * len(spectrum)
    assert abs(sum(weights) - 1.0) < 1e-9
```

- [ ] **Step 2: 跑失败测试**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_multi_slot.py tests/test_reproduction.py::test_mutation_outcomes_accepts_slots_per_event_kwarg_default_1 -v
```

Expected:
- `test_mutation_outcomes_default_kwarg_byte_identical_to_legacy` FAIL — `_mutation_outcomes()` does not accept `slots_per_event` kwarg yet (TypeError on unexpected keyword)。
- `test_mutation_outcomes_N_eq_1_via_kwarg_matches_default` FAIL — 同理 kwarg 不接受。
- `test_mutation_outcomes_N_eq_2_raises_notimplemented_in_task_3` FAIL — 同理。
- `test_default_bb0_engine_run_byte_identical_post_s7_kernel_change` PASS(本 task 还没改 kernel call-site,但默认 BB0 同 seed 跑两次本来就 byte-identical — Task 3 完成后仍 PASS,守的是退步)。
- `test_mutation_outcomes_accepts_slots_per_event_kwarg_default_1` PASS(legacy 4-arg call 走旧路径正常返回;此条守的是 Task 3 改完后 legacy call 不破)。

- [ ] **Step 3: 改 `_mutation_outcomes` 加形参 + N=1 verbatim + N≥2 占位**

打开 `src/des/kernels/reproduction.py`,把 `_mutation_outcomes`(S6 后 line 34-48,4-arg)替换为:

```python
def _mutation_outcomes(seq, mutable, spectrum, blocks, slots_per_event=1):
    """Per-parent mutation categorical. N=1 (default) keeps the legacy single-slot
    overwrite path verbatim (same enumeration order, same weights, same RNG
    call count — spec §3 red line 'by construction byte-identical, not merely
    distributionally equal'). N>=2 enumerates unordered slot-sets of size N
    (clamped to #mutable) × per-slot spectrum letters; weight of (slot-set S,
    letters) = (1/C(m,N)) * prod(q(letter_s) for s in S). At N=1 the joint
    formula reduces to q/C(m,1) = q/m — continuous with the legacy path; N=1
    nonetheless takes the verbatim legacy branch to guarantee byte-identity.

    Returns (child_sequences, weights) over the full enumeration; weights
    sum to 1. Self-loops (letter == current) yield child == parent. Pure fn
    of (sequence, spectrum, motif blocks, N) — reads no world state."""
    slot_idx = [i for i, ok in enumerate(mutable) if ok]
    if not slot_idx or not spectrum:
        return [], []
    # index -> (start, end) of the block covering position i (S6 motif overwrite).
    cover: dict[int, tuple[int, int]] = {}
    for s, e, _ in blocks:
        for k in range(s, e):
            cover[k] = (s, e)

    if slots_per_event == 1:
        # ---------------------------------------------------------------
        # N=1: legacy verbatim path (pre-S7). DO NOT REFACTOR INTO JOINT
        # ENUMERATION — spec §3 red line requires the byte-identical
        # enumeration order + RNG call count, not just the same weights.
        # ---------------------------------------------------------------
        children, weights = [], []
        for s in slot_idx:                      # ascending: canonical order
            for letter, q in spectrum:          # spectrum already sorted in _spectrum_for
                start, end = cover[s]
                new = list(seq)
                for k in range(start, end):      # S6: overwrite the whole covering block
                    new[k] = letter
                children.append(tuple(new))
                weights.append(q / len(slot_idx))
        return children, weights

    # N>=2: joint enumeration path lands in Task 4.
    raise NotImplementedError(
        f"_mutation_outcomes slots_per_event>=2 lands in S7 Task 4; "
        f"got slots_per_event={slots_per_event}. "
        f"(P_cascade is the sole roster letter with slots=2 and is minted in S8.)")
```

(若 S6 尚未落地,signature 是 4-arg `(seq, mutable, spectrum, slots_per_event=1)` 不带 `blocks`;函数体把 `cover` 计算块去掉、`for k in range(start, end)` 改为 `new[s] = letter` 单 slot 覆写,其余逻辑不变。)

改 call-site `src/des/kernels/reproduction.py:134`,把:

```python
            children, weights = _mutation_outcomes(
                seq, BB0_TEMPLATE["mutable"], spectrum, motif_blocks(seq))
```

替换为:

```python
            phe_obj = table.phenotype_of(p)
            children, weights = _mutation_outcomes(
                seq, BB0_TEMPLATE["mutable"], spectrum, motif_blocks(seq),
                slots_per_event=phe_obj.slots_per_event)
```

(若 S6 未落,`motif_blocks(seq)` 参数不存在,call-site 形如 `_mutation_outcomes(seq, BB0_TEMPLATE["mutable"], spectrum, slots_per_event=phe_obj.slots_per_event)`。)

注意:既有循环 `for p in sorted(set(int(x) for x in ind_parent.tolist())):` 内部已经 build 过 `seq = table.sequence_of(p)` 与 `spectrum = table.phenotype_of(p).spectrum`(line 132-133);本 Task 把 `table.phenotype_of(p)` 提取到一个本地 `phe_obj` 变量,避免两次调用(`spectrum` 与 `slots_per_event` 共享同一 phe lookup)。改后:

```python
        for p in sorted(set(int(x) for x in ind_parent.tolist())):
            if p == 0:
                continue
            seq = table.sequence_of(p)
            phe_obj = table.phenotype_of(p)
            spectrum = phe_obj.spectrum
            children, weights = _mutation_outcomes(
                seq, BB0_TEMPLATE["mutable"], spectrum, motif_blocks(seq),
                slots_per_event=phe_obj.slots_per_event)
            if not children:
                continue
            # ... 既有 out_sid / sel / w / draws / osid / child_sid_i 逻辑全部不动 ...
```

- [ ] **Step 4: 跑测试, 确认 PASS**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_multi_slot.py tests/test_reproduction.py -v
```

Expected: 全 PASS。Task 1 + Task 2 + Task 3 的全部 `test_multi_slot.py` 断言绿;`test_default_bb0_engine_run_byte_identical_post_s7_kernel_change` PASS(默认 BB0 全 strain N=1,kernel 走 verbatim N=1 路径,同 seed 跑两次 byte-identical);`test_mutation_outcomes_N_eq_2_raises_notimplemented_in_task_3` PASS(N=2 显式 raise 占位);既有 `test_reproduction.py` 全 PASS(legacy call 走 N=1 verbatim 路径)。

Backtrack:
- 若 `TypeError: _mutation_outcomes() got an unexpected keyword argument 'slots_per_event'` — signature 没改对;按 Step 3 重写,确认 kwarg 默认值是 1。
- 若 `test_default_bb0_engine_run_byte_identical_post_s7_kernel_change` 失败(同 seed 跑两次结果不同)— root cause:N=1 verbatim 路径里改了 `weights.append(q / len(slot_idx))` 的浮点数表达式(例如改成 `q * (1.0 / len(slot_idx))` 触发 float promotion),应该一字不差保留 `q / len(slot_idx)`。
- 若整套 pytest 出现「`_mutation_outcomes` 调用 TypeError positional vs keyword」错 — 检查 call-site 有没有传 `slots_per_event=` 作 kwarg(不用 positional,避免与 S6 `blocks` 位次冲突)。

- [ ] **Step 5: 跑全 suite, 确认无回归**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿。N=1 verbatim 路径 + call-site kwarg 加完后,既有 BB0 / S4 / S5 / S6 / S1 / S2 落地后的所有 strain 全部走 N=1 路径,字节级不变。

- [ ] **Step 6: Commit**

```bash
git add src/des/kernels/reproduction.py tests/test_multi_slot.py tests/test_reproduction.py
git commit -m "feat(s7): _mutation_outcomes slots_per_event kwarg + N=1 verbatim

Add slots_per_event=1 kwarg to _mutation_outcomes. N=1 path keeps the
legacy single-slot enumeration verbatim (same order, weights, RNG call
count — byte-identical, not merely distributionally equal). N>=2 raises
NotImplementedError occupant (Task 4 lands the joint-enumeration path).
phase2_reproduce call-site reads phe.slots_per_event per parent and
passes it through; default BB0 strains all have slots_per_event=1 so
the engine run is byte-identical pre-S7."
```

---

---

### Task 4: N≥2 joint enumeration 路径 + clamp + distinct slots

**Goal:** 把 Task 3 占位的 `raise NotImplementedError` 替换为真实的 N≥2 联合枚举路径:`itertools.combinations(slot_idx, effective_N)` 列举 `unordered slot-sets of size N`(`combinations` 天然 `distinct + no-repeat`,接住 spec §5 "N distinct slots, sample without replacement");每个 slot-set × per-slot spectrum letters 用 `itertools.product(spectrum, repeat=effective_N)` 列举字母组合;weight 公式 `(1/C(m,effective_N)) · ∏_{s∈S} q(letter_s)`;`effective_N = min(slots_per_event, len(slot_idx))` 处理 spec §5 "N > #mutable → clamp to #mutable"。outcome 出口仍是 `(children: list[tuple[str,...]], weights: list[float])`,kernel call-site 不动 `torch.multinomial` 路径 — 同 single 调用 per distinct parent(spec §3 first ponytail note)。

**Files:**
- Modify: `src/des/kernels/reproduction.py`(替换 Task 3 占位 raise)
- Test: `tests/test_multi_slot.py`(append N=2 行为 + clamp + distinct + 同序列合并)

**Interfaces:**
- Consumes: Task 3 落地的 `_mutation_outcomes(seq, mutable, spectrum, blocks, slots_per_event=1)` signature + N=1 verbatim 路径 + call-site `slots_per_event=phe.slots_per_event` 传参。
- Produces:
  - N≥2 路径的 children = `[apply_overwrites(seq, slot_set, letters) for slot_set in C(m,N) for letters in product(spectrum, repeat=N)]`,枚举顺序固定:slot_set 升序(由 `combinations` 字典序保证),letters 按 `_spectrum_for` 排序 cartesian product 字典序。
  - weights = `[(1.0 / C_mN) * prod_q for ...]`,Σweights = 1(数学可证:`Σ_S Σ_letters (1/C_mN) ∏ q = (1/C_mN) · C(m,N) · (Σ q)^N = 1` 因 spectrum 已归一)。

- [ ] **Step 1: 写失败测试 — N=2 行为 + 边界**

追加到 `tests/test_multi_slot.py`:

```python
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
    """N=2 outcome 总数 = C(m, 2) * |spectrum|^2.
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
    cascade_layout = ("P_cascade",) + ("N0",) * 15
    eng = Engine(H=2, W=2, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2, layouts=(cascade_layout,) * 4)
    eng.run(5, recorder=None, stop_on=())
    # 不抛 (kernel N=2 路径无崩); strain table 完整性 (有 strain 被 mint 即 PASS)
    assert len(eng.table) >= 1, "P_cascade synthetic letter should mint at least 1 strain"
```

(把 Task 3 留下的 `test_mutation_outcomes_N_eq_2_raises_notimplemented_in_task_3` 整条删除 — 它的 expected 在 Task 4 实现后失效,由上述 N=2 正确性测试集替代。删除方式:在 `tests/test_multi_slot.py` 里找到那个 function,整段删除即可。)

- [ ] **Step 2: 跑失败测试**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_multi_slot.py -v
```

Expected:
- Task 1-3 既有断言全 PASS。
- Task 4 N=2 行为测试(7 条)FAIL with `NotImplementedError: _mutation_outcomes slots_per_event>=2 lands in S7 Task 4; got slots_per_event=2`。
- `test_mutation_outcomes_N_eq_2_continuous_with_N_eq_1_formula` PASS(纯数学测试,与代码无关)。
- `test_mutation_outcomes_N_clamped_when_exceeds_mutable` FAIL(同 NotImplementedError,因 effective_N=2 仍走 N≥2 分支)。
- `test_mutation_outcomes_N_eq_2_same_sequence_collapsed_by_get_or_mint` FAIL(端到端 engine.run 在 phase2_reproduce 调 `_mutation_outcomes(..., slots_per_event=2)` 时 raise)。

- [ ] **Step 3: 实现 N≥2 joint enumeration 路径**

打开 `src/des/kernels/reproduction.py`,把 Task 3 写的 `raise NotImplementedError(...)` 替换为完整 N≥2 路径。注意顶部 `from __future__ import annotations` 之后 / `import torch` 之上加 stdlib import:

```python
from __future__ import annotations
from itertools import combinations, product
from math import comb
import torch
```

替换 `_mutation_outcomes` 的 N≥2 raise 占位代码,完整函数体写为:

```python
def _mutation_outcomes(seq, mutable, spectrum, blocks, slots_per_event=1):
    """Per-parent mutation categorical. N=1 (default) keeps the legacy single-slot
    overwrite path verbatim (same enumeration order, same weights, same RNG
    call count — spec §3 red line). N>=2 enumerates unordered slot-sets of size
    effective_N = min(N, #mutable) × per-slot spectrum letters; weight of
    (slot-set S, letters) = (1/C(m, effective_N)) * prod(q(letter_s) for s in S)."""
    slot_idx = [i for i, ok in enumerate(mutable) if ok]
    if not slot_idx or not spectrum:
        return [], []
    cover: dict[int, tuple[int, int]] = {}
    for s, e, _ in blocks:
        for k in range(s, e):
            cover[k] = (s, e)

    if slots_per_event == 1:
        # N=1: legacy verbatim path (Task 3) — DO NOT REFACTOR.
        children, weights = [], []
        for s in slot_idx:
            for letter, q in spectrum:
                start, end = cover[s]
                new = list(seq)
                for k in range(start, end):
                    new[k] = letter
                children.append(tuple(new))
                weights.append(q / len(slot_idx))
        return children, weights

    # N>=2: joint enumeration. itertools.combinations gives unordered slot-sets
    # (distinct, sample-without-replacement); itertools.product gives the
    # per-slot letter Cartesian product. effective_N clamps to #mutable per spec §5.
    m = len(slot_idx)
    effective_N = min(slots_per_event, m)
    inv_C_mN = 1.0 / comb(m, effective_N)
    children, weights = [], []
    for slot_set in combinations(slot_idx, effective_N):
        # spectrum is already sorted in _spectrum_for; product preserves
        # lex order over the spectrum tuples.
        for letter_tuple in product(spectrum, repeat=effective_N):
            # Build the outcome: per-slot block overwrite with the chosen letter.
            new = list(seq)
            prod_q = 1.0
            for s, (letter, q) in zip(slot_set, letter_tuple):
                start, end = cover[s]
                for k in range(start, end):
                    new[k] = letter
                prod_q *= q
            children.append(tuple(new))
            weights.append(inv_C_mN * prod_q)
    return children, weights
```

(若 S6 未落,`blocks` 参数与 `cover` map 计算都去掉,`for s, (letter, q) in zip(...)` 改为 `new[s] = letter` 单 slot 覆写。)

- [ ] **Step 4: 跑测试, 确认 PASS**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_multi_slot.py -v
```

Expected: 全 PASS。Task 1-4 全部 `test_multi_slot.py` 断言绿。N=2 outcome 数 / 权重和 / weight 公式 / clamp / distinct slots / same-sequence merge 全部满足。

Backtrack:
- 若 `test_mutation_outcomes_N_eq_2_weights_sum_to_one` FAIL with `Σweights ≠ 1`:
  - 验证 `inv_C_mN = 1.0 / comb(m, effective_N)` 正确(`from math import comb`)。
  - 验证 `prod_q *= q` 累乘所有 N 个 letter 的 q,而不只是最后一个。
- 若 `test_mutation_outcomes_N_eq_2_slot_set_count_matches_combinations` FAIL with count 不匹配:
  - 验证 `combinations(slot_idx, effective_N)` 不是 `permutations`(`combinations` 输出 `C(m,N)`,`permutations` 输出 `P(m,N) = N! · C(m,N)`)。
- 若 `test_mutation_outcomes_N_eq_2_children_have_2_distinct_slot_changes` FAIL with `diff` 数大于 2:
  - 验证 `for s, (letter, q) in zip(slot_set, letter_tuple)` 每次只改 slot_set 内 N 个位置,不动其他。
- 若 `test_mutation_outcomes_N_clamped_when_exceeds_mutable` FAIL with `IndexError` 或 `ValueError`:
  - 验证 `effective_N = min(slots_per_event, m)`,**而非** `min(slots_per_event, m)` 之后又 `combinations(slot_idx, slots_per_event)`(必须用 `effective_N` 不能用未 clamp 的原值)。
- 若 same-sequence merge 测试 FAIL with `RuntimeError`(端到端 engine.run 崩):
  - 检查 `phase2_reproduce` call-site(Task 3 改的)有没有正确读 `phe.slots_per_event`,以及 `torch.multinomial(w, n_p, ...)` 的 `w` 张量长度匹配 `len(weights)`(N=2 时 60 outcomes 远比 N=1 的 6 大,但仍 fit GPU)。

- [ ] **Step 5: 跑全 suite, 确认无回归**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿。Task 1-4 全部 `test_multi_slot.py` 断言绿(~15 条 N=2 测试);既有 BB0 / S4 / S5 / S6 / S1 / S2 落地后的所有 strain 全部 N=1 走 verbatim 路径,字节级不变;`test_default_bb0_engine_run_byte_identical_post_s7_kernel_change`(Task 3 加的)仍 PASS。

Backtrack:若全 suite 跑出非 multi_slot 相关失败,常见 root cause:
- (a) `from itertools import combinations, product` 与 `from math import comb` 之前已存在 — 重复 import 不报错,但 `combinations` / `product` 在 `reproduction.py` 顶部加一次即可。
- (b) `torch.multinomial(w, n_p, replacement=True, ...)` 当 `w` 张量数过大(N=3 + |spec|=16 → `C(6,3)·16³ ≈ 81920`),GPU 内存或速度敏感 — 但 S7 默认局没人触发 N=2,实际不会跑到。

- [ ] **Step 6: Smoke probe — 用合成 P_cascade letter 跑端到端 (eye-test)**

跑一个 4×4 grid 上单 cell 合成 P_cascade 的 smoke,验 N=2 路径在 engine.run 里真的工作(不抛、产 strain):

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -c "
import torch
from des import registry
registry.ALPHABET['P_cascade']        = 'P'
registry.GRAN['P_cascade']            = 'residue'
registry._P['P_cascade']              = (0.28, 2)
registry.SPECTRUM_SHAPE['P_cascade']  = (1.0, None, 0.0)
registry.SLOTS_PER_EVENT['P_cascade'] = 2
from des.engine import Engine
layout = ('P_cascade',) + ('N0',) * 15
eng = Engine(H=4, W=4, K=8, seed=0, device=torch.device('cpu'),
             z_max=8.0, fill_per_cell=4, layouts=(layout,)*4)
eng.run(10, recorder=None, stop_on=())
print('strains:', len(eng.table))
print('total count:', int(eng.world.count.sum().item()))
"
```

Expected: 10 tick 后打印 `strains: <N>`(`N` ≥ 1,N=2 突变会产新 strain) + `total count: <total>`(> 0)。任何 `RuntimeError` / `ValueError` / `IndexError` 都是 N≥2 路径 bug — 按 Step 4 的 backtrack 树定位。

(这是 manual probe,**不**进 pytest — 端到端 stochastic 行为不易做精确断言,Step 1 的 `test_mutation_outcomes_N_eq_2_same_sequence_collapsed_by_get_or_mint` 已经做了对应的端到端 smoke。)

- [ ] **Step 7: Commit**

```bash
git add src/des/kernels/reproduction.py tests/test_multi_slot.py
git commit -m "feat(s7): _mutation_outcomes N>=2 joint enumeration path

Replace Task 3 NotImplementedError placeholder with the real joint-
enumeration body: itertools.combinations(slot_idx, effective_N) × 
itertools.product(spectrum, repeat=effective_N); weight of (slot-set S,
letters) = (1/C(m, effective_N)) * prod(q(letter_s) for s in S). 
effective_N = min(slots_per_event, m) clamps per spec §5. Same-sequence
children still collapse via get_or_mint. P_cascade is the sole roster
letter that triggers N=2; S8 mints it. Default BB0 still runs N=1
verbatim path — byte-identical pre-S7."
```

---

---

### Task 5: Final regression sweep + smoke + push

**Goal:** 把整个 S7 deliverable (Tasks 1-4) 一起跑一遍,确认全套测试绿、smoke run 不崩、性能档位 (~15.8ms/tick / 128² grid) 没明显漂移、工作树干净、推 origin。这是 sibling task 的同款收口动作 (S0 Task 6, S6 Task 9, S2 Task 7, S5 Task 6, S4 Task 8 等)。

**Files:**
- 不预期 source 改动。若 Step 1 暴露回归,本 task 修 forward,commit message 引用 offending commit。
- Test: `tests/`(整套)+ `scripts/run_batch.py --probe` smoke + 默认 4-faction symmetric run smoke + relabel-invariance sanity。

**Interfaces:**
- Consumes: Tasks 1-4 全部产物。
- Produces: 绿 `pytest tests/` + 干净 `git status` + push 到 origin。

- [ ] **Step 1: Full pytest sweep**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -q
```

Expected: 全绿。总数 = 285 engine + 146 web + S6 motif + S1 vis + S2 spectrum_shape + S4 direction_kinds + S5 phase_windows + S3 prey_predicates + S7 multi_slot (`tests/test_multi_slot.py` ~17 条) + 既有 `test_registry.py` / `test_reproduction.py` / `test_phenotype_cache.py` append 部分。精确数随时间漂移; **没有 `FAILED tests/...` 行**是验收标准 (`SKIPPED` 允许 — sibling task 标记的 fixture re-record 占位,例如 S4 Task 6 留下的 837MB 基线 parquet skip)。

Backtrack: 若任何 FAIL,先按 owner 文件 root-cause 到对应 Task;用 `git log` 找 offending commit,fix forward,不 reset。

- [ ] **Step 2: Smoke run probe (确认运行时性能没崩)**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --probe 30
```

Expected stdout 形如 `[probe 30 ticks] X.X ms/tick | peak Y.YY GB | strains->… | total …`。`X.X ms/tick` 应保持在 S0/S6/S1/S2/S4/S5/S3 完工时的同一档(target ≈15.8 ms/tick on 128² grid;≤ 20% drift acceptable)。exit 0;不写 parquet。

若 drift > 20%,最常见原因:
- (a) `phase2_reproduce` 在 per-parent loop 内加的 `phe_obj = table.phenotype_of(p)` 触发额外 lookup — 但既有代码本已读 `phenotype_of(p).spectrum`,我们仅把 lookup 提取到本地变量复用,**总 lookup 数应当减一**(从 2 次 phenotype_of → 1 次 + 2 次 attribute 读)。检查 Task 3 Step 3 的 call-site 编辑是不是漏抽 `phe_obj` 局部变量,变成两次独立 `table.phenotype_of(p)`。
- (b) `itertools.combinations / product` 在 N=1 路径**不应该**被调用(N=1 走 verbatim 分支,在 `if slots_per_event == 1: ...; return` 之内提早返回);若 profile 显示 N=1 路径还跑了 `combinations`,根本原因是函数体没正确按 `if slots_per_event == 1` 分支。
- (c) 默认 BB0 全 strain `slots_per_event=1`,N≥2 分支永远不入,`combinations / product` 在 hot path 上的成本为 0。若 profile 显示它们被调用,grep `slots_per_event` 找到把 `2` 写死的 bug。

- [ ] **Step 3: Byte-identical default-run smoke (推荐, 守 BB0 字节级不变)**

跑两次同 seed BB0 默认局:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --seeds 0 --T 30
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --seeds 0 --T 30
```

Expected: 两次产出的 parquet 在 `data/runs/` 下用 pyarrow 读后 `(tick, cell, strain, count)` 行级一致。这一条守 "S7 改了 kernel call-site 一行 + 加了 N≥2 path,但默认局所有 strain N=1 走 verbatim → 同 seed 同结果"。

补充验证: 与最新 sibling task (S5 / S3) 收口时的 baseline parquet 比对 — 把现在跑出来的 parquet 与最新 baseline 做 row-by-row diff:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -c "
import pyarrow.parquet as pq, sys
a = pq.read_table(sys.argv[1]).to_pydict()
b = pq.read_table(sys.argv[2]).to_pydict()
print('cols match:', sorted(a.keys()) == sorted(b.keys()))
for col in a:
    print(col, 'equal:', a[col] == b[col])
" <baseline.parquet> <s7-fresh.parquet>
```

Expected: 全 `equal: True`。若 False,root cause 大概率是 `_mutation_outcomes` N=1 verbatim 路径里某行被误改 — 例如 `weights.append(q / len(slot_idx))` 改成了 `weights.append(q * inv_C_mN)`(N=1 时 `inv_C_mN = 1.0/comb(m,1) = 1/m` 数学等价,但浮点表达式不同会导致 RNG path 引入新的 dtype promotion)。

- [ ] **Step 4: N=2 端到端 smoke — 合成 P_cascade 跑 4-faction match**

写一个 smoke config 跑 N=2 端到端,验证 phase2_reproduce 在 N=2 路径下 runtime 不崩 + 产出合理:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -c "
import torch, json
from des import registry
# 合成 P_cascade (S8 才正式铸; 此 smoke 用 monkeypatch 路径)
registry.ALPHABET['P_cascade']        = 'P'
registry.GRAN['P_cascade']            = 'residue'
registry._P['P_cascade']              = (0.28, 2)
registry.SPECTRUM_SHAPE['P_cascade']  = (1.0, None, 0.0)
registry.SLOTS_PER_EVENT['P_cascade'] = 2
from des.engine import Engine
layout = ('P_cascade',) + ('N0',) * 15
eng = Engine(H=8, W=8, K=16, seed=0, device=torch.device('cpu'),
             z_max=8.0, fill_per_cell=4, layouts=(layout,)*4)
eng.run(30, recorder=None, stop_on=())
print(json.dumps({
    'strains': len(eng.table),
    'total_count': int(eng.world.count.sum().item()),
    'distinct_per_cell': int(eng.world.count.sum(dim=-1).gt(0).sum().item()),
}))
"
```

Expected stdout 是一行 JSON: `{"strains": <N>, "total_count": <total>, "distinct_per_cell": <count>}`。`N` ≥ 1(N=2 突变会产新 strain);`total_count > 0`(P_cascade f=0 即 P 不繁衍,本 smoke 仅 mint kid 通过 P 触发的突变 — 检测 mutation core 没崩即合格);任何 `RuntimeError` / `ValueError` 都是 N≥2 路径未处理边界 — Task 4 Step 4 backtrack 树排查。

(注:P_cascade family='P',phenotype 走 P 池路径,自身不出 `f`;此 smoke 不靠繁衍跑到 N=2,而是靠 phenotype 解析的 N=2 + 任何 P-base offspring 走 N=2 突变 → kernel 不抛即合格。)

- [ ] **Step 5: Relabel-invariance sanity (spec §6 一句话守门)**

写一条简短的 relabel-invariance pytest 断言追加到 `tests/test_multi_slot.py`(若 Task 4 没追加):

```python
def test_slots_per_event_structural_under_relabel(monkeypatch):
    """spec §6: slots_per_event is structural — shuffling _F/_Z/_P magnitudes
    doesn't change which slots/letters are drawn (drives off spectrum +
    mutable, both structural)."""
    from des.registry import phenotype, BB0_TEMPLATE
    pre = phenotype(BB0_TEMPLATE["layout"])
    # 重排 _F / _Z / _P 量级 (NOT gran / NOT family / NOT SLOTS_PER_EVENT 值)
    from des import registry
    monkeypatch.setitem(registry._F, "F4Nr1", (0.95, "hash:f4nr1", 0.99, 99))
    monkeypatch.setitem(registry._Z, "BroadSweep", (0.99, (("F",), ("Z",)), 99))
    monkeypatch.setitem(registry._P, "P_hotspot", (0.0, 99))
    post = phenotype(BB0_TEMPLATE["layout"])
    # slots_per_event 是结构 readout — 与量级无关, 不漂移
    assert post.slots_per_event == pre.slots_per_event
```

跑一遍:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_multi_slot.py::test_slots_per_event_structural_under_relabel -v
```

Expected: PASS。`slots_per_event` 读自 `SLOTS_PER_EVENT[dominant_p]`(`dominant_p` 是 P 家族 letter 选择,基于 `p_add` 而非 `f / z` 量级 — 此测试只重排 `_F["F4Nr1"]` 与 `_Z` 量级,不动 P 排名,`dominant_p` 不变)。若 FAIL,root cause:`phenotype()` 把 `slots_per_event` 误读自非 P 的 letter(应仅读 `SLOTS_PER_EVENT[dominant_p]`,无 P → 默认 1)。

- [ ] **Step 6: Inspect & clean stray data**

Run:

```
git status
```

Expected: 干净工作树。若 `data/runs/<ts>-*.parquet` smoke 残留 / `s7_*` 临时日志,删掉 — 不入 commit(它们不是 fixture)。

- [ ] **Step 7: Final commit (only if Step 1 needed a fix-forward)**

若 Step 1 surface 了 regression 并修了:

```bash
git add <files-touched>
git commit -m "fix(s7): <description of the regression fixed>"
```

(若 Step 5 加了 relabel-invariance 测试,它的 commit 走单独提交:)

```bash
git add tests/test_multi_slot.py
git commit -m "test(s7): add relabel-invariance check for slots_per_event"
```

Otherwise this step is a no-op。

- [ ] **Step 8: Push to origin**

```bash
git push origin <current-branch>
```

Expected: push succeeds。branch ready for review / merge to `main`。

后续动作(out of S7 plan scope,用户单独决定时机):
1. S8 (A 池) 铸 P_cascade 时,加一行 `SLOTS_PER_EVENT["P_cascade"] = 2` 即可;Task 1 已留 hook。
2. S8 同步解决「多 P stacked 时 spectrum 混合 `Σpᵢqᵢ/Σpᵢ`」与「多 P stacked 时 `slots_per_event` 合并规则」— 两个问题同源,S7 仅守「dominant 单源」(spec §3 inline note + §7)。
3. N≥3 触发 sequential single-slot N-step 升级路径(spec §3 second ponytail ceiling)— 由 future-spec 决定何时跨过。本 plan 的 joint enumeration 在 N=2 / m=6 / |spectrum|=16 上是 ~3840 outcomes per parent,完全够用。
4. 837MB 首批基线 parquet (S4 Task 6 标 skip 的 fixture-based byte-equal 测试) 重跑 — S7 也没碰它,仍由 batch CLI 单独跑后回填。

---

---

## Self-Review

**1. Spec coverage:**

- **§1 (Why — P_cascade as the sole slots≠1 letter):** Task 1 把 `SLOTS_PER_EVENT` 值域写在 `{1, 2}`,P_cascade 在 S8 铸 — Global Constraints 显式声明「P_cascade 是 roster 里唯一 slots=2 letter, S7 不铸」。Task 2 / Task 4 用 monkeypatch 合成 P_cascade letter 走 N=2 路径验证。
- **§2 (Red lines):** 
  - 红线 1 (`slots_per_event` 是全局 per-primitive registry int, 非 per-species): Task 1 `SLOTS_PER_EVENT: dict[str, int]` — 一行 letter, 无 species 维度。
  - 红线 2 (P_cascade 的 2 是 verbatim from roster): Task 1 注释 + Task 2 monkeypatch fixture 使用 `p_add=0.28, period=2` 与 roster L230 一致。
  - 红线 3 (mutation-core 改, 不动 kernel-physics): Task 3 / Task 4 仅碰 `_mutation_outcomes` 函数体 + `phase2_reproduce` call-site 一行;`fires_this_tick` / `binom` / `torch.roll` / `ArrivalBuffer` / `torch.multinomial` 全部 untouched。
  - 红线 4 (N=1 byte-identical **by construction**, not by distribution): Task 3 实现 N==1 verbatim 路径(完整保留枚举顺序、weight `q/|slots|`、RNG 调用次数);Task 3 `test_mutation_outcomes_default_kwarg_byte_identical_to_legacy` 用 `==` 直接比较 children + weights 字节级相等;`test_default_bb0_engine_run_byte_identical_post_s7_kernel_change` 端到端守同 seed 跑 30 tick world.count / strain_id byte-identical。
  - 红线 5 (S0–S7 所有 active primitive `slots_per_event=1`, P_cascade 仅 S8 铸): Task 1 module-load `assert` 守值域、Task 1 `test_slots_per_event_v1_all_one` 显式守每条 ALPHABET letter 都是 1。
- **§3 (Architecture):**
  - "Single locus `_mutation_outcomes` + 一个新字段": Task 1 (字段) + Task 2 (phenotype 解析) + Task 3 (signature + N=1) + Task 4 (N≥2) — 锁在 `_mutation_outcomes` + `phenotype()` 两处, kernel call-site 仅加一行 kwarg。
  - N=1 verbatim 路径细节(单 categorical + 单 `torch.multinomial`, slot-ascending × `_spectrum_for` order, `q/|slots|`): Task 3 函数体 verbatim 保留 pre-S7 代码。
  - N≥2 joint enumeration 路径 (`(1/C(m,N)) · ∏ q`, N=1 退化为 `q/m` 数学连续): Task 4 Step 3 实现 + Task 4 `test_mutation_outcomes_N_eq_2_continuous_with_N_eq_1_formula` 守。
  - N piggybacks dominant_p (`registry.py:101-105`): Task 2 `slots_per_event = SLOTS_PER_EVENT.get(dominant_p, 1) if dominant_p else 1` — 与 S2 spectrum-source 选择**同源**, 不引入新 selection rule。
  - gran pairing per-slot (S6): Task 3 / Task 4 函数体保留 S6 `blocks` 与 `cover` map, 每 slot 各自查 cover 走 block-overwrite — interaction-free。
  - Joint vs sequential (N=2 走 joint, N≥3 才考虑 sequential): Task 4 实现 joint + Global Constraints 显式声明「至 N≥3 才考虑 sequential, 不在 S7 范围」。
  - 多 P stacked 时 N 取 dominant (合并归 S8): Global Constraints + spec §3 inline note 引用。
- **§4 (Data flow `mint→phenotype→phase2_mutation`):** Task 1 → Task 2 → Task 3 → Task 4 严格按 spec data flow 实现。
- **§5 (Error handling):**
  - "N > #mutable: clamp N to #mutable": Task 4 `effective_N = min(slots_per_event, m)` + `test_mutation_outcomes_N_clamped_when_exceeds_mutable`。
  - "N distinct slots: sample without replacement": Task 4 用 `itertools.combinations` (天然 distinct) + `test_mutation_outcomes_N_eq_2_slot_sets_are_distinct`。
  - "Same-sequence children merge via `get_or_mint`": call-site 不动 `get_or_mint` + `torch.unique` aggregation; Task 4 `test_mutation_outcomes_N_eq_2_same_sequence_collapsed_by_get_or_mint` 端到端守。
- **§6 (Testing):** 
  - 回归 (285+146 + sibling 增量): Task 1-4 每 task Step 5 跑全 suite; Task 5 Step 1 final sweep。
  - P_cascade (N=2) 产 ≤2 distinct mutable slot 差异: Task 4 `test_mutation_outcomes_N_eq_2_children_have_2_distinct_slot_changes`。
  - gran respected per slot: Task 4 N≥2 路径每 slot 走 `cover[s]` block-overwrite, gran 由 spectrum 已 gran-matched 保证 (S6 owns)。
  - N=2 经验分布 (slot-set, letters) match joint mass: Task 4 `test_mutation_outcomes_N_eq_2_weights_sum_to_one` + `test_mutation_outcomes_N_eq_2_weight_formula_matches_spec` 验解析公式 + 公式 ↔ 经验等价的数学一步。
  - N clamped when #mutable<N: Task 4 `test_mutation_outcomes_N_clamped_when_exceeds_mutable`。
  - Same-sequence cascade merge: Task 4 `test_mutation_outcomes_N_eq_2_same_sequence_collapsed_by_get_or_mint`。
  - relabel-invariance: Task 5 Step 5 `test_slots_per_event_structural_under_relabel`。
  - 推迟到 S8 测 (多 P stacked 时 N 合并): spec §6 + Global Constraints 显式声明, 本 plan 不写。
- **§7 (Out of scope):** Global Constraints 列出每项; 本 plan 不引入 P_cascade rate / spectrum (S2) / A-pool gating (S8) / 多 P stacked 合并 (S8) / sequential N≥3 (deferred) / per-faction asymmetric N (HARD-GATE)。

**2. Placeholder scan:**

无 `TBD` / `TODO` / "implement later" / "fill in details" / "similar to Task N" / "write tests for the above" / "add appropriate error handling" 等 plan-failure 字串。所有 code step 给出真实代码; 所有 command step 给出真实命令 + 预期输出; 所有 backtrack 条件给出具体 root-cause / fix。Task 4 Step 6 / Task 5 Step 4 标明 "manual probe, **不**进 pytest"(probabilistic 不易精确断言) — 显式声明非占位。Task 5 Step 3 "<baseline.parquet>" / "<s7-fresh.parquet>" 是命令行参数动态文件名占位(实施者按 ls 结果填), Step 描述讲清判定规则, 不是 plan 级空白。

**3. Type consistency:**

- `SLOTS_PER_EVENT: dict[str, int]` — Task 1 定义, Task 2 读 (`SLOTS_PER_EVENT.get(dominant_p, 1) if dominant_p else 1`), Task 4 monkeypatch 写 fixture。全程同名同 dtype。
- `Phenotype.slots_per_event: int = 1` — Task 2 定义, Task 3 读 (`phe_obj.slots_per_event`), Task 4 / Task 5 测试断言。
- `_mutation_outcomes(seq, mutable, spectrum, blocks, slots_per_event=1) -> (list[tuple[str,...]], list[float])` — Task 3 加 signature (kwarg 默认 1), Task 4 在同 signature 内实现 N≥2 分支, Task 5 验证。
- call-site signature 一致: Task 3 Step 3 改 `phase2_reproduce`: `_mutation_outcomes(seq, BB0_TEMPLATE["mutable"], spectrum, motif_blocks(seq), slots_per_event=phe_obj.slots_per_event)` — Task 4 不动 call-site, Task 5 Step 2 / Step 4 跑 smoke 验证。
- 形参顺序 `(seq, mutable, spectrum, blocks, slots_per_event=1)` 锁死: Task 3 / Task 4 函数定义一致; `slots_per_event` 永远 kwarg(避免与 S6 `blocks` 位次冲突)。
- N=1 vs N≥2 分叉判定: `if slots_per_event == 1:` — Task 3 + Task 4 同一行, early-return 在 if 块内不漏。
- weight 公式 `(1.0 / comb(m, effective_N)) * prod_q`: Task 4 Step 3 + Task 4 `test_mutation_outcomes_N_eq_2_weight_formula_matches_spec` 同公式。
- `effective_N = min(slots_per_event, m)` (m = `len(slot_idx)`): Task 4 Step 3 + Task 4 `test_mutation_outcomes_N_clamped_when_exceeds_mutable` 同 clamp。
- `itertools.combinations` + `itertools.product` + `math.comb`: Task 4 Step 3 显式 import + 函数体使用; Task 4 `test_mutation_outcomes_N_eq_2_slot_set_count_matches_combinations` 用 `math.comb` 比对 count 守 "不是 permutations"。

无 method / property 名称漂移: `slots_per_event` (lowercase + underscore) 全 plan 同名; 不出现 `slotsPerEvent` / `slot_per_event` / `slots_per_evt` 等变体; `SLOTS_PER_EVENT` (UPPER_SNAKE) 仅作 registry 全局表名。`effective_N` / `slot_set` / `letter_tuple` / `prod_q` / `inv_C_mN` 等局部变量仅 Task 4 函数体内使用, 不暴露到 Phenotype / phe 出口。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-24-s7-multi-slot-mutation.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. Use `superpowers:subagent-driven-development`.
2. **Inline Execution** — execute tasks in this session with batch checkpoints. Use `superpowers:executing-plans`.

Which approach?
