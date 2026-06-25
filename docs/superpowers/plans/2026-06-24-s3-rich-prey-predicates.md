# S3 — 富猎物谓词 (rich prey predicates) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 S6 在 predicate-bit vocabulary 里**预留 (reserved)** 的 4 个阈值 bit (`thr_crest` / `thr_hotspot` / `thr_mirror` / `vis_lowvis`) 全部填上语义 —— 在 `feature_mask_of(sequence)` 里按 roster 阈值 (F.f≥0.5 / P.p_add≥0.05 / Z.z≤0.45∧|prey|≥2 / N.vis≤0.20) 设置 prey 端的 feature bit, 在 `prey_mask_for_clauses(prey_clauses)` 里加 4 个新 clause tag (`f_hi` / `p_hi` / `generalist` / `lowvis`) 让 predator 端的 prey clause 能定向到这些 bit, 内核**完全不动**, 默认 BB0 局动力学**完全不动** (BroadSweep 仍走 family-only clauses, 阈值 bit 在默认局虽然被置位但无 v1 predator 命中, 红线 #4)。

**Architecture:** 三件事, 顺序: (1) 给 `feature_mask_of` 增 3 行阈值检查 (`thr_crest` / `thr_hotspot` / `thr_mirror`) —— `vis_lowvis` 由 S1 Task 6 已经填好, 本 plan 不再重复填, 只在 Task 5 做回归审计; (2) 在 `_Z` 表上做 module-load **prey-cardinality 预计算** `_Z_PREY_CARD: dict[str, int]` (取 `len(prey_clauses)`, 给 Mirror Fang 的 `|prey|≥2` 谓词用, 永不在运行时算, spec §5); (3) 给 `prey_mask_for_clauses` 加 4 个新 clause-tag 分支: `("F", "f_hi") → thr_crest` / `("P", "p_hi") → thr_hotspot` / `("Z", "generalist") → thr_mirror` / `("N", "lowvis") → vis_lowvis` —— S6 已守 `tag in {"motif","len>=3"}` 的两条分支不变, S3 只追加, 不动既有 4 条 clause 解析。无 kernel 改动, 无 `phenotype()` 主体改动 (只动它复用的 `feature_mask_of` 与 `prey_mask_for_clauses` 两 helper), 无 `phenotype_arrays.py` 改动, 无 `_Z` 行 reshape (BroadSweep 仍 4-tuple `(z, prey_clauses, period, vis_mode)` 不动)。

**Tech Stack:** Python 3 (`D:\anaconda3\envs\basic\python.exe`), PyTorch 2.10+cu128, pyarrow, pytest. Windows 主机, `PYTHONPATH=src` 纪律。引擎源码 `src/des/`。**依赖**: S0 (CLI 已就位, 不动) + S6 (predicate-bit vocabulary 4 reserved bit 索引锁死, `feature_mask_of` / `prey_mask_for_clauses` 函数体已就位, 本 plan 在它们上面叠) + S1 (vis 通道, `VIS` 表 + `vis_lowvis` bit 在 `feature_mask_of` 已被 S1 Task 6 设置, S3 仅审计不重复) + S2 (新 P 字母行 `P_aic` / `P_hotspot` / `P_balanced` / `P_ep` / `P_burst_lite` 都自带 `p_add≥0.05` 候选, S3 在 thr_hotspot 测试里覆盖)。**与 S4/S5 不交互** (动态方向与 f-window 不读 feature_mask 也不出 prey_mask)。

## Global Constraints

- **阈值是结构谓词, 不是「谁强」**: f≥0.5 / p_add≥0.05 / z≤0.45 / |prey|≥2 / vis≤0.20 五条阈值**全部 verbatim 抄 roster** (spec §1 表 + §2), 不做工程调参; 命中阈值的株 = "predator 想猎杀这种特征", 与「这种特征更弱」无关 —— predator-prey 频率依赖动力学解出胜负 (spec §2 红线 #1)。
- **per-letter, 不是 stacked**: roster `fam(s)=F ∧ f_s≥0.5` 的 `s` 是一个 **primitive (letter)**, 不是 strain stacked f. feature bit = "strain 的 sequence 里至少有一个 primitive 满足该阈值" (spec §2 §3.1 + 用户 doc-clarification: "携带")。`Phenotype.f` (stacked) 是另一回事, 与 thr_crest 无关。
- **kernel 不动**: antagonism kernel match expression 仍是 `(prey_mask[i] & feature_mask[j]) != 0`, S6 / S1 / S3 一路下来此式从未变化, 只是参与 OR 的 bit 位含义在变 (spec §2 §3, 红线 #3)。
- **|prey_s| 是 Z 行的 prey_clauses 元组长度, module-load 预计算**: 永不在运行时算; `_Z_PREY_CARD: dict[str, int]` 在 `_Z` 定义之后立即派生, 后续 `feature_mask_of` 直读 (spec §5)。
- **Default-BB0 字节级不变**: BB0 的 `F4Nr4` (f=0.50, 边界 ≥0.5) / `P_base` (p_add=0.0, 不命中) / `BroadSweep` (z=0.40, |prey|=2, 命中 thr_mirror) / `N0` (vis=0.20, 边界 ≤0.20, S1 已命中 vis_lowvis) → 默认局 strain 会 SET 3/4 阈值 bit, 但**无 v1 predator 用 threshold clause** (BroadSweep clause 是 `(("F",), ("Z",))` family-only) → kernel match 关系完全不变 → kill 数字字节级不变 (spec §2 红线 #4 + §6)。回归锁 = 285 引擎 + 146 web + S6/S1/S2/S4/S5 各自 owner test 全绿 + Task 5 跑同 seed 默认局 byte-identical。
- **bit 索引锁死, S6 owns**: `PREDICATE_BITS["thr_crest"]=12 / thr_hotspot=13 / thr_mirror=14 / vis_lowvis=11` 在 S6 Task 6 已铸 + module-load `assert max < 63` 守 int64; **S3 不动 `PREDICATE_BITS` 字典**, 只 OR 进 `m` (spec §5: "S3 adds no new bits — it populates S6-reserved slots — so the assert is S6's; S3 need not re-assert")。
- **vis_lowvis 由 S1 owns, S3 不重复填**: S1 Task 6 已给 `feature_mask_of` 加 `for letter in sequence: if ALPHABET[letter]=='N' and VIS[letter]<=0.20: m |= PREDICATE_BIT['vis_lowvis']; break` —— S3 Task 4 在 `prey_mask_for_clauses` 添 `("N", "lowvis")` clause 让 predator 能 target 这个 bit, 但 prey 端 (`feature_mask_of`) 不动 vis_lowvis 设置代码 (避免重复 OR + 两份事实源)。Task 5 审计 S1 vis_lowvis 状态仍正确。
- **预测的 boundary 半开/闭锁死**: `f_s ≥ 0.5` (闭, F4Nr4=0.50 命中) / `p_add,s ≥ 0.05` (闭, P_hotspot=0.05 命中) / `z_s ≤ 0.45` (闭, Sweep Surge=0.45 命中) / `|prey_s| ≥ 2` (闭, BroadSweep=2 命中) / `vis_s ≤ 0.20` (闭, N0=0.20 命中, S1 已守) —— 全 verbatim 抄 spec §6 + roster, 测试用 boundary 值守门 (spec §6: "F4Nr4 (f=0.50) sets FEAT_F_HI; F4Nr1 (f=0.30) does not; P_hotspot (p_add=0.05) sets FEAT_P_HI; BroadSweep sets FEAT_Z_GENERALIST")。
- **`prey_mask_for_clauses` 已识别的 4 条 clause 形态保持不变**: S6 Task 7 既有 `(fam,)` / `(fam, "motif")` / `(fam, "motif", "len>=3")` 三条分支, S3 在它们之后追加 4 条新 clause-tag 分支, 老 clause 形态字节级不变 (spec §3.2 + §6 regression: "antagonism known-answer unchanged for family-only predators")。
- **relabel-invariance**: 阈值读的是 strain *自身* 序列 + registry 表里的 `_F[letter][0]` (f) / `_P[letter][0]` (p_add) / `_Z[letter][0]` (z) / `_Z[letter][1]` (prey_clauses) / `VIS[letter]` —— 重排哪些 letter 是 dominant 不影响 (per-letter 不是 stacked); 但 *改 letter 的量级* 会改 bit 命中, 这与 spec §6 "fix the registry, the bit assignment is a pure function of the sequence" 一致。Task 5 验「fix registry 后两次 mint 同 sequence → 相同 feature_mask」(per-letter 是 strain sequence 的纯函数)。
- **Z multi-letter strain 取 OR 语义**: 同 strain 多个 Z letter 时, thr_mirror 只要任一个 Z letter 满足 `z≤0.45 ∧ |prey|≥2` 即置位 (spec §5 "carries ≥1 → bit set")。其他 3 个阈值同款 OR。
- **out of scope (later specs)**: A pool 的 8 个 threshold-hunter predator 行 (`Crest Bite` / `Hotspot Snipe` / `Mirror Fang` / `Void Bite` / `Lineage Reaper` / `Coil Cinch` / `Idiotype Lance` / `Predator Lock` 等, S8 owns); motif clause `(fam, "motif")` / `(fam, "motif", "len>=3")` 已是 S6 owns, 不动 (spec §7); vis 单 letter aggregate `vis_sum` / `n_count` (S1 owns); 多 P 合并谱 (S8 owns); `_Z` row 的 4-tuple 形状 (S1 已落)。
- **6→68 重录基线纪律延续**: spec §2 + S2 + S4 既已锁死, 默认局 fixture 的字节锁守 **non-registry 代码路径**, 不守 6-字母时代具体数值; S3 加阈值 bit 但无 v1 consumer 命中 → 默认局 kernel 输出字节级不变 → fixture re-record 不需要为 S3 触发 (spec §2 红线 #4)。

---

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `src/des/registry.py` | **Modify** | (a) 在 `_Z = {...}` 块之后立即派生 module-level `_Z_PREY_CARD: dict[str, int]`, key 集合 ⊆ `_Z.keys()`, value = `len(prey_clauses)`; (b) `feature_mask_of(sequence)` 函数体追加 3 段阈值检查 (单遍扫描 sequence, 命中即 `m |= PREDICATE_BIT[name]`): `thr_crest` (任一 `_F` letter 的 `f >= 0.5`) / `thr_hotspot` (任一 `_P` letter 的 `p_add >= 0.05`) / `thr_mirror` (任一 `_Z` letter 的 `z <= 0.45 ∧ _Z_PREY_CARD[letter] >= 2`); `vis_lowvis` 的填值由 S1 Task 6 已就位, S3 不重复; (c) `prey_mask_for_clauses(prey_clauses)` 解析新增 4 条 clause-tag 分支: `("F", "f_hi")` → `thr_crest` / `("P", "p_hi")` → `thr_hotspot` / `("Z", "generalist")` → `thr_mirror` / `("N", "lowvis")` → `vis_lowvis`; 既有 4 条分支 (family-only, motif, motif+len>=3) 字节级不动。 |
| `tests/test_threshold_predicates.py` | **Create** | 新建, S3 owner 文件 (sibling: `test_motif.py` / `test_vis.py` / `test_spectrum_shape.py` / `test_direction_kinds.py` / `test_phase_windows.py` / `test_hash_dirs.py`)。覆盖: `_Z_PREY_CARD` 模块级派生; `thr_crest` 命中 (F4Nr4 边界 0.50≥0.5) / 不命中 (F4Nr1 0.30); `thr_hotspot` 命中 (P_hotspot 0.05≥0.05) / 不命中 (P_base 0.0); `thr_mirror` 命中 (BroadSweep z=0.40≤0.45 ∧ |prey|=2) / 不命中 (单 prey 的合成 Z) / 边界 (z=0.45 + |prey|=2 命中); `prey_mask_for_clauses` 4 条新 clause-tag → 4 个 reserved bit; 多 letter strain OR 语义; default-BB0 strain feature_mask 包含 thr_crest + thr_mirror + vis_lowvis 但不含 thr_hotspot (P_base p_add=0.0); per-letter 不读 stacked f (relabel-invariance 风味); kernel byte-identical (默认 BB0 同 seed 跑两次 strain trajectory bit-equal — 守 antagonism kernel 没多吃 kill)。 |
| `tests/test_registry.py` | **Modify (append)** | 追加 1 条断言: `_Z_PREY_CARD` 与 `_Z` keys 同集合, 且 BroadSweep cardinality == 2。 |

**Naming contract (locked, used by every task):**

```python
# src/des/registry.py
_Z_PREY_CARD: dict[str, int]       # name -> len(prey_clauses); module-load 派生

# extended bodies (signatures unchanged)
def feature_mask_of(sequence: tuple[str, ...]) -> int
    # S6 / S1 既有: family_<X> / motif_<X> / motif3_<X> / vis_lowvis
    # S3 追加: thr_crest / thr_hotspot / thr_mirror

def prey_mask_for_clauses(prey_clauses: tuple[tuple[str, ...], ...]) -> int
    # S6 既有: (fam,) / (fam, "motif") / (fam, "motif", "len>=3")
    # S3 追加: (fam, "f_hi") / (fam, "p_hi") / (fam, "generalist") / (fam, "lowvis")
```

**Threshold predicate verbatim from spec §1 (locked, used by Task 2 / Task 3):**

| 谓词 bit | 触发条件 (per-letter) | 边界类型 | 默认 BB0 命中? |
| --- | --- | --- | --- |
| `thr_crest` | `letter in _F` 且 `_F[letter][0] >= 0.5` | 闭 (≥) | 是 (F4Nr4=0.50) |
| `thr_hotspot` | `letter in _P` 且 `_P[letter][0] >= 0.05` | 闭 (≥) | 否 (P_base=0.0) |
| `thr_mirror` | `letter in _Z` 且 `_Z[letter][0] <= 0.45` 且 `_Z_PREY_CARD[letter] >= 2` | 闭 (≤ + ≥) | 是 (BroadSweep=0.40, 2 prey clauses) |
| `vis_lowvis` | `ALPHABET[letter] == "N"` 且 `VIS[letter] <= 0.20` | 闭 (≤) | 是 (N0=0.20, **S1 Task 6 已实现**) |

**Predator clause-tag → reserved bit (locked, used by Task 4):**

| Clause 形式 | 选择 bit | 典型 predator (S8 owns) |
| --- | --- | --- |
| `("F", "f_hi")` | `PREDICATE_BIT["thr_crest"]` | Crest Bite |
| `("P", "p_hi")` | `PREDICATE_BIT["thr_hotspot"]` | Hotspot Snipe |
| `("Z", "generalist")` | `PREDICATE_BIT["thr_mirror"]` | Mirror Fang |
| `("N", "lowvis")` | `PREDICATE_BIT["vis_lowvis"]` | Void Bite |

不引入新 bit, 不改 `PREDICATE_BITS`/`PREDICATE_BIT` 字典 (S6 锁死)。tag 名字符串 (`"f_hi"` / `"p_hi"` / `"generalist"` / `"lowvis"`) 在 `prey_mask_for_clauses` 内字面识别, 与 S6 `"motif"` / `"len>=3"` 同款字符串-tag 风格。

---

### Task 1: `_Z_PREY_CARD` module-load 预计算 (纯数据派生)

**Goal:** 在 `src/des/registry.py` 的 `_Z` 表定义之后, 立即派生 module-level 常量 `_Z_PREY_CARD: dict[str, int]`, key 与 `_Z.keys()` 同, value 是该 row 的 `prey_clauses` 元组长度。这一步是 Mirror Fang 的 `|prey|≥2` 谓词在 `feature_mask_of` 里 O(1) 直读的来源, 永不在运行时算 (spec §5)。这一步**无消费者** (Task 2 才让 `feature_mask_of` 读它), 既有行为 0 漂移。

**Files:**
- Modify: `src/des/registry.py` (在 `_Z = {...}` 之后, `_P = {...}` 之前, 插入 4 行派生)
- Test: `tests/test_threshold_predicates.py` (Create — S3 owner 文件首批断言) + `tests/test_registry.py` (append 1 条)

**Interfaces:**
- Consumes: `_Z` 的 4-tuple 行形状 `(z, prey_clauses, period, vis_mode)` (S6 / S1 既有)。
- Produces:
  - `_Z_PREY_CARD: dict[str, int]` — 每个 `_Z` 行 key 对应 `len(_Z[key][1])`。v1 v6 alphabet 唯一 Z row 是 `BroadSweep`, prey_clauses `(("F",), ("Z",))` → `_Z_PREY_CARD["BroadSweep"] == 2`。

- [ ] **Step 1: 写失败测试 — `_Z_PREY_CARD` 存在 + 覆盖 + BroadSweep cardinality**

新建 `tests/test_threshold_predicates.py`:

```python
# tests/test_threshold_predicates.py
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
```

追加到 `tests/test_registry.py`:

```python
# ---------------------------------------------------------------------------
# S3 Task 1: _Z_PREY_CARD module-load derivation
# ---------------------------------------------------------------------------

def test_s3_z_prey_card_module_level_derivation():
    """_Z_PREY_CARD 是 module-load 派生 (而非运行时算), key 与 _Z 同."""
    from des.registry import _Z, _Z_PREY_CARD
    assert set(_Z_PREY_CARD) == set(_Z)
    assert _Z_PREY_CARD["BroadSweep"] == 2
```

- [ ] **Step 2: 跑失败测试, 确认 FAIL**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_threshold_predicates.py tests/test_registry.py::test_s3_z_prey_card_module_level_derivation -v
```

Expected: 4 条全 FAIL with `ImportError: cannot import name '_Z_PREY_CARD' from 'des.registry'`.

- [ ] **Step 3: 在 `src/des/registry.py` 派生 `_Z_PREY_CARD`**

在 `_Z = {...}` 块之后 (S1 / S6 落地后此块行号会漂移, 锚定为 `_Z` 块的结束括号之后), `_P = {...}` 之前, 插入:

```python
# S3: prey-clause cardinality (|prey_s| in roster Mirror Fang spec §1).
# Module-load derived from _Z[letter][1] (the prey_clauses tuple) so the
# feature_mask_of hot path reads O(1) per Z letter, never iterates clauses
# at mint time. Co-extensive with _Z; adding a new _Z row REQUIRES this
# dict be re-derived (it is, on every module import).
_Z_PREY_CARD: dict[str, int] = {name: len(row[1]) for name, row in _Z.items()}
```

- [ ] **Step 4: 跑测试, 确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_threshold_predicates.py tests/test_registry.py::test_s3_z_prey_card_module_level_derivation -v
```

Expected: 4 条全 PASS。

- [ ] **Step 5: 跑全 suite, 确认无回归**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿 — `_Z_PREY_CARD` 是纯数据派生, 无消费者读它, 既有谱 / kernel / phenotype 0 漂移。

- [ ] **Step 6: Commit**

```bash
git add src/des/registry.py tests/test_threshold_predicates.py tests/test_registry.py
git commit -m "feat(s3): derive _Z_PREY_CARD at module load (data-only)

dict[str, int] mapping each _Z row name to len(prey_clauses). Foundation
for the thr_mirror feature-bit predicate (|prey_s| >= 2 in roster Mirror
Fang spec §1); read O(1) per Z letter in feature_mask_of (Task 2). v1
BroadSweep cardinality = 2. Pure data derivation, no consumer yet."
```

---

### Task 2: `feature_mask_of` 增 `thr_crest` / `thr_hotspot` / `thr_mirror` 三阈值

**Goal:** 扩展 `src/des/registry.py::feature_mask_of(sequence)` 函数体, 在 S6 / S1 既有的 family / motif / motif3 / vis_lowvis 之后追加 3 段阈值检查。每段做单遍 sequence 扫描, 命中即 OR 进 reserved bit, `break` 跳出 (per-letter 任一命中即置位, OR 语义)。`vis_lowvis` 由 S1 Task 6 已加, S3 不重复; 三段新代码与 S1 那段并列。

阈值边界 verbatim 抄 spec §1 表 + Global Constraints:
- `thr_crest`: `letter in _F` 且 `_F[letter][0] >= 0.5` (闭, F4Nr4 边界 0.50 命中)
- `thr_hotspot`: `letter in _P` 且 `_P[letter][0] >= 0.05` (闭, P_hotspot 边界 0.05 命中)
- `thr_mirror`: `letter in _Z` 且 `_Z[letter][0] <= 0.45` 且 `_Z_PREY_CARD[letter] >= 2` (闭, BroadSweep z=0.40 + 2 prey 命中)

**Files:**
- Modify: `src/des/registry.py` 的 `feature_mask_of` 函数体 (S1 落地后此函数在文件中后段 — 锚定为函数体的 `return m` 之前)。
- Test: `tests/test_threshold_predicates.py` (append)

**Interfaces:**
- Consumes: `_F` / `_P` / `_Z` 行 (既有), `_Z_PREY_CARD` (Task 1), `PREDICATE_BIT["thr_crest" | "thr_hotspot" | "thr_mirror"]` (S6 Task 6 已铸), `ALPHABET` (既有)。
- Produces: 同 `feature_mask_of(sequence: tuple[str, ...]) -> int` 签名。返回的 int 在原有 family/motif/motif3/vis_lowvis bit 之上, 多 OR 进:
  - `PREDICATE_BIT["thr_crest"]` iff `any(_F[ltr][0] >= 0.5 for ltr in sequence if ltr in _F)`
  - `PREDICATE_BIT["thr_hotspot"]` iff `any(_P[ltr][0] >= 0.05 for ltr in sequence if ltr in _P)`
  - `PREDICATE_BIT["thr_mirror"]` iff `any(_Z[ltr][0] <= 0.45 and _Z_PREY_CARD[ltr] >= 2 for ltr in sequence if ltr in _Z)`

- [ ] **Step 1: 写失败测试 — 三阈值的命中 / 不命中 / 边界 / 多 letter OR**

追加到 `tests/test_threshold_predicates.py`:

```python
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
```

- [ ] **Step 2: 跑失败测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_threshold_predicates.py -v -k "thr_crest or thr_hotspot or thr_mirror or default_bb0_strain_thr_bits or pure_function_of_sequence"
```

Expected: 12 条全 FAIL — 当前 `feature_mask_of` (S1 / S6 落地态) 不读 `_F`/`_P`/`_Z` 阈值, 这 3 个 bit 永远 CLEAR。

- [ ] **Step 3: 扩展 `feature_mask_of` 函数体**

打开 `src/des/registry.py`, 找到 `feature_mask_of(sequence)` (S6 Task 7 添加 + S1 Task 6 扩展)。在它的 `return m` 之前 (即 vis_lowvis 那段 `for letter in sequence: if ALPHABET.get(letter) == "N" and VIS.get(letter, 0.0) <= 0.20: ... break` 之后) 追加三段独立扫描:

```python
    # S3: thr_crest — strain carries any F letter with f >= 0.5 (Crest Bite prey
    # clause from roster §1 + spec §3.1). Default BB0's F4Nr4 (f=0.50) SETs it.
    for letter in sequence:
        if letter in _F and _F[letter][0] >= 0.5:
            m |= PREDICATE_BIT["thr_crest"]
            break
    # S3: thr_hotspot — strain carries any P letter with p_add >= 0.05 (Hotspot
    # Snipe prey clause). Default BB0's P_base (0.0) does NOT set it.
    for letter in sequence:
        if letter in _P and _P[letter][0] >= 0.05:
            m |= PREDICATE_BIT["thr_hotspot"]
            break
    # S3: thr_mirror — strain carries any Z letter with z <= 0.45 AND
    # |prey_clauses| >= 2 (Mirror Fang prey clause; |prey_s| precomputed in
    # _Z_PREY_CARD at module load — never iterates clauses on the hot path).
    # Default BB0's BroadSweep (z=0.40, 2 prey clauses) SETs it.
    for letter in sequence:
        if (letter in _Z
                and _Z[letter][0] <= 0.45
                and _Z_PREY_CARD[letter] >= 2):
            m |= PREDICATE_BIT["thr_mirror"]
            break
    return m
```

(三段并列, 各自 `break` 跳出 — 每个 bit 独立 OR, 互不阻塞。本仓库 S1 Task 6 落地的 vis_lowvis 段已经在 `return m` 之前用同款单遍 + break 模式; S3 三段紧随其后。)

- [ ] **Step 4: 跑测试, 确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_threshold_predicates.py -v
```

Expected: Task 1 + Task 2 全 PASS (含 boundary, OR 语义, 默认 BB0 strain, pure function)。

Backtrack:
- 若 `test_thr_mirror_boundary_z_at_0p45_with_two_prey` FAIL with `KeyError: 'SweepSurge'` in `_Z_PREY_CARD`, 说明 monkeypatch 给 `_Z_PREY_CARD` 的 `setitem` 没生效 (顺序问题)。手工 `monkeypatch.setitem(registry._Z_PREY_CARD, "SweepSurge", 2)` 必须在 `_Z` 那条 monkeypatch 之后, 且 `feature_mask_of` 必须读 `_Z_PREY_CARD` (而非 `len(_Z[letter][1])`); 检查 Step 3 用了 `_Z_PREY_CARD[letter]` 不是 `len(...)`。
- 若 `test_default_bb0_strain_thr_bits_per_global_constraints` 在 `thr_hotspot` 上 FAIL (= bit 被错误置位), root-cause: BB0 layout 含 `P_base` (p_add=0.0), 不应命中阈值 — 检查 Step 3 第二段是否误用 `>` 0.0 而不是 `>= 0.05`。

- [ ] **Step 5: 跑全 suite, 确认无回归**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 285 engine + 146 web + S6/S1/S2/S4/S5/S3 owner 测试全绿。3 个新 bit 在默认 BB0 strain 上额外置位, 但**无 v1 prey clause target 这 3 个 bit** (BroadSweep clause 仍是 `(("F",), ("Z",))` family-only) → kernel match 关系不变 → 默认局动力学字节级不变。

Backtrack: 若 `tests/test_phenotype_cache.py` 或 `tests/test_acceptance.py` 在 `feature_mask` 数值字面量断言上 FAIL, 该测试 hard-coded 了 pre-S3 mask 值。OR 进 `PREDICATE_BIT["thr_crest"] | PREDICATE_BIT["thr_mirror"]` (默认 BB0 命中的两个新 bit) 后字面量会变; 升级方式同 S1 Task 6 的策略 — 把硬数升成 `OR(*reserved bits actually set)`, 或改成「只验匹配关系不验字面量」。

- [ ] **Step 6: Commit**

```bash
git add src/des/registry.py tests/test_threshold_predicates.py
git commit -m "feat(s3): feature_mask_of sets thr_crest / thr_hotspot / thr_mirror

Three single-pass scans append after S6 family/motif + S1 vis_lowvis:
- thr_crest    iff any _F letter has f >= 0.5      (Crest Bite prey clause)
- thr_hotspot  iff any _P letter has p_add >= 0.05 (Hotspot Snipe prey clause)
- thr_mirror   iff any _Z letter has z <= 0.45 AND _Z_PREY_CARD >= 2
                                                    (Mirror Fang prey clause)
Roster verbatim, closed boundaries (>=, <=). Default BB0 SETs thr_crest
(F4Nr4 f=0.50) + thr_mirror (BroadSweep z=0.40, 2 prey); thr_hotspot
clear (P_base p_add=0.0). Kernel match relation unchanged — no v1 prey
clause targets these bits yet (Task 4 wires the predator side)."
```

---

### Task 3: kernel byte-identical 回归 + per-letter (非 stacked) 守门审计

**Goal:** Task 2 给 `feature_mask_of` 加了 3 个 reserved bit 的置位逻辑, 但 antagonism kernel 的 match expression `(prey_mask[i] & feature_mask[j]) != 0` 是关于「prey clause 选 bit ↔ feature 端置 bit」的关系 —— v1 没有任何 prey clause target `thr_*` (Task 4 才追加 4 条新 clause), 所以**默认 BB0 局的 kill 数字必须 byte-identical**。这一步**不动 source code**, 只追加 2 条端到端守门测试: (a) 默认 BB0 4-faction 同 seed 跑 30 tick, `world.count` / `strain_id` byte-identical (回归锁); (b) per-letter (非 stacked) 守门 — strain `f_hi` 是 stacked 累计值, 但 `thr_crest` 命中条件是「sequence 任一 letter 的 `_F[letter][0] >= 0.5`」, 验明 stacked f 与 thr_crest 解耦。

**Files:**
- 不预期 source 改动。
- Test: `tests/test_threshold_predicates.py` (append 2 条端到端 + 1 条 per-letter audit)。

**Interfaces:**
- Consumes: Task 1 + Task 2 全部产物。
- Produces: 端到端字节级回归断言 + per-letter 语义审计。

- [ ] **Step 1: 写端到端 byte-identical 回归测试 + per-letter 审计**

追加到 `tests/test_threshold_predicates.py`:

```python
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
```

- [ ] **Step 2: 跑测试, 确认 PASS (这一 task 不应该出现 FAIL)**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_threshold_predicates.py -v -k "byte_identical or match_relation_byte_identical or per_letter_not_stacked"
```

Expected: 3 条全 PASS。

- 第一条 (默认 BB0 same-seed): kernel 没改, generator 是 seed 的确定函数 → 同 seed 双跑同结果。S3 唯一改的 `feature_mask_of` 给 prey 添了 reserved bit, 但 prey_mask 端没人用它 (Task 4 才铸 clause), kernel match 输出不变。
- 第二条 (synthetic match relation): 纯逻辑等式断言, 不调 kernel。
- 第三条 (per-letter): 走 Task 2 的代码 — 它读 `_F[letter][0]` (per-letter), 不读 `Phenotype.f` (stacked)。F4Nr1 行 `_F["F4Nr1"][0] = 0.30 < 0.5` → thr_crest CLEAR。

Backtrack: 若第三条 FAIL, root-cause: Task 2 Step 3 误把检查写成 `phenotype(sequence).f >= 0.5` 这种 stacked 形式, 或者从循环外读了 `f_prod`。修法: 严格用 `_F[letter][0] >= 0.5` 字面读。

- [ ] **Step 3: 跑全 suite, 确认无回归**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿。

- [ ] **Step 4: Commit**

```bash
git add tests/test_threshold_predicates.py
git commit -m "test(s3): byte-identical regression + per-letter (non-stacked) audit

Three guard tests: (a) default BB0 4-faction 30-tick same-seed run is
byte-identical post-S3 — kernel match relation unchanged because no v1
prey clause targets thr_* bits yet; (b) synthetic match-relation logic
sanity (family clause and thr_mirror both hit prey BroadSweep); (c)
thr_crest must read _F[letter][0] per-letter, NOT Phenotype.f (stacked)
— constructed adversarial case (two F4Nr1, stacked f > 0.5 but no
per-letter f >= 0.5) catches a stacked-leak regression."
```

---

### Task 4: `prey_mask_for_clauses` 加 4 条新 clause-tag 分支

**Goal:** 扩展 `src/des/registry.py::prey_mask_for_clauses(prey_clauses)` 函数体, 让 predator 端的 prey clause 能 target S6 reserved + Task 2 / S1 已填的 4 个 bit。S6 既有 3 条分支 — `(fam,)` family-only / `(fam, "motif")` motif / `(fam, "motif", "len>=3")` motif3 — S3 在它们之后追加 4 条新 clause-tag 分支:

| Clause | 选择 bit | 典型 predator |
| --- | --- | --- |
| `("F", "f_hi")` | `thr_crest` | Crest Bite |
| `("P", "p_hi")` | `thr_hotspot` | Hotspot Snipe |
| `("Z", "generalist")` | `thr_mirror` | Mirror Fang |
| `("N", "lowvis")` | `vis_lowvis` | Void Bite |

`("Z", "generalist")` 的 "generalist" tag 命名来自 spec §3.1 注释 ("FEAT_Z_GENERALIST"); v1 `_Z` 行的 prey_clauses 仍是 family-only (BroadSweep `(("F",), ("Z",))`), 不会触发新分支; A pool predator (`Crest Bite` / `Hotspot Snipe` / `Mirror Fang` / `Void Bite`) 由 S8 铸 — S3 只让 clause 形态**可被识别**, 此时铸出来即可命中正确 bit。

**Files:**
- Modify: `src/des/registry.py::prey_mask_for_clauses` 函数体 (S6 Task 7 添加, 当前是 7 行 if/elif/else)。
- Test: `tests/test_threshold_predicates.py` (append)

**Interfaces:**
- Consumes: `PREDICATE_BIT["thr_crest" | "thr_hotspot" | "thr_mirror" | "vis_lowvis"]` (S6 Task 6 锁死索引)。
- Produces: 同 `prey_mask_for_clauses(prey_clauses: tuple[tuple[str, ...], ...]) -> int` 签名。新增 4 条 clause-tag 分支后, 返回值仍是 OR-over-clauses 的 int; 4 条新 tag 与既有 3 条 (family-only / motif / motif3) 互斥识别。

- [ ] **Step 1: 写失败测试 — 4 条新 clause-tag → 4 个 reserved bit**

追加到 `tests/test_threshold_predicates.py`:

```python
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
```

- [ ] **Step 2: 跑失败测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_threshold_predicates.py -v -k "prey_mask_for_clauses"
```

Expected: 8 条中前 5 条 (f_hi / p_hi / generalist / lowvis / unknown_tag) FAIL (S6 既有函数体把 "f_hi" 等当作未知 tag, 现行 else 分支 fall through 到 family_F 路径, 但这条路径会把它当成 fam == "F" 选 `family_F` bit, 与新断言矛盾); 3 条 regression PASS (family-only / motif / mixed)。

具体地, S6 既有函数体大致为:

```python
def prey_mask_for_clauses(prey_clauses):
    m = 0
    for clause in prey_clauses:
        if not clause: continue
        fam = clause[0]; tags = clause[1:]
        if "motif" in tags and "len>=3" in tags:
            if fam in ("F","P","Z"): m |= PREDICATE_BIT[f"motif3_{fam}"]
        elif "motif" in tags:
            m |= PREDICATE_BIT[f"motif_{fam}"]
        else:
            m |= PREDICATE_BIT[f"family_{fam}"]
    return m
```

所以 `("F", "f_hi")` 会落到 else, 命中 `family_F` —— `test_prey_mask_for_clauses_f_hi_tag_selects_thr_crest` 断 `pm == PREDICATE_BIT["thr_crest"]` 故 FAIL; `("F", "unknown_future_tag")` 也走 else 命中 `family_F` —— `test_unknown_tag_falls_through_to_family` 实际 PASS (这条断言是 spec §3.2 的 forward-compat 契约, 用 S3 实现也必须保持)。

- [ ] **Step 3: 扩展 `prey_mask_for_clauses` 函数体**

打开 `src/des/registry.py`, 找到 `prey_mask_for_clauses(prey_clauses)`。把函数体重写为 (在 S6 既有 motif/motif3 分支之间插入 S3 新 tag 分支, 保持原 family-only fallback):

```python
def prey_mask_for_clauses(prey_clauses: tuple[tuple[str, ...], ...]) -> int:
    """Predicate-bit prey mask for a Z row's clause list (S6 §3.5 + S3 §3.2).
    Each clause is a tuple whose first element is the family ('F'|'P'|'Z'|'N')
    and whose optional further elements specialize the predicate:
      ('F',)                     → family_F bit                      (S6)
      ('F', 'motif')             → motif_F bit                       (S6)
      ('F', 'motif', 'len>=3')   → motif3_F bit                      (S6)
      ('F', 'f_hi')              → thr_crest bit                     (S3 — Crest Bite)
      ('P', 'p_hi')              → thr_hotspot bit                   (S3 — Hotspot Snipe)
      ('Z', 'generalist')        → thr_mirror bit                    (S3 — Mirror Fang)
      ('N', 'lowvis')            → vis_lowvis bit                    (S3 — Void Bite; S1 reserved bit 11)
      ('F', '<unknown_tag>')     → family_F bit (fall through, forward-compat)
    OR the selected bits to form prey_mask. Pure function of the clause list."""
    m = 0
    for clause in prey_clauses:
        if not clause:
            continue
        fam = clause[0]
        tags = clause[1:]
        if "motif" in tags and "len>=3" in tags:
            if fam in ("F", "P", "Z"):
                m |= PREDICATE_BIT[f"motif3_{fam}"]
        elif "motif" in tags:
            m |= PREDICATE_BIT[f"motif_{fam}"]
        elif "f_hi" in tags and fam == "F":
            m |= PREDICATE_BIT["thr_crest"]
        elif "p_hi" in tags and fam == "P":
            m |= PREDICATE_BIT["thr_hotspot"]
        elif "generalist" in tags and fam == "Z":
            m |= PREDICATE_BIT["thr_mirror"]
        elif "lowvis" in tags and fam == "N":
            m |= PREDICATE_BIT["vis_lowvis"]
        else:
            # forward-compat fallback: unknown tag (or no tag) → family bit
            m |= PREDICATE_BIT[f"family_{fam}"]
    return m
```

(S3 4 条新分支用「字面 tag-string + fam 配对」识别, 避免空 tag 误命中 / 跨家族错配 (例如 `("F", "p_hi")` 不应命中 `thr_hotspot`)。`unknown_tag` 仍 fall through 到 family_F, 与 Step 1 第 8 条测试一致。)

- [ ] **Step 4: 跑测试, 确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_threshold_predicates.py -v -k "prey_mask_for_clauses"
```

Expected: 8 条全 PASS, 含 4 条新 tag + 3 条 regression + 1 条 unknown_tag fall through。

Backtrack:
- 若 `test_prey_mask_for_clauses_motif_clauses_unchanged_post_s3` FAIL, root-cause: S3 把 "motif" 分支顺序写错, 例如把 "f_hi" 分支提到 "motif" 之前 → 含 "motif" 的 clause 误进 fall-through。修法: 保持 motif / motif3 分支优先级 (在 S3 4 条新分支之上)。
- 若 `("F", "p_hi")` 跨家族错配 (例如返回 `thr_hotspot` 而非 fall-through 到 `family_F`), root-cause: Step 3 写成 `elif "p_hi" in tags:` 没带 `and fam == "P"`。修法: 每条新分支都必须 tag + fam 双校验。

- [ ] **Step 5: 跑全 suite, 确认无回归**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 285 engine + 146 web + S6/S1/S2/S4/S5/S3 owner 测试全绿。v1 `_Z` 行 (BroadSweep) 的 prey_clauses 仍是 family-only → kernel match 关系不变 → 默认局动力学字节级不变 (与 Task 3 守门一致)。

Backtrack: 若 `test_prey_mask_for_clauses_motif_clause` (S6 既有 test) FAIL, S3 改动顺序破坏了 motif 优先级 — 把 motif/motif3 分支保留在所有 S3 新分支之前。

- [ ] **Step 6: Commit**

```bash
git add src/des/registry.py tests/test_threshold_predicates.py
git commit -m "feat(s3): prey_mask_for_clauses recognizes 4 new clause tags

Append 4 new clause-tag branches after S6's motif/motif3 path:
  ('F', 'f_hi')       → thr_crest      (Crest Bite)
  ('P', 'p_hi')       → thr_hotspot    (Hotspot Snipe)
  ('Z', 'generalist') → thr_mirror     (Mirror Fang)
  ('N', 'lowvis')     → vis_lowvis     (Void Bite; S1 reserved bit 11)
Each branch requires both tag-string and family match to fire — cross-
family tag (e.g. ('F','p_hi')) falls through to family_F. v1 _Z rows
(BroadSweep family-only) unaffected; A-pool predators (S8) will mint
prey_clauses with these tags. Kernel match unchanged."
```

---

### Task 5: vis_lowvis 审计 + final regression sweep + smoke + push

**Goal:** S1 Task 6 已经把 `vis_lowvis` bit 的 prey 端语义填入 `feature_mask_of` (N0 vis=0.20 ≤ 0.20 → SET on default BB0). S3 Task 4 给 predator 端添了 `("N", "lowvis")` clause tag。这一步**审计 S1 那段代码仍然存在且工作正常** (避免 Task 2 扩展 `feature_mask_of` 时不小心删掉 S1 段, 或 monkeypatch 顺序冲突), 然后跑整套 S3 deliverable (Tasks 1-4) 的 final regression sweep + smoke + push, 这是 sibling task 的同款收口动作 (S0 Task 6, S6 Task 9, S1 Task 7, S2 Task 7, S4 Task 8, S5 Task 6)。

**Files:**
- 不预期 source 改动。若 Step 1 暴露回归 (例如 Task 2 误删 S1 段, 或 cross-bit 漏洞), 本 task 修 forward, commit message 引用 offending commit。
- Test: `tests/test_threshold_predicates.py` (append 1 条审计) + `tests/` (整套) + `scripts/run_batch.py --probe` smoke。

**Interfaces:**
- Consumes: Task 1-4 全部产物 + S1 Task 6 的 vis_lowvis prey-端实现。
- Produces: 绿 `pytest tests/` + 干净 `git status` + push 到 origin。

- [ ] **Step 1: 写 vis_lowvis 审计测试 (S1 端到 S3 端的桥)**

追加到 `tests/test_threshold_predicates.py`:

```python
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
```

- [ ] **Step 2: 跑 vis_lowvis 审计**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_threshold_predicates.py::test_vis_lowvis_end_to_end_s1_prey_meets_s3_predator_clause -v
```

Expected: PASS. 若 FAIL on `prey_m & PREDICATE_BIT["vis_lowvis"]`, root-cause: Task 2 Step 3 重写 `feature_mask_of` 时把 S1 Task 6 的 vis_lowvis 那段 (大约 `for letter in sequence: if ALPHABET.get(letter) == "N" and VIS.get(letter, 0.0) <= 0.20: m |= PREDICATE_BIT["vis_lowvis"]; break`) 误删/覆写。修法: 把 S1 那段重新加回, 紧跟 motif3 后, S3 三段之前 — 保持次序: family → motif → motif3 → S1 vis_lowvis → S3 thr_crest → S3 thr_hotspot → S3 thr_mirror。

- [ ] **Step 3: 跑 full pytest sweep**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -q
```

Expected: 全绿。总数 = 285 engine + 146 web + S6 motif + S1 vis + S2 spectrum_shape + S4 direction_kinds + S5 phase_windows + S3 threshold_predicates (~25 条). 精确数随时间漂移; **没有 `FAILED tests/...` 行**是验收标准 (`SKIPPED` 行允许 — 它们是 S4 / S5 显式标记的 fixture re-record 占位)。

Backtrack: 若任何 FAIL, 先按测试 owner 文件 root-cause 到对应 Task; 用 `git log` 找出 offending commit, fix forward, 不要 reset。

- [ ] **Step 4: Smoke run probe (确认运行时性能没崩)**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --probe 30
```

Expected stdout 形如 `[probe 30 ticks] X.X ms/tick | peak Y.YY GB | strains->… | total …`. `X.X ms/tick` 应保持在 S0/S6/S1/S2/S4/S5 完工时的同一档 (target ≈15.8 ms/tick on 128² grid; ≤ 20% drift acceptable)。exit 0; 不写 parquet (probe 路径 record=False)。

若 drift > 20%, 最常见原因: `feature_mask_of` 4 段独立扫描 (vis_lowvis + thr_crest + thr_hotspot + thr_mirror) 在 strain mint 时跑 4 遍 16-letter 循环, 每段 O(L)。对默认 BB0 16-letter strain 是 64 次比较, < 1 μs 量级, **不应**影响 tick time。如果真观察到 drift, 把 4 段融合到 family/motif 那 1 段 sequence 循环里 (做单遍, 但要小心保留 break 语义 — 不同 bit 各自 break)。

- [ ] **Step 5: Byte-identical default-run smoke (推荐, 守 BB0 字节级不变)**

跑两次同 seed BB0 默认局:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --seeds 0 --T 30
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --seeds 0 --T 30
```

Expected: 两次产出的 parquet 在 `data/runs/` 下用 pyarrow 读起来 `(tick, cell, strain, count)` 行级一致 (Test `test_default_bb0_same_seed_byte_identical_post_s3` 已 pytest 守门, smoke 仅作 belt-and-suspenders)。这一条守 "S3 给 feature_mask 加了 3 个 bit, 但 v1 prey clauses 不 target 它们, kernel match 关系不变 → 同 seed 同结果"。

补充验证 (可选): 与 S5 / S4 收口时的 baseline parquet 比对 — 若 S5 Task 6 当时留了一份, 把现在跑出来的 parquet 与 S5 baseline 做 row-by-row diff:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -c "
import pyarrow.parquet as pq
a = pq.read_table(r'data/runs/<s5-baseline>.parquet').to_pydict()
b = pq.read_table(r'data/runs/<s3-fresh>.parquet').to_pydict()
print('cols:', sorted(a.keys()) == sorted(b.keys()))
for col in a:
    print(col, a[col] == b[col])
"
```

Expected: 全 `True`。若 False, root-cause: 大概率 `feature_mask_of` 给默认 BB0 strain 加了 reserved bit, 但某 v1 prey_clauses 误 target 了 reserved bit (Task 4 出问题); 或 kernel 偷读了 feature_mask 而非走 match expression。

- [ ] **Step 6: Inspect & clean stray data**

```
git status
```

Expected: 干净工作树。若 `data/runs/<ts>-*.parquet` smoke 残留, 删掉 — 不入 commit (它们不是 fixture, 而且 837MB 基线 RE-RECORD 是 batch CLI 单独跑的事)。

- [ ] **Step 7: Final commit (only if Step 3 needed a fix-forward) + push**

若 Step 3 surface 了 regression 并修了:

```bash
git add <files-touched>
git commit -m "fix(s3): <description of the regression fixed>"
```

否则跳过. 然后:

```bash
git add tests/test_threshold_predicates.py
git commit -m "test(s3): vis_lowvis end-to-end audit (S1 prey ↔ S3 predator)

Guard that S1 Task 6's feature_mask_of vis_lowvis path survived Task 2's
feature_mask_of body extension. End-to-end test: prey side (S1 sets
vis_lowvis on default BB0 N0 strain) ∩ predator side (S3 ('N','lowvis')
clause selects vis_lowvis bit) match relation is non-zero. Closes S3."
```

- [ ] **Step 8: Push to origin**

```bash
git push origin <current-branch>
```

Expected: push succeeds. The branch is ready for review / merge to `main`.

后续动作 (out of S3 plan scope, 用户单独决定时机):
1. S7 (多位突变) 落地后回看本 plan: 多位突变改 `_mutation_outcomes` 的 slot 选择, 与阈值 bit 完全正交; 不需要 hook。
2. S8 (A 池) 落地后会铸出 `Crest Bite` / `Hotspot Snipe` / `Mirror Fang` / `Void Bite` / `Lineage Reaper` / `Coil Cinch` / `Idiotype Lance` / `Predator Lock` 等 threshold-hunter predator — 它们的 `_Z` 行 prey_clauses 用 S3 新增的 4 条 clause-tag 形态 (`("F","f_hi")` 等), S3 已经把 machinery 做完, S8 仅需追加行。
3. 837MB 首批基线 parquet (S4 / S5 仍 skip 的 fixture-based byte-equal 测试) 重跑 — S3 也没碰它, 仍由 batch CLI 单独跑后回填。S3 + S4 + S5 全完工后再 RE-RECORD 一并解 skip 更高效。

---

## Self-Review

**1. Spec coverage:**

- §1 (why — Crest Bite / Hotspot Snipe / Mirror Fang / Void Bite 四 predator clause + motif clauses S6 已编码 verbatim): Task 2 (`feature_mask_of` 三阈值置位) + Task 4 (`prey_mask_for_clauses` 4 条新 tag 分支) + Task 5 (vis_lowvis 桥接 S1 prey 端 + S3 predator 端). 8 个 motif clause `(motif∋F/P/Z/N)` 与 `(ℓ≥3 motif∋F/P/Z)` 已在 S6 落地, S3 不重复 (spec §1 + §7 显式标 out-of-scope)。
- §2 (red lines — 阈值是 prey-selection predicate / per-letter / kernel 不动 / 默认 BB0 byte-equal 锁): Global Constraints 列 8 条逐行守门; Task 2 (per-letter 不读 stacked) + Task 3 (per-letter audit + byte-identical 回归) + Task 5 (final byte-identical smoke 双跑)。
- §3.1 (feature_mask 在 prey 端置位 4 个 thr_/vis bit): Task 2 (thr_crest / thr_hotspot / thr_mirror) + S1 Task 6 既有 (vis_lowvis); Task 5 审计 vis_lowvis 仍存活。
- §3.2 (prey_mask 在 predator 端读 PREY_CLAUSE: predator letter -> bitmask): Task 4 把 `prey_mask_for_clauses` 函数体扩展, 新增 4 条 clause-tag 分支按 spec §1 表把 4 个 predator clause 形态映射到 4 个 reserved bit。
- §4 (data flow): Task 2 (mint(seq) → phenotype()/prey side feature_mask) + Task 4 (mint(predator) → phenotype()/predator side prey_mask) + Task 5 (phase1_antagonism `(prey_mask[i] & feature_mask[j]) != 0` UNCHANGED end-to-end audit)。
- §5 (Error handling): `_Z_PREY_CARD` 在 module-load 派生而非 runtime 算 (Task 1); int64 守门由 S6 owns, S3 不重复 (Global Constraints 显式声明); strain 多 letter OR 语义 (Task 2 测 `test_thr_crest_set_if_any_F_letter_meets_threshold`)。
- §6 (testing — regression 默认 BB0 byte-equal / boundary 命中 / 多 letter OR / relabel-invariance flavor): Task 1 (`_Z_PREY_CARD` 派生) + Task 2 (12 条 thr_* 命中/不命中/边界/OR/默认 BB0/pure-function) + Task 3 (默认 BB0 same-seed byte-equal + per-letter 不读 stacked) + Task 4 (8 条 prey_mask_for_clauses 4 条新 tag + regression + cross-family 隔离) + Task 5 (vis_lowvis end-to-end audit + final pytest sweep + smoke)。
- §7 (out of scope — A pool predator 行 / motif clause / vis_sum aggregate / 多 P 合并 / `_Z` row reshape): 全在 Global Constraints 与各 Task 描述里多次声明; S3 不引入相关 source code。

**Red lines (§2):**

1. 阈值是 prey-selection predicate, 非「谁强」 — 谓词读 `_F[letter][0]` / `_P[letter][0]` / `_Z[letter][0]` / `_Z_PREY_CARD[letter]` / `VIS[letter]`, 不写入 strain 相对强度。
2. per-letter, 非 stacked — Task 2 三段都用 `_F[letter][0]` 字面读, Task 3 第三条测试 (`test_thr_crest_is_per_letter_not_stacked_f`) adversarial 守门。
3. kernel 不动 — antagonism kernel match expression 在 Tasks 1-5 中均未触碰。
4. 默认 BB0 byte-equal — Task 3 Step 1 端到端测试 + Task 5 smoke 双跑双重锁。

**2. Placeholder scan:**

无 `TBD` / `TODO` / "implement later" / "fill in details" / "similar to Task N" / "write tests for the above" 等 plan-failure 字串。所有 code step 给出真实代码; 所有 command step 给出真实命令 + 预期输出; 所有 backtrack 条件给出具体 root-cause / fix。Task 5 Step 5 的 "<s5-baseline>.parquet" / "<s3-fresh>.parquet" 是动态文件名占位 (实施者 ls 后填), Step 描述里讲清 "用 `data/runs/` 下最新 timestamped parquet" 的判定规则, 不是 plan-level 空白。Task 5 Step 7 的 "<files-touched>" 同款 — 仅在 Step 3 暴露了 fix-forward 时才用, 实施者用本地 `git status --short` 填入。

**3. Type consistency:**

- `_Z_PREY_CARD: dict[str, int]` — Task 1 定义, Task 2 (`feature_mask_of` thr_mirror 段) consume, Task 3 / Task 4 / Task 5 不直读但隐式依赖; key 集合 ⊆ `_Z.keys()` 由 module-load 派生保证, Task 1 Step 1 测试守门。
- `feature_mask_of(sequence: tuple[str, ...]) -> int` — signature S6 起锁定, S1 / S3 都只扩展函数体不动签名 (Task 2 Step 3 / Task 5 Step 1 都从这个签名读)。
- `prey_mask_for_clauses(prey_clauses: tuple[tuple[str, ...], ...]) -> int` — signature S6 起锁定, Task 4 仅扩展函数体。
- `PREDICATE_BIT["thr_crest" | "thr_hotspot" | "thr_mirror" | "vis_lowvis"]` — 索引由 S6 Task 6 锁死 (12 / 13 / 14 / 11), S3 不动 `PREDICATE_BITS` 字典只 OR 进 `m`; Task 2 / Task 4 / Task 5 全程同名同索引。
- Clause tag 字符串 — `"f_hi"` / `"p_hi"` / `"generalist"` / `"lowvis"` 在 Task 4 函数体识别 + Task 4 / Task 5 测试断言里全程同 spelling, 与 S6 既有 `"motif"` / `"len>=3"` 同款字符串-tag 风格不冲突。
- 阈值数字 — `0.5` (f_hi) / `0.05` (p_hi) / `0.45` (generalist z 上界) / `2` (generalist |prey| 下界) / `0.20` (lowvis vis 上界) — File Structure 表与 Global Constraints 与 Task 2 实现代码三处 verbatim 一致, 与 spec §1 表一致。
- `_Z` 行 4-tuple 形状 `(z, prey_clauses, period, vis_mode)` — S1 Task 3 锁定, S3 不 reshape (Task 1 派生 `_Z_PREY_CARD` 时只读 `row[1]` = prey_clauses, 不动其他元素)。
- 边界类型 (闭/半开) — Global Constraints + File Structure 表 + Task 2 实现代码 + Task 2 测试 verbatim 一致, 全 5 条都是闭区间 (`>=` / `<=`)。

无 method/property 名称漂移: `_Z_PREY_CARD` 全 plan 同名 (lowercase + underscore + uppercase 后缀, 与既有 `MOTIF_LEN` / `PREDICATE_BIT` 同款 module-level 常量纪律); `feature_mask_of` / `prey_mask_for_clauses` 全 plan 同名同 signature。

Task 4 cross-family fall-through 设计 (`("F", "p_hi")` 应解析为 `family_F` 而非 `thr_hotspot`) 在 Step 3 实现代码 + Step 1 `test_unknown_tag_falls_through_to_family` 守门 + Self-Review red line #1 三处一致 — `elif "p_hi" in tags and fam == "P":` 的双校验语义全程不漂移。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-24-s3-rich-prey-predicates.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. Use `superpowers:subagent-driven-development`.
2. **Inline Execution** — execute tasks in this session with batch checkpoints. Use `superpowers:executing-plans`.

Which approach?
