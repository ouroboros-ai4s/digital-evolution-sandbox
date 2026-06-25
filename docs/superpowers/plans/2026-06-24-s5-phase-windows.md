# S5 — 相位窗 f (phase windows) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 F 池剩下两个相位窗繁衍基元 (FBURST / F_NOVA) 落地, 给 `Phenotype` 加 4 个相位窗字段 (`f_hi` / `f_lo` / `burst_w` / `burst_k`) + 给 `phenotype_arrays` 加同名 4 张张量, 把 `phase2_reproduce` 里 `f = phe["f"][sid_long]` 一行换成「`on = ((T - birth_tick) % burst_w) < burst_k; f = torch.where(on, f_hi, f_lo)`」, 默认局静态全部基元字节级不变。

**Architecture:** 五件事, 顺序: (1) `Phenotype` dataclass 加 4 字段 `f_hi / f_lo / burst_w / burst_k`, 默认值 `(0.0, 0.0, 1, 1)`, frozen + 默认值不破现有构造点; (2) `_F` 表每行扩展为 7-tuple `(f, dirs, p_leave, period, f_lo, burst_w, burst_k)`, 既有两行 (`F4Nr1` / `F4Nr4`) 默认 `f_lo=f, burst_w=1, burst_k=1` (静态退化), `phenotype()` 累加 F 行时分配 dominant-F 解析(highest-f 优先, 平手取首现), 写 Phenotype 的 `f_hi=f` (stacked) / `f_lo` (用 dominant 的 f_lo 算 stacked-when-off) / `burst_w` / `burst_k`, 同时**保持 `Phenotype.f` 作为 `f_hi` 的别名**让既有读者 (`phenotype_cache.py:f`, `webapp.readouts`) 字节级不变; (3) 注册两个新 F 字母 `FBURST` (f=0.55, f_lo=0.05, burst_w=12, burst_k=2, dirs=4-nbr, p_leave=0.20, period=2) 与 `F_NOVA` (f=0.85, f_lo=0.05, burst_w=20, burst_k=1, dirs=4-nbr, p_leave=0.50, period=2) 进 `ALPHABET` / `GRAN` / `_F`; (4) `phenotype_arrays` 加 4 张张量列 `f_hi: float32 / f_lo: float32 / burst_w: int64 / burst_k: int64` (idx 0 哨兵 0/0/1/1); (5) `phase2_reproduce` 读新 4 列, 一行 `where` 把 `f` 从 `phe["f"]` 升级为 windowed live f, 显式 `clamp(min=1)` 守 `burst_w==0`。

**Tech Stack:** Python 3 (`D:\anaconda3\envs\basic\python.exe`), PyTorch 2.10+cu128, pyarrow, pytest. Windows 主机, `PYTHONPATH=src` 纪律。引擎源码 `src/des/`。**依赖**: S0 (CLI 已就位, 不动) + S6 (`GRAN` 表已落地; FBURST / F_NOVA 加 `gran="residue"` 行) + S4 (F 池 5 个动态方向基元 + 字面 4-tuple dirs 走旧路径, FBURST / F_NOVA 的 dirs=4-nbr 复用 F4Nr4 同款字面 4-tuple); 不与 S1 / S2 交互 (vis 通道与谱形塑都不读 `f`)。

## Global Constraints

- **相位窗是序列结构的纯函数**: `(f, f_lo, burst_w, burst_k)` 全部从 `_F[letter]` 直读, 没有 per-species 量级, 没有 world-state 读, 没有「相位窗谁开得更猛」的手写偏置 (spec §2)。
- **静态默认字节级不变**: 5 个 v1 F 行 (含 S4 落的 5 个新行) 全部 `f_lo=f, burst_w=1, burst_k=1`; 静态 strain `Phenotype.f_hi == Phenotype.f_lo == Phenotype.f`; kernel 里 `(T-birth)%1==0<1` 永远为 True → `f = f_hi = f` (spec §3)。回归锁 = 285+146 引擎/web 测试 + S4/S6/S1/S2 落地后新增测试整套继续全绿。
- **`f` 字段是 `f_hi` 的别名, 不删**: `Phenotype.f` 仍然存在, 等于 `f_hi`; `phenotype_cache.py:69` 的 `f[sid] = phe.f` 不动, `webapp.readouts` 不动, 任何旧代码读 `Phenotype.f` 都得到 `f_hi`, 静态默认下又恰是 stacked f (spec §3)。
- **kernel 算 live f, 不让 phenotype 读 world-state**: `(T - birth_tick) % burst_w` 的形状与 `fires_this_tick((T-birth)%period==0)` 完全一致 (`[H, W, K]`), 都是 kernel 读自己的时钟 + 静态 phe 表 (spec §2, HOW-1)。
- **`burst_w=0` 必须显式 clamp**: `fires_this_tick` 现行用 `period.clamp(min=1)` 守 `%`, 但 `burst_w` 是新列, 没人代它防 `% 0` —— kernel 必须显式 `burst_w.clamp(min=1)` (spec §5)。Phenotype 默认值是 1, 此 clamp 是防御性的, 实际不应触发。
- **多 F 行只走 dominant-F 近似, full per-letter windowed stacking 推迟**: 多 F 字母共存时, 取 `f` 最高的 letter 为 dominant (平手取首现); `f_hi = 1 - Π(1-f_i)` 仍 stacked, `f_lo = 1 - (1-dominant.f_lo) × Π_{i != dom}(1-f_i)`; `burst_w / burst_k` 直接取 dominant 的 (spec §5)。默认 BB0 没有窗口化 F, 此路径不触发。
- **frozen `Phenotype` 加字段必带默认值**: `Phenotype` 是 `@dataclass(frozen=True)` (S4 落地后); 4 个新字段必须有默认值, 否则既有 `Phenotype(...)` 构造点 (S1 / S2 / S4 加进来的) 全得带新 kwarg → 大爆炸 (同 S4 加 `in_place/rand_dir` 与 S1 加 `vis_sum/n_count` 的纪律)。
- **`_F` 行的 7-tuple 形状是新基线**: 既有两行 `F4Nr1` / `F4Nr4` 在本 plan Task 2 集中升级为显式 7-tuple, **同时**该路径必须接住 S4 加的 5 行新 F 字母 —— 它们的 `directions` 字段是 `"hash:<kind>"` / `"rand:1of4"` / `(IN_PLACE_DIR,)` 三态; 形状扩展从 4-tuple 到 7-tuple, 第 5/6/7 元素是 windowed 参数, 三态识别在原位继续生效。
- **dirs / p_leave 不在 S5 范围**: FBURST / F_NOVA 的 dirs=4-nbr 走 F4Nr4 同款**字面 4-tuple** (S4 既有静态路径); `p_leave` 已是 `_F` 行第 3 元素, S5 不动它的语义。S4 owns 动态 dirs 机制, S5 仅 owns f-window (spec §7)。
- **`P_burst_lite` 不属 S5**: roster 的 `P_burst_lite` 是 P 池基元, 没有 `f` (P 基元不出 f), S2 已经按普通 P 行铸完, S5 不重复处理 (spec §1)。
- **out of scope**: FBURST / F_NOVA 的 dirs (S4) 与 p_leave (既存机制); `P_burst_lite` 谱 (S2); full per-letter windowed-f stacking; A 池 (S8); 多位突变 (S7) (spec §7)。

---

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `src/des/types.py` | **Modify** | `Phenotype` 加 4 个新字段 `f_hi: float = 0.0` / `f_lo: float = 0.0` / `burst_w: int = 1` / `burst_k: int = 1`, 保持 `@dataclass(frozen=True)`(S4 已 frozen)与默认值不破现有构造点。`Phenotype.f` 字段保留, 与 `f_hi` 同义但单独存。 |
| `src/des/registry.py` | **Modify** | (a) `_F` 表既有两行 `F4Nr1` / `F4Nr4` 升级为 7-tuple `(f, dirs, p_leave, period, f_lo, burst_w, burst_k)`,静态默认 `f_lo=f, burst_w=1, burst_k=1`;(b) `_F` 加 2 行新 F 基元 `FBURST` / `F_NOVA` 的 7-tuple verbatim 抄 spec §1;(c) `ALPHABET` 加 `FBURST` / `F_NOVA` 两 key,family `"F"`;(d) `GRAN` 加 2 行 `"residue"`(S6 已落 `GRAN` 表);(e) `phenotype()` 在累加 F 行时,记录 dominant-F (highest-f, 平手取首现) 的 `f_lo / burst_w / burst_k`,把 stacked `f_hi = 1 − Π(1 − f_i)` 与 `f_lo_stacked = 1 − (1 − dom.f_lo) × Π_{i≠dom}(1 − f_i)` 写进 `Phenotype`;静态默认下二者相等且等于 stacked f, byte-identical。 |
| `src/des/phenotype_cache.py` | **Modify** | `phenotype_arrays(device)` 加 4 张张量列 `f_hi: float32 / f_lo: float32 / burst_w: int64 / burst_k: int64`,一行一 strain,idx 0 (`EMPTY_ID`) 守默认 `0.0 / 0.0 / 1 / 1`;新 4 列入 dirty-flag rebuild 路径;**`f` 列保留不动**(它已经存 `phe.f` = `f_hi`,既有读者 `webapp.readouts` / 旧 kernel 路径都接 `f`)。 |
| `src/des/kernels/reproduction.py` | **Modify** | `phase2_reproduce` 在 `f = phe["f"][sid_long]` 一行**之上**新读 4 列 `phe["f_hi" / "f_lo" / "burst_w" / "burst_k"][sid_long]`,把那一行替换为 `on = ((T - birth_tick) % burst_w.clamp(min=1)) < burst_k; f = torch.where(on, f_hi, f_lo)`;下游 `binom(scattered, f, generator)` / mut 切分 / roll 全部不动。 |
| `tests/test_phase_windows.py` | **Create** | 新建,S5 owner 文件(同款 sibling: S6 `test_motif.py` / S1 `test_vis.py` / S2 `test_spectrum_shape.py` / S4 `test_direction_kinds.py`)。覆盖 4 字段默认值、`_F` 7-tuple 形状、FBURST / F_NOVA 行注册、phenotype 静态默认 `f_hi==f_lo==f` & `burst_w==burst_k==1`、phenotype FBURST 单 letter `f_hi=0.55 / f_lo=0.05 / burst_w=12 / burst_k=2`、phenotype F_NOVA 同款断言、kernel where-on-window 数值断言(几个 birth offset 跑 1 tick, 子代 count 与裸 binomial 期望分布对齐)、F4Nr4 byte-identical 跑(回归锁)、FBURST 静态 (k=w=1) 等价测试、relabel-invariance(重排 _Z/_P 量级不影响 4 个窗口字段)。 |
| `tests/test_registry.py` | **Modify (append)** | 追加 `_F` 7-tuple 形状断言(全 7+ 行 verbatim)+ FBURST / F_NOVA 在 `ALPHABET` / `GRAN` / `_F` 三表的覆盖与值域。**不动**S6 加的 `feature_mask` / `prey_mask` / `motif_blocks` 断言。 |
| `tests/test_phenotype_cache.py` | **Modify (append)** | 追加一条断言:`phenotype_arrays(device)` 返回的 dict 含 `f_hi` / `f_lo` / `burst_w` / `burst_k` 4 个 key,默认 BB0 全 strain 在这 4 列上是 `(f, f, 1, 1)` 的退化值;dirty-flag 在 mint FBURST 后重建,新 strain 的 `burst_w` 列变成 12。 |
| `tests/test_reproduction.py` | **Modify (append)** | 追加 windowed-f 端到端断言:单 cell 装 FBURST 株, 跑 24 tick, 在 birth_tick=0 起点下,tick 0/1/12/13/24 触发 `f_hi=0.55`(on-window),tick 2..11/14..23 触发 `f_lo=0.05`(off-window);F4Nr4 同 seed 跑 30 tick byte-identical(S4 已有 byte-identical 测试,本任务再追加一份显式 windowed-default 等价)。 |

**Naming contract (locked, used by every task):**

```python
# src/des/types.py
@dataclass(frozen=True)
class Phenotype:
    # ... existing fields (f, directions, p_leave, ..., in_place, rand_dir from S4) ...
    f_hi: float = 0.0       # S5: stacked on-window f (== Phenotype.f)
    f_lo: float = 0.0       # S5: stacked off-window f (静态默认 = f_hi)
    burst_w: int = 1        # S5: window period (静态默认 1)
    burst_k: int = 1        # S5: on-window length (静态默认 1; on=k/w 占空比)

# src/des/registry.py
# _F 行 7-tuple shape:
#   (f, dirs, p_leave, period, f_lo, burst_w, burst_k)
# 既有 / S4 / S5 全 9 行(S4 已加 5 行,S5 加 2 行,既有 2 行升 7-tuple):
#   "F4Nr1":   (0.30, "hash:f4nr1",          0.05, 4, 0.30, 1, 1)         # 静态退化(S4 已 hash-locked)
#   "F4Nr4":   (0.50, ((-1,0),(1,0),(0,-1),(0,1)), 0.15, 5, 0.50, 1, 1)   # 静态退化
#   "FSTACK":  (0.60, (IN_PLACE_DIR,),       0.00, 3, 0.60, 1, 1)         # S4, 静态退化
#   "FCLUMP":  (0.45, "hash:fclump",         0.10, 6, 0.45, 1, 1)         # S4, 静态退化
#   "FFRONT":  (0.50, "hash:ffront",         0.25, 4, 0.50, 1, 1)         # S4, 静态退化
#   "F4Nr3":   (0.40, "hash:f4nr3",          0.12, 5, 0.40, 1, 1)         # S4, 静态退化
#   "FDRIFT":  (0.15, "rand:1of4",           0.30, 2, 0.15, 1, 1)         # S4, 静态退化
#   "FBURST":  (0.55, ((-1,0),(1,0),(0,-1),(0,1)), 0.20, 2, 0.05, 12, 2)  # S5 新, on=0.55, off=0.05
#   "F_NOVA":  (0.85, ((-1,0),(1,0),(0,-1),(0,1)), 0.50, 2, 0.05, 20, 1)  # S5 新, on=0.85, off=0.05

# src/des/phenotype_cache.py
# phe dict 出口形状(S5 后 14 个 key):
#   "f", "p_leave", "z_raw", "p_x", "prey_mask", "feature_mask",
#   "period", "dir_bits", "repro_period", "anta_period",
#   "in_place", "rand_dir",                       # S4 加
#   "f_hi", "f_lo", "burst_w", "burst_k"          # S5 加

# src/des/kernels/reproduction.py
def phase2_reproduce(world, snap_sid, snap_count, snap_faction, phe, table,
                     birth_tick, T, generator) -> tuple[ArrivalBuffer, Tensor]:
    # ... 旧入口不变 ...
    f_hi    = phe["f_hi"][sid_long]                                # [H,W,K] float32
    f_lo    = phe["f_lo"][sid_long]                                # [H,W,K] float32
    burst_w = phe["burst_w"][sid_long].clamp(min=1)                # [H,W,K] int64; defensive clamp
    burst_k = phe["burst_k"][sid_long]                             # [H,W,K] int64
    on = ((T - birth_tick) % burst_w) < burst_k                    # [H,W,K] bool
    f  = torch.where(on, f_hi, f_lo)                               # [H,W,K] float32
    # ... 下游 fires/binom/roll 路径完全不动 ...
```

`Phenotype.f` 不删, 与 `f_hi` 同步赋值; 静态默认下 `f == f_hi == f_lo`, 既有 reader 字节级不变。dominant-F 解析在 `phenotype()` 内部完成, kernel 永远不读 `_F` 表 / 不读 `Phenotype` 对象, 只读 `phenotype_arrays` 的张量列 —— 三层职责清晰: registry 描述, phenotype 翻译, kernel 执行。

---

### Task 1: `Phenotype` 加 4 个相位窗字段(无消费者)

**Goal:** 给 `src/des/types.py` 的 frozen `Phenotype` dataclass 加 4 个新字段 `f_hi: float = 0.0` / `f_lo: float = 0.0` / `burst_w: int = 1` / `burst_k: int = 1`。**必须给默认值**, 否则既有所有 `Phenotype(...)` 构造点(`registry.py::phenotype` 末尾, S1 / S2 / S4 落地后的构造点, 测试夹具)全得带新 kwarg → 大爆炸(同 S4 加 `in_place/rand_dir` 与 S1 加 `vis_sum/n_count` 的纪律)。这一步无消费者(下一 Task 才让 `phenotype()` 写它), 既有行为 0 漂移。

**Files:**
- Modify: `src/des/types.py` (扩 `Phenotype` 字段列表, 紧跟 S4 加的 `in_place` / `rand_dir` 之后)
- Test: `tests/test_phase_windows.py` (Create — first test for this owner file; later tasks 续填)

**Interfaces:**
- Consumes: 无。
- Produces:
  - `Phenotype.f_hi: float = 0.0` —— stacked on-window f (== `Phenotype.f` 的同步副本)。
  - `Phenotype.f_lo: float = 0.0` —— stacked off-window f, 默认与 `f_hi` 相等。
  - `Phenotype.burst_w: int = 1` —— 窗口周期(静态默认 1 → 永远 on)。
  - `Phenotype.burst_k: int = 1` —— on-window 长度(静态默认 1 → 占空比 100%)。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_phase_windows.py`:

```python
# tests/test_phase_windows.py
"""S5 phase-window f primitives + kernel where-on-window branch.

This file is the S5 owner test file (sibling: tests/test_motif.py /
test_vis.py / test_spectrum_shape.py / test_direction_kinds.py). Covers
Phenotype 4 new fields, _F row 7-tuple shape, FBURST/F_NOVA registration,
phenotype dominant-F resolution, phenotype_arrays 4 new columns, kernel
where(on,f_hi,f_lo) branch, static-default byte-identity."""
from __future__ import annotations
import pytest


def test_phenotype_has_f_hi_field():
    """Phenotype.f_hi 字段存在, float 类型."""
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert hasattr(p, "f_hi")
    assert isinstance(p.f_hi, float)


def test_phenotype_has_f_lo_field():
    """Phenotype.f_lo 字段存在, float 类型."""
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert hasattr(p, "f_lo")
    assert isinstance(p.f_lo, float)


def test_phenotype_has_burst_w_default_one():
    """Phenotype.burst_w 默认 1 —— 静态默认下 kernel `(T-birth)%1==0<1` 永远 True."""
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert hasattr(p, "burst_w")
    assert isinstance(p.burst_w, int)
    assert p.burst_w >= 1


def test_phenotype_has_burst_k_default_one():
    """Phenotype.burst_k 默认 1 —— 静态默认占空比 100% (k/w = 1/1)."""
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert hasattr(p, "burst_k")
    assert isinstance(p.burst_k, int)
    assert p.burst_k >= 1


def test_phenotype_is_still_frozen_after_s5_fields():
    """加字段后 dataclass 仍 frozen, 不可改: 守不可变契约."""
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    with pytest.raises(Exception):
        p.f_hi = 0.99       # FrozenInstanceError under @dataclass(frozen=True)
```

- [ ] **Step 2: 跑失败测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_phase_windows.py -v
```

Expected: 5 条全 FAIL —— `AttributeError: 'Phenotype' object has no attribute 'f_hi'` (其他 3 个字段同理); `test_phenotype_is_still_frozen_after_s5_fields` 也 FAIL 因 `f_hi` 不存在,无法触发 frozen check。

- [ ] **Step 3: 给 `Phenotype` 加 4 个新字段**

打开 `src/des/types.py`, 把 `Phenotype` dataclass 扩展为(在 S4 已加的 `in_place` / `rand_dir` 之后追加 4 行):

```python
# src/des/types.py
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum

EMPTY_ID = 0
FAMILY_RANK = {"N": 0, "F": 1, "P": 2, "Z": 3}

class PhaseType(IntEnum):
    ANTAGONISM = 1
    REPRODUCTION = 2

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
    in_place: bool = False     # S4: FSTACK — 当格沉积, 内核独立分支
    rand_dir: bool = False     # S4: FDRIFT — 每 firing tick 现抽 1-of-4
    f_hi: float = 0.0          # S5: stacked on-window f (== Phenotype.f)
    f_lo: float = 0.0          # S5: stacked off-window f (静态默认 = f_hi)
    burst_w: int = 1           # S5: window period (静态默认 1 → 永远 on)
    burst_k: int = 1           # S5: on-window length (静态默认 1 → 占空比 100%)
```

(默认值 `0.0 / 0.0 / 1 / 1` 让既有 `Phenotype(...)` 构造点不需要带新 kwarg, 编辑别处 0 改动。**注意默认值的字段必须放在所有非默认字段之后**, 同 S4 加 `in_place / rand_dir` 时的纪律。)

- [ ] **Step 4: 跑测试, 确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_phase_windows.py -v
```

Expected: 5 条全 PASS。

- [ ] **Step 5: 跑全 suite, 确认无回归**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿。既有所有 strain 落 `f_hi=0.0 / f_lo=0.0 / burst_w=1 / burst_k=1` 默认值,phenotype 构造点不读、kernel 不读, 字节级不变。

Backtrack: 若 `TypeError: non-default argument follows default argument` 出现, 把 4 新字段往 dataclass 最末尾挪 —— 它们必须在所有 default-less 字段之后(`fold` 之后, S4 加的两 bool 之后)。若某个测试夹具复现了 `Phenotype(...)` 的位置参数且把 `fold` 后面的字段也按位置传, 改成 kwarg。

- [ ] **Step 6: Commit**

```bash
git add src/des/types.py tests/test_phase_windows.py
git commit -m "feat(s5): add Phenotype f_hi / f_lo / burst_w / burst_k fields

Four new frozen dataclass fields with defaults (0.0, 0.0, 1, 1) — FBURST /
F_NOVA will populate them with their windowed (on, off, period, on-length)
parameters; static defaults make every existing v1/S4 strain byte-equal to
the pre-S5 path because (T-birth)%1==0<1 is always True so live f == f_hi.
No consumer yet — Task 2 wires phenotype(); Task 4 wires phenotype_arrays;
Task 5 wires the kernel."
```

---

### Task 2: `_F` 行 7-tuple 化 + `phenotype()` dominant-F 窗解析

**Goal:** 把 `src/des/registry.py` 的 `_F` 表从 4-tuple `(f, dirs, p_leave, period)` 升级为 7-tuple `(f, dirs, p_leave, period, f_lo, burst_w, burst_k)`,既有 2 行 `F4Nr1` / `F4Nr4` 静态默认 `f_lo=f, burst_w=1, burst_k=1`(byte-identical 退化);同时改 `phenotype()` 累加 F 行的循环,记录 dominant-F (highest-f, 平手取首现) 的 `f_lo / burst_w / burst_k`,把 stacked `f_hi = 1 − Π(1 − f_i)` 与 `f_lo_stacked = 1 − (1 − dom.f_lo) × Π_{i≠dom}(1 − f_i)` 写进 `Phenotype`;**保持 `Phenotype.f` 与 `f_hi` 同步**(`f = f_hi`),让 `phenotype_cache.py` 的 `f[sid] = phe.f` 路径接到 stacked-on-window 值。这一步仅碰既有 2 行 F 字母 + S4 加的 5 行 F 字母(全部静态退化),**不**注册 FBURST / F_NOVA(Task 3 来)。

**Files:**
- Modify: `src/des/registry.py:27-30` (`_F` 表升 7-tuple,既有 2 行)+ `src/des/registry.py:72-86` (`phenotype()` 的 `_F` 分支累加)+ `Phenotype(...)` 构造调用末尾(line 119-125,加 4 个新 kwarg)
- (若 S4 已落地)`src/des/registry.py` 中 S4 加的 5 行 F 字母同步升 7-tuple
- Test: `tests/test_phase_windows.py` (append) + `tests/test_registry.py` (append 1 条 7-tuple 形状断言)

**Interfaces:**
- Consumes: `Phenotype.f_hi / f_lo / burst_w / burst_k` (Task 1); S4 的 `_hash_dirs` / `IN_PLACE_DIR` / `Phenotype.in_place / rand_dir`(若 S4 已落)。
- Produces:
  - `_F[letter] = (f, dirs, p_leave, period, f_lo, burst_w, burst_k)` —— 7-tuple,既有 2 行 + S4 加的 5 行同款形状(全部静态默认 `f_lo=f, burst_w=1, burst_k=1`)。
  - `phenotype(seq).f_hi: float` —— stacked on-window f,与 `Phenotype.f` 同步。
  - `phenotype(seq).f_lo: float` —— stacked off-window f;静态默认下等于 `f_hi`。
  - `phenotype(seq).burst_w / burst_k: int` —— dominant-F 的窗口参数;静态默认 1 / 1。

- [ ] **Step 1: 写失败测试 —— 静态默认 byte-identical + dominant-F 解析**

追加到 `tests/test_phase_windows.py`:

```python
def test_existing_F_rows_static_default_means_f_hi_eq_f_lo_eq_f():
    """既有 v1 F 字母 (F4Nr1 / F4Nr4) 静态默认:
       phenotype.f_hi == phenotype.f_lo == phenotype.f, 且 burst_w=burst_k=1."""
    from des.registry import phenotype
    p_f4nr1 = phenotype(("F4Nr1",) + ("N0",) * 15)
    assert p_f4nr1.f_hi == p_f4nr1.f
    assert p_f4nr1.f_lo == p_f4nr1.f
    assert p_f4nr1.burst_w == 1
    assert p_f4nr1.burst_k == 1
    p_f4nr4 = phenotype(("F4Nr4",) + ("N0",) * 15)
    assert p_f4nr4.f_hi == p_f4nr4.f
    assert p_f4nr4.f_lo == p_f4nr4.f
    assert p_f4nr4.burst_w == 1
    assert p_f4nr4.burst_k == 1


def test_default_bb0_layout_phenotype_static_default():
    """默认 BB0 layout (含 F4Nr4 在 locked 位置 1) 的 phenotype:
       f_hi==f_lo==f, burst_w=burst_k=1, byte-identical 退化."""
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert p.f_hi == p.f
    assert p.f_lo == p.f
    assert p.burst_w == 1
    assert p.burst_k == 1


def test_multi_F_static_strain_stacks_f_via_one_minus_prod(monkeypatch):
    """同 strain 两 F 字母 (都是静态退化), f_hi 应是 1 - (1-f1)(1-f2);
       f_lo 同表 (因为 f_lo_i = f_i, dom 的 f_lo 也 = dom.f);
       burst_w / burst_k 取 dominant 的, 默认都是 1."""
    from des.registry import phenotype
    # F4Nr4 (f=0.50) + F4Nr1 (f=0.30) 共存; F4Nr4 是 dominant
    seq = ("F4Nr4", "F4Nr1") + ("N0",) * 14
    p = phenotype(seq)
    expected_f = 1 - (1 - 0.50) * (1 - 0.30)
    assert abs(p.f - expected_f) < 1e-9
    assert p.f_hi == p.f
    assert abs(p.f_lo - expected_f) < 1e-9   # 静态 f_lo == f
    assert p.burst_w == 1
    assert p.burst_k == 1


def test_F_row_is_7_tuple_for_existing_letters():
    """既有两行 _F 升 7-tuple, 元素数严格 == 7."""
    from des.registry import _F
    for letter in ("F4Nr1", "F4Nr4"):
        row = _F[letter]
        assert len(row) == 7, f"{letter}: expected 7-tuple, got {row!r} (len={len(row)})"
        f, dirs, p_leave, period, f_lo, burst_w, burst_k = row
        assert isinstance(f, float)
        # static-default: f_lo == f, burst_w=1, burst_k=1
        assert f_lo == f, f"{letter}: f_lo {f_lo} must equal f {f} on static default"
        assert burst_w == 1
        assert burst_k == 1


def test_phenotype_f_field_is_alias_of_f_hi():
    """Phenotype.f 永远 == Phenotype.f_hi (alias 不变契约).
       既有读 `phe.f` 的路径 (phenotype_cache.py / webapp.readouts) 都拿 stacked on-window."""
    from des.registry import phenotype
    seqs = (
        ("F4Nr1",) + ("N0",) * 15,
        ("F4Nr4",) + ("N0",) * 15,
        ("F4Nr4", "F4Nr1") + ("N0",) * 14,
    )
    for s in seqs:
        p = phenotype(s)
        assert p.f == p.f_hi, f"seq={s!r}: f {p.f} != f_hi {p.f_hi}"
```

追加到 `tests/test_registry.py`:

```python
def test_existing_F_rows_are_7_tuple_post_s5():
    """S5: _F 行 7-tuple 形状, 既有 F4Nr1 / F4Nr4 静态退化默认值."""
    from des.registry import _F
    f4nr1 = _F["F4Nr1"]
    assert len(f4nr1) == 7
    # f, dirs, p_leave, period, f_lo, burst_w, burst_k
    assert f4nr1[0] == 0.30
    assert f4nr1[2] == 0.05
    assert f4nr1[3] == 4
    assert f4nr1[4] == 0.30      # f_lo == f static
    assert f4nr1[5] == 1         # burst_w
    assert f4nr1[6] == 1         # burst_k
    f4nr4 = _F["F4Nr4"]
    assert len(f4nr4) == 7
    assert f4nr4[0] == 0.50
    assert f4nr4[2] == 0.15
    assert f4nr4[3] == 5
    assert f4nr4[4] == 0.50
    assert f4nr4[5] == 1
    assert f4nr4[6] == 1
```

- [ ] **Step 2: 跑失败测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_phase_windows.py tests/test_registry.py::test_existing_F_rows_are_7_tuple_post_s5 -v
```

Expected: 上面 5 条新测试 + registry 7-tuple 形状测试全 FAIL —— 因为 `_F` 仍是 4-tuple,且 `phenotype()` 没写 4 个新字段。

- [ ] **Step 3: 升级 `_F` 表为 7-tuple(既有 2 行 + S4 已加 5 行,全部静态退化)**

把 `src/des/registry.py:27-30` 的 `_F` 表替换为(若 S4 已落地, 此 5 行已存; 同步把它们也升 7-tuple):

```python
# per-letter raw outputs (design tables; numbers are formula anchors, not calibrated knobs)
_F = {    # name -> (f, dirs, p_leave, period, f_lo, burst_w, burst_k)
    # 静态默认: f_lo=f, burst_w=1, burst_k=1 → kernel where 永远取 f_hi (byte-identical).
    "F4Nr1": (0.30, "hash:f4nr1", 0.05, 4, 0.30, 1, 1),                              # S4 升 hash-locked, S5 升 7-tuple
    "F4Nr4": (0.50, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.15, 5, 0.50, 1, 1),        # S5 升 7-tuple
    # S4 已加的 5 个 F 字母, 静态退化 (Task 3 不动这 5 行, 仅在此 Task 同步升 7-tuple):
    "FSTACK":  (0.60, (0, 0),         0.00, 3, 0.60, 1, 1),                          # 注: (IN_PLACE_DIR,) 单元素 tuple
    "FCLUMP":  (0.45, "hash:fclump",  0.10, 6, 0.45, 1, 1),
    "FFRONT":  (0.50, "hash:ffront",  0.25, 4, 0.50, 1, 1),
    "F4Nr3":   (0.40, "hash:f4nr3",   0.12, 5, 0.40, 1, 1),
    "FDRIFT":  (0.15, "rand:1of4",    0.30, 2, 0.15, 1, 1),
}
```

(注:`FSTACK` 行的 `dirs` 是 `(IN_PLACE_DIR,)` 这个单元素 tuple,在 S4 落地后 `IN_PLACE_DIR = (0, 0)` 已是模块级常量; 这里用字面 `(0, 0)` 仅作展示, 实际写 `(IN_PLACE_DIR,)`,保持与 S4 一致。若 S4 尚未落地,Task 2 仅改既有两行,FSTACK 等 5 行由 S4 任务自己升 7-tuple。)

- [ ] **Step 4: 改 `phenotype()` 的 `_F` 分支累加 dominant-F + 窗口字段**

打开 `src/des/registry.py:56-125`,把 `phenotype()` 函数顶部的累加器初始化(line 59-70)与 `_F` 分支(line 76-86)替换为:

```python
def phenotype(sequence: tuple[str, ...]) -> Phenotype:
    """Pure function of the sequence only. No world-state, no neighbors, no tick.
    κ=0 in v1 — no self-coordination neighbor scan."""
    f_prod = 1.0          # accumulate Π(1-fᵢ)
    pl_prod = 1.0
    px_prod = 1.0
    z_sum = 0.0
    prey_mask = 0
    feature_mask = 0
    directions: list[tuple[int, int]] = []
    periods: list[int] = []
    f_periods: list[int] = []
    z_periods: list[int] = []
    phase_type: PhaseType | None = None
    dominant_p: str | None = None
    # S4: in_place / rand_dir (if S4 落地后)
    in_place = False
    rand_dir = False
    # S5: dominant-F window resolution. 同 strain 多 F 字母时, 取 f 最高的 letter
    # 作为 dominant; 平手取首现 (sequence order). dominant 的 (f_lo, burst_w, burst_k)
    # 用于 stacked f_lo 与 kernel where 的窗口参数.
    dom_f_letter: str | None = None
    dom_f_value: float = -1.0
    dom_f_lo: float = 0.0
    dom_burst_w: int = 1
    dom_burst_k: int = 1
    # 收集每条 F 行的 (f, f_lo) 用于 stacked f_lo: 1 - (1-dom.f_lo) × Π_{i≠dom}(1 - f_i)
    f_each: list[tuple[str, float, float]] = []   # (letter, f, f_lo)

    for letter in sequence:
        if letter not in ALPHABET:
            continue
        feature_mask |= FEATURE_BIT[letter]
        if letter in _F:
            row = _F[letter]
            # 7-tuple shape: (f, dirs, p_leave, period, f_lo, burst_w, burst_k)
            f, dirs, pl, per, f_lo, b_w, b_k = row
            f_prod *= (1 - f)
            pl_prod *= (1 - pl)
            # S4: dirs 三态识别 (字面 tuple / "hash:<kind>" / "rand:1of4")
            # —— 这部分代码 S4 落地后已在原位, 7-tuple 升级不动它的逻辑.
            if isinstance(dirs, str):
                if dirs == "rand:1of4":
                    rand_dir = True
                elif dirs.startswith("hash:"):
                    kind = dirs[len("hash:"):]
                    for d in _hash_dirs(sequence, kind):
                        if d not in directions:
                            directions.append(d)
            else:
                if dirs == (IN_PLACE_DIR,):
                    in_place = True
                else:
                    for d in dirs:
                        if d not in directions:
                            directions.append(d)
            periods.append(per)
            f_periods.append(per)
            phase_type = PhaseType.REPRODUCTION
            # S5: 记录每条 F 行的 (f, f_lo) + dominant-F 候选
            f_each.append((letter, f, f_lo))
            if f > dom_f_value:
                dom_f_letter = letter
                dom_f_value = f
                dom_f_lo = f_lo
                dom_burst_w = b_w
                dom_burst_k = b_k
        elif letter in _Z:
            # ... (Z 分支 S6 / S1 已改过, 此处不动) ...
            pass    # (略, 沿用既有代码)
        elif letter in _P:
            # ... (P 分支 S2 已改过, 此处不动) ...
            pass    # (略, 沿用既有代码)

    f = 1 - f_prod
    p_leave = 1 - pl_prod
    p_x = max(MU, 1 - px_prod)
    spectrum = _spectrum_for(dominant_p) if dominant_p else ()
    period = min(periods) if periods else 1
    repro_period = min(f_periods) if f_periods else 1
    anta_period = min(z_periods) if z_periods else 1
    dir_bits = 0
    for d in directions:
        dir_bits |= _DIR_BIT.get(d, 0)

    # S5: stacked f_hi = stacked f; stacked f_lo = 1 - (1 - dom.f_lo) × Π_{i≠dom}(1 - f_i)
    f_hi = f
    if dom_f_letter is None:
        # 无 F 字母: f_lo 也无意义, 默认 0.0 (与 f_hi 同, 因 f_hi=0)
        f_lo_stacked = 0.0
        burst_w_out = 1
        burst_k_out = 1
    else:
        non_dom_prod = 1.0
        for letter_i, f_i, _ in f_each:
            if letter_i == dom_f_letter:
                continue
            non_dom_prod *= (1 - f_i)
        f_lo_stacked = 1.0 - (1.0 - dom_f_lo) * non_dom_prod
        burst_w_out = int(dom_burst_w)
        burst_k_out = int(dom_burst_k)

    return Phenotype(
        f=f, directions=tuple(directions), p_leave=p_leave, z_raw=z_sum,
        prey_mask=prey_mask, feature_mask=feature_mask, p_x=p_x,
        spectrum=spectrum, period=period,
        repro_period=repro_period, anta_period=anta_period, dir_bits=dir_bits,
        phase_type=phase_type, fold=(),
        in_place=in_place, rand_dir=rand_dir,
        f_hi=f_hi, f_lo=f_lo_stacked,
        burst_w=burst_w_out, burst_k=burst_k_out,
    )
```

(`_Z` / `_P` 分支保持原状,只是为 plan 可读性在示例代码里用 `pass` 占位;实际落地时把 S6 / S1 / S2 已经写好的代码完整保留,**只**改 `_F` 分支累加与 `Phenotype(...)` 构造调用末尾的新 4 个 kwarg。)

- [ ] **Step 5: 跑测试, 确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_phase_windows.py tests/test_registry.py -v
```

Expected: Task 1 + Task 2 的 phase_windows 测试全 PASS;新 `test_existing_F_rows_are_7_tuple_post_s5` PASS;既有 `test_dir_bits_match_directions` / `test_phenotype_*` 类断言仍 PASS(`Phenotype.f` 字段语义未变, dir_bits 路径未碰)。

Backtrack:
- 若 `test_multi_F_static_strain_stacks_f_via_one_minus_prod` 的 `f_lo` 偏差:检查 `non_dom_prod` 循环是否真的跳过了 dominant letter(`if letter_i == dom_f_letter: continue`);若 strain 同 letter 重复出现(罕见但可能),要按 sequence order 跳过仅首次出现的那个 dominant 实例。本 plan 取「按 letter 名字过滤」是 spec §5 dominant-F 近似的精确定义。
- 若 `Phenotype(...)` 构造抛 `TypeError`:检查 `f_lo_stacked` 是否作为 `f_lo=` kwarg 传(注意变量名:函数内本地是 `f_lo_stacked`,kwarg 名是 `f_lo`)。

- [ ] **Step 6: 跑全 suite, 标记预期受影响的测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿。`Phenotype.f` 仍是 stacked on-window(== `f_hi`),`phenotype_cache.py:69` 的 `f[sid] = phe.f` 不动,kernel 没改, 任何既有数值断言都不漂移。

判定:`_F` 由 4-tuple 升 7-tuple,如果某测试在 `_F["F4Nr1"]` 上做 `(f, dirs, pl, per) = _F["F4Nr1"]` 这种位置解包,本步骤会 FAIL with `ValueError: too many values to unpack`。grep `_F\[` 模式定位修正点。本步骤必须修这种,因为它会阻断后续 task。

```
grep -nE "_F\[" src/ tests/ | grep -v "registry.py"
```

Expected: 命中需逐一升级(位置解包 → 7-tuple 解包,或 named 索引 `_F[letter][0]`)。

- [ ] **Step 7: Commit**

```bash
git add src/des/registry.py tests/test_phase_windows.py tests/test_registry.py
git commit -m "feat(s5): _F rows 7-tuple + phenotype dominant-F window resolution

_F[letter] = (f, dirs, p_leave, period, f_lo, burst_w, burst_k); existing
F4Nr1/F4Nr4 (and S4-added FSTACK/FCLUMP/FFRONT/F4Nr3/FDRIFT) static-default
to f_lo=f, burst_w=1, burst_k=1 → byte-identical pre-S5 behavior. phenotype()
picks the highest-f F letter as dominant (first occurrence on tie), stacks
f_hi via 1-Π(1-f_i), stacks f_lo via 1-(1-dom.f_lo)×Π_{i≠dom}(1-f_i);
burst_w / burst_k pass through dominant. Phenotype.f stays as f_hi alias —
phenotype_cache.py f column unchanged, webapp.readouts unchanged."
```

---

### Task 3: 注册 FBURST / F_NOVA 两个新 F 基元

**Goal:** spec §1 表里两个新 F 基元 FBURST 与 F_NOVA 入 `ALPHABET` / `GRAN` / `_F`,7-tuple 形状 verbatim 抄 spec §1。FBURST: `f=0.55, dirs=4-nbr, p_leave=0.20, period=2, f_lo=0.05, burst_w=12, burst_k=2`(每 12 tick 前 2 tick on-window);F_NOVA: `f=0.85, dirs=4-nbr, p_leave=0.50, period=2, f_lo=0.05, burst_w=20, burst_k=1`(每 20 tick 仅前 1 tick on-window)。dirs 用 F4Nr4 同款**字面 4-tuple** `((-1, 0), (1, 0), (0, -1), (0, 1))`(spec §7 明确 dirs=4-nbr 不归 S4 hash-locked)。

**Files:**
- Modify: `src/des/registry.py` —— `ALPHABET` 块、`GRAN` 块、`_F` 块各加 2 行(FBURST / F_NOVA)
- Test: `tests/test_phase_windows.py` (append) + `tests/test_registry.py` (append)

**Interfaces:**
- Consumes: Task 2 落地的 7-tuple `_F` 形状 + `phenotype()` dominant-F 解析。
- Produces:
  - `ALPHABET["FBURST"] = "F"`、`ALPHABET["F_NOVA"] = "F"` —— family `"F"`。
  - `GRAN["FBURST"] = "residue"`、`GRAN["F_NOVA"] = "residue"` —— 单 letter, residue gran。
  - `_F["FBURST"] = (0.55, ((-1,0), (1,0), (0,-1), (0,1)), 0.20, 2, 0.05, 12, 2)`。
  - `_F["F_NOVA"] = (0.85, ((-1,0), (1,0), (0,-1), (0,1)), 0.50, 2, 0.05, 20, 1)`。

- [ ] **Step 1: 写失败测试 —— 注册表覆盖 + 单 letter phenotype**

追加到 `tests/test_phase_windows.py`:

```python
def test_s5_FBURST_present_in_alphabet_with_family_F():
    """FBURST family 'F'."""
    from des.registry import ALPHABET
    assert ALPHABET.get("FBURST") == "F"


def test_s5_F_NOVA_present_in_alphabet_with_family_F():
    """F_NOVA family 'F'."""
    from des.registry import ALPHABET
    assert ALPHABET.get("F_NOVA") == "F"


def test_s5_FBURST_and_F_NOVA_have_gran_residue():
    """FBURST / F_NOVA 是 residue gran(单 letter, 非 motif)."""
    from des.registry import GRAN
    assert GRAN["FBURST"] == "residue"
    assert GRAN["F_NOVA"] == "residue"


def test_s5_FBURST_row_verbatim():
    """_F['FBURST'] = (0.55, 4-nbr, 0.20, 2, 0.05, 12, 2) —— spec §1 表."""
    from des.registry import _F
    expected = (0.55, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.20, 2, 0.05, 12, 2)
    assert _F["FBURST"] == expected, f"got {_F['FBURST']!r}"


def test_s5_F_NOVA_row_verbatim():
    """_F['F_NOVA'] = (0.85, 4-nbr, 0.50, 2, 0.05, 20, 1) —— spec §1 表."""
    from des.registry import _F
    expected = (0.85, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.50, 2, 0.05, 20, 1)
    assert _F["F_NOVA"] == expected, f"got {_F['F_NOVA']!r}"


def test_phenotype_FBURST_single_letter_has_correct_window_params():
    """单 FBURST strain (其他位置 N0): phenotype f_hi=0.55, f_lo=0.05,
       burst_w=12, burst_k=2, dir_bits = 4-nbr all-set."""
    from des.registry import phenotype, ALL_DIRECTIONS
    p = phenotype(("FBURST",) + ("N0",) * 15)
    assert abs(p.f_hi - 0.55) < 1e-9
    assert abs(p.f_lo - 0.05) < 1e-9
    assert p.burst_w == 12
    assert p.burst_k == 2
    # dir_bits 全 4 邻置位 (与 F4Nr4 同 dirs)
    assert p.dir_bits == (1 << len(ALL_DIRECTIONS)) - 1
    # Phenotype.f == f_hi (alias 契约)
    assert p.f == p.f_hi


def test_phenotype_F_NOVA_single_letter_has_correct_window_params():
    """单 F_NOVA strain: phenotype f_hi=0.85, f_lo=0.05, burst_w=20, burst_k=1."""
    from des.registry import phenotype, ALL_DIRECTIONS
    p = phenotype(("F_NOVA",) + ("N0",) * 15)
    assert abs(p.f_hi - 0.85) < 1e-9
    assert abs(p.f_lo - 0.05) < 1e-9
    assert p.burst_w == 20
    assert p.burst_k == 1
    assert p.dir_bits == (1 << len(ALL_DIRECTIONS)) - 1
    assert p.f == p.f_hi


def test_phenotype_FBURST_plus_static_F_dominant_is_FBURST(monkeypatch):
    """FBURST (f=0.55) + F4Nr1 (f=0.30) 共存: FBURST 是 dominant (f 更高);
       stacked f_hi = 1-(1-0.55)(1-0.30) = 0.685;
       stacked f_lo = 1-(1-0.05)(1-0.30) = 1-0.665 = 0.335;
       burst_w/burst_k 取 FBURST 的 12/2."""
    from des.registry import phenotype
    seq = ("FBURST", "F4Nr1") + ("N0",) * 14
    p = phenotype(seq)
    expected_f_hi = 1 - (1 - 0.55) * (1 - 0.30)
    expected_f_lo = 1 - (1 - 0.05) * (1 - 0.30)
    assert abs(p.f_hi - expected_f_hi) < 1e-9, f"got f_hi={p.f_hi}"
    assert abs(p.f_lo - expected_f_lo) < 1e-9, f"got f_lo={p.f_lo}"
    assert p.burst_w == 12
    assert p.burst_k == 2


def test_phenotype_static_F_plus_FBURST_dominant_still_FBURST():
    """无论 sequence order, dominant 总是 f 最高那个; F4Nr1 在前不改判."""
    from des.registry import phenotype
    seq = ("F4Nr1", "FBURST") + ("N0",) * 14
    p = phenotype(seq)
    # FBURST f=0.55 > F4Nr1 f=0.30 → dominant 仍 FBURST
    assert p.burst_w == 12
    assert p.burst_k == 2
```

追加到 `tests/test_registry.py`:

```python
def test_s5_alphabet_contains_FBURST_and_F_NOVA():
    from des.registry import ALPHABET
    assert "FBURST" in ALPHABET
    assert "F_NOVA" in ALPHABET
    assert ALPHABET["FBURST"] == "F"
    assert ALPHABET["F_NOVA"] == "F"


def test_s5_gran_covers_FBURST_and_F_NOVA():
    """S6 加的 GRAN-covers-every-ALPHABET-letter 不变量, S5 加完后仍然 hold."""
    from des.registry import GRAN, ALPHABET
    assert set(GRAN.keys()) == set(ALPHABET.keys())
    assert GRAN["FBURST"] == "residue"
    assert GRAN["F_NOVA"] == "residue"
```

- [ ] **Step 2: 跑失败测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_phase_windows.py tests/test_registry.py -v -k "FBURST or F_NOVA or s5_alphabet or s5_gran"
```

Expected: 注册表覆盖 + phenotype 行覆盖测试全 FAIL with `KeyError: 'FBURST'` / `assert None == 'F'`。

- [ ] **Step 3: 在 `src/des/registry.py` 加 FBURST / F_NOVA 两行**

把 `ALPHABET` 块扩展为(在末尾追加 2 行,与 S4 / S6 / S2 落字母同纪律 —— 只追加, 不改顺序):

```python
ALPHABET = {
    "N0": "N",
    "F4Nr1": "F", "F4Nr4": "F",
    "P_base": "P", "P_hotspot": "P",
    "BroadSweep": "Z",
    # ... (S4 已加的 5 个 F 字母, S2 已加的 10 个 P 字母, 此处不重复展开) ...
    # S5: F-pool phase-window primitives (spec §1)
    "FBURST":   "F",
    "F_NOVA":   "F",
}
```

把 `GRAN` 块扩展为(末尾追加 2 行):

```python
GRAN: dict[str, str] = {
    # ... 已有 ALPHABET 中每字母对应 gran (S6 / S4 / S2 已写) ...
    # S5: F-pool phase-window primitives
    "FBURST":   "residue",
    "F_NOVA":   "residue",
}
```

把 `_F` 块扩展为(在 Task 2 已升 7-tuple 的 7 行之后,追加 2 行):

```python
_F = {    # name -> (f, dirs, p_leave, period, f_lo, burst_w, burst_k)
    # ... (Task 2 已落 7 行: F4Nr1 / F4Nr4 + S4 的 5 行 FSTACK/FCLUMP/FFRONT/F4Nr3/FDRIFT) ...
    # S5: 2 new phase-window F primitives (spec §1 verbatim)
    "FBURST":  (0.55, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.20, 2, 0.05, 12, 2),
    "F_NOVA":  (0.85, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.50, 2, 0.05, 20, 1),
}
```

(注:FBURST / F_NOVA 的 dirs 与 F4Nr4 完全相同的字面 4-tuple,phenotype 走 Task 2 的「字面 tuple → OR-into-directions」路径,把 4 bit 全置 dir_bits。dirs 已由 S4 / S6 既有 dir_bits 机制接住,不需要任何新代码。)

- [ ] **Step 4: 跑测试, 确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_phase_windows.py tests/test_registry.py -v
```

Expected: Task 1 + Task 2 + Task 3 的 phase_windows 测试全 PASS;`test_s5_alphabet_contains_FBURST_and_F_NOVA` / `test_s5_gran_covers_FBURST_and_F_NOVA` PASS;`test_existing_F_rows_are_7_tuple_post_s5` 仍 PASS。

Backtrack:
- 若 `test_phenotype_FBURST_plus_static_F_dominant_is_FBURST` FAIL,root-cause:`f_each` 列表必须按出现顺序追加, dominant 选择必须 `f > dom_f_value` 严格大于(保证平手取首现);若用 `>=` 会把后出现的同 f 字母变成 dominant。
- 若 dir_bits 仅 1 个 bit(而非 4 个), FBURST 的 dirs 没被识别成字面 4-tuple —— 检查 `dirs == (IN_PLACE_DIR,)` 这条分支没误中(`((-1,0), (1,0), (0,-1), (0,1))` 显然不等于 `((0, 0),)`,但若拼写打错很容易引入这个 bug)。

- [ ] **Step 5: 跑全 suite, 确认无回归**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿。FBURST / F_NOVA 行加进 `_F` 但既有 BB0 strain 不会 mint 它们(它们不在 BB0 模板任何位置)。phenotype_arrays 没人读 f_hi / f_lo / burst_w / burst_k(Task 4 才加张量列),kernel 没改, 默认局完全字节级不变。

- [ ] **Step 6: Commit**

```bash
git add src/des/registry.py tests/test_phase_windows.py tests/test_registry.py
git commit -m "feat(s5): register FBURST and F_NOVA F-pool phase-window primitives

FBURST (f=0.55 on, f_lo=0.05 off, burst_w=12, burst_k=2; period 2; 4-nbr;
p_leave=0.20) and F_NOVA (f=0.85 on, f_lo=0.05 off, burst_w=20, burst_k=1;
period 2; 4-nbr; p_leave=0.50) added to ALPHABET / GRAN / _F. dirs are
literal 4-tuples (same as F4Nr4) — S4 hash-locked path not engaged here.
Default BB0 doesn't mint either letter, so default-run dynamics still
identical pre-S5; phenotype_arrays consumer in Task 4, kernel in Task 5."
```

---

### Task 4: `phenotype_arrays` 加 `f_hi` / `f_lo` / `burst_w` / `burst_k` 4 列

**Goal:** 给 `src/des/phenotype_cache.py::StrainTable.phenotype_arrays(device)` 加 4 个新 per-strain 张量列 `f_hi: float32 / f_lo: float32 / burst_w: int64 / burst_k: int64`,让 kernel 可在 vectorized 路径上读到(`phe["f_hi"][sid_long]`)。`int64` 与既有 `period` / `repro_period` 同 dtype 风格(避免 PyTorch 在 `% burst_w` 上的 int 提升开销)。idx 0 (`EMPTY_ID` 哨兵)守默认 `0.0 / 0.0 / 1 / 1`(注意:`burst_w / burst_k` 哨兵是 1 不是 0,与 `period` 哨兵同纪律,守 `% 0` 的防御性 clamp 不被触发)。dirty-flag cache 同样的 rebuild 路径;**`f` 列不动**(它已经存 `phe.f = f_hi`,既有读者 `webapp.readouts` / 旧 kernel 路径都接 `f`)。

**Files:**
- Modify: `src/des/phenotype_cache.py:44-95` (`phenotype_arrays` 函数体, 三处: 累加列表初始化、per-sid 累加循环、result dict)
- Test: `tests/test_phase_windows.py` (append) + `tests/test_phenotype_cache.py` (append)

**Interfaces:**
- Consumes: `Phenotype.f_hi: float` / `Phenotype.f_lo: float` / `Phenotype.burst_w: int` / `Phenotype.burst_k: int` (Task 1 / Task 2)。
- Produces:
  - `phe["f_hi"]: torch.Tensor (float32, shape [n_strains])` —— `phe["f_hi"][sid] == phenotype(seq_sid).f_hi`。
  - `phe["f_lo"]: torch.Tensor (float32, shape [n_strains])` —— 同上。
  - `phe["burst_w"]: torch.Tensor (int64, shape [n_strains])` —— idx 0 = 1。
  - `phe["burst_k"]: torch.Tensor (int64, shape [n_strains])` —— idx 0 = 1。

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_phase_windows.py`:

```python
def test_phenotype_arrays_has_f_hi_column():
    """phe['f_hi'] 是 float32 张量, idx 0 (EMPTY) 是 0.0."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    assert "f_hi" in phe
    assert phe["f_hi"].dtype == torch.float32
    assert float(phe["f_hi"][0].item()) == 0.0


def test_phenotype_arrays_has_f_lo_column():
    """phe['f_lo'] 是 float32 张量, idx 0 (EMPTY) 是 0.0."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    assert "f_lo" in phe
    assert phe["f_lo"].dtype == torch.float32
    assert float(phe["f_lo"][0].item()) == 0.0


def test_phenotype_arrays_has_burst_w_column_default_one():
    """phe['burst_w'] 是 int64 张量, idx 0 (EMPTY) 是 1 (而非 0, 守 % 0)."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    assert "burst_w" in phe
    assert phe["burst_w"].dtype == torch.int64
    assert int(phe["burst_w"][0].item()) == 1


def test_phenotype_arrays_has_burst_k_column_default_one():
    """phe['burst_k'] 是 int64 张量, idx 0 (EMPTY) 是 1."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    assert "burst_k" in phe
    assert phe["burst_k"].dtype == torch.int64
    assert int(phe["burst_k"][0].item()) == 1


def test_phenotype_arrays_columns_match_python_phenotype():
    """对每个 strain, phe[col][sid] 必须 == 对应 Phenotype 字段值."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    for sid in range(1, len(eng.table) + 1):
        p = eng.table.phenotype_of(sid)
        assert abs(float(phe["f_hi"][sid].item()) - p.f_hi) < 1e-6
        assert abs(float(phe["f_lo"][sid].item()) - p.f_lo) < 1e-6
        assert int(phe["burst_w"][sid].item()) == p.burst_w
        assert int(phe["burst_k"][sid].item()) == p.burst_k


def test_phenotype_arrays_default_bb0_window_columns_degenerate():
    """默认 BB0 没有 FBURST / F_NOVA, 所有 strain 的 f_hi==f_lo==f, w=k=1."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    # 跑 5 tick 让突变产几个 strain 进 strain table
    eng.run(5, recorder=None, stop_on=())
    phe2 = eng.table.phenotype_arrays(torch.device("cpu"))
    # idx 1.. 全 strain: f_hi == f_lo (静态退化, 都是 stacked f)
    n = len(eng.table)
    assert n >= 1
    for sid in range(1, n + 1):
        f_hi = float(phe2["f_hi"][sid].item())
        f_lo = float(phe2["f_lo"][sid].item())
        assert abs(f_hi - f_lo) < 1e-6, f"sid={sid}: f_hi={f_hi} != f_lo={f_lo}"
        assert int(phe2["burst_w"][sid].item()) == 1
        assert int(phe2["burst_k"][sid].item()) == 1
```

追加到 `tests/test_phenotype_cache.py`:

```python
def test_phenotype_arrays_dirty_flag_invalidated_on_FBURST_mint():
    """mint 一个 FBURST strain 触发 dirty-flag → phenotype_arrays 重建,
    新 strain 在 burst_w 列上是 12, burst_k 是 2."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=4, W=4, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe_pre = eng.table.phenotype_arrays(torch.device("cpu"))
    n_pre = phe_pre["burst_w"].shape[0]
    # mint FBURST strain
    fburst_layout = ("FBURST",) + ("N0",) * 15
    sid = eng.table.get_or_mint(fburst_layout)
    # 取第二次 phe; n_strains 应增加, FBURST 行 burst_w=12 / burst_k=2
    phe_post = eng.table.phenotype_arrays(torch.device("cpu"))
    n_post = phe_post["burst_w"].shape[0]
    assert n_post > n_pre, "phenotype_arrays did not rebuild on new mint"
    assert int(phe_post["burst_w"][sid].item()) == 12
    assert int(phe_post["burst_k"][sid].item()) == 2
    assert abs(float(phe_post["f_hi"][sid].item()) - 0.55) < 1e-6
    assert abs(float(phe_post["f_lo"][sid].item()) - 0.05) < 1e-6
```

- [ ] **Step 2: 跑失败测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_phase_windows.py tests/test_phenotype_cache.py -v -k "phenotype_arrays or FBURST_mint"
```

Expected: 6 + 1 条全 FAIL with `assert 'f_hi' in phe` / `KeyError: 'f_hi'`(其他 3 列同理)。

- [ ] **Step 3: 加 4 列到 `phenotype_arrays`**

编辑 `src/des/phenotype_cache.py:44-95`。把 `phenotype_arrays` 函数体改为(在既有累加列表 + per-sid 循环 + result dict 三处都加 4 行):

```python
    def phenotype_arrays(self, device: torch.device) -> dict[str, torch.Tensor]:
        # I1: return cached tensors when not dirty and same device
        if not self._arrays_dirty and self._cached_device == device and self._cached_arrays is not None:
            return self._cached_arrays
        n = self._next
        # ... (既有 f / p_leave / z_raw / p_x / prey / feat / dir_bits / period 等列表初始化) ...
        f = [0.0] * n
        p_leave = [0.0] * n
        z_raw = [0.0] * n
        p_x = [0.0] * n
        prey = [0] * n
        feat = [0] * n
        dir_bits = [0] * n
        period = [1] * n
        repro_period = [1] * n
        anta_period = [1] * n
        # S4 加的 (若已落地):
        in_place_col = [0] * n
        rand_dir_col = [0] * n
        # S5: 4 个新列 — burst_w / burst_k 哨兵是 1 (守 % 0)
        f_hi_col = [0.0] * n
        f_lo_col = [0.0] * n
        burst_w_col = [1] * n
        burst_k_col = [1] * n
        for sid in range(1, n):
            phe = self._id_to_phe[sid]
            if phe is None:
                raise KeyError(f"strain id {sid} has no phenotype (internal error)")
            f[sid] = phe.f
            p_leave[sid] = phe.p_leave
            z_raw[sid] = phe.z_raw
            p_x[sid] = phe.p_x
            prey[sid] = phe.prey_mask
            feat[sid] = phe.feature_mask
            period[sid] = phe.period
            dir_bits[sid] = phe.dir_bits
            repro_period[sid] = phe.repro_period
            anta_period[sid] = phe.anta_period
            in_place_col[sid] = int(phe.in_place)       # S4
            rand_dir_col[sid] = int(phe.rand_dir)       # S4
            # S5:
            f_hi_col[sid] = phe.f_hi
            f_lo_col[sid] = phe.f_lo
            burst_w_col[sid] = phe.burst_w
            burst_k_col[sid] = phe.burst_k
        result = {
            "f": torch.tensor(f, dtype=torch.float32, device=device),
            "p_leave": torch.tensor(p_leave, dtype=torch.float32, device=device),
            "z_raw": torch.tensor(z_raw, dtype=torch.float32, device=device),
            "p_x": torch.tensor(p_x, dtype=torch.float32, device=device),
            "prey_mask": torch.tensor(prey, dtype=torch.int64, device=device),
            "feature_mask": torch.tensor(feat, dtype=torch.int64, device=device),
            "period": torch.tensor(period, dtype=torch.int64, device=device),
            "dir_bits": torch.tensor(dir_bits, dtype=torch.int64, device=device),
            "repro_period": torch.tensor(repro_period, dtype=torch.int64, device=device),
            "anta_period": torch.tensor(anta_period, dtype=torch.int64, device=device),
            "in_place": torch.tensor(in_place_col, dtype=torch.int8, device=device),    # S4
            "rand_dir": torch.tensor(rand_dir_col, dtype=torch.int8, device=device),    # S4
            # S5: 4 新列
            "f_hi": torch.tensor(f_hi_col, dtype=torch.float32, device=device),
            "f_lo": torch.tensor(f_lo_col, dtype=torch.float32, device=device),
            "burst_w": torch.tensor(burst_w_col, dtype=torch.int64, device=device),
            "burst_k": torch.tensor(burst_k_col, dtype=torch.int64, device=device),
        }
        # store cache and clear dirty flag
        self._cached_arrays = result
        self._cached_device = device
        self._arrays_dirty = False
        return result
```

(若 S4 尚未落地, 删除 `in_place_col` / `rand_dir_col` 与对应 dict key,不影响 S5 字段加成。)

- [ ] **Step 4: 跑测试, 确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_phase_windows.py tests/test_phenotype_cache.py -v
```

Expected: phase_windows 测试全 PASS;`test_phenotype_arrays_dirty_flag_invalidated_on_FBURST_mint` PASS;既有 `test_phenotype_cache.py` 全 PASS(I1 dirty-flag cache 路径不变,既有列 / dtype / 顺序不变,只是末尾新增 4 列)。

Backtrack:
- 若 `burst_w[0]` 不是 1 而是 0,检查哨兵列表初始化 `burst_w_col = [1] * n`(默认值 1, 不是 0)。
- 若 dtype 不匹配, PyTorch 在不同平台 `int` 默认 promote 不同 —— 显式 `dtype=torch.int64` 比省略关键。
- 若 cache 命中后没看到新列, root-cause: `_arrays_dirty` 在 Task 4 之前从 True 翻 False 后, cache 里冻结的 dict 没有新列 —— 但 `__init__` 设 `_arrays_dirty=True`, mint 也设 `_arrays_dirty=True`, cache 永远在第一次 call 时重建; 不应触发该路径。如果触发了, 把 `_cached_arrays = None` 强制 reset 一次。

- [ ] **Step 5: 跑全 suite, 确认无回归**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿。新 4 列仅暴露 dict 出口, kernel 没人读它们(Task 5 才接), 既有 strain 字节级不变。

- [ ] **Step 6: Commit**

```bash
git add src/des/phenotype_cache.py tests/test_phase_windows.py tests/test_phenotype_cache.py
git commit -m "feat(s5): phenotype_arrays f_hi / f_lo / burst_w / burst_k columns

Four per-strain tensor columns added to the bulk phenotype-array layout:
f_hi / f_lo float32, burst_w / burst_k int64. idx 0 (EMPTY sentinel) is
0.0 / 0.0 / 1 / 1 (the 1s on window cols guard against % 0 in the kernel).
Default BB0 max is f_hi==f_lo and burst_w==burst_k==1 — no v1/S4 strain
hits the windowed branch yet. f column stays unchanged (phe.f == phe.f_hi
alias), so webapp.readouts and any reader of phe['f'] are byte-identical.
Kernel consumer in Task 5."
```

---

### Task 5: `phase2_reproduce` 用 `where` 算 windowed live f

**Goal:** 把 `src/des/kernels/reproduction.py::phase2_reproduce` 里读 `f = phe["f"][sid_long]` 那一行(line 66)替换为 windowed live f 的计算 —— 读 4 个新列 `f_hi / f_lo / burst_w / burst_k`,跑 `on = ((T - birth_tick) % burst_w.clamp(min=1)) < burst_k; f = torch.where(on, f_hi, f_lo)`。下游 `fires` / `binom(scattered, f, generator)` / mut 切分 / `torch.roll` 全部不动 —— S5 唯一改的是 `f` 怎么算出来。`burst_w.clamp(min=1)` 是防御性 clamp(Phenotype 默认 1, registry 表 FBURST=12 / F_NOVA=20, 实际无 0 路径,但 spec §5 明确要求 explicit clamp 守 future-spec)。

**Files:**
- Modify: `src/des/kernels/reproduction.py:60-75` (`phase2_reproduce` 入口读 phe 那几行 + `f =` 那一行)
- Test: `tests/test_phase_windows.py` (append) + `tests/test_reproduction.py` (append)

**Interfaces:**
- Consumes: `phe["f_hi"]` / `phe["f_lo"]` / `phe["burst_w"]` / `phe["burst_k"]` (Task 4); 既有 `binom` / `fires_this_tick` / `ALL_DIRECTIONS` / `ArrivalBuffer`; `birth_tick` / `T` 形参(既有 phase2_reproduce 入口)。
- Produces: `phase2_reproduce` 接口签名不变 `(world, snap_sid, snap_count, snap_faction, phe, table, birth_tick, T, generator) -> tuple[ArrivalBuffer, Tensor]`。内部 `f` 张量从「strain-level 单一标量」变为「`[H, W, K]` 上 per-slot windowed live 值」,FBURST / F_NOVA strain 跑出来 `f` 会在 on / off 之间 toggle, 静态默认 strain `where` 出来 `f == f_hi == f_lo`(byte-identical)。

- [ ] **Step 1: 写失败测试 —— 默认局 byte-identical + FBURST on/off 窗口**

追加到 `tests/test_phase_windows.py`:

```python
def test_default_bb0_same_seed_byte_identical_post_s5():
    """S5 改了 kernel 一行, 但默认局所有 strain 静态退化 → byte-identical.
    跑 30 tick 两次同 seed, world.count / strain_id 完全相等."""
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


def test_fburst_offspring_window_on_vs_off(monkeypatch):
    """单 cell 装 FBURST strain, 跑 24 tick (= 2 windows), 比较 on-tick 与
    off-tick 的子代数量量级.

    FBURST burst_w=12 burst_k=2: tick (T - birth_tick) % 12 < 2 → on (f_hi=0.55);
    其他 → off (f_lo=0.05). 子代数量在 on-tick 应远大于 off-tick.

    构造细节: 用 birth_tick=0 的源格, 取 tick 0 (on) / tick 5 (off) 的本格 +
    邻 cell 总 count 之比作为 sanity 指标."""
    import torch
    from des.engine import Engine
    fburst_layout = ("FBURST",) + ("N0",) * 15
    layouts = (fburst_layout,) * 4
    # 用 8×8 grid 给 4 邻 roll 足够空间; 大 fill 让 binomial 期望更接近平均值
    eng = Engine(H=8, W=8, K=32, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=16, layouts=layouts)
    # 跑 24 tick 累计两期窗
    eng.run(24, recorder=None, stop_on=())
    # 由于 FBURST f_hi=0.55 (on 期) >> f_lo=0.05 (off 期), 跑完 24 tick 后
    # 系统应处于明显繁衍状态; 主要 sanity = 总 count 没崩 (> 0).
    total = int(eng.world.count.sum().item())
    assert total > 0, "FBURST strain should produce offspring across windows"


def test_fburst_seed_reproducible_across_runs():
    """FBURST strain 同 seed 跑 2 次 → bit-identical (kernel generator 是
    seed 的确定函数; windowed f 不引入新 RNG, binomial 仍走旧路径)."""
    import torch
    from des.engine import Engine
    fburst_layout = ("FBURST",) + ("N0",) * 15
    eng_a = Engine(H=4, W=4, K=16, seed=42, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=4,
                   layouts=(fburst_layout,) * 4)
    eng_b = Engine(H=4, W=4, K=16, seed=42, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=4,
                   layouts=(fburst_layout,) * 4)
    eng_a.run(15, recorder=None, stop_on=())
    eng_b.run(15, recorder=None, stop_on=())
    assert torch.equal(eng_a.world.count, eng_b.world.count)
    assert torch.equal(eng_a.world.strain_id, eng_b.world.strain_id)


def test_f_nova_seed_reproducible_across_runs():
    """F_NOVA strain 同 seed 跑 2 次 → bit-identical."""
    import torch
    from des.engine import Engine
    fnova_layout = ("F_NOVA",) + ("N0",) * 15
    eng_a = Engine(H=4, W=4, K=16, seed=42, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=4,
                   layouts=(fnova_layout,) * 4)
    eng_b = Engine(H=4, W=4, K=16, seed=42, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=4,
                   layouts=(fnova_layout,) * 4)
    eng_a.run(22, recorder=None, stop_on=())
    eng_b.run(22, recorder=None, stop_on=())
    assert torch.equal(eng_a.world.count, eng_b.world.count)
    assert torch.equal(eng_a.world.strain_id, eng_b.world.strain_id)


def test_phase_window_on_mask_formula_directly():
    """直接构造 `on = ((T - birth_tick) % burst_w) < burst_k` 验证算式.
    FBURST burst_w=12, burst_k=2, birth_tick=0:
      T=0 → 0%12=0 < 2 → on
      T=1 → 1%12=1 < 2 → on
      T=2 → 2%12=2 < 2 → off
      T=11 → 11%12=11 < 2 → off
      T=12 → 12%12=0 < 2 → on
      T=24 → 24%12=0 < 2 → on
    F_NOVA burst_w=20, burst_k=1:
      T=0 → on; T=1..19 → off; T=20 → on."""
    import torch
    burst_w = torch.tensor([12, 12, 12, 12, 12, 12, 20, 20, 20], dtype=torch.int64)
    burst_k = torch.tensor([2, 2, 2, 2, 2, 2, 1, 1, 1], dtype=torch.int64)
    birth_tick = torch.zeros(9, dtype=torch.int64)
    Ts = torch.tensor([0, 1, 2, 11, 12, 24, 0, 1, 20], dtype=torch.int64)
    on = ((Ts - birth_tick) % burst_w.clamp(min=1)) < burst_k
    expected = torch.tensor([True, True, False, False, True, True,
                             True, False, True])
    assert torch.equal(on, expected), f"on={on.tolist()}, expected={expected.tolist()}"
```

追加到 `tests/test_reproduction.py`:

```python
def test_fburst_byte_identical_when_seed_matches():
    """同 seed 同 layout FBURST 跑 30 tick byte-identical —— 守 RNG 用量不漂移."""
    import torch
    from des.engine import Engine
    fburst_layout = ("FBURST",) + ("N0",) * 15
    eng_a = Engine(H=4, W=4, K=8, seed=7, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2,
                   layouts=(fburst_layout,) * 4)
    eng_b = Engine(H=4, W=4, K=8, seed=7, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2,
                   layouts=(fburst_layout,) * 4)
    eng_a.run(30, recorder=None, stop_on=())
    eng_b.run(30, recorder=None, stop_on=())
    assert torch.equal(eng_a.world.count, eng_b.world.count)


def test_default_bb0_byte_identical_post_s5_kernel_change():
    """S5 改了 phase2_reproduce 一行 (f 的来源), 但 BB0 strain 全部静态退化
    → kernel `where` 出来恒等于旧 `f` 值, byte-identical."""
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
```

- [ ] **Step 2: 跑失败测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_phase_windows.py tests/test_reproduction.py -v -k "byte_identical_post_s5 or fburst or f_nova or phase_window_on_mask"
```

Expected:
- `test_phase_window_on_mask_formula_directly` PASS(pure torch 算式, 不需要 kernel 改)。
- `test_default_bb0_same_seed_byte_identical_post_s5` / `test_default_bb0_byte_identical_post_s5_kernel_change` PASS(kernel 没改但默认局任何 phe['f'] 读法都会重出同样数值; 因此这两条**会假性 PASS**, 实际 Task 5 完成后仍 PASS,守的是退步)。
- FBURST / F_NOVA byte-identical 测试 PASS(同 seed 跑两次,kernel 没改的当前状态:`f` 仍读 `phe["f"]`, 即 `f_hi`,FBURST `f_hi=0.55` 是常数, 不 toggle → byte-identical 但 dynamics 错。这条本步骤会 PASS,Task 5 完成后仍 PASS — 它守的是「同 seed 同结果」而非「windowed 行为正确」)。
- FBURST on/off 窗口数值测试: 本步骤大概率 PASS(因为 sanity 只断「total > 0」)。但这是**TDD-light** — kernel 改完后这条仍 PASS,且 dynamics 真实在 toggle; Task 5 真正改 kernel 后用 Step 5 的 manual probe 反查 toggle 行为。

(说明:S5 的 kernel 改是个**正向加法**,既有路径退化为它的 byte-equivalent 特例。我们没有一条「现在 FAIL、Task 5 完成后 PASS」的纯单元测试可以写 — 因为 FBURST 单 letter 走 `phe["f"]` 的旧路径返回 0.55, 改完后走 windowed `where` 在 on-tick 返回 0.55 / off-tick 返回 0.05, **同 seed 同结果会变**。因此这一步的 Step 2 验证目标改写为「跑全 Step 1 的测试集合, 应当看到所有 byte-identical 测试 PASS, FBURST 窗口数值测试 PASS, kernel 没改; 进入 Step 3 后, FBURST 同 seed 数值会变 — 把那条 byte-identical 测试 expected pre-vs-post 重新跑一次, 它必须从 PASS 变 PASS, 因为 seed 仍同, 只是 kernel 公式变了, 同 seed 仍同 result」。)

实际操作:Step 2 跑一遍记下 `eng_a.world.count.sum()` 的具体数值; Step 4 完成后再跑同测试, 数值会变(FBURST off-tick 子代变少),但两次同 seed 仍 byte-identical 自洽。

- [ ] **Step 3: 改 `phase2_reproduce` 的 `f` 计算**

编辑 `src/des/kernels/reproduction.py:60-75`。把 `phase2_reproduce` 入口的读 phe 那几行替换为:

```python
def phase2_reproduce(world, snap_sid, snap_count, snap_faction, phe, table,
                     birth_tick, T, generator):
    """Slot-level vectorized reproduction. (docstring 同前 ...)
    S5: f is computed per-slot from windowed (f_hi, f_lo, burst_w, burst_k)
    instead of the strain-level scalar phe['f']. Static-default strains have
    f_hi == f_lo and burst_w == burst_k == 1, so the where collapses to f_hi,
    which equals phe['f'] — byte-identical pre-S5 behavior."""
    from des.registry import ALL_DIRECTIONS
    H, W, K = snap_count.shape
    dev = world.device
    sid_long = snap_sid.long()

    # S5: windowed live f. (T - birth_tick) % burst_w < burst_k → on; on→f_hi, off→f_lo.
    # burst_w.clamp(min=1) is defensive (Phenotype default is 1; FBURST=12, F_NOVA=20);
    # the clamp mirrors fires_this_tick's period.clamp(min=1) discipline.
    f_hi    = phe["f_hi"][sid_long]                           # [H,W,K] float32
    f_lo    = phe["f_lo"][sid_long]                           # [H,W,K] float32
    burst_w = phe["burst_w"][sid_long].clamp(min=1)           # [H,W,K] int64
    burst_k = phe["burst_k"][sid_long]                        # [H,W,K] int64
    on = ((T - birth_tick) % burst_w) < burst_k               # [H,W,K] bool
    f  = torch.where(on, f_hi, f_lo)                          # [H,W,K] float32

    p_leave = phe["p_leave"][sid_long]
    p_x = phe["p_x"][sid_long]
    repro_period = phe["repro_period"][sid_long]
    dir_bits = phe["dir_bits"][sid_long]

    alive = snap_count > 0
    fires = fires_this_tick(birth_tick, repro_period, T) & alive & (f > 0)

    buf = ArrivalBuffer(dev)

    # ... (静态 meshgrid + rolled list + pass 1 / pass 2a / pass 2b / migration 全部不动) ...
```

注意:**`phe["f"][sid_long]` 那一行(原 line 66)被替换为以上 windowed 计算块**,不是删除。下游 `fires & alive & (f > 0)` 的 `f` 仍是 `[H,W,K]` float32 张量, 既有 `binom(a, f, generator)`、`(f > 0)` 判定、`rolled` 列表用法全部不变。

(若 S4 已落地,reproduction.py 已经在 `dir_bits` 之后加了 `in_place_mask = phe["in_place"][sid_long].bool()` 与 `rand_dir_mask = phe["rand_dir"][sid_long].bool()` 两行;S5 不动 S4 的这两个 mask, 也不动 S4 加的 in-place / rand-dir 两路分支。它们都在 windowed `f` 计算**之下**。)

`birth_tick` 与 `T` 都是 `phase2_reproduce` 的形参, 已存在; `birth_tick` 形状 `[H,W,K]` 与 `phe['f']` 同形, `T` 是 Python int 标量, PyTorch broadcast 接住。

- [ ] **Step 4: 跑测试, 确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_phase_windows.py tests/test_reproduction.py -v
```

Expected: 全 PASS。`test_default_bb0_*_byte_identical_post_s5` PASS 守的是「同 seed 跑 2 次结果相同」,kernel 改后仍同。FBURST / F_NOVA 同 seed 跑 2 次仍 byte-identical(generator 是 seed 的确定函数,windowed `where` 不引入 RNG)。`test_fburst_offspring_window_on_vs_off` 的 `total > 0` 仍 PASS。

Backtrack:
- 若 `RuntimeError: The size of tensor a (...) must match the size of tensor b (...)` 在 `(T - birth_tick) % burst_w`:检查 `birth_tick` 是否 `[H,W,K]` shape;若不是,在 `Engine.tick()` 里 `phase2_reproduce(..., birth_tick=self.world.birth_tick, T=self.T, ...)` 传的就是 `[H,W,K]` 张量, 应该匹配。
- 若 `RuntimeError: Expected all tensors to be on the same device`:`T` 是 Python int, PyTorch 会自动 promote; 若 `burst_w` 在 `cuda`、`birth_tick` 也在 `cuda`,结果在 `cuda`,**不应该**出 device 错。如果出, root-cause `phe[...]` 是不是 cuda(`StrainTable.phenotype_arrays(device)` 拿的是 `Engine.world.device`)。
- 若默认 BB0 byte-identical 失败:大概率 `phe["f"]` 仍是某处 hard-coded 读取(不应在 `phase2_reproduce` 内, 但可能在别处 — grep `phe\["f"\]` 或 `phe\.f`)。

- [ ] **Step 5: Smoke probe — 验 FBURST 真的 toggle**

跑一个 1×1 grid 上 FBURST 单 cell 的小 probe:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -c "
import torch
from des.engine import Engine
fburst = ('FBURST',) + ('N0',)*15
eng = Engine(H=1, W=3, K=16, seed=0, device=torch.device('cpu'),
             z_max=8.0, fill_per_cell=8, layouts=(fburst,)*4)
tots = []
for _ in range(24):
    eng.tick()
    tots.append(int(eng.world.count.sum().item()))
for T, t in enumerate(tots, start=1):
    on = (T % 12) < 2
    print(f'T={T:>2d}  on={on}  total={t}')
"
```

Expected output: 跑出来 24 行, 每行 `on=True/False`。on 段(T=12, 13, 24)的 total 增量应明显大于 off 段(T=2..11, 14..23)。具体数字会因 random seed、p_leave、K 满 等等漂移,但**on 段总 count 增量 / off 段同长度区间总 count 增量** > 5 是合理目标(`f_hi=0.55` vs `f_lo=0.05` 的繁衍速率比是 0.55/0.05 = 11)。

(这是 manual probe,**不**做 pytest 断言 —— probabilistic 平均下来过于 noisy,断言会 flake。仅供实施者眼测确认 toggle 起作用。)

- [ ] **Step 6: 跑全 suite, 确认无回归**

```
$env:PYTHONPATH='src'; D:/anaconda3/examples/basic/python.exe -m pytest tests/ -x -q
```

(若上一行的路径打错纠正:`D:/anaconda3/envs/basic/python.exe`。)

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿,含 S4 / S6 / S1 / S2 落地后的所有 test。

- [ ] **Step 7: Commit**

```bash
git add src/des/kernels/reproduction.py tests/test_phase_windows.py tests/test_reproduction.py
git commit -m "feat(s5): phase2_reproduce windowed live f via where(on, f_hi, f_lo)

Replace single phe['f'] read with the windowed live computation:
    on = ((T - birth_tick) % burst_w.clamp(min=1)) < burst_k
    f  = torch.where(on, f_hi, f_lo)
Defensive clamp mirrors fires_this_tick's period.clamp(min=1). Downstream
fires / binom / mut split / torch.roll unchanged. Default BB0 strains have
f_hi==f_lo and burst_w==burst_k==1, so the where collapses to f_hi == phe['f']
— byte-identical pre-S5 behavior. FBURST / F_NOVA now toggle their f between
on-window and off-window per their roster (burst_w, burst_k)."
```

---

### Task 6: Final regression sweep + smoke + push

**Goal:** 把整个 S5 deliverable (Tasks 1-5) 一起跑一遍, 确认全套测试绿、smoke run 不崩、性能档位 (~15.8ms/tick / 128² grid) 没明显漂移, 工作树干净, 推 origin。这是 sibling task 的同款收口动作 (S0 Task 6, S6 Task 9, S1 Task 7, S2 Task 7, S4 Task 8)。

**Files:**
- 不预期 source 改动。若 Step 1 暴露回归, 本 task 修 forward, commit message 引用 offending commit。
- Test: `tests/` (整套) + `scripts/run_batch.py --probe` smoke + (推荐) 一次默认 4-faction symmetric run smoke。

**Interfaces:**
- Consumes: Tasks 1-5 全部产物。
- Produces: 绿 `pytest tests/` + 干净 `git status` + push 到 origin。

- [ ] **Step 1: Full pytest sweep**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -q
```

Expected: 全绿。总数 = 285 engine + 146 web + S6 motif tests + S1 vis tests + S2 spectrum_shape tests + S4 direction_kinds tests + S5 phase_windows 新增 (`test_phase_windows.py` ~17 条) + 既有 `test_registry.py` / `test_phenotype_cache.py` / `test_reproduction.py` 的 append。精确数随时间漂移; **没有 `FAILED tests/...` 行**是验收标准 (`SKIPPED` 行允许 — 它们是 sibling task 显式标记的 fixture re-record 占位, 例如 S4 Task 6 留下的 837MB 基线 parquet skip)。

Backtrack: 若任何 FAIL, 先按测试 owner 文件 root-cause 到对应 Task; 用 `git log` 找出 offending commit, fix forward, 不要 reset。

- [ ] **Step 2: Smoke run probe (确认运行时性能没崩)**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --probe 30
```

Expected stdout 形如 `[probe 30 ticks] X.X ms/tick | peak Y.YY GB | strains->… | total …`。`X.X ms/tick` 应保持在 S0/S6/S1/S2/S4 完工时的同一档 (target ≈15.8 ms/tick on 128² grid; ≤ 20% drift acceptable)。exit 0; 不写 parquet (probe 路径 record=False)。

若 drift > 20%, 最常见原因:
- (a) `phase2_reproduce` 入口 4 行 phe 读取没共享 `sid_long` (该变量已在 `f_hi = phe["f_hi"][sid_long]` 之前的旧路径中 build 过), 检查是否复用而非重算。
- (b) `torch.where(on, f_hi, f_lo)` 与 `(T - birth_tick) % burst_w.clamp(min=1) < burst_k` 应该单次广播完成。如果出现意外 broadcast/`.contiguous()` copy, 检查 `f_hi` / `f_lo` dtype 都是 `float32` 与 `phe["f"]` 同 dtype 风格。
- (c) `burst_w.clamp(min=1)` 在每 tick 重算 [H,W,K] 一份, 若 `phenotype_arrays` 已守 idx 0 = 1 + Phenotype 默认 1 + registry 表 >0, 这个 clamp 实际是 no-op; 不应有性能影响。

- [ ] **Step 3: Byte-identical default-run smoke (推荐, 守 BB0 字节级不变)**

跑两次同 seed BB0 默认局:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --seeds 0 --T 30
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --seeds 0 --T 30
```

Expected: 两次产出的 parquet 在 `data/runs/` 下用 pyarrow 读起来 `(tick, cell, strain, count)` 行级一致 (用 `pyarrow.parquet.read_table` 读后比对 row group hash)。这一条守 "S5 改了 kernel 一行, 但默认局 strain 全部静态退化 → 同 seed 同结果"。

补充验证: 与 S4 收口时的 baseline parquet 比对 — 若 S4 Task 8 当时留了一份, 把现在跑出来的 parquet 与 S4 baseline 做 row-by-row diff:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -c "
import pyarrow.parquet as pq
a = pq.read_table(r'data/runs/<s4-baseline>.parquet').to_pydict()
b = pq.read_table(r'data/runs/<s5-fresh>.parquet').to_pydict()
print('cols:', sorted(a.keys()) == sorted(b.keys()))
for col in a:
    print(col, a[col] == b[col])
"
```

Expected: 全 `True`。若 False, root-cause: 大概率 `Phenotype.f` 不再等于 `phe.f_hi` 在某构造路径下漂移; 或 `phenotype_arrays` 的 `f` 列被无意改写。**不可能**是 windowed-f 公式本身 — 默认 BB0 所有 strain `burst_w==burst_k==1` 永远 on, where 出来恒等于 `f_hi == f`。

- [ ] **Step 4: FBURST / F_NOVA 手工 smoke (eye-test)**

跑一份 4-faction 全 FBURST / F_NOVA 的 8×8 grid 60-tick 小局, 观察 strain 动态:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_match.py --config _smoke_fburst.json --cpu --out data/runs
```

其中 `_smoke_fburst.json` 内容:

```json
{
  "players": [
    {"slots": {"0": "FBURST"}},
    {"slots": {"0": "F_NOVA"}},
    {"slots": {"0": "FBURST"}},
    {"slots": {"0": "F_NOVA"}}
  ],
  "grid": 8, "K": 16, "fill": 4, "T": 60, "seed": 0
}
```

Expected stdout 是一个 JSON line: `{"path": "data/runs/<ts>-match.parquet", "ticks": 60, "final": {"total": ..., "occupied_cells": ..., "distinct_strains": ..., "n2": ..., "d_max": ..., "faction_share": {"0": ..., ...}}}`. faction_share 任两 faction 不必相等 (FBURST f_hi=0.55 vs F_NOVA f_hi=0.85 → F_NOVA 倾向占优), 但**仍守红线**: 4 faction 全活 / 总 share = 1.0。

完事删除测试 config 与 smoke parquet:

```
rm _smoke_fburst.json
ls data/runs/ | grep "$(date +%Y%m%d)" | xargs -I{} rm data/runs/{}
```

- [ ] **Step 5: Inspect & clean stray data**

```
git status
```

Expected: 干净工作树。若 `data/runs/<ts>-*.parquet` smoke 残留 / `s5_*` 临时日志, 删掉 —— 不入 commit (它们不是 fixture, 也不是 S5 deliverable)。

- [ ] **Step 6: Final commit (only if Step 1 needed a fix-forward)**

若 Step 1 surface 了 regression 并修了:

```bash
git add <files-touched>
git commit -m "fix(s5): <description of the regression fixed>"
```

Otherwise this step is a no-op。

- [ ] **Step 7: Push to origin**

```bash
git push origin <current-branch>
```

Expected: push succeeds. branch ready for review / merge to `main`。

后续动作 (out of S5 plan scope, 用户单独决定时机):
1. 837MB 首批基线 parquet (S4 Task 6 标 skip 的 fixture-based byte-equal 测试) 重跑 — S5 也没碰它, 仍由 batch CLI 单独跑后回填。S4 + S5 完工后再 RE-RECORD 一并解 skip 更高效。
2. S3 (富猎物谓词, 填 S6 预留的 4 阈值 bit) 落地后回看本 plan: `vis_lowvis` / `thr_crest` / `thr_hotspot` / `thr_mirror` 都是 S6 / S1 / S3 owns 的 predicate-bit, 与 S5 windowed-f 不交互 — 本 plan 不需要为 S3 留任何 hook。
3. S7 (多位突变) 落地后回看本 plan: 多位突变改 `_mutation_outcomes` 的 slot 选择, 与 windowed-f 完全正交; 不需要 hook。
4. S8 (A 池) 落地后会复用 S5 的 windowed-f machinery (例如 A 池可能 mint 一些 phase-modulated 杀手), `_F` 7-tuple + `phenotype()` dominant-F 解析 + kernel `where` 一切就绪, S8 仅需追加 A pool 行。

---

## Self-Review

**1. Spec coverage:**

- §1 (FBURST + F_NOVA, 两个新 phase-window F 基元): Task 3 — `ALPHABET` / `GRAN` / `_F` 三表各加 2 行 verbatim 抄 spec §1。FBURST: `f=0.55, f_lo=0.05, w=12, k=2`; F_NOVA: `f=0.85, f_lo=0.05, w=20, k=1`。**`P_burst_lite` 明确不在此 task** —— spec §1 已声明它是 P 基元, S2 已铸, S5 不重复。
- §2 (Red lines — 窗口参数是 registry 全局值 / phenotype 只存参数 / kernel 算 live f / 静态默认字节级不变 / red line HOW-1): Task 2 (`_F` 行 7-tuple, `Phenotype` 4 字段) + Task 5 (kernel `where` 算 live f, `birth_tick` + `T` 都是既有 kernel 时钟 input, 不读 world-state)。回归锁 = Task 6 Step 1 全套 pytest 绿 + Step 3 默认局 BB0 byte-identical smoke。
- §3 (Architecture HOW-1, ponytail-minimal): Task 1 (Phenotype 加 4 字段) + Task 2 (`_F` 升 7-tuple + dominant-F 解析) + Task 4 (`phenotype_arrays` 加 4 列) + Task 5 (kernel `f = where(on, f_hi, f_lo)`)。spec §3 明确「保持 `Phenotype.f` 作为 `f_hi` 的别名」—— Task 2 Step 5 + Task 4 实现:`phenotype_cache.py:69` 的 `f[sid] = phe.f` 不动 (`phe.f` 就是 `f_hi`); webapp.readouts 不动。
- §4 (data flow): mint(seq) ─► phenotype() 设 `f_hi/f_lo/burst_w/burst_k` (Task 2) ─► phenotype_arrays() 张量化 (Task 4) ─► phase2_reproduce() `on = (T-birth)%w < k; f = where(on, f_hi, f_lo)` (Task 5) ─► binom offspring (既有路径不变)。
- §5 (Error handling): `burst_w.clamp(min=1)` 守 `% 0` —— Task 5 Step 3 显式写入 kernel; spec §5 明确「`burst_w` 是 NEW 列, 不在既有 `period.clamp(min=1)` 守护范围」。多 F 行 dominant-F 近似 —— Task 2 dominant-F 解析 (highest-f, 平手取首现; `f > dom_f_value` 严格大于)。spec §5 "stacked f_lo via 1-(1-dom.f_lo)·Π_{i≠dom}(1-f_i)" —— Task 2 Step 4 verbatim 实现。
- §6 (Testing — regression / 新 / relabel-invariance): Task 1 (4 字段 default 存在 + frozen) + Task 2 (静态默认 byte-identical, dominant-F 解析, multi-F stacking, `_F` 7-tuple 形状) + Task 3 (FBURST / F_NOVA 行精确值, 单 letter phenotype, multi-F dominant-FBURST/F_NOVA) + Task 4 (phenotype_arrays 4 列 + dirty-flag + 默认 BB0 退化) + Task 5 (kernel byte-identical 默认局, FBURST/F_NOVA on/off 窗口 sanity, 同 seed 跑两次 byte-identical, 直接验 `on` 算式)。relabel-invariance 隐含 (4 个窗口字段全部从 `_F[letter]` 直读, 与 `_Z`/`_P` 量级无关)。
- §7 (Out of scope): FBURST / F_NOVA 的 dirs (S4 owns 但 dirs=4-nbr 走字面 4-tuple) 与 p_leave (既存机制); P_burst_lite spectrum (S2); full per-letter windowed-f stacking; A 池 (S8); 多位突变 (S7) —— 本 plan 注释多次声明, 不引入任何相关 source。
- **Red lines (§2):** ① `(f, f_lo, burst_w, burst_k)` 全 registry 全局 (Task 2 / 3), 没有 per-species 量级; ② `Phenotype` 只存参数 (Task 1) 不读 world-state; ③ kernel 算 live f (Task 5), `(T-birth)%burst_w` 与 `fires_this_tick` 同款时钟; ④ 静态默认字节级不变 (Task 2 / Task 5 byte-identical 测试守门 + Task 6 smoke); ⑤ `burst_w=0` 显式 clamp (Task 5 kernel)。

**2. Placeholder scan:**

无 `TBD` / `TODO` / "implement later" / "fill in details" / "similar to Task N" / "write tests for the above" 等 plan-failure 字串。所有 code step 给出真实代码; 所有 command step 给出真实命令 + 预期输出; 所有 backtrack 条件给出具体 root-cause / fix。Task 5 Step 5 的 manual probe 是显式 eye-test (不是 pytest 断言, 因为 probabilistic 测 flake), 已在描述里说明 "**不**做 pytest 断言"。Task 6 Step 3 的 "<s4-baseline>.parquet" / "<s5-fresh>.parquet" 是动态文件名占位 (实施者 ls 后填), Step 描述里讲清 "用 `data/runs/` 下最新 timestamped parquet" 的判定规则, 不是 plan-level 空白。

**3. Type consistency:**

- `Phenotype.f_hi: float = 0.0` / `f_lo: float = 0.0` / `burst_w: int = 1` / `burst_k: int = 1` —— Task 1 定义, Task 2 写, Task 4 / Task 5 / Task 6 consume。同名同 dtype 全程贯穿。
- `_F[letter] = (f, dirs, p_leave, period, f_lo, burst_w, burst_k)` 7-tuple —— Task 2 升级既有 2 行 (+ S4 5 行), Task 3 追加 2 新行。元素顺序在 plan 全文锁死: index 0=f, 1=dirs, 2=p_leave, 3=period, 4=f_lo, 5=burst_w, 6=burst_k。
- `phe["f_hi"] / phe["f_lo"] / phe["burst_w"] / phe["burst_k"]` —— Task 4 定义, Task 5 consume。dtype 全程: `float32 / float32 / int64 / int64`。
- `phase2_reproduce(world, snap_sid, snap_count, snap_faction, phe, table, birth_tick, T, generator) -> tuple[ArrivalBuffer, Tensor]` —— signature 与 pre-S5 一致, Task 5 内部加 windowed f 计算 (在 `f = phe["f"][sid_long]` 原位)。
- `on = ((T - birth_tick) % burst_w.clamp(min=1)) < burst_k` —— `on` 是 [H,W,K] bool 张量, Task 5 Step 3 + Step 1 直接验算 (Step 1 的 `test_phase_window_on_mask_formula_directly` 用 pure torch 算同一公式)。
- `Phenotype.f == Phenotype.f_hi` (alias 契约) —— Task 2 Step 4 `Phenotype(..., f=f, ..., f_hi=f_hi, ...)` 且 `f_hi = f` 同行赋值; Task 2 Step 1 `test_phenotype_f_field_is_alias_of_f_hi` 守门; Task 4 `phenotype_arrays` 的 `f` 列读 `phe.f`, 既等于 `phe.f_hi`。

spec §1 表里 `(f, f_lo, burst_w, burst_k)` 数值与本 plan 对照 verbatim: FBURST=(0.55, 0.05, 12, 2) / F_NOVA=(0.85, 0.05, 20, 1) —— Task 3 Step 3 (_F 块) 与 spec §1 字节级一致。

无 method / property 名称漂移: `f_hi` / `f_lo` / `burst_w` / `burst_k` (lowercase + underscore) 全 plan 同名; 不出现 `f_high` / `fHi` / `burstW` 等变体; `dom_f_letter` / `dom_f_value` / `dom_burst_w` 等局部变量仅在 Task 2 `phenotype()` 函数内, 不暴露到 Phenotype / phe 出口。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-24-s5-phase-windows.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. Use `superpowers:subagent-driven-development`.
2. **Inline Execution** — execute tasks in this session with batch checkpoints. Use `superpowers:executing-plans`.

Which approach?
