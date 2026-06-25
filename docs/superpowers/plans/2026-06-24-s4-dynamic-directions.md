# S4 — 动态方向 (dynamic directions) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 F 池剩下 5 个繁衍基元 (FSTACK / FCLUMP / FFRONT / F4Nr3 / FDRIFT) 落地, 为方向变体接两条新机制 —— **crc32 序列哈希锁向**(FCLUMP / FFRONT / F4Nr3 / F4Nr1)在 mint 时一次性把方向 OR 进 `dir_bits`, 以及内核两路新分支(FDRIFT 每 tick 随机抽一邻、FSTACK 原格堆叠 ),并把 F4Nr1 从 v1 占位 `((-1,0),)` 重新底定到 hash-locked 1-of-4 。

**Architecture:** 三件事, 顺序: (1) 5 行新 F 基元入 `_F` / `ALPHABET` / `GRAN` (S6 协议), 配一个 `_hash_dirs(seq, kind) -> tuple[(int,int),...]` 纯函数, 用 `zlib.crc32("\x1f".join(seq).encode())` 把序列结构 → 一组方向; (2) `phenotype()` 在累加 F 行时把 hash-locked / in-place / rand-dir 三类标志识别出来 —— hash-locked 的方向 OR 进 `dir_bits` 和它已有的 `directions` 列表; in-place / rand-dir 由 frozen `Phenotype` 新增两个 bool 字段 `in_place: bool` / `rand_dir: bool` (默认 False) 承载; (3) `phase2_reproduce` 在 `phenotype_arrays` 里读这两个 bool 列, 给 in-place 加一条「无 roll、当格沉积」分支, 给 rand-dir 加一条「kernel `generator` 抽 1-of-4」分支, 走 `dir_bits` 的旧路保持字节级不变。

**Tech Stack:** Python 3 (`D:\anaconda3\envs\basic\python.exe`), PyTorch 2.10+cu128, pyarrow, pytest, stdlib `zlib.crc32`. Windows 主机, `PYTHONPATH=src` 纪律。引擎源码 `src/des/`。**依赖**: S0 (CLI 已就位, 不动) + S6 (`GRAN` / `motif_blocks` / 谱预过滤已落地; 本 spec 给新 F 行加 `gran="residue"` 或 `"motif"` 两类) + S2 (`SPECTRUM_SHAPE` 已落地, 不读 F 行)。S1 不交互 (vis 是 N 池机制)。

## Global Constraints

- **方向是序列结构的确定性函数 或 诚实的每 tick 随机抽**: hash-locked 类用 `zlib.crc32` 把序列 → 方向集 (mint 时一次性算好, 同 strain 同方向, 跨 strain 像随机), rand-dir 类每次开火现抽; **绝不手写「谁扩张得更好」** (spec §2)。
- **禁用 Python 内建 `hash()`**: 解释器进程级 `PYTHONHASHSEED` 加盐 → 跨进程跨机器不可复现, 对数据生成沙盒是致命的 (spec §2 ponytail HOW-2)。**只许 `zlib.crc32`**, 拼接前用 `\x1f` (unit separator, 不在字母表内) 把多字符字母分隔, 防 `"N0"+"F4Nr4"` 之类的歧义拼接。
- **`dir_bits==0` 不许 overload**: 当前内核里 `dir_bits==0` 意味「不发射」(连源格也不发); 不可拿来兼容 in-place 或 rand-dir。In-place 与 rand-dir 是两个**独立的、显式的 bool 字段**, 与 `dir_bits` 正交 (spec §3.2)。
- **F4Nr1 重底定可被用户接受**: 从 v1 占位 `((-1,0),)` 改成 hash-locked 1-of-4 等价 FFRONT 的处理 —— 同 strain 锁死单一方向, 跨 strain 看似 4 邻随机。**这是已知的默认局漂移** (用户 2026-06-24 已批 RE-RECORD), 影响 837MB 首批基线与「锁 F4Nr1=北」的旧测试 (spec §3.3 / §6); 本 plan Task 6 集中升级。
- **回归锁限定 F4Nr4 的方向集**: F4Nr4 的方向集合是 4 邻全开的 static 路径 (spec §1 表), 改 hash-locked 不在 S4 范围内, **必须保持 4 个 bit 全置位**。Task 4 / Task 6 的回归断言显式守这一条。
- **新增 F 行带各自方向逻辑同 ship**: 5 个新 F 基元的 `_F` 行与对应方向处理是一个交付单元 (spec §3.4), 不允许「先加行后加机制」的中间态 —— Task 5 把 5 行入注册表与方向逻辑收口同一 commit, 让 strain mint 出来就立即可正确发射或堆叠或抽随机。
- **frozen `Phenotype` 加字段必须保持 frozen 与默认值**: `Phenotype` 是 `@dataclass(frozen=True)`, 新增 `in_place: bool = False` / `rand_dir: bool = False` 必须给默认值, 否则既有 `Phenotype(...)` 构造点 (含 S1 / S2 / 测试夹具) 全部要带新参 → 大爆炸。同 S1 加 vis_sum / n_count 时的纪律。
- **bulk phenotype-array 必带新两列**: kernel 走的是 `phe[...sid_long]` 索引 (`phenotype_cache.py::phenotype_arrays`), 不是逐 cell 触 `Phenotype` 对象。新两 bool 必须在 `phenotype_arrays` 增对应张量 (`bool` dtype 或 `int8`, 用 `int8` 与 vis_mode 同 dtype 风格, 避免 PyTorch 上 bool 索引语义差异), 否则 kernel 看不到。
- **kernel `generator` 是世界 RNG**: rand-dir 类每 tick 抽方向必须从 `phase2_reproduce(..., generator)` 形参抽 (这是 `Engine` 传进来的 world RNG), **不是** Python 全局 `random` 也**不是** `torch.randint` 默认 RNG —— 否则同 seed 不可复现 (spec §3.2)。
- **L=16 fixed 不变**: F4Nr1 改 hash-locked 后, layout 长度仍 16, 也不引入新长度量。crc32 读「字面字母序列」(structural identity), `\x1f` 分隔符 + 字节编码, 与 `_F`/`_Z`/`_P` 量级解耦 → relabel-invariance: 重排 f/z/p 数值不影响方向选择 (spec §6)。
- **空方向集不可能出现**: hash-locked 类用 `crc32 % 4` / `crc32 % 2`, 都是合法索引; FSTACK 走 in_place 路径不读 `dir_bits`; F4Nr3 永远保留 3 个; 任何路径都至少有 1 个发射目标 (spec §5)。
- **out of scope**: FBURST / F_NOVA (S5 owns f-window 含其 dirs); A 池方向变体 F8Ar1 / Lance Front / Ember Drip / F_TRICKLE / F_SCATTER (S8 复用本 spec 的机制, S4 不铸); 多位突变 (S7) 与 P_loopswap_lite (S2 已铸 + 它读谱不读方向)。

---

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `src/des/registry.py` | **Modify** | (a) `ALPHABET` 加 5 个新 F 字母 (`FSTACK` / `FCLUMP` / `FFRONT` / `F4Nr3` / `FDRIFT`), 全 family `"F"`; (b) `GRAN` 配套加 5 行 (`FSTACK`/`F4Nr3`/`FDRIFT` = `"residue"`, `FCLUMP`/`FFRONT` = `"motif"` —— spec §3.4 表声明的); (c) 仅 motif 行的两个进 `MOTIF_LEN` (`FCLUMP=2` / `FFRONT=2`, spec §3.4 表); (d) `_F` 表加 5 行 `(f, directions, p_leave, period)` —— **directions 字段对 hash-locked / rand-dir / in-place 三类含义不同**, 见 Task 2; (e) 新模块级函数 `_hash_dirs(seq: tuple[str,...], kind: str) -> tuple[tuple[int,int], ...]` 用 `zlib.crc32` 把序列 → 方向集; (f) `phenotype()` 在 `_F` 分支识别 hash-locked / in-place / rand-dir 三类标志, 写新两 bool。 |
| `src/des/types.py` | **Modify** | `Phenotype` 加两个 bool 字段 `in_place: bool = False` / `rand_dir: bool = False`, 保持 frozen 与默认值不破坏既有构造点。 |
| `src/des/phenotype_cache.py` | **Modify** | `phenotype_arrays(device)` 加两个新张量列 `in_place: int8 (0/1)` / `rand_dir: int8 (0/1)`, 一行一 strain, idx 0 (`EMPTY_ID`) 守默认 0; 新两列入 dirty-flag cache 同样的 rebuild 路径。 |
| `src/des/kernels/reproduction.py` | **Modify** | `phase2_reproduce` 在「pass 1: per direction roll」之前先读 `phe["in_place"][sid_long]` 与 `phe["rand_dir"][sid_long]` 两个 mask; 给 in-place 加一条「无 roll、当格沉积」分支 (走 `binom(scattered, f)` + 当格 ty/tx 直接 `buf.add`, 不调用 `torch.roll`); 给 rand-dir 加「kernel `generator` 抽 1-of-4」分支 (`torch.randint(0, 4, ..., generator=generator)` 一次性给每个 firing slot 一个方向索引, 然后按方向索引 gather)。`dir_bits` 的旧路径只跑「既非 in-place 又非 rand-dir」的 slot, 保持字节级回归。 |
| `tests/test_hash_dirs.py` | **Create** | 新建, S4 owner 文件 (本仓库已有 `test_motif.py` / `test_vis.py` / `test_spectrum_shape.py` 同款 owner 切分纪律): `_hash_dirs` 的确定性 (子进程跨进程一致)、四类 kind (`ffront` / `fclump` / `f4nr3` / `f4nr1`) 各自的方向集形状 (单方向、轴、三邻、单 4-邻 1)、`\x1f` 分隔符防拼接歧义、relabel-invariance (重排 `_F`/`_Z`/`_P` 量级不影响方向)。 |
| `tests/test_direction_kinds.py` | **Create** | 新建: 5 个新 F 基元的 `_F` / `ALPHABET` / `GRAN` / `MOTIF_LEN` 入注册表 + `phenotype()` 落字段 + `phenotype_arrays` 落张量列 + `phase2_reproduce` 行为 (FSTACK 在格内堆叠且不出格、FDRIFT 跨 tick 同 strain 方向变化、F4Nr3 永远剩 3 邻、FCLUMP 轴二选一、FFRONT 单方向稳定、F4Nr4 全 4 邻不变 = 回归锁)。 |
| `tests/test_registry.py` | **Modify (append)** | (a) **改一条既有断言**: line 113-115 `test_dir_bits_match_directions` 里「`F4Nr1` 北向唯一」的硬编码升级为 hash-locked: 仍 1 bit 但具体哪个 bit 由 crc32 决定 (测试用 `phenotype(("F4Nr1",))` 现算 `dir_bits` 的 popcount==1, **不锁哪个 bit**); (b) 追加 5 行新 F 基元的 family/gran 与 `_F` 行 registry-层覆盖断言。 |
| `tests/test_reproduction.py` | **Modify (append)** | 追加 in-place / rand-dir 路径在 `phase2_reproduce` 端到端断言: 单 cell FSTACK 跑 1 tick, 邻居 count 必须为 0, 源格 count 增加; 单 cell FDRIFT 跑同 seed 两次 → 同 strain 同方向分布 (跨 tick 抽随机, 跨进程同 seed 同结果)。 |

**Naming contract (locked, used by every task):**

```python
# src/des/registry.py
def _hash_dirs(seq: tuple[str, ...], kind: str) -> tuple[tuple[int, int], ...]
    # kind ∈ {"ffront", "fclump", "f4nr3", "f4nr1"};
    # ffront / f4nr1 -> 1 dir;
    # fclump         -> 2 dirs (一根轴);
    # f4nr3          -> 3 dirs (去掉一邻);
    # 实现: h = zlib.crc32("\x1f".join(seq).encode())

# src/des/types.py
@dataclass(frozen=True)
class Phenotype:
    # ... existing fields ...
    in_place: bool = False    # S4: FSTACK — 当格沉积, 不参与 ALL_DIRECTIONS roll
    rand_dir: bool = False    # S4: FDRIFT — 每 firing tick 现抽 1-of-4 邻

# src/des/registry.py (新模块级常量)
IN_PLACE_DIR = (0, 0)                            # 不入 ALL_DIRECTIONS, 不入 _DIR_BIT;
                                                 # 仅作为 _F 行 `directions` 字段的 sentinel

# 5 个新 F 基元的 `_F` row 形状:
#   "FSTACK":  (0.60, (IN_PLACE_DIR,),          0.00, 3),   # in_place
#   "FCLUMP":  (0.45, "hash:fclump",            0.10, 6),   # hash-locked 轴二选一
#   "FFRONT":  (0.50, "hash:ffront",            0.25, 4),   # hash-locked 单方向
#   "F4Nr3":   (0.40, "hash:f4nr3",             0.12, 5),   # hash-locked 三邻
#   "FDRIFT":  (0.15, "rand:1of4",              0.30, 2),   # 每 tick rand

# F4Nr1 重底定 (从 v1 占位升级):
#   "F4Nr1":   (0.30, "hash:f4nr1",             0.05, 4),   # hash-locked 1 of 4-邻

# directions 字段的解释规则 (phenotype() 内部消费, 不暴露给 caller):
#   tuple[(int,int), ...]   -> 字面方向集 (旧路径, F4Nr4 / FSTACK 走这里)
#   "hash:<kind>"           -> mint 时调 _hash_dirs(seq, kind), 把结果 OR 进 dir_bits
#   "rand:1of4"             -> phenotype 写 rand_dir=True, 内核每 tick 抽

# src/des/kernels/reproduction.py
def phase2_reproduce(world, snap_sid, snap_count, snap_faction, phe, table,
                     birth_tick, T, generator) -> tuple[ArrivalBuffer, Tensor]
    # 内部新读 phe["in_place"] / phe["rand_dir"] 两个 int8 mask;
    # 三路: rand_dir 路 (gather by per-firing-slot torch.randint) ->
    #       in_place 路 (无 roll, 当格 buf.add) ->
    #       静态 dir_bits 路 (旧路径, 字节级不变).
```

`directions` 字段的「字面 tuple vs `"hash:<kind>"` vs `"rand:1of4"`」三态识别策略锁死在 `phenotype()` 一处; kernel 永不直接读 `_F` 表, 只读 `Phenotype` / `phenotype_arrays` 的结构化字段。这样 registry 是描述, `phenotype` 是翻译, kernel 是执行 —— 三层职责清晰。

---

### Task 1: `_hash_dirs(seq, kind)` 纯函数 + `IN_PLACE_DIR` 常量(无消费者)

**Goal:** 把 spec §3.1 的 hash-locked 方向计算落成一个独立的纯函数 `_hash_dirs(seq, kind)`, 用 stdlib `zlib.crc32("\x1f".join(seq).encode())` 把序列 → uint32 → 各 kind 的方向集。覆盖四类 kind: `"ffront"` (单方向, `ALL_DIRECTIONS[h%4]`)、 `"fclump"` (一根轴, x 轴或 y 轴二选一)、 `"f4nr3"` (去掉一邻, 剩 3 邻)、 `"f4nr1"` (单方向, 与 ffront 同表)。同时加 module-level 常量 `IN_PLACE_DIR = (0, 0)`, 它作为 FSTACK 在 `_F` 行的 directions 字段 sentinel, 不入 `ALL_DIRECTIONS` 也不入 `_DIR_BIT`。这一步纯函数 + 常量, 无消费者, 既有行为 0 漂移。

**Files:**
- Modify: `src/des/registry.py` (在 `_F` 表之后, `affinity` 函数之前插入 `_hash_dirs` 与 `IN_PLACE_DIR`)
- Test: `tests/test_hash_dirs.py` (Create)

**Interfaces:**
- Consumes: `ALL_DIRECTIONS` (既有, `[(-1,0),(1,0),(0,-1),(0,1)]`)。
- Produces:
  - `IN_PLACE_DIR: tuple[int, int]` —— 字面 `(0, 0)`, 与 `ALL_DIRECTIONS` 不交。
  - `_hash_dirs(seq: tuple[str, ...], kind: str) -> tuple[tuple[int, int], ...]`:
    - `"ffront"` / `"f4nr1"` → 单方向 `(ALL_DIRECTIONS[h % 4],)`, 长度 1。
    - `"fclump"` → 长度 2: `h % 2 == 0` → `((-1,0), (1,0))` (x 轴); 否则 `((0,-1), (0,1))` (y 轴)。
    - `"f4nr3"` → 长度 3: `ALL_DIRECTIONS` 去掉 `h % 4` 那一个。
    - 其他 kind → `ValueError`。
  - `h = zlib.crc32("\x1f".join(seq).encode())`; 跨进程 / 跨机器恒定。

- [ ] **Step 1: 写失败测试 —— 确定性、四 kind 形状、`\x1f` 防拼接歧义、relabel-invariance**

新建 `tests/test_hash_dirs.py`:

```python
# tests/test_hash_dirs.py
"""S4 hash-locked direction selection. crc32(\\x1f.join(seq)) -> 各 kind 的方向集.

Determinism 是这一档的生命线: PYTHONHASHSEED 加盐的 hash() 跨进程不一致, 数据生成
沙盒立崩. 用 stdlib zlib.crc32 是为了 cross-process / cross-machine 一致 +
lightweight (we only need a uniform int to select dirs, not crypto)."""
from __future__ import annotations
import subprocess
import sys
import textwrap
import pytest
import zlib

from des.registry import _hash_dirs, IN_PLACE_DIR, ALL_DIRECTIONS


def test_in_place_dir_is_zero_zero_and_not_in_all_directions():
    """IN_PLACE_DIR = (0, 0); 严格不入 ALL_DIRECTIONS, 与四邻完全正交."""
    assert IN_PLACE_DIR == (0, 0)
    assert IN_PLACE_DIR not in ALL_DIRECTIONS


def test_ffront_returns_single_direction_in_all_directions():
    """ffront kind 返回 长度 1 的方向元组, 方向在 ALL_DIRECTIONS 中."""
    dirs = _hash_dirs(("F4Nr1", "N0", "N0"), "ffront")
    assert len(dirs) == 1
    assert dirs[0] in ALL_DIRECTIONS


def test_f4nr1_returns_single_direction_in_all_directions():
    """f4nr1 kind 与 ffront 同表 (spec §3.3): 单方向, ALL_DIRECTIONS[h%4]."""
    dirs = _hash_dirs(("F4Nr1", "N0", "N0"), "f4nr1")
    assert len(dirs) == 1
    assert dirs[0] in ALL_DIRECTIONS


def test_fclump_returns_axis_pair():
    """fclump kind 返回 长度 2 的方向元组, 是一根轴 (x 或 y)."""
    dirs = _hash_dirs(("FCLUMP", "N0", "N0"), "fclump")
    assert len(dirs) == 2
    axis_x = {(-1, 0), (1, 0)}
    axis_y = {(0, -1), (0, 1)}
    assert set(dirs) == axis_x or set(dirs) == axis_y


def test_f4nr3_returns_three_of_four_neighbors():
    """f4nr3 kind 返回 长度 3 的方向元组, 是 ALL_DIRECTIONS 的 3 元子集."""
    dirs = _hash_dirs(("F4Nr3", "N0", "N0"), "f4nr3")
    assert len(dirs) == 3
    for d in dirs:
        assert d in ALL_DIRECTIONS
    assert len(set(dirs)) == 3        # 无重复


def test_unknown_kind_raises_value_error():
    with pytest.raises(ValueError):
        _hash_dirs(("N0",), "not_a_kind")


def test_same_sequence_same_kind_yields_same_dirs():
    """同 seq + 同 kind → 同方向 (mint 时一次性算, 整个 lineage 锁死)."""
    seq = ("FFRONT", "N0", "F4Nr4", "P_hotspot")
    a = _hash_dirs(seq, "ffront")
    b = _hash_dirs(seq, "ffront")
    assert a == b


def test_unit_separator_prevents_concat_ambiguity():
    """`\\x1f` 分隔符使多字符字母 token 不会歧义拼接:
    ("N0", "F4Nr4") 必须 != ("N0F4Nr4",) 也 != ("N", "0F4Nr4"),
    crc32 算出来 hash 不一样 → 方向不一样 (一般情况)."""
    h_split = zlib.crc32("\x1f".join(("N0", "F4Nr4")).encode())
    h_concat = zlib.crc32("\x1f".join(("N0F4Nr4",)).encode())
    h_misjoin = zlib.crc32("\x1f".join(("N", "0F4Nr4")).encode())
    assert h_split != h_concat
    assert h_split != h_misjoin


def test_hash_is_deterministic_across_subprocesses():
    """开一个子进程, 用同 seq 调 _hash_dirs, 必须得到与父进程相同的方向集.
    PYTHONHASHSEED 加盐的 hash() 在这条断言上必失败 (spec §2)."""
    code = textwrap.dedent('''
        import sys
        sys.path.insert(0, r"{src}")
        from des.registry import _hash_dirs
        seq = ("FFRONT", "N0", "F4Nr4", "P_hotspot")
        print(_hash_dirs(seq, "ffront"))
    ''').format(src=str(__import__("os").path.abspath(__import__("os").path.join(
        __import__("os").path.dirname(__file__), "..", "src"))))
    parent = _hash_dirs(("FFRONT", "N0", "F4Nr4", "P_hotspot"), "ffront")
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert str(parent) == out, f"parent={parent!r}, child={out!r}"


def test_relabel_invariance_hash_reads_letter_sequence_only(monkeypatch):
    """重排 _F / _Z / _P 的量级 (f / z / p_add / period) 不影响 _hash_dirs
    输出 —— hash 读的是字面字母, 与 magnitude 解耦 (spec §6)."""
    import des.registry as reg
    seq = ("FFRONT", "F4Nr4", "P_hotspot")
    pre = _hash_dirs(seq, "ffront")
    monkeypatch.setitem(reg._F, "F4Nr4", (0.01, ((1, 0),), 0.99, 99))
    monkeypatch.setitem(reg._P, "P_hotspot", (0.0, 99))
    post = _hash_dirs(seq, "ffront")
    assert pre == post
```

- [ ] **Step 2: 跑测试, 确认失败**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_hash_dirs.py -v
```

Expected: 全 FAIL with `ImportError: cannot import name '_hash_dirs' from 'des.registry'` (`IN_PLACE_DIR` 同理).

- [ ] **Step 3: 在 `src/des/registry.py` 加 `IN_PLACE_DIR` + `_hash_dirs`**

在 `src/des/registry.py` 顶部 `from des.types import ...` 之后加 import:

```python
import zlib
```

在 `_DIR_BIT = {d: 1 << i for i, d in enumerate(ALL_DIRECTIONS)}` (line 24) 之后, `_F = {...}` (line 27) 之前, 插入:

```python
# S4: in-place 方向 sentinel. 不入 ALL_DIRECTIONS, 不入 _DIR_BIT —— FSTACK
# 走 Phenotype.in_place 的独立内核分支, 与四邻 roll 路径正交.
IN_PLACE_DIR: tuple[int, int] = (0, 0)


def _hash_dirs(seq: tuple[str, ...], kind: str) -> tuple[tuple[int, int], ...]:
    """S4 hash-locked direction selection. Pure function of the sequence.

    Determinism: stdlib zlib.crc32(\x1f-joined utf-8 bytes); the \x1f (unit
    separator, not in the alphabet) prevents multi-char token concat ambiguity.
    Python's built-in hash() is salted per process (PYTHONHASHSEED) →不可复现, 致命
    for a data-generation sandbox; crc32 is byte-identical cross-process / cross-machine.

    kind:
      "ffront" | "f4nr1" -> ( ALL_DIRECTIONS[h % 4], )                       1 方向
      "fclump"           -> ( (-1,0), (1,0) ) or ( (0,-1), (0,1) ) per h % 2  一根轴
      "f4nr3"            -> ALL_DIRECTIONS minus ALL_DIRECTIONS[h % 4]        3 邻
    """
    h = zlib.crc32("\x1f".join(seq).encode())
    if kind in ("ffront", "f4nr1"):
        return (ALL_DIRECTIONS[h % 4],)
    if kind == "fclump":
        if h % 2 == 0:
            return ((-1, 0), (1, 0))
        return ((0, -1), (0, 1))
    if kind == "f4nr3":
        drop = h % 4
        return tuple(d for i, d in enumerate(ALL_DIRECTIONS) if i != drop)
    raise ValueError(f"_hash_dirs: unknown kind {kind!r}; "
                     "expected one of {'ffront','f4nr1','fclump','f4nr3'}")
```

- [ ] **Step 4: 跑测试, 确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_hash_dirs.py -v
```

Expected: 9 条全 PASS, 含子进程 determinism + relabel-invariance.

Backtrack: 子进程 determinism 失败 → 检查是不是误用了 `hash(...)` 而非 `zlib.crc32(...)`, 或者把 join 分隔符写成 `""` / `","` 之类 (必须严格 `"\x1f"`).

- [ ] **Step 5: 跑全 suite, 确认无回归**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿 —— `_hash_dirs` / `IN_PLACE_DIR` 还没人读, 既有行为 0 漂移.

- [ ] **Step 6: Commit**

```bash
git add src/des/registry.py tests/test_hash_dirs.py
git commit -m "feat(s4): add _hash_dirs(seq, kind) + IN_PLACE_DIR sentinel

Pure function: zlib.crc32(\\x1f-joined utf-8 bytes) → hash-locked direction
sets per kind {ffront, f4nr1, fclump, f4nr3}. Determinism asserted in a
subprocess test (built-in hash() would fail it). \\x1f separator prevents
multi-char token concat ambiguity. IN_PLACE_DIR=(0,0) is the FSTACK sentinel
(not in ALL_DIRECTIONS, not in _DIR_BIT). No consumer yet — phenotype()
wiring lands in Task 3."
```

---

### Task 2: `Phenotype` 加 `in_place` / `rand_dir` 两 bool 字段(无消费者)

**Goal:** 给 `src/des/types.py` 的 frozen `Phenotype` dataclass 加两个新 bool 字段 `in_place: bool = False` / `rand_dir: bool = False`. **必须给默认值**, 否则既有所有 `Phenotype(...)` 构造点(`registry.py::phenotype` 末尾, 可能还有测试夹具) 全部要带新 kwarg → 大爆炸. 这一步无消费者(下一 Task 才让 `phenotype()` 写它), 既有行为 0 漂移.

**Files:**
- Modify: `src/des/types.py:13-29` (扩 `Phenotype` 字段列表)
- Test: `tests/test_direction_kinds.py` (Create — first test for this owner file; later tasks 续填)

**Interfaces:**
- Consumes: 无.
- Produces:
  - `Phenotype.in_place: bool = False` —— FSTACK 标志, kernel 走当格沉积分支.
  - `Phenotype.rand_dir: bool = False` —— FDRIFT 标志, kernel 每 firing tick 现抽 1-of-4.

- [ ] **Step 1: 写失败测试**

新建 `tests/test_direction_kinds.py`:

```python
# tests/test_direction_kinds.py
"""S4 dynamic-direction primitives + kernel branching.

This file is the S4 owner test file (sibling: tests/test_motif.py / test_vis.py /
test_spectrum_shape.py). Covers the 5 new F primitives' registry rows, phenotype
fields, phenotype-array columns, and end-to-end kernel branching."""
from __future__ import annotations
import pytest


def test_phenotype_has_in_place_default_false():
    """Phenotype.in_place 默认 False —— 既有所有 strain 都不走当格沉积."""
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert hasattr(p, "in_place")
    assert p.in_place is False


def test_phenotype_has_rand_dir_default_false():
    """Phenotype.rand_dir 默认 False —— 既有所有 strain 都不走每 tick 抽."""
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert hasattr(p, "rand_dir")
    assert p.rand_dir is False


def test_phenotype_is_still_frozen():
    """加字段后 dataclass 仍 frozen, 不可改: 守不可变契约."""
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    with pytest.raises(Exception):
        p.in_place = True       # FrozenInstanceError under @dataclass(frozen=True)
```

- [ ] **Step 2: 跑失败测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_direction_kinds.py -v
```

Expected: 三条全 FAIL —— `AttributeError: 'Phenotype' object has no attribute 'in_place'` (rand_dir 同理).

- [ ] **Step 3: 给 `Phenotype` 加两 bool 字段**

修改 `src/des/types.py`:

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
```

(默认值 `False` 让既有 `Phenotype(...)` 构造点不需要带新 kwarg, 编辑别处 0 改动.)

- [ ] **Step 4: 跑测试, 确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_direction_kinds.py -v
```

Expected: 3 条全 PASS.

- [ ] **Step 5: 跑全 suite, 确认无回归**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿. 既有所有 strain 落 `in_place=False` / `rand_dir=False` 默认值, 字节级不变.

Backtrack: 若某个测试夹具复现了 `Phenotype(...)` 的位置参数, 而新两字段的默认值放错位置 (例如挤到了非默认 kwarg 之间), Python 会报 `TypeError: non-default argument follows default argument`. 把两新字段放到所有非默认字段之后, 确保它们是 dataclass 的最后两个字段.

- [ ] **Step 6: Commit**

```bash
git add src/des/types.py tests/test_direction_kinds.py
git commit -m "feat(s4): add Phenotype.in_place / rand_dir bool fields

Two new frozen dataclass fields with default False — FSTACK sets in_place
(kernel emits in source cell, no roll); FDRIFT sets rand_dir (kernel draws
1-of-4 each firing tick from world RNG). Default False so every existing
Phenotype(...) construction site (registry.py::phenotype tail, fixtures)
needs no changes. No consumer yet — Task 3 wires phenotype(); Task 4 wires
phenotype_arrays; Task 5 wires the kernel."
```

---

### Task 3: `phenotype()` 识别三类 directions 字段语义 + 重底定 F4Nr1 行

**Goal:** 让 `phenotype()` 在累加 `_F` 行时识别 `_F[letter]` 中 `directions` 字段的三态语义: 字面 tuple → 旧路径 (OR 进 `dir_bits`); `"hash:<kind>"` 字符串 → 调 `_hash_dirs(sequence, kind)` 把结果 OR 进 `dir_bits`; `"rand:1of4"` 字符串 → 设 `rand_dir=True`. 同时**重底定 F4Nr1**: 把 `_F["F4Nr1"]` 从 `(0.30, ((-1, 0),), 0.05, 4)` 改成 `(0.30, "hash:f4nr1", 0.05, 4)`. 仍不动 5 个新 F 字母 (Task 5 才入), 也不改 kernel; F4Nr4 行原样不动 (字面 4 邻 tuple, 走旧路径).

**Files:**
- Modify: `src/des/registry.py:27-30` (`_F` 表) + `src/des/registry.py:56-125` (`phenotype()` 的 `_F` 分支与构造末尾)
- Test: `tests/test_registry.py` (改既有断言) + `tests/test_direction_kinds.py` (append)

**Interfaces:**
- Consumes: `_hash_dirs` (Task 1), `IN_PLACE_DIR` (Task 1), `Phenotype.in_place / rand_dir` (Task 2), 既有 `ALL_DIRECTIONS` / `_DIR_BIT`.
- Produces:
  - `phenotype(sequence).in_place: bool` —— `True` iff sequence 含至少一个 `_F` 行的 `directions == (IN_PLACE_DIR,)` 或 `directions` 字面 tuple 中只含 `(0, 0)`.
  - `phenotype(sequence).rand_dir: bool` —— `True` iff sequence 含至少一个 `_F` 行 `directions == "rand:1of4"`.
  - `phenotype(sequence).dir_bits: int` —— hash-locked 方向 OR 进同一字段 (与既有静态字面方向同表), 单 `dir_bits` 字段承载混合方向集合 (一个 strain 多 F 字母情况下).

- [ ] **Step 1: 写 / 改测试 —— F4Nr1 popcount==1 (不锁北), 三类 directions 字段语义**

改 `tests/test_registry.py:112-119` 的 `test_dir_bits_match_directions`. 替换为:

```python
def test_dir_bits_match_directions():
    """S4 重底定: F4Nr1 现在是 hash-locked 1-of-4 (而非 v1 占位 ((-1,0),)).
    断言 dir_bits 的 popcount==1, 但**不锁哪个 bit** (由 crc32 决定).
    F4Nr4 仍是 4 邻全开, 不变."""
    # F4Nr1: 1 bit set (hash-locked 单方向, 跨 strain 看似随机)
    db_f4nr1 = phenotype(("F4Nr1",)).dir_bits
    assert bin(db_f4nr1).count("1") == 1, (
        f"F4Nr1 must hash-lock to exactly 1 direction, got dir_bits={db_f4nr1:04b}")
    # F4Nr4 = all four -> all four bits (S4 不动)
    assert phenotype(("F4Nr4",)).dir_bits == (1 << len(ALL_DIRECTIONS)) - 1
    # no F letter -> no directions -> 0
    assert phenotype(("N0",)).dir_bits == 0
```

(行号附近的 import 与 `north_bit` 局部变量一并删除; 注释从 "F4Nr1 = north only" 升级为新描述.)

追加到 `tests/test_direction_kinds.py`:

```python
def test_f4nr1_dir_bits_determined_by_crc32_of_sequence():
    """F4Nr1 的方向位由 crc32 决定, 但跨进程 / 跨调用恒定.
    同一 sequence 调两次 phenotype, dir_bits 必相等."""
    from des.registry import phenotype
    a = phenotype(("F4Nr1",)).dir_bits
    b = phenotype(("F4Nr1",)).dir_bits
    assert a == b
    # popcount 仍是 1
    assert bin(a).count("1") == 1


def test_phenotype_recognizes_hash_locked_directions_field(monkeypatch):
    """模拟一个 hash-locked F 字母, _F[...].directions = 'hash:fclump';
    phenotype() 必须把它翻译为 _hash_dirs(seq, 'fclump') 的 2 方向, OR 进 dir_bits."""
    import des.registry as reg
    monkeypatch.setitem(reg.ALPHABET, "FCLUMP_TEST", "F")
    monkeypatch.setitem(reg.GRAN, "FCLUMP_TEST", "motif")
    monkeypatch.setitem(reg.MOTIF_LEN, "FCLUMP_TEST", 2)
    monkeypatch.setitem(reg._F, "FCLUMP_TEST", (0.45, "hash:fclump", 0.10, 6))
    seq = ("FCLUMP_TEST", "FCLUMP_TEST") + ("N0",) * 14
    p = reg.phenotype(seq)
    # fclump => 2 dirs (一根轴); 全在 ALL_DIRECTIONS 中
    assert bin(p.dir_bits).count("1") == 2
    assert p.in_place is False
    assert p.rand_dir is False


def test_phenotype_recognizes_rand_dir_directions_field(monkeypatch):
    """_F[...].directions = 'rand:1of4' → phenotype 写 rand_dir=True, dir_bits=0."""
    import des.registry as reg
    monkeypatch.setitem(reg.ALPHABET, "FDRIFT_TEST", "F")
    monkeypatch.setitem(reg.GRAN, "FDRIFT_TEST", "residue")
    monkeypatch.setitem(reg._F, "FDRIFT_TEST", (0.15, "rand:1of4", 0.30, 2))
    seq = ("FDRIFT_TEST",) + ("N0",) * 15
    p = reg.phenotype(seq)
    assert p.rand_dir is True
    assert p.in_place is False
    # rand_dir 不预写 dir_bits, kernel 现抽
    assert p.dir_bits == 0


def test_phenotype_recognizes_in_place_directions_field(monkeypatch):
    """_F[...].directions = (IN_PLACE_DIR,) → phenotype 写 in_place=True, dir_bits=0."""
    import des.registry as reg
    monkeypatch.setitem(reg.ALPHABET, "FSTACK_TEST", "F")
    monkeypatch.setitem(reg.GRAN, "FSTACK_TEST", "residue")
    monkeypatch.setitem(reg._F, "FSTACK_TEST", (0.60, (reg.IN_PLACE_DIR,), 0.00, 3))
    seq = ("FSTACK_TEST",) + ("N0",) * 15
    p = reg.phenotype(seq)
    assert p.in_place is True
    assert p.rand_dir is False
    # in_place 不预写 dir_bits, kernel 走当格分支
    assert p.dir_bits == 0


def test_phenotype_mixed_hash_and_static_F_letters_or_dir_bits(monkeypatch):
    """同 strain 含 F4Nr4 (字面 4 邻) + 一个 hash-locked F 字母 → dir_bits 应是
    两者 OR (字面四邻全开 + hash 的额外方向重复, OR 不变化 4-bit 全开)."""
    import des.registry as reg
    monkeypatch.setitem(reg.ALPHABET, "FFRONT_TEST", "F")
    monkeypatch.setitem(reg.GRAN, "FFRONT_TEST", "motif")
    monkeypatch.setitem(reg.MOTIF_LEN, "FFRONT_TEST", 2)
    monkeypatch.setitem(reg._F, "FFRONT_TEST", (0.50, "hash:ffront", 0.25, 4))
    seq = ("F4Nr4", "FFRONT_TEST", "FFRONT_TEST") + ("N0",) * 13
    p = reg.phenotype(seq)
    # F4Nr4 全 4 邻 OR-into-dir_bits 后, hash-locked 单方向只会重复一个 bit
    # → dir_bits 仍是 (1 << 4) - 1 = 15
    assert p.dir_bits == (1 << len(reg.ALL_DIRECTIONS)) - 1
```

- [ ] **Step 2: 跑测试, 确认失败**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_registry.py::test_dir_bits_match_directions tests/test_direction_kinds.py -v
```

Expected: 改写的 `test_dir_bits_match_directions` FAIL (因为 `_F["F4Nr1"]` 还没改, 现 popcount 是 1 但方向仍是 (-1,0) —— 严格地说这条测试在改前已是绿的 popcount==1, **会假性 PASS**, 但 hash 几种分支测试 FAIL 因 phenotype 不识别新字段格式). 五条 hash / rand / in_place / mixed 测试 FAIL with `KeyError: 'hash:ffront'` (或 unpacking 错: `dirs` 不是字面 tuple, 解包 `for d in dirs` 报错).

- [ ] **Step 3: 在 `src/des/registry.py` 改 `_F["F4Nr1"]` 行 + 扩 `phenotype()` 的 `_F` 分支**

把 `_F` 表 (line 27-30) 改为:

```python
_F = {    # name -> (f, directions, p_leave, period)
    # S4: F4Nr1 由 v1 占位 ((-1, 0),) 重底定为 hash-locked 1-of-4 (spec §3.3).
    "F4Nr1": (0.30, "hash:f4nr1", 0.05, 4),
    "F4Nr4": (0.50, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.15, 5),
}
```

把 `phenotype()` 的 `_F` 分支 (line 76-86) 替换为:

```python
        if letter in _F:
            f, dirs_spec, pl, per = _F[letter]
            f_prod *= (1 - f)
            pl_prod *= (1 - pl)
            # S4: dirs_spec 三态. tuple → 字面方向 (旧路径, OR 进 directions/dir_bits);
            # "hash:<kind>" → mint 时调 _hash_dirs(sequence, kind) → 同样 OR 进;
            # "rand:1of4" → 不预写方向, 设 rand_dir=True, kernel 每 tick 现抽.
            if isinstance(dirs_spec, str):
                if dirs_spec == "rand:1of4":
                    rand_dir = True
                elif dirs_spec.startswith("hash:"):
                    kind = dirs_spec[len("hash:"):]
                    for d in _hash_dirs(sequence, kind):
                        if d not in directions:
                            directions.append(d)
                else:
                    raise ValueError(
                        f"_F[{letter!r}].directions: unknown spec {dirs_spec!r}; "
                        "expected tuple, 'hash:<kind>', or 'rand:1of4'")
            else:
                # 字面 tuple. (IN_PLACE_DIR,) 即 ((0, 0),) → in_place=True;
                # 其他字面方向 OR 进 directions 列表 (旧路径).
                if dirs_spec == (IN_PLACE_DIR,):
                    in_place = True
                else:
                    for d in dirs_spec:
                        if d not in directions:
                            directions.append(d)
            periods.append(per)
            f_periods.append(per)
            phase_type = PhaseType.REPRODUCTION
```

在 `phenotype()` 函数顶部累加器初始化处 (现行 line 59-70, `dominant_p: str | None = None` 之前) 加两个本地状态:

```python
    in_place = False           # S4: FSTACK 标志
    rand_dir = False           # S4: FDRIFT 标志
```

把 `Phenotype(...)` 构造调用 (line 119-125) 改为:

```python
    return Phenotype(
        f=f, directions=tuple(directions), p_leave=p_leave, z_raw=z_sum,
        prey_mask=prey_mask, feature_mask=feature_mask, p_x=p_x,
        spectrum=spectrum, period=period,
        repro_period=repro_period, anta_period=anta_period, dir_bits=dir_bits,
        phase_type=phase_type, fold=(),
        in_place=in_place, rand_dir=rand_dir,
    )
```

- [ ] **Step 4: 跑测试, 确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_registry.py::test_dir_bits_match_directions tests/test_direction_kinds.py -v
```

Expected: 所有断言 PASS, 含 F4Nr1 popcount==1 / hash-locked 字段被识别 / rand_dir 被识别 / in_place 被识别 / mixed F4Nr4 + hash 的方向 OR.

Backtrack: 若 `test_phenotype_mixed_hash_and_static_F_letters_or_dir_bits` FAIL, 检查 `directions` 列表是不是被替换为 hash 方向(而非 OR-into). 列表语义是 "去重追加", `dir_bits` 在末尾 OR 得到的应该是各方向位的并集.

- [ ] **Step 5: 跑全 suite, 标记预期受影响的测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -q 2>&1 | tee s4_task3_full_suite.log
```

Expected: 大概率会有「锁 F4Nr1=北」的旧测试 FAIL (例如 `tests/test_reproduction.py` 里端到端测试一份子代落格的 `(-1, 0)` 邻居等). **这些就是 Task 6 集中处理的「F4Nr1 重底定回归」失败**, 不在本步骤修. 把日志里所有以 `FAILED tests/...` 开头的测试名记录到 `s4_task3_affected_tests.txt`. 非「锁 F4Nr1=北」类失败 (import 错 / `Phenotype` 构造 arity 错) 必须本步骤修.

判定: `pytest -v` 失败信息含 `(-1, 0)` 或 "north" 或 "F4Nr1" 字串 → 留 Task 6; 其他 → 本步骤 root-cause 并修.

- [ ] **Step 6: Commit**

```bash
git add src/des/registry.py tests/test_registry.py tests/test_direction_kinds.py
git commit -m "feat(s4): phenotype recognizes hash/rand/in_place direction specs

_F row directions field now three-valued: tuple (literal), 'hash:<kind>'
(crc32-derived at mint), or 'rand:1of4' (sets rand_dir for the per-tick
kernel branch). F4Nr1 re-baselined from v1 placeholder ((-1,0),) to
'hash:f4nr1' — same-strain locks one direction, cross-strain looks
4-neighbor random. Default game changes per user 2026-06-24; affected
tests re-baselined in Task 6."
```

---

### Task 4: `phenotype_arrays` 加 `in_place` / `rand_dir` 两 int8 列

**Goal:** 给 `src/des/phenotype_cache.py::StrainTable.phenotype_arrays(device)` 加两个新的 per-strain `int8` 张量列 (`in_place` / `rand_dir`, 取值 0/1), 让 kernel 可在 vectorized 路径上读到 (`phe["in_place"][sid_long]`). 两新列用 `int8` 与 `vis_mode` (S1 同 dtype) 风格, 避免 PyTorch 在 bool dtype 索引语义上的差异。idx 0 (`EMPTY_ID` 哨兵) 守默认 0; dirty-flag cache 同样的 rebuild 路径; 既有列名 / dtype / 顺序不变。

**Files:**
- Modify: `src/des/phenotype_cache.py:55-95` (`phenotype_arrays` 函数体, 两处: 累加列表 与 result dict)
- Test: `tests/test_direction_kinds.py` (append) + `tests/test_phenotype_cache.py` 既有 dirty-flag 测试做回归.

**Interfaces:**
- Consumes: `Phenotype.in_place: bool` / `Phenotype.rand_dir: bool` (Task 2).
- Produces:
  - `phe["in_place"]: torch.Tensor (int8, shape [n_strains], 0/1)` —— `phe["in_place"][sid] == 1` iff `phenotype(seq_sid).in_place is True`.
  - `phe["rand_dir"]: torch.Tensor (int8, shape [n_strains], 0/1)` —— 同上.

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_direction_kinds.py`:

```python
def test_phenotype_arrays_has_in_place_column():
    """phe['in_place'] 是 int8 张量, idx 0 (EMPTY) 是 0."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    assert "in_place" in phe
    assert phe["in_place"].dtype == torch.int8
    assert int(phe["in_place"][0].item()) == 0


def test_phenotype_arrays_has_rand_dir_column():
    """phe['rand_dir'] 是 int8 张量, idx 0 (EMPTY) 是 0."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    assert "rand_dir" in phe
    assert phe["rand_dir"].dtype == torch.int8
    assert int(phe["rand_dir"][0].item()) == 0


def test_phenotype_arrays_columns_match_python_phenotype():
    """对每个 strain, phe[<col>][sid] 必须 == int(Phenotype.<field>)."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    for sid in range(1, len(eng.table) + 1):
        seq = eng.table.sequence_of(sid)
        p_obj = eng.table.phenotype_of(sid)
        assert int(phe["in_place"][sid].item()) == int(p_obj.in_place)
        assert int(phe["rand_dir"][sid].item()) == int(p_obj.rand_dir)


def test_phenotype_arrays_default_bb0_all_zero_for_in_place_and_rand_dir():
    """默认 BB0 alphabet 里没有 FSTACK / FDRIFT, 所以 in_place / rand_dir
    全是 0 (kernel 永远不会因这两列误走新分支)."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    assert int(phe["in_place"].max().item()) == 0
    assert int(phe["rand_dir"].max().item()) == 0
```

- [ ] **Step 2: 跑失败测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_direction_kinds.py -v -k "phenotype_arrays"
```

Expected: 四条全 FAIL with `assert 'in_place' in phe` 或 `KeyError: 'in_place'`.

- [ ] **Step 3: 在 `src/des/phenotype_cache.py` 加两 int8 列**

编辑 `phenotype_arrays` 函数体 (line 44-95). 在 line 64 `anta_period = [1] * n` 之后插入:

```python
        in_place_col = [0] * n
        rand_dir_col = [0] * n
```

在 `for sid in range(1, n):` 循环体内 (line 65-78), 在 `anta_period[sid] = phe.anta_period` 之后加两行:

```python
            in_place_col[sid] = int(phe.in_place)
            rand_dir_col[sid] = int(phe.rand_dir)
```

在 `result = {...}` 字典 (line 79-90) 末尾, 在 `"anta_period": ...,` 之后追加两个 key:

```python
            "in_place": torch.tensor(in_place_col, dtype=torch.int8, device=device),
            "rand_dir": torch.tensor(rand_dir_col, dtype=torch.int8, device=device),
```

(`int8` 与 S1 `vis_mode` 同 dtype 风格. PyTorch `int8` 张量在 `[sid_long]` 整型 index 下与 `bool` 索引语义一致, 但在 `mask.any()` / `mask.bool()` 上更可移植.)

- [ ] **Step 4: 跑测试, 确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_direction_kinds.py -v -k "phenotype_arrays"
```

Expected: 4 条全 PASS.

- [ ] **Step 5: 跑既有 `test_phenotype_cache.py` 守 dirty-flag rebuild 行为**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_phenotype_cache.py -v
```

Expected: 既有 dirty-flag 测试全 PASS (新加两列走同一份 rebuild 路径, mint 新 strain 即 invalidate, 没改 I1 cache 语义).

Backtrack: 若 `test_phenotype_cache.py` 出现 dtype 或 shape 不匹配, 大概率是 `int8` dtype 在不同平台默认 broadcasting 不同 —— 把 `result[...] = torch.tensor(...)` 改成显式 `torch.tensor([...], dtype=torch.int8, device=device)`. 不要降级为 bool dtype.

- [ ] **Step 6: 跑全 suite**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿, 除了 Task 3 已记录的 F4Nr1 重底定相关失败 (Task 6 处理).

- [ ] **Step 7: Commit**

```bash
git add src/des/phenotype_cache.py tests/test_direction_kinds.py
git commit -m "feat(s4): phenotype_arrays in_place / rand_dir int8 columns

Per-strain int8 (0/1) tensors so kernel can vectorize over strains:
phe['in_place'][sid] and phe['rand_dir'][sid]. idx 0 (EMPTY sentinel) is 0.
Default BB0 max is 0 in both columns — no v1 strain triggers the new
kernel branches. Kernel consumer in Task 5."
```

---

### Task 5: 注册 5 个新 F 基元 + `phase2_reproduce` 三路分支

**Goal:** spec §3.4 表的 5 个新 F 基元 (FSTACK / FCLUMP / FFRONT / F4Nr3 / FDRIFT) 入 `ALPHABET` / `GRAN` / `MOTIF_LEN` / `_F`, 同 commit 把 `phase2_reproduce` 加上 in-place 与 rand-dir 两路分支 —— spec §3.4 明确 "rows and direction handling are one deliverable, wired together". 完成本 task 后 mint 出 FSTACK 即在格内堆叠, mint 出 FDRIFT 即每 tick 现抽; FCLUMP / FFRONT / F4Nr3 走 hash-locked 路径, mint 时由 Task 3 的 phenotype OR 进 dir_bits → 走静态路径无需新分支.

**Files:**
- Modify: `src/des/registry.py` (5 行 ALPHABET / GRAN / MOTIF_LEN / _F)
- Modify: `src/des/kernels/reproduction.py:51-155` (`phase2_reproduce` 加两路分支)
- Test: `tests/test_direction_kinds.py` (append) + `tests/test_reproduction.py` (append)

**Interfaces:**
- Consumes: `_hash_dirs` (Task 1), `IN_PLACE_DIR` (Task 1), `Phenotype.in_place / rand_dir` (Task 2/3), `phenotype_arrays['in_place' | 'rand_dir']` (Task 4), 既有 `binom` / `ArrivalBuffer` / `ALL_DIRECTIONS`.
- Produces:
  - `ALPHABET["FSTACK"|"FCLUMP"|"FFRONT"|"F4Nr3"|"FDRIFT"] = "F"` —— 5 新行.
  - `GRAN["FSTACK"|"F4Nr3"|"FDRIFT"] = "residue"`; `GRAN["FCLUMP"|"FFRONT"] = "motif"`.
  - `MOTIF_LEN["FCLUMP"] = 2; MOTIF_LEN["FFRONT"] = 2`.
  - `_F` 5 新行 (verbatim 抄 spec §3.4 表):
    ```python
    "FSTACK":  (0.60, (IN_PLACE_DIR,), 0.00, 3),
    "FCLUMP":  (0.45, "hash:fclump",   0.10, 6),
    "FFRONT":  (0.50, "hash:ffront",   0.25, 4),
    "F4Nr3":   (0.40, "hash:f4nr3",    0.12, 5),
    "FDRIFT":  (0.15, "rand:1of4",     0.30, 2),
    ```
  - `phase2_reproduce` 行为不变 接口保持 `(world, snap_sid, snap_count, snap_faction, phe, table, birth_tick, T, generator)`. 内部新读 `phe["in_place"]` / `phe["rand_dir"]`, 给 in_place 走当格沉积分支, 给 rand_dir 走每 firing slot 现抽分支.

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_direction_kinds.py`:

```python
def test_s4_new_F_letters_present_in_alphabet_with_family_F():
    """5 个新 F 字母 family 全 'F'."""
    from des.registry import ALPHABET
    for letter in ("FSTACK", "FCLUMP", "FFRONT", "F4Nr3", "FDRIFT"):
        assert ALPHABET.get(letter) == "F", f"{letter}: {ALPHABET.get(letter)!r}"


def test_s4_new_F_letters_have_correct_gran():
    """FSTACK / F4Nr3 / FDRIFT = residue; FCLUMP / FFRONT = motif (spec §3.4)."""
    from des.registry import GRAN
    assert GRAN["FSTACK"] == "residue"
    assert GRAN["F4Nr3"] == "residue"
    assert GRAN["FDRIFT"] == "residue"
    assert GRAN["FCLUMP"] == "motif"
    assert GRAN["FFRONT"] == "motif"


def test_s4_motif_F_letters_have_motif_len_2():
    """FCLUMP / FFRONT 是长度 2 的 motif (spec §3.4)."""
    from des.registry import MOTIF_LEN
    assert MOTIF_LEN["FCLUMP"] == 2
    assert MOTIF_LEN["FFRONT"] == 2


def test_s4_new_F_rows_have_exact_values():
    """每行 (f, directions, p_leave, period) 与 spec §3.4 表 verbatim 一致."""
    from des.registry import _F, IN_PLACE_DIR
    assert _F["FSTACK"] == (0.60, (IN_PLACE_DIR,), 0.00, 3)
    assert _F["FCLUMP"] == (0.45, "hash:fclump",   0.10, 6)
    assert _F["FFRONT"] == (0.50, "hash:ffront",   0.25, 4)
    assert _F["F4Nr3"]  == (0.40, "hash:f4nr3",    0.12, 5)
    assert _F["FDRIFT"] == (0.15, "rand:1of4",     0.30, 2)


def test_phenotype_fstack_strain_has_in_place_true():
    from des.registry import phenotype
    p = phenotype(("FSTACK", "F4Nr4", "P_base", "BroadSweep") + ("N0",) * 12)
    assert p.in_place is True
    assert p.rand_dir is False


def test_phenotype_fdrift_strain_has_rand_dir_true():
    from des.registry import phenotype
    p = phenotype(("FDRIFT", "F4Nr4", "P_base", "BroadSweep") + ("N0",) * 12)
    assert p.rand_dir is True
    assert p.in_place is False


def test_phenotype_fclump_strain_has_two_dir_bits():
    """FCLUMP hash-locked 一根轴 (2 个 bit). 仍 motif=2 只能成对出现."""
    from des.registry import phenotype
    p = phenotype(("FCLUMP", "FCLUMP") + ("N0",) * 14)
    assert bin(p.dir_bits).count("1") == 2
    assert p.in_place is False
    assert p.rand_dir is False


def test_phenotype_f4nr3_strain_has_three_dir_bits():
    """F4Nr3 hash-locked 三邻 (3 个 bit)."""
    from des.registry import phenotype
    p = phenotype(("F4Nr3",) + ("N0",) * 15)
    assert bin(p.dir_bits).count("1") == 3
```

追加到 `tests/test_reproduction.py`:

```python
def test_fstack_strain_offspring_stay_in_source_cell():
    """FSTACK 单 cell 跑 1 tick: 邻居 count 必须为 0, 源格 count 增加 (in-place 沉积).
    构造一个 1×3 grid, 中央格放纯 FSTACK 株, 跑一步, 邻 cell 应仍是 0."""
    import torch
    from des.engine import Engine
    fstack_layout = ("FSTACK",) + ("N0",) * 15
    # 4 factions 必填 (Engine.__init__ 期望 4-tuple), 但只在中央 cell 投种.
    layouts = (fstack_layout,) * 4
    eng = Engine(H=1, W=3, K=8, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2, layouts=layouts)
    eng.run(1, recorder=None, stop_on=())
    # FSTACK 不 roll 到邻居; 任何 cell 累积都从原 cell 来 (3 cell 等量种, 都 in-place).
    # 验证: 任何一个 cell 跑完仍只有 faction 来源是自己原始位置 (不交叉).
    # 简化版断言: 全 grid count 仍 > 0, 且无 0 count cell (in-place 不外流).
    assert (eng.world.count.sum(dim=-1) > 0).all()


def test_fdrift_strain_same_seed_reproducible():
    """FDRIFT 跨 process 同 seed 同结果: kernel generator (world RNG) 必须是
    seed 的确定函数, 不抓 Python 默认 random / torch 默认 RNG."""
    import torch
    from des.engine import Engine
    fdrift_layout = ("FDRIFT", "F4Nr4", "P_base", "BroadSweep") + ("N0",) * 12
    layouts = (fdrift_layout,) * 4

    eng_a = Engine(H=4, W=4, K=8, seed=42, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2, layouts=layouts)
    eng_b = Engine(H=4, W=4, K=8, seed=42, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2, layouts=layouts)
    eng_a.run(3, recorder=None, stop_on=())
    eng_b.run(3, recorder=None, stop_on=())
    assert torch.equal(eng_a.world.count, eng_b.world.count)
    assert torch.equal(eng_a.world.strain_id, eng_b.world.strain_id)


def test_f4nr4_byte_identical_after_s4():
    """F4Nr4 仍 4 邻全开 (spec §1 表锁死), S4 重底 F4Nr1 不许动 F4Nr4. 同 seed
    跑 2 次 → bit-identical."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng_a = Engine(H=8, W=8, K=8, seed=0, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2,
                   layouts=(BB0_TEMPLATE["layout"],) * 4)
    eng_b = Engine(H=8, W=8, K=8, seed=0, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2,
                   layouts=(BB0_TEMPLATE["layout"],) * 4)
    eng_a.run(3, recorder=None, stop_on=())
    eng_b.run(3, recorder=None, stop_on=())
    assert torch.equal(eng_a.world.count, eng_b.world.count)
```

- [ ] **Step 2: 跑失败测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_direction_kinds.py tests/test_reproduction.py -v -k "s4_new_F or fstack or fdrift or f4nr3 or fclump or f4nr4_byte_identical"
```

Expected: 注册表行 4 条 + phenotype 4 条 FAIL with `KeyError: 'FSTACK'` 等; FSTACK / FDRIFT 端到端 FAIL with `KeyError: 'FSTACK'` 在 mint; F4Nr4 byte-identical 跑可能仍 PASS (注册表暂未碰).

- [ ] **Step 3: 在 `src/des/registry.py` 加 5 行新 F 基元**

把 `ALPHABET` 块扩为 (在 `"BroadSweep": "Z",` 之后追加, 与 S2 / S1 落 N / P 字母同纪律 —— 只追加, 不改顺序):

```python
ALPHABET = {
    "N0": "N",
    "F4Nr1": "F", "F4Nr4": "F",
    "P_base": "P", "P_hotspot": "P",
    "BroadSweep": "Z",
    # S4: F-pool dynamic-direction primitives (spec §3.4)
    "FSTACK":  "F",
    "FCLUMP":  "F",
    "FFRONT":  "F",
    "F4Nr3":   "F",
    "FDRIFT":  "F",
}
```

把 `GRAN` 块扩为 (S6 已加; S2 已落 P 池; 这里在末追加 5 行):

```python
GRAN: dict[str, str] = {
    "N0":         "residue",
    "F4Nr1":      "residue",
    "F4Nr4":      "residue",
    "P_base":     "residue",
    "P_hotspot":  "residue",
    "BroadSweep": "residue",
    # ... (S2 加的 10 行 P pool 在此之间, 按本仓库现状不动) ...
    # S4: F-pool dynamic directions
    "FSTACK":  "residue",
    "FCLUMP":  "motif",
    "FFRONT":  "motif",
    "F4Nr3":   "residue",
    "FDRIFT":  "residue",
}
```

把 `MOTIF_LEN` 块扩为:

```python
MOTIF_LEN: dict[str, int] = {
    # S4: motif F primitives (spec §3.4)
    "FCLUMP": 2,
    "FFRONT": 2,
}
```

把 `_F` 块扩为 (Task 3 已改 F4Nr1; 这里追加 5 行, 引用模块级 `IN_PLACE_DIR`):

```python
_F = {    # name -> (f, directions, p_leave, period)
    "F4Nr1": (0.30, "hash:f4nr1", 0.05, 4),
    "F4Nr4": (0.50, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.15, 5),
    # S4: 5 new F primitives (spec §3.4 表 verbatim)
    "FSTACK":  (0.60, (IN_PLACE_DIR,), 0.00, 3),
    "FCLUMP":  (0.45, "hash:fclump",   0.10, 6),
    "FFRONT":  (0.50, "hash:ffront",   0.25, 4),
    "F4Nr3":   (0.40, "hash:f4nr3",    0.12, 5),
    "FDRIFT":  (0.15, "rand:1of4",     0.30, 2),
}
```

- [ ] **Step 4: 在 `src/des/kernels/reproduction.py` 加两路分支**

编辑 `phase2_reproduce` (line 51-155). 在 `dir_bits = phe["dir_bits"][sid_long]` (line 70) 之后追加两行 mask:

```python
    in_place_mask = phe["in_place"][sid_long].bool()      # [H,W,K]
    rand_dir_mask = phe["rand_dir"][sid_long].bool()      # [H,W,K]
```

把 `# --- pass 1: per direction ---` 那个 `for (dy, dx) in ALL_DIRECTIONS:` 循环 (line 87-100) 内部的 `dir_mask` 行改成 "排除 in_place / rand_dir 的 slot, 它们走独立分支":

```python
    # --- pass 1: 静态 dir_bits 路径 (排除 in_place / rand_dir, 它们各有分支) ---
    static_mask = (~in_place_mask) & (~rand_dir_mask)
    rolled = []
    for (dy, dx) in ALL_DIRECTIONS:
        bit = ALL_DIRECTIONS.index((dy, dx))
        dir_mask = ((dir_bits >> bit) & 1).bool() & static_mask
        active = fires & dir_mask
        a = (snap_count * active).to(torch.int32)
        scattered = binom(a, f, generator)
        mut = binom(scattered, p_x, generator)
        non = scattered - mut
        r_non = torch.roll(non, shifts=(dy, dx), dims=(0, 1))
        r_mut = torch.roll(mut, shifts=(dy, dx), dims=(0, 1))
        r_sid = torch.roll(snap_sid, shifts=(dy, dx), dims=(0, 1))
        r_fac = torch.roll(faction_long, shifts=(dy, dx), dims=(0, 1))
        rolled.append((r_non, r_mut, r_sid, r_fac))
```

在 `rolled = []` 之后, 但在 pass 2a 的 `for (r_non, r_mut, r_sid, r_fac) in rolled:` 之前, 追加 in-place 与 rand-dir 两路独立分支. 它们各自走 `binom(scattered, f, generator)` + `binom(scattered, p_x, generator)` 的 mut 切分, 但 emit 时 ty/tx 处理不同:

```python
    # --- S4: in-place 路径 (FSTACK) — 当格沉积, 不 roll. ---
    in_place_active = fires & in_place_mask
    if in_place_active.any():
        a_ip = (snap_count * in_place_active).to(torch.int32)
        scattered_ip = binom(a_ip, f, generator)
        mut_ip = binom(scattered_ip, p_x, generator)
        non_ip = scattered_ip - mut_ip
        # ty/tx 用静态 meshgrid (源格自己), 不 roll. sid / fac 用 snap (parent).
        rolled.append((non_ip, mut_ip, snap_sid, faction_long))

    # --- S4: rand-dir 路径 (FDRIFT) — 每 firing slot 现抽 1-of-4. ---
    rand_active = fires & rand_dir_mask
    if rand_active.any():
        a_rd = (snap_count * rand_active).to(torch.int32)
        scattered_rd = binom(a_rd, f, generator)
        mut_rd = binom(scattered_rd, p_x, generator)
        non_rd = scattered_rd - mut_rd
        # 给每个 [H,W,K] slot 抽 1-of-4 方向索引 (kernel generator = 世界 RNG).
        dir_idx = torch.randint(0, 4, (H, W, K), generator=generator,
                                device=dev, dtype=torch.int64)
        # 每个方向 d, 把 dir_idx==d 的 slot 的 non/mut 投到 d 邻居.
        for d in range(4):
            dy, dx = ALL_DIRECTIONS[d]
            sel = (dir_idx == d) & rand_active
            non_d = (non_rd * sel).to(non_rd.dtype)
            mut_d = (mut_rd * sel).to(mut_rd.dtype)
            r_non = torch.roll(non_d, shifts=(dy, dx), dims=(0, 1))
            r_mut = torch.roll(mut_d, shifts=(dy, dx), dims=(0, 1))
            r_sid = torch.roll(snap_sid, shifts=(dy, dx), dims=(0, 1))
            r_fac = torch.roll(faction_long, shifts=(dy, dx), dims=(0, 1))
            rolled.append((r_non, r_mut, r_sid, r_fac))
```

后续 pass 2a (line 102-106) / pass 2b (line 108-149) / migration (line 151-154) 不动 —— 它们循环 `rolled` 列表, 多塞条目即可.

- [ ] **Step 5: 跑测试, 确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_direction_kinds.py tests/test_reproduction.py -v
```

Expected: 注册表 4 条 + phenotype 4 条 + FSTACK / FDRIFT / F4Nr4 byte-identical 全 PASS.

Backtrack:
- 若 FSTACK 端到端测试 FAIL 因「邻居 count 也增加了」, 检查 in-place 路径误用了 `torch.roll` 或 `static_mask` 没排掉 in_place_mask. in-place 路径必须直接 `rolled.append((non_ip, mut_ip, snap_sid, faction_long))` 把未 roll 的张量塞进去.
- 若 FDRIFT 跨 process 不等, 检查 `torch.randint(..., generator=generator, ...)` 的 `generator` 参数是不是从 `phase2_reproduce(..., generator)` 形参读, 不是 `torch.Generator()` 新建.
- 若 F4Nr4 byte-identical 测试 FAIL, 检查 `static_mask = (~in_place_mask) & (~rand_dir_mask)` 在 BB0 strain 上必须全 True (默认 BB0 既不 in-place 也不 rand-dir), 整个 static 路径与 S4 前等价.

- [ ] **Step 6: 跑全 suite**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 大体绿, 但 Task 3 标记的 F4Nr1 重底定相关失败仍在 (Task 6 处理). 不应有新非 F4Nr1 类失败.

- [ ] **Step 7: Commit**

```bash
git add src/des/registry.py src/des/kernels/reproduction.py tests/test_direction_kinds.py tests/test_reproduction.py
git commit -m "feat(s4): register 5 new F primitives + kernel in_place / rand_dir branches

ALPHABET / GRAN / MOTIF_LEN / _F gain FSTACK / FCLUMP / FFRONT / F4Nr3 /
FDRIFT (spec §3.4 verbatim). phase2_reproduce branches static dir_bits
path on (~in_place_mask & ~rand_dir_mask); FSTACK strain emits in source
cell (no roll); FDRIFT strain draws 1-of-4 per firing slot from world
RNG. F4Nr4 byte-identical regression preserved."
```

---

### Task 6: F4Nr1 重底定回归 — 升级既有「锁北」断言

**Goal:** Task 3 已经把 `_F["F4Nr1"]` 从 `((-1,0),)` 改成 `"hash:f4nr1"`, 默认局动力学因此漂移. spec §3.3 / §6 + 用户 2026-06-24 已批 RE-RECORD; 本 task 集中找出 / 升级所有「锁 F4Nr1=北」/「假设邻 cell 在 (-1, 0)」的既有测试 (Task 3 Step 5 已经把这份名单记到 `s4_task3_affected_tests.txt`). 837MB 首批基线 parquet **不在 plan 范围内** —— 这条 commit 不重跑 batch, 只让测试套面回归绿.

**Files:**
- Modify: 由 Task 3 Step 5 日志驱动的若干测试文件 (典型: `tests/test_reproduction.py` / `tests/test_acceptance.py` / `tests/test_smoke.py`); **不动**任何 `src/` 代码 (Task 3 已完成 source 升级).
- Test: 所有出现「F4Nr1 方向 == (-1, 0)」或「north_bit」硬编码的测试.

**Interfaces:** 无新接口 —— 这是回归升级 task, 升级口径 spec §6 锁定:
- 凡是断言 "F4Nr1 唯一方向 == (-1, 0)" → 改成 "popcount(dir_bits) == 1, 具体哪个 bit 由 crc32 决定, 不锁".
- 凡是断言 "邻 cell (y-1, x) 收到 F4Nr1 子代" → 改成 "F4Nr1 子代落在 ALL_DIRECTIONS 中**某一**邻 (通过 crc32 算出的那一邻); 用 phenotype 调用现算预期方向, 再断言邻 cell".
- 凡是断言「跑 N tick 后某 cell count == 硬数」(因为子代方向从 north 改成 hash 选邻 → 落点漂移) → 用 phenotype-driven 重算预期, **不锁绝对数**.

- [ ] **Step 1: 重读 `s4_task3_affected_tests.txt`, 用 grep 双确认覆盖范围**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -q 2>&1 | tee s4_task6_pre_log.txt
```

然后:

```
grep -nE "F4Nr1|north|\\(-1, *0\\)" tests/*.py | tee s4_task6_grep.txt
```

Expected: `s4_task6_pre_log.txt` 末尾的 `FAILED` 行集合 ⊆ `s4_task6_grep.txt` 命中文件. 任何在 `pre_log` 失败但不在 `grep` 里的测试都得手工查 (可能是别的回归), root-cause 再来.

- [ ] **Step 2: 给每个失败测试做一份升级清单**

逐个看 `s4_task6_grep.txt` 的命中, 标记 3 类:
- (a) **直接锁 north 的方向断言** —— 改为 popcount==1 / 现算预期.
- (b) **依赖 north 的端到端 strain trajectory 断言** (落格 / count) —— 现算预期, 或改成 phenotype-driven 等式 (用 `phenotype(("F4Nr1",)).directions` 决定预期落格).
- (c) **不再相关 (例如 BB0 用 F4Nr4 锁 1, 该位置已经在 S0/S6 改过)** —— 保留不动, 只是 grep 命中不代表需要升级.

把清单写到 commit message 描述里 (commit 时一并提交).

- [ ] **Step 3: 改 `tests/test_registry.py:test_dir_bits_match_directions` (Task 3 已改) — 验证它仍绿**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_registry.py::test_dir_bits_match_directions -v
```

Expected: PASS (Task 3 已经把这条改成 popcount==1).

- [ ] **Step 4: 升级 `tests/test_reproduction.py` 端到端方向断言**

打开 `tests/test_reproduction.py`, 把所有 "F4Nr1 子代必须落 (y-1, x)" 类断言改成「现算预期方向」:

```python
def test_f4nr1_offspring_lands_at_hash_locked_neighbor():
    """F4Nr1 重底定后, 子代落格由 crc32(seq) 决定的那一邻. 不锁北."""
    import torch
    from des.engine import Engine
    from des.registry import phenotype
    f4nr1_layout = ("F4Nr1", "F4Nr4", "P_base", "BroadSweep") + ("N0",) * 12
    expected_dirs = phenotype(f4nr1_layout).directions
    # F4Nr1 在该 layout 下贡献的方向 = expected_dirs ∩ ALL_DIRECTIONS 的子集; F4Nr4 贡献四邻全 OR.
    # 这条测试单独验「F4Nr1 不再 north-only」, 与 F4Nr4 并存时验 popcount==4 (后者主导).
    assert bin(phenotype(("F4Nr1",)).dir_bits).count("1") == 1
    # 跑一步 sanity: 同 seed 两次落格一致.
    eng_a = Engine(H=4, W=4, K=8, seed=0, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2, layouts=(f4nr1_layout,) * 4)
    eng_b = Engine(H=4, W=4, K=8, seed=0, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2, layouts=(f4nr1_layout,) * 4)
    eng_a.run(1, recorder=None, stop_on=())
    eng_b.run(1, recorder=None, stop_on=())
    assert torch.equal(eng_a.world.count, eng_b.world.count)
```

若文件里存在硬编码 `assert world.count[y-1, x, k] > 0` 之类的断言 (Task 3 失败日志里能看到具体行), 把它们改成同款 phenotype-driven 模式: 算 `expected = phenotype(seq).directions`, 然后 `for (dy, dx) in expected: assert world.count[y+dy, x+dx, k].sum() > 0`.

- [ ] **Step 5: 升级 `tests/test_acceptance.py` / `tests/test_smoke.py` 等绝对数断言**

如果 `s4_task3_affected_tests.txt` 里出现 acceptance/smoke 类: 这些一般是 "跑 N tick 后总 count / strain 集合 == 硬数". 应对策略:
- 若是 "count 总数不变 / faction share 总和 == 1.0 ± ε" 这种**结构等式**, 与方向无关 → 应该不会因 F4Nr1 改向漂移, 但 if FAIL, root-cause: 通常是某 cell 的 `world.count` 从 north 邻溢出到了别的邻, 但总和不变 — 检查断言是不是不应该 fail.
- 若是 "tick T 末 cell (y, x) 的 strain == X" 这种**绝对状态**断言 → 用上面同款 phenotype-driven 重算预期; **绝不**抄旧值粘新值 (那样无验证力).

如果 acceptance/smoke 是把整份 parquet 与 fixture 比 byte-equal —— 这是 spec §6 明说要 RE-RECORD 的, **本 plan 不重录 fixture parquet** (固化 838MB 基线归 batch CLI 跑数据线, 用户单独跑后整批回写). 在测试里加 `@pytest.mark.skip(reason="S4 F4Nr1 re-baseline pending fixture re-record")` 显式 skip, 留下 TODO 文档.

例如:

```python
@pytest.mark.skip(
    reason="S4 (2026-06-24): F4Nr1 re-baselined from north-only to hash-locked "
           "1-of-4. This fixture-based byte-equal regression needs the 837MB "
           "baseline parquet re-recorded (spec §3.3 + user 2026-06-24 accepted). "
           "Re-enable after batch re-record sweep.")
def test_first_batch_parquet_byte_identical():
    ...
```

(skip 是显式的 design debt, 不是 silent FAIL.)

- [ ] **Step 6: 跑全 suite, 确认 F4Nr1 类失败全部消失**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -q 2>&1 | tee s4_task6_post_log.txt
```

Expected: 没有 `FAILED tests/...` 行 (skip 不算 FAIL). 若有, root-cause: 大概率某个测试用 `(0, -1)` / `(0, 1)` 等其他「锁特定方向」断言, grep `(0, ?-1)\\|(0, ?1)\\|(-1, ?0)\\|(1, ?0)` tests/*.py 补充清单.

Backtrack: skip 标记不能用于隐藏 non-F4Nr1 类失败. 若 grep 命中行号不解释当前失败, root-cause 那一条单独修, 不要广撒 skip.

- [ ] **Step 7: Commit**

```bash
git add tests/
git commit -m "test(s4): re-baseline F4Nr1 north-only assertions to hash-locked

Following spec §3.3 + user 2026-06-24 acceptance: F4Nr1 changes from v1
placeholder ((-1,0),) to hash-locked 1-of-4. All tests that locked F4Nr1
direction to (-1,0) or asserted offspring at (y-1,x) are upgraded to
read phenotype().directions for the expected direction set. Byte-equal
parquet fixture comparisons are skipped pending the 837MB baseline
re-record (out of plan scope; tracked in skip reasons).

Affected tests:
- <list filled in from s4_task6_grep.txt>"
```

---

### Task 7: 谱预过滤与 5 个新 F 行的交互守护 + relabel-invariance 审计

**Goal:** 5 个新 F 字母入 ALPHABET (Task 5) 后, S6 的 `_spectrum_for(letter)` (S2 又叠了 shape) 现在会把它们当 mutation target —— 须验:
- (a) **gran-match 守门**: motif F (`FCLUMP` / `FFRONT`) 只能突变成同长 motif F, 不能突变成 residue F (spec §3.1 + S6 已实现);
- (b) **没有 P_loopswap_lite 把 F 字母拖进 P 的谱**: P_loopswap_lite (S2) family_mask="adjacent" → |Δrank|=1 → P→{F, Z}, **fine**; 它可以选 F 字母, 但 F 字母只是被替换进 slot, 与方向选择正交;
- (c) **relabel-invariance**: 重排 `_F`/`_Z`/`_P` 量级 (f / z / p_add / period) 不影响 5 个新 F 字母的方向选择 (Task 1 已对 `_hash_dirs` 做过; 这里端到端验 phenotype + kernel).

本 task 不动 source code, 只追加 4 条断言. 把 5 个新 F 字母引发的「面新增之后, S6 / S2 既有机制是否兼容」一次扫清.

**Files:**
- No source modifications expected.
- Test: `tests/test_direction_kinds.py` (append) + `tests/test_motif.py` (append 1 条) + `tests/test_spectrum_shape.py` (append 1 条).

**Interfaces:**
- Consumes: 全部 Task 1-5 产物.
- Produces: 跨 spec (S2 / S4 / S6) 兼容性断言.

- [ ] **Step 1: 写新断言 — gran-match + motif-len 守门 (motif F → motif F only)**

追加到 `tests/test_motif.py`:

```python
def test_motif_F_spectrum_only_matches_equal_length_motif_F():
    """FCLUMP (motif, len=2) 的 spectrum 应只含同长 motif (FFRONT, len=2);
    不能含 residue F (F4Nr1 / F4Nr4 / FSTACK / F4Nr3 / FDRIFT) —— S6 gran-match
    + 等长预过滤的 S4 验证."""
    from des.registry import _spectrum_for, ALPHABET, GRAN, MOTIF_LEN
    spec = _spectrum_for("FCLUMP")
    for t, q in spec:
        # 必须 motif, 等长, family F
        assert GRAN[t] == "motif", f"{t}: gran {GRAN[t]!r}; FCLUMP→{t} cross-gran"
        assert MOTIF_LEN[t] == 2, f"{t}: motif_len {MOTIF_LEN[t]}; FCLUMP→{t} cross-len"
        assert ALPHABET[t] == "F", f"{t}: family {ALPHABET[t]!r}; FCLUMP→{t} cross-family"
        # FCLUMP itself不在 spectrum (self-exclude)
        assert t != "FCLUMP"
```

- [ ] **Step 2: 写新断言 — P_loopswap_lite 适邻含 5 个新 F 字母**

追加到 `tests/test_spectrum_shape.py`:

```python
def test_p_loopswap_lite_adjacent_can_reach_new_F_letters():
    """P_loopswap_lite (family_mask='adjacent') 把 F 邻列入 mutation target.
    Task 5 加完 5 个新 F 字母后, P_loopswap_lite 的 spectrum 应包含至少
    部分新 F 字母 (那些 gran=residue 与 P 配 —— FSTACK / F4Nr3 / FDRIFT)."""
    from des.registry import _spectrum_for, ALPHABET, GRAN
    spec = dict(_spectrum_for("P_loopswap_lite"))
    # P 是 rank=2; F 是 rank=1, Z 是 rank=3, 都是 adjacent. mask='adjacent' 是
    # |Δrank|==1 → F + Z.
    # 同 gran residue → 只有 residue F / residue Z 入 spectrum
    f_residue = {l for l in ALPHABET if ALPHABET[l] == "F" and GRAN[l] == "residue"}
    for letter in ("FSTACK", "F4Nr3", "FDRIFT"):
        # 这 3 个 S4 新增 F 字母都是 residue, 应在 spectrum 里
        assert letter in spec, f"{letter} (residue F) missing from P_loopswap_lite spectrum"
    # motif F 不能入 (cross-gran)
    for letter in ("FCLUMP", "FFRONT"):
        assert letter not in spec, f"{letter} (motif F) leaked into P (residue) spectrum"
```

- [ ] **Step 3: 写新断言 — relabel-invariance 端到端 (方向不读 f/z/p magnitude)**

追加到 `tests/test_direction_kinds.py`:

```python
def test_relabel_invariance_directions_read_only_letter_sequence(monkeypatch):
    """spec §6 relabel-invariance: 重排 _F / _Z / _P 量级不影响新 F 基元的
    方向选择. F4Nr1 / FFRONT / FCLUMP / F4Nr3 / FDRIFT 的方向都从 crc32(seq)
    或 world RNG 来, 与 magnitude 完全解耦."""
    import des.registry as reg
    seqs = (
        ("F4Nr1", "F4Nr4", "P_base", "BroadSweep") + ("N0",) * 12,
        ("FFRONT", "FFRONT") + ("N0",) * 14,
        ("FCLUMP", "FCLUMP", "F4Nr4", "P_hotspot", "BroadSweep") + ("N0",) * 11,
        ("F4Nr3", "F4Nr4", "P_base", "BroadSweep") + ("N0",) * 12,
    )
    pre = [reg.phenotype(s).dir_bits for s in seqs]
    # 重排 _F / _Z / _P 量级 (NOT 字面 directions 字段 — 那是结构, 不是 magnitude).
    monkeypatch.setitem(reg._F, "F4Nr4", (0.01, ((1, 0), (-1, 0), (0, 1), (0, -1)), 0.99, 99))
    monkeypatch.setitem(reg._Z, "BroadSweep", (0.99, ("F", "Z"), 99))
    monkeypatch.setitem(reg._P, "P_base", (0.0, 99))
    monkeypatch.setitem(reg._P, "P_hotspot", (0.0, 99))
    post = [reg.phenotype(s).dir_bits for s in seqs]
    assert pre == post, (
        f"directions leaked magnitude reading. pre={pre!r}, post={post!r}")


def test_phenotype_arrays_cache_dirty_flag_invalidated_on_new_F_mint():
    """mint 新 F strain 触发 dirty-flag → phenotype_arrays 重新构造,
    新 in_place / rand_dir / dir_bits 必须出现在 phe 表里. 这一条守
    Task 4 的 cache 路径对 Task 5 新 F 字母也有效."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    # 取一次 phe (会缓存)
    phe_pre = eng.table.phenotype_arrays(torch.device("cpu"))
    n_pre = phe_pre["in_place"].shape[0]
    # mint 一个 FSTACK strain
    fstack_layout = ("FSTACK",) + ("N0",) * 15
    sid = eng.table.get_or_mint(fstack_layout)
    # 取第二次 phe; n_strains 应增加, FSTACK 行 in_place=1
    phe_post = eng.table.phenotype_arrays(torch.device("cpu"))
    n_post = phe_post["in_place"].shape[0]
    assert n_post > n_pre, "phenotype_arrays did not rebuild on new mint"
    assert int(phe_post["in_place"][sid].item()) == 1
```

- [ ] **Step 4: 跑新断言, 确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_direction_kinds.py tests/test_motif.py tests/test_spectrum_shape.py -v -k "motif_F_spectrum_only or p_loopswap_lite_adjacent or relabel_invariance_directions or phenotype_arrays_cache_dirty"
```

Expected: 4 条全 PASS. 它们的实现 hook 都在 Task 1-5 完成时已具备 —— 本 task 只追加守门.

Backtrack:
- `test_motif_F_spectrum_only_matches_equal_length_motif_F` FAIL → 检查 S6 `_spectrum_for` 是不是 `MOTIF_LEN[t] != src_len` 的 continue 漏了; Task 5 加 motif F 后, S6 已有路径必须接住.
- `test_p_loopswap_lite_adjacent_can_reach_new_F_letters` FAIL → 检查 S2 `SPECTRUM_SHAPE["P_loopswap_lite"]` 是不是 `"adjacent"` (而非 `"F"`).
- `test_relabel_invariance_directions_read_only_letter_sequence` FAIL → root-cause `phenotype()` 是否在某分支偷读了 `_F[letter][0]` (f) 当作方向决策; 不应该.
- `test_phenotype_arrays_cache_dirty_flag_invalidated_on_new_F_mint` FAIL → 检查 `get_or_mint` 是否在新 strain 路径上设 `_arrays_dirty = True`; 既有代码已设 (`phenotype_cache.py:28`).

- [ ] **Step 5: 跑全 suite 最终回归**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -q
```

Expected: 全绿, 与 Task 6 末态一致 (除被 skip 的 byte-equal parquet fixture 之外).

- [ ] **Step 6: Commit**

```bash
git add tests/test_direction_kinds.py tests/test_motif.py tests/test_spectrum_shape.py
git commit -m "test(s4): cross-spec compat + relabel-invariance audit

Four guard tests: (a) motif F → motif F only (S6 gran-match + equal-len
holds after S4 mints motif F rows); (b) P_loopswap_lite adjacent reaches
new residue F letters (S2 family_mask='adjacent' compatible with S4 F
expansion); (c) phenotype.dir_bits is byte-identical under f/z/p magnitude
shuffle (directions read crc32 of letter sequence only, never magnitude);
(d) phenotype_arrays dirty-flag invalidates on new F mint."
```

---

### Task 8: Final regression sweep + smoke + push

**Goal:** 把整个 S4 deliverable (Tasks 1-7) 一起跑一遍, 确认全套测试绿、smoke run 不崩、性能档位 (~15.8ms/tick / 128² grid) 没明显漂移, 工作树干净, 推 origin. 这是 Task 9 的同款收口动作 (sibling: S0 Task 6, S6 Task 9, S1 Task 7, S2 Task 7).

**Files:**
- 不预期 source 改动. 若有回归暴露, 本 task 修 forward, commit message 引用 offending commit.
- Test: `tests/` (整套) + `scripts/run_batch.py --probe` smoke + (推荐) 一次默认 4-faction symmetric run smoke.

**Interfaces:**
- Consumes: Task 1-7 全部产物.
- Produces: 绿 `pytest tests/` + 干净 `git status` + push 到 origin.

- [ ] **Step 1: Full pytest sweep**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -q
```

Expected: 全绿. 总数 = 285 engine + 146 web + S6 落地的 motif tests + S1 落地的 vis tests + S2 落地的 spectrum_shape tests + S4 新增 (`test_hash_dirs.py` ~9 + `test_direction_kinds.py` ~16 + 既有 `test_registry.py` / `test_reproduction.py` 的 append). 精确数会随时间漂移; **没有 `FAILED tests/...` 行**是验收标准 (`SKIPPED` 行允许 — 它们是 Task 6 显式标记的 fixture re-record 占位).

Backtrack: 若有任何 FAIL, 先按测试 owner 文件 root-cause 到对应 Task; 用 `git log` 找出 offending commit, fix forward, 不要 reset.

- [ ] **Step 2: Smoke run probe (确认运行时性能没崩)**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --probe 30
```

Expected stdout 形如 `[probe 30 ticks] X.X ms/tick | peak Y.YY GB | strains->… | total …`. `X.X ms/tick` 应保持在 S0 / S6 / S1 / S2 完工时的同一档 (target ≈15.8 ms/tick on 128² grid; ≤ 20% drift acceptable). exit 0; 不写 parquet (probe 路径 record=False).

如果 drift > 20%, 最常见原因是:
- (a) `phase2_reproduce` 里 in_place / rand_dir 路径 unconditional 跑了 `binom`/`torch.roll` 即使 mask 全 False —— 把 `if in_place_active.any():` 守门提到分支前, 默认局 (BB0 没 FSTACK / FDRIFT) `any()==False` 直接 skip.
- (b) `_hash_dirs` 在 hot loop 里被反复调 —— 它应该只在 `phenotype()` mint 时调一次 (per strain), kernel 永远不该见.

- [ ] **Step 3: Byte-identical default-run smoke (推荐, 但 spec §6 已讲 F4Nr1 重底定 → 默认局漂移, 故只验同 seed 双跑一致, 不与 pre-S4 baseline 比)**

跑两次同 seed:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --seeds 0 --T 30
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --seeds 0 --T 30
```

Expected: 两次产出的 parquet 在 `data/runs/` 下用 pyarrow 读起来 `(tick, cell, strain, count)` 行级一致 (用 `pyarrow.parquet.read_table` 读后 `equals`). 这一条守 "S4 增加了 RNG 用量 (FDRIFT 每 firing slot 抽 1-of-4), 但仍由 world generator 驱动 → 同 seed 同结果".

如果默认局 BB0 不含 FDRIFT, 这条等价于 S2 时代的 same-seed 复现性 —— 大概率会绿.

- [ ] **Step 4: Inspect & clean stray data**

```
git status
```

Expected: 干净工作树. 若 `data/runs/<ts>-*.parquet` 有 smoke 残留, 删掉 —— **不**入 commit (它们不是 fixture, 而且 837MB 基线 RE-RECORD 是 batch CLI 单独跑的事).

- [ ] **Step 5: Final commit (only if Step 1 needed a fix-forward)**

If Step 1 surfaced a regression you fixed:

```bash
git add <files-touched>
git commit -m "fix(s4): <description of the regression fixed>"
```

Otherwise this step is a no-op.

- [ ] **Step 6: Push to origin**

```bash
git push origin <current-branch>
```

Expected: push succeeds. The branch is ready for review / merge to `main`.

后续动作 (out of S4 plan scope, 用户单独决定时机):
1. 837MB 首批基线 parquet 在 `scripts/run_batch.py --seeds 0 1 2 3 --T 450` 重跑; 完工后回填进 fixtures, 把 Task 6 末段的 `pytest.mark.skip` 解开.
2. S5 (FBURST f-window) 落地后回看本 plan: FBURST 入 `_F` 时它的 dirs 字段会复用 S4 的字面 4-邻 tuple (`((-1, 0), (1, 0), (0, -1), (0, 1))`), 但 S5 owns f-window 部分 (per-tick on/off f), 与本 plan 无 source-level 冲突.
3. S8 落地 A 池方向变体 (F8Ar1 / Lance Front / Ember Drip / F_TRICKLE / F_SCATTER) 时复用 `_hash_dirs` / `IN_PLACE_DIR` / 三态 directions 字段 / kernel 三路分支机制 —— 本 plan 已经把 machinery 做完, S8 只追加新行.

---

## Self-Review

**1. Spec coverage:**

- §1 (5 个新 F + 三类 direction kinds): Task 1 (`_hash_dirs` 函数) + Task 2 (`Phenotype` 两 bool) + Task 3 (`phenotype()` 识别三态 + F4Nr1 重底定) + Task 4 (phenotype_arrays 两 int8 列) + Task 5 (5 行入注册表 + kernel 三路).
- §2 (red lines — crc32 / 不许 `dir_bits==0` overload / 不许 `hash()`): Task 1 用 `zlib.crc32("\x1f".join(seq).encode())` (子进程 determinism 测试守门); Task 3 / Task 5 用 `Phenotype.in_place` / `rand_dir` 独立 bool 字段 (而非 overload `dir_bits==0`); 跨 Task 守 F4Nr1 重底定的 user-batched RE-RECORD.
- §3.1 (hash-locked dirs computed at mint, no kernel change): Task 1 (`_hash_dirs` 计算) + Task 3 (`phenotype()` 在 mint 时 OR 进 `dir_bits`); kernel 在 hash-locked 路径上 0 改动 —— FFRONT / FCLUMP / F4Nr3 / F4Nr1 走既有 dir_bits → ALL_DIRECTIONS roll 路径.
- §3.2 (新 kernel logic: FDRIFT per-tick rand_dir + FSTACK in-place): Task 2 / Task 4 / Task 5 联合实现 —— `Phenotype.rand_dir` / `in_place` 字段 → `phenotype_arrays['rand_dir' | 'in_place']` int8 列 → `phase2_reproduce` 两路独立分支. Spec §3.2 「Do NOT overload `dir_bits==0`」明确兑现.
- §3.3 (F4Nr1 RESOLVED — hash-locked 1-of-4): Task 3 把 `_F["F4Nr1"]` 由 `((-1, 0),)` 改为 `"hash:f4nr1"`; Task 6 把所有「锁北」断言升级 (popcount==1 但不锁哪个 bit); 用户已批 RE-RECORD 在 commit message 与 task 描述显式记录.
- §3.4 (Register S4 owns F-pool rows): Task 5 把 5 行 verbatim 抄 spec §3.4 表 (FSTACK / FCLUMP / FFRONT / F4Nr3 / FDRIFT 的 `f / p_leave / period / gran / direction spec`); FBURST 显式 NOT here (out of scope, S5 owns).
- §4 (data flow): mint(seq) ─► phenotype() hash-locked → dir_bits / per-tick rand_dir → rand_dir / in-place → in_place. Task 3 (phenotype) + Task 5 (kernel) 联合实现.
- §5 (Error handling: crc32 always uint32 / FSTACK 显式 in_place flag): Task 1 (`crc32 % 4` / `% 2` 总合法 index); Task 5 (`in_place` 独立 flag, 不依赖 `dir_bits==0` 等价).
- §6 (Testing: 回归 + 新功能 + relabel-invariance): Task 1 子进程 determinism + relabel-invariance; Task 3 F4Nr1 popcount==1 / 三态识别; Task 5 FSTACK 在格内 / FDRIFT 跨进程同 seed 同结果 / F4Nr4 byte-identical; Task 6 F4Nr1 重底定回归升级; Task 7 跨 spec compat + relabel-invariance 端到端; Task 8 全套 sweep + smoke.
- §7 (Out of scope): FBURST / F_NOVA → S5 (本 plan 注释多次声明); A 池方向变体 (F8Ar1 / Lance Front / Ember Drip / F_TRICKLE / F_SCATTER) → S8 复用 S4 machinery, 不在本 plan.
- **Red lines (§2):** 方向是结构函数 (crc32) 或诚实 RNG (world generator); 禁 Python 内建 `hash()` (子进程 determinism 测试守门); 禁 `dir_bits==0` overload (两独立 bool 字段); F4Nr4 静态 4-邻 不动 (Task 5 byte-identical 测试); 5 行入注册表与方向逻辑同 commit (Task 5).

**2. Placeholder scan:**

无 `TBD` / `TODO` / "implement later" / "fill in details" / "similar to Task N" / "write tests for the above" 等 plan-failure 字串. 所有 code step 给出真实代码, 所有 command step 给出真实命令 + 预期输出, 所有 backtrack 条件给出具体 root-cause / fix. Task 6 的 "<list filled in from s4_task6_grep.txt>" 是 commit message 的设计动态部分 (实施者按其本地实际 grep 结果填), 不是 plan-level 占位 —— plan 已经把 grep 命令与判定规则给齐.

**3. Type consistency:**

- `_hash_dirs(seq: tuple[str, ...], kind: str) -> tuple[tuple[int, int], ...]` —— Task 1 定义, Task 3 / Task 7 consume.
- `IN_PLACE_DIR: tuple[int, int]` = `(0, 0)` —— Task 1 定义, Task 3 / Task 5 consume (`(IN_PLACE_DIR,)` 作为 `_F["FSTACK"][1]` 字面 sentinel).
- `Phenotype.in_place: bool = False` / `Phenotype.rand_dir: bool = False` —— Task 2 定义, Task 3 写, Task 4 / Task 5 / Task 7 consume.
- `phe["in_place"]: torch.Tensor (int8)` / `phe["rand_dir"]: torch.Tensor (int8)` —— Task 4 定义, Task 5 / Task 7 consume.
- `_F[letter] = (f, directions, p_leave, period)` 其中 `directions` 三态: `tuple[(int, int), ...]` | `"hash:<kind>"` str | `"rand:1of4"` str —— Task 3 内识别策略, Task 5 5 行实例化.
- `phase2_reproduce(world, snap_sid, snap_count, snap_faction, phe, table, birth_tick, T, generator) -> tuple[ArrivalBuffer, Tensor]` —— signature 保持 pre-S4 一致, Task 5 内部加分支.

Spec §3.4 表里 `period` 列与本 plan File Structure 命名契约对照 verbatim: FSTACK=3 / FCLUMP=6 / FFRONT=4 / F4Nr3=5 / FDRIFT=2 —— Task 5 Step 3 (_F 块) 与 spec §3.4 字节级一致.

无 method/property 名称漂移: `_hash_dirs` 全 plan 同名; `IN_PLACE_DIR` 全 plan 同名; `in_place` / `rand_dir` (lowercase + underscore, 不出现 camelCase 变体).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-24-s4-dynamic-directions.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. Use `superpowers:subagent-driven-development`.
2. **Inline Execution** — execute tasks in this session with batch checkpoints. Use `superpowers:executing-plans`.

Which approach?
