# DES 可视化 Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 Digital Evolution Sandbox 引擎做一个单页 web app:默认整屏 = 实时游戏界面(验收之眼,肉眼看红皇后四阵营共存动力学),探索分析/数据展示为可折叠侧栏。

**Architecture:** aiohttp 后台 asyncio 任务跑现有 `Engine`,每 tick 把紧凑格子帧 + 读数打包成 JSON 经 WebSocket 推给浏览器实时渲染,同时复用现有 `Recorder` 边跑边写 parquet 到隔离 playground 目录。前端纯 HTML/CSS/JS 无框架,128×128 `ImageData` 最近邻放大到 canvas、单图层合成混色(阵营=色相、密度=alpha、争夺格=混色)。

**Tech Stack:** Python 3.12+(conda `basic` env,真解释器 `D:\anaconda3\envs\basic\python.exe`,torch 2.10+cu128/CUDA/RTX 5080)、aiohttp 3.13.3(env 已装,零新依赖)、pyarrow 22(已装)、pytest 9(已装)。前端纯浏览器原生(无框架、无构建步骤、无图表库)。

## Global Constraints

以下为 spec(`docs/superpowers/specs/2026-06-22-viz-web-app-design.md`)的项目级硬要求,**每个 task 的要求都隐含包含本节**。值逐字照抄:

- **环境:** 跑测试与脚本一律用真解释器 `D:/anaconda3/envs/basic/python.exe`;裸 `python` 是另一套 cpu-only torch 2.5.1,**禁用**。**绝不 pip/conda install**(后端依赖 aiohttp 已在 env 内)。测试从 repo 根跑 `D:/anaconda3/envs/basic/python.exe -m pytest <path> -v`(pyproject 设了 `pythonpath=["src"]`)。`scripts/`/`webapp/` 下脚本需 PowerShell `$env:PYTHONPATH='src'`。
- **格子默认 128×128。** 网格尺寸是全局参(默认 128,可在配置面板调),默认与首批一致。
- **BB0 对称局,四阵营同条 layout。** 四始祖株拿同一条填好的 layout,差异只靠突变涌现。**不碰角色系统(被 HARD-GATE 住、本轮不设计)。**
- **基元调色板 = 6 个 v1 子集:** `N0 / F4Nr1 / F4Nr4 / P_base / P_hotspot / BroadSweep`(= `registry.ALPHABET` 的键全集)。
- **locked 位只读、只 6 个 mutable 槽可改:** locked 位 `{1:"F4Nr4", 5:"BroadSweep", 7:"P_base"}`(0-indexed,`registry._LOCKED`);mutable 槽 `{0,2,3,9,10,13}`(`registry._SLOTS`);其余非锁非槽位固定 `"N0"`。
- **红线 1 — 数据隔离:** playground 跑出的 parquet **必须隔离**(写到 `data/playground/`),绝不与正式采集 run(`data/runs/`)混进同一池。
- **红线 2 — live 帧忠实张量:** 前端合成混色 OK,但绝不做平滑/插值让前沿「好看」。帧编码只忠实读 world tensor 的真值。
- **红线 3 — 无手写谁强:** 配置只让玩家选基元字母(对称、四家同条),绝不暴露任何「相对优势」系数。
- **红线 4 — 四阵营对称不变量:** 由 `init_factions` 守门校验硬保(locked 位 == `_LOCKED`、差异只落在 `_SLOTS`、四阵营注入同一条 layout)。
- **红线 5 — 读数单一来源:** 占比/distinct/N2/d_max/occupied 只有一份纯函数定义(`webapp/readouts.py`),live 端与 `analyze_batch` 共用,杜绝两套公式漂移。
- **节奏 = 引擎出多快播多快:** 算完一 tick 即推一帧,不限速、不缓冲、不插值(守红线 2)。
- **YAGNI:** 帧传输先用 JSON(不预先做二进制编码);Tier B 钻取靠 parquet 谓词下推不预建索引;前端是 demo,人工验收画面,不引前端测试框架。

---

## File Structure

新增/改动文件及各自单一职责:

- **`src/des/registry.py`**(改):加 `validate_bb0_layout(layout)` 守门校验(挨着 `_LOCKED`/`_SLOTS`,registry 拥有 BB0 不变量)。
- **`src/des/world.py`**(改):`init_factions` 加可选 `layout` 参,进函数先校验,默认行为不变。
- **`src/des/engine.py`**(改):`Engine.__init__` 加可选 `layout` 参,透传给 `init_factions`,默认不变。
- **`webapp/readouts.py`**(新):读数纯函数 `compute_readouts` + `occupied_cells`,被 server 与 analyze_batch 共用(红线 5)。
- **`scripts/analyze_batch.py`**(改):占比/distinct/N2/d_max/occupied 改调 `webapp.readouts`,锁死单一来源。
- **`webapp/frame.py`**(新):world tensor → JSON 帧编码(紧凑格子帧 + 读数)。
- **`webapp/server.py`**(新):aiohttp app + WebSocket + 后台引擎任务 + 静态资源 + 钻取 HTTP 路由 + playground Recorder 接线。含可单测的 `build_engine_from_config` 纯助手。
- **`webapp/static/index.html` / `app.js` / `style.css`**(新):单页前端(无框架),人工验收。
- **`webapp/README.md`**(新):一句话启动命令。
- **`data/playground/`**(新,gitignore):隔离 parquet 目录。
- **测试:** `tests/test_world_layout.py`(校验器 + layout 参)、`tests/test_readouts.py`、`tests/test_frame.py`、`tests/test_server_config.py`、`tests/test_drilldown.py`。`tests/test_analyze_batch.py` 已存在,作 Task 3 回归闸。

---

### Task 1: BB0 layout 守门校验器(registry)

**Files:**
- Modify: `src/des/registry.py`(在 `_SLOTS`/`_LOCKED` 定义之后、`BB0_TEMPLATE` 之后追加函数)
- Test: `tests/test_world_layout.py`(新建)

**Interfaces:**
- Consumes: `registry._LOCKED`(`{1:"F4Nr4", 5:"BroadSweep", 7:"P_base"}`)、`registry._SLOTS`(`{0,2,3,9,10,13}`)、`registry.ALPHABET`(6 基元键)。
- Produces: `validate_bb0_layout(layout: tuple[str, ...]) -> None` —— 校验通过返回 None,违反任一不变量 raise `ValueError`。Task 2 的 `init_factions` 调它。

不变量(红线 4):locked 位必须等于 `_LOCKED` 的值;非锁非槽位(backbone)必须是 `"N0"`;只有 `_SLOTS` 位可变,且其值必须是 6 基元之一。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_world_layout.py
import pytest
from des.registry import BB0_TEMPLATE, validate_bb0_layout, _SLOTS, _LOCKED


def _canonical():
    return list(BB0_TEMPLATE["layout"])


def test_canonical_bb0_passes():
    # default BB0 (all slots N0) is the canonical symmetric genotype
    assert validate_bb0_layout(BB0_TEMPLATE["layout"]) is None


def test_slot_change_passes():
    lay = _canonical()
    lay[0] = "P_hotspot"          # slot 0 is mutable
    lay[13] = "F4Nr1"             # slot 13 is mutable
    assert validate_bb0_layout(tuple(lay)) is None


def test_tampered_locked_position_rejected():
    lay = _canonical()
    lay[1] = "N0"                 # position 1 must stay F4Nr4
    with pytest.raises(ValueError, match="locked"):
        validate_bb0_layout(tuple(lay))


def test_tampered_backbone_position_rejected():
    lay = _canonical()
    lay[4] = "BroadSweep"         # position 4 is backbone-fixed N0, not a slot
    with pytest.raises(ValueError, match="backbone"):
        validate_bb0_layout(tuple(lay))


def test_unknown_primitive_in_slot_rejected():
    lay = _canonical()
    lay[2] = "ZZZ_not_a_primitive"
    with pytest.raises(ValueError, match="palette"):
        validate_bb0_layout(tuple(lay))


def test_wrong_length_rejected():
    with pytest.raises(ValueError, match="16"):
        validate_bb0_layout(("N0",) * 15)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_world_layout.py -v`
Expected: FAIL — `ImportError: cannot import name 'validate_bb0_layout' from 'des.registry'`

- [ ] **Step 3: Write minimal implementation**

在 `src/des/registry.py` 末尾(`BB0_TEMPLATE = {...}` 之后)追加:

```python
def validate_bb0_layout(layout: tuple[str, ...]) -> None:
    """Enforce the BB0 symmetry invariant (viz spec §5 / red-line 4).
    locked positions must equal _LOCKED; backbone (non-locked, non-slot)
    positions must stay "N0"; only _SLOTS positions may vary, and only to a
    primitive in the 6-letter palette. Raises ValueError on any violation."""
    if len(layout) != 16:
        raise ValueError(f"BB0 layout must have 16 positions, got {len(layout)}")
    for i, letter in enumerate(layout):
        if i in _LOCKED:
            if letter != _LOCKED[i]:
                raise ValueError(
                    f"position {i} is locked to {_LOCKED[i]!r}, got {letter!r}")
        elif i in _SLOTS:
            if letter not in ALPHABET:
                raise ValueError(
                    f"slot {i} = {letter!r} not in palette {sorted(ALPHABET)}")
        else:
            if letter != "N0":
                raise ValueError(
                    f"position {i} is backbone-fixed to 'N0', got {letter!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_world_layout.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/des/registry.py tests/test_world_layout.py
git commit -m "feat: BB0 layout gatekeeper validator (red-line 4)"
```

---

### Task 2: 引擎触点 —— `init_factions` + `Engine` 接 layout 参(spec §5)

**Files:**
- Modify: `src/des/world.py`(`init_factions` 签名 + 函数体)
- Modify: `src/des/engine.py`(`Engine.__init__` 签名 + 传参)
- Test: `tests/test_world_layout.py`(在 Task 1 文件追加)

**Interfaces:**
- Consumes: `registry.validate_bb0_layout`(Task 1)、`registry.BB0_TEMPLATE`、`StrainTable.get_or_mint`。
- Produces:
  - `init_factions(H, W, K, device, table, fill_per_cell, n_fac=4, layout=None) -> World` —— `layout=None` 时用 `BB0_TEMPLATE["layout"]`(默认行为完全不变);非 None 时先 `validate_bb0_layout(layout)` 再 mint。四阵营注入同一条 layout。
  - `Engine(H, W, K, seed, device, z_max=8.0, fill_per_cell=None, check_every=10, layout=None)` —— `layout` 透传给 `init_factions`。Task 7 的 server 用 `layout=` 起局。

唯一引擎改动。`layout` 默认 None → 走老路径 → 红线 4 由校验器在非默认时硬保。

- [ ] **Step 1: Write the failing test**

在 `tests/test_world_layout.py` 末尾追加:

```python
import torch
from des.world import init_factions
from des.phenotype_cache import StrainTable
from des.engine import Engine

DEV = torch.device("cpu")


def test_init_factions_default_layout_unchanged():
    # layout=None must reproduce the canonical BB0 seeding (behavior-neutral)
    t = StrainTable()
    w = init_factions(8, 8, 16, DEV, t, fill_per_cell=10, n_fac=4)
    bb0 = t.get_or_mint(BB0_TEMPLATE["layout"])
    assert int((w.count.sum(dim=-1) > 0).sum()) == 4   # 4 seeded cells
    centers = [(2, 2), (2, 6), (6, 2), (6, 6)]
    for (cy, cx) in centers:
        assert int(w.strain_id[cy, cx, 0]) == bb0       # same BB0 everywhere


def test_init_factions_custom_layout_minted_for_all_four():
    t = StrainTable()
    lay = list(BB0_TEMPLATE["layout"]); lay[0] = "P_hotspot"   # legal slot change
    custom = tuple(lay)
    w = init_factions(8, 8, 16, DEV, t, fill_per_cell=10, n_fac=4, layout=custom)
    expect = t.get_or_mint(custom)
    centers = [(2, 2), (2, 6), (6, 2), (6, 6)]
    seen = set()
    for (cy, cx) in centers:
        assert int(w.strain_id[cy, cx, 0]) == expect      # all four = same custom layout
        seen.add(int(w.faction[cy, cx, 0]))
    assert seen == {0, 1, 2, 3}


def test_init_factions_rejects_tampered_layout():
    t = StrainTable()
    bad = list(BB0_TEMPLATE["layout"]); bad[1] = "N0"      # tampered locked position
    with pytest.raises(ValueError, match="locked"):
        init_factions(8, 8, 16, DEV, t, fill_per_cell=10, n_fac=4, layout=tuple(bad))


def test_engine_passes_layout_through():
    lay = list(BB0_TEMPLATE["layout"]); lay[2] = "F4Nr1"
    e = Engine(H=8, W=8, K=16, seed=0, device=DEV, fill_per_cell=10, layout=tuple(lay))
    expect = e.table.get_or_mint(tuple(lay))
    # the seeded strain at a quadrant center is the custom layout
    assert int(e.world.strain_id[2, 2, 0]) == expect


def test_engine_default_layout_unchanged():
    e = Engine(H=8, W=8, K=16, seed=0, device=DEV, fill_per_cell=10)
    bb0 = e.table.get_or_mint(BB0_TEMPLATE["layout"])
    assert int(e.world.strain_id[2, 2, 0]) == bb0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_world_layout.py -v`
Expected: FAIL — `init_factions() got an unexpected keyword argument 'layout'`

- [ ] **Step 3: Write minimal implementation**

`src/des/world.py` —— 改 `init_factions` 签名与 mint 行:

```python
def init_factions(H: int, W: int, K: int, device: torch.device,
                  table: StrainTable, fill_per_cell: int, n_fac: int = 4,
                  layout: tuple[str, ...] | None = None) -> World:
    """Seed BB0 at the four quadrant centers, one faction each, everything else empty.
    The four centers are the D4-symmetric orbit of one point (equal to grid center,
    equal nearest-wall distance, pairwise-symmetric) → no faction gets a geometric edge.
    layout=None uses the canonical BB0_TEMPLATE["layout"] (default behavior unchanged);
    a custom layout is gatekeeper-validated (red-line 4) and injected identically into
    all four factions."""
    assert fill_per_cell <= K, "fill must fit in K slots"
    assert n_fac == 4, "v1 seeds exactly 4 factions at the 4 quadrant centers"
    if layout is None:
        layout = BB0_TEMPLATE["layout"]
    else:
        validate_bb0_layout(layout)
    w = World(H, W, K, device)
    bb0 = table.get_or_mint(layout)
    centers = [(H // 4, W // 4), (H // 4, 3 * W // 4),
               (3 * H // 4, W // 4), (3 * H // 4, 3 * W // 4)]
    for fac, (cy, cx) in enumerate(centers):
        w.strain_id[cy, cx, 0] = bb0
        w.count[cy, cx, 0] = fill_per_cell
        w.faction[cy, cx, 0] = fac
    return w
```

同文件顶部 import 追加 `validate_bb0_layout`:

```python
from des.registry import BB0_TEMPLATE, validate_bb0_layout
```

`src/des/engine.py` —— 改 `Engine.__init__` 签名与 `init_factions` 调用:

```python
    def __init__(self, H, W, K, seed, device, z_max=8.0, fill_per_cell=None,
                 check_every=10, layout=None):
        self.H, self.W, self.K, self.device, self.z_max = H, W, K, device, z_max
        self.check_every = check_every
        self.table = StrainTable()
        fill = K // 2 if fill_per_cell is None else fill_per_cell
        self.world = init_factions(H, W, K, device, self.table,
                                   fill_per_cell=fill, n_fac=NFAC, layout=layout)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.birth = torch.zeros((H, W, K), dtype=torch.int32, device=device)
        self.T = 0
        self._phe = self.table.phenotype_arrays(device)
        self._phe_n = len(self.table)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_world_layout.py tests/test_world.py tests/test_engine.py -v`
Expected: PASS — Task 1 + Task 2 新测全过,且 `test_world.py`/`test_engine.py` 旧测无回归(默认 layout 行为不变)。

- [ ] **Step 5: Commit**

```bash
git add src/des/world.py src/des/engine.py tests/test_world_layout.py
git commit -m "feat: optional validated layout param for init_factions + Engine (spec §5)"
```

---

### Task 3: 读数纯函数(`webapp/readouts.py`,红线 5 单一来源)

**Files:**
- Create: `webapp/readouts.py`
- Create: `webapp/__init__.py`(空,让 `webapp` 成包,供 `from webapp.readouts import ...`)
- Test: `tests/test_readouts.py`(新建)

**Interfaces:**
- Consumes: 无(stdlib only —— 不 import pandas / torch,保持纯)。
- Produces: `compute_readouts(cell_x, cell_y, strain, faction, count) -> dict` —— 五条等长序列(一条记录 = 一个 tick 的一个非空 (cell, slot)),返回 `{total:int, occupied_cells:int, distinct_strains:int, n2:float, d_max:float, faction_share:dict[int,float]}`。Task 4(analyze_batch)与 Task 5(frame)都调它。

定义(逐字锁死,与现 `analyze_batch` 一致,否则 Task 4 回归闸破):
`total=Σcount`;`faction_share[f]=Σcount_f/total`;`distinct_strains` = 出现的不同 strain 数;`n2=1/Σp_s²`(`p_s=strain_total/total`,空则 0);`d_max=max_s p_s`(空则 0);`occupied_cells` = 不同 `(cell_x,cell_y)` 数。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_readouts.py
from webapp.readouts import compute_readouts


def test_empty_is_all_zero():
    r = compute_readouts([], [], [], [], [])
    assert r == {"total": 0, "occupied_cells": 0, "distinct_strains": 0,
                 "n2": 0.0, "d_max": 0.0, "faction_share": {}}


def test_two_equal_strains_one_cell_each():
    # two strains 10/10 -> freqs .5/.5 -> N2 = 1/(.25+.25) = 2.0, d_max = .5
    cx = [0, 1]; cy = [0, 0]; strain = ["A", "B"]; fac = [0, 0]; cnt = [10, 10]
    r = compute_readouts(cx, cy, strain, fac, cnt)
    assert r["total"] == 20
    assert r["occupied_cells"] == 2
    assert r["distinct_strains"] == 2
    assert abs(r["n2"] - 2.0) < 1e-9
    assert abs(r["d_max"] - 0.5) < 1e-9
    assert r["faction_share"] == {0: 1.0}


def test_faction_share_quarter_each():
    cx = [0, 1, 2, 3]; cy = [0, 0, 0, 0]
    strain = ["S", "S", "S", "S"]; fac = [0, 1, 2, 3]; cnt = [10, 10, 10, 10]
    r = compute_readouts(cx, cy, strain, fac, cnt)
    for f in (0, 1, 2, 3):
        assert abs(r["faction_share"][f] - 0.25) < 1e-9
    assert r["distinct_strains"] == 1        # all the same strain
    assert abs(r["d_max"] - 1.0) < 1e-9       # one strain owns everything


def test_skewed_strain_freqs():
    # A=30 B=10 -> .75/.25 -> N2 = 1/(.5625+.0625)=1.6, d_max=.75
    cx = [0, 1]; cy = [0, 0]; strain = ["A", "B"]; fac = [0, 0]; cnt = [30, 10]
    r = compute_readouts(cx, cy, strain, fac, cnt)
    assert abs(r["n2"] - 1.6) < 1e-9
    assert abs(r["d_max"] - 0.75) < 1e-9


def test_same_cell_multiple_slots_counts_once_as_occupied():
    # two records in the same cell (different slots) -> 1 occupied cell
    cx = [5, 5]; cy = [7, 7]; strain = ["A", "B"]; fac = [0, 1]; cnt = [3, 4]
    r = compute_readouts(cx, cy, strain, fac, cnt)
    assert r["occupied_cells"] == 1
    assert r["total"] == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_readouts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'webapp'`

- [ ] **Step 3: Write minimal implementation**

`webapp/__init__.py`:

```python
# webapp package
```

`webapp/readouts.py`:

```python
# webapp/readouts.py
"""Single-source per-tick readouts (viz spec red-line 5). Pure: stdlib only,
no pandas/torch import. The live path (webapp/frame.py) feeds world-tensor
records; scripts/analyze_batch.py feeds parquet-df records. One definition,
shared, so the live acceptance picture and the offline report never drift."""
from __future__ import annotations


def compute_readouts(cell_x, cell_y, strain, faction, count) -> dict:
    """All five sequences are equal length; one entry per non-empty
    (cell, slot) record of ONE tick. Returns:
      total            int   = Σ count
      occupied_cells   int   = # distinct (cell_x, cell_y)
      distinct_strains int   = # distinct strain present
      n2               float = 1 / Σ p_s²   (p_s = strain_total / total); 0 if empty
      d_max            float = max_s p_s; 0 if empty
      faction_share    dict[int,float] = Σcount_f / total per faction
    """
    total = 0
    by_strain: dict = {}
    by_faction: dict = {}
    occ = set()
    for i in range(len(count)):
        c = count[i]
        total += c
        by_strain[strain[i]] = by_strain.get(strain[i], 0) + c
        by_faction[faction[i]] = by_faction.get(faction[i], 0) + c
        occ.add((cell_x[i], cell_y[i]))
    n2 = 0.0
    d_max = 0.0
    if total:
        sumsq = sum((v / total) ** 2 for v in by_strain.values())
        n2 = 1.0 / sumsq if sumsq else 0.0
        d_max = max(v / total for v in by_strain.values())
    denom = total or 1
    return {
        "total": int(total),
        "occupied_cells": len(occ),
        "distinct_strains": len(by_strain),
        "n2": float(n2),
        "d_max": float(d_max),
        "faction_share": {int(f): v / denom for f, v in by_faction.items()},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_readouts.py -v`
Expected: PASS — 5 passed

- [ ] **Step 5: Commit**

```bash
git add webapp/__init__.py webapp/readouts.py tests/test_readouts.py
git commit -m "feat: single-source per-tick readout pure function (red-line 5)"
```

---

### Task 4: `analyze_batch` 改调共享读数(锁死单一来源)

**Files:**
- Modify: `scripts/analyze_batch.py`(`diversity_metrics` + `survival_spatial_metrics` 的 per-tick 占比/distinct/n2/d_max/occupied 段)
- Test: `tests/test_analyze_batch.py`(已存在,**回归闸**,不改)

**Interfaces:**
- Consumes: `webapp.readouts.compute_readouts`(Task 3)。
- Produces: 无新公共接口;行为对现有测试不变(known-answer 全过)。

约束:`tests/test_analyze_batch.py` 的 `test_survival_spatial_known_answers`、`test_diversity_known_answers` 是逐值断言(share=0.25、n2=2.0/1.6、d_max=0.75、occupied、distinct)。改完它们必须仍 PASS —— 这是「单一来源不漂移」的可执行证明。

`analyze_batch` 跑在 `scripts/` 下,`tests/test_analyze_batch.py:2` 已把 `scripts/` 插进 `sys.path`;`webapp` 包要可 import。从 repo 根跑 pytest 时 `pythonpath=["src"]` 不含 repo 根,故在 `analyze_batch.py` 顶部按 `scripts→repo-root` 相对路径把 repo 根插入 `sys.path`(下方实现含)。

- [ ] **Step 1: 先确认回归闸当前为绿(baseline)**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_analyze_batch.py -v`
Expected: PASS — 全过(改动前基线)。

- [ ] **Step 2: 改 `analyze_batch.py` 调用共享读数**

`scripts/analyze_batch.py` 顶部 import 段(`import pandas as pd` 之后)追加 —— 让 `webapp` 可 import(scripts 不在 src 路径下):

```python
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from webapp.readouts import compute_readouts
```

在 `diversity_metrics` 内,把 per-tick 的 `distinct_strains/n2/d_max` 三行计算替换为调用共享函数(`freqs` 仍按原样保留,established_flux/leader 不动)。原循环体:

```python
    for t in ticks:
        s = by_strain.loc[t]
        s = s[s > 0]
        tot = float(s.sum()) or 1.0
        f = (s / tot)
        freqs[t] = {str(k): float(v) for k, v in f.items()}
        distinct_strains[t] = int((s > 0).sum())
        n2[t] = float(1.0 / (f ** 2).sum()) if len(f) else 0.0
        d_max[t] = float(f.max()) if len(f) else 0.0
```

替换为(freqs 保留;三标量改走 compute_readouts,喂该 tick 的活记录):

```python
    for t in ticks:
        s = by_strain.loc[t]
        s = s[s > 0]
        tot = float(s.sum()) or 1.0
        f = (s / tot)
        freqs[t] = {str(k): float(v) for k, v in f.items()}
        lt = df[(df["tick"] == t) & (df["count"] > 0)]
        r = compute_readouts(lt["cell_x"].tolist(), lt["cell_y"].tolist(),
                             lt["strain"].tolist(), lt["faction"].tolist(),
                             lt["count"].tolist())
        distinct_strains[t] = r["distinct_strains"]
        n2[t] = r["n2"]
        d_max[t] = r["d_max"]
```

在 `survival_spatial_metrics` 内,`faction_share` 与 `occupied_cells` 改走共享函数。原段:

```python
    def _occ(g):
        return g[["cell_x", "cell_y"]].drop_duplicates().shape[0]
    occ = live.groupby("tick", group_keys=False)[["cell_x", "cell_y"]].apply(_occ)
    occupied_cells = {int(t): int(occ.get(t, 0)) for t in ticks}
    fill_tick = next((t for t in ticks if occupied_cells[t] >= n_cells), None)
```

后面单独的 faction_share 段:

```python
    fac_cnt = live.groupby(["tick", "faction"])["count"].sum()
    faction_share = {int(t): {} for t in ticks}
    for (t, f), v in fac_cnt.items():
        denom = total_count.get(int(t), 0) or 1
        faction_share[int(t)][int(f)] = float(v) / denom
```

两段合并为按 tick 调一次共享函数(`faction_occupied`/`per_cell_fac`/`first_cross` 维持原样,不动):

```python
    occupied_cells = {}
    faction_share = {int(t): {} for t in ticks}
    for t in ticks:
        lt = live[live["tick"] == t]
        r = compute_readouts(lt["cell_x"].tolist(), lt["cell_y"].tolist(),
                             lt["strain"].tolist(), lt["faction"].tolist(),
                             lt["count"].tolist())
        occupied_cells[int(t)] = r["occupied_cells"]
        faction_share[int(t)] = r["faction_share"]
    fill_tick = next((t for t in ticks if occupied_cells[t] >= n_cells), None)
```

- [ ] **Step 3: 跑回归闸验证不变**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_analyze_batch.py -v`
Expected: PASS — 全过(share=0.25 / n2=2.0,1.6 / d_max=0.75 / occupied / distinct 逐值仍对)。

- [ ] **Step 4: Commit**

```bash
git add scripts/analyze_batch.py
git commit -m "refactor: analyze_batch delegates per-tick readouts to shared fn (red-line 5)"
```

---

### Task 5: 帧编码(`webapp/frame.py`,红线 2 忠实张量)

**Files:**
- Create: `webapp/frame.py`
- Test: `tests/test_frame.py`(新建)

**Interfaces:**
- Consumes: `webapp.readouts.compute_readouts`(Task 3)、`des.world.World`(读 `.strain_id/.count/.faction`,均 `[H,W,K]` tensor)、`StrainTable.sequence_of`。
- Produces:
  - `encode_frame(world, table, tick, H, W, top_n=5) -> dict` —— 返回可 `json.dumps` 的帧:
    ```
    {"tick": int, "H": int, "W": int,
     "cells": [[y, x, c0, c1, c2, c3], ...],   # 每个占用格一行:四阵营各自 count 和
     "readouts": {... compute_readouts 输出 ...},
     "leaderboard": [{"strain": seq_str, "faction": f, "count": c, "share": p}, ...]}
    ```
    `cells` 只含非空格(忠实张量,无插值/平滑)。四个 `cN` = 该格 faction N 的 count 之和(前端按 count 加权混色 + 总密度 alpha)。`leaderboard` = 主导株排行榜(spec §4):全局 count 降序 top_n 株,每株记其总 count、占比(`count/total`)、主导阵营(该株 count 最大的阵营)。**leaderboard 是 live-only**,在 `encode_frame` 内算,不进 `compute_readouts`(避免牵动 analyze_batch / 守红线 5)。
  - `cell_detail(world, table, y, x) -> dict` —— Tier A 钻取:`{"y":y,"x":x,"strains":[{"strain":seq_str,"faction":f,"count":c}, ...]}`,该格非空 slot 逐条。Task 7 的钻取路由 live 端用它。

红线 2:`encode_frame` 只读 world tensor 真值聚合,绝不平滑。`readouts` 复用 Task 3 同一函数(红线 5)。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_frame.py
import json
import torch
from des.world import World
from des.phenotype_cache import StrainTable
from des.registry import BB0_TEMPLATE
from webapp.frame import encode_frame, cell_detail

DEV = torch.device("cpu")


def _world_with(table):
    # 4x4 world, K=8. cell (1,1): faction0 BB0 count5. cell (2,2): faction1 BB0 count3 + faction2 BB0 count2.
    w = World(4, 4, 8, DEV)
    bb0 = table.get_or_mint(BB0_TEMPLATE["layout"])
    w.strain_id[1, 1, 0] = bb0; w.count[1, 1, 0] = 5; w.faction[1, 1, 0] = 0
    w.strain_id[2, 2, 0] = bb0; w.count[2, 2, 0] = 3; w.faction[2, 2, 0] = 1
    w.strain_id[2, 2, 1] = bb0; w.count[2, 2, 1] = 2; w.faction[2, 2, 1] = 2
    return w


def test_encode_frame_only_nonempty_cells():
    t = StrainTable(); w = _world_with(t)
    fr = encode_frame(w, t, tick=7, H=4, W=4)
    assert fr["tick"] == 7 and fr["H"] == 4 and fr["W"] == 4
    assert len(fr["cells"]) == 2                      # only the 2 occupied cells
    # frame must be JSON-serializable (no torch scalars leaking through)
    json.dumps(fr)


def test_encode_frame_faction_counts_per_cell():
    t = StrainTable(); w = _world_with(t)
    fr = encode_frame(w, t, tick=0, H=4, W=4)
    cells = {(c[0], c[1]): c[2:] for c in fr["cells"]}
    assert cells[(1, 1)] == [5, 0, 0, 0]              # faction0 = 5
    assert cells[(2, 2)] == [0, 3, 2, 0]              # faction1 = 3, faction2 = 2


def test_encode_frame_readouts_match_shared_fn():
    t = StrainTable(); w = _world_with(t)
    fr = encode_frame(w, t, tick=0, H=4, W=4)
    r = fr["readouts"]
    assert r["total"] == 10                           # 5 + 3 + 2
    assert r["occupied_cells"] == 2
    assert abs(r["faction_share"][0] - 0.5) < 1e-9    # 5/10


def test_cell_detail_lists_strains_in_cell():
    t = StrainTable(); w = _world_with(t)
    d = cell_detail(w, t, y=2, x=2)
    assert d["y"] == 2 and d["x"] == 2
    seq = ".".join(BB0_TEMPLATE["layout"])
    assert {"strain": seq, "faction": 1, "count": 3} in d["strains"]
    assert {"strain": seq, "faction": 2, "count": 2} in d["strains"]
    assert len(d["strains"]) == 2


def test_cell_detail_empty_cell():
    t = StrainTable(); w = _world_with(t)
    d = cell_detail(w, t, y=0, x=0)
    assert d["strains"] == []


def test_leaderboard_ranks_strains_by_total_count():
    # add a second distinct strain so the leaderboard has >1 entry to rank
    t = StrainTable(); w = _world_with(t)
    lay2 = list(BB0_TEMPLATE["layout"]); lay2[0] = "P_hotspot"
    s2 = t.get_or_mint(tuple(lay2))
    w.strain_id[0, 0, 0] = s2; w.count[0, 0, 0] = 100; w.faction[0, 0, 0] = 3
    fr = encode_frame(w, t, tick=0, H=4, W=4, top_n=5)
    lb = fr["leaderboard"]
    # BB0 total = 5+3+2 = 10; s2 total = 100 -> s2 ranks first
    assert lb[0]["strain"] == ".".join(lay2) and lb[0]["count"] == 100
    assert lb[0]["faction"] == 3                       # dominant faction of s2
    assert abs(lb[0]["share"] - 100 / 110) < 1e-9      # total world = 110
    assert lb[1]["count"] == 10                        # BB0 second


def test_leaderboard_top_n_caps_length():
    t = StrainTable(); w = _world_with(t)
    fr = encode_frame(w, t, tick=0, H=4, W=4, top_n=1)
    assert len(fr["leaderboard"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_frame.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'webapp.frame'`

- [ ] **Step 3: Write minimal implementation**

`webapp/frame.py`:

```python
# webapp/frame.py
"""world tensor -> JSON frame (viz spec §2/§6). Faithful to the tensor: only
non-empty cells, per-faction count sums, NO smoothing/interpolation (red-line 2).
Readouts reuse the single-source pure function (red-line 5)."""
from __future__ import annotations
import torch
from webapp.readouts import compute_readouts

NFAC = 4


def encode_frame(world, table, tick: int, H: int, W: int, top_n: int = 5) -> dict:
    cnt = world.count.to("cpu")
    sid = world.strain_id.to("cpu")
    fac = world.faction.to("cpu")
    nz = torch.nonzero(cnt > 0, as_tuple=False)        # [M,3] = (y,x,k)
    ys = nz[:, 0].tolist(); xs = nz[:, 1].tolist(); ks = nz[:, 2]
    facs = fac[nz[:, 0], nz[:, 1], ks].tolist()
    cnts = cnt[nz[:, 0], nz[:, 1], ks].tolist()
    sids = sid[nz[:, 0], nz[:, 1], ks].tolist()
    # per-cell per-faction count sum
    by_cell: dict = {}
    for y, x, f, c in zip(ys, xs, facs, cnts):
        row = by_cell.setdefault((y, x), [0, 0, 0, 0])
        row[f] += c
    cells = [[y, x, *row] for (y, x), row in by_cell.items()]
    strains = [".".join(table.sequence_of(int(s))) for s in sids]
    readouts = compute_readouts(xs, ys, strains, facs, cnts)
    leaderboard = _leaderboard(strains, facs, cnts, readouts["total"], top_n)
    return {"tick": int(tick), "H": int(H), "W": int(W),
            "cells": cells, "readouts": readouts, "leaderboard": leaderboard}


def _leaderboard(strains, facs, cnts, total: int, top_n: int) -> list:
    """Dominant-strain ranking (spec §4). live-only — NOT in compute_readouts,
    so analyze_batch / red-line 5 stay untouched. Per strain: total count, share
    (count/total), and dominant faction (the faction holding the most of it)."""
    agg: dict = {}   # strain -> {"count": int, "fac": {f: c}}
    for s, f, c in zip(strains, facs, cnts):
        e = agg.setdefault(s, {"count": 0, "fac": {}})
        e["count"] += c
        e["fac"][f] = e["fac"].get(f, 0) + c
    denom = total or 1
    ranked = sorted(agg.items(), key=lambda kv: kv[1]["count"], reverse=True)[:top_n]
    out = []
    for s, e in ranked:
        dom_fac = max(e["fac"], key=e["fac"].get)
        out.append({"strain": s, "faction": int(dom_fac),
                    "count": int(e["count"]), "share": e["count"] / denom})
    return out


def cell_detail(world, table, y: int, x: int) -> dict:
    cnt = world.count[y, x].to("cpu")
    sid = world.strain_id[y, x].to("cpu")
    fac = world.faction[y, x].to("cpu")
    out = []
    for k in range(cnt.shape[0]):
        c = int(cnt[k])
        if c > 0:
            out.append({"strain": ".".join(table.sequence_of(int(sid[k]))),
                        "faction": int(fac[k]), "count": c})
    return {"y": int(y), "x": int(x), "strains": out}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_frame.py -v`
Expected: PASS — 7 passed

- [ ] **Step 5: Commit**

```bash
git add webapp/frame.py tests/test_frame.py
git commit -m "feat: world-tensor -> JSON frame encoder + cell drilldown (red-line 2)"
```

---

### Task 6: 配置→引擎 纯助手(`build_engine_from_config`)

**Files:**
- Create: `webapp/server.py`(本 task 只写纯助手段 + 模块级常量;aiohttp app 在 Task 7 加)
- Test: `tests/test_server_config.py`(新建)

**Interfaces:**
- Consumes: `des.engine.Engine`(Task 2 带 layout 参)、`des.registry`(`_SLOTS`/`_LOCKED`/`ALPHABET`)、`des.world.validate_bb0_layout` 间接经 Engine。
- Produces:
  - `PALETTE: list[str]` —— 6 基元有序表 `["N0","F4Nr1","F4Nr4","P_base","P_hotspot","BroadSweep"]`(前端下拉同源)。
  - `layout_from_slots(slots: dict[int,str]) -> tuple[str,...]` —— 收 `{slot_index: primitive}`,拼 16 位 layout:locked 位填 `_LOCKED`、`_SLOTS` 位填传入(缺省 `"N0"`)、其余 backbone 填 `"N0"`。非法 slot key(不在 `_SLOTS`)或非法基元 raise `ValueError`。
  - `build_engine_from_config(cfg: dict, device) -> tuple[Engine, dict]` —— `cfg` 形如 `{"slots":{0:"N0",...}, "grid":128, "K":64, "fill":20, "T":450, "seed":0, "z_max":8.0}`,缺省用括号内默认。返回 `(engine, resolved_cfg)`,`resolved_cfg` 是补全默认后的配置(含拼好的 `layout` tuple),供 server 回显与起 Recorder。

红线 3/4:config 只收基元字母与 run 级参;layout 经 `layout_from_slots` 拼好后由 `Engine`→`init_factions`→`validate_bb0_layout` 硬校验。无任何「谁强」系数入口。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server_config.py
import pytest
import torch
from webapp.server import PALETTE, layout_from_slots, build_engine_from_config
from des.registry import BB0_TEMPLATE, _SLOTS, _LOCKED

DEV = torch.device("cpu")


def test_palette_is_the_six_v1_primitives():
    assert PALETTE == ["N0", "F4Nr1", "F4Nr4", "P_base", "P_hotspot", "BroadSweep"]


def test_layout_from_empty_slots_is_canonical_bb0():
    assert layout_from_slots({}) == BB0_TEMPLATE["layout"]


def test_layout_from_slots_fills_mutable_positions():
    lay = layout_from_slots({0: "P_hotspot", 13: "F4Nr1"})
    assert lay[0] == "P_hotspot" and lay[13] == "F4Nr1"
    assert lay[1] == _LOCKED[1]                # locked untouched
    assert lay[4] == "N0"                       # backbone untouched


def test_layout_from_slots_rejects_non_slot_index():
    with pytest.raises(ValueError, match="slot"):
        layout_from_slots({4: "F4Nr1"})         # 4 is backbone, not a slot


def test_layout_from_slots_rejects_unknown_primitive():
    with pytest.raises(ValueError, match="palette"):
        layout_from_slots({0: "NOPE"})


def test_build_engine_defaults():
    eng, cfg = build_engine_from_config({}, DEV)
    assert cfg["grid"] == 128 and cfg["K"] == 64 and cfg["fill"] == 20
    assert cfg["T"] == 450 and cfg["seed"] == 0 and cfg["z_max"] == 8.0
    assert cfg["layout"] == BB0_TEMPLATE["layout"]
    assert eng.H == 128 and eng.W == 128 and eng.K == 64


def test_build_engine_custom_slot_reaches_world():
    cfg_in = {"slots": {2: "F4Nr1"}, "grid": 16, "K": 8, "fill": 4, "seed": 1}
    eng, cfg = build_engine_from_config(cfg_in, DEV)
    expect = eng.table.get_or_mint(cfg["layout"])
    assert int(eng.world.strain_id[4, 4, 0]) == expect   # quadrant center of 16-grid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_server_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'webapp.server'`

- [ ] **Step 3: Write minimal implementation**

`webapp/server.py`(本 task 只到纯助手;Task 7 续 aiohttp):

```python
# webapp/server.py
"""aiohttp viz server for the DES engine. This module's pure helpers
(layout_from_slots / build_engine_from_config) are unit-tested without the
event loop; the aiohttp app + WebSocket live loop are added on top."""
from __future__ import annotations
import torch
from des.engine import Engine
from des.registry import _SLOTS, _LOCKED, ALPHABET

# 6 v1 primitives, ordered (front-end dropdown reads the same list via /config)
PALETTE = ["N0", "F4Nr1", "F4Nr4", "P_base", "P_hotspot", "BroadSweep"]

_DEFAULTS = {"grid": 128, "K": 64, "fill": 20, "T": 450, "seed": 0, "z_max": 8.0}


def layout_from_slots(slots: dict) -> tuple[str, ...]:
    """Assemble a 16-position BB0 layout from mutable-slot choices.
    locked positions := _LOCKED; slot positions := given (default 'N0');
    backbone positions := 'N0'. Rejects non-slot indices and off-palette letters."""
    out = []
    for i in range(16):
        if i in _LOCKED:
            out.append(_LOCKED[i])
        elif i in _SLOTS:
            letter = slots.get(i, "N0")
            if letter not in ALPHABET:
                raise ValueError(f"slot {i} = {letter!r} not in palette {PALETTE}")
            out.append(letter)
        else:
            out.append("N0")
    for k in slots:
        if k not in _SLOTS:
            raise ValueError(f"position {k} is not a mutable slot {sorted(_SLOTS)}")
    return tuple(out)


def build_engine_from_config(cfg: dict, device) -> tuple[Engine, dict]:
    """cfg keys (all optional): slots{int:str}, grid, K, fill, T, seed, z_max.
    Returns (engine, resolved_cfg) where resolved_cfg has defaults filled and a
    'layout' tuple. Engine -> init_factions -> validate_bb0_layout enforces
    red-line 4; only primitive letters + run knobs ever enter here (red-line 3)."""
    slots = {int(k): v for k, v in (cfg.get("slots") or {}).items()}
    layout = layout_from_slots(slots)
    resolved = dict(_DEFAULTS)
    for k in _DEFAULTS:
        if cfg.get(k) is not None:
            resolved[k] = cfg[k]
    resolved["slots"] = slots
    resolved["layout"] = layout
    g = int(resolved["grid"])
    eng = Engine(H=g, W=g, K=int(resolved["K"]), seed=int(resolved["seed"]),
                 device=device, z_max=float(resolved["z_max"]),
                 fill_per_cell=int(resolved["fill"]), layout=layout)
    return eng, resolved
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_server_config.py -v`
Expected: PASS — 7 passed

- [ ] **Step 5: Commit**

```bash
git add webapp/server.py tests/test_server_config.py
git commit -m "feat: config->engine builder (layout_from_slots, red-lines 3/4)"
```

---

### Task 7: 回放钻取(parquet 读,Tier A 必做 + Tier B 懒算)

**Files:**
- Create: `webapp/drilldown.py`
- Test: `tests/test_drilldown.py`(新建)

**Interfaces:**
- Consumes: `pyarrow.parquet`(已装)、`webapp.frame`(无,自给)。
- Produces:
  - `frame_at_tick(path: str, tick: int) -> dict` —— 读某 parquet 该 tick 的所有行,聚合成与 `encode_frame` 同形帧(`cells` 每格四阵营 count + `readouts`),供回放某帧。
  - `cell_at_tick(path: str, tick: int, y: int, x: int) -> dict` —— 该 tick 该格逐株 `{strain,faction,count}`(Tier A 点格子)。
  - `strain_trajectory(path: str, strain: str) -> list[dict]` —— Tier B:谓词下推 `strain == X` filtered read,返回 `[{tick, total_count, occupied_cells}, ...]` 按 tick。`# ponytail: 谓词下推,不预建索引;实测真慢再懒建 strain→row_group 轻索引`。memoize 由调用方(server)按 strain 缓存,本函数不缓存(纯)。

红线 1/2:只读已落盘的 playground parquet 真值,不回写、不美化。读数复用 Task 3。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_drilldown.py
import pandas as pd
from webapp.drilldown import frame_at_tick, cell_at_tick, strain_trajectory


def _toy(path):
    # asymmetric cells pin the (y,x) ordering. row = (tick, cell_x, cell_y, strain, faction, count).
    # tick1: cell cx=1,cy=3 -> f0 A=5; cell cx=2,cy=2 -> f1 A=3, f2 B=2.  tick2: cx=1,cy=3 f0 A=7.
    rows = [
        (1, 1, 3, "A", 0, 5), (1, 2, 2, "A", 1, 3), (1, 2, 2, "B", 2, 2),
        (2, 1, 3, "A", 0, 7),
    ]
    df = pd.DataFrame(rows, columns=["tick", "cell_x", "cell_y", "strain", "faction", "count"])
    df.to_parquet(str(path))


def test_frame_at_tick_aggregates_cells(tmp_path):
    p = tmp_path / "r.parquet"; _toy(p)
    fr = frame_at_tick(str(p), tick=1)
    assert fr["tick"] == 1
    # cells rows are [y, x, c0..c3] (same order as encode_frame) -> key by (y, x)
    cells = {(c[0], c[1]): c[2:] for c in fr["cells"]}
    assert sorted(cells.keys()) == sorted([(3, 1), (2, 2)])   # (cy,cx): (3,1) and (2,2)
    assert cells[(2, 2)] == [0, 3, 2, 0]
    assert fr["readouts"]["total"] == 10


def test_frame_at_tick_second_tick(tmp_path):
    p = tmp_path / "r.parquet"; _toy(p)
    fr = frame_at_tick(str(p), tick=2)
    assert len(fr["cells"]) == 1
    assert fr["cells"][0][0] == 3 and fr["cells"][0][1] == 1   # y=3, x=1 ordering pinned
    assert fr["readouts"]["total"] == 7


def test_cell_at_tick_lists_strains(tmp_path):
    p = tmp_path / "r.parquet"; _toy(p)
    d = cell_at_tick(str(p), tick=1, y=2, x=2)
    assert {"strain": "A", "faction": 1, "count": 3} in d["strains"]
    assert {"strain": "B", "faction": 2, "count": 2} in d["strains"]
    assert len(d["strains"]) == 2


def test_cell_at_tick_empty(tmp_path):
    p = tmp_path / "r.parquet"; _toy(p)
    d = cell_at_tick(str(p), tick=2, y=2, x=2)
    assert d["strains"] == []


def test_strain_trajectory_predicate_pushdown(tmp_path):
    p = tmp_path / "r.parquet"; _toy(p)
    traj = strain_trajectory(str(p), strain="A")
    by_tick = {t["tick"]: t for t in traj}
    assert by_tick[1]["total_count"] == 8       # 5 + 3
    assert by_tick[2]["total_count"] == 7
    assert by_tick[1]["occupied_cells"] == 2    # A at (cy3,cx1) + (cy2,cx2)
    # strain B never appears at tick2
    trajB = strain_trajectory(str(p), strain="B")
    assert [t["tick"] for t in trajB] == [1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_drilldown.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'webapp.drilldown'`

- [ ] **Step 3: Write minimal implementation**

`webapp/drilldown.py`:

```python
# webapp/drilldown.py
"""Replay drilldown over recorded playground parquet (viz spec §6).
Tier A (frame_at_tick / cell_at_tick): cheap single-frame reads.
Tier B (strain_trajectory): predicate-pushdown filtered read, no prebuilt index.
Reads recorded truth only — never rewrites, never beautifies (red-lines 1/2)."""
from __future__ import annotations
import pyarrow.parquet as pq
import pyarrow.compute as pc
from webapp.readouts import compute_readouts

NFAC = 4


def _rows(path: str, filt=None):
    tbl = pq.read_table(path, filters=filt) if filt else pq.read_table(path)
    return tbl.to_pydict()


def frame_at_tick(path: str, tick: int) -> dict:
    d = _rows(path, filt=[("tick", "==", tick)])
    n = len(d["tick"])
    by_cell: dict = {}
    for i in range(n):
        y, x = d["cell_y"][i], d["cell_x"][i]
        row = by_cell.setdefault((y, x), [0, 0, 0, 0])
        row[d["faction"][i]] += d["count"][i]
    cells = [[y, x, *row] for (y, x), row in by_cell.items()]
    readouts = compute_readouts(d["cell_x"], d["cell_y"], d["strain"],
                                d["faction"], d["count"])
    return {"tick": int(tick), "cells": cells, "readouts": readouts}


def cell_at_tick(path: str, tick: int, y: int, x: int) -> dict:
    d = _rows(path, filt=[("tick", "==", tick), ("cell_y", "==", y), ("cell_x", "==", x)])
    n = len(d["tick"])
    strains = [{"strain": d["strain"][i], "faction": int(d["faction"][i]),
                "count": int(d["count"][i])} for i in range(n)]
    return {"tick": int(tick), "y": int(y), "x": int(x), "strains": strains}


def strain_trajectory(path: str, strain: str) -> list:
    # ponytail: predicate pushdown, no prebuilt index; lazy-build a
    # strain->row_group index only if profiling proves this too slow.
    d = _rows(path, filt=[("strain", "==", strain)])
    n = len(d["tick"])
    per_tick: dict = {}
    for i in range(n):
        t = int(d["tick"][i])
        e = per_tick.setdefault(t, {"tick": t, "total_count": 0, "_cells": set()})
        e["total_count"] += int(d["count"][i])
        e["_cells"].add((d["cell_y"][i], d["cell_x"][i]))
    out = []
    for t in sorted(per_tick):
        e = per_tick[t]
        out.append({"tick": t, "total_count": e["total_count"],
                    "occupied_cells": len(e["_cells"])})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_drilldown.py -v`
Expected: PASS — 5 passed

- [ ] **Step 5: Commit**

```bash
git add webapp/drilldown.py tests/test_drilldown.py
git commit -m "feat: replay drilldown Tier A + Tier B over playground parquet (spec §6)"
```

---

### Task 8: aiohttp app —— 路由 + WebSocket live loop + playground Recorder

**Files:**
- Modify: `webapp/server.py`(在 Task 6 纯助手之上加 aiohttp app)
- Modify: `.gitignore`(确认 `data/playground/` 被忽略)
- Test: `tests/test_server_config.py`(追加 app 装配 smoke test)

**Interfaces:**
- Consumes: `aiohttp.web`(已装 3.13.3)、Task 5 `encode_frame`、Task 7 `frame_at_tick`/`cell_at_tick`/`strain_trajectory`、`des.recorder.Recorder`、Task 6 `build_engine_from_config`。
- Produces:
  - `PLAYGROUND_DIR = "data/playground"` —— 隔离目录常量(红线 1)。
  - `make_app(device=None) -> web.Application` —— 装配路由:`GET /` → static index.html;`GET /config` → `{"palette":PALETTE, "slots":sorted(_SLOTS), "locked":_LOCKED, "defaults":...}`;`GET /api/frame_at_tick`、`GET /api/cell`、`GET /api/trajectory` → Task 7 函数;`GET /ws` → WebSocket。static 目录挂 `webapp/static`。
  - WebSocket 协议:client 发 `{"cmd":"start", "config":{...}}` → server `build_engine_from_config`、起 playground `Recorder`、跑 `engine.run`-式循环:每 tick `engine.step()` → `recorder.dump` → `encode_frame` → `ws.send_json(frame)`,**算完即发,不限速**(节奏红线)。跑满 `T` 或 client 断开即停、`recorder.close()`。
- 启动:`main()` → `web.run_app(make_app(), port=8000)`。

红线 1:Recorder 路径 = `data/playground/<timestamp>-live.parquet`,绝不写 `data/runs/`。节奏红线:循环内无 sleep/缓冲。

- [ ] **Step 1: Write the failing test**

在 `tests/test_server_config.py` 末尾追加:

```python
def test_make_app_registers_routes():
    from webapp.server import make_app, PLAYGROUND_DIR
    app = make_app(device=torch.device("cpu"))
    paths = {r.resource.canonical for r in app.router.routes() if r.resource is not None}
    assert "/" in paths
    assert "/config" in paths
    assert "/ws" in paths
    assert "/api/frame_at_tick" in paths
    assert "/api/cell" in paths
    assert "/api/trajectory" in paths
    # red-line 1: playground dir is isolated, never data/runs
    assert "playground" in PLAYGROUND_DIR
    assert "runs" not in PLAYGROUND_DIR
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_server_config.py::test_make_app_registers_routes -v`
Expected: FAIL — `ImportError: cannot import name 'make_app' from 'webapp.server'`

- [ ] **Step 3: Write minimal implementation**

`webapp/server.py` 顶部 import 段补:

```python
import os
import datetime
from aiohttp import web
from des.recorder import Recorder
from webapp.frame import encode_frame
from webapp.drilldown import frame_at_tick, cell_at_tick, strain_trajectory

PLAYGROUND_DIR = os.path.join("data", "playground")
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
```

文件末尾追加 app 装配 + 路由 handler + live loop + main:

```python
def _device(device):
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


async def _config(request):
    return web.json_response({
        "palette": PALETTE, "slots": sorted(_SLOTS),
        "locked": {str(k): v for k, v in _LOCKED.items()},
        "defaults": _DEFAULTS,
    })


async def _frame_at_tick(request):
    path = request.query["path"]; tick = int(request.query["tick"])
    return web.json_response(frame_at_tick(path, tick))


async def _cell(request):
    q = request.query
    return web.json_response(cell_at_tick(
        q["path"], int(q["tick"]), int(q["y"]), int(q["x"])))


async def _trajectory(request):
    q = request.query
    return web.json_response(strain_trajectory(q["path"], q["strain"]))


async def _ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    device = request.app["device"]
    async for msg in ws:
        if msg.type != web.WSMsgType.TEXT:
            continue
        data = msg.json()
        if data.get("cmd") != "start":
            continue
        eng, cfg = build_engine_from_config(data.get("config") or {}, device)
        os.makedirs(PLAYGROUND_DIR, exist_ok=True)
        tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(PLAYGROUND_DIR, f"{tag}-live.parquet")
        rec = Recorder(path, eng.table)
        await ws.send_json({"event": "started", "config": _jsonable(cfg), "path": path})
        try:
            for _ in range(int(cfg["T"])):
                if ws.closed:
                    break
                eng.step()
                rec.dump(eng.T, eng.world)
                frame = encode_frame(eng.world, eng.table, eng.T, eng.H, eng.W)
                await ws.send_json(frame)   # engine-speed: no sleep, no buffer
        finally:
            rec.close()
        if not ws.closed:
            await ws.send_json({"event": "done", "path": path})
    return ws


def _jsonable(cfg: dict) -> dict:
    out = dict(cfg)
    out["layout"] = list(cfg["layout"])
    out["slots"] = {str(k): v for k, v in cfg["slots"].items()}
    return out


def make_app(device=None) -> web.Application:
    app = web.Application()
    app["device"] = _device(device)
    app.router.add_get("/config", _config)
    app.router.add_get("/api/frame_at_tick", _frame_at_tick)
    app.router.add_get("/api/cell", _cell)
    app.router.add_get("/api/trajectory", _trajectory)
    app.router.add_get("/ws", _ws)
    app.router.add_get("/", lambda r: web.FileResponse(os.path.join(_STATIC_DIR, "index.html")))
    app.router.add_static("/static", _STATIC_DIR)
    return app


def main() -> None:
    web.run_app(make_app(), port=8000)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_server_config.py -v`
Expected: PASS — 8 passed(含新 smoke test)。

- [ ] **Step 5: 确认 gitignore 忽略 playground parquet**

`.gitignore` 已有 `*.parquet`(全局忽略所有 parquet),playground 自动覆盖。无需改动则跳过;若要显式标注,在 `# --- Digital Evolution Sandbox project-specific ---` 段下加一行:

```
# live playground parquet (isolated from data/runs formal collection)
data/playground/
```

- [ ] **Step 6: Commit**

```bash
git add webapp/server.py tests/test_server_config.py .gitignore
git commit -m "feat: aiohttp app — routes + WebSocket live loop + playground recorder (red-line 1)"
```

---

### Task 9: 前端单页(无框架,人工验收)

**Files:**
- Create: `webapp/static/index.html`
- Create: `webapp/static/style.css`
- Create: `webapp/static/app.js`

**Interfaces:**
- Consumes: `GET /config`(palette/slots/locked/defaults)、`GET /ws`(live 帧)、`GET /api/cell`/`/api/trajectory`/`/api/frame_at_tick`(钻取/回放)。
- Produces: 无代码接口(浏览器人工验收,spec §8,**不引前端测试框架**)。

前端复刻已批准的 mockup(`docs/mockups/main-screen.html`)布局与合成逻辑,把假数据换成真 WS 帧:左=配置栏(genome 16 位 + 全局参 + 开始)、中=验收之眼 128² canvas + 图例 + transport、右=读数(数字牌/占比图/多样性图/该格明细)+ 折叠的探索分析/数据展示。合成:`ImageData` 单图层,阵营 count 加权混色(色相)+ 总密度→alpha(over black),争夺格自然两色混。

验收清单(人工,跑起来肉眼核):① 四象限扩张→前沿相遇→填满→四阵营共存;② 占比四线全贴 0.25 无某条冲顶;③ 点格子弹出该格 `{strain:count,faction}`;④ 配置 locked 位置灰只读、6 槽可改、默认全 N0;⑤ 前沿无平滑(像素硬边,守红线 2)。

- [ ] **Step 1: 写 `webapp/static/index.html`**

```html
<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>DES 验收之眼</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body>
<div class="app">
  <aside class="rail" id="rail">
    <button class="railbtn" onclick="document.getElementById('rail').classList.toggle('collapsed')">⚙</button>
    <h2><span>配置(对称局)</span></h2>
    <div class="body">
      <div class="note">这是<b>对称起始基因型</b>(四阵营全同)<br>不是角色/阵营差异系统</div>
      <h2 style="margin:0 0 6px"><span>BB0 基因型(16 位)</span></h2>
      <div class="genome" id="genome"></div>
      <div style="font-size:11px;color:var(--dim);margin-bottom:10px">
        灰=锁死功能位&nbsp; 蓝=可选插槽</div>
      <h2 style="margin:8px 0 6px"><span>全局参数</span></h2>
      <div class="row"><span>网格</span><input id="cfg-grid" value="128"></div>
      <div class="row"><span>K</span><input id="cfg-K" value="64"></div>
      <div class="row"><span>fill/格</span><input id="cfg-fill" value="20"></div>
      <div class="row"><span>T (ticks)</span><input id="cfg-T" value="450"></div>
      <div class="row"><span>seed</span><input id="cfg-seed" value="0"></div>
      <div class="row"><span>z_max</span><input id="cfg-zmax" value="8.0"></div>
      <button class="go" id="startBtn">▶ 开始游戏</button>
    </div>
  </aside>

  <main class="stage">
    <div class="stagehead">
      <span class="title">验收之眼 — <span id="gridLabel">128×128</span> 世界</span>
      <div class="legend">
        <span><i style="background:var(--f0)"></i>阵营0</span>
        <span><i style="background:var(--f1)"></i>阵营1</span>
        <span><i style="background:var(--f2)"></i>阵营2</span>
        <span><i style="background:var(--f3)"></i>阵营3</span>
        <span>亮度=密度&nbsp;·&nbsp;混色=争夺格</span>
      </div>
    </div>
    <div class="canvaswrap">
      <canvas id="grid" width="128" height="128"></canvas>
    </div>
    <div class="transport">
      <span class="ticklabel" id="tickLabel">tick 0 / 0</span>
      <span id="status" style="color:var(--dim)">未开始</span>
      <span style="color:var(--dim);margin-left:auto">点格子 → 看 {strain:count}</span>
    </div>
  </main>

  <aside class="read">
    <div class="cards">
      <div class="card"><div class="k">tick</div><div class="v" id="m-tick">0</div></div>
      <div class="card"><div class="k">总个体</div><div class="v" id="m-total">0</div></div>
      <div class="card"><div class="k">占用格</div><div class="v" id="m-occ">0</div></div>
      <div class="card"><div class="k">活株</div><div class="v" id="m-distinct">0</div></div>
      <div class="card"><div class="k">N2</div><div class="v" id="m-n2">0</div></div>
      <div class="card"><div class="k">d_max</div><div class="v" id="m-dmax">0</div></div>
    </div>
    <h3>阵营占比(主判决曲线 → 全贴 0.25=共存)</h3>
    <canvas class="chart" id="shareChart"></canvas>
    <h3>多样性(distinct / N2)</h3>
    <canvas class="chart" id="divChart"></canvas>
    <h3>主导株排行(top 5)</h3>
    <ul class="lead" id="lead"></ul>
    <h3>该格明细(点格子)</h3>
    <ul class="lead" id="cellDetail"><li style="color:var(--dim)">点中心 canvas 的格子查看</li></ul>
    <details class="foldpanel"><summary>探索分析(钻取)</summary>
      <div style="font-size:11px;color:var(--dim);padding:8px 0">
        点格子看该 tick {strain:count};跨 tick 株轨迹经 /api/trajectory 懒算。</div>
    </details>
    <details class="foldpanel"><summary>数据产品展示</summary>
      <div style="font-size:11px;color:var(--dim);padding:8px 0" id="dataInfo">
        本局 → parquet(playground 隔离目录)。开始后显示路径。</div>
    </details>
  </aside>
</div>
<script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: 写 `webapp/static/style.css`**(复刻 mockup 样式)

```css
:root{
  --bg:#0d1117; --panel:#161b22; --line:#30363d; --txt:#c9d1d9; --dim:#8b949e;
  --f0:#ff4d4d; --f1:#4d9bff; --f2:#3fd07f; --f3:#ffcc4d;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
     font:13px/1.4 ui-sans-serif,system-ui,"Segoe UI",sans-serif}
.app{display:grid;grid-template-columns:auto 1fr 320px;height:100vh}
.rail{background:var(--panel);border-right:1px solid var(--line);
      width:260px;transition:width .15s;overflow:hidden}
.rail.collapsed{width:40px}
.rail h2{font-size:12px;letter-spacing:.05em;color:var(--dim);
         text-transform:uppercase;margin:14px 12px 8px}
.rail .body{padding:0 12px 16px}
.collapsed .body, .collapsed h2 span{display:none}
.railbtn{width:40px;height:40px;background:none;border:0;color:var(--txt);
         cursor:pointer;font-size:16px}
.note{font-size:11px;color:var(--dim);background:#1f2937;border:1px solid var(--line);
      border-radius:6px;padding:6px 8px;margin-bottom:12px}
.genome{display:grid;grid-template-columns:repeat(8,1fr);gap:3px;margin-bottom:10px}
.pos{aspect-ratio:1;border:1px solid var(--line);border-radius:4px;display:flex;
     align-items:center;justify-content:center;font-size:9px;text-align:center;padding:1px}
.pos.locked{background:#21262d;color:var(--dim);cursor:not-allowed}
.pos.slot{background:#1c2733;color:#79c0ff;cursor:pointer}
.pos.slot select{background:#0d1117;border:0;color:#79c0ff;font-size:9px;width:100%;cursor:pointer}
.row{display:flex;justify-content:space-between;align-items:center;margin:6px 0}
.row input{background:#0d1117;border:1px solid var(--line);color:var(--txt);
     border-radius:4px;padding:2px 6px;width:90px}
.go{width:100%;margin-top:12px;padding:9px;background:#238636;border:0;border-radius:6px;
    color:#fff;font-weight:600;cursor:pointer;font-size:14px}
.stage{display:flex;flex-direction:column;min-width:0}
.stagehead{display:flex;align-items:center;gap:14px;padding:10px 16px;
           border-bottom:1px solid var(--line)}
.stagehead .title{font-weight:600}
.legend{display:flex;gap:12px;margin-left:auto;font-size:11px;color:var(--dim)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;
          vertical-align:-1px}
.canvaswrap{flex:1;display:flex;align-items:center;justify-content:center;padding:16px;
            min-height:0}
#grid{image-rendering:pixelated;background:#000;border:1px solid var(--line);
      width:min(72vh,100%);height:min(72vh,100%);aspect-ratio:1;cursor:crosshair}
.transport{display:flex;align-items:center;gap:12px;padding:10px 16px;
           border-top:1px solid var(--line)}
.ticklabel{font-variant-numeric:tabular-nums;color:var(--dim);min-width:120px}
.read{background:var(--panel);border-left:1px solid var(--line);overflow-y:auto;padding:14px}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px}
.card{background:#0d1117;border:1px solid var(--line);border-radius:6px;padding:8px 10px}
.card .k{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:18px;font-weight:600;font-variant-numeric:tabular-nums}
.read h3{font-size:11px;letter-spacing:.05em;color:var(--dim);text-transform:uppercase;
         margin:16px 0 6px}
.chart{width:100%;height:90px;background:#0d1117;border:1px solid var(--line);border-radius:6px}
.lead{list-style:none;padding:0;margin:0;font-size:12px}
.lead li{display:flex;align-items:center;gap:6px;padding:3px 0}
.lead .seq{font-family:ui-monospace,monospace;font-size:10px;color:var(--dim);
           white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:160px}
.foldpanel{border-top:1px solid var(--line);margin-top:14px;padding-top:10px}
.foldpanel summary{cursor:pointer;color:var(--dim);font-size:11px;
     text-transform:uppercase;letter-spacing:.05em}
```


- [ ] **Step 3: 写 `webapp/static/app.js`**(配置渲染 + WS live + 合成 + 手绘图 + 点格子钻取)

```javascript
// webapp/static/app.js — DES 验收之眼前端。无框架。
const FC = [[255, 77, 77], [77, 155, 255], [63, 208, 127], [255, 204, 77]];
const $ = (id) => document.getElementById(id);

let CFG = null;            // /config 返回(palette/slots/locked/defaults)
const slotState = {};      // slot index -> chosen primitive
let ws = null, livePath = null;
const shareSeries = [[], [], [], []];   // per-faction share over ticks
const distinctSeries = [], n2Series = [];

// ---- build the 16-position genome grid from /config ----
async function loadConfig() {
  CFG = await (await fetch("/config")).json();
  const g = $("genome");
  const locked = CFG.locked;             // {"1":"F4Nr4",...} string keys
  const slots = new Set(CFG.slots);      // [0,2,3,9,10,13]
  for (let i = 0; i < 16; i++) {
    const d = document.createElement("div");
    if (locked[String(i)] !== undefined) {
      d.className = "pos locked"; d.textContent = locked[String(i)];
    } else if (slots.has(i)) {
      d.className = "pos slot";
      const sel = document.createElement("select");
      CFG.palette.forEach((p) => {
        const o = document.createElement("option"); o.value = p; o.textContent = p; sel.appendChild(o);
      });
      sel.value = "N0"; slotState[i] = "N0";
      sel.onchange = () => { slotState[i] = sel.value; };
      d.appendChild(sel);
    } else {
      d.className = "pos locked"; d.textContent = "N0";   // backbone-fixed
    }
    g.appendChild(d);
  }
}

function readConfig() {
  return {
    slots: { ...slotState },
    grid: +$("cfg-grid").value, K: +$("cfg-K").value, fill: +$("cfg-fill").value,
    T: +$("cfg-T").value, seed: +$("cfg-seed").value, z_max: +$("cfg-zmax").value,
  };
}

// ---- single-layer compositing: faction-count-weighted hue + density alpha ----
let imgW = 128, img = null, ctx = null;
function setupCanvas(n) {
  const cv = $("grid"); cv.width = n; cv.height = n; imgW = n;
  ctx = cv.getContext("2d"); img = ctx.createImageData(n, n);
}
function drawFrame(frame) {
  const n = frame.H;
  if (n !== imgW || !img) setupCanvas(n);
  img.data.fill(0);
  for (let i = 3; i < img.data.length; i += 4) img.data[i] = 255;   // opaque black
  let kCap = +$("cfg-K").value || 64;
  for (const c of frame.cells) {
    const y = c[0], x = c[1], cf = [c[2], c[3], c[4], c[5]];
    const tot = cf[0] + cf[1] + cf[2] + cf[3];
    if (tot <= 0) continue;
    let r = 0, gg = 0, bb = 0;
    for (let f = 0; f < 4; f++) { const w = cf[f] / tot; r += FC[f][0] * w; gg += FC[f][1] * w; bb += FC[f][2] * w; }
    const dens = Math.min(1, tot / kCap), a = 0.15 + 0.85 * dens;   // density -> brightness
    const o = (y * n + x) * 4;
    img.data[o] = r * a; img.data[o + 1] = gg * a; img.data[o + 2] = bb * a; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// ---- readout panel + hand-drawn charts (no chart lib) ----
function line(id, series, colors, ymax) {
  const cv = $(id), c = cv.getContext("2d");
  cv.width = cv.clientWidth; cv.height = cv.clientHeight;
  const W = cv.width, H = cv.height; c.clearRect(0, 0, W, H);
  series.forEach((s, si) => {
    if (!s.length) return;
    c.strokeStyle = colors[si]; c.lineWidth = 1.5; c.beginPath();
    s.forEach((v, i) => {
      const px = s.length > 1 ? i / (s.length - 1) * W : 0;
      const py = H - (v / ymax) * H * 0.92 - 2;
      i ? c.lineTo(px, py) : c.moveTo(px, py);
    });
    c.stroke();
  });
}
function updateReadouts(frame) {
  const r = frame.readouts;
  $("m-tick").textContent = frame.tick;
  $("m-total").textContent = r.total;
  $("m-occ").textContent = r.occupied_cells;
  $("m-distinct").textContent = r.distinct_strains;
  $("m-n2").textContent = r.n2.toFixed(1);
  $("m-dmax").textContent = r.d_max.toFixed(3);
  $("tickLabel").textContent = `tick ${frame.tick} / ${readConfig().T}`;
  for (let f = 0; f < 4; f++) shareSeries[f].push(r.faction_share[f] || 0);
  distinctSeries.push(r.distinct_strains); n2Series.push(r.n2);
  line("shareChart", shareSeries, ["#ff4d4d", "#4d9bff", "#3fd07f", "#ffcc4d"], 0.6);
  const dmax = Math.max(1, ...distinctSeries), nmax = Math.max(1, ...n2Series);
  line("divChart",
    [distinctSeries.map((v) => v / dmax), n2Series.map((v) => v / nmax)],
    ["#a371f7", "#39c5cf"], 1.0);
  renderLeaderboard(frame.leaderboard || []);
}

// ---- 主导株排行榜 (spec §4) ----
function renderLeaderboard(lb) {
  const ul = $("lead"); ul.innerHTML = "";
  lb.forEach((e) => {
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="seq" title="${e.strain}">${e.strain}</span>` +
      `<span style="color:rgb(${FC[e.faction]})">f${e.faction}</span>` +
      `<span style="margin-left:auto;font-variant-numeric:tabular-nums">${(e.share * 100).toFixed(1)}%</span>`;
    ul.appendChild(li);
  });
}

// ---- WebSocket live loop ----
function start() {
  if (ws) ws.close();
  for (const s of shareSeries) s.length = 0;
  distinctSeries.length = 0; n2Series.length = 0;
  $("gridLabel").textContent = `${readConfig().grid}×${readConfig().grid}`;
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { $("status").textContent = "运行中"; ws.send(JSON.stringify({ cmd: "start", config: readConfig() })); };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.event === "started") { livePath = m.path; $("dataInfo").textContent = `本局 → ${m.path}`; return; }
    if (m.event === "done") { $("status").textContent = "完成"; return; }
    drawFrame(m); updateReadouts(m);
  };
  ws.onclose = () => { if ($("status").textContent === "运行中") $("status").textContent = "已停止"; };
}

// ---- Tier A drilldown: click a cell -> that tick's {strain:count,faction} ----
function onCanvasClick(ev) {
  if (!livePath) return;
  const cv = $("grid"), rect = cv.getBoundingClientRect();
  const x = Math.floor((ev.clientX - rect.left) / rect.width * imgW);
  const y = Math.floor((ev.clientY - rect.top) / rect.height * imgW);
  const tick = +$("m-tick").textContent;
  fetch(`/api/cell?path=${encodeURIComponent(livePath)}&tick=${tick}&y=${y}&x=${x}`)
    .then((r) => r.json()).then((d) => {
      const ul = $("cellDetail"); ul.innerHTML = "";
      if (!d.strains.length) { ul.innerHTML = `<li style="color:var(--dim)">格 (${y},${x}) 空</li>`; return; }
      d.strains.forEach((s) => {
        const li = document.createElement("li");
        li.innerHTML = `<span class="seq" title="${s.strain}">${s.strain}</span>` +
          `<span style="color:rgb(${FC[s.faction]})">f${s.faction}</span>` +
          `<span style="margin-left:auto;font-variant-numeric:tabular-nums">${s.count}</span>`;
        ul.appendChild(li);
      });
    });
}

window.addEventListener("DOMContentLoaded", () => {
  loadConfig();
  $("startBtn").onclick = start;
  $("grid").addEventListener("click", onCanvasClick);
});
```

- [ ] **Step 4: 人工验收(无自动化测试,spec §8)**

启动(PowerShell,从 repo 根):

```powershell
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe webapp/server.py
```

浏览器开 `http://localhost:8000`,点「开始游戏」,核验收清单 ①–⑤。128² 满世界约 2–5 tick/秒,整局一两分钟跑完。

- [ ] **Step 5: Commit**

```bash
git add webapp/static/index.html webapp/static/style.css webapp/static/app.js
git commit -m "feat: single-page frontend — 验收之眼 live canvas + readouts + drilldown"
```

---

### Task 10: 启动文档 + 全套回归验证闸

**Files:**
- Create: `webapp/README.md`
- Test: 无新测(集成闸,跑全套)

**Interfaces:**
- Consumes: 前 9 个 task 的全部产出。
- Produces: 无代码;一份启动说明 + 一次全绿验证。

- [ ] **Step 1: 写 `webapp/README.md`**

```markdown
# DES 可视化 Web App

实时游戏界面(验收之眼):选 BB0 对称局起始基因型 → 四阵营从四象限扩张 →
肉眼看红皇后共存动力学。后端 aiohttp 跑现有 Engine 流帧,前端纯 HTML/CSS/JS。

## 启动

从 repo 根(PowerShell):

```powershell
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe webapp/server.py
```

浏览器开 http://localhost:8000。

## 数据

每局边跑边写 parquet 到 `data/playground/<timestamp>-live.parquet`(隔离目录,
gitignore;绝不与 `data/runs/` 正式采集混池)。schema 同正式 run:
`(tick, cell_x, cell_y, strain, faction, count)`。

## 红线

对称局四阵营同条 layout;locked 位只读、只 6 槽可改;live 帧忠实张量(无平滑);
读数与 `scripts/analyze_batch.py` 共用 `webapp/readouts.py` 单一来源。
```

- [ ] **Step 2: 提交 README**

```bash
git add webapp/README.md
git commit -m "docs: webapp start command + data isolation note"
```

- [ ] **Step 3: 全套回归 —— 确认引擎触点零回归 + 新模块全绿**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest -q`
Expected: PASS — 既有 285 测试无回归 + 本计划新增(test_world_layout / test_readouts / test_frame / test_server_config / test_drilldown)全过。若任何既有测试因引擎触点变红,回 Task 2 修(默认 layout 行为必须完全不变)。

- [ ] **Step 4: live 端到端人工冒烟(一次)**

按 Task 9 Step 4 启动,点开始,跑完一局(默认 128²/T=450)。肉眼核:四象限扩张→相遇→填满→四阵营占比贴 0.25;确认 `data/playground/` 下生成了本局 parquet 且 `data/runs/` 未被写入(红线 1)。

- [ ] **Step 5: 最终提交(若有未决的 README/收尾)**

```bash
git add -A
git commit -m "chore: viz web app complete — 验收之眼 live + replay drilldown"
```

---

## 验收对照(spec → task 映射,自检用)

| spec 节 | 要求 | task |
|---|---|---|
| §2 | aiohttp 后端 + WS + 前端单图层合成 | 8, 9 |
| §3(a) | genome 配置:locked 只读 / 6 槽 / 默认 N0 | 6, 9 |
| §3(b) | 全局参(grid/K/fill/T/seed/z_max) | 6, 9 |
| §4 | 读数(数字牌/占比线/多样性线/主导株排行榜) | 3, 5, 9 |
| §4 红线 | 读数单一来源 | 3, 4 |
| §5 | init_factions layout 参 + 守门校验 | 1, 2 |
| §6 Tier A | 点格子单帧 + 回放帧 | 5, 7, 9 |
| §6 Tier B | 跨 tick 株轨迹懒算 | 7 |
| §7 红线1 | playground 数据隔离 | 8 |
| §7 红线2 | live 帧忠实张量 | 5 |
| §7 红线3 | 无手写谁强 | 6 |
| §7 红线4 | 四阵营对称不变量 | 1, 2 |
| §8 | 帧/读数/校验单测 + 前端人工验收 | 3,5,1,2 / 9 |

---

## Execution Notes

- **环境铁律:** 全程 `D:/anaconda3/envs/basic/python.exe`,绝不 pip/conda install(aiohttp 已在 env)。
- **commit 粒度:** 每 task 末尾一 commit(已在各 Step 5 写明)。
- **引擎触点零回归是硬闸:** Task 2 改 `init_factions`/`Engine` 后,`test_world.py`/`test_engine.py`/全套必须仍绿。
- **前端无自动化测试是 YAGNI 决定(spec §8),非遗漏:** demo 人工验收,验收清单在 Task 9。
