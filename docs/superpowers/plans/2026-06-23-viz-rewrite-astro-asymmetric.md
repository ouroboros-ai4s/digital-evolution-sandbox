# 验收之眼 v2 — Astro 重写 + 放开四阵营对称 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 V3 mockup 搬进真应用,放开引擎使其能真跑「四阵营各自独立基因型」,前端用 Astro 重写,由 aiohttp 托管。

**Architecture:** 引擎 `init_factions` 接受 4 条 layout(保留单条向后兼容);server 协议从单 `slots` 改为 `players:[{slots}×4]`,4 条 layout inline 生成、各自过现有守门;前端 Astro `output:'static'` 构建 `dist/`,渲染/读数逻辑从现 `app.js` 逐字搬运(守红线 2),由 aiohttp 单端口托管。

**Tech Stack:** Python 3 / torch / aiohttp / pytest(后端);Astro 4 + 原生 ES 模块 + Canvas 2D(前端,无 UI 框架)。

## Global Constraints

本轮主动放开 red-line 4 的「四阵营基因型全同」约束(降级为「同模板结构」),其余四条红线原样守死。每个 task 的要求隐含包含本节:

- **① playground 数据隔离:** 仍写 `data/playground/`,绝不碰 `data/runs/`。
- **② live 帧忠实张量、无平滑:** `drawFrame` 合成逐字搬运(faction-count 加权混色 + density alpha + `image-rendering:pixelated` 最近邻),不引入任何插值。
- **③ 配置只暴露基元字母 + run 参:** `/config` 仍只给 `palette`(6 字母)/`slots`/`locked`/`defaults`,无任何「谁强」系数。
- **④ 四阵营对称:** 放开「全同」为「同模板结构」——四阵营共用同一 BB0 模板(锁死位、骨架位、插槽位置与调色板一致),仅各自插槽取值可不同;四中心仍是 D4 对称轨道(无几何偏袒),仍走同一固定 G→P 函数;严禁手写「谁强」。
- **⑤ 读数单一来源:** `webapp/readouts.py`、`webapp/frame.py` 的 `encode_frame`/`cell_detail` 一字不改;known-answer 回归测试必须仍绿。
- **环境:** `D:\anaconda3\envs\basic\python.exe`;测试 `$env:PYTHONPATH='src'; python -m pytest`;前端 `cd webapp/frontend && npm install && npm run build`。

---
### Task 1: 引擎放开 red-line 4 —— `init_factions` 接受四条 layout

**Files:**
- Modify: `src/des/world.py:35-58` (`init_factions`)
- Modify: `src/des/engine.py:14-21` (`Engine.__init__`)
- Test: `tests/test_world_layout.py` (追加测试,现有测试保持绿)

**Interfaces:**
- Consumes: `validate_bb0_layout(layout)`、`BB0_TEMPLATE["layout"]`、`StrainTable.get_or_mint(layout)`(均已存在,不改)。
- Produces:
  - `init_factions(H, W, K, device, table, fill_per_cell, n_fac=4, layout=None, layouts=None) -> World`
    —— `layouts` 是 4 条 layout 的序列;`layout` 单条(四阵营全同,向后兼容);二者同时非 None → `ValueError`;`layouts` 长度≠4 → `ValueError`;每条独立过 `validate_bb0_layout`;第 `fac` 中心注入第 `fac` 条。
  - `Engine(..., layout=None, layouts=None)` —— `layouts` 透传给 `init_factions`。

- [ ] **Step 1: 写失败测试(四条不同 layout + 互斥 + 长度 + 非法第 N 条 + Engine 透传)**

在 `tests/test_world_layout.py` 末尾追加:

```python
def _four_distinct():
    """Four layouts differing only in slot 0 (a legal mutable slot)."""
    out = []
    for letter in ("N0", "F4Nr1", "P_base", "P_hotspot"):
        lay = _canonical(); lay[0] = letter
        out.append(tuple(lay))
    return out


def test_init_factions_four_distinct_layouts():
    t = StrainTable()
    layouts = _four_distinct()
    w = init_factions(8, 8, 16, DEV, t, fill_per_cell=10, n_fac=4, layouts=layouts)
    centers = [(2, 2), (2, 6), (6, 2), (6, 6)]
    seen_strains = set()
    for fac, (cy, cx) in enumerate(centers):
        expect = t.get_or_mint(layouts[fac])
        assert int(w.strain_id[cy, cx, 0]) == expect      # fac-th center = fac-th layout
        assert int(w.faction[cy, cx, 0]) == fac
        seen_strains.add(int(w.strain_id[cy, cx, 0]))
    assert len(seen_strains) == 4                          # four genuinely different strains


def test_init_factions_layout_and_layouts_mutually_exclusive():
    t = StrainTable()
    with pytest.raises(ValueError, match="not both"):
        init_factions(8, 8, 16, DEV, t, fill_per_cell=10,
                      layout=BB0_TEMPLATE["layout"], layouts=_four_distinct())


def test_init_factions_layouts_must_be_exactly_four():
    t = StrainTable()
    three = _four_distinct()[:3]
    with pytest.raises(ValueError, match="exactly 4"):
        init_factions(8, 8, 16, DEV, t, fill_per_cell=10, layouts=three)


def test_init_factions_rejects_tampered_nth_layout():
    t = StrainTable()
    layouts = _four_distinct()
    bad = list(layouts[2]); bad[1] = "N0"        # tamper faction-2's locked position
    layouts[2] = tuple(bad)
    with pytest.raises(ValueError, match="locked"):
        init_factions(8, 8, 16, DEV, t, fill_per_cell=10, layouts=layouts)


def test_engine_passes_layouts_through():
    layouts = _four_distinct()
    e = Engine(H=8, W=8, K=16, seed=0, device=DEV, fill_per_cell=10, layouts=layouts)
    centers = [(2, 2), (2, 6), (6, 2), (6, 6)]
    for fac, (cy, cx) in enumerate(centers):
        assert int(e.world.strain_id[cy, cx, 0]) == e.table.get_or_mint(layouts[fac])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_world_layout.py -k "layouts or four_distinct" -v`
Expected: FAIL —— `init_factions() got an unexpected keyword argument 'layouts'`(及 `Engine` 同样)。

- [ ] **Step 3: 改 `init_factions`(`src/des/world.py`)**

把 `init_factions`(35–58 行)整体替换为:

```python
def init_factions(H: int, W: int, K: int, device: torch.device,
                  table: StrainTable, fill_per_cell: int, n_fac: int = 4,
                  layout: tuple[str, ...] | None = None,
                  layouts: "tuple[tuple[str, ...], ...] | None" = None) -> World:
    """Seed BB0 at the four quadrant centers, one faction each, everything else empty.
    The four centers are the D4-symmetric orbit of one point (equal to grid center,
    equal nearest-wall distance, pairwise-symmetric) → no faction gets a geometric edge.

    Genotype seeding (red-line 4, downgraded "全同"→"同模板结构"):
      - layouts: a sequence of exactly 4 layouts, one per faction (asymmetric start).
      - layout: a single layout shared by all four factions (backward-compatible).
      - both None → canonical BB0_TEMPLATE["layout"] shared by all four (default unchanged).
    Passing both layout and layouts is a ValueError. Every layout is independently
    gatekeeper-validated (same fixed template structure; only slot choices may differ)."""
    assert fill_per_cell <= K, "fill must fit in K slots"
    assert n_fac == 4, "v1 seeds exactly 4 factions at the 4 quadrant centers"
    if layouts is not None and layout is not None:
        raise ValueError("pass either layout (single, shared) or layouts (4), not both")
    if layouts is None:
        single = BB0_TEMPLATE["layout"] if layout is None else layout
        layouts = (single, single, single, single)
    if len(layouts) != 4:
        raise ValueError(f"layouts must have exactly 4 entries, got {len(layouts)}")
    for lay in layouts:
        validate_bb0_layout(lay)
    w = World(H, W, K, device)
    centers = [(H // 4, W // 4), (H // 4, 3 * W // 4),
               (3 * H // 4, W // 4), (3 * H // 4, 3 * W // 4)]
    for fac, (cy, cx) in enumerate(centers):
        bb0_f = table.get_or_mint(layouts[fac])
        w.strain_id[cy, cx, 0] = bb0_f
        w.count[cy, cx, 0] = fill_per_cell
        w.faction[cy, cx, 0] = fac
    return w
```

- [ ] **Step 4: 改 `Engine.__init__`(`src/des/engine.py`)**

把 14–21 行的签名与 `init_factions` 调用改为:

```python
    def __init__(self, H, W, K, seed, device, z_max=8.0, fill_per_cell=None,
                 check_every=10, layout=None, layouts=None):
        self.H, self.W, self.K, self.device, self.z_max = H, W, K, device, z_max
        self.check_every = check_every
        self.table = StrainTable()
        fill = K // 2 if fill_per_cell is None else fill_per_cell
        self.world = init_factions(H, W, K, device, self.table,
                                   fill_per_cell=fill, n_fac=NFAC,
                                   layout=layout, layouts=layouts)
```

- [ ] **Step 5: 跑新测试 + 全 layout 回归确认通过**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_world_layout.py -v`
Expected: PASS —— 新 5 个 + 现有 9 个(含 `test_init_factions_default_layout_unchanged`、`test_engine_default_layout_unchanged` 等向后兼容用例)全绿。

- [ ] **Step 6: 提交**

```bash
git add src/des/world.py src/des/engine.py tests/test_world_layout.py
git commit -m "feat(engine): init_factions accepts 4 per-faction layouts (open red-line 4 全同→同模板结构)"
```

---
### Task 2: server 协议 —— 单 slots → 四 players

**Files:**
- Modify: `webapp/server.py:47-64` (`build_engine_from_config`)、`webapp/server.py:139-143` (`_jsonable`)、`webapp/server.py:106-136` (`_ws` 错误回传)
- Test: `tests/test_server_config.py` (追加测试,现有测试保持绿)

**Interfaces:**
- Consumes: `layout_from_slots(slots)`(不改)、Task 1 的 `Engine(layouts=...)`、`_DEFAULTS`、`encode_frame`(不改)。
- Produces:
  - `build_engine_from_config(cfg, device) -> (Engine, resolved)`,`cfg` 新结构:
    `{ players:[{slots:{int:str}} ×4], grid,K,fill,T,seed,z_max }`;`players`≠4 → `ValueError`;
    `resolved` 含 `players`(原样回带)与 `layouts`(4 条 tuple)。
  - `_jsonable(cfg)` 序列化 `layouts`(4 条 → list of list)与 `players`。
  - `_ws`:`build_engine_from_config` 抛 `ValueError` → 回 `{"event":"error","msg":str(e)}`,不开跑。

- [ ] **Step 1: 写失败测试(四 players、长度校验、非法插槽、全局共享、jsonable)**

在 `tests/test_server_config.py` 末尾追加(并在文件顶部 import 处加 `from webapp.server import _jsonable`):

```python
def _four_players(slots_list):
    return {"players": [{"slots": s} for s in slots_list]}


def test_build_engine_four_distinct_players():
    cfg_in = {**_four_players([{0: "N0"}, {0: "F4Nr1"}, {0: "P_base"}, {0: "P_hotspot"}]),
              "grid": 16, "K": 8, "fill": 4, "seed": 1}
    eng, cfg = build_engine_from_config(cfg_in, DEV)
    assert len(cfg["layouts"]) == 4
    centers = [(4, 4), (4, 12), (12, 4), (12, 12)]      # quadrant centers of 16-grid
    seen = set()
    for fac, (cy, cx) in enumerate(centers):
        assert int(eng.world.strain_id[cy, cx, 0]) == eng.table.get_or_mint(cfg["layouts"][fac])
        seen.add(int(eng.world.strain_id[cy, cx, 0]))
    assert len(seen) == 4                               # four genuinely different seeds


def test_build_engine_requires_exactly_four_players():
    with pytest.raises(ValueError, match="4 players"):
        build_engine_from_config(_four_players([{}, {}, {}]), DEV)   # only 3


def test_build_engine_rejects_illegal_slot_in_a_player():
    cfg_in = _four_players([{}, {}, {4: "F4Nr1"}, {}])   # pos 4 is backbone, not a slot
    with pytest.raises(ValueError, match="slot"):
        build_engine_from_config(cfg_in, DEV)


def test_build_engine_global_params_shared_not_per_player():
    eng, cfg = build_engine_from_config(
        {**_four_players([{}, {}, {}, {}]), "grid": 32, "K": 16}, DEV)
    assert cfg["grid"] == 32 and cfg["K"] == 16          # single global value
    assert eng.H == 32 and eng.K == 16


def test_jsonable_serializes_players_and_layouts():
    _, cfg = build_engine_from_config(_four_players([{}, {}, {}, {}]), DEV)
    j = _jsonable(cfg)
    assert isinstance(j["layouts"], list) and len(j["layouts"]) == 4
    assert all(isinstance(lay, list) for lay in j["layouts"])
    assert isinstance(j["players"], list) and len(j["players"]) == 4
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_server_config.py -k "four or players or jsonable_serializes" -v`
Expected: FAIL —— `build_engine_from_config` 现读 `cfg["slots"]`,无 `players`/`layouts`。

- [ ] **Step 3: 改 `build_engine_from_config`(`webapp/server.py:47-64`)**

整体替换为(单条 layout 走 `layout_from_slots`,4 条 inline 生成 —— 不另立命名函数):

```python
def build_engine_from_config(cfg: dict, device) -> tuple[Engine, dict]:
    """cfg keys: players (list of exactly 4 {slots:{int:str}}), grid, K, fill,
    T, seed, z_max (globals shared across all four factions). Returns (engine,
    resolved) where resolved has defaults filled, a 'players' echo, and a 4-tuple
    'layouts'. Each layout -> layout_from_slots -> validate_bb0_layout enforces the
    template structure (red-line 4: same template, slot choices may differ); only
    primitive letters + run knobs ever enter here (red-line 3)."""
    players = cfg.get("players")
    if not isinstance(players, list) or len(players) != 4:
        raise ValueError(f"config must have exactly 4 players, got {players!r}")
    layouts = tuple(
        layout_from_slots({int(k): v for k, v in (p.get("slots") or {}).items()})
        for p in players
    )
    resolved = dict(_DEFAULTS)
    for k in _DEFAULTS:
        if cfg.get(k) is not None:
            resolved[k] = cfg[k]
    resolved["players"] = players
    resolved["layouts"] = layouts
    g = int(resolved["grid"])
    eng = Engine(H=g, W=g, K=int(resolved["K"]), seed=int(resolved["seed"]),
                 device=device, z_max=float(resolved["z_max"]),
                 fill_per_cell=int(resolved["fill"]), layouts=layouts)
    return eng, resolved
```

- [ ] **Step 4: 改 `_jsonable`(`webapp/server.py:139-143`)**

替换为:

```python
def _jsonable(cfg: dict) -> dict:
    out = dict(cfg)
    out["layouts"] = [list(lay) for lay in cfg["layouts"]]
    out["players"] = [{"slots": {str(k): v for k, v in (p.get("slots") or {}).items()}}
                      for p in cfg["players"]]
    return out
```

- [ ] **Step 5: 改 `_ws` 错误回传(`webapp/server.py:114-122` 区段)**

把 `_ws` 里 `data.get("cmd") != "start"` 判断之后、`build_engine_from_config` 调用一段改为(用 try 包住 build,失败回 error 事件、continue 不开跑):

```python
        if data.get("cmd") != "start":
            continue
        try:
            eng, cfg = build_engine_from_config(data.get("config") or {}, device)
        except ValueError as e:
            await ws.send_json({"event": "error", "msg": str(e)})
            continue
        os.makedirs(PLAYGROUND_DIR, exist_ok=True)
```

(其余 `_ws` 主体 —— tag/path/Recorder/`_LIVE_KEY`/全速 `for` 循环/`encode_frame`/done —— 一字不改。)

- [ ] **Step 6: 跑新测试 + 现有 server 测试回归**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_server_config.py -v`
Expected: PASS。注意:现有 `test_build_engine_defaults`、`test_build_engine_custom_slot_reaches_world`(用旧单 `slots` 结构)会因新签名失败 —— 把这两个旧用例改写为四 players 等价形式:

```python
def test_build_engine_defaults():
    eng, cfg = build_engine_from_config(
        {"players": [{"slots": {}} for _ in range(4)]}, DEV)
    assert cfg["grid"] == 128 and cfg["K"] == 64 and cfg["fill"] == 20
    assert cfg["T"] == 450 and cfg["seed"] == 0 and cfg["z_max"] == 8.0
    assert all(lay == BB0_TEMPLATE["layout"] for lay in cfg["layouts"])
    assert eng.H == 128 and eng.W == 128 and eng.K == 64


def test_build_engine_custom_slot_reaches_world():
    cfg_in = {"players": [{"slots": {2: "F4Nr1"}} for _ in range(4)],
              "grid": 16, "K": 8, "fill": 4, "seed": 1}
    eng, cfg = build_engine_from_config(cfg_in, DEV)
    expect = eng.table.get_or_mint(cfg["layouts"][0])
    assert int(eng.world.strain_id[4, 4, 0]) == expect   # quadrant center of 16-grid
```

- [ ] **Step 7: 提交**

```bash
git add webapp/server.py tests/test_server_config.py
git commit -m "feat(server): config protocol single slots -> 4 players + WS error event"
```

---
### Task 3: 后端全量回归闸 —— 证明红线 5 无漂移

**Files:** 无代码改动。这是 backend→frontend 之间的 reviewer 闸口:证明引擎/server 改动未污染帧格式/读数。

**Interfaces:** Consumes: 全部后端测试。Produces: 全绿后端,作为前端开工前提。

- [ ] **Step 1: 跑 frame/readouts known-answer 回归(红线 5)**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_frame.py tests/test_readouts.py tests/test_analyze_batch.py -v`
Expected: PASS —— 全绿。这些测试锁死 `encode_frame`/`cell_detail`/`compute_readouts` 的输出;它们仍绿即证明 Task 1/2 没碰帧契约。

- [ ] **Step 2: 跑全量后端套件**

Run: `$env:PYTHONPATH='src'; python -m pytest -q`
Expected: PASS —— 全部测试绿(含 Task 1/2 新增)。若有红,回到对应 task 修复,不进前端。

- [ ] **Step 3: (无提交)** 本任务不产代码;若 Step 1/2 发现需修的回归,修复并补提到对应 task 的提交里。

---
### Task 4: Astro 项目脚手架 + 构建冒烟

**Files:**
- Create: `webapp/frontend/package.json`
- Create: `webapp/frontend/astro.config.mjs`
- Create: `webapp/frontend/src/pages/index.astro`(临时占位,Task 6 填真内容)
- Create: `webapp/frontend/.gitignore`

**Interfaces:** Produces: `npm run build` 退 0,产出 `webapp/frontend/dist/index.html`;dev server 代理 `/ws`、`/config`、`/api/*` → `localhost:8000`。

- [ ] **Step 1: 写 `package.json`**

```json
{
  "name": "des-viz-frontend",
  "type": "module",
  "private": true,
  "scripts": {
    "dev": "astro dev",
    "build": "astro build",
    "preview": "astro preview"
  },
  "dependencies": {
    "astro": "^4.16.0"
  }
}
```

- [ ] **Step 2: 写 `astro.config.mjs`**

`output:'static'`,`outDir` 默认即 `webapp/frontend/dist/`;dev 时把后端三组路由代理到 8000:

```js
import { defineConfig } from 'astro/config';

// output:'static' → SSG build into dist/, hosted by aiohttp in production.
// dev server proxies the live backend (WS + config + drilldown) to :8000.
export default defineConfig({
  output: 'static',
  server: { port: 4321 },
  vite: {
    server: {
      proxy: {
        '/ws': { target: 'ws://localhost:8000', ws: true },
        '/config': 'http://localhost:8000',
        '/api': 'http://localhost:8000',
      },
    },
  },
});
```

- [ ] **Step 3: 写临时 `src/pages/index.astro` 占位**

```astro
---
// placeholder — replaced by Task 6 with the real three-column UI
---
<!doctype html>
<html lang="zh">
<head><meta charset="utf-8"><title>DES 验收之眼</title></head>
<body><div id="app-placeholder">build smoke ok</div></body>
</html>
```

- [ ] **Step 4: 写 `.gitignore`**

```
node_modules/
dist/
.astro/
```

- [ ] **Step 5: 安装依赖**

Run: `cd webapp/frontend; npm install`
Expected: 退 0,生成 `node_modules/` 与 `package-lock.json`。

- [ ] **Step 6: 构建冒烟**

Run: `cd webapp/frontend; npm run build`
Expected: 退 0,生成 `webapp/frontend/dist/index.html`(含 `build smoke ok`)。

- [ ] **Step 7: 提交(含 lock,排除 node_modules/dist)**

```bash
git add webapp/frontend/package.json webapp/frontend/astro.config.mjs \
        webapp/frontend/src/pages/index.astro webapp/frontend/.gitignore \
        webapp/frontend/package-lock.json
git commit -m "build(frontend): scaffold Astro project (static output + dev proxy)"
```

---
### Task 5: 前端运行时 ES 模块(render / charts / config / sim)

**Files:**
- Create: `webapp/frontend/src/scripts/render.js`
- Create: `webapp/frontend/src/scripts/charts.js`
- Create: `webapp/frontend/src/scripts/config.js`
- Create: `webapp/frontend/src/scripts/sim.js`

**Interfaces:**
- Produces:
  - `render.js`: `FC`(4 阵营 RGB)、`setupCanvas(n)`、`gridW()`、`drawFrame(frame, kCap)`。
  - `charts.js`: `line(id, series, colors, ymax)`。
  - `config.js`: `CFG`、`playerSlots`(`[{slotIdx:letter}×4]`)、`loadConfig()`、`collectPayload()`。
  - `sim.js`: `start()`、`reset()`、`onCanvasClick(ev)`。
- Consumes: `/config`、`/ws`、`/api/cell`(后端,Task 1/2);DOM id(Task 6 提供)。
- 校验:每个模块 `node --check` 必须过(纯语法,无运行时依赖)。

- [ ] **Step 1: 写 `render.js`(drawFrame 逐字搬运,红线 2)**

```js
// webapp/frontend/src/scripts/render.js — canvas compositing.
// drawFrame copied VERBATIM from the old webapp/static/app.js (red-line 2:
// faithful tensor — faction-count-weighted hue + density alpha, pixelated
// nearest-neighbor, NO smoothing). Only change: kCap is a param, not a DOM read.
export const FC = [[255, 77, 77], [77, 155, 255], [63, 208, 127], [255, 204, 77]];

let imgW = 128, img = null, ctx = null;
export function setupCanvas(n) {
  const cv = document.getElementById("grid"); cv.width = n; cv.height = n; imgW = n;
  ctx = cv.getContext("2d"); img = ctx.createImageData(n, n);
}
export function gridW() { return imgW; }
export function drawFrame(frame, kCap) {
  const n = frame.H;
  if (n !== imgW || !img) setupCanvas(n);
  img.data.fill(0);
  for (let i = 3; i < img.data.length; i += 4) img.data[i] = 255;   // opaque black
  for (const c of frame.cells) {
    const y = c[0], x = c[1], cf = [c[2], c[3], c[4], c[5]];
    const tot = cf[0] + cf[1] + cf[2] + cf[3];
    if (tot <= 0) continue;
    let r = 0, gg = 0, bb = 0;
    for (let f = 0; f < 4; f++) { const w = cf[f] / tot; r += FC[f][0] * w; gg += FC[f][1] * w; bb += FC[f][2] * w; }
    const dens = Math.min(1, tot / kCap), a = 0.15 + 0.85 * dens;
    const o = (y * n + x) * 4;
    img.data[o] = r * a; img.data[o + 1] = gg * a; img.data[o + 2] = bb * a; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
```

- [ ] **Step 2: 写 `charts.js`(line 逐字搬运)**

```js
// webapp/frontend/src/scripts/charts.js — hand-drawn line charts,
// copied VERBATIM from the old webapp/static/app.js (no chart lib).
export function line(id, series, colors, ymax) {
  const cv = document.getElementById(id), c = cv.getContext("2d");
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
```

- [ ] **Step 3: 写 `config.js`(4 玩家基因型编辑器 + payload)**

```js
// webapp/frontend/src/scripts/config.js — /config load, 4-player genome editors,
// legend. Editor structure is built from /config at runtime (palette/slots/locked);
// only primitive letters + run knobs ever leave here (red-line 3).
import { FC } from "./render.js";

const FNAME = ["阵营0", "阵营1", "阵营2", "阵营3"];
export let CFG = null;
export const playerSlots = [0, 1, 2, 3].map(() => ({}));   // faction -> {slotIdx: letter}

export async function loadConfig() {
  CFG = await (await fetch("/config")).json();
  const locked = CFG.locked, slots = new Set(CFG.slots);
  for (let f = 0; f < 4; f++) {
    const host = document.querySelector(`.genome-list[data-faction="${f}"]`);
    host.innerHTML = "";
    for (let i = 0; i < 16; i++) {
      const row = document.createElement("div");
      const gi = `<span class="gi">#${i}</span>`;
      if (locked[String(i)] !== undefined) {
        row.className = "grow fn";
        row.innerHTML = `${gi}<span class="gtype">功能·锁死</span><span class="gtag">${locked[String(i)]}</span>`;
      } else if (slots.has(i)) {
        row.className = "grow slot";
        row.innerHTML = `${gi}<span class="gtype">插槽</span>`;
        const sel = document.createElement("select");
        CFG.palette.forEach((p) => { const o = document.createElement("option"); o.value = p; o.textContent = p; sel.appendChild(o); });
        sel.value = "N0"; playerSlots[f][i] = "N0";
        sel.onchange = () => { playerSlots[f][i] = sel.value; };
        row.appendChild(sel);
      } else {
        row.className = "grow bb";
        row.innerHTML = `${gi}<span class="gtype">骨架</span><span class="gtag">N0</span>`;
      }
      host.appendChild(row);
    }
  }
  const rows = [0, 1, 2, 3].map((f) =>
    `<div class="lrow"><i style="background:rgb(${FC[f]})"></i>${FNAME[f]}<span class="lpct" data-f="${f}">25.0%</span></div>`);
  rows.push(`<div class="lrow" style="color:var(--dim);min-width:0">亮度=密度 · 混色=争夺</div>`);
  document.getElementById("legend").innerHTML = rows.join("");
}

const num = (id) => +document.getElementById(id).value;
export function collectPayload() {
  return {
    players: playerSlots.map((s) => ({ slots: { ...s } })),
    grid: num("cfg-grid"), K: num("cfg-K"), fill: num("cfg-fill"),
    T: num("cfg-T"), seed: num("cfg-seed"), z_max: num("cfg-zmax"),
  };
}
```

- [ ] **Step 4: 写 `sim.js`(WS 状态机 + 读数 + 排行 + % + 钻取)**

```js
// webapp/frontend/src/scripts/sim.js — WS live loop, readout/leaderboard/% updates,
// start/reset state machine (FULL-SPEED, no pause/speed), cell drilldown.
// updateReadouts/renderLeaderboard/onCanvasClick adapted from the old app.js.
import { FC, drawFrame, gridW } from "./render.js";
import { line } from "./charts.js";
import { collectPayload } from "./config.js";

const $ = (id) => document.getElementById(id);
const fmtPct = (v) => (v * 100).toFixed(1) + "%";
let ws = null, livePath = null, started = false;
const shareSeries = [[], [], [], []], distinctSeries = [], n2Series = [];
function resetSeries() { for (const s of shareSeries) s.length = 0; distinctSeries.length = 0; n2Series.length = 0; }

export function start() {
  if (started) { reset(); return; }                  // button doubles as 重置 once running
  resetSeries();
  const cfg = collectPayload();
  $("gridLabel").textContent = `${cfg.grid}×${cfg.grid}`;
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => {
    started = true; $("status").textContent = "运行中"; $("status").style.color = "var(--dim)";
    $("startBtn").textContent = "⟳ 重置";
    ws.send(JSON.stringify({ cmd: "start", config: cfg }));
  };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.event === "error") { $("status").textContent = "配置错误:" + m.msg; $("status").style.color = "#ff6b6b"; return; }
    if (m.event === "started") { livePath = m.path; return; }
    if (m.event === "done") { $("status").textContent = "完成"; return; }
    drawFrame(m, +$("cfg-K").value || 64); updateReadouts(m);
  };
  ws.onclose = () => { if ($("status").textContent === "运行中") $("status").textContent = "已停止"; };
}

export function reset() {
  if (ws) { ws.close(); ws = null; }
  resetSeries(); started = false; livePath = null;
  $("status").textContent = "未开始"; $("status").style.color = "var(--dim)";
  $("startBtn").textContent = "▶ 开始";
  $("tickLabel").textContent = "tick 0 / " + (+$("cfg-T").value || 450);
}

function updateReadouts(frame) {
  const r = frame.readouts;
  $("m-tick").textContent = frame.tick;
  $("m-total").textContent = r.total;
  $("m-occ").textContent = r.occupied_cells;
  $("m-distinct").textContent = r.distinct_strains;
  $("m-n2").textContent = (r.n2 ?? 0).toFixed(1);
  $("m-dmax").textContent = (r.d_max ?? 0).toFixed(3);
  $("tickLabel").textContent = `tick ${frame.tick} / ${+$("cfg-T").value || 450}`;
  const share = r.faction_share || {};
  for (let f = 0; f < 4; f++) shareSeries[f].push(share[f] || 0);
  distinctSeries.push(r.distinct_strains); n2Series.push(r.n2);
  document.querySelectorAll(".player").forEach((p, f) => {
    const pct = p.querySelector(".pct"); if (pct) pct.textContent = fmtPct(share[f] || 0);
  });
  for (let f = 0; f < 4; f++) {
    const el = document.querySelector(`.lpct[data-f="${f}"]`);
    if (el) el.textContent = fmtPct(share[f] || 0);
  }
  line("shareChart", shareSeries, ["#ff4d4d", "#4d9bff", "#3fd07f", "#ffcc4d"], 1.0);
  const dmax = Math.max(1, ...distinctSeries), nmax = Math.max(1, ...n2Series);
  line("divChart", [distinctSeries.map((v) => v / dmax), n2Series.map((v) => v / nmax)],
    ["#a371f7", "#39c5cf"], 1.0);
  renderLeaderboard(frame.leaderboard || []);
}

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

export function onCanvasClick(ev) {
  if (!livePath) return;
  const cv = $("grid"), rect = cv.getBoundingClientRect(), n = gridW();
  const x = Math.floor((ev.clientX - rect.left) / rect.width * n);
  const y = Math.floor((ev.clientY - rect.top) / rect.height * n);
  const tick = +$("m-tick").textContent;
  fetch(`/api/cell?path=${encodeURIComponent(livePath)}&tick=${tick}&y=${y}&x=${x}`)
    .then((r) => { if (!r.ok) throw new Error("cell fetch failed"); return r.json(); })
    .then((d) => {
      if (!d || !d.strains) return;
      const ul = $("cellDetail"); ul.innerHTML = "";
      if (!d.strains.length) { ul.innerHTML = `<li style="color:var(--dim)">格 (${y},${x}) 空</li>`; return; }
      d.strains.forEach((s) => {
        const li = document.createElement("li");
        li.innerHTML = `<span class="seq" title="${s.strain}">${s.strain}</span>` +
          `<span style="color:rgb(${FC[s.faction]})">f${s.faction}</span>` +
          `<span style="margin-left:auto;font-variant-numeric:tabular-nums">${s.count}</span>`;
        ul.appendChild(li);
      });
    })
    .catch(() => { $("cellDetail").innerHTML = `<li style="color:var(--dim)">钻取暂不可用,请重试</li>`; });
}
```

- [ ] **Step 5: 语法校验四模块**

Run: `cd webapp/frontend; node --check src/scripts/render.js; node --check src/scripts/charts.js; node --check src/scripts/config.js; node --check src/scripts/sim.js`
Expected: 四条均退 0、无输出(`node --check` 对 ES module 语法静态检查)。

- [ ] **Step 6: 提交**

```bash
git add webapp/frontend/src/scripts/
git commit -m "feat(frontend): runtime ES modules — render/charts/config/sim (drawFrame verbatim)"
```

---
### Task 6: Astro 组件 + 页面 + 样式(UI 装配)

**Files:**
- Create: `webapp/frontend/src/styles/global.css`
- Create: `webapp/frontend/src/components/GenomeList.astro`
- Create: `webapp/frontend/src/components/PlayerConfig.astro`
- Create: `webapp/frontend/src/components/PlayerRail.astro`
- Create: `webapp/frontend/src/components/Stage.astro`
- Create: `webapp/frontend/src/components/ReadPanel.astro`
- Modify: `webapp/frontend/src/pages/index.astro`(替换 Task 4 占位)

**Interfaces:**
- Consumes: Task 5 模块(`loadConfig`/`start`/`reset`/`onCanvasClick`/`setupCanvas`);DOM id 契约(`grid`、`legend`、`players`、`startBtn`、`status`、`tickLabel`、`gridLabel`、`m-*`、`shareChart`、`divChart`、`lead`、`cellDetail`、`cfg-*`)。
- Produces: 三栏 grid 页面 + 两 resizer;`<script>` 绑定运行时;退场 mockup 的播放/暂停/速度件。

- [ ] **Step 1: 写 `src/styles/global.css`(从 mockup `<style>` 搬,去掉播放/速度相关)**

```css
:root{
  --bg:#0d1117; --panel:#161b22; --line:#30363d; --txt:#c9d1d9; --dim:#8b949e;
  --f0:#ff4d4d; --f1:#4d9bff; --f2:#3fd07f; --f3:#ffcc4d;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
     font:13px/1.4 ui-sans-serif,system-ui,"Segoe UI",sans-serif}
.app{display:grid;grid-template-columns:var(--lw,268px) 5px 1fr 5px var(--rw,320px);height:100vh}
.resizer{background:var(--line);cursor:col-resize;transition:background .1s}
.resizer:hover,.resizer.drag{background:#1f6feb}
.rail{background:var(--panel);border-right:1px solid var(--line);overflow-y:auto}
.rail h2{font-size:12px;letter-spacing:.05em;color:var(--dim);
         text-transform:uppercase;margin:14px 12px 8px}
.rail .body{padding:0 12px 16px}
.genome-list{display:flex;flex-direction:column;gap:2px;margin-bottom:8px}
.grow{display:flex;align-items:center;gap:6px;padding:3px 8px;border-radius:4px;
      font-size:11px;border:1px solid var(--line)}
.grow .gi{color:var(--dim);font-variant-numeric:tabular-nums;width:32px;font-size:10px}
.grow .gtype{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:.04em}
.grow .gtag{margin-left:auto;font-family:ui-monospace,monospace}
.grow.bb{background:#161b22;border-color:#21262d;opacity:.55}
.grow.bb .gtag{color:var(--dim)}
.grow.fn{background:#21262d}
.grow.fn .gtag{color:var(--txt)}
.grow.slot{background:#1c2733;border-color:#1f6feb55}
.grow.slot select{margin-left:auto;background:#0d1117;border:1px solid var(--line);
     color:#79c0ff;font-size:11px;border-radius:4px;padding:1px 6px;cursor:pointer}
.row{display:flex;justify-content:space-between;align-items:center;margin:6px 0}
.row input{background:#0d1117;border:1px solid var(--line);color:var(--txt);
     border-radius:4px;padding:2px 6px;width:90px}
.go{width:100%;margin-top:12px;padding:9px;background:#238636;border:0;border-radius:6px;
    color:#fff;font-weight:600;cursor:pointer;font-size:14px}
.player{border:1px solid var(--line);border-radius:6px;margin-bottom:8px;overflow:hidden}
.player>summary{list-style:none;cursor:pointer;display:flex;align-items:center;gap:8px;
     padding:8px 10px;background:#0d1117;font-weight:600}
.player>summary::-webkit-details-marker{display:none}
.player>summary .dot{width:11px;height:11px;border-radius:3px;flex:none}
.player>summary .pct{margin-left:auto;color:var(--dim);font-variant-numeric:tabular-nums;font-weight:400}
.player>summary .chev{color:var(--dim);transition:transform .15s}
.player[open]>summary .chev{transform:rotate(90deg)}
.player .pbody{padding:8px 10px;border-top:1px solid var(--line)}
.stage{display:flex;flex-direction:column;min-width:0}
.stagehead{display:flex;align-items:center;gap:14px;padding:10px 16px;
           border-bottom:1px solid var(--line)}
.stagehead .title{font-weight:600}
.legend{margin-left:auto;font-size:11px;color:var(--dim);
        display:flex;flex-direction:column;gap:4px;align-items:flex-start}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;
          vertical-align:-1px}
.legend .lrow{display:flex;align-items:center;gap:4px;min-width:140px}
.legend .lrow .lpct{margin-left:auto;font-variant-numeric:tabular-nums;color:var(--txt)}
.canvaswrap{flex:1;display:flex;align-items:center;justify-content:center;padding:16px;min-height:0}
#grid{image-rendering:pixelated;background:#000;border:1px solid var(--line);
      width:min(72vh,100%);height:min(72vh,100%);aspect-ratio:1;cursor:crosshair}
.transport{display:flex;align-items:center;gap:12px;padding:10px 16px;border-top:1px solid var(--line)}
.ticklabel{font-variant-numeric:tabular-nums;color:var(--dim);min-width:120px}
.read{background:var(--panel);border-left:1px solid var(--line);overflow-y:auto;padding:14px}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px}
.card{background:#0d1117;border:1px solid var(--line);border-radius:6px;padding:8px 10px}
.card .k{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:18px;font-weight:600;font-variant-numeric:tabular-nums}
.read h3{font-size:11px;letter-spacing:.05em;color:var(--dim);text-transform:uppercase;margin:16px 0 6px}
.chart{width:100%;height:90px;background:#0d1117;border:1px solid var(--line);border-radius:6px}
.lead{list-style:none;padding:0;margin:0;font-size:12px}
.lead li{display:flex;align-items:center;gap:6px;padding:3px 0}
.lead .seq{font-family:ui-monospace,monospace;font-size:10px;color:var(--dim);
           white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:160px}
```

- [ ] **Step 2: 写 `src/components/GenomeList.astro`(空容器,运行时由 config.js 填)**

```astro
---
const { faction } = Astro.props;
---
<div style="font-size:10px;color:var(--dim);margin-bottom:6px">
  BB0 基因型(16 位)· 灰=骨架噪声 · 深=功能锁死 · 蓝=可选插槽</div>
<div class="genome-list" data-faction={faction}></div>
```

- [ ] **Step 3: 写 `src/components/PlayerConfig.astro`(单玩家手风琴)**

```astro
---
import GenomeList from "./GenomeList.astro";
const { faction } = Astro.props;
const FC = ["255,77,77", "77,155,255", "63,208,127", "255,204,77"];
const FNAME = ["阵营0", "阵营1", "阵营2", "阵营3"];
---
<details class="player" open={faction === 0}>
  <summary>
    <span class="dot" style={`background:rgb(${FC[faction]})`}></span>
    <span>玩家 {faction} · {FNAME[faction]}</span>
    <span class="pct">25.0%</span>
    <span class="chev">▸</span>
  </summary>
  <div class="pbody"><GenomeList faction={faction} /></div>
</details>
```

- [ ] **Step 4: 写 `src/components/PlayerRail.astro`(左栏:4 玩家 + 全局参数 + 开始)**

```astro
---
import PlayerConfig from "./PlayerConfig.astro";
---
<aside class="rail">
  <h2><span>玩家配置</span></h2>
  <div class="body" id="players">
    <PlayerConfig faction={0} />
    <PlayerConfig faction={1} />
    <PlayerConfig faction={2} />
    <PlayerConfig faction={3} />
  </div>
  <div class="body">
    <h2 style="margin:8px 0 6px"><span>全局参数</span></h2>
    <div class="row"><span>网格</span><input id="cfg-grid" value="128"></div>
    <div class="row"><span>K</span><input id="cfg-K" value="64"></div>
    <div class="row"><span>fill/格</span><input id="cfg-fill" value="20"></div>
    <div class="row"><span>T (ticks)</span><input id="cfg-T" value="450"></div>
    <div class="row"><span>seed</span><input id="cfg-seed" value="0"></div>
    <div class="row"><span>z_max</span><input id="cfg-zmax" value="8.0"></div>
    <button class="go" id="startBtn">▶ 开始</button>
  </div>
</aside>
```

- [ ] **Step 5: 写 `src/components/Stage.astro`(中栏:标题 + 竖排图例 + canvas + transport,无播放/速度)**

```astro
---
---
<main class="stage">
  <div class="stagehead">
    <span class="title">验收之眼 — <span id="gridLabel">128×128</span> 世界</span>
    <div class="legend" id="legend"></div>
  </div>
  <div class="canvaswrap"><canvas id="grid" width="128" height="128"></canvas></div>
  <div class="transport">
    <span class="ticklabel" id="tickLabel">tick 0 / 450</span>
    <span id="status" style="color:var(--dim)">未开始</span>
    <span style="margin-left:auto;color:var(--dim)">点格子 → 看 {"{strain:count}"}</span>
  </div>
</main>
```

- [ ] **Step 6: 写 `src/components/ReadPanel.astro`(右栏:6 卡 + 2 图 + 排行 + 明细)**

```astro
---
---
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
  <h3>主导株排行(top 5)</h3><ul class="lead" id="lead"></ul>
  <h3>该格明细(点格子)</h3>
  <ul class="lead" id="cellDetail"><li style="color:var(--dim)">点中心 canvas 的格子查看</li></ul>
</aside>
```

- [ ] **Step 7: 写 `src/pages/index.astro`(装配 + 脚本胶水 + resizer)**

```astro
---
import PlayerRail from "../components/PlayerRail.astro";
import Stage from "../components/Stage.astro";
import ReadPanel from "../components/ReadPanel.astro";
import "../styles/global.css";
---
<!doctype html>
<html lang="zh">
<head><meta charset="utf-8"><title>DES 验收之眼</title></head>
<body>
  <div class="app">
    <PlayerRail />
    <div class="resizer" id="lresizer"></div>
    <Stage />
    <div class="resizer" id="rresizer"></div>
    <ReadPanel />
  </div>
  <script>
    import { setupCanvas } from "../scripts/render.js";
    import { line } from "../scripts/charts.js";
    import { loadConfig } from "../scripts/config.js";
    import { start, reset, onCanvasClick } from "../scripts/sim.js";

    // draggable column resizers — offsetWidth version copied VERBATIM from mockup.
    function setupResizers() {
      const app = document.querySelector(".app");
      function drag(el, varName, col) {
        el.addEventListener("mousedown", (e) => {
          e.preventDefault(); el.classList.add("drag");
          const startX = e.clientX, startW = app.children[col].offsetWidth;
          const move = (ev) => {
            const dx = ev.clientX - startX;
            const w = Math.max(160, Math.min(560, startW + (col === 0 ? dx : -dx)));
            app.style.setProperty(varName, w + "px");
          };
          const up = () => {
            el.classList.remove("drag");
            document.removeEventListener("mousemove", move);
            document.removeEventListener("mouseup", up);
          };
          document.addEventListener("mousemove", move);
          document.addEventListener("mouseup", up);
        });
      }
      drag(document.getElementById("lresizer"), "--lw", 0);   // col 0 = .rail
      drag(document.getElementById("rresizer"), "--rw", 4);   // col 4 = .read
    }

    window.addEventListener("DOMContentLoaded", () => {
      setupCanvas(128); loadConfig(); setupResizers(); reset();
      document.getElementById("startBtn").onclick = start;
      document.getElementById("grid").addEventListener("click", onCanvasClick);
    });
  </script>
</body>
</html>
```

- [ ] **Step 8: 构建冒烟**

Run: `cd webapp/frontend; npm run build`
Expected: 退 0,`dist/index.html` 含三栏结构与 4 个 `.player` 手风琴;`dist/` 下有打包后的 JS。

- [ ] **Step 9: 提交**

```bash
git add webapp/frontend/src/
git commit -m "feat(frontend): Astro components + page + styles (V3 UI, no playback controls)"
```

---
### Task 7: aiohttp 托管 dist/ + e2e 冒烟 + 删旧 static

**Files:**
- Modify: `webapp/server.py:15-16`(`_STATIC_DIR` 指向 `dist/`)、`webapp/server.py:102-103`(`_index`)、`webapp/server.py:146-158`(`make_app`)
- Test: `tests/test_server_config.py`(改 `test_make_app_registers_routes` 不依赖 static 目录建出)
- Delete: `webapp/static/index.html`、`webapp/static/app.js`、`webapp/static/style.css`(e2e 通过后)

**Interfaces:**
- Produces: aiohttp `/` 返回 `dist/index.html`;`/static`(若保留)与构建产物对齐;`dist/` 缺失时启动期明确报错,不静默 500。
- Consumes: Task 6 的 `webapp/frontend/dist/`。

- [ ] **Step 1: 改 `_STATIC_DIR` 与 `_index`(`webapp/server.py`)**

把 16 行 `_STATIC_DIR` 改为指向 Astro 产物,并在 `_index` 里对缺失文件给明确错误:

```python
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
```

`_index`(102–103 行)改为:

```python
async def _index(request):
    index = os.path.join(_STATIC_DIR, "index.html")
    if not os.path.exists(index):
        raise web.HTTPServiceUnavailable(
            text="前端未构建:先 cd webapp/frontend && npm install && npm run build")
    return web.FileResponse(index)
```

- [ ] **Step 2: 改 `make_app`(`webapp/server.py:146-158`)**

把 `os.makedirs(_STATIC_DIR, ...)`(150 行)删掉(dist/ 由 build 产出,不该由 server mkdir 掩盖未构建),并把静态根挂到 dist/ 的资源子目录。`make_app` 路由段改为:

```python
def make_app(device=None) -> web.Application:
    app = web.Application()
    app[_DEVICE_KEY] = _device(device)
    app[_LIVE_KEY] = {}
    app.router.add_get("/config", _config)
    app.router.add_get("/api/frame_at_tick", _frame_at_tick)
    app.router.add_get("/api/cell", _cell)
    app.router.add_get("/api/trajectory", _trajectory)
    app.router.add_get("/ws", _ws)
    app.router.add_get("/", _index)
    if os.path.isdir(os.path.join(_STATIC_DIR, "_astro")):
        app.router.add_static("/_astro", os.path.join(_STATIC_DIR, "_astro"))
    return app
```

(`/_astro` 是 Astro 默认打包资源目录;仅当存在时挂载,避免未构建即 raise。)

并把 `main()`(161–162 行)加一句启动期检查,让「未构建」在启动日志就显形(spec §5:不静默,启动期明确报),而非只在首个请求 503:

```python
def main() -> None:
    if not os.path.exists(os.path.join(_STATIC_DIR, "index.html")):
        print("[warn] 前端未构建:先 cd webapp/frontend && npm install && npm run build")
    web.run_app(make_app(), port=8000)
```

- [ ] **Step 3: 改 `test_make_app_registers_routes`(`tests/test_server_config.py`)**

该用例曾断言 `make_app` 建出 static 目录;现 dist/ 不再由 server 创建。把它改为只验路由 + playground 隔离(去掉对目录创建的依赖):

```python
def test_make_app_registers_routes():
    from webapp.server import make_app, PLAYGROUND_DIR
    app = make_app(device=torch.device("cpu"))
    paths = {r.resource.canonical for r in app.router.routes() if r.resource is not None}
    assert {"/", "/config", "/ws", "/api/frame_at_tick", "/api/cell", "/api/trajectory"} <= paths
    assert "playground" in PLAYGROUND_DIR and "runs" not in PLAYGROUND_DIR
```

- [ ] **Step 4: 跑后端测试确认仍绿**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_server_config.py -v`
Expected: PASS。

- [ ] **Step 5: e2e 冒烟(手动,四玩家不同插槽)**

构建并起服务:

```
cd webapp/frontend; npm run build; cd ../..
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m webapp.server
```

浏览器开 http://localhost:8000 ,验:
1. 4 个玩家手风琴各 16 位基因型,插槽下拉可改;
2. 给四玩家设不同插槽(如 f0 全 N0、f1 slot0=F4Nr1、f2 slot0=P_base、f3 slot0=P_hotspot),点「开始」;
3. canvas 四阵营色块真不同(非全同扩张),图例 4 条 live% 加总≈100%;
4. 点一个有色格子 → 右下「该格明细」出 `{strain:count}` list;
5. 跑满 T → 状态「完成」;按钮「⟳ 重置」点击回到「▶ 开始」、canvas 清空。

记录结果到提交信息或 context;任一项失败则回对应 task 修复,不删旧 static。

- [ ] **Step 6: e2e 通过后删旧 static(ponytail:不留影子)**

```bash
git rm webapp/static/index.html webapp/static/app.js webapp/static/style.css
git commit -m "feat(server): host Astro dist/ + e2e verified; remove old static frontend"
```

(回退靠 git;若 `webapp/static/` 还有其他被引用文件,先 `git status` 确认仅这三个再删。)

---
### Task 8: README 更新 + 终审全量回归

**Files:**
- Modify: `README.md`(viz 启动段:加前端构建步)
- 无新代码

**Interfaces:** Produces: 可复现启动指令;全绿测试套件作为合并前提。

- [ ] **Step 1: 更新 README 启动段**

找到现有 webapp 启动说明(`python -m webapp.server`),把它替换/补成两步(先构建前端,再起服务):

```markdown
## 验收之眼(可视化 web app)

前端用 Astro 构建静态产物,由 aiohttp 单端口托管。

1. 构建前端(首次或前端改动后):
   ```
   cd webapp/frontend
   npm install
   npm run build
   ```
2. 起服务(务必模块形式 —— server.py 用绝对导入 `from webapp.frame import`):
   ```
   $env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m webapp.server
   ```
3. 开 http://localhost:8000 。

四玩家可各设不同插槽基因型(放开 red-line 4「全同」→「同模板结构」);全速直播,无暂停/调速。
```

- [ ] **Step 2: 终审全量后端回归**

Run: `$env:PYTHONPATH='src'; python -m pytest -q`
Expected: PASS —— 全部测试绿(含 Task 1/2 新增、Task 2/7 改写)。

- [ ] **Step 3: 终审前端构建**

Run: `cd webapp/frontend; npm run build`
Expected: 退 0,`dist/index.html` 产出。

- [ ] **Step 4: 提交**

```bash
git add README.md
git commit -m "docs: README viz startup — Astro build + module-form server launch"
```

---

## 自审结果(plan 对照 spec)

**1. spec 覆盖:**
- §0 红线表 → Global Constraints(逐条)+ Task 1(④)+ Task 3/7(⑤回归)+ Task 5 注释(②③)。✅
- §1 引擎放开 → Task 1。✅
- §2 server 协议(4 处:players 结构 / build_engine / /config 不变 / 错误回传)→ Task 2。`/config` 不变:Task 2 显式「不动 `_config`」。✅
- §3 前端 Astro 结构(astro.config / 组件 / scripts / styles)→ Task 4(脚手架)+ Task 5(scripts)+ Task 6(组件/页面/样式)。✅
- §4 数据流 → Task 5 `sim.js` 状态机 + Task 6 装配。✅
- §5 错误处理(非法 slots / WS 断 / cell 失败 / dist 缺失)→ Task 2 Step 5(error 事件)+ Task 5 `sim.js`(onclose/catch)+ Task 7 Step 1(dist 缺失明确报错)。✅
- §6 测试(Python 单元 / 构建冒烟 / e2e)→ Task 1/2(单元)+ Task 4/6(构建冒烟)+ Task 7(e2e)。✅
- §7 构建集成 + 删旧 → Task 4(astro.config 代理)+ Task 7(托管 dist + git rm static)。✅
- §8 范围边界 → 全 plan 仅放开「起始基因型不必相同」,无每阵营 K/突变率,无暂停/调速,无 scrubber。✅

**2. 占位扫描:** 无 TBD/TODO/"适当处理";每个代码步给完整代码。✅

**3. 类型一致性:**
- `init_factions(..., layout=None, layouts=None)` —— Task 1 定义,Task 1 Step 4 `Engine` 透传 `layouts`,Task 2 `build_engine_from_config` 传 `layouts=layouts`。一致。✅
- `build_engine_from_config` 返回 `resolved` 含 `layouts`(4-tuple)+ `players` —— Task 2 定义,`_jsonable` 序列化二者,测试断言 `cfg["layouts"]`。一致。✅
- 前端 DOM id 契约:`grid/legend/players/startBtn/status/tickLabel/gridLabel/m-*/shareChart/divChart/lead/cellDetail/cfg-*` —— Task 6 组件提供,Task 5 模块消费。逐一核对一致(`gridLabel` 在 Stage、`cfg-zmax` 对应 `z_max`)。✅
- `drawFrame(frame, kCap)`、`gridW()`、`setupCanvas(n)` —— Task 5 render.js export,Task 5 sim.js + Task 6 index.astro 调用,签名一致。✅








