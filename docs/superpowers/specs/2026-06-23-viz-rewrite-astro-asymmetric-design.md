# 验收之眼 v2 — Astro 重写 + 放开四阵营对称(red-line 4)设计

**日期:** 2026-06-23
**分支:** `feat/ui-mockups`(沿用,后续可改 `feat/viz-rewrite`)
**前置:** V3 mockup(`docs/mockups/v3-enhanced.html/.js`)已选定为目标界面。本 spec 把它搬进真应用,并放开引擎使其能真跑「四阵营各自独立基因型」。

---

## 0. 红线状态(置顶 · 实现时不得违反)

本轮**主动放开** red-line 4 的「四阵营基因型全同」约束,其余四条红线原样守死。

| 红线 | 本轮状态 |
|---|---|
| ① playground 数据隔离 | **守。** 仍写 `data/playground/`,绝不碰 `data/runs/`。 |
| ② live 帧忠实张量、无平滑 | **守。** `drawFrame` 合成逻辑逐字搬运:faction-count 加权混色 + density alpha + `image-rendering:pixelated` 最近邻放大。不引入任何插值。 |
| ③ 配置只暴露基元字母 + run 参 | **守。** `/config` 仍只给 `palette`(6 字母)/ `slots`(可编辑位)/ `locked` / `defaults`。无任何「谁强」系数。 |
| ④ 四阵营对称 | **放开「全同」,降级为「同模板结构」。** 见 §1。四中心仍是 D4 对称轨道(无几何偏袒),四阵营仍走同一固定 G→P 函数。 |
| ⑤ 读数单一来源 | **守。** `webapp/readouts.py`、`webapp/frame.py` 的 `encode_frame`/`cell_detail` **一字不改**。known-answer 回归测试必须仍绿。 |

**纪律澄清(写死):** 破的是「四阵营起始基因型必须相同」这条**对称对照**约束 —— 不破「无私货」纪律。四阵营仍共用同一 BB0 模板(锁死位、骨架位、插槽位置与调色板四阵营一致),仅各自插槽取值可不同;表型仍是序列的固定函数,严禁手写「谁强」。

---

## 1. 引擎/world 放开 red-line 4(唯一引擎触点)

**改一处:** `src/des/world.py` 的 `init_factions`(现 35–58 行)。

**现状:** 接单条 `layout: tuple[str,...] | None`,mint 一条 bb0,广播注入四阵营中心。

**改后:** 接四条 layout,各 mint 各注。

- 新增参数 `layouts`:四条 layout 的序列(list/tuple of 4)。
- 保留 `layout`(单条):传单条时四阵营全同 —— **向后兼容**,现有调用与测试不破。
- `layouts=None and layout=None` → 仍走 `BB0_TEMPLATE["layout"]`(正典默认行为不变)。
- 互斥校验:`layouts` 与 `layout` 同时非 None → `ValueError`;给了 `layouts` 时必须恰好 4 条,否则 `ValueError`。
- 循环里第 `fac` 个象限中心注入第 `fac` 条:`bb0_f = table.get_or_mint(layouts[fac])`;`w.strain_id[cy,cx,0]=bb0_f`;`count`/`faction` 同现状。
- **每条** layout 独立过 `validate_bb0_layout`(该函数**不动** —— 它从来只验单条结构:锁死位==`_LOCKED`、骨架位=="N0"、插槽∈6字母调色板;从不跨阵营比较)。

**Engine 透传:** `src/des/engine.py`(现 15、21 行)`__init__` 加 `layouts` 参,透传给 `init_factions`。保留 `layout` 参。

**四中心对称不变:** 注入位置仍为 `[(H//4,W//4),(H//4,3W//4),(3H//4,W//4),(3H//4,3W//4)]`(D4 轨道),只换每中心注入的 bb0,几何对称不动。

---

## 2. server 协议(单 slots → 四 slots)

**改 `webapp/server.py`,三处:**

**(1) 配置结构 single → players**
```
现:  cfg = { slots:{i:letter}, grid,K,fill,T,seed,z_max }
改:  cfg = { players:[ {slots:{i:letter}} ×4 ], grid,K,fill,T,seed,z_max }
```
- `players` 必须恰好 4 个;少/多即 `ValueError`。
- 全局参数(grid/K/fill/T/seed/z_max)**四阵营共享**,不进 players(符合 mockup,未画每阵营不同 K/突变率)。

**(2) `build_engine_from_config`**
- 四条 layout 一行 inline 生成(复用现有单条 `layout_from_slots`,**不另立命名函数** —— ponytail):
  `layouts = tuple(layout_from_slots({int(k):v for k,v in (p.get("slots") or {}).items()}) for p in players)`
- 每条经 `layout_from_slots` 已含 `validate_bb0_layout` 守门(沿用现有逻辑)。
- `Engine(layouts=layouts, ...)`。
- `resolved` 回带 `players` 与 `layouts`(四条,`_jsonable` 序列化)供前端回显。

**(3) `/config`(GET)不变**
- 仍只返回模板级元信息:`palette / slots / locked / defaults`。
- 理由:模板四阵营共用,是结构契约;每玩家「当前选择」是前端状态,不入 /config。前端拿 /config 画 4 份相同结构编辑器,各自维护选择。

**(4) 错误回传(本轮新增一小点)**
- 某玩家 slots 非法(如插槽位填锁死字母、字母不在调色板)→ `build_engine_from_config` 抛 `ValueError` → WS 回 `{event:"error", msg:str(e)}`,前端状态栏红字提示、不开跑。

**不动:** WS 全速直播循环(`_ws` 跑 T 次 `eng.step()` + `encode_frame` 直推,无 sleep、无暂停/调速并发改造)。`encode_frame`/`cell_detail`/readouts/`/api/cell`(live 读内存)全部一字不改。

---

## 3. 前端:Astro 重写

**栈决议:** 用户明确选 Astro(已知账:单页实时应用里 SSG/islands 收益小、引入 Node 工具链+build 步代价大;用户接受)。`output:'static'`,构建出 `dist/`,由 aiohttp 托管(单端口单部署)。

**落地结构:**
```
webapp/frontend/
  astro.config.mjs      # output:'static'; outDir→aiohttp 可读的 dist/;
                        # dev 时 /ws /config /api/* 代理到 localhost:8000
  package.json          # astro 单依赖
  src/
    pages/index.astro          # 单页骨架:三栏 grid(rail/stage/read)+ 两 resizer
    components/
      PlayerRail.astro         # 左栏:4×PlayerConfig + 全局参数 + 开始/重置
      PlayerConfig.astro       # 单玩家手风琴(details/summary + GenomeList)
      GenomeList.astro         # 16 位竖排 list(骨架/功能锁死/插槽下拉)
      Stage.astro              # 中栏:标题 + 竖排图例(live%)+ canvas + 底部 tick/状态
      ReadPanel.astro          # 右栏:6 卡片 + 2 图表 + 排行榜 + 该格明细
    scripts/
      sim.js     # WS 连接 + 帧解析 + start/reset 状态机
      render.js  # drawFrame 合成(从现 app.js 逐字搬,守红线2)
      charts.js  # line() 手绘折线(从现 app.js 搬)
      config.js  # 读 /config、收集 4 玩家 slots、组 {players:[…]} payload
    styles/global.css          # CSS 变量 + 三栏布局(从 mockup 搬)
```

**关键决策:**
- **组件=静态结构(.astro),交互=共享 ES 模块。** Astro 组件渲染骨架 HTML;运行时逻辑(canvas/WS/图表/resizer/状态机)是普通 ES 模块,`<script>` import。不引 React/Vue island。
- **渲染/读数逻辑从现 `webapp/static/app.js` 逐字搬运,不重写。** `drawFrame`(faction 加权混色+density alpha+pixelated)、`line()` 手绘图表。搬运而非重造,杜绝 UI 漂移。
- **4 玩家编辑器:** `PlayerConfig` 渲染 4 次,同一 `GenomeList`(结构来自 /config:locked 灰显、骨架 N0 灰显、6 插槽下拉)。每玩家独立 `slots` 状态(数组,index=faction)。summary 显示阵营色点+名+live%。
- **保留的 mockup 件:** 4 玩家手风琴、竖排基因型 list、竖排图例带 live%、左右栏可拖 resizer(用已修好的 `offsetWidth` 版逻辑,**照搬不优化**)、开始/重置按钮(开始=跑新局,跑起来变重置)。
- **退场的 mockup 件:** ▶播放/⏸暂停、速度滑块、速度指示 —— 全速直播不需要。底部 transport 只剩 `tick X / T` + 状态(未开始/运行中/完成/错误)+「点格子→看 {strain:count}」提示。
- **点格子钻取:** live 读内存(server `/api/cell` 命中 `_LIVE_KEY` 走内存 world),前端照调。

---

## 4. 数据流

```
加载 → config.js fetch /config → {palette,slots,locked,defaults}
     → PlayerRail 渲染 4×PlayerConfig(同结构)+ 全局参数填 defaults
     → canvas 画 tick0 空世界,状态"未开始"

开始 → 收集 4 玩家 slots+全局参数 → {cmd:start, config:{players:[…4],…}}
     → sim.js 开 WS 发出 → 按钮变"重置",状态"运行中"
     → server 全速跑,逐帧 WS 推
     → 每帧:render.drawFrame + ReadPanel 更新(6卡/2图/排行)+ 图例&手风琴 live%
     → {event:done} → 状态"完成"

重置 → 关 WS、清序列、canvas 回 tick0、按钮回"开始"
点格 → /api/cell?path&tick&y&x(live 读内存)→ 该格明细 list
```

---

## 5. 错误处理

- 玩家 slots 非法 → WS `{event:error,msg}` → 状态栏红字,不开跑。
- WS 意外断 → `onclose` 若仍"运行中"→ 状态"已停止"。
- `/api/cell` 失败 → 明细区显"钻取暂不可用,请重试"(沿用现行为)。
- `dist/index.html` 缺失(未 build)→ aiohttp 启动日志明确报"先 build 前端",不静默 500。

---

## 6. 测试

**(1) Python 单元(pytest,现有体系):**
- `init_factions`:四条不同 layout → 四阵营 strain_id 各异、各过 validate;非法第 N 条被拒;单条/None 向后兼容。
- `build_engine_from_config`:新 `players` 结构 → 4 条正确 layout;`players≠4` 报错;非法插槽报错;全局参数仍共享。
- **红线回归:** `encode_frame`/readouts 的 known-answer 测试**必须仍绿**(证明帧格式/读数无漂移)。

**(2) 前端构建冒烟:** `npm run build` 退 0 出 `dist/index.html`;搬运的 JS 模块 `node --check` 过。

**(3) e2e 冒烟:** aiohttp 托管 dist,起一局四玩家不同插槽 → 四阵营色块真不同、live% 四条加总≈1、点格子出明细。

---

## 7. 构建集成 + 收尾

- `astro.config.mjs`:`output:'static'`;`outDir` 指 aiohttp 可读的 `dist/`;dev server 代理 `/ws`、`/config`、`/api/*` → `localhost:8000`。
- aiohttp `make_app`:`/` 与静态根改指 `dist/`(**唯一 server 路由改动**,其余路由不动)。
- **删旧不留影子(ponytail):** Astro `dist/` 经 e2e 验收通过后,`git rm` 掉 `webapp/static/`(index.html/app.js/style.css)。回退靠 git,不养并行死代码。`app.js` 里被搬运的逻辑搬走后即删原文件。
- README 更新:`cd webapp/frontend && npm install && npm run build`,再 `$env:PYTHONPATH='src'; python -m webapp.server` → http://localhost:8000。

---

## 8. 范围边界(本轮不做)

- **不做** 每阵营不同 K/突变率/机制 —— 那是完整非对称角色系统(更大,需专门 brainstorm)。本轮只放开「起始基因型不必相同」这一最小非对称。
- **不做** server 端暂停/调速 —— 全速直播。
- **不做** 时间轴 scrubber / 帧缓存重放。
