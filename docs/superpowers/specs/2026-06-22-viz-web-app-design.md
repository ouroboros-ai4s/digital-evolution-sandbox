# Digital Evolution Sandbox — 可视化 Web App 设计

**日期:** 2026-06-22
**状态:** 设计 — 待用户 review
**作者:** hub (Claude) + 设计评审 subagent(基准核对 + ponytail)

## 0. 概述

为 Digital Evolution Sandbox 引擎做一个单页 web app。默认整屏 = 实时游戏界面(「验收之眼」:肉眼看红皇后动力学——四象限扩张→前沿相遇→填满→四阵营共存、无塌缩/无内战)。「探索分析」和「数据产品展示」作为可折叠侧栏。

引擎是 torch/CUDA Python(浏览器跑不了),所以形态必然是「后端跑引擎、流帧到前端」:后端跑现有 `Engine`,每 tick 把世界状态推给浏览器实时渲染,同时复用现有 `Recorder` 边跑边落 parquet。

**视觉基准:** `docs/mockups/main-screen.html`(brainstorming 期草图,假数据真混色逻辑,throwaway,实现期重写)。

## 1. 范围与边界(已锁)

- **格子 128×128。** 基准设计本体是 512²(`context/.../2026-06-11-16-10-design.md` line 7),但 512² 太大跑不动;128² 是首批跑参,用户已明确锁 128。网格尺寸是 §3(b) 的全局参之一(默认 128,可在配置面板调),但默认与首批一致。
- **BB0 对称局,四阵营同条。** 角色系统(让四阵营不同 = 非对称)是 CLAUDE.md 里另一轮被 HARD-GATE 住、尚未设计的,**本轮不碰**。四始祖株拿同一条 layout,差异靠突变涌现。
- **基元调色板 = 6 个 v1 子集**(N0/F4Nr1/F4Nr4/P_base/P_hotspot/BroadSweep),即 registry 首版实现的精选子集,非基准 22 个可选池全集。
- **本轮只做可视化**,不引入新引擎机制(除一个最小触点,见 §5)。

## 2. 架构

```
浏览器(单页, 纯 HTML/CSS/JS 无框架)
   │  WebSocket(JSON 帧)         HTTP(钻取/回放查询)
   ▼                                  ▼
FastAPI + uvicorn(纯 Python, 轻量装进 basic env, 版本 pin 死)
   │  后台任务驱动
   ▼
现有 Engine(torch/CUDA, RTX 5080)  ──每 tick──▶  现有 Recorder ──▶ parquet(playground 隔离目录)
```

**后端(FastAPI + uvicorn + WebSocket):**
- 纯 Python,不碰 torch CUDA 二进制,轻量装进 conda `basic` env,版本 pin 死(用户已授权可轻量装后端依赖)。
- 一个后台任务跑现有 `Engine`。每 tick 完成后:① 从 world tensors 算紧凑格子帧 ② 算标量/时序读数。两者打包成 JSON 帧经 WebSocket 推给前端。
- **节奏 = 引擎出多快播多快**(已锁):算完一 tick 即推一帧,不限速、不缓冲、不插值。128² 满世界约 2–5 tick/秒,整局约一两分钟。零额外节奏逻辑,且忠实引擎真实速度(不美化,守验收红线)。
- 同时复用现有 `Recorder` 边跑边写 parquet 到 playground 隔离目录(见 §6 数据隔离)。
- 帧传输先用 JSON(后期占用格 ~8k,每帧几十 KB,WebSocket 直发够用)。`# ponytail: JSON frames, switch to binary if profiler says too slow` —— 不预先优化二进制编码。

**前端(单页, 纯 HTML/CSS/JS, 无框架):**
- 主画面 = 128×128 `ImageData`,最近邻放大到 canvas(128² 对 canvas 极小,一个 ImageData 缓冲轻松 60fps,WebGL/库纯属过度)。
- 单图层合成(用户明确要求,不分图层):阵营=色相,各阵营 count 加权混色,总密度=alpha(密度低发暗),争夺格自然呈两阵营混色。
- 读数面板:手绘小 canvas 折线图(不引图表库)+ 数字牌 + 主导株排行榜。
- 「探索分析」「数据产品展示」为可折叠侧栏。

## 3. 配置流程

开始游戏前的配置面板,两块:

**(a) 基因配置(共享 BB0,四阵营同条)**
- 展示 BB0 的 16 位布局。**3 个 locked 功能基元位**(0-indexed 位 1=F4Nr4、位 5=BroadSweep、位 7=P_base,以 `registry.py:130` `_LOCKED` 为准)显示但**置灰只读**——表达「锁死即锁死」并防误改。
- **6 个 mutable 插槽**(0-indexed 位 0,2,3,9,10,13,即 `_SLOTS`)各一个下拉框,选项 = 6 个 v1 基元(N0/F4Nr1/F4Nr4/P_base/P_hotspot/BroadSweep),**默认全选 N0**(= 正典 BB0)。
- 面板顶部一行红线标注:「这是**对称起始基因型**,四阵营全同;不是角色/阵营差异系统。」
- 默认全 N0 时拼出的就是现有 `BB0_TEMPLATE["layout"]`,走**同一条代码路径**,不开模式分支。

**(b) 全局参数(run 级旋钮,非「谁强」系数)**
- 网格(默认 128×128)、K(默认 64)、fill_per_cell(默认 20)、T 总 ticks(默认 450)、seed(默认 0)、z_max(默认 8.0)。

**为什么填插槽不撞 HARD-GATE(评审裁决):** 角色系统那道闸守的是**非对称**(给阵营不同角色,让占比脱离 0.25 生出选择信号)。这里四阵营拿**同一条**填好的 layout,零非对称引入。判别测试:玩家的选择是否打破四阵营对称?不打破 → 不是角色系统,只是「选对称局的起始基因型」(像对称棋盘换个开局)。也不碰「无手写谁强」红线:玩家选的是基元字母不是适应度数字,表型仍由固定 `phenotype()` 裁定,四家全同没有任何**相对**优势被写入。成立强依赖两条不变量(§5 守门校验硬保):① 只暴露 6 个 mutable 槽,locked 位只读;② 四阵营注入同一条 layout。

## 4. 实时读数(单一来源)

右栏读数,全部来自一个**纯函数**(见 §5):
- **数字牌:** tick / 总个体数 / 占用格数 / 活株数(distinct strains)/ N2(有效多样性)/ d_max(最大单株占比)。
- **阵营占比时间线**(主判决曲线):四条线随 tick,全贴 0.25=共存,某条冲顶=塌缩。
- **多样性曲线:** distinct strains + N2 随 tick。
- **主导株排行榜:** top-N 株及其占比/阵营。

**单一来源红线:** 这些量 `analyze_batch.py` 已经在算。本 app **不再开第二套统计代码**——把占比/distinct/N2/d_max/occupied 抽成一个纯函数,live 端喂 world tensor、`analyze_batch` 喂 df,共用一份定义,杜绝两套公式漂移导致验收画面与离线报告对不上。

## 5. 引擎触点(最小)

唯一的引擎改动:
- 给 `world.init_factions` 加可选 `layout` 参数(默认 `BB0_TEMPLATE["layout"]`,保持现行为)。
- 进函数先 **assert 守门校验**:`layout` 的 locked 位(`_LOCKED` 的键)内容 == `_LOCKED` 的值、差异只落在 `_SLOTS` 位;四阵营注入同一条 layout。校验器既是功能也是红线 enforcement——任何破坏对称或改锁死位的请求被拒。
- `get_or_mint` 本就吃任意 layout,除校验器外无别的引擎代码要写。
- **默认 layout 行为不变**,有回归测试兜底。

其余引擎一字不动。读数纯函数(§4)抽取属于新增工具模块,不改引擎逻辑。

## 6. 钻取与回放(分档)

用户要但排次要,按读取成本切两档:

**Tier A(便宜,单帧内查)——必做:**
- 「点格子 → 看该 tick 的 `{strain:count, faction}`」。live 读内存当前帧;回放读该 tick 那一个 parquet row-group 的该格行。无扫描。
- 「拖时间轴重放帧」:跑完的 parquet 每 tick 一个 row-group,随机读某 tick 便宜。

**Tier B(贵,跨 tick 某株时空轨迹)——次要,按需懒算:**
- 用户点某株时**懒算一次该株轨迹 + 缓存**(按 strain memoize)。
- 靠 parquet 谓词下推(`strain == X` filtered read)让 reader 按 row-group 跳读,**不预建索引**。
- caveat:string 字典的 row-group 统计未必剪得干净,最坏接近全扫。可接受(按需/每株一次/已缓存/次要)。`# ponytail: 实测真慢再懒建一次 strain→row_group 轻索引,别预先建`。

回放读的是已记录 parquet 真值,不回写、不美化(守 §7 红线)。

## 7. 红线(数据不被私货污染)

继承基准设计「守门人判据」(`2026-06-11-16-10-design.md` line 18):

- **数据隔离:** playground 跑出的 parquet **必须隔离**(独立目录或 `_playground` 后缀),绝不与「锁定配置的正式采集 run」混进同一产出池。否则「调参→看动力学→再调」的循环会把手工标定污染流回 selop。
- **live 帧忠实张量:** 前端合成混色 OK,但绝不做平滑/插值让前沿「好看」——验收之眼一旦美化就不再是验收。
- **无手写谁强:** 配置只让玩家选基元字母(对称、四家同条),不暴露任何「相对优势」系数。
- **四阵营对称不变量:** 由 §5 守门校验硬保。

## 8. 测试

- 后端帧编码:给定 world tensor → 期望 JSON 帧,单元测试。
- 读数纯函数:给定 `(strain_id, count, faction)` → 期望占比/distinct/N2/d_max,单元测试(并与 `analyze_batch` 同函数复用,锁死单一来源)。
- `init_factions` 守门校验:默认 layout 回归测试(行为不变)+ 非法 layout(改锁死位/破对称)被拒测试。
- 前端是 demo,人工验收画面(不引前端测试框架,YAGNI)。

## 9. 文件结构(预估,实现期细化)

- 后端:`webapp/server.py`(FastAPI app + WebSocket + 后台引擎任务)、`webapp/readouts.py`(读数纯函数,被 server 与 analyze_batch 共用)、`webapp/frame.py`(world tensor → JSON 帧编码)。
- 前端:`webapp/static/index.html` + `app.js` + `style.css`(单页,无框架)。
- 引擎触点:`src/des/world.py`(`init_factions` 加 layout 参 + 校验)。
- playground 数据:`data/playground/`(隔离目录,gitignore)。
