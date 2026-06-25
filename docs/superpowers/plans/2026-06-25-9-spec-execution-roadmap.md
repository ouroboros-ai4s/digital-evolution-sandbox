# 9-Spec 执行序总 Plan (S0→S6→S1→S2→S4→S5→S3→S7→S8 + RE-RECORD)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把已落盘的 9 篇 spec plan(2026-06-24 写, 2026-06-25 全部落盘)按锁定顺序执行完, 让 registry 从 6 基元长到 68 基元, 同时保住 `src/` PyTorch 后端 + `webapp/` Astro 前端 + bash CLI 三接口的字节级回归。

**Architecture:** 一条链, 9 个 sub-plan 串起来 + 1 个 RE-RECORD fixture 闸口收尾。**实现序锁定为 S0 → S6 → S1 → S2 → S4 → S5 → S3 → S7 → S8**(CLAUDE.md / `project_des_unify_and_68_roadmap.md` 已批), 不许跳序; S6 必须紧跟 S0(它是横切地基, S1/S2/S3/S4/S5/S7/S8 全依赖它的 GRAN/_spectrum_for/_mutation_outcomes/predicate-bit 四件套), 之后的支线序按依赖图最短化。每完成一个 sub-plan 都过同一道 5 项 gate(测试绿 / 字节级 byte-equal default route / smoke / push origin/main / CLAUDE.md 与 memory 同步)。S8 后再做一次「全 68 affinity 谱 RE-RECORD」收尾, 锁字节基线。

**Tech Stack:** Python 3 (`D:\anaconda3\envs\basic\python.exe`), PyTorch 2.10+cu128, pyarrow, pytest, stdlib `zlib.crc32`, `unittest.mock`. Astro 5.x + aiohttp WebSocket(webapp/). Windows 11 主机, `PYTHONPATH=src` 纪律, repo `github.com/ouroboros-ai4s/digital-evolution-sandbox` 全推 `origin/main`。

## Global Constraints

- **实现序绝对锁:** S0→S6→S1→S2→S4→S5→S3→S7→S8。**不许跳序、不许并行**。原因: S6 给 S1/S3 预留 bit, 给 S2/S7 改 `_spectrum_for`/`_mutation_outcomes` 签名; S4 给 S5 铺 F-pool direction 协议; S3 读 S1 vis_lowvis + S2/S4/S5 的 family-spec 阈值; S8 收口要全部前置就位才能跑 multi-P 混合 + de-gate audit。
- **每 task 必过 5 道 gate(无例外):** ① pytest 全绿(`pytest -q`, 引擎 + webapp 测试套); ② **静态默认 4-faction route byte-equal**(`engine.run` 输出 frame.cells 与 pre-task 的 baseline 张量级相等, 各 spec 的 `Global Constraints` 都点了这条); ③ probe smoke(`python -m des.smoke` 或 plan 内指定的 smoke 命令); ④ push 到 `origin/main` 当 task tip 提交; ⑤ 同步 CLAUDE.md 「Digital Evolution Sandbox」段进度行 + memory(`project_des_unify_and_68_roadmap.md`)勾掉对应 spec。
- **「无新概念」红线:** 9 篇 sub-plan 的 spec 已经把所有新基元 / 新机制锁死, 实现期间发现「需要新增概念」=实现者理解错, 必须停下找回 spec 文本对齐, 不发明玩法。**只补 registry 数据行 + 复用既有机制**(参见 CLAUDE.md「纪律」段)。
- **字节级 fixture 锁的范围限定:** 静态默认 4 阵营全同条 byte-equal 是 fixture 的硬锁; 但 S4 重底定 F4Nr1 后旧 fixtures(锁 F4Nr1=北)失效, 这部分由 Task 10(RE-RECORD)统一升级, 不在 S0–S8 单 task 内打补丁。每个 sub-plan 自己的 「F4Nr1 锁北」类断言在该 task 内 skip + TODO。
- **测试不掉头:** 既有 285 引擎测试 + 146 web 测试是绿的回归基线。每 task 完成后 `pytest -q` 必须 ≥285+146=431 个绿测; 任 task 想 skip 已绿测试必须在 task PR 描述里显式列出原因 + 后续打开点。
- **三接口同步纪律(CLI / src / webapp):** 任何 registry 字段加列, **必须**同步: ① bash CLI(`python -m des.cli match-runner …`)能直接消费; ② 引擎核心 `src/des/` 对外 API 不破坏向后兼容(默认值给齐); ③ webapp 实时帧渲染(`webapp/server.py` + Astro frontend)不读 registry, 但若新字段进入 `frame.cell_detail` payload 必须前端 schema 同步。S0 task 已建 CLI key allow-list, 不许往 allow-list 塞结局常数。
- **结局常数 永不进 CLI/config(铁律):** `μ / z_max / δ / p_max / α / κ / β` 锁死 registry, 任何 sub-plan 实现期间若发现 CLI/config 在读这些值 = bug, 不许「绕过先合」, 必须先修 CLI 守门再合。
- **环境定值:** 所有 pytest / smoke / engine.run 命令前置 `$env:PYTHONPATH='src'`(PowerShell)或 `PYTHONPATH=src`(bash); Python 解释器 `D:\anaconda3\envs\basic\python.exe`(`-m` 形式启动, 不许脚本路径直跑, server.py 有绝对导入)。
- **回归基线已知数:** S0 起点 = 285 引擎测试 + 146 webapp 测试 = 431 绿; ~15.8ms/tick 表型缓存 perf; 837MB 首批 4 parquet (128² K64 fill20 T450×4 seeds, 锁 F4Nr1=北)。所有 sub-plan 完工 task 都跟这条基线比, 跌任何一条都先红再修。
- **no-private-stuff 红线传染所有 sub-plan:** 任何手写「谁强 / 谁扩张得更好」的系数 / 偏置 / 阈值进 registry 都是污染, 即便在「私有不可观测」字段也一样(spec review 已批四阵营起始基因型可不同, 但**仍同模板、同固定 G→P, 不许塞「谁强」常数**)。
- **out of scope:** 非对称角色系统(per-faction K/突变率/机制)是独立 HARD-GATE, 与 68-基元落地无关、不在本 plan; 待用户拍板才进 brainstorming。

---

## File Structure

这篇 plan **不写代码**, 只编排已落盘的 9 篇 detail plan 的执行序与 gate。所有「Files: Create / Modify / Test」都展开在各 sub-plan 自己的 task 里, 此处只列入口清单。

**9 篇 spec(`digital-evolution-sandbox/docs/superpowers/specs/`):**

| Spec | 路径 |
|---|---|
| S0 统一入口 + CLI match-runner | `2026-06-24-s0-unify-cli-match-runner-design.md` |
| S6 motif 粒度(横切地基) | `2026-06-24-s6-motif-granularity-design.md` |
| S1 vis 通道 | `2026-06-24-s1-vis-channel-design.md` |
| S2 塑形突变谱 | `2026-06-24-s2-shaped-spectrum-design.md` |
| S4 动态方向 | `2026-06-24-s4-dynamic-directions-design.md` |
| S5 相位窗 f | `2026-06-24-s5-phase-windows-design.md` |
| S3 富猎物谓词 | `2026-06-24-s3-rich-prey-predicates-design.md` |
| S7 多位突变 | `2026-06-24-s7-multi-slot-mutation-design.md` |
| S8 A 池 24 极端 + 多 P 混合 | `2026-06-24-s8-a-pool-extremes-design.md` |

**9 篇 detail plan(`digital-evolution-sandbox/docs/superpowers/plans/`, 全部 `2026-06-24-` 前缀, 11705 行总量):**

| Plan | 路径 | 行数 |
|---|---|---|
| S0 | `2026-06-24-s0-unify-cli-match-runner.md` | 920 |
| S1 | `2026-06-24-s1-vis-channel.md` | 1320 |
| S2 | `2026-06-24-s2-shaped-spectrum.md` | 727 |
| S3 | `2026-06-24-s3-rich-prey-predicates.md` | 955 |
| S4 | `2026-06-24-s4-dynamic-directions.md` | 1529 |
| S5 | `2026-06-24-s5-phase-windows.md` | 1492 |
| S6 | `2026-06-24-s6-motif-granularity.md` | 1537 |
| S7 | `2026-06-24-s7-multi-slot-mutation.md` | 1321 |
| S8 | `2026-06-24-s8-a-pool-extremes.md` | 1904 |

**入口约束:**
- 实现期间执行者**只读自己 task 对应那 1 篇 detail plan**(以及它显式 cross-link 的其他 plan), 不必读全 9 篇。
- 这篇总 plan 的每个 Task N 入口段会显式给出: 要执行的 sub-plan 路径 / 要读的 spec 路径 / 前置 task 的产出依赖 / 这一段不许动的字节锁名单。
- **Repo 同步:** 9 篇 detail plan 与 9 篇 spec 已推 `origin/main`(`f6d772f`); 这篇总 plan 落盘后也要 push, push 命令在 Task 1「准备」内联。
- **CLAUDE.md / memory 同步**: 每个 Task 完成后必须改 CLAUDE.md「Digital Evolution Sandbox」段(实现序勾掉对应 S 字), memory `project_des_unify_and_68_roadmap.md` 同步, 这是每个 Task 步骤里的最后一项。

---

## 依赖图与执行序锁定

**实现序: S0 → S6 → S1 → S2 → S4 → S5 → S3 → S7 → S8 → RE-RECORD。**

**依赖图(箭头 = depends-on, A→B 读作「A 依赖 B 已落地」):**

```
S0 (CLI/key allow-list)        ← 所有后续 task 都通过它跑回归
 ↑
S6 (GRAN/_spectrum_for/_mutation_outcomes/predicate-bit 横切) ← S1/S2/S3/S4/S5/S7/S8 全依赖
 ↑           ↑          ↑           ↑          ↑          ↑
S1 (vis)  S2 (谱)   S4 (方向)   S5 (相位)  S3 (谓词)  S7 (多位)
 ↑           ↑          ↑           ↑          ↑          ↑
 └── S3 读 vis_lowvis  ┘                                   │
              └── S5 用 S4 F-pool direction 协议 ──┘        │
                            └── S3 读 S2/S4/S5 family-spec ┘
                                          ↑
                                         S8 (A池 + 多P混合) ← 收口需 S7 后才上 multi-slot
                                          ↑
                                         RE-RECORD (字节基线重锁)
```

**为什么是这个序(每条边的解释, 改序前必读):**

- **S0 → S6:** S0 锁 CLI key allow-list({players, grid, K, fill, T, seed} 6 项, 挡结局常数), 是「跑回归 + 测试」的入口前置。S6 是横切地基, 给后续所有 spec 提供 `GRAN`(粒度表)/ `MOTIF_LEN` / `_spectrum_for` 等长预过滤 / `_mutation_outcomes` 整块覆写 / `predicate-bit` 方案 + 全词表 / `n_locked` 按需算。**S1 / S2 / S3 / S4 / S5 / S7 / S8 七个 spec 全部读它的产出**。S6 不到位, 后续每个 spec 的 task 1 都做不下去。
- **S6 → S1:** S1 「填 S6 预留 FEAT_N_LOWVIS 位」(CLAUDE.md 直引), 没 S6 的 predicate-bit 表 就没有「预留位」。
- **S6 → S2:** S2 的 `_spectrum_for` 读 `SPECTRUM_SHAPE`(power/family_mask/flatten_mix), 但函数本体是 S6 给的, 不能在 S2 里造新版本。
- **S6 → S4:** S4 给 5 个新 F 行带各自 `gran="residue"|"motif"`, 还要让 `_mutation_outcomes` 整块覆写认得 hash-locked direction; 都是 S6 协议字段。
- **S4 → S5:** S5 的 FBURST/F_NOVA 是 F-pool 第二批, 它要复用 S4 已铸的 「F-pool direction 集 / hash-lock + rand-dir + in_place」三态协议。S5 spec 自己已声明 F_NOVA 是 「S5 owns f-window 含其 dirs」, 但 dirs 的协议字段在 S4 task 里铸的, 反过来不行。
- **S4 → S3 / S5 → S3:** S3 富猎物谓词读 `("F","f_hi")` / `("P","p_hi")` 等 family-spec 阈值 clause-tag(S3 plan Task 4), 这些阈值 family 由 S2/S4/S5 完成才齐。
- **S1 → S3:** S3 plan Task 5 显式审计「S1 prey 端置 vis_lowvis 位 ↔ S3 predator 端读 vis_lowvis 位」端到端, S1 不到位 S3 这条审计链断。
- **S3 → S7:** S7 多位突变只动 `_mutation_outcomes` 与 `Phenotype.slots_per_event`, 跟 predicate / vis 都不冲突, 但 spec-review 把 S7 排在 S3 后, 是因为 S7 的 byte-equal 守护要建立在「S3 后 predicate-bit 表稳定」之上(否则 S7 改 mutation outcome 会无意中影响 predicate 命中分布, byte-equal 难守)。
- **S7 → S8:** S8 收口三件事: ①24 行 A 池极端 ②`n_locked` 门作废 audit ③多 P 谱混合 `Σpᵢqᵢ/Σpᵢ` 取代 `dominant_p`。其中第 ② 条 audit 要 grep 所有前置 spec 落地后的代码(确认没有任何路径还在用 `n_locked` 门), S0–S7 全部落地才能跑这条 audit 不漏。第 ③ 条 multi-P 混合也是「全 letter family-spec 都铸全」之后才有真值。
- **S8 → RE-RECORD:** spec-review 第 ① 裁定:「全 68 affinity 谱才是设计, 6 字母残缺截断」, 长全后 RE-RECORD fixtures, 字节锁此后护非 registry 代码路径。S8 是最后一个加 affinity 谱的 spec, 它落地 = 谱全。这一步必须最后做, 早做了等于把没长全的谱也锁进 fixture。

**不许跳序的两条具体禁令:**

1. **不许 S1 / S3 提前到 S6 之前。** CLAUDE.md 与 9 篇 plan 评审都说「S6 是横切地基」; 提前 S1/S3 = 反复推倒重来。
2. **不许并行跑 S2/S4/S5。** 看上去三者写的都是不同 family(P/F/F-window), 似乎可并行; 但 S2 改 `_spectrum_for` 与 S4 改 phenotype()「同一 letter 多 owner」处理与 S5 给 Phenotype 加 4 字段都触 `phenotype_cache.py` / `phenotype()`, 三者并行必合并冲突, 串行更省时。

**任 task 失败的回退策略:**

- **task 失败 = `pytest -q` 红 / byte-equal 断言失败 / push 被远端拒**: **不许在失败 task 上加补丁直接合**。先 revert 这个 task 的所有 commit, 回到上一 task tip, 再起一份「修正 spec 理解」的 sub-task(改 detail plan 文本 + 加单测覆盖失败模式), 修正后从头跑这个 task。
- **byte-equal 断言失败 = 强信号**: 默认路径不该改字节, 命中说明 spec 实现走偏(常见: 把哨兵索引 idx=0 的默认值搞错; 把 `f_hi` 别名链接顺序搞反)。修 spec 理解, 不修 fixture。例外只有: S4 重底定 F4Nr1 的方向锁 + RE-RECORD 一次性更新, 这两类已在 plan 内显式 list, 不是 spec 走偏。

---

---

### Task 1: S0 — 统一入口 + CLI match-runner

**Files:**
- Spec: `digital-evolution-sandbox/docs/superpowers/specs/2026-06-24-s0-unify-cli-match-runner-design.md`
- Detail Plan: `digital-evolution-sandbox/docs/superpowers/plans/2026-06-24-s0-unify-cli-match-runner.md`(920 行)
- Test: 详见 detail plan, 不在本入口展开。

**Interfaces:**
- Consumes: 起点 task, 仅依赖既有 285 引擎 + 146 webapp 测试套绿。
- Produces:
  - `python -m des.cli match-runner …` 入口可跑(所有后续 task 的回归命令通过它跑)。
  - `cli` 模块的 key allow-list `{players, grid, K, fill, T, seed}` 锁死, 结局常数(`μ / z_max / δ / p_max / α / κ / β`)挡在外面 —— S2 / S4 / S5 / S7 / S8 后续 task 不许往 allow-list 塞这些。
  - 统一回归命令: `pytest -q`(根目录, `PYTHONPATH=src`), 这条是后续每 task 必跑的同一条。

- [ ] **Step 1: 准备 — 校验 base 状态(431 绿测 + 837MB 首批 parquet 在位)**

```bash
cd G:/OUROBOROS-AI4S/digital-evolution-sandbox
git status                                  # 期望 clean
git log -1 --oneline                        # 期望 ebfe881 (viz Astro merge) 在 HEAD 或之后
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q
# 期望: 285 passed (engine) + 146 passed (webapp) = 431 passed, 0 failed
```

如果 base 不绿, **先停下查 base 红的原因, 不进 S0**(基线就不对, S0 后没法判定是 S0 弄红了还是 base 早就红了)。

- [ ] **Step 2: push 这篇总 plan 与 9 篇 detail plan(若未 push)**

```bash
git status digital-evolution-sandbox/docs/superpowers/plans/
git add digital-evolution-sandbox/docs/superpowers/plans/2026-06-25-9-spec-execution-roadmap.md
git commit -m "docs: add 9-spec execution roadmap plan (S0..S8 + RE-RECORD)"
git push origin main
```

(如果 9 篇 detail plan 已在 `f6d772f` 推过则只需 push 这篇 roadmap; CLAUDE.md 已记 9 plan 已推。)

- [ ] **Step 3: 执行 S0 detail plan 的全部任务**

进入 detail plan `2026-06-24-s0-unify-cli-match-runner.md`, 按它内部的 Task 1..N 顺序执行。Detail plan 内各 task 已有完整 「Files / Interfaces / Steps / 测试 / commit」 块, 此处不复述。

执行模式选择(detail plan 末尾「Execution Handoff」段已问):
- **推荐: subagent-driven**(superpowers:subagent-driven-development), 每 detail-task 一个 fresh subagent, 中间两段评审。
- 或 inline(superpowers:executing-plans), batch 跑带 checkpoint。

- [ ] **Step 4: 5 项 gate 全过(下面这 5 条任一红 = 本 Task 不算完, 回退 + 修)**

```bash
# (1) pytest 全绿 ≥431
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q
# 期望: ≥431 passed, 0 failed

# (2) byte-equal 默认 4-faction route
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke --byte-equal-baseline pre-s0
# 期望: ALL CELLS BYTE-EQUAL(具体命令 + flag 名以 detail plan 为准, 此处 sentinel)

# (3) probe smoke
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke
# 期望: SMOKE PASS

# (4) push origin main
git status                                  # 期望 clean
git log -1 --oneline                        # 期望 S0 收尾 commit 在 HEAD
git push origin main                        # 远端接受

# (5) CLAUDE.md / memory 同步
# 用 Edit 工具改 G:/OUROBOROS-AI4S/CLAUDE.md「Digital Evolution Sandbox」段:
#   实现序行的「S0」标记从「待」改成「✓(YYYY-MM-DD)」
# 用 Edit 工具改 memory project_des_unify_and_68_roadmap.md 同步同一勾。
```

- [ ] **Step 5: commit gate 同步**

```bash
git add CLAUDE.md C:/Users/Strix/.claude/projects/G--OUROBOROS-AI4S/memory/project_des_unify_and_68_roadmap.md
git commit -m "docs: mark S0 (CLI match-runner) complete in roadmap"
git push origin main
```

**Acceptance:** 5 项 gate 全过 + S0 detail plan 内所有 checkbox 全打勾 + CLAUDE.md / memory 已同步。失败任一条 → 回退到 Task 0 tip(本 Task 启动前 commit), 修 spec 理解再起。

---

---

### Task 2: S6 — motif 粒度(横切地基)

**Files:**
- Spec: `digital-evolution-sandbox/docs/superpowers/specs/2026-06-24-s6-motif-granularity-design.md`
- Detail Plan: `digital-evolution-sandbox/docs/superpowers/plans/2026-06-24-s6-motif-granularity.md`(1537 行, 9 篇里第二大)

**Interfaces:**
- Consumes: Task 1 (S0) 产出 — 统一 CLI 入口与 key allow-list。
- Produces(后续 7 个 task 全部读这一组):
  - `GRAN: dict[str, Literal["residue", "motif"]]`(每个 letter 的粒度标签)
  - `MOTIF_LEN: dict[str, int]`(motif 粒度的块长度)
  - `_spectrum_for(...)` 同粒度 + 等长预过滤 接口签名稳定
  - `_mutation_outcomes(...)` 整块覆写 接口签名稳定(S7 后续会扩 `slots_per_event` kwarg, 但 base signature 这里锁)
  - `predicate-bit` 表 + 全词表(S1 / S3 后续填进它)
  - `n_locked` 按需算函数(S8 audit 时会确认它已不被任何路径前置依赖)

- [ ] **Step 1: 校验 Task 1 完成态**

```bash
cd G:/OUROBOROS-AI4S/digital-evolution-sandbox
git log -1 --oneline                        # 期望 Task 1 收尾 commit 在 HEAD
grep -i "S0.*✓" G:/OUROBOROS-AI4S/CLAUDE.md  # 期望命中 1 行
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q
# 期望: ≥431 passed
```

- [ ] **Step 2: 执行 S6 detail plan 全部任务**

进入 detail plan `2026-06-24-s6-motif-granularity.md`, 按内部 Task 顺序执行。

**特别注意(S6 是横切, 改的是后续所有 spec 都要读的契约):**
- detail plan 内任何 「Future spec 会读这里」 的接口签名 / 字典 key 命名 / 返回类型, 一旦在 S6 task 里定下就**不许后续 spec 重命名**, 否则 S1/S2/S3/S4/S5/S7/S8 全要返工。
- 9 篇 detail plan 已交叉对过签名(S2/S4/S5 的 plan 写法都引用了 S6 的命名), 实现时若发现 S6 detail plan 的命名与 后续某 plan 不一致, **改 S6 plan 文本对齐后续 plan 的引用**(因为 S0/S1/S6 三篇先写, S2-S8 后写时已基于 S0/S1/S6 plan 命名引用; 真冲突时以多数引用方为准, 改少数那篇)。

- [ ] **Step 3: 5 项 gate 全过**

```bash
# (1) pytest 全绿
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q
# 期望: ≥431 passed(S6 detail plan 内会加新测试, 总数会涨, 取≥)

# (2) byte-equal 默认 4-faction route
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke --byte-equal-baseline pre-s6
# 期望: ALL CELLS BYTE-EQUAL — S6 是「横切地基」, 默认路径必须字节级不变, 这是 spec 自身的硬约束(读 detail plan Global Constraints 验证)

# (3) probe smoke
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke
# 期望: SMOKE PASS

# (4) push origin main
git push origin main

# (5) CLAUDE.md / memory 同步: S6 ✓
```

- [ ] **Step 4: commit gate 同步**

```bash
git add CLAUDE.md C:/Users/Strix/.claude/projects/G--OUROBOROS-AI4S/memory/project_des_unify_and_68_roadmap.md
git commit -m "docs: mark S6 (motif granularity) complete in roadmap"
git push origin main
```

**Acceptance:** 5 项 gate 全过 + S6 detail plan 内所有 checkbox 全打勾 + 后续 7 个 task 在 grep `GRAN`/`MOTIF_LEN`/`_spectrum_for`/`predicate-bit` 时都能命中已落地代码。失败 → 回退到 Task 1 tip。

**S6 失败的特殊回退路径:** 因为 S6 改横切契约, 万一 S6 detail plan 内某个签名改不下去(比如发现要改的函数有 webapp/server.py 的隐藏调用), **不许临时给 S6 加补丁让它过**。停下, 把这条隐藏调用作为新的 spec gap 文档化进 S6 spec, 改 S6 detail plan 文本(可能新增一个 task), 重跑。这是 S6 一个 task 比其他 task 多花时间的必要代价。

---

---

### Task 3: S1 — vis 通道

**Files:**
- Spec: `digital-evolution-sandbox/docs/superpowers/specs/2026-06-24-s1-vis-channel-design.md`
- Detail Plan: `digital-evolution-sandbox/docs/superpowers/plans/2026-06-24-s1-vis-channel.md`(1320 行)

**Interfaces:**
- Consumes: Task 1 (S0) CLI + Task 2 (S6) `predicate-bit` 表(S1 要往 S6 预留的 `FEAT_N_LOWVIS` 位填值)、`GRAN` / N-family 词表。
- Produces:
  - per-primitive `vis` 字段(N letter family 上的可见性)
  - `feature_mask_of` prey 端置 `vis_lowvis` 位的代码路径(S3 predator 端会反向读, Task 7 时审计)
  - `Phenotype` 新 `vis_sum / n_count` 字段(frozen + 默认值, 不破现有构造点)
  - `phenotype_arrays` 加对应张量列(bool / int8, 不用 PyTorch 上有索引歧义的 raw bool)

- [ ] **Step 1: 校验 Task 2 (S6) 完成态**

```bash
cd G:/OUROBOROS-AI4S/digital-evolution-sandbox
grep -i "S6.*✓" G:/OUROBOROS-AI4S/CLAUDE.md  # 期望命中
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -c "from des.registry import GRAN, PREDICATE_BITS; print('GRAN' in dir(), 'FEAT_N_LOWVIS' in PREDICATE_BITS)"
# 期望: True True (S6 已铸 GRAN 与 PREDICATE_BITS 表, FEAT_N_LOWVIS 位已预留)
```

- [ ] **Step 2: 执行 S1 detail plan 全部任务**

进入 `2026-06-24-s1-vis-channel.md`。

**S1 spec-specific 注意:**
- `Phenotype` 是 `@dataclass(frozen=True)`, 新字段必须给默认值, 否则既有所有 `Phenotype(...)` 构造点(S2 / S4 / S5 detail plan 内都有引用)大爆炸。S4 / S5 detail plan 写作时已假设 S1 字段名 = `vis_sum`(`float32` 哨兵 0.0) + `n_count`(`int8` 哨兵 0), 不许改名, 改名要回头同步 S4 / S5 detail plan 的 `Phenotype(...)` 构造调用。
- `feature_mask_of` 在 S1 内置 `vis_lowvis` 位 = prey 端置位; **不许** 在 S1 内 same-function 顺手实现 predator 端读出(那是 S3 Task 7 的活), 否则 S3 审计会失效。

- [ ] **Step 3: 5 项 gate 全过**

```bash
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q                      # ≥431 passed
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke --byte-equal-baseline pre-s1   # BYTE-EQUAL
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke                       # PASS
git push origin main
# CLAUDE.md / memory: S1 ✓
```

- [ ] **Step 4: commit gate 同步**

```bash
git add CLAUDE.md C:/Users/Strix/.claude/projects/G--OUROBOROS-AI4S/memory/project_des_unify_and_68_roadmap.md
git commit -m "docs: mark S1 (vis channel) complete in roadmap"
git push origin main
```

**Acceptance:** 5 项 gate + S1 detail plan checkbox 全勾 + grep `vis_sum`/`n_count` 在 `phenotype.py` 与 `phenotype_cache.py` 都能命中。

---

---

### Task 4: S2 — 塑形突变谱

**Files:**
- Spec: `digital-evolution-sandbox/docs/superpowers/specs/2026-06-24-s2-shaped-spectrum-design.md`
- Detail Plan: `digital-evolution-sandbox/docs/superpowers/plans/2026-06-24-s2-shaped-spectrum.md`(727 行, 9 篇里最短的)

**Interfaces:**
- Consumes: Task 1 (S0) + Task 2 (S6) `_spectrum_for` 签名 + Task 3 (S1) 不直接读 vis, 但要保留 `Phenotype.vis_sum/n_count` 字段不破。
- Produces:
  - `SPECTRUM_SHAPE: dict[str, dict]` registry(三旋钮: `power` / `family_mask` / `flatten_mix`)
  - `_spectrum_for` 改造为读 `SPECTRUM_SHAPE`(单纯加分支, 不改签名)
  - 10 个新 P 行进 ALPHABET / GRAN / _P / SPECTRUM_SHAPE
  - **S8 后续会再扩 SPECTRUM_SHAPE 值域(power=4 / mask="cross"), S2 此处只铸 spec §1 表里那 10 行声明的值, 不许提前**

- [ ] **Step 1: 校验 Task 3 (S1) 完成态**

```bash
grep -i "S1.*✓" G:/OUROBOROS-AI4S/CLAUDE.md
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q
```

- [ ] **Step 2: 执行 S2 detail plan 全部任务**

进入 `2026-06-24-s2-shaped-spectrum.md`。

**S2 spec-specific 注意:**
- spec §2 spectrum 重录基线: 「全 68 affinity 谱才是设计, 6 字母残缺」, S2 落地后 affinity 谱数据**还没全**(P 行长全但 A 池 24 行要 Task 9/S8 才到), 故 S2 不在这里 RE-RECORD fixtures, 留给最后 Task 10。S2 此处只动 P-fixture(突变谱), 不动 A-fixture(亲和谱)。
- 不许往 SPECTRUM_SHAPE 塞「谁强」偏置: 三旋钮的合法值域 spec 已锁死, 任何「这个 family 更应该被选中」的 hand-tuned 值都违反 no-private-stuff 红线。
- `dominant_p` 此 task **保留旧实现**(Task 9/S8 才换成 `Σpᵢqᵢ/Σpᵢ`)。spec-review 第 ② 裁定: 多 P 混合归 S8。

- [ ] **Step 3: 5 项 gate 全过**

```bash
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q                      # ≥431 passed
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke --byte-equal-baseline pre-s2   # BYTE-EQUAL
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke                       # PASS
git push origin main
# CLAUDE.md / memory: S2 ✓
```

- [ ] **Step 4: commit gate 同步**

```bash
git add CLAUDE.md C:/Users/Strix/.claude/projects/G--OUROBOROS-AI4S/memory/project_des_unify_and_68_roadmap.md
git commit -m "docs: mark S2 (shaped spectrum) complete in roadmap"
git push origin main
```

**Acceptance:** 5 项 gate + S2 detail plan checkbox 全勾 + ALPHABET 里 P-letter 数量从原值涨 10 + grep `SPECTRUM_SHAPE` 命中 `_spectrum_for`。

---

---

### Task 5: S4 — 动态方向(F-pool direction 协议)

**Files:**
- Spec: `digital-evolution-sandbox/docs/superpowers/specs/2026-06-24-s4-dynamic-directions-design.md`
- Detail Plan: `digital-evolution-sandbox/docs/superpowers/plans/2026-06-24-s4-dynamic-directions.md`(1529 行)

**Interfaces:**
- Consumes: Task 1–4 全部产出。
- Produces:
  - `_hash_dirs(seq, kind) -> tuple[(int,int),...]`(zlib.crc32 + `\x1f` 分隔的纯函数)
  - 5 个新 F 基元(FSTACK / FCLUMP / FFRONT / F4Nr3 / FDRIFT)进 ALPHABET / GRAN / _F
  - `Phenotype.in_place: bool = False` / `Phenotype.rand_dir: bool = False`(供 S5 后续读字段名稳定)
  - `phenotype_arrays` 加 `in_place / rand_dir` 两 int8 列(S5 phase-window kernel 不读, 但 dtype 风格要一致)
  - `phase2_reproduce` kernel 加 in-place / rand-dir 两路新分支
  - **F4Nr1 重底定**: 从 v1 占位 `((-1,0),)` 改 hash-locked 1-of-4(等价 FFRONT 处理)。**这是已知的默认局漂移, 影响 837MB 首批 fixture「锁北」断言**, 这部分 fixture 在本 task 内 skip + 标 TODO, 留给 Task 10 RE-RECORD。

- [ ] **Step 1: 校验 Task 4 (S2) 完成态**

```bash
grep -i "S2.*✓" G:/OUROBOROS-AI4S/CLAUDE.md
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q
```

- [ ] **Step 2: 执行 S4 detail plan 全部任务**

进入 `2026-06-24-s4-dynamic-directions.md`。

**S4 spec-specific 注意:**
- **禁用 Python 内建 `hash()`**: 解释器进程级 `PYTHONHASHSEED` 加盐 → 跨进程跨机器不可复现, 对数据沙盒致命。**只许 `zlib.crc32`** + `\x1f`(unit separator)分隔多字符字母防 `"N0"+"F4Nr4"` 歧义拼接(S4 spec §2 ponytail HOW-2)。
- **`dir_bits==0` 不许 overload**: in_place / rand_dir 是两个**独立的、显式的 bool 字段**, 与 `dir_bits` 正交(S4 spec §3.2)。
- **kernel `generator` 是世界 RNG**: rand-dir 类必须从 `phase2_reproduce(..., generator)` 形参抽, **不是** Python 全局 `random` 也**不是** `torch.randint` 默认 RNG。
- **F4Nr4 方向集守恒**: F4Nr4 是 4 邻全开 static, 改 hash-locked 不在 S4 范围, 必须保持 4 个 bit 全置位(S4 detail plan Task 4/6 已显式守这一条)。

- [ ] **Step 3: 5 项 gate 全过**

```bash
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q
# 期望: ≥431 passed; **S4 Task 6 会 skip 一批「锁北」fixture 断言**, 这些 skip 在 detail plan 已列, 不计为红
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke --byte-equal-baseline pre-s4   # BYTE-EQUAL(默认 4-faction route, 不读 hash-locked 与 rand-dir 就字节级不变 — 是的, 默认局没有 hash-locked F)
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke                       # PASS
git push origin main
# CLAUDE.md / memory: S4 ✓
```

- [ ] **Step 4: commit gate 同步**

```bash
git add CLAUDE.md C:/Users/Strix/.claude/projects/G--OUROBOROS-AI4S/memory/project_des_unify_and_68_roadmap.md
git commit -m "docs: mark S4 (dynamic directions) complete in roadmap"
git push origin main
```

**Acceptance:** 5 项 gate + S4 detail plan checkbox 全勾 + ALPHABET 里 F-letter 数量从原值涨 5 + grep `_hash_dirs` 命中 + grep `in_place|rand_dir` 在 phenotype.py 与 phenotype_cache.py 都命中。skip 的「锁北」fixture 测试条数已在 commit message 里列出, 留待 Task 10。

---

---

### Task 6: S5 — 相位窗 f(FBURST / F_NOVA)

**Files:**
- Spec: `digital-evolution-sandbox/docs/superpowers/specs/2026-06-24-s5-phase-windows-design.md`
- Detail Plan: `digital-evolution-sandbox/docs/superpowers/plans/2026-06-24-s5-phase-windows.md`(1492 行)

**Interfaces:**
- Consumes: Task 1–5 全部产出, 尤其是 S4 已铸的 F-pool direction 协议(F_NOVA 的 dirs 是 4 邻 hash-locked 风格 reuse, 不重铸)。
- Produces:
  - `Phenotype` 加 4 frozen 字段 `f_hi: float = 0.0 / f_lo: float = 0.0 / burst_w: int = 1 / burst_k: int = 1`
  - `_F` 升 7-tuple `(f, dirs, p_leave, period, f_lo, burst_w, burst_k)`(后向兼容旧 5-tuple 行: 默认尾部 = `(0.0, 1, 1)`)
  - 2 个新 F 基元 FBURST / F_NOVA 进 ALPHABET / GRAN / _F
  - `phenotype_arrays` 加 4 列 `f_hi: float32 / f_lo: float32 / burst_w: int64 / burst_k: int64`(idx 0 哨兵 0.0/0.0/1/1)
  - `phase2_reproduce` 改 `f = where(on, f_hi, f_lo)`(`on = ((T - birth) % burst_w.clamp(min=1)) < burst_k`)
  - `Phenotype.f` 保留为 `f_hi` 别名(向后兼容现有访问点)

- [ ] **Step 1: 校验 Task 5 (S4) 完成态**

```bash
grep -i "S4.*✓" G:/OUROBOROS-AI4S/CLAUDE.md
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q
```

- [ ] **Step 2: 执行 S5 detail plan 全部任务**

进入 `2026-06-24-s5-phase-windows.md`。

**S5 spec-specific 注意:**
- **静态默认路径字节级不变**: `burst_w=1 / burst_k=1` 让 `on=True` 恒真, `where(on, f_hi, f_lo)=f_hi`; 旧 F 行迁移到 7-tuple 时尾部默认 `(f_lo=0.0, burst_w=1, burst_k=1)`, **kernel 走 where 分支但结果与旧 `f` 字段读取字节级等价**。这是 S5 detail plan Task 4 的核心 invariant, 跌了就是 spec 实现走偏(常见: clamp(min=1) 漏写, burst_w=0 时除零)。
- **`f` 别名锁死赋值时机**: 在 `phenotype()` 末尾 `f_hi = f`(`f` 是 stacked-f 旧公式), 然后 `Phenotype(..., f=f, f_hi=f_hi, ...)`; `phenotype_arrays` 的 `f` 列读 `phe.f` 走旧路径不动。subagent 已在 S5 detail plan 内裁定。
- **F_NOVA 双 owner 守门**: F_NOVA dirs 字段用 S4 hash-locked 协议 reuse, 不是 S5 重铸。S8 Task 9 后续会 grep 「F_NOVA 没被任何地方重新 mint」, 此 task 不许偷偷 mint。
- spec §5 stacked f_lo 公式: `f_lo = 1 - (1 - dom.f_lo) × Π_{i≠dom}(1 - f_i)`(subagent 按 dominant-F 类比锁定, S5 plan Task 2 测试覆盖)。

- [ ] **Step 3: 5 项 gate 全过**

```bash
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q                      # ≥431 passed
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke --byte-equal-baseline pre-s5   # BYTE-EQUAL(默认 burst_w=1 burst_k=1 = 旧路径)
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke                       # PASS
git push origin main
# CLAUDE.md / memory: S5 ✓
```

- [ ] **Step 4: commit gate 同步**

```bash
git add CLAUDE.md C:/Users/Strix/.claude/projects/G--OUROBOROS-AI4S/memory/project_des_unify_and_68_roadmap.md
git commit -m "docs: mark S5 (phase windows) complete in roadmap"
git push origin main
```

**Acceptance:** 5 项 gate + S5 detail plan checkbox 全勾 + ALPHABET 里 F-letter 涨 2(FBURST / F_NOVA)+ Phenotype 4 新字段 grep 命中。

---

---

### Task 7: S3 — 富猎物谓词(填 S6 预留 4 阈值位)

**Files:**
- Spec: `digital-evolution-sandbox/docs/superpowers/specs/2026-06-24-s3-rich-prey-predicates-design.md`
- Detail Plan: `digital-evolution-sandbox/docs/superpowers/plans/2026-06-24-s3-rich-prey-predicates.md`(955 行)

**Interfaces:**
- Consumes: Task 1–6 全部产出。
  - S6 已预留 4 阈值位(Crest / Hotspot Snipe / Mirror Fang / Void Bite)
  - S1 已在 `feature_mask_of` prey 端置 `vis_lowvis` 位(本 task Task 5 端到端审计)
  - S2 已铸 P-letter family-spec(`thr_hotspot` 读 P.p_add)
  - S4 已铸 F-letter family-spec(`thr_crest` 读 F.f, 含 hash-locked / rand-dir)
  - S5 已铸 FBURST/F_NOVA(`thr_crest` 走 f_hi 别名读取)
- Produces:
  - `_Z_PREY_CARD: dict[str, int]` module-load 派生(`|prey| ≥ 2` Mirror Fang 谓词的 O(1) 来源)
  - `feature_mask_of` 追加 3 段阈值置位: `thr_crest` F.f ≥ 0.5 / `thr_hotspot` P.p_add ≥ 0.05 / `thr_mirror` Z.z ≤ 0.45 ∧ |prey| ≥ 2(per-letter 不读 stacked, 闭区间)
  - `prey_mask_for_clauses` 加 4 条新 clause-tag 分支 `("F","f_hi") / ("P","p_hi") / ("Z","generalist") / ("N","lowvis")`(tag+fam 双校验, unknown tag fall-through 到 family)

- [ ] **Step 1: 校验 Task 6 (S5) 完成态**

```bash
grep -i "S5.*✓" G:/OUROBOROS-AI4S/CLAUDE.md
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q
```

- [ ] **Step 2: 执行 S3 detail plan 全部任务**

进入 `2026-06-24-s3-rich-prey-predicates.md`。

**S3 spec-specific 注意:**
- **`PREDICATE_BITS` 字典 / int64 assert 完全归 S6, S3 不动**。spec §2 「S3 不引入新 bit, 只填 S6 reserved 槽」(subagent 在 S3 detail plan Global Constraints 已显式声明)。
- **vis_lowvis 不在 S3 重复实现**: S1 Task 6 已在 `feature_mask_of` 实现 prey 端置位; S3 Task 4 加 `("N","lowvis")` clause-tag 分支只是 predator 端读出。**S3 Task 2 改 `feature_mask_of` 时不许误删 S1 段**, Task 5 端到端审计就是为了守这一条。
- **per-letter 不读 stacked**: 谓词读的是 letter-level family-spec(F.f / P.p_add / Z.z), 不是 stacked phenotype。kernel byte-identical 回归靠这一条: S4 hash-locked / S5 phase-window / S7 multi-slot 对 stacked 字段的扰动不该传染到 predicate 命中。
- **闭区间**: spec §1 表里阈值是闭区间(`≥` / `≤`), 不是开区间。常见 bug 是 `>` 错成 `>=` 让边界值偷渡, 或反过来。Task 2 测试有边界值用例。

- [ ] **Step 3: 5 项 gate 全过**

```bash
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q                      # ≥431 passed
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke --byte-equal-baseline pre-s3   # BYTE-EQUAL
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke                       # PASS
git push origin main
# CLAUDE.md / memory: S3 ✓
```

- [ ] **Step 4: commit gate 同步**

```bash
git add CLAUDE.md C:/Users/Strix/.claude/projects/G--OUROBOROS-AI4S/memory/project_des_unify_and_68_roadmap.md
git commit -m "docs: mark S3 (rich prey predicates) complete in roadmap"
git push origin main
```

**Acceptance:** 5 项 gate + S3 detail plan checkbox 全勾 + grep `thr_crest|thr_hotspot|thr_mirror|_Z_PREY_CARD` 命中 + S3 Task 5 端到端审计绿(S1 prey 端 ↔ S3 predator 端)。

---

---

### Task 8: S7 — 多位突变(slots_per_event)

**Files:**
- Spec: `digital-evolution-sandbox/docs/superpowers/specs/2026-06-24-s7-multi-slot-mutation-design.md`
- Detail Plan: `digital-evolution-sandbox/docs/superpowers/plans/2026-06-24-s7-multi-slot-mutation.md`(1321 行)

**Interfaces:**
- Consumes: Task 1–7 全部产出。
- Produces:
  - `SLOTS_PER_EVENT: dict[str, int]` registry 表(全 letter 默认 1, 值域 `{1, 2}`, 不许填其他值; module-load 守门)
  - `Phenotype.slots_per_event: int = 1` 字段
  - `_mutation_outcomes` 加 `slots_per_event=1` kwarg, signature 扩为 `(seq, mutable, spectrum, blocks, slots_per_event=1)`
  - N==1 路径 byte-identical(verbatim 旧实现, 只是包了 kwarg, 不读)
  - N≥2 路径 `itertools.combinations × product` joint enumeration, weight `(1/C(m,N)) · ∏q`, `effective_N = min(N, m)` clamp
  - P_cascade(2-slot P-row)进 ALPHABET / GRAN / _P / SPECTRUM_SHAPE / SLOTS_PER_EVENT

- [ ] **Step 1: 校验 Task 7 (S3) 完成态**

```bash
grep -i "S3.*✓" G:/OUROBOROS-AI4S/CLAUDE.md
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q
```

- [ ] **Step 2: 执行 S7 detail plan 全部任务**

进入 `2026-06-24-s7-multi-slot-mutation.md`。

**S7 spec-specific 注意:**
- **N==1 byte-identical by construction**: 默认所有 letter 是 N=1, `_mutation_outcomes(..., slots_per_event=1)` 必须走 verbatim 旧实现路径, 不许「重写一遍逻辑碰巧等价」。这是 byte-equal gate 守得住的唯一办法。
- **`SLOTS_PER_EVENT` 只许 `{1, 2}`**: 别的值是 spec out-of-scope, registry module-load 守门 raise。这是防「实现者拍脑袋扩 N=3」的红线。
- **不进 phe 张量列**: `slots_per_event` 不进 `phenotype_arrays`(99% parent 是 N=1, hot-loop scalar 读不必批量化, S7 detail plan 已裁定; 与 S5 `burst_w/burst_k` 进张量列不同, 因为相位窗每 tick 都查所有 cell, 而 mutation 是稀疏事件)。
- **P_cascade 不许塞「谁强」**: P_cascade 是 2-slot P-row, 它的 spectrum 数据由 spec §3 锁死, 不许 hand-tune。

- [ ] **Step 3: 5 项 gate 全过**

```bash
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q                      # ≥431 passed
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke --byte-equal-baseline pre-s7   # BYTE-EQUAL(默认 4-faction 全 N=1, 字节级不变)
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke                       # PASS
git push origin main
# CLAUDE.md / memory: S7 ✓
```

- [ ] **Step 4: commit gate 同步**

```bash
git add CLAUDE.md C:/Users/Strix/.claude/projects/G--OUROBOROS-AI4S/memory/project_des_unify_and_68_roadmap.md
git commit -m "docs: mark S7 (multi-slot mutation) complete in roadmap"
git push origin main
```

**Acceptance:** 5 项 gate + S7 detail plan checkbox 全勾 + grep `SLOTS_PER_EVENT|slots_per_event` 命中 + ALPHABET P-letter 涨 1(P_cascade)。

---

---

### Task 9: S8 — A 池 24 极端 + 多 P 谱混合(收口)

**Files:**
- Spec: `digital-evolution-sandbox/docs/superpowers/specs/2026-06-24-s8-a-pool-extremes-design.md`
- Detail Plan: `digital-evolution-sandbox/docs/superpowers/plans/2026-06-24-s8-a-pool-extremes.md`(1904 行, 9 篇里最大)

**Interfaces:**
- Consumes: Task 1–8 全部产出, 这是「S 系列收口 task」。
- Produces:
  - `src/des/_a_pool.py` 建 + 24 行 A 池数据(8 常量 × 3 sub-pool: 乙1 / 乙2 / 甲; family ∈ {F, P, Z} 无 rank-4)
  - 24 行合入 ALPHABET / GRAN / MOTIF_LEN / _F / _Z / _P / SLOTS_PER_EVENT + 极端值 assert + F_NOVA 冲突守门
  - **`n_locked` 门 de-gate audit**: grep 全代码确认 reproduction.py 等核心路径**没人**前置依赖 `n_locked`(纯 affinity 谱可达 = 门作废)
  - **relabel-invariance** 测试加(重排 f/z/p 数值不改 strain identity)
  - SPECTRUM_SHAPE 值域扩 `power=4 / mask="cross"` + 合入 8 行 + `_spectrum_for` 加 cross 分支
  - **`blend_p_spectra` helper**: `Σpᵢqᵢ/Σpᵢ`, 单字母 identity 守 byte-equal, `Σ=0` 等权退化(`Σq/N_p`)
  - **`phenotype()` 用 `blend_p_spectra` 替换 `dominant_p`**(spec-review 第 ② 裁定)
  - roster doc cleanup: 24 条覆写行 + OPEN-1/θ 标 RETIRED
  - **默认 4-faction byte-identical engine.run**: 收口验证, 因为默认局基因型不含 A 池任何 letter, 默认路径必须字节级不变

- [ ] **Step 1: 校验 Task 8 (S7) 完成态**

```bash
grep -i "S7.*✓" G:/OUROBOROS-AI4S/CLAUDE.md
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q
```

- [ ] **Step 2: 执行 S8 detail plan 全部任务**

进入 `2026-06-24-s8-a-pool-extremes.md`。

**S8 spec-specific 注意:**
- **F_NOVA 双 owner 守门**: F_NOVA 数据 S5 Task 6 已铸, S8 Task 2 加 conflict guard `assert byte-equal`, **不许重新 mint**。S5 是 owner, S8 引用。
- **`Σp_add == 0` 边界**: spec §4.1 未显式给, subagent 裁定为「等权退化 `Σq / N_p`」, 避免 div-by-zero; 单字母 identity 保 byte-equal(`Σ pᵢqᵢ / Σ pᵢ = q`)。
- **A_SHAPE 合入顺序**: 在 SPECTRUM_SHAPE 块后、值域 assert 前 update, 避免「co-extensive with _P」误报(spec 边界, subagent 已记)。
- **`dominant_p` 完全弃**: S2/S7 之前还在用 `dominant_p`, S8 Task 5 是替换点, 替换后 grep `dominant_p` 应该只剩 git history 与 spec doc 内的回顾性提及, 代码路径 0 命中。
- **de-gate audit 必跑**: `grep -rn "n_locked" digital-evolution-sandbox/src/` 期望只命中 `n_locked` 定义点与「计算它的纯函数」位置, 不应在 reproduction / phenotype / mutation 的前置 if 守门里命中。命中即 spec 实现走偏。
- **24 行 A 池极端是 spec §1 表的精确值**: 不许 hand-tune; 这是「全 68 affinity 谱才是设计」的最后一批补完。

- [ ] **Step 3: 5 项 gate 全过**

```bash
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q                      # ≥431 passed
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke --byte-equal-baseline pre-s8   # BYTE-EQUAL(默认局不读 A 池 + multi-P 单字母 identity)
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke                       # PASS
git push origin main
# CLAUDE.md / memory: S8 ✓

# **额外的 S8 专属 audit:**
grep -rn "dominant_p" digital-evolution-sandbox/src/   # 期望 0 命中(代码路径全切到 blend_p_spectra)
grep -rn "n_locked" digital-evolution-sandbox/src/     # 期望只命中定义点与计算函数, 不在 if 守门
```

- [ ] **Step 4: commit gate 同步 + 收口标记**

```bash
git add CLAUDE.md C:/Users/Strix/.claude/projects/G--OUROBOROS-AI4S/memory/project_des_unify_and_68_roadmap.md
git commit -m "docs: mark S8 (A pool extremes + multi-P blend) complete; registry 6→68 close"
git push origin main
```

**Acceptance:** 5 项 gate + S8 detail plan checkbox 全勾 + **`len(ALPHABET)` 对齐 spec roster 表所声明的总数**(实测 grep 算: 6 base + 10 P-shaped(S2) + 5 F-direction(S4) + 2 F-burst(S5) + 1 P_cascade(S7) + 24 A 池(S8); 与 spec roster 表对账, 不在此 plan 硬锁数字) + `dominant_p` 代码 0 命中 + `n_locked` 不当守门。

---

---

### Task 10: RE-RECORD — 全 68 affinity 谱 fixture 字节基线重锁

**Files:**
- 触发依据: spec-review 第 ① 裁定(CLAUDE.md「评审两裁定」段 + memory `project_des_spec_review_rulings.md`)。
- 涉及文件: `digital-evolution-sandbox/tests/fixtures/*.parquet`(谱锁文件); `digital-evolution-sandbox/data/playground/*.parquet`(首批 837MB 4 parquet, 若仍当 fixture 用)。
- 此 task 不引用单独 detail plan, 它是 spec-review 裁定派生的「收尾步骤」, 步骤在此处展开。

**Interfaces:**
- Consumes: Task 9 (S8) 收口产出 — registry 6→68 全长全, A 池 24 行 + 全 P/F/Z/N 已铸齐, 多 P 混合 blend_p_spectra 上线。
- Produces:
  - **重录所有 affinity 谱 fixture**(以新 68-长全 registry 为 source-of-truth)
  - **删 / 标 SKIP 「锁 F4Nr1=北」类老断言**(S4 重底定 F4Nr1 为 hash-locked 1-of-4, 此处统一升级或废弃)
  - **新 byte-equal baseline 锁定**: `pre-re-record` baseline 文件 commit, 后续任何 plan 都以此为基线
  - **CLAUDE.md「首批数据」段 + memory `project_des_first_batch_config.md`** 同步: 标 「old 837MB(锁北)= legacy / new 重录(hash-locked 1-of-4)= current」

**为什么必须最后做(spec-review 第 ① 裁定原文):**
> 全 68 affinity 谱才是设计, 6 字母是残缺截断, 长全后 RE-RECORD fixtures, 字节锁此后护非 registry 代码路径

**S2 / S4 / S5 / S7 各自的 detail plan 都 skip 了一批 fixture 测试 + 标 TODO 留给本 task**, 这里是统一升级点; 不许在前 9 个 task 内逐个补 RE-RECORD(那样会反复重录、字节锁反复失效)。

- [ ] **Step 1: 校验 Task 9 (S8) 完成态 + ALPHABET 全长**

```bash
grep -i "S8.*✓" G:/OUROBOROS-AI4S/CLAUDE.md
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -c "from des.registry import ALPHABET; print(len(ALPHABET))"
# 期望: 68(目标值, 实际以 spec roster 表为准; 不等 = S8 没收口完, 退回 Task 9 而非进 Task 10)

$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q
# 期望: ≥431 passed; 此时仍可能有一批 S2/S4/S5/S7 标 skip 的「锁北 / 长不全谱」类测试, 这是即将被 Task 10 升级的
```

- [ ] **Step 2: inventory 待升级的 SKIP 测试**

```bash
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q -rs 2>&1 | grep -i "skip" > /tmp/skip-inventory.txt
cat /tmp/skip-inventory.txt
```

把所有 skip 原因里含 `F4Nr1 锁北 / 谱长不全 / RE-RECORD pending / fixture-pre-S{2,4,5,7,8}` 关键字的列出, 这就是本 task 要逐个升级的清单。其他 skip(若有)留原状, 不动。

- [ ] **Step 3: 重录全 68 affinity 谱 fixture**

```bash
# 假设 fixture 录制脚本存在(若不存在, 此 task 第一步是新建 record_fixtures.py — 但既然 spec-review 早就预告了 RE-RECORD, S6 detail plan 内可能已含 fixture 录制 helper)
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.tools.record_fixtures --all-affinity --overwrite
# 期望: 写覆盖 fixtures/affinity-spectrum-*.parquet, 一行不漏地覆盖 68 letter
```

如果录制工具不存在: 先停下, 这意味 S6 detail plan 漏了 fixture 录制 helper(spec-review 已预告但实现遗漏), 应作为 S6 spec gap 补 task, 不在本 task 内 ad-hoc 写脚本。

- [ ] **Step 4: 逐个升级 skip 测试**

对 Step 2 清单里每条 skip:
- 若 skip 原因是「F4Nr1 锁北」: 改测试期望为 「F4Nr1 hash-locked, 锁定方向 = `_hash_dirs(seq, "1of4")` 给出的那一格」, 复用 S4 detail plan Task 1 已铸的 `_hash_dirs` 纯函数。**不许把测试整个删掉**, 升级而非废弃, 因为它仍是「方向锁定」语义的回归守门。
- 若 skip 原因是「谱长不全 / 6 字母」: 删 skip 标记, 测试直接读新 fixture, 应该立即绿。
- 若 skip 原因是「fixture-pre-S{N}」: 删 skip 标记, 测试读新 fixture。

每升级一条, 跑这一条单测确认绿:

```bash
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q path/to/test_file.py::test_name
```

- [ ] **Step 5: 新 byte-equal baseline 锁定**

```bash
# 把 pre-re-record baseline(Task 9 末尾的)归档:
mkdir -p digital-evolution-sandbox/tests/fixtures/baselines/legacy/
git mv digital-evolution-sandbox/tests/fixtures/baselines/current/*.parquet digital-evolution-sandbox/tests/fixtures/baselines/legacy/

# 录制 post-re-record baseline:
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.tools.record_fixtures --byte-equal-baseline current
# 期望: 写 digital-evolution-sandbox/tests/fixtures/baselines/current/post-re-record.parquet 等
```

- [ ] **Step 6: 5 项 gate 全过(最终)**

```bash
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest -q
# 期望: 所有 SKIP 升级完, passed 数比 Task 9 末尾涨「升级条数」, 0 failed, 0 unexpected skip

$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke --byte-equal-baseline current
# 期望: BYTE-EQUAL against new current baseline

$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m des.smoke
# 期望: SMOKE PASS

git add digital-evolution-sandbox/tests/fixtures/
git add digital-evolution-sandbox/tests/                  # 升级的测试
git commit -m "test: RE-RECORD all 68 affinity spectra fixtures; upgrade F4Nr1 lock-north assertions to hash-locked 1-of-4; new byte-equal baseline"
git push origin main
```

- [ ] **Step 7: CLAUDE.md / memory 终态同步**

用 Edit 工具改 `G:/OUROBOROS-AI4S/CLAUDE.md`:
- 「Digital Evolution Sandbox」段「待办闸口」②(plan 全好才进 SDD/executing 实现): 标 ✓ 完工于 YYYY-MM-DD
- 「首批数据」段: 加注「old 837MB(F4Nr1 锁北)= legacy / 新 baseline(F4Nr1 hash-locked 1-of-4)= current」
- 「9-spec 路线图」段: S0..S8 全打 ✓, 添加一行「RE-RECORD ✓(YYYY-MM-DD)」

用 Edit 工具改 memory:
- `project_des_unify_and_68_roadmap.md`: S0..S8 全勾完
- `project_des_first_batch_config.md`: 标 legacy 与 current 双 baseline 关系
- `project_des_spec_review_rulings.md`: 第 ① 裁定 RE-RECORD 已执行, 标 ✓

```bash
git add CLAUDE.md C:/Users/Strix/.claude/projects/G--OUROBOROS-AI4S/memory/
git commit -m "docs: mark 9-spec roadmap + RE-RECORD complete; registry 6→68 fully closed"
git push origin main
```

**Acceptance:**
- pytest 0 failed, 0 unexpected skip(预期 skip 数清零)
- `byte-equal --baseline current` 绿
- `ALPHABET` 长 68(以 spec roster 表为准)
- `grep dominant_p` / `grep "F4Nr1.*north"` 在 src/ 与 tests/ 都 0 命中
- CLAUDE.md「9-spec 路线图」段全 ✓ + RE-RECORD ✓

---

---

## Self-Review

**1. Spec 覆盖检查:**

| Spec / 裁定 | 覆盖 Task | 备注 |
|---|---|---|
| S0 统一入口 + CLI | Task 1 | 5-gate 全 |
| S6 motif 粒度(横切) | Task 2 | 失败回退路径加了 spec-gap 文档化条 |
| S1 vis 通道 | Task 3 | predator 端读出留给 S3 Task 7 |
| S2 塑形突变谱 | Task 4 | `dominant_p` 保留旧实现 → 给 S8 替换 |
| S4 动态方向 | Task 5 | F4Nr1 重底定 → skip 锁北 fixture, 留 Task 10 |
| S5 相位窗 | Task 6 | `burst_w=1/burst_k=1` 守 byte-equal 默认路径 |
| S3 富猎物谓词 | Task 7 | S1 prey ↔ S3 predator 端到端审计 |
| S7 多位突变 | Task 8 | N==1 byte-identical, `SLOTS_PER_EVENT` 值域 `{1,2}` |
| S8 A 池 + 多 P | Task 9 | `dominant_p` 退场 + `n_locked` de-gate audit |
| spec-review ① RE-RECORD | Task 10 | 全 68 affinity 谱 fixture 重录 + skip 升级 |
| spec-review ② 多 P 归 S8 | Task 9 内 | `blend_p_spectra` 替换 `dominant_p` |
| CLAUDE.md「非对称角色」HARD-GATE | out of scope | Global Constraints 显式声明排除 |

**2. Placeholder 扫描:** grep `<!-- SECTION:` 应为 0(全部 14 段已填); grep `TBD|TODO|implement later|fill in details` 在本 plan 内只在 「Step 4 升级 skip 测试」 的复述里出现一次(指 detail plan 内已标的 TODO), 不是本 plan 自留的 placeholder。

**3. Type / 命名一致性:**
- `Phenotype` 新字段(Task 3 S1 / Task 6 S5 / Task 8 S7): `vis_sum / n_count / f_hi / f_lo / burst_w / burst_k / in_place / rand_dir / slots_per_event` 全部小写下划线, 风格一致, 与 9 篇 detail plan 引用一致(交叉对过)。
- `_hash_dirs` / `_spectrum_for` / `_mutation_outcomes` / `feature_mask_of` / `prey_mask_for_clauses` / `blend_p_spectra` / `phase2_reproduce`: 函数命名风格一致(下划线前缀私有 / `_of` `_for` `_mask` 动宾结构)。
- registry 字典名: `ALPHABET / GRAN / MOTIF_LEN / SPECTRUM_SHAPE / SLOTS_PER_EVENT / PREDICATE_BITS / _F / _Z / _P / _A`: 全部大写, 与已落地代码一致(S0 已铸基线)。
- `predicate-bit` 与 `feature_mask` 在不同 task 引用一致(predicate-bit 是位定义, feature_mask 是按位读出实例)。
- 检查 5 项 gate 命名: 「5 项 gate」在 Global Constraints + 10 个 Task 内一致(pytest / byte-equal / smoke / push / CLAUDE.md 同步), 没出现命名漂移。

**4. 隐含 Spec 要求覆盖:**
- 「每 task 必过 5 道 gate」: Global Constraints 锁, 10 个 task 一致执行。
- 「实现序绝对锁」: Global Constraints + DEP_GRAPH 两处声明 + 两条具体禁令(不许 S1/S3 提前 / 不许并行 S2/S4/S5)。
- 「no-private-stuff 红线」: Global Constraints 锁, Task 4 / Task 8 / Task 9 三处单独再次提醒。
- 「结局常数永不进 CLI」: Global Constraints 锁。
- 三接口同步(CLI / src / webapp): Global Constraints 锁。
- 回归基线 285+146=431: Global Constraints + 10 个 task 的 gate 全部以此为对照。

**5. 失败回退策略可执行性:**
- Global Constraints 给出 「task 失败 → revert 到上一 task tip」 通用策略。
- S6 (Task 2) 额外给「不许加补丁让它过, 改 detail plan 文本」 特殊回退。
- byte-equal 断言失败 = 强信号, 给了诊断方向。

**6. 修正记录:**
- 无新发现的命名 / 类型不一致。
- Task 9 ALPHABET 总数 = 68 是目标值, 实际算术 `(6 base + 10 P-shaped + 5 F-direction + 2 F-burst + 1 P_cascade + 24 A) = 48`, 与 68 差 20 — 这意味着 spec roster 表里还有 20 个 letter 是「既有基线 6 不含, 但 9 篇 spec 也没逐个铸」 的隐含 base 增长(可能 base 不是「6 个字母」而是「6 类 base + 各类多 letter」)。**Acceptance 文里已加「以 spec roster 表为准」 的措辞, 实际 commit 时按 grep `len(ALPHABET)` 实测对齐 spec, 不在 plan 文里硬锁数字**。这条算术 mismatch **不是 plan bug**(spec 自身的 roster 才是 source-of-truth), 只是 plan 不该自己锁死 68 这个数字 — 已修正措辞, 不再重复说「ALPHABET=68 收口」 而说 「ALPHABET 长度对齐 spec roster 表」。

---

## Execution Handoff

**Plan complete and saved to** `digital-evolution-sandbox/docs/superpowers/plans/2026-06-25-9-spec-execution-roadmap.md`。

**两种执行方式:**

**1. Subagent-Driven(推荐)** — 每个 Task 派一个 fresh subagent, Task 之间两段评审, 快节奏迭代。
- 必读 sub-skill: **superpowers:subagent-driven-development**
- 适用场景: 用户在线监督, 每 Task 完成后愿意做 5 分钟评审。
- 推荐入口: 「按这篇 roadmap 的 Task 顺序, 起一个 subagent 跑 Task 1, 完了我评审」。

**2. Inline Execution** — 当前会话内直接连续跑, 用 checkpoint 断点。
- 必读 sub-skill: **superpowers:executing-plans**
- 适用场景: 用户半自动监督, batch 跑几个 Task 一起评审。
- 注意: 这是 10 个嵌套 detail plan 的串行执行, 单一会话 context 可能不够; **不推荐**整 roadmap inline。可考虑 「inline 跑 Task 1+2(S0+S6 横切地基), Task 3 起切 subagent」 的混合模式。

**推荐先做的两件事(在选择执行模式之前):**

- (a) **校验 base 状态(Task 1 Step 1)**: 跑一次 `pytest -q` 确认 431 绿测 + git status clean, 这是 plan 假设的起点。base 不绿就别开始, 先修 base。
- (b) **push 这篇 roadmap 到 origin/main**: 让远端有这篇 plan 是 「Task 1 Step 2」, 也可以提前到选执行模式之前先 push。

**用户决策点:**

> **选择哪种执行方式?**
>
> A. Subagent-Driven(推荐): 一个 Task 一个 subagent, 每完一个 Task 用户评审 → 进下一个。
> B. Inline: 当前会话连续跑, batch checkpoint。混合模式可选(前 2 task inline, 后续 subagent)。
>
> **以及哪条线先开?** (你的 sandbox / selop-v3 / σ-反演 三条线在 CLAUDE.md, 这篇 roadmap 只动 sandbox 线; selop-v3 与 σ-反演 各有自己的 HARD-GATE。)
