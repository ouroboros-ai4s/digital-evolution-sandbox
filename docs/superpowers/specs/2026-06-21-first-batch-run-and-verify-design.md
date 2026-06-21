# 首批运行 + 数据核验 — Design Spec

**日期:** 2026-06-21
**项目:** Digital Evolution Sandbox(数字进化沙盒,目标 512×512 红皇后混战,采集无偏时序数据集供 selop 反演)
**类型:** 运行 + 核验任务(**非引擎改动**)。引擎代码已冻结在 main(含 per-individual 突变修复 `3dd4923`);本 spec 只做「跑首批 + 核验数据是否符合设想动力学」。
**上游(权威,不复述机制):**
- `docs/superpowers/specs/2026-06-21-faction-and-vectorization-redesign.md` — §7 私货红线 11 条 / §8 验收标准 11 条
- `context/2026-06-20-numerical-round-protocol.md` — 动力学健康度判据(GATE0–4 / 8 病态标签 / 三零模型 / N2 / β<0)
- `context/2026-06-11-16-10-design.md` — 目标宏观动力学 / 世界初始化 / fixation

---

## 0. 核心纪律(顶置,上批教训)

上一批数据是 B1+B2 两 bug 产生的伪现象,却被误读成「红皇后正频率依赖/寡头化」还下了 selop 反演结论。本 spec 的结构性兜底:

1. **脚本只报数,绝不下 PASS/FAIL 判决。** 「数据对不对」的裁定权完全留给用户看报告定。脚本不预设阈值(避免阈值魔数把正常涨落判成 FAIL,或反之)。
2. **反推量一律标 proxy。** parquet schema 只有 `(tick, cell_x, cell_y, strain, faction, count)` 6 列;「kills / 对抗减员」**不是列**,只能从快照间 count 负 delta 反推,会混入 K 墙蒸发 + p_leave 迁出 + 仲裁。报告中标 proxy、注明「不可从快照辨识」,**严禁叫 kills**。
3. **不脑补。** 数据没明说的(机制归因、因果),报告写「不可从快照辨识」,不替用户臆测。
4. **裁定是人不是机器**(§4)。

---

## 1. 范围

**范围内:**
- 改 `scripts/run_batch.py`:取消提前停机(`stop_on=()`),保证 4 锅都跑满 T=450。
- 新增 `scripts/analyze_batch.py`:读 parquet → 报数(stdout 文本 + JSON 留档),不判决。
- 报告指标集(§2)。
- `tests/test_analyze_batch.py`:人造小 parquet 单测指标计算(已知答案)。

**明确不做(YAGNI,理由见各条):**
- **β<0 频率依赖回归** — 红皇后核心信号,但本批稳态窗 ~130 tick < 协议要求 200 tick(GATE0 NA-SHORT-RUN),且结局常数未标定 → 算出来的 β 协议自己都说不可信。defer 到 selop / GATE4 阶段。
- **ACF 负瓣 / PSD-vs-AR(1) / cyc_frac**(协议 §2.1 判据1)— GATE4 红皇后旋转指标,需 ≥200-tick 稳态窗,本批不够。
- **三零模型阈值合成**(协议 §3.0)— 需独立标定跑;报告只打原始 N2/flux/D_max 值,不合成阈值。
- **κ / α₀ / β_fold 上位效应** — 首批全关(=0,spec §9),无可测。
- **z_eff soft-cap / self_loss 标定曲线** — 标定阶段,本批 z 为占位值。
- **任何 GATE 闸门判决** — 脚本算 GATE0–4 的*输入*,但永不触发闸门(纪律1)。

---

## 2. 报告指标集

两层结构:per-seed 块 ×4 + 跨 seed 聚合块 ×1。协议依据:per-seed 可合法出现单局碾压(=自发对称破缺=想要的涌现);跨 seed 才是「分布/期望层」,1/4 CI 对称检验只在此层成立(design 行 53)。

### 2.1 Per-seed 块(每 parquet 一份,4 份)

| 指标 | parquet 算法 | 验 |
|---|---|---|
| 总活量 + 灭绝 tick | `groupby(tick).count.sum()`;首个 →0 tick | NULL-DEAD / GATE1(放第一行,最致命最便宜) |
| distinct 阵营数 + fixation tick | `df[count>0].groupby(tick).faction.nunique()`;→1 tick | §8.8 / FIXATION |
| 占用格时间线 | per-tick 非空格数;首个跨 faction 同格 tick;填满 tick | §8.2(~160 相遇 / ~320 填满) |
| occ 占比 + 谷深 | 占用格/(H·W);逐 tick 大跌幅分布 | GATE1b CHAOTIC-CRASH |
| 株多样性 | per-tick distinct 株数 S(t);N2=1/Σpₛ²;新序列首现率 | 多样性 / 协议 N2 |
| D_max 包络 | per-tick 最大单株频率;是否单调爬顶不回 | GATE2 寡头化 |
| L_ε turnover / flux | established 集(全局频率≥1%);flux=½Σ\|Δpₛ\|(Δ=5);leader 变更数 | GATE3 在动(FROZEN↔CHURN) |
| 阵营份额轨迹 | per-tick 四阵营 count + 占格份额;本 seed 赢家 | §8 份额 |
| 减员 proxy | 快照间 per-(格,faction) count 负 delta;是否集中相遇带 | §8.3,**标 proxy** |
| t=0 播种检查 | `tick==0`:distinct 序列数(期望 1)、distinct faction(期望 4) | §8.1 / §7-H |
| 表型-faction 独立性交叉表 | 高频株 × faction 计数;同株在 ≥2 faction 下生长率比对 | §7-G 核心不变量 |
| 同/多阵营格稳定性 | 单 faction 格 vs 多 faction 格 count 稳定性 | §8.4 **弱 proxy** |

### 2.2 跨 seed 聚合块(4 seed 合一)— 私货自检主战场

| 指标 | 算法 | 依据 |
|---|---|---|
| **胜率 1/4 CI(头条)** | 4 seed 各赢家分布 vs 1/4 二项 CI | §7 对称性定理;系统偏离=私货泄漏 |
| 前沿 D4 对称性 | per-tick 各 faction 占格数;四轨迹离散度(max−min) | §7-C/D 环面 D4 等变 |
| 竞争格 faction 存活率 | 满竞争格内各 faction 存活率(应 ≈ avail,无偏) | §7-J K墙 faction 盲 |
| GATE0 短窗旗标 | 稳态窗长 ~130 < 200,**显著打印一次** | 协议 GATE0,提醒 β 类不可算 |
| 时间线对账 | 实测 meet/fill tick vs spec§1.9(160/320) vs design 行44(32/60) | 让用户看现实匹配哪个模型 |

---

## 3. 文件产出与脚本结构

### 3.1 改 `scripts/run_batch.py`(微调)
- `e.run(ticks, recorder=rec)` → `e.run(ticks, recorder=rec, stop_on=())`,跨完整 T=450 不提前停。fixation tick 照样从数据读出(distinct_factions→1),但世界继续跑,fixation 后动力学也采得到。
- 其余不动(配置常数 / parquet 命名 `{timestamp}-seed{N}.parquet` / `--probe` / `--phase-probe`)。

### 3.2 新增 `scripts/analyze_batch.py`
```
analyze_batch.py [--runs-dir data/runs] [--glob "*-seed*.parquet"]
  ├─ load(path)                         # pyarrow 读 parquet → DataFrame
  ├─ per_seed_metrics(df) -> dict       # §2.1 全部指标
  ├─ cross_seed_metrics([df...]) -> dict # §2.2 聚合块
  ├─ render_text(report)                # stdout 结构化文本(人看,无 PASS/FAIL)
  └─ dump_json(report, path)            # data/runs/analysis-{timestamp}.json
```
- **纯函数式**:`*_metrics` 只吃 DataFrame 吐 dict,不打印不判决 → 可单测、可复跑。
- **报数不判决**:`render_text` 只排版数字 + 标注(proxy / GATE0 短窗 / 时间线三模型并列),无 PASS/FAIL。
- **依赖**:pyarrow(recorder 已用)+ pandas/numpy。**不引 matplotlib**(文本+JSON,无图)。
- **入参**:默认 `data/runs/` 抓最近一批 4 parquet;`--glob` 可指定。

### 3.3 产出物落点
`data/runs/` 下:4 个 parquet(跑批)+ 1 个 `analysis-{timestamp}.json`(核验留档)+ stdout 文本报告。

### 3.4 最小自检
`tests/test_analyze_batch.py`:人造小 parquet(已知答案,如 t=0 单序列4阵营、某 tick fixation 的玩具表),断言关键指标算对(N2、fixation tick、t=0 检查、1/4 CI 计算)。不跑真数据,CI 快。

---

## 4. 执行流程与介入点

```
1. [机器] 改 run_batch stop_on=()  →  --probe 60 估时/显存(确认不炸)
2. [机器] 跑满 4 seed × T=450     →  data/runs/ 落 4 parquet
3. [机器] analyze_batch 读 4 parquet → stdout 文本报告 + analysis-{ts}.json
4. [人]   看报告裁定:数据是否符合设想的红皇后扩张动力学
            ├─ 符合 → selop 反演分析输入就绪
            └─ 不符 → 异常指标定位到机制 → 回 brainstorming / 修 bug
```

**第 4 步是人不是机器**——脚本只把数摆出来(相遇 tick、四阵营份额、N2 曲线、1/4 CI 偏不偏、表型-faction 是否独立),「对不对」用户定。上批教训的结构性兜底。

**两个诚实提醒:**
- **GATE0 短窗**:稳态窗 ~130 tick < 200,β<0 红皇后核心信号本就算不了。「数据符合设想」的标准是**扩张 + 相遇 + 对称性**这些 transient 层面,不是「红皇后旋转已证实」(后者需更长跑或满格后专门批次)。
- **代码对 ≠ 数据对**:四条 founding bug 全修、85 测试绿,但只有这步真数据看完才算确认整套动力学符合设想。跑批烧算力,跑前 probe 估时确认。

---

## 5. 验收(本任务自身完成的标志)

1. `run_batch.py` 改 `stop_on=()`,4 seed 各产一个 450-tick parquet。
2. `analyze_batch.py` 跑通,产出 stdout 文本报告 + `analysis-{ts}.json`,§2 指标全在,无 PASS/FAIL 字样。
3. `test_analyze_batch.py` 绿(人造小 parquet 验指标计算)。
4. 报告交用户,用户完成 §4 第 4 步裁定。

**不在本任务范围:** selop 反演分析(数据符合后另起)、512² sparse 重写(defer)、数值轮标定。
