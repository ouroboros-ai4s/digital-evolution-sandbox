# PHASE3 K-wall 仲裁性能优化设计

**日期:** 2026-06-21
**状态:** 设计 — 待用户 review
**作者:** hub (Claude) + 3 路 subagent(性能工程 / 设计保真审计 / ponytail)

## 1. 问题陈述

首批 T=450×4 种子跑批在当前引擎实现下**计算上不可行**(约比可接受速度慢 100 倍)。本设计只解决性能瓶颈,**不改变任何仲裁语义**。

### 1.1 实测诊断(measured + 代码确认,非臆测)

phase-probe(128×128 / K=64 / FILL=20,跑到 tick 240,世界逐渐填满)实测每 phase 耗时:

| phase | ms/tick | 占比 |
|---|---|---|
| antagonism | 24.5 | 1.6% |
| reproduction | 123.0 | 8% |
| **arbitration(PHASE3 K墙仲裁)** | **1323.9** | **90%** |

per-tick step 时间随世界占用增长:tick60(世界近空)~32ms → tick340(世界填满,33 万非空行)~2234ms。`step()` 计算累计 759.8s vs `dump()` 写盘累计 4.3s —— **瓶颈是计算不是写盘**(此前两次假设"写盘"/"对抗 kernel"均被实测证伪,本设计只基于实测)。

### 1.2 根因(代码定位)

`src/des/kernels/arbitration.py` 函数 `phase3_arbitrate_vec`,争用分支(约 63-90 行):

- L67 `labels = torch.repeat_interleave(c_idx, c_counts)` —— 把每个争用记录摊开成**逐个体**,张量长度 `n_ind = Σ merged[contested]`。
- L69 `torch.rand(labels.shape[0], ...)` —— 对每个个体掷随机数。
- L72-73 `composite = ind_cell.float64 + keys.float64; order = torch.argsort(composite)` —— 对长度 `n_ind` 的 float64 向量做 **O(n_ind·log n_ind)** 排序。这是主导项。
- L78-89 segment-rank / cumsum / searchsorted / bincount —— 均为 O(n_ind) 遍历。

**为何随占用恶化:** 世界填满后(a)几乎每个格子都 `total > avail` 进入争用分支,(b)繁衍持续往每格灌大 count,`n_ind`(争用个体总数)冲到千万级。排序成本随之爆炸。

**核心浪费(决定性观察):** 世界填满时 `avail = K − 已占用 ≈ 0`,代码仍生成千万级随机 key,**最后只留下约 0 个幸存者**。算力被淘汰个体(losers,无信息)主导。

**结论:** 任何仍触及"逐个体"的修法都注定失败(`total` 无上限)。修法必须把成本从 **个体数(无界)** 移到 **记录数(distinct `(cell, sid, faction)` 元组,有界)**。

## 2. 必须保全的契约(锁定于 design.md + 测试)

任何重写必须**逐字**保全以下语义(否则污染 ground truth):

1. **K墙抽样:** 目标格 `total_arriving > available = K − 占用` 时,精确抽 `available` 个幸存者;每个到达个体存活概率 = `available/total`(无放回的多元超几何 / sampling-without-replacement)。— design.md L201
2. **等比例、无隐藏权重:** 幸存分配只依赖送达量,**绝不**依赖 `sid` 或 `faction`。design.md L170-171:"加权重 = 手写 φ = 数据作废. 必须保持等比例." 这是 ground truth 藏身处。
3. **faction-blind(公平红线 7-J):** 存活概率不得依赖阵营 id。
4. **蒸发不挤活体:** 只填空位(`avail`),**绝不**驱逐在格活体。design.md L173。
5. **硬上限 ≤ K** / **同 sid 不同 faction 分槽** / **收敛同 `(sid,faction)` 合并**。
6. **枚举顺序无关:** `test_kwall_order_independent` 强制(均值 pairwise max/min ratio < 1.10;旧 sequential-binom 给 ~1.24 被否)。
7. **确定性:** 给定种子结果确定且统计正确(跨种子胜率 CI 需要)。**注意:property 测试是统计性的(ratio 区间 / 数千种子平均),非 bit-exact** —— 故抽样算法**可以替换**,只要保全分布。

## 3. 方案对比(3 路 subagent 已审)

### 方案 A — 逐空位抽签(record 粒度,精确)【推荐,三方一致】

**核心:** 不摊开个体。对争用格构建 `[n_contested_cells, R_max]` 的"剩余 per-record 计数"矩阵(`R_max` = 任一争用格内 distinct 记录数的最大值,有界 padding 矩阵)。逐个抽幸存者:`for j in 0..max(avail)-1`,跨格向量化,每格按 **Gumbel-max 加权(权重 = 剩余计数)** `argmax(log(remaining_count) + gumbel_noise)` 挑一条记录,该记录幸存计数 +1、剩余 −1,已达自身 `avail` 的格子被 mask 掉。这是无放回抽样的**精确**瓮过程(urn process)。

- **分布:** 精确多元超几何,非近似。权重只是计数 → 结构上 faction-blind & sid-blind。
- **复杂度:** `O(max_avail · n_cells · R_max)`,与个体数无关。`max_avail ≤ K=64`,且**满世界热区里 `max_avail ≈ 0`**,循环几乎不转 —— 恰好在原代码最慢处它最快。
- **依赖:** 仅 `torch.rand`(已有),无新依赖,确定性干净。
- **设计审计:** CLEAN。**ponytail:** 推荐,且指出 A 是最朴素正确的方案(非过度工程)。

### 方案 B — order-statistic capped expansion(精确)【淘汰】

保留现有 keys+top-avail 机制,但每记录只取最小 `min(count, avail)` 个顺序统计量(经 exponential-spacings + 一个 Gamma 尾抽生成)。数学精确。**淘汰理由:** 需要 Gamma 采样器,torch 的 Gamma 对 `Generator` 支持弱 → 威胁跨种子复现性(科学需要);且仍要 argsort;moving parts 更多。ponytail 判定这才是"为优雅过度工程"。

### 方案 C — multinomial/hybrid(近似)【淘汰】

按 `multinomial(avail, count/total)` 直接抽 per-record 幸存计数(有放回,近似无放回)。**淘汰理由:** 引入 `O(avail/total)` 偏差;"满世界时可忽略"的辩护**恰好在四阵营扩张期失效**(`total ≈ avail`,正是本批数据的真实起始状态);需 clamp 防止伪造个体;调参 `c` 须冻结;残差须披露。设计审计:NEEDS-CAVEAT(不算夹私货,但偏离精确 law)。

### 决议

**采用方案 A。** 三方独立收敛:性能 subagent 推荐 A(热区零循环);设计审计 A CLEAN;ponytail 选 A 并反驳 B/C。A 是唯一同时满足"精确 + 无新依赖 + 按构造杀掉热点"的方案。

## 4. 实现范围

**只改一处:** `src/des/kernels/arbitration.py` 的 `phase3_arbitrate_vec`,Section 2 争用分支(约 48-90 行)。

- Section 1(coalesce 到 record 粒度,L21-39)**不动** —— 已是 record 粒度。
- Section 3(writeback,L92-141)**不动** —— 已是 record 粒度。
- 引擎其它部分、reproduction、antagonism、recorder、world、StrainTable **全部不动**(引擎冻结纪律)。

## 5. 验证策略

### 5.1 强制 de-risk 闸门(动手前先做)

新增一个独立的"矩"检验(synthetic 单争用格,~30 行):
- 输入如 counts `[100, 200, 700]`,`avail=10`,跑 ~10k 种子。
- 对比 A 的 per-record 幸存**经验均值/方差** vs 闭式**多元超几何**(`mean = avail · cᵢ/total`,方差用 MVHG 闭式)。
- 通过条件:均值匹配 + 顺序无关 ratio < 1.10。

**理由:** 统计 bug 不会抛异常,只会悄悄污染数据(本项目已被 B1/B2/B3 此类失败烧过)。此检验**强制非可选**,且作为永久回归守卫留下。

### 5.2 既有测试套(必须全绿)

`tests/test_arbitration.py` + `tests/test_arbitration_properties.py`:hard cap ≤ K、faction-blind ratio 区间、resident-not-evicted、same-sid-diff-faction 分槽、convergent-merge、`test_kwall_order_independent`(顺序无关 <1.10)。

### 5.3 性能验收

满世界快照上测单 tick 仲裁耗时,应从 ~1324ms 降到与 reproduction(123ms)同量级或更低,使 T=450×4 跑批回到可行区间。

### 5.4 环境(铁律)

- 解释器 **`D:/anaconda3/envs/basic/python.exe`**(torch 2.10+cu128 / CUDA True / RTX 5080);裸 `python` 是另一个 cpu-only torch,**永远用显式 basic 路径**。
- imports 经 `PYTHONPATH=src`,repo root 运行;**不 pip/conda install**。

## 6. YAGNI / 显式排除

- **不**重写 dense `[H,W,K,K]` 对抗张量(C1)—— 实测对抗仅 24.5ms,非当前瓶颈;512² 时再单独处理。
- **不**改任何仲裁语义、阈值、结局常数。
- **不**改 recorder / 写盘路径 —— 实测非瓶颈。
- **不**做泛化或可配置化 —— 只换争用分支的抽样算法。

## 7. 风险

**头号风险:** 向量化的"逐格 Gumbel-max + 减计数 + avail-mask"记账 fiddly,失败模式**静默**(产生看似合理但错误的分布,可能悄悄越过 <1.10 band 或扭曲幸存比例)。**最便宜的 de-risk = §5.1 的矩检验闸门**(动手前先证明分布正确,30 行换一个完整周期的代价)。

## 8. 杂项

- `scripts/diag_recorder.py`(诊断期产物,untracked)— 实现轮决定保留(归入诊断工具)还是删除。
