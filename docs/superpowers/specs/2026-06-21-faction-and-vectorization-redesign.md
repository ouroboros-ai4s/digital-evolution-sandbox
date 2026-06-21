# 阵营扩张 + 全向量化重构 — Design Spec

**日期:** 2026-06-21
**项目:** Digital Evolution Sandbox(数字进化沙盒,目标 512×512 红皇后混战,采集无偏时序数据集供 selop 反演)
**类型:** 合并重写 = ① 阵营扩张重构(承 `2026-06-21-faction-expansion-redesign.md`,已锁)+ ② 引擎热路径全向量化(P1/P2/P3,本轮新增)
**取代:** `2026-06-21-faction-expansion-redesign.md`(faction 部分原样并入本文 §2/§4,该文件标 superseded)

---

## 0. 背景:为什么合并

两件事撞在同样两个文件上,合起来一次做最省、零返工:

- **faction 重构**要重写 `reproduction.py`(分组键 `sid`→`(sid,faction)`、ArrivalBuffer 加第 5 列)和 `arbitration.py`(coalesce key 带 faction、writeback 合并判据带 faction)。
- **性能优化 P1/P2** 也要重写**同样这两个文件**:首批 12.8 s/tick(RTX 5080 上 128×128/K64,慢得离谱)的真凶不是 dense 张量(那是显存问题),而是 `arbitration.py` 与 `reproduction.py` 里的 **Python-per-element 循环 + GPU↔CPU 同步**。

分开做 = 两文件改两遍、第二遍还在向量化后的复杂代码上加 faction;合并 = 一份 diff、一套回归。**用户已拍:合并一次重写,B2 也现在全向量化。**

**首批废数据的三个 bug(承前,本文 §4/§5/§7 修复):** B1 BB0 铺满全场 / B2 对抗按序列免疫而非阵营 / B3 BB0 繁衍基元 F4Nr1 只向北。详见 design.md 新章节「世界初始化 · 阵营模型 · 目标宏观动力学」(权威源)。

**性能三热点(本轮新增,profiler 未跑、静态归因,见 §6 验证):**

| # | 热点 | 代码位置 | 病因 |
|---|---|---|---|
| **P1** | arbitration K墙抽样 | `src/des/kernels/arbitration.py` section 2(per-cell `for` 循环) | 每 contested cell 数次 `int(tensor[...])` = GPU→CPU 同步;满格期每 tick 数千 cell × 数次 = 数万次 GPU stall〔头号真凶〕 |
| **P2** | reproduction 繁衍 | `src/des/kernels/reproduction.py`(`present.tolist()` per-strain 循环 + `if mut.sum()>0`) | ① 每株每方向一次 `.sum()`/`.item()` 同步;② 逐株在全 [H,W] 网格跑 binom/roll,T=450 末期上千株时爆炸 |
| **P3** | engine 停机检查 | `src/des/engine.py` `run()` 每 tick `total_count()`+`_fixated()` | 每 tick 两次全张量 `.sum()`/`unique()`+`.item()`,无谓同步 |

> **代码位置按符号定位,不按行号。** 本文所有代码引用以**函数/符号名 + 字典键**为准(`init_bb0` / `_F[...]` / `_LOCKED` / `phase3_arbitrate_vec` / `ArrivalBuffer`);源树在 `src/des/`,行号随编辑漂移。

---

## 1. 用户已拍板的决议(本 spec 的硬约束)

**faction 部分(承前,已锁):**
1. **阵营血统永不变** — faction 从始祖继承的不可变标签,突变只改序列、不改 faction。
2. **对抗只有中和** — 双方各自按既有公式减员(`kills=min(b,a·z)`,自损 `kills/z_eff`)。**无"猎杀/同化"概念**。
3. **四阵营全用 BB0** — 仅 faction 不同。最纯对照组,私货免疫定理成立。
4. **四象限中心播种** — 四个始祖格子隔得尽量远,给扩张和相遇留空间。
5. **meta 地基②修订** — 「人人互为对手」→「群结构混战:4 阵营、群内联盟、群间对抗」。
6. **BB0 繁衍基元改 F4Nr4** — 四方向 f=0.50,使阵营能二维扩张相遇。
7. **fixation = 单阵营统一场** — 某 faction 干掉其他所有阵营即停机。
8. **世界边界 = 环面 torus** — 保持 `torch.roll` 现状,四象限对称等距无偏袒。
9. **T = 450** — F4Nr4 period=5 → 扩张 1格/5tick/轴 → 填满 ~tick 320,T 覆盖 前沿瞬态(0-160)+ 相遇带对抗(160-320)+ 满格后红皇后竞争(320-450)。

**性能部分(本轮,已拍):**
10. **P1/P2/P3 全做,合并进同一次重写** — 不分两轮、不留后续 pass。
11. **B2(reproduction per-strain 循环)也现在全向量化** — 不先 profile、直接做透。
12. **新抽样不与旧参考实现 bit-identical** — 旧 227 个 `vec==reference` 等价测试作废(旧数据本就是废的,无需复刻旧 RNG draw),换性质测试(§6)。

> **权威指针:** 宏观动力学、双正交身份不变量、对称群→可交换系综→1/4 CI 私货自检、faction 红线裁定,均以 design.md 章节「世界初始化 · 阵营模型 · 目标宏观动力学」为**权威源**。本 spec 是落到代码改动的实现规格,冲突以 design.md 为准。

---

## 2. faction 架构(方案 C — 平行 faction 张量 + 双正交身份,承前已锁)

**核心不变量(写进守门人审计):**
> `sid` 是唯一表型 key(phenotype gather 路径 `phe[sid]` 在任何 kernel 中**禁止**出现 faction 索引);`faction` 是 team key;占位 slot 身份 = `(sid, faction)` 对。

**为什么平行张量而非复合 id:** 把 (faction,sequence) 编进 sid 会让表型缓存膨胀 4×,且红-BB0/蓝-BB0 表型相等从"结构保证"降级为"约定"。平行张量下 `phe[sid]` 物理上看不到 faction,同序列不同阵营自动共享缓存行,结构性保住红线。

### 2.1 world.py
- `World.__init__` 加 `self.faction = torch.zeros((H,W,K), dtype=torch.int8, device=device)`(128²/K64=1MB,512²/K256=64MB,可忽略)。
- `snapshot()` 返回 `(strain_id, count, faction)` 三元组。
- 新增 `init_factions(H,W,K,device,table,fill_per_cell,n_fac=4)` **替代** `init_bb0` 全场铺地:只在四象限中心 `(H//4,W//4)`/`(H//4,3W//4)`/`(3H//4,W//4)`/`(3H//4,3W//4)` 各播 `fill_per_cell` 个 BB0,faction=0/1/2/3,其余格全空。
- **D4 对称校验:** 四播种点是同一点在 D4 群下的精确轨道(四象限中心天然满足)。

### 2.2 antagonism.py(P4 — 只换谓词,无性能改)
- gather:faction 是 slot 级状态,直接 `fac_slot = faction.long()`(`[H,W,K]`,**不经 sid 索引**)。
- `diff_strain = sid_i!=sid_j` **替换为** `diff_faction = fac_i!=fac_j`(`fac_i=fac_slot.unsqueeze(-1)`、`fac_j=fac_slot.unsqueeze(-2)`)。
- gate 必须是**对称等价谓词** `fight ⟺ faction_i≠faction_j`,对所有阵营对逐字相同。**禁** 4×4 非对称对阵矩阵。
- prey_mask/feature_mask(选靶按族)仍纯序列算;z 全程只读攻方序列;faction 只 gate"打不打"。
- dense `[H,W,K,K]` 内存**不因 faction 增长**(一个 bool 比较换另一个)。**C1 dense→sparse 重写 defer 到 512 跑前**(本批 128 网格 1GB 可跑)。

### 2.3 recorder.py(schema 加 faction)
- schema 加 `("faction", pa.int8())`;`dump` gather `world.faction[nz]`。
- 每行 6 列 `(tick, cell_x, cell_y, strain, faction, count)`,每帧自包含。

### 2.4 表型缓存 StrainTable —— 完全不动
- 同序列不同阵营共享同一缓存行 = 结构性保住"表型 = 序列固定函数"。faction 永不进 `phe[...]` 路径。

---

## 3. P1 — arbitration 全向量化(头号真凶)

`phase3_arbitrate_vec` 现状:section 3 writeback 已向量化,但 **section 1(coalesce)+ section 2(K墙抽样)仍是真凶**。section 1 其实已是张量算(`unique`+`scatter_add`),只需把 key 带 faction;**section 2 是那个 per-cell Python 循环 + 每元素 `int(tensor[...])` 同步**,是 stall 源。本节把 section 2 改成纯张量,并把整个 kernel 的 (sid)→(sid,faction) 键控做掉。

### 3.1 coalesce(section 1):key 带 faction
- arrivals 现含第 5 列 `a_fac`(见 §4 reproduction)。
- `key = cell*(MAXSID*NFAC) + a_sid.long()*NFAC + a_fac.long()`(NFAC=4;int64 安全:128² × MAXSID × 4 远未溢出)。
- `uniq, inv = unique(key)`;`scatter_add` 合并 count;反解 `u_y/u_x/u_sid/u_fac`。**仍全向量化,无循环。**

### 3.2 K墙多元超几何抽样(section 2):随机键法替代 per-cell 循环

**等价定理:** 多元超几何"从 cell 内 total 个个体无放回抽 avail 个" ⟺ "给每个个体一个 i.i.d. 均匀键,保留 cell 内键最小的 avail 个"。两者给出同一个无放回均匀抽样分布。

**向量化算法(无 Python 循环、无 `.item()`):**
1. 只取 contested cell:`per_cell_total`(按 u_cell 用 scatter_add 聚合 merged)、`per_cell_avail = (K - resident_occ)[cell]`;mask = `total > avail`(装得下的 cell 整段直接全存活,不抽)。
2. 对 contested 记录展开个体:`labels = repeat_interleave(arange(n_records), merged_records)`,每个个体带 `(cell, record_idx)`。**只展开 contested,稀疏期几乎空;最坏全满 ~2.1M int32 ≈ 25MB。**
3. `keys = torch.rand(n_individuals, generator=gen)`(**i.i.d.,不读 sid/faction**)。
4. 按 `(cell, key)` 排序;cell 内 rank = `arange - group_start`(分段 rank,**复用现有 section-3 writeback 同套 searchsorted/cumsum 技巧**)。
5. `survived_individual = rank < avail[cell]`;`bincount(record_idx[survived])` → 每 (sid,faction) 记录的存活数。
6. 装得下的 cell:`survived_record = merged_record`(全存活)。

**公平性证明(红线 §7-J,替代旧 spec「逐字不动」):** 键 i.i.d. 均匀、全程**不读 sid/faction** → 每个体存活概率恒 `avail/total`,与株/阵营无关 → faction-盲 + strain-盲**由构造保证**,非靠"原样不动"。RNG 单 generator 顺序消费 → 跨运行可复现(不与旧 draw bit-identical,§1.12 已拍废旧等价测试)。

### 3.3 writeback(section 3):合并判据带 faction
- 现 vec writeback 按 `(cell,sid)` 判 resident-hit;改为 `(slots_sid==sid) & (slots_fac==fac) & (cnt>0)`(否则红/蓝 BB0 错并进一个 slot)。
- new 记录找空槽逻辑不变;写回同步写 `new_faction`。
- **无 per-faction 配额**:K 是全局个体上限,蒸发 faction-盲。

### 3.4 删除参考实现
- `phase3_arbitrate`(非 vec 参考版)整个删掉:它存在的唯一理由是 227 个 bit-identical 等价测试,这些测试 §1.12 已废。留着 = 两份 per-cell 循环代码债。

---

## 4. P2 — reproduction 全向量化 + faction

现状 `phase2_reproduce`:`present.tolist()` 后**逐株** Python 循环,每株每方向跑 binom/roll,且 `if mut.sum()>0` 每方向一次 GPU→CPU 同步。两个病:① 同步;② per-strain 循环(T=450 末期上千株爆炸)。本节全做掉,并加 faction。

### 4.1 ArrivalBuffer 加第 5 列 faction
- `add(ty,tx,sid,cnt,fac)`;内部 `_fac` list;`tensors()` 返回 `(ty,tx,sid,cnt,fac)` 五元组。
- 空 buffer 返回 5 个空张量。

### 4.2 per-strain 循环消除(B2 全向量化)
**核心观察:** 方向集合是表型属性 `phe_obj.directions`,F4Nr4 四阵营同 BB0 → 早期所有存活株共享同一方向集。但突变后不同株可有不同 directions,故按 **directions 分组**而非按 strain:
- 对每个**唯一方向 (dy,dx)**(全表至多 8 种:4N+4D):
  - mask = "该方向在自己 directions 里"的所有 slot(`phe["has_dir_{dy}{dx}"][sid_long]`,表型预计算的方向 bitmask,gather 即得,无循环)。
  - `a = (snap_count * mask). ` 按 (sid,faction) 聚合到 [H,W] —— 用 slot 张量直接算,不 tolist。
  - `scattered = binom(a, f_per_slot)`(f 已 gather 成 [H,W,K] 张量),`roll` 到目标格。
  - 落地按 slot 的 (sid,faction) 进 buffer,faction 跟 slot 走(子代继承父 faction 天然成立)。
- **方向至多 8 种 = 至多 8 次张量算,与株数无关。** 替代"逐株循环"。

### 4.3 突变铸造:批量、去同步
- 突变 split 仍在落地处:`mut = binom(scattered, p_x)`,`non = scattered - mut`。
- **去掉 `if mut.sum()>0`**:无条件算 mut 张量;铸造按"本 tick 实际出现的突变母株集合"**批量**做——收集所有 `mut>0` 的 (sid) 唯一集,一次性对每个母株算 `_mutate_sequence` + `get_or_mint`,建 `sid→child_id` 映射张量,再 gather 给 mut 落地记录。
- **`get_or_mint` 是唯一必须留 CPU 的点**(序列是 Python tuple、StrainTable 是 dict):但从"每株每方向一次"降到"每 tick 一次批量",同步次数 O(方向×株) → O(1)。
- `_mutate_sequence` 不变(只读序列,纯)。

### 4.4 migration out 不变
- `leave = min(binom(snap_count, p_leave), snap_count)`;`live = (world.count - leave).clamp(min=0)`。已是张量算,无改。p_leave 迁出消失语义不变(design 已锁:迁出是繁衍代价,非搬迁)。

---

## 5. P3 — engine 停机检查降频

- 现 `run()` 每 tick 调 `total_count()`(全张量 sum+item)+ `_fixated()`(全张量 unique+item)。
- 改:**每 `check_every`(默认 10)tick 才查一次**停机;跑过头几 tick 无害(数据每 tick 仍 dump,不丢)。
- `_fixated` 用 int8 `faction` 张量:`distinct_factions = unique(faction[count>0])`,只 4 个值,极便宜;`==1` 即单阵营统一场。
- extinction 检查同理并进同一次降频查。

---

## 6. 验证 & 测试套重写

**旧 227 个 `vec==reference` bit-identical 等价测试作废**(§1.12):新抽样不复刻旧 RNG draw,旧数据本就是废的。删 `phase3_arbitrate` 参考版后这些测试无对照物,换**性质测试**:

| 测试 | 断言 | 守的不变量 |
|---|---|---|
| 抽样公平性 | 大样本下每株/每阵营存活率 ≈ avail/total(卡方不显著) | §7-J strain/faction 盲(私货红线) |
| 硬上限 | 任何 cell 抽后 ≤ K,且 contested cell 恰好 avail 个存活 | K 墙 |
| 守恒 | 非 contested cell 全员存活,无凭空增减 | 仲裁不造不灭 |
| faction 继承 | 子代 faction == 母株 faction(突变只改 sid) | §1.1 血统永不变 |
| 同阵营免疫 | 同 faction(任意 sid)对抗后双方 count 不变 | B2 修复 |
| 表型纯序列 | `phe[...]` gather 路径静态无 faction 索引(签名/grep 断言) | 核心不变量 §2 |
| 向量==逐株(reproduction) | 临时 scalar 参考实现对随机种子产同分布(性质,非 bit) | P2 正确性 |
| 播种 | t=0 恰 4 格非空、faction=0/1/2/3、其余空 | B1 修复 |

**性能标尺(profiler 真测,验证 §0 静态归因):** 跑 T=60 单 seed,记录每 phase 的 ms/tick(用 `torch.cuda.synchronize()` 围栏分段)。验收:128/K64 满格期 ms/tick 比首批 12.8s **降一个量级以上**(目标 <1s/tick;不达标则 profiler 重新定位)。这把"P1 是真凶"从静态推断变实测。

**回归:** 现有 285 测试中与 faction/抽样无关的(world/types/registry/recorder/phenotype_cache/common)仍全过。

---

## 7. 私货红线裁定(守门人,11 条审计清单)

- A. **禁**任何结局常数(base/z/f/μ/z_max/δ/α₀/β_fold/κ)按 faction 键控 —— 全局标量,永不 faction-keyed〔头号死罪〕
- B. **禁** 4×4 非对称对阵矩阵;kill 效率禁被 (攻方,守方) faction 对调制
- C. 播种几何 D4 精确轨道(四象限中心满足);seed 数=1、序列逐字节同 BB0
- D. 环面拓扑对四象限一视同仁,无象限独享安全边
- E. 快照同步(已锁);**空名额抽样 faction 盲(§3.2 随机键不读 sid/faction,由构造保证)**;tie-break RNG 对称
- F. 同阵营免疫 = **恰好**跳过这次 kill,不多给一分(禁同阵营共享资源/互升f/联防bonus)
- G. faction **唯一**允许出现在 Z 对抗资格谓词一个点;P 内零出现、F 强度零出现、z 量级零出现〔红线〕
- H. 四 BB0 始祖除 faction 一个 bit 外,序列+内建基元(F4Nr4/BroadSweep/P_base)逐位全同
- I. 族瞄准(序列算)在前,faction gate 资格在后,两 gate 对称复合
- J. K 墙仲裁无 per-faction 保护配额,蒸发 faction 盲
- K. 动力学**禁回读** faction 聚合量(如阵营总人口)调强度——频率依赖只许从 per-individual 对抗涌现

**对称群私货免疫定理(承 design.md):** 联合动力学在 `G=(阵营置换 S₄)⋉(空间 D4)` 下等变 → 四阵营可交换系综 → 跨 RNG 种子阵营胜率落 ~1/4 二项 CI。系统性偏离 = 私货泄漏信号(给反推侧的可证伪自检);单局碾压 = SSB = 想要的涌现,非违规。**向量化不改这个定理**(P1 新抽样 §3.2 仍 faction 盲)。

---

## 8. 验收标准(实现后必须满足)

1. **播种正确:** t=0 只 4 格非空,各 fill_per_cell 个 BB0,faction=0/1/2/3,其余全空
2. **扩张发生:** 占用格数从 4 单调增长;前沿 ~tick 160 相遇、世界 ~tick 320 填满(F4Nr4 period=5)
3. **相遇对抗:** ~tick 160 起跨 faction 同格 → 对抗触发 → 减员可观测
4. **同阵营不互斗:** 同 faction(任意 sid)永不对抗(单元测试)
5. **表型纯序列:** gather 路径无 faction 索引(签名/grep 断言)
6. **仲裁 faction 盲:** §3.2 随机键不读 sid/faction;跨种子阵营胜率落 1/4 CI
7. **schema 含 faction:** parquet 每行 6 列,每帧自包含
8. **fixation 可触发:** 单阵营统一场停机(T=450 留 ~130 tick 给机会;不强求每锅触发,SSB 随机发生)
9. **私货审计:** §7 的 11 条全过
10. **性能达标:** 128/K64 满格期 ms/tick < 1s(降一量级以上),profiler 分段验证 P1/P2 确为大头
11. **测试套绿:** 新性质测试全过 + 无关回归不破;`phase3_arbitrate` 参考版已删

---

## 9. 不在本 spec 范围(YAGNI)

- C1 dense 对抗张量 sparse 重写 —— defer 到 512 跑前(128 批 1GB 可跑)。**注:本 spec 砍掉了 arbitration/reproduction 的 Python 循环,但 antagonism 的 dense [H,W,K,K] 是另一回事(显存非速度),不在本轮。**
- 非对称始祖 / 角色系统 / 更多 backbone 模板 —— 本体验证后专门设计
- 数值轮标定(μ/K/z_max/δ/p_max/α₀/β_fold/κ 常数搜索)—— 开发期跑,本 spec 用协议占位值
- α₀/β_fold/κ 上位效应 —— 首批全关(=0),先验证阵营扩张本体
