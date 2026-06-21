# 阵营扩张混战重构 — Design Spec

> **⚠ SUPERSEDED (2026-06-21).** 本文已被 `2026-06-21-faction-and-vectorization-redesign.md` 取代 —— faction 重构 + 引擎热路径全向量化(P1/P2/P3)合并为一份 spec(同样两个 kernel 文件要一起重写,分开做会改两遍)。本文 faction 部分原样并入新文 §2/§4,保留作历史存档,**不再维护**。

**日期:** 2026-06-21
**项目:** Digital Evolution Sandbox(数字进化沙盒,512×512 红皇后混战,采集无偏时序数据集供 selop 反演)
**类型:** 设计修正(伤筋动骨)——纠正三个根本性实现偏差 + 一处 meta 地基修订

---

## 0. 背景:为什么要重构

首批四组数据(2026-06-20 跑)与用户设想**完全不符**。根因有三处偏差,经代码核查 + 三路 subagent(机制保全 / 架构 / 私货守门人)验证确认:

| # | 偏差 | 代码位置 | 后果 |
|---|---|---|---|
| **B1 播种** | BB0 铺满全部 16384 格,世界一开始就满 | `src/des/world.py` `init_bb0`(`strain_id[:,:,0]=bb0`) | 无扩张过程、无相遇前沿、空间均质 CV≈0 |
| **B2 免疫** | 对抗免疫按**序列**判(`sid_i!=sid_j`),非按阵营 | `src/des/kernels/antagonism.py`(diff_strain 谓词) | 同阵营突变后代互斗(内战);异阵营收敛到同序列反而免疫 |
| **B3 扩张** | BB0 自带繁衍基元 `F4Nr1` 只向北一个方向 | `src/des/registry.py` `_F["F4Nr1"]`(`((-1,0),)`) | 即使改 4 格起步,也只长成竖条、永不相遇 |

> **代码位置按符号定位,不按行号。** 上表与下文 §2 引的位置以**函数/符号名 + 字典键**为准(`init_bb0` / `_F[...]` / `_LOCKED` / diff_strain 谓词);源树在 `src/des/`,行号会随编辑漂移,实现时按符号 grep 定位。已知锚点:`registry.py` 的 `_F`/`_Z`/`_P` 基元字典 + `_LOCKED = {1:"F4Nr1", 5:"BroadSweep", 7:"P_base"}`(0-indexed 槽位)。

**用户设想的正确玩法:** 128×128 世界初始只有 4 个格子(四象限中心),红/黄/蓝/绿四阵营各占一格,初始全是同一条 BB0 序列、仅阵营标签不同。各阵营从自己那一格繁衍、突变、二维扩张铺开,在扩张前沿**相遇处**发生**跨阵营**对抗(中和,双方各自减员)。同阵营(含其所有突变后代)血统永不互斗。

---

## 1. 用户已拍板的决议(本 spec 的硬约束)

1. **阵营血统永不变** — faction 从始祖继承的不可变标签,突变只改序列、不改 faction。
2. **对抗只有中和** — 双方各自按既有公式减员(`kills=min(b,a·z)`,自损 `kills/z_eff`)。**无"猎杀/同化"概念**(那是错误臆造,删除)。
3. **四阵营全用 BB0** — 仅 faction 不同。最纯对照组,私货免疫定理成立。
4. **四象限中心播种** — 四个始祖格子隔得尽量远,给扩张和相遇留空间。
5. **meta 地基②修订** — 从「人人互为对手」改为「群结构混战:4 阵营、群内联盟、群间对抗」。对称不偏袒。
6. **BB0 繁衍基元改 F4Nr4** — 四方向 f=0.50,使阵营能二维扩张相遇。
7. **fixation = 单阵营统一场** — 某 faction 干掉其他所有阵营即停机(不管它内部多少突变序列)。
8. **世界边界 = 环面 torus** — 保持 `torch.roll` 现状,四象限对称等距无偏袒。
9. **T = 450**(从首批锁定的 200 上调)— F4Nr4 period=5 → 扩张 1格/5tick/轴 → 填满 ~tick 320,T 须覆盖 前沿瞬态(0-160)+ 相遇带对抗(160-320)+ 满格后红皇后竞争(320-450)。**T 是纯运行层旋钮,不碰任何机制/基线/私货。**(memory `project_des_first_batch_config` 同步:T200→T450)

> **权威指针:** 本 spec 的宏观动力学、双正交身份不变量、对称群→可交换系综→1/4 CI 私货自检、faction 红线裁定,均以 design.md 新章节「世界初始化 · 阵营模型 · 目标宏观动力学」为**权威源**。本 spec 是该章节落到 5 处 kernel 改动的**实现规格**,若两者冲突以 design.md 为准。

---

## 2. 架构方案(三路验证后锁定:方案 C — 平行 faction 张量 + 双正交身份)

**核心不变量(写进守门人审计):**
> `sid` 是唯一表型 key(phenotype gather 路径 `phe[sid]` 在任何 kernel 中**禁止**出现 faction 索引);`faction` 是 team key;占位 slot 身份 = `(sid, faction)` 对。

**为什么不选复合 id 方案:** 把 (faction,sequence) 编进 sid 会让表型缓存膨胀 4×,且红-BB0/蓝-BB0 必须靠 builder 自觉令表型相等——红线从"结构保证"降级为"约定"。方案 C 下 `phe[sid]` 物理上看不到 faction,同序列不同阵营自动共享同一缓存行,结构性保住红线。

### 2.1 数据结构改动

**`world.py`:**
- `World.__init__` 新增 `self.faction = torch.zeros((H,W,K), dtype=torch.int8, device=device)`(128²/K=64 = 1MB;512²/K=256 = 64MB,相对 dense 对抗张量可忽略)
- `snapshot()` 返回三元组 `(strain_id, count, faction)`
- 新增 `init_factions(H,W,K,device,table,fill_per_cell,n_fac=4)`:**替代** `init_bb0` 的全场铺地。只在四象限中心 `(H//4,W//4)`、`(H//4,3W//4)`、`(3H//4,W//4)`、`(3H//4,3W//4)` 各播种 `fill_per_cell` 个 BB0,faction 分别 = 0/1/2/3。其余格全空。
- **D4 对称校验:** 四个播种点必须是同一点在 D4 群(旋转+镜像)下的精确轨道——四象限中心天然满足(等距中心、等距最近壁、成对距离对称)。

### 2.2 对抗免疫改动(`antagonism.py`)

- gather faction:`fac = faction.long()` 然后 `fac[sid_long]`?**否**——faction 是 slot 级状态,直接 `fac_slot = faction.long()`(已是 `[H,W,K]`,不经 sid 索引)
- 把 `diff_strain = sid_i != sid_j`(L61)**替换**为 `diff_faction = fac_i != fac_j`,其中 `fac_i = fac_slot.unsqueeze(-1)`、`fac_j = fac_slot.unsqueeze(-2)`
- gate 必须是**对称等价谓词** `fight ⟺ faction_i ≠ faction_j`,对所有阵营对逐字相同。**禁** 4×4 非对称对阵矩阵。
- prey_mask / feature_mask(选靶,按族)仍纯由序列算,faction 只 gate"打不打"不 gate"打谁/打多狠"。z 全程只读攻击方序列。
- dense `[H,W,K,K]` 内存**不因 faction 增长**(一个 bool 比较替换另一个)。

### 2.3 繁衍改动(`reproduction.py`)

- **分组键从 `sid_val` 升为 `(sid_val, fac_val)` 对**(这是 faction 架构唯一实质改造点,也是陷阱所在:当前按 sid 分组会把红-BB0/蓝-BB0 混进同一次 scatter 丢 faction)
- `present` = 在 `(snap_sid, faction)[fires]` 上取 unique 对
- `slotmask = (snap_sid==sid_val) & (snap_faction==fac_val) & fires`
- `ArrivalBuffer` 加第五列 `_fac`;`add(ty,tx,sid,cnt,fac)`;`tensors()` 同步返回 faction
- 子代:`child_id = table.get_or_mint(child_seq)`(由序列定,不变),`fac` 携带父 `fac_val`(faction 跟父走、不随 sid 变)——突变继承 faction 在此天然成立
- `_mutate_sequence` 不变(只读序列,纯)
- BB0 locked 繁衍基元从 F4Nr1 改 F4Nr4(见 §3)
- p_leave 迁出消失语义不变(稀疏扩张期实测非问题:30% 流入 ≫ 5% 迁出成本)

### 2.4 K 墙仲裁改动(`arbitration.py`)

- **抽样 randperm 段逐字不动**(faction-盲是公平性来源,等同 strain-盲;若仲裁偏袒某阵营=隐藏权重=私货)
- coalesce key 升为 `key = cell*(MAXSID*NFAC) + sid*NFAC + fac`(int64 安全)
- slot 合并判据 `(slots_sid==s) & (slots_fac==fac)`(否则红/蓝 BB0 错误并进一个 slot)
- writeback 同步写 `new_faction`
- **无 per-faction 配额**:K 是全局个体上限,蒸发 faction-盲

### 2.5 数据采集改动(`recorder.py`)

- schema 加 `("faction", pa.int8())`
- `dump` gather `world.faction[nz]`
- 每行变 `(tick, cell_x, cell_y, strain, faction, count)`——这样数据分析才能区分阵营

### 2.6 引擎改动(`engine.py`)

- `step()` 三处快照/写回带上 faction
- `_fixated()` 改为**单阵营统一场**:`distinct_factions == 1`(全场存活个体只剩一个 faction)
- 停机仍是 run 层职责,不进 kernel

---

## 3. F4Nr4 作为 BB0 繁衍基元(B3 修复)

- 当前 BB0 layout 的 locked F 位坐 `F4Nr1`(`registry.py` 单向北 `((-1,0),)`,f=0.30)
- 改为 `F4Nr4`(四方向 `((-1,0),(1,0),(0,-1),(0,1))`,f=0.50,**period=5**)——`src/des/registry.py` 已定义为"标准扩张基准(=原F4)"
- 改动点:`BB0_TEMPLATE["layout"]` 里把 F 位的字母从 F4Nr1 换成 F4Nr4
- **★扩张速率 = 1 格 / 5 tick / 轴(不是 1 格/tick)**:F4Nr4 period=5,繁衍基元每 5 tick 才触发一次,故钻石波每 5 tick 推进 1 格。**时间线(已按 period=5 重算):**
  - 相邻象限中心隔 64 格,两前沿相向各走 32 格 ×5 = **前沿约 tick 160 相遇**
  - 最远空角(世界正中/四壁角)离最近殖民地 64 格 ×5 = **世界约 tick 320 填满**(占用率→100%)
  - tick 320 之后满格,换血靠"对抗杀人→空名额→繁衍/突变填"(design line 101 满格冻结动力学)
  - **T=450**(锁定值,memory 同步更新):覆盖 前沿瞬态(0-160)+ 相遇带混合对抗(160-320)+ 满格后红皇后竞争(320-450)。fixation 不强求每锅触发(对称系统靠 SSB 随机发生,共存也是合法结局)
- **私货检查:** 四阵营全用同一条改后 BB0,无阵营拿不同基元 → 对称,不偏袒。F4Nr4 是设计文档已锁的中性扩张基准,非为某方加强。period=5 不动(它是 birth 盲定常数、四阵营同 BB0 故对称,不属 7 个结局常数,不碰基线)。

---

## 4. 私货红线裁定(守门人,已通过 + 边界条件)

**Q1 阵营免疫 = CLEAN(带硬边界):** faction gate 决定"打不打"非"打多狠",z 仍纯由序列算,落在判据①合法侧;faction 是明文世界状态可被反推者重建,过判据②。与既有 gate("猎物瞄族""空间邻接")同构。**边界:** gate 必须是对称等价谓词,一旦退化成非对称对阵矩阵或某阵营免被打特权 = VIOLATION。

**Q2 私货免疫定理 = 升级成立:** 旧"四全同休眠"论证作废(现在 t=0 即互殴);新定理 = 对称群 `G=(阵营置换)⋉(空间D4)` 下联合动力学**等变** → 四阵营**可交换系综** → 任意阵营 `E_seed[优势_k]` 全相等,无偏袒。**关键:** 断言的是**分布层/期望层**可交换性(跨 RNG 种子胜率落在 ~1/4 二项 CI 内),**非单局相等**——单局某阵营碾压是自发对称破缺(SSB),正是想要的涌现。这给了反推者一个**可经验证伪的检验**(系统性偏离 1/4 = 私货泄漏信号)。**四前提:** ①四序列逐字节全同 ②规则对 faction 对称(无 faction-索引常数)③播种是 D4 精确对称轨道 ④结算对 faction 盲(快照同步 + tie-break RNG 对称、禁"faction-id 小者胜")。

**Q3 审计清单(11 条,写进 spec 防渗入):**
- A. **禁** 任何结局常数(base/z/f/μ/z_max/δ/α₀/β_fold)按 faction 键控 —— 全局标量,永不 faction-keyed〔头号死罪〕
- B. **禁** 4×4 非对称对阵矩阵;kill 效率禁被 (攻方,守方) faction 对调制
- C. 播种几何必须 D4 精确轨道(四象限中心满足);seed 数=1、序列逐字节同 BB0
- D. 环面拓扑对四象限一视同仁,无象限独享安全边
- E. 快照同步(已锁安全);空名额仲裁 faction 盲;tie-break RNG 对称
- F. 同阵营免疫 = **恰好**跳过这次 kill,不多给一分(禁同阵营共享资源/互升f/联防bonus);Z+P/Z+F 折叠跨类注入在同阵营跳过时须同步跳过
- G. faction **唯一**允许出现在 Z 对抗资格谓词一个点;P 内零出现、F 强度零出现、z 量级零出现〔红线〕
- H. 四 BB0 始祖除 faction 一个 bit 外,序列+内建基元(F4Nr4/BroadSweep/P_base)逐位全同
- I. 族瞄准(序列算)在前,faction gate 资格在后,两 gate 对称复合
- J. K 墙仲裁无 per-faction 保护配额,蒸发 faction 盲
- K. 动力学**禁回读** faction 聚合量(如阵营总人口)调强度——频率依赖只许从 per-individual 对抗涌现

---

## 5. 采集价值(对 selop 反演的意义)

经机制保全审计师确认:**扩张+相遇前沿数据 ≫ 全场满播数据**(对 selop 反演),条件是上述三处修复全部到位。

- 全场满播 BB0:空间均质 CV≈0,μ_t 近乎静止 delta,无空间传输、无频率梯度、无相遇——measure-flow 几乎没东西可反演
- 扩张+前沿:行波(Fisher-KPP 式)产生真实空间梯度、前沿奠基者瓶颈(allele surfing,扩张遗传学教科书现象)、前沿混合带多株共存变频率 → 真正的**频率依赖克隆竞争** = selop 要的非平凡 μ_t 传输 + β 信号
- **可交换性检验**(§4-Q2)同时给反演侧一个独立的私货自检指标

---

## 6. 不在本 spec 范围(YAGNI)

- C1 dense 对抗张量 sparse 重写 —— 用户已 defer 到 512 目标跑前,128 批 1GB 可跑
- 非对称始祖 / 角色系统 / 更多 backbone 模板 —— 本体验证后专门设计
- 数值轮标定(μ/K/z_max/δ/p_max/α₀/β_fold/κ 具体常数搜索)—— 开发期跑,本 spec 用协议占位值
- α₀/β_fold/κ 上位效应 —— 首批保持全关(=0),先验证阵营扩张本体

---

## 7. 验收标准(实现后必须满足)

1. **播种正确:** t=0 只有 4 格非空,各 `fill_per_cell` 个 BB0,faction=0/1/2/3,其余 16380 格全空
2. **扩张发生:** 占用格数从 4 单调增长(非 t=0 即满);前沿 ~tick 160 相遇、世界 ~tick 320 填满(F4Nr4 period=5,1格/5tick/轴,见 §3)
3. **相遇对抗:** ~tick 160 起存在跨 faction 同格 → 对抗触发 → 减员事件可观测
4. **同阵营不互斗:** 同 faction 个体(任意序列)永不对抗(单元测试断言)
5. **表型纯序列:** 表型 gather 路径无 faction 索引(签名/审计断言)
6. **仲裁 faction 盲:** randperm 段不读 faction,跨种子阵营胜率落在 1/4 二项 CI 内
7. **schema 含 faction:** parquet 每行 6 列,可区分阵营,每帧自包含
8. **fixation 可触发:** 单阵营统一场触发停机(T=450 留 ~130 tick 满格后竞争给 fixation 机会;**不强求每锅触发**,对称系统靠 SSB 随机发生,跑满 T 共存也是合法结局)
9. **私货审计:** §4 的 11 条清单全过(无 faction-索引强度常数)
10. **回归不破:** 现有 285 测试中与 faction 无关的部分仍全过

---

## 附:三个 bug 的设计文档源头(需同步回写)

- design.md **第 320 行**「一个模板多拷贝**铺地图**」← B1 的设计源头,改为"四象限中心四格播种"
- design.md **第 10 行**「四种**不同的**始祖株」vs 第 320 行「初始相同」内在矛盾 ← 统一为"四阵营同 BB0、仅 faction 不同"
- G10「同株免疫」(第 335 行)← B2 源头,泛化为"同阵营免疫"
- BB0 layout F 位 F4Nr1 ← B3 源头,改 F4Nr4
- meta 地基②「人人互为对手」← 修订为"群结构混战"(§1.5)
