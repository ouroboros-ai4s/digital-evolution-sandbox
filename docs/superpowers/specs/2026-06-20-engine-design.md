# Digital Evolution Sandbox — 引擎 (1/3) 设计 spec

> **状态**: 设计已逐节用户确认 (2026-06-20)，待 spec 审阅后进 writing-plans。
> **范围**: 本 spec 只做**引擎**。指标/健康闸 (2/3)、标定 harness (3/3) 是后续独立 spec。
> **上游设计 (权威，本 spec 不复述机制细节，只定实现契约)**:
> - `context/2026-06-11-16-10-design.md` — 全貌设计 (机制锁定)
> - `context/2026-06-20-numerical-round-protocol.md` — 数值轮协议
> - `context/2026-06-20-22-00-descaffold-unified-primitive.md` — 拆脚手架统一基元模型
>
> 上游与本 spec 冲突时，**上游机制定义为准**；本 spec 只决定「怎么把锁定的机制实现成 GPU-int 引擎」。

---

## 0. 目的与验收标准

引擎的唯一职责：忠实运行红皇后对称混战世界，每 tick dump 整个 512×512 格子状态，产出一批干净的涌现动力学时序数据集。

**验收标准 (本 spec 完成 = 下列全 PASS)**:
1. BB0 单始祖 (四全同拷贝铺地图) 在小网格上跑得起来，不报错、不 NaN、不一步全灭。
2. 每 tick dump 出符合已锁 schema 的 parquet (long format，`(tick, cell_x, cell_y, strain, count)`)。
3. **肉眼本体活性**: 突变体会出现、株频率在动、不冻死也不一步全灭 (这是「效果好不好」的肉眼判据，不需要指标闸)。
4. 私货回归全过 (见 §6)。
5. 同 seed 可复现；多 seed 跑出方差。

**非目标 (明确排除，留后续 spec)**: 健康判据 GATE0-4、三零模型、N2/L_ε 指标、标定 sweep、κ 自协同的实际计算、折叠组合技的实际效果、全基元表 (22+36)、始祖株特色角色系统、变体株。

---

## 1. 关键架构决策 (已逐项用户拍定)

| # | 决策 | 取舍理由 (三方 subagent 合议结论) |
|---|---|---|
| D1 | **count = int32 随机** (非 float 确定性) | float mean-field 是穿 agent-based 外衣的确定性 ODE，结构性摧毁 drift/founder/克隆干涉 (P1 原理上无法显现)，且低 μ 下把稀疏株集炸成稠密尘云反而更慢。int 随机的 `Binomial(n,μ)` 多数恰为 0 → 突变真稀有整数 → 株集真稀疏。 |
| D2 | **复现靠固定 seed + 多 seed ensemble** (非删 RNG) | 确定性要从「可复现」拿，不从「删随机」拿。同 seed 可复现/可二分，多 seed 测方差 = 数据本身。 |
| D3 | **world = per-cell slot 张量, C=K** | `strain_id[512,512,K]` + `count[512,512,K]`。鸽笼原理 (K 个体最多 K 个型) → **溢出概率严格 0** → 无溢出策略 → 无「strain-blind 还是私货」纠结，整类 bug + 数据污染一次清零。K=256 时 536MB，研究沙盒可接受。 |
| D4 | **绝不 dense `[512,512,S]`** | S (活跃株数) 随突变增长，dense 张量随 S 平方级炸内存 (S=10k→10GB 全是零)。 |
| D5 | **对抗配对走稀疏非零株** | 每格实际 ~5-10 株 (两方评审 init 收敛值)，不建 dense K×K 配对矩阵；只在真实存在株对上算掩码 AND。 |
| D6 | **一步到位 GPU-int** (从第一版奔最终形态) | 用户长期诉求 (算力不设限)。GPU/int/512×512 全保留。 |
| D7 | **scatter-merge 第一版纯 torch 算子**，profile 确认热点后再写自定义 CUDA kernel | 先要正确再要快。株 id 对齐合并是唯一真需自定义 kernel 的热点，但本体动力学没证明前不投重工程。 |
| D8 | **第一版就异步双缓冲 dump** | 每 tick 全图 dump 是潜在真 wall-clock 瓶颈 (可能比 kernel 还拖)，后台线程双缓冲不阻塞 GPU 计算。 |
| D9 | **v1 stub: κ=0 / α₀=0 折叠 / 基元子集** | 设计自己背书 (「κ 可先取 0 跑通本体」)。与 D6 不冲突 (GPU 全保留，只收窄机制丰富度)，最快验证本体活性。 |

---

## 2. 模块边界 (责任隔离)

6 个边界清晰、可独立测试的单元：

| 模块 | 职责 | 依赖 | 关键接口 |
|---|---|---|---|
| **registry** | 基元注册表 `name→formula`；字母表 + 字母族 (4 族 N<F<P<Z)；BB0 backbone 模板 (locked 布局 + 折叠表) | 无 (纯数据 + 纯函数) | `phenotype(sequence) → 表型束` |
| **phenotype-cache** | 序列→表型解析与缓存 (每条新株一辈子算一次)；全局 `strain_id ↔ sequence` intern 表 | registry | `get_or_mint(sequence) → (id, 表型)` |
| **world** | 状态张量 `strain_id`/`count [512,512,K]`；快照 | 无 | `snapshot()` / 格子读写 |
| **kernels** | 四 PHASE 张量运算：对抗 (稀疏配对) / 繁衍 scatter / 突变采样 / K墙仲裁 | world, phenotype-cache | `phase1/2/3(world, snap) → world'` |
| **engine** | tick 循环编排 PHASE0-3；RNG seed 管理；停机条件 | 以上全部 | `step()` / `run(T)` |
| **recorder** | 异步双缓冲 parquet dump (后台线程) | world | `dump(tick, world)` |

私货红线落点：**registry** 表型只读序列 (守门人判据)；**kernels** 的 K墙/溢出/对抗全 strain-blind。

---

## 3. 数据表示

### 3.1 world 状态

```
strain_id : int32 [512, 512, K]   # 每格 K 个 slot，存株 id；空 slot = 0 (保留 id)
count     : int32 [512, 512, K]   # 对应 count；空 slot count=0
```

- C=K (D3)：slot 数 = 每格个体上限，溢出物理不可能。
- 空 slot 约定：`strain_id=0` 为保留「空」哨兵 id，`count=0`。dump 时只写 `count>0` 的 slot (§7「只存非空记录」)，哨兵 id 永不泄漏进数据集。
- 同一格内同一株只占一个 slot (合并保证由 scatter-merge 维护，见 §4)。

### 3.2 全局表型表 (phenotype-cache)

```
strain_id (int32, 由原子计数器铸造，0 保留给空)
  → sequence (str, 完整序列，只此一处持有；dump 时映射回字符串)
  → 表型束:
      f            : float   繁衍比例
      affected     : 方向掩码 / 波及格规则 (4N/8A/hash 方向…)
      p_leave      : float   迁出概率
      z_raw        : float   对抗交换比 (soft cap 前)
      prey_mask    : int64   猎物特征掩码 (瞄哪些族/motif)
      feature_mask : int64   自身特征掩码 (被对手 prey_mask AND)
      p_x          : float   突变 rate
      spectrum     : 候选表 (id/族 → q，Σq=1)
      period       : int     时钟周期
      phase_type   : enum    {Z→1, F→2, P→嵌入2}
      fold         : 折叠组合元数据 (v1 stub，存着不发力)
```

- 新株罕见 (D1) → 解析摊销，每 tick 只 batch-parse 当 tick 新出现的 id。
- 表型是 sequence 的纯函数 (守 1.3 + 守门人判据)：解析过程**只读 sequence**，绝不读 count/对手/步数/占用。

### 3.3 对抗稀疏表示 (D5)

PHASE1 不建 dense `[512,512,K,K]` 配对张量。每格 gather 实际非零株 (~5-10) 的 `prey_mask`/`feature_mask`，在这个小集合上算配对命中 `(attacker.prey_mask & prey.feature_mask) != 0`。配对命中是株对级、可缓存。

---

## 4. tick 计算管线 (PHASE0-3 → 张量运算)

严格照上游锁定时序：**对抗 → 繁衍(含突变) → K墙仲裁**，异相位时钟 `(T-birth)%period`。

### PHASE 0 — 取快照
```
snap = world.clone()        # strain_id + count 双张量
```
本 tick 所有「量」读 snap，不读活世界 (同 tick 内先动基元不影响后动)。

### PHASE 1 — 对抗 (读快照)
- 触发：`(T-birth) % Z_period == phase_offset` 且攻方 Z 基元猎物掩码命中被攻方特征掩码。
- 稀疏株对结算 (§3.3)：
  ```
  z_eff = z_max · z_raw / (z_max + z_raw)        # Michaelis-Menten soft cap
  kills = min(b, a · z_eff)                       # b,a 读快照
  被攻方 count -= kills
  攻方 count   -= kills / z_eff                   # 攻击自损 (也用 z_eff，治本)
  ```
- 双向同时：用开打前快照 (a,b) 同时算两个方向，再一起扣 (不依赖出手顺序)。
- 多株/多攻击叠加 v0：`raw_damage > b_snap` 时 `ratio=b_snap/raw` 按贡献比例分摊，攻方自损同比例缩，双向不出负 count。
- delta 暂存批量写入 world。

### PHASE 2 — 繁衍 + 突变 (量读快照，空位读对抗后活世界)
- 触发：`(T-birth) % F_period == offset`。
- scatter：`torch.roll` 把 slot 张量按方向掩码移位，撒到 `[512,512,9,K]` 到达缓冲；源格某株 a 个 → 每被波及邻格各收 `a·f` 个 (各格各 af，非总量分摊)。
- 突变嵌入 (SHM，落地那刻)：
  ```
  mutate_n = Binomial(到达数, p_x)               # 多数恰为 0 (低 μ)
  spectrum 抽样 = Gumbel-argmax(log q)            # 避开变长 multinomial
  → 拆为 [原株 (到达-mutate_n) 个 + mutant(s) mutate_n 个] 进 pending
  ```
  - `pending_arrivals` 是**动态结构** (非固定 C 数组)：tick 内 distinct 量比 resident 大，蒸发前可超 K。固定 K slot 只存仲裁后 resident。
- 迁出：`world[g][s] -= min(a_snap · p_leave, a_snap)` 写活世界。
- **PHASE 2 只生成 pending，绝不提前落地。**

### PHASE 3 — K墙仲裁 (统一消化 pending)
- 每目标格：`available = K − 占用`，`total_arriving = Σ pending`。
- 装得下全落；装不下按 **Binomial 抽稀** `survival = available / total_arriving`：
  ```
  幸存 = Binomial(arriving_s, survival)           # 均值恰好等比例 → 守「无隐藏权重」红线
  ```
  超额蒸发，**绝不挤掉活体**。±√ 非守恒 = 合法 demographic noise。
- 折叠强化在此算 (落地写入前一次性，非每 tick 维护状态) — **v1 stub α₀=0，不发力**。
- mutant(s) 是全新序列 → `phenotype-cache.get_or_mint` 插新 id；收敛突变 (变出已有序列) 按序列作 key 自动合并。

### scatter-merge (株 id 对齐合并，唯一真热点)
邻格「slot 3」≠ 本格「slot 3」，到达必须按 strain_id 合并进本格 slot。
- **v1 = 纯 torch 算子** (D7)：利用局部性 (到达来自 ≤8 邻格+自身+突变，bounded ≤9K entries → K slots)，本地 coalesce + segment-reduce，不做全局 sort。
- 后续 (profile 确认热点)：自定义 CUDA scatter kernel (open-addressing hash slot + atomicAdd + CAS 抢空 slot，O(arrivals))。

### 时钟调度
异相位 (已锁)：个体记 `birth_tick`，触发 `(T-birth_tick)%period==0`。多基元独立时钟无冲突。

---

## 5. v1 故意 stub (YAGNI，设计自己背书)

| 机制 | v1 状态 | 解锁时机 |
|---|---|---|
| **κ 同通道族自协同** | `κ=0`，`n_same` 扫描不实现 (dead code) | 本体活性证明后 |
| **折叠组合技** (Z+P/Z+F/P+F 新通道 + α₀ + β_fold) | `α₀=0`，BB0 折叠区当元数据存着不发力 | 本体活性证明后 |
| **基元表** | 只实现 BB0 locked 集 (F4Nr1 / BroadSweep / P_base / N₀) + 给突变留阶梯的 ~2 个 (1 更强 F、1 个 P) | 逐步扩到 22+36 |
| **变体株 / 始祖特色角色** | 不做 | 本体验证通过后专门 spec |

不与 D6 (GPU-int 一步到位) 冲突：GPU/int/512×512/异步 dump 全保留，只收窄机制丰富度，最快验证本体活性。

---

## 6. 测试与验收

### 单元测试
- registry 表型是序列纯函数：同序列 → 同表型 (确定性、无副作用)。
- K墙仲裁：抽稀均值 = 等比例 `available/total` (统计验，多次采样均值收敛)。
- 溢出永不触发：构造高到达场景，验 distinct ≤ K 恒成立。
- 对抗双向对称：交换甲乙序列，kills/自损对称。
- z_eff soft cap：z_raw→∞ 时 z_eff→z_max。

### 私货回归 (项目存亡线)
- **G10 同株免疫**：四全同拷贝 BB0 在 t=0 不自相残杀 (内建 BroadSweep 在 a==b 阶段不结算)。
- **表型不越界**：解析路径静态/运行时验证只读 sequence，不触 count/对手/步数/占用。
- **K墙无隐藏权重**：多源抢同空格，幸存比例只由送达量定，无 strain-level 偏置。

### 本体活性闸 (验收，肉眼 + 最简统计)
小网格 (如 128×128) BB0 跑 N tick：
- 突变体会出现 (新 strain_id 被铸造)。
- 株频率在动 (非全程冻结)。
- 不一步全灭、不冻死。

### 复现
- 同 seed → bit-identical 结果。
- 多 seed → 跑出轨迹方差 (验随机性真在起作用)。

---

## 7. 引擎产出契约 (后两个 spec 的输入接口)

- **数据集**: 每局一个 parquet，文件名 = 时间戳。long format，每行 `(tick, cell_x, cell_y, strain, count)`，strain 列存完整序列字符串 (parquet 字典编码压缩)。只存非空记录，每 tick 全量快照 (任意帧自包含)。
- **运行接口**: `engine.run(T, seed) → 写出 parquet 路径`；停机条件 = 某谱系统一全场 / 全灭 / 跑够 T 步 (运行层，不告诉系统什么算赢)。
- 指标闸 (2/3) 消费 parquet；标定 harness (3/3) 驱动 `engine.run` 跑 sweep。

---

## 附：开放的实现细节 (留 writing-plans / 实现期定，不挡设计)

- slot 内空位整理策略 (compaction 时机)。
- `pending_arrivals` 动态结构的具体形态 (COO 事件列表 vs per-cell 临时 dict)。
- 方向掩码 (4N/8A/hash 方向) 的张量编码。
- batch-parse 新株的触发与批大小。
- 异步 dump 的缓冲深度与背压策略。
