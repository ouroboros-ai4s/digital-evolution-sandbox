# 基元总名册 (Primitive Roster) — config

> DES 五池基元 config。格式: `name: 功能` + `formula`(markdown math)。未设计=留空。
> family rank: `N=0 < F=1 < P=2 < Z=3 < A=4`。容量: N8 / F8 / P12 / Z16 / A24 = 68。

## OPEN

- **[OPEN-1] 突变选择机制**: ✅已解(2026-06-24 重订)。A 池覆写列表 = backbone 上下文门控谓词 $\{\text{株}:n_{\text{locked}}(\text{chan})\ge\theta_{\text{chan}}\}$——某 A 只在 backbone 已充分该通道专化的物种内、由 gran 匹配的可变插槽突变涌现。「建难」由门控自然涌现(极端 F 变体只在 F-重株里冒头),非手写阶梯。**废弃**:逐条点名前驱、「非对称律/阶梯/破易建难」(均非用户设计的概念)。
- **[OPEN-2] Z 单字母狙击猎物清单**: ✅已解,改特征阈值(见 Z 池末三条)。
- **[OPEN-3] backbone 大幅扩充**: 长度待定(给「主体覆写 N」提供更多 N0 槽)。基元 config 与此无关。
- **A 池(24)**: 8 个 FPZ 精选拷贝 + 16 专属(甲新通道/乙极端变体)。Phase 2 设计中,新通道须过闸。

---

## N 池 — 中性 filler (rank 0, 8)

仅产 vis∈[0,1](可见度,影响被 N-猎对抗命中概率),无 f/z/p。
> from(前驱谱):$N_{res}=\{N0,N1,N2,N3,N5,N6\}$(残基)、$N_{mtf}=\{N4,N7\}$(motif)。中性互漂仅同粒度 N 互换;N0 额外是功能丧失唯一落点(任意 F/P/Z→N0 单步,功能丧失落点)。

- **N0**: 正典零位 filler,BB0 六插槽初始值,vis 基线。
  - $\text{vis}=0.20,\; f=z=p_{\text{add}}=0,\; \text{spectrum}=\varnothing,\; \text{gran}=\text{residue}$
  - from: $(N_{res}\setminus\{N0\})\cup\{s:\text{fam}(s)\in\{F,P,Z\}\}$ — 中性残基互漂 + 功能漂回正典空白(均 residue 粒度)
- **N1**: 折叠端点 filler,可登记折叠区成员当对侧端点,vis 略升。
  - $\text{vis}=0.40,\; f=z=p_{\text{add}}=0,\; \text{fold-endpoint},\; \text{gran}=\text{residue}$
  - from: $N_{res}\setminus\{N1\}$ — 中性残基互漂(fold-endpoint 亦 residue)
- **N2**: 高暴露 filler,N-猎对抗优先靶,占此位=自愿当诱饵。
  - $\text{vis}=0.70,\; f=z=p_{\text{add}}=0,\; \text{gran}=\text{residue}$
  - from: $N_{res}\setminus\{N2\}$ — 中性残基互漂
- **N3**: 隐身 filler,极低 vis,几乎逃过 vis 加权命中。
  - $\text{vis}=0.15,\; f=z=p_{\text{add}}=0,\; \text{gran}=\text{residue}$
  - from: $N_{res}\setminus\{N3\}$ — 中性残基互漂
- **N4**: 中性 motif 块,整块占位无产出,作大跳变中性落脚。
  - $\text{vis}=0.35,\; f=z=p_{\text{add}}=0,\; \text{gran}=\text{motif}$
  - from: $(N_{mtf}\setminus\{N4\})\cup\{\text{gran}=\text{motif}\ni F/P/Z\}$ — motif 块互换 + 功能 motif 漂回中性 motif
- **N5**: 真零可见 filler,vis=0 序列空间盲区,终极隐身位。
  - $\text{vis}=0.00,\; f=z=p_{\text{add}}=0,\; \text{gran}=\text{residue}$
  - from: $N_{res}\setminus\{N5\}$ — 中性残基互漂
- **N6**: 满暴露诱饵,vis=1.0 必中靶,拉走对抗火力。
  - $\text{vis}=1.00,\; f=z=p_{\text{add}}=0,\; \text{gran}=\text{residue}$
  - from: $N_{res}\setminus\{N6\}$ — 中性残基互漂
- **N7**: 隐身 motif 块,整块低 vis 占位。
  - $\text{vis}=0.10,\; f=z=p_{\text{add}}=0,\; \text{gran}=\text{motif}$
  - from: $(N_{mtf}\setminus\{N7\})\cup\{\text{gran}=\text{motif}\ni F/P/Z\}$ — motif 块互换

### 弱 FPZ 骨架料子表(2026-06-24 加;backbone-eligible 标记,不占 N8 名额、不改原 family/rank)

> 用户裁定(2026-06-24):无视疑似私货,按量级直接取各池最低档当骨架可坐料。这 6 条仍留在原 F/P/Z 池、保留原 rank(rank 是 $q=\text{aff}$ 的输入,改成 rank0 会污染突变谱——正确性问题,故不动),仅在此登记 backbone-eligible,把「可坐 backbone 的近中性料集合」从 N8 扩为 N8 + 本子表(供 OPEN-3「更多骨架槽」)。

- **FDRIFT** (F, f=0.15) / **F4Nr1** (F, f=0.30) — F 池最低两档
- **P_base** (P, rate=μ) / **P_slow_drift** (P, rate=μ) — 唯二零加码纯地板
- **BroadSweep** (Z, z=0.40) / **Scatter Nip** (Z, z=0.40) — Z 池最低 z
  - ⚠ Z 坐 backbone = 永久开火战士(非低扰动 filler),与 F/P 语气不同;按用户「全 6 条含 Z」纳入,数值轮可复核去留。

## F 池 — 繁衍 (rank 1, 8)

吐 f(繁衍比例) / dirs(波及格) / p_leave(迁出率) + T(周期)。后代=Binom(count,f) 散到 dirs;迁出=Binom(count,p_leave) 消失。堆叠 $f=1-\prod(1-f_i)$。
> from(前驱谱):获得繁衍功能 = 从 N_res 或同族邻档单步可达(Δrank≤1)。residue 档前驱 = $\{N_{res}\}\cup\{\text{同 gran 的 F-residue 邻档}\}$;motif/相位档(FBURST)前驱含 motif N。**禁从 rank≥3(Z)直接变出 F**(Δrank≥2 跳须经 P_zscan_invert 等中转,见 A 池)。

- **F4Nr1**: 最弱档脉冲,4 邻随机 1 格,BB0 可选起点。
  - $f=0.30,\; \text{dirs}=\{(-1,0)\}\,(\text{1 rand of 4-nbr}),\; p_{\text{leave}}=0.05,\; T=4$
  - from: $N_{res}\cup\{\text{F4Nr3}\}$ — N 漂出最弱繁衍 / 强档退一阶(降档易)
- **F4Nr4**: 标准强档,四方向全 4 格全向扩张,BB0 locked@pos1。
  - $f=0.50,\; \text{dirs}=\{(-1,0),(1,0),(0,-1),(0,1)\},\; p_{\text{leave}}=0.15,\; T=5$
  - from: $\{\text{F4Nr3},\text{FCLUMP}\}$ — 由邻近繁衍档升/侧变
- **F4Nr3**: 中强档,4 邻随机 3 格(hash 留缺口)。
  - $f=0.40,\; \text{dirs}=\{\text{3 rand of 4-nbr, by seq-hash}\},\; p_{\text{leave}}=0.12,\; T=5$
  - from: $\{\text{F4Nr1},\text{F4Nr4}\}$ — 同族繁衍档互变
- **FSTACK**: 守孤岛,原地堆叠零外扩,垫高 count 抢满格名额。
  - $f=0.60,\; \text{dirs}=\{(0,0)\},\; p_{\text{leave}}=0.00,\; T=3$
  - from: $N_{res}\cup\{\text{FCLUMP}\}$ — N 漂出囤积型 / 团块收敛为原地
- **FFRONT**: 单向锋线,固定 1 方向(hash 锁)楔形推进,高迁出露背。
  - $f=0.50,\; \text{dirs}=\{d_{\text{hash}}\}\,(\text{1 fixed}),\; p_{\text{leave}}=0.25,\; T=4$
  - from: $\{\text{F4Nr1},\text{FDRIFT}\}$ — 由低档定向化(随机→固定方向)
- **FDRIFT**: 低量高频游走,随机 1 格漂移,最高迁出造飞地。
  - $f=0.15,\; \text{dirs}=\{\text{1 rand of 4-nbr/tick}\},\; p_{\text{leave}}=0.30,\; T=2$
  - from: $N_{res}\cup\{\text{F4Nr1}\}$ — N 漂出最低量游走 / 弱档高迁出化
- **FCLUMP**: 团块扩张,沿单轴向 2 格(hash 锁轴)互撒。
  - $f=0.45,\; \text{dirs}=\{(-1,0),(1,0)\}\text{ or }\{(0,-1),(0,1)\}\,(\text{axis by hash}),\; p_{\text{leave}}=0.10,\; T=6$
  - from: $\{\text{F4Nr3},\text{FSTACK}\}$ — 全向收敛为单轴 / 囤积转双向撒
- **FBURST**: 爆发-静息相位,短窗强爆其余蛰伏,博弈节奏。
  - $\text{dirs}=\text{4-nbr},\; p_{\text{leave}}=0.20,\; T=2;\;(T-\text{birth})\bmod 12<2\Rightarrow f=0.55,\text{ else }f=0.05$
  - from: $\{\text{F4Nr4},\text{FDRIFT}\}$ — 由强全向档获得相位调制(获得时间结构,非升量)

## P 池 — 突变 (rank 2, 12)

吐 p(x)(率,$\text{rate}=\min(p_{\max},\mu+p_{\text{add}})$,$\mu=0.01$,$p_{\max}=0.08$) + spectrum(变成谁,由家族距离 aff 0.70/0.25/0.05 算) + T。逐个体:Binom(后代,p_x) 个突变,各挑插槽+从谱抽字母覆写。堆叠 $p_x=\max(\mu,1-\prod(1-p_i))$。
> from(前驱谱):P 族 rank2,获得突变功能从 $N_{res}$ 或邻 rank 单步(F/Z 均 Δrank1)。谱偏置档(fscan/zscan/neutral_sink)前驱含对应目标族邻基元 = 谱倾向源于结构邻近。

- **P_base**: 中性基线,无加码纯 μ 漂变,谱全向。
  - $p_{\text{add}}=0,\; \text{rate}=\mu,\; q(t)\propto\text{aff}(\text{fam}(x),\text{fam}(t)),\; T=1$
  - from: $N_{res}\cup\{\text{P\_slow\_drift},\text{P\_entropy\_brake}\}$ — N 漂出最低突变档 / 同族档互变
- **P_hotspot**: AID 热点加码,中速,谱全向提通量。
  - $p_{\text{add}}=0.05,\; \text{rate}=0.06,\; q(t)\propto\text{aff}(\cdot),\; T=3$
  - from: $\{\text{P\_balanced},\text{P\_burst\_lite}\}$ — 由邻近 P 档升/侧变
- **P_aic**: AID-core 保守加码,谱锐化压同族近邻,定向不离族。
  - $p_{\text{add}}=0.03,\; \text{rate}=0.04,\; q(t)\propto\text{aff}(\cdot)^{2},\; T=2$
  - from: $\{\text{P\_base},\text{P\_balanced}\}$ — 由全向档锐化谱(获得定向性)
- **P_ep**: error-prone 宽谱,谱拉平向均匀,四散探索。
  - $p_{\text{add}}=0.04,\; \text{rate}=0.05,\; q(t)\propto\tfrac{1}{2}\text{aff}(\cdot)+\tfrac{1}{2}\tfrac{1}{|\mathcal{A}|-1},\; T=2$
  - from: $\{\text{P\_base},\text{P\_balanced}\}$ — 由全向档拉平谱(获得宽谱)
- **P_fscan**: 谱偏置投 F 族,加码小,拉向繁衍功能获得。
  - $p_{\text{add}}=0.02,\; \text{rate}=0.03,\; q(t)\propto\text{aff}(\cdot)\cdot\mathbb{1}[\text{fam}(t)=F],\; T=3$
  - from: $\{\text{P\_base}\}\cup\{s:\text{fam}(s)=F\}$ — 由基线获谱偏置 + F 族邻近(投 F 倾向源于结构邻 F)
- **P_zscan**: 谱偏置投 Z 族,加码小,拉向对抗功能获得。
  - $p_{\text{add}}=0.02,\; \text{rate}=0.03,\; q(t)\propto\text{aff}(\cdot)\cdot\mathbb{1}[\text{fam}(t)=Z],\; T=3$
  - from: $\{\text{P\_base}\}\cup\{s:\text{fam}(s)=Z\}$ — 由基线获谱偏置 + Z 族邻近
- **P_entropy_brake**: 熵刹车,谱高阶幂收窄(降多样性),加码极小。
  - $p_{\text{add}}=0.01,\; \text{rate}=0.02,\; q(t)\propto\text{aff}(\cdot)^{3},\; T=4$
  - from: $\{\text{P\_base},\text{P\_aic}\}$ — 由基线/保守档进一步收窄谱
- **P_loopswap_lite**: 短 motif 小跳,谱偏置相邻族(rank±1)穿浅谷。
  - $p_{\text{add}}=0.03,\; \text{rate}=0.04,\; q(t)\propto\text{aff}(\cdot)\cdot\mathbb{1}[|\Delta\text{rank}|=1],\; T=5$
  - from: $\{\text{P\_base},\text{P\_balanced}\}$ — 由全向档获相邻族跳跃偏置
- **P_neutral_sink**: 谱偏置投 N 族,功能位漂回中性(功能丧失)。
  - $p_{\text{add}}=0.02,\; \text{rate}=0.03,\; q(t)\propto\text{aff}(\cdot)\cdot\mathbb{1}[\text{fam}(t)=N],\; T=4$
  - from: $\{\text{P\_base}\}\cup N_{res}$ — 由基线获投 N 偏置 + N 族邻近(谱倾向 N = 邻 N)
- **P_slow_drift**: 慢钟全向基线变体,加码同 base 但触发极慢。
  - $p_{\text{add}}=0,\; \text{rate}=\mu,\; q(t)\propto\text{aff}(\cdot),\; T=6$
  - from: $N_{res}\cup\{\text{P\_base}\}$ — N 漂出慢钟档 / 基线改 T(同 rate 仅时间结构变)
- **P_burst_lite**: 周期热点脉冲,加码达本池上限,慢钟相位窗突变爆发。
  - $p_{\text{add}}=0.07,\; \text{rate}=0.08,\; q(t)\propto\text{aff}(\cdot),\; T=6$
  - from: $\{\text{P\_hotspot},\text{P\_balanced}\}$ — 由邻近 P 档获相位调制
- **P_balanced**: 中速标准加码全向,hotspot 与 base 间稳态工作档。
  - $p_{\text{add}}=0.04,\; \text{rate}=0.05,\; q(t)\propto\text{aff}(\cdot),\; T=4$
  - from: $\{\text{P\_base},\text{P\_aic},\text{P\_hotspot}\}$ — 同族 P 档互变

## Z 池 — 对抗 (rank 3, 16)

吐 z(交换比) + prey(猎物清单) + T。$\text{kills}=\min(b,a\cdot z)$,自损 $\text{kills}/z$,快照同时结算。高 z↔窄清单(反相关,防无敌)。多条非传递克制环。
> from(前驱谱):Z 族 rank3,**入口只经 P_zscan(rank2 邻)或 A 池 Z 变体(rank4 邻)**——N(Δrank3)/F(Δrank2)非邻不直达。通才(宽清单)由 P_zscan 变出,motif 专才由通才特化,长 motif/单字母狙击由邻近窄档再特化。**前驱认 z 档位/清单粒度的结构邻近。**

- **BroadSweep**: 通才,最低 z 最宽清单(两整族),克制链1基座。BB0 locked@pos5。
  - $z=0.40,\; \text{prey}=\{F,Z\},\; T=5$
  - from: $\{\text{P\_zscan}\}\cup\{\text{Scatter Nip},\text{Epitope Bleed}\}$ — 突变投 Z 获最弱通才 / 邻通才侧变(入口档)
- **Scatter Nip**: 通才打暴露中性,vis 链基座,命中按 vis 加权(高 vis 被剔)。
  - $z=0.40,\; \text{prey}=\{N\},\; p_{\text{hit}}=\tfrac1L\sum_{i\in N}vis_i,\; T=3$
  - from: $\{\text{P\_zscan}\}\cup\{\text{BroadSweep},\text{Ghost Spike}\}$ — 入口档 / 通才转 N 猎
- **Epitope Bleed**: 通才反战士/反突变,链1第二环(也猎 Z)。
  - $z=0.45,\; \text{prey}=\{Z,P\},\; T=6$
  - from: $\{\text{BroadSweep},\text{Attrition Bite}\}$ — 由基座升档 / 邻环侧变
- **Ghost Spike**: 隐位中性猎手,Scatter Nip 镜像,命中按 (1−vis) 加权。
  - $z=0.50,\; \text{prey}=\{N\},\; p_{\text{hit}}=\tfrac1L\sum_{i\in N}(1-vis_i),\; T=4$
  - from: $\{\text{Scatter Nip},\text{Frame Pincer}\}$ — N 猎档互变(vis 取向翻转)
- **Attrition Bite**: 整 Z 族猎手,链1第三环;自身为 Z 故被 Z-motif 专才克。
  - $z=0.55,\; \text{prey}=\{Z\},\; T=3$
  - from: $\{\text{Epitope Bleed},\text{Clade Snare}\}$ — 链1中继,清单收窄一阶
- **Hapten Graze**: 整 P 族猎手,对 P 重群多数派诅咒,稀少则饿。
  - $z=0.55,\; \text{prey}=\{P\},\; T=7$
  - from: $\{\text{Epitope Bleed},\text{Burst Leech}\}$ — 由含 P 通才收窄为整 P 族 / motif 专才放宽
- **Burst Leech**: P-motif 专才,猎带 P 的 motif(突变代价捕食者)。
  - $z=0.62,\; \text{prey}=\{\text{motif}\ni P\},\; T=8$
  - from: $\{\text{Hapten Graze},\text{Idiotype Lance}\}$ — 整 P 族特化为 P-motif / 长 P-motif 放宽
- **Ambush Coil**: F-motif 专才,链2第一环,猎繁衍 motif 株。
  - $z=0.65,\; \text{prey}=\{\text{motif}\ni F\},\; T=7$
  - from: $\{\text{BroadSweep},\text{Lineage Reaper}\}$ — 通才特化为 F-motif / 长 F-motif 放宽
- **Clade Snare**: Z-motif 专才,链1第三环,猎 Attrition 携带的 Z-motif。
  - $z=0.68,\; \text{prey}=\{\text{motif}\ni Z\},\; T=6$
  - from: $\{\text{Attrition Bite},\text{Coil Cinch}\}$ — 整 Z 族特化为 Z-motif / 长 Z-motif 放宽
- **Frame Pincer**: N-motif 专才,vis 链第三环,构象命中中性结构块。
  - $z=0.72,\; \text{prey}=\{\text{motif}\ni N\},\; T=5$
  - from: $\{\text{Ghost Spike},\text{Scatter Nip}\}$ — N 猎档特化为 N-motif 专才
- **Lineage Reaper**: 长 F-motif 专才,链2第二环,猎≥3字母繁衍 motif。
  - $z=0.78,\; \text{prey}=\{\ell\ge3\text{ motif}\ni F\},\; T=6$
  - from: $\{\text{Ambush Coil}\}$ — 由 F-motif 专才特化为长 motif(同 gran 邻档)
- **Coil Cinch**: 长 Z-motif 专才,链1第四环,≥3字母战士 motif。
  - $z=0.80,\; \text{prey}=\{\ell\ge3\text{ motif}\ni Z\},\; T=8$
  - from: $\{\text{Clade Snare}\}$ — Z-motif 专才升为长 motif
- **Idiotype Lance**: 长 P-motif 专才,≥3字母突变 motif,慢钟防失控。
  - $z=0.85,\; \text{prey}=\{\ell\ge3\text{ motif}\ni P\},\; T=9$
  - from: $\{\text{Burst Leech}\}$ — P-motif 专才升为长 motif
- **Crest Bite**: 单字母 F 狙击,链2第三环,猎高扩张繁衍基元(特征阈值)。
  - $z=0.90,\; \text{prey}=\{s:\text{fam}(s)=F \wedge f_s\ge0.5\},\; T=6$
  - from: $\{\text{Lineage Reaper}\}$ — 长 F-motif 特化为单字母 F 狙击(收窄到极)
- **Hotspot Snipe**: 单字母 P 狙击,猎高加码突变基元(特征阈值)。
  - $z=0.95,\; \text{prey}=\{s:\text{fam}(s)=P \wedge p_{\text{add},s}\ge0.05\},\; T=7$
  - from: $\{\text{Idiotype Lance}\}$ — 长 P-motif 特化为单字母 P 狙击
- **Mirror Fang**: 单字母 Z 狙击,链1收口,猎宽谱通才对抗基元(特征阈值)。
  - $z=1.00,\; \text{prey}=\{s:\text{fam}(s)=Z \wedge z_s\le0.45 \wedge |\text{prey}_s|\ge2\},\; T=8$
  - from: $\{\text{Coil Cinch}\}$ — 长 Z-motif 特化为单字母 Z 狙击(同 gran 邻档)

## A 池 — 突变可达极端变体 (rank 4, 24)

全部只产 f/p/z(无第五通道;功能只有 F/P/Z 三种)。全部突变专属(骨架坐不上,只能突变出)。极端区间: f≤0.85 / z≤1.5(窄清单守反相关) / p_add≤0.34(rate cap 0.35)。
> 覆写列表(2026-06-24 重订,替代逐条点名前驱):某 A 的覆写列表 = backbone 上下文门控谓词 $\{\text{株}:n_{\text{locked}}(\text{chan}(A))\ge\theta_{\text{chan}(A)}\}$——即只在 backbone-locked 组成里**该 A 所产通道对应族**已达数量阈 θ 的物种内,由一个 gran 匹配的可变插槽突变涌现。「建难」由门控自然涌现(极端 F 变体只在 F-重株里冒头,无需手写阶梯/禁 N)。同通道 A 共用同一覆写门,彼此差异全在各自 formula。θ 数值轮占位。**copy-of 仅血统注释(标该 A 照哪条常规档放大),不进覆写逻辑(突变核不读 copy-of)。**

### 乙1 — 8 个 FPZ 精选拷贝(escalation,带 copy-of 理由)

- **Apex Bloom**: 最强繁衍爆点,四方向全向扩张推到极限,源格高自损。[copy-of F4Nr4]
  - $f=0.85,\; \text{dirs}=\text{4-nbr},\; p_{\text{leave}}=0.20,\; T=4$
  - 覆写: $\{\text{株}:n_{\text{locked}}(F)\ge\theta_F\}$ — F-重 backbone 物种内由 gran 匹配插槽突变涌现
- **Ember Drip**: 废档脉冲,极低 f 单向慢钟几乎不扩张(下坠样本)。[copy-of F4Nr1]
  - $f=0.05,\; \text{dirs}=\{d_{\text{hash}}\}(\text{1 of 4-nbr}),\; p_{\text{leave}}=0.04,\; T=9$
  - 覆写: $\{\text{株}:n_{\text{locked}}(F)\ge\theta_F\}$ — F-重 backbone 物种内由 gran 匹配插槽突变涌现
- **Bastion Pile**: 极限囤积,原地零外扩高 f 垫满 count 死守名额。[copy-of FSTACK]
  - $f=0.85,\; \text{dirs}=\{(0,0)\},\; p_{\text{leave}}=0.00,\; T=3$
  - 覆写: $\{\text{株}:n_{\text{locked}}(F)\ge\theta_F\}$ — F-重 backbone 物种内由 gran 匹配插槽突变涌现
- **Apex Fang**: 单字母 Z 狙击 z 顶到极限,最窄清单换最高效率。[copy-of Mirror Fang]
  - $z=1.50,\; \text{prey}=\{s:\text{fam}(s)=Z\wedge z_s\le0.45\wedge|\text{prey}_s|\ge2\},\; T=9$
  - 覆写: $\{\text{株}:n_{\text{locked}}(Z)\ge\theta_Z\}$ — Z-重 backbone 物种内由 gran 匹配插槽突变涌现
- **Pan Sweep**: 三族通吃宽扫(F∪Z∪P),z 仍压低(宽谱代价=高自损)。[copy-of BroadSweep]
  - $z=0.50,\; \text{prey}=\{F,Z,P\},\; T=6$
  - 覆写: $\{\text{株}:n_{\text{locked}}(Z)\ge\theta_Z\}$ — Z-重 backbone 物种内由 gran 匹配插槽突变涌现
- **Hotspot Amp**: AID 热点加码推到上限,全向谱高通量。[copy-of P_hotspot]
  - $p_{\text{add}}=0.30,\; \text{rate}=0.31,\; q(t)\propto\text{aff}(\cdot),\; T=3$
  - 覆写: $\{\text{株}:n_{\text{locked}}(P)\ge\theta_P\}$ — P-重 backbone 物种内由 gran 匹配插槽突变涌现
- **Sink Cascade**: 高率功能丧失,谱锁投 N 族快速漂回中性(率强产出垃圾)。[copy-of P_neutral_sink]
  - $p_{\text{add}}=0.25,\; \text{rate}=0.26,\; q(t)\propto\text{aff}(\cdot)\cdot\mathbb{1}[\text{fam}(t)=N],\; T=3$
  - 覆写: $\{\text{株}:n_{\text{locked}}(P)\ge\theta_P\}$ — P-重 backbone 物种内由 gran 匹配插槽突变涌现
- **Glacial Drift**: 极慢全向漂变,零加码 + 最慢钟近不动(废档慢端)。[copy-of P_slow_drift]
  - $p_{\text{add}}=0,\; \text{rate}=\mu,\; q(t)\propto\text{aff}(\cdot),\; T=12$
  - 覆写: $\{\text{株}:n_{\text{locked}}(P)\ge\theta_P\}$ — P-重 backbone 物种内由 gran 匹配插槽突变涌现

### 乙2 — 8 个 A 原生极端变体

- **F_NOVA**: 极端强爆-长静息,爆窗一次性向四邻投 0.85 但 50% 自迁出露背。
  - $\text{dirs}=\text{4-nbr},\; p_{\text{leave}}=0.50,\; T=2;\;(T-\text{birth})\bmod 20<1\Rightarrow f=0.85,\text{ else }0.05$
  - 覆写: $\{\text{株}:n_{\text{locked}}(F)\ge\theta_F\}$ — F-重 backbone 物种内由 gran 匹配插槽突变涌现
- **F_TRICKLE**: 绝对地板繁衍,f=0.02 单向慢钟效率近零(阶梯最底)。
  - $f=0.02,\; \text{dirs}=\{d_{\text{hash}}\}(\text{1 of 4-nbr}),\; p_{\text{leave}}=0.02,\; T=8$
  - 覆写: $\{\text{株}:n_{\text{locked}}(F)\ge\theta_F\}$ — F-重 backbone 物种内由 gran 匹配插槽突变涌现
- **F_SCATTER**: 奇异超薄云团,低 f 随机 3 格 + 最高迁出 0.60 清空源格成移动飞地。
  - $f=0.12,\; \text{dirs}=\{\text{3 of 4-nbr by hash}\},\; p_{\text{leave}}=0.60,\; T=3$
  - 覆写: $\{\text{株}:n_{\text{locked}}(F)\ge\theta_F\}$ — F-重 backbone 物种内由 gran 匹配插槽突变涌现
- **Predator Lock**: 长 Z-motif 专杀的杀手,z=1.45 守最窄长 motif(克制链收口)。
  - $z=1.45,\; \text{prey}=\{\ell\ge3\text{ motif}\ni Z\},\; T=9$
  - 覆写: $\{\text{株}:n_{\text{locked}}(Z)\ge\theta_Z\}$ — Z-重 backbone 物种内由 gran 匹配插槽突变涌现
- **Void Bite**: 破隐身流,只瞄低 vis 中性,命中按 (1−vis) 加权,专罚盲区策略。
  - $z=0.95,\; \text{prey}=\{s:\text{fam}(s)=N\wedge vis_s\le0.20\},\; p_{\text{hit}}=\tfrac1L\sum_i(1-vis_i),\; T=5$
  - 覆写: $\{\text{株}:n_{\text{locked}}(Z)\ge\theta_Z\}$ — Z-重 backbone 物种内由 gran 匹配插槽突变涌现
- **P_cascade**: 双位点级联,单次事件覆写 2 插槽(SHM 连发),高率全向。
  - $p_{\text{add}}=0.28,\; \text{rate}=0.29,\; q(t)\propto\text{aff}(\cdot),\; \text{2 slots/event},\; T=2$
  - 覆写: $\{\text{株}:n_{\text{locked}}(P)\ge\theta_P\}$ — P-重 backbone 物种内由 gran 匹配插槽突变涌现
- **P_crossclan_surge**: 跨族大跳,谱只投远族(|Δrank|≥2,亲和 0.05 罕跳)穿鞍点。
  - $p_{\text{add}}=0.20,\; \text{rate}=0.21,\; q(t)\propto\text{aff}(\cdot)\cdot\mathbb{1}[|\Delta\text{rank}|\ge2],\; T=4$
  - 覆写: $\{\text{株}:n_{\text{locked}}(P)\ge\theta_P\}$ — P-重 backbone 物种内由 gran 匹配插槽突变涌现
- **P_frozen**: 近锁突变,零加码 + 谱高阶幂锐化锁同族(几乎只自环)。
  - $p_{\text{add}}=0,\; \text{rate}=\mu,\; q(t)\propto\text{aff}(\cdot)^{4},\; T=8$
  - 覆写: $\{\text{株}:n_{\text{locked}}(P)\ge\theta_P\}$ — P-重 backbone 物种内由 gran 匹配插槽突变涌现

### 甲 — 8 个 A 原生极端变体(补满 24,均产 f/p/z)

- **F8Ar1**: 最快钟搅动机,单随机格中 f 但 T2 极速(churn 引擎持续薄撒)。
  - $f=0.25,\; \text{dirs}=\{\text{1 rand of 4-nbr/tick}\},\; p_{\text{leave}}=0.10,\; T=2$
  - 覆写: $\{\text{株}:n_{\text{locked}}(F)\ge\theta_F\}$ — F-重 backbone 物种内由 gran 匹配插槽突变涌现
- **Lance Front**: 极限楔形锋线,单固定方向(hash 锁)f=0.80,高迁出露背。
  - $f=0.80,\; \text{dirs}=\{d_{\text{hash}}\}(\text{1 fixed}),\; p_{\text{leave}}=0.30,\; T=4$
  - 覆写: $\{\text{株}:n_{\text{locked}}(F)\ge\theta_F\}$ — F-重 backbone 物种内由 gran 匹配插槽突变涌现
- **Ambush Venom**: F-motif 专杀强档,z=1.30 守最窄繁衍 motif 清单。
  - $z=1.30,\; \text{prey}=\{\text{motif}\ni F\},\; T=7$
  - 覆写: $\{\text{株}:n_{\text{locked}}(Z)\ge\theta_Z\}$ — Z-重 backbone 物种内由 gran 匹配插槽突变涌现
- **Sweep Surge**: 极端低-z 快钟通才,猎 F∪P 但 z 仅 0.45(宽=高自损,守反相关)。
  - $z=0.45,\; \text{prey}=\{F,P\},\; T=3$
  - 覆写: $\{\text{株}:n_{\text{locked}}(Z)\ge\theta_Z\}$ — Z-重 backbone 物种内由 gran 匹配插槽突变涌现
- **Nip Whisper**: 废档对抗,z=0.15 几乎不划算,瞄暴露中性按 vis 加权(下坠端)。
  - $z=0.15,\; \text{prey}=\{N\},\; p_{\text{hit}}=\tfrac1L\sum_{i\in N}vis_i,\; T=3$
  - 覆写: $\{\text{株}:n_{\text{locked}}(Z)\ge\theta_Z\}$ — Z-重 backbone 物种内由 gran 匹配插槽突变涌现
- **Coil Null**: 废档 Z 猎手,z=0.20 慢钟瞄整 Z 族,猎物稀少即自损饿死。
  - $z=0.20,\; \text{prey}=\{Z\},\; T=8$
  - 覆写: $\{\text{株}:n_{\text{locked}}(Z)\ge\theta_Z\}$ — Z-重 backbone 物种内由 gran 匹配插槽突变涌现
- **P_zscan_invert**: 攻击转扩张,谱反投 F 族(获得对抗→获得繁衍),中档加码。
  - $p_{\text{add}}=0.10,\; \text{rate}=0.11,\; q(t)\propto\text{aff}(\cdot)\cdot\mathbb{1}[\text{fam}(t)=F],\; T=4$
  - 覆写: $\{\text{株}:n_{\text{locked}}(P)\ge\theta_P\}$ — P-重 backbone 物种内由 gran 匹配插槽突变涌现
- **P_stutter**: 伪突变废档,率推到上限但谱高阶幂锐化近自环(高率零产出)。
  - $p_{\text{add}}=0.32,\; \text{rate}=0.33,\; q(t)\propto\text{aff}(\cdot)^{4},\; T=2$
  - 覆写: $\{\text{株}:n_{\text{locked}}(P)\ge\theta_P\}$ — P-重 backbone 物种内由 gran 匹配插槽突变涌现

## 覆写列表 (overwrite/predecessor list)

✅ 两种形态(2026-06-24 重订):

- **A 池(24)** = backbone 上下文门控谓词 $\{\text{株}:n_{\text{locked}}(\text{chan})\ge\theta_{\text{chan}}\}$,inline 在每条 A 的 `覆写:` 行。同通道共用同一门,差异在各自 formula。「建难」由门控涌现(极端变体只在该通道已专化的物种内冒头),非手写阶梯。
- **N/F/P/Z 池** 仍保留各自 `from:` 行(中性互漂 / 功能获得方向),为后续设计的参考线,非红线判据。

**废弃**(均非用户设计的概念,2026-06-24 删):逐条点名前驱当红线、「三结构律(粒度/阶梯/非对称)」、「破易建难非对称律」、对称性。粒度配对(residue↔residue / motif↔motif)是突变核机制本身,不需另立为律。
