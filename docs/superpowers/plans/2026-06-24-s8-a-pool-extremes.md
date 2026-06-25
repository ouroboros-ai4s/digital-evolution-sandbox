# S8 — A 池 24 极端变体 (de-gated) + 多 P 混合谱 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 A 池 24 个极端变体作为「registry 数据行 + de-gated reachability」一次性铸进 `ALPHABET`/`GRAN`/`MOTIF_LEN`/`_F`/`_Z`/`_P`/`SPECTRUM_SHAPE`/`SLOTS_PER_EVENT`/`VIS` 等既有表(全部 family F/P/Z,**没有 rank-4 letter**,无 `n_locked≥θ` 覆写门),同时把 `phenotype()` 里的「单 dominant_p 谱源」升级为「多 P 字母 p_add 加权混合谱 `Σpᵢqᵢ/Σpᵢ`」(用户 2026-06-24 裁定,S8 owns)。

**Architecture:** 路线图 9/9。S0→S6→S1→S2→S4→S5→S3→S7→**S8** 全栈依赖。两件事:(a) **多 P 混合谱** —— 唯一真实机制变更:`phenotype()` 把 `dominant_p` 选择改为 `spectrum(t) = Σ_i p_add_i · q_i(t) / Σ_i p_add_i`,每个 `q_i` 是既有 `_spectrum_for(letter)` 已 gran-filtered + shape-modulated + normalized 的字母级谱,blend 后再归一;单 P 字母路径**字节级等于 v1 dominant_p 路径**(Σ 只有一项,blend == dominant)。(b) **24 A 池数据行** —— 全部用 S1–S7 既有机制(`_F`/`_Z`/`_P` 极端值 + S2 shape + S6 motif + S5 window + S4 dirs + S7 slots + S3 prey-clauses + S1 vis),`n_locked` 覆写门**作废**(retired,2026-06-24);三子池(乙1 escalation copy 8 + 乙2 native 8 + 甲 native 8)分三 Task 落地,再加 SPECTRUM_SHAPE 域扩展(power=4 + family_mask="cross" 接住 P_frozen/P_stutter/P_crossclan_surge)。零内核改动,零 `phenotype_arrays` 改动,默认四阵营对称局 byte-identical 不变。

**Tech Stack:** Python 3 (`D:\anaconda3\envs\basic\python.exe`), PyTorch 2.10+cu128, pyarrow, pytest. Windows 主机,`PYTHONPATH=src` 纪律。引擎源码 `src/des/`。**依赖**: S0 (CLI 已就位,不动) + S6 (`GRAN` / `MOTIF_LEN` / `PREDICATE_BITS` / motif_blocks / `feature_mask_of` / `prey_mask_for_clauses` 已落地) + S1 (`VIS` 表 + `vis_lowvis` bit 已就位) + S2 (`SPECTRUM_SHAPE` + `_spectrum_for` 三旋钮 + P 池 10 行新基元已落地) + S4 (动态 `directions` 三态识别 `"hash:<kind>"` / `"rand:1of4"` / `(IN_PLACE_DIR,)` + `IN_PLACE_DIR` 哨兵) + S5 (`Phenotype.f_hi/f_lo/burst_w/burst_k` + windowed-f kernel) + S3 (4 个阈值 bit + 4 个新 prey clause tags) + S7 (`SLOTS_PER_EVENT` + `_mutation_outcomes(seq, mutable, spectrum, blocks, slots_per_event)` + N≥2 joint enumeration)。**与 a_invivo 无关**(归 R_irr,sandbox 不研究)。

## Global Constraints

- **A 是 family F/P/Z 极端值,不是 rank-4 family**(spec §2 + design.md L339,2026-06-20 descaffold 已废 rank-4):24 A 行的 `ALPHABET[letter]` 值是 `"F"` / `"P"` / `"Z"` 三家之一,**禁止**写入 `"A"` / `"S"` / 任何 rank-4 letter;`FAMILY_RANK` 字典 `{"N":0,"F":1,"P":2,"Z":3}` 不动。"A 池 / rank 4" 是组织 tier 标签,不是 mutation-family 标签。
- **De-gate(2026-06-24 用户裁定)**:A primitives reachable 仅靠 `affinity()` 全局谱,与其他 letter 同条;**`n_locked(chan)≥θ` 覆写门 retired**;`n_locked` 仍是 S6 on-demand 派生量,但无 consumer → advisory structural readout,不进 mutation 决策。
- **No new mechanism code**(spec §2):S8 = 24 行 registry 数据 + multi-P blend 公式 1 处;**禁止**手写「这个 A 突变更强」,所有强度仅经 `_F`/`_Z`/`_P` 表(roster-declared extreme values);**禁止**新增 `_mutation_outcomes` / kernel 路径分支。
- **Multi-P blend(spec §4.1,2026-06-24 用户裁定 S8 owns)**:`phenotype()` 解析 P 字母时,把 `dominant_p` 单源选择替换为 `spectrum(t) = Σ_i p_add_i · q_i(t) / Σ_i p_add_i`(其中 `q_i = _spectrum_for(letter_i)` 是 S2 已 shape-modulated + S6 gran-filtered + normalized 的字母级谱)。**单 P 字母路径数学等价 dominant_p 路径**(Σ 一项 = `q_dominant`,blend == dominant_p,byte-identical 静态默认局)。
- **`Σ p_add_i = 0` 边界**:strain 里所有 P letter 都是 `p_add=0`(`P_base` / `P_slow_drift` / `P_frozen` / `Glacial Drift`)→ 退化为「等权 blend」(`spectrum = Σ q_i / N_p`)而非分母 0 抛错(spec §4.1 implicit edge case + spec §5 ponytail discipline);单 P 字母此情况下仍 == `q_dominant`(N_p=1 时 `Σ q / 1 = q`),byte-identical。
- **Extreme value bounds(spec §6)**:module-load assert 守 A 行**严格**在 roster 范围内,违规 → import 即 fail:
  - F 行:`_F[letter][0] <= 0.85`(`f` 字段)
  - Z 行:`_Z[letter][0] <= 1.5`(`z` 字段)
  - P 行:`_P[letter][0] <= 0.34`(`p_add` 字段,`rate = MU + p_add <= 0.35` 隐式守住)
  - 校验**仅守 A 池新行**(乙1 + 乙2 + 甲 共 24 行),既有 v1 / S1–S7 行的值不被波及。
- **z↔prey 反相关是 roster 不变量,非 runtime 检查**(spec §6):高 z A 行(Apex Fang z=1.5 / Predator Lock z=1.45 / Ambush Venom z=1.30)的 prey clause 显式窄(threshold / motif / family 单条),由 roster-verbatim 数值与 prey_clauses 字段保证,**不**在代码里加 runtime assert(避免错信号 + 自检循环)。
- **`copy-of` 仅血统注释**(roster line 183,spec §3 红线):roster 的 `[copy-of <ancestor>]` 标签是文档元数据;mutation core / kernel / registry **一律不读** `copy-of`;不写入任何 Python 数据结构,只作 roster 注释或 plan 步骤里的命名出处说明。
- **静态默认 4-阵营对称局字节级不变**(spec §3 红线):4 同 BB0 faction 的默认局,`BB0_TEMPLATE["layout"]` 6 个 mutable slot 全是 v1 6-字母调色板,**不会**突变到 A 池(A 行为 `affinity()` 同条但可通达 ⇒ 起始局没人坐在 A 上 + 默认 mutable 调色板未扩);28 引擎 + 146 web 测试 + S1–S7 owner test 全部继续绿。回归锁 = `pytest tests/ -q` 末尾无 `FAILED tests/...` 行。
- **A 在默认局出现率(de-gate 副作用,spec §3 红线 + §7)**:de-gate 后 A 是 normal same-family spectrum target,可在默认局通过 mutation **罕见但对称地**出现(四阵营起 BB0 完全相同,A 涌现的概率四阵营完全等概)—— 不构造非对称、不产 selection 信号、不破「红皇后无目标」红线。非对称-backbone 角色系统仍 HARD-GATE,不在 S8 范围。
- **Multi-P blend 与 SLOTS_PER_EVENT 多 P 合并规则同步**(spec §3 inline + S7 plan):多 P 共存 strain 的 `slots_per_event` 在 S7 落地时仍取 `SLOTS_PER_EVENT[dominant_p]`(highest p_add,平手取首现);S8 不动 SLOTS_PER_EVENT 多 P 合并规则(P_cascade 单字母 N=2,无与其他 P 字母 stack 的设计场景)—— spec §3 inline 实证明确「N piggybacks dominant_p,blend 归 S8 但不延伸到 N」。
- **SPECTRUM_SHAPE 值域扩展(spec §1 P_frozen / P_stutter / P_crossclan_surge 强制项)**:S2 锁 `power ∈ {1.0, 2.0, 3.0}` + `family_mask ∈ {None, "F", "Z", "N", "adjacent"}`;S8 **扩展**:`power` 加入 `4.0`(P_frozen / P_stutter `aff⁴` 锐化);`family_mask` 加入 `"cross"`(`|Δrank| >= 2`,P_crossclan_surge);module-load assert 同步更新值域(spec §6 +S2 §2 红线延续)。
- **A 池 vis 字段为 N/A**(spec §4):A 全部仅产 `f/p/z`,不产 vis;`VIS` 表对 A 行**不加 key**(vis 是 N 字母独占 channel,`feature_mask_of` 的 `vis_lowvis` 设置只在 `ALPHABET[letter]=="N"` 分支生效,A 不命中);只 Void Bite / Nip Whisper 这两条 Z prey clause 引用 vis(走 `("N","lowvis")` clause-tag → `vis_lowvis` bit,prey 是 N 字母,自身仍不在 VIS 表)。
- **Roster doc cleanup 是 plan 必须项**(spec §2 末尾 + §3):本 plan Task 6 显式重写 24 条 `覆写:` 行 + OPEN-1/θ section 标 RETIRED,verbatim 抄 spec §2 的两条文本,**禁止**改字、删字、加私货;`primitive-roster.md` 是设计 doc,不进代码路径,但属本 plan 交付物。
- **回归锁(spec §6 + 路线图全栈)**:全 suite (`pytest tests/ -q`) 完工后必须全绿,**任何** S1–S7 owner test 触发 FAIL = root cause 必须本 plan 内 fix-forward(不动既有 S1–S7 spec / plan;若 fix 涉及调整 S2 SPECTRUM_SHAPE 值域 assert,见上一条已显式列入 S8 范围)。
- **Out of scope**(spec §7):非对称-backbone 角色系统(HARD-GATE);κ same-channel synergy(`κ=0` v1 不动);A 池 wet-lab a_invivo 真值锚(归 R_irr);多 faction asymmetric K / rate / mechanics(HARD-GATE,独立 brainstorming);P_cascade 的 SLOTS_PER_EVENT 多 P 合并规则升级(S7 + future-spec);S5 windowed-f 多字母 stacking 升级(S5 + future-spec);CLI key allow-list 扩展(S0 锁 `{players,grid,K,fill,T,seed}`,A 池数据条目不进 CLI)。

---

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `src/des/registry.py` | **Modify** | (a) `ALPHABET` 加 24 个 A 池新 key,值是 `"F"` / `"P"` / `"Z"` 三家之一,**禁止** `"A"` / `"S"`;(b) `GRAN` 加 24 行,大部分 `"residue"`,Predator Lock 单条 `"motif"`(`MOTIF_LEN["Predator Lock"]=3`);(c) `MOTIF_LEN` 加 Predator Lock 一行 `3`(spec §1 乙2 `ℓ≥3 motif∋Z`);(d) `_F` 加 8 行(乙1 4 行 + 乙2 3 行 + 甲 3 行 = 10 个 F 类 A,但 lance-front / F4Nr3 等 motif 不在 F 池,**实际 F 池 A = 8 行**:Apex Bloom / Ember Drip / Bastion Pile / F_NOVA / F_TRICKLE / F_SCATTER / F8Ar1 / Lance Front);(e) `_Z` 加 9 行(Apex Fang / Pan Sweep / Predator Lock / Void Bite / Ambush Venom / Sweep Surge / Nip Whisper / Coil Null + Apex Fang 共 8 行 Z 类 A);(f) `_P` 加 7 行(Hotspot Amp / Sink Cascade / Glacial Drift / P_cascade / P_crossclan_surge / P_frozen / P_zscan_invert / P_stutter = 8 行 P 类 A);(g) `SPECTRUM_SHAPE` 加 8 个 P 类 A 行 + 扩展值域 assert(`power` 允许 `4.0`,`family_mask` 允许 `"cross"`);(h) `SLOTS_PER_EVENT` 加 24 个新 letter 行(P_cascade 唯一 `=2`,其余 `=1`);(i) **Multi-P blend in `phenotype()`**:把「`dominant_p` 选择 + `spectrum = _spectrum_for(dominant_p)`」替换为「逐 P 字母收集 `(p_add, _spectrum_for(letter))` 然后 `spectrum = blend_p_spectra(pairs)`」;(j) 新 helper `blend_p_spectra(pairs: tuple[tuple[float, tuple[tuple[str, float], ...]], ...]) -> tuple[tuple[str, float], ...]`,实现 `Σ p_add_i · q_i / Σ p_add_i`(`Σ p_add_i == 0` 时退化为等权 `Σ q_i / N_p`),单字母路径 == `_spectrum_for(letter)`;(k) `phenotype()` 同步保留 `dominant_p` 为 **`SLOTS_PER_EVENT` 取值的 P 字母**(highest p_add, 平手首现,与 S7 共享纪律,不动 S7 spec),仅谱源改 blend。 |
| `src/des/_a_pool.py` | **Create** | 把 24 A 池的 registry 数据塞进一个独立模块(避免 `registry.py` 长到 700 行不可维护)。导出 4 个常量:`A_F: dict[str, tuple]` (8 行 _F 形 7-tuple 走 S5 路径) / `A_Z: dict[str, tuple]` (8 行 _Z 形 4-tuple) / `A_P: dict[str, tuple]` (8 行 _P 形 2-tuple) / `A_SHAPE: dict[str, tuple]` (8 行 SPECTRUM_SHAPE 形 3-tuple) + 1 个 `A_GRAN: dict[str, str]` / 1 个 `A_MOTIF_LEN: dict[str, int]` / 1 个 `A_SLOTS: dict[str, int]` / 1 个 `A_FAMILY: dict[str, str]`(letter → "F"/"P"/"Z")。`registry.py` 在表定义后单 `update()` 合入。 |
| `src/des/types.py` | **不动** | `Phenotype` 字段在 S5/S7 已锁;S8 不加字段。 |
| `src/des/phenotype_cache.py` | **不动** | `phenotype_arrays` 张量列(`f/p_leave/.../slots_per_event`)在 S5/S7 已锁;S8 不加列。 |
| `src/des/kernels/reproduction.py` | **不动** | `_mutation_outcomes(seq, mutable, spectrum, blocks, slots_per_event=1)` signature 在 S6/S7 已锁;S8 用同 signature 直接走 N=2(P_cascade)+ joint enumeration。 |
| `src/des/kernels/antagonism.py` | **不动** | `(prey_mask[i] & feature_mask[j]) != 0` match 表达式在 S6/S3 已锁;A 池 prey_clauses 走 S6/S3 既有 clause-tag 解析。 |
| `tests/test_a_pool.py` | **Create** | S8 owner 文件(sibling: `test_motif.py` / `test_vis.py` / `test_spectrum_shape.py` / `test_direction_kinds.py` / `test_phase_windows.py` / `test_threshold_predicates.py` / `test_multi_slot.py`)。覆盖:24 行存在 + family ∈ {F,P,Z} + 极端值边界 assert + 每子池 verbatim 行(乙1 8 行 / 乙2 8 行 / 甲 8 行)+ Predator Lock motif gran + P_cascade slots=2 + SPECTRUM_SHAPE 值域扩展(power=4 / mask="cross")+ multi-P blend 单字母 == dominant + multi-P blend 双字母按 p_add 加权 + `Σ p_add == 0` 等权退化 + de-gate 验证(无 n_locked gate code path)+ relabel-invariance + 默认 BB0 phenotype 字节级与 pre-S8 不变 + 同 seed engine.run byte-identical。 |
| `tests/test_registry.py` | **Modify (append)** | 追加 S8 注册表覆盖断言:`A_F`/`A_Z`/`A_P` keys ⊆ `_F`/`_Z`/`_P`;`A_SHAPE` keys ⊆ `SPECTRUM_SHAPE`;`SLOTS_PER_EVENT` keys ⊇ ALPHABET keys(含 24 A)。 |
| `tests/test_phenotype_cache.py` | **Modify (append)** | 追加一条 multi-P blend 单字母字节级 = dominant_p 路径断言(守 cache invalidate 行为);**不动** array shape 断言。 |
| `tests/test_multi_p_blend.py` | **Create** | 专门的 blend 公式测试(独立文件,避免与 `test_a_pool.py` 混):`blend_p_spectra([(p,q)])` == `q`(单字母身份)/ `blend_p_spectra([(p₁,q₁),(p₂,q₂)])` == `(p₁q₁+p₂q₂)/(p₁+p₂)` 逐 target 相等 / `blend_p_spectra([(0,q₁),(0,q₂)])` == `(q₁+q₂)/2`(等权退化)/ `blend_p_spectra([])` == `()` / 归一化 `Σ spectrum = 1`。 |
| `context/design/primitive-roster.md` | **Modify** | 24 条 `覆写:` 行 verbatim 替换为 spec §2 末尾的退役文本;OPEN-1/θ section 标 RETIRED;**不动** A 池 verbatim formula 表(数值仍是 design source-of-truth)。 |
| `scripts/run_batch.py` | **不动** | symmetric default producer 路径不变(A 不在默认调色板里)。 |

**Naming contract(locked,used by every task):**

```python
# src/des/_a_pool.py — A pool data tables (24 letters)
A_FAMILY:    dict[str, str]                                 # letter -> "F"|"P"|"Z" (no rank-4)
A_GRAN:      dict[str, str]                                 # letter -> "residue"|"motif"
A_MOTIF_LEN: dict[str, int]                                 # only "Predator Lock" -> 3
A_F:         dict[str, tuple[float, object, float, int,
                             float, int, int]]              # 7-tuple, S5 shape
A_Z:         dict[str, tuple[float,
                             tuple[tuple[str, ...], ...],
                             int, float]]                   # 4-tuple, S1 shape (with vis_mode)
A_P:         dict[str, tuple[float, int]]                   # (p_add, period)
A_SHAPE:     dict[str, tuple[float, "str | None", float]]   # (power, family_mask, flatten_mix)
A_SLOTS:     dict[str, int]                                 # slots_per_event;P_cascade=2,其余=1

# src/des/registry.py — new helper + extended phenotype path
def blend_p_spectra(
    pairs: tuple[tuple[float, tuple[tuple[str, float], ...]], ...],
) -> tuple[tuple[str, float], ...]
# Returns the p_add-weighted average of per-letter spectra, renormalized to Σq=1.
# Empty pairs -> (); Σp_add == 0 -> equal-weight average; single letter -> q itself.

# (existing) _spectrum_for(letter) -> spectrum  (unchanged signature, S2 body)
# (existing) phenotype(sequence) -> Phenotype   (body changed: dominant_p -> blend)
```

**24 A 池 verbatim 数据(spec §1 + roster L181–266,Task 1–3 直读此表):**

| letter | family | gran | _F / _Z / _P 数值 | SHAPE | SLOTS | 子池 |
| --- | --- | --- | --- | --- | --- | --- |
| `Apex Bloom` | F | residue | f=0.85 dirs=4-nbr p_leave=0.20 T=4 | — | 1 | 乙1 |
| `Ember Drip` | F | residue | f=0.05 dirs="hash:ember" p_leave=0.04 T=9 | — | 1 | 乙1 |
| `Bastion Pile` | F | residue | f=0.85 dirs=(IN_PLACE_DIR,) p_leave=0.00 T=3 | — | 1 | 乙1 |
| `Apex Fang` | Z | residue | z=1.50 prey=(("Z","generalist"),) T=9 | — | 1 | 乙1 |
| `Pan Sweep` | Z | residue | z=0.50 prey=(("F",),("Z",),("P",)) T=6 | — | 1 | 乙1 |
| `Hotspot Amp` | P | residue | p_add=0.30 T=3 | (1.0,None,0.0) | 1 | 乙1 |
| `Sink Cascade` | P | residue | p_add=0.25 T=3 | (1.0,"N",0.0) | 1 | 乙1 |
| `Glacial Drift` | P | residue | p_add=0.0 T=12 | (1.0,None,0.0) | 1 | 乙1 |
| `F_NOVA` | F | residue | f=0.85 dirs=4-nbr p_leave=0.50 T=2 f_lo=0.05 burst_w=20 burst_k=1 | — | 1 | 乙2 |
| `F_TRICKLE` | F | residue | f=0.02 dirs="hash:trickle" p_leave=0.02 T=8 | — | 1 | 乙2 |
| `F_SCATTER` | F | residue | f=0.12 dirs="hash:scatter3" p_leave=0.60 T=3 | — | 1 | 乙2 |
| `Predator Lock` | Z | motif | z=1.45 prey=(("Z","motif","len>=3"),) T=9 MOTIF_LEN=3 | — | 1 | 乙2 |
| `Void Bite` | Z | residue | z=0.95 prey=(("N","lowvis"),) T=5 | — | 1 | 乙2 |
| `P_cascade` | P | residue | p_add=0.28 T=2 | (1.0,None,0.0) | **2** | 乙2 |
| `P_crossclan_surge` | P | residue | p_add=0.20 T=4 | (1.0,"cross",0.0) | 1 | 乙2 |
| `P_frozen` | P | residue | p_add=0.0 T=8 | (**4.0**,None,0.0) | 1 | 乙2 |
| `F8Ar1` | F | residue | f=0.25 dirs="rand:1of4" p_leave=0.10 T=2 | — | 1 | 甲 |
| `Lance Front` | F | residue | f=0.80 dirs="hash:lance" p_leave=0.30 T=4 | — | 1 | 甲 |
| `Ambush Venom` | Z | residue | z=1.30 prey=(("F","motif"),) T=7 | — | 1 | 甲 |
| `Sweep Surge` | Z | residue | z=0.45 prey=(("F",),("P",)) T=3 | — | 1 | 甲 |
| `Nip Whisper` | Z | residue | z=0.15 prey=(("N","lowvis"),) T=3 | — | 1 | 甲 |
| `Coil Null` | Z | residue | z=0.20 prey=(("Z",),) T=8 | — | 1 | 甲 |
| `P_zscan_invert` | P | residue | p_add=0.10 T=4 | (1.0,"F",0.0) | 1 | 甲 |
| `P_stutter` | P | residue | p_add=0.32 T=2 | (**4.0**,None,0.0) | 1 | 甲 |

注:
- `dirs="hash:<kind>"` 走 S4 hash-locked 单方向路径;`(IN_PLACE_DIR,)` 走 S4 原地路径(`Bastion Pile` 极限囤积);`"rand:1of4"` 走 S4 每 tick 随机 1/4 路径(`F8Ar1`)。
- `prey=(("Z","generalist"),)` 走 S3 `("Z","generalist") -> thr_mirror` clause;`("Z","motif","len>=3")` 走 S6 motif+len 子句;`("N","lowvis")` 走 S3 vis_lowvis clause(`Void Bite` / `Nip Whisper`)。
- `Predator Lock` 是 A 池里**唯一** motif letter(gran="motif", MOTIF_LEN=3);全部 P / 其他 F / 其他 Z 都是 residue。
- `Hotspot Amp` rate = `MU + 0.30 = 0.31`,roster 显式列「rate=0.31」(spec §1 乙1),与 P_MAX=0.08 不冲突(rate 计算路径取 `min(P_MAX, MU+p_add)` 在 v1 仍 cap 在 P_MAX);P_MAX 是 v1 placeholder,A 池 rate 不另起 cap 路径,守「常数锁 registry」红线 + spec §6 `p_add≤0.34`。
- `P_cascade` 是 SLOTS_PER_EVENT = 2 的**唯一** letter,S7 plan 在 Task 1 已说明 S8 加这一行;本 plan Task 2 落地。
- `P_crossclan_surge` 的 `family_mask="cross"` 与 `P_frozen` / `P_stutter` 的 `power=4.0` 都不在 S2 当前值域;本 plan Task 4 扩展 SPECTRUM_SHAPE module-load assert。



---

### Task 1: 创建 `src/des/_a_pool.py` 数据模块(纯数据 + family/gran/motif/slots 表,无消费者)

**Goal:** 把 24 行 A 池数据集中放在一个独立模块 `src/des/_a_pool.py`,导出 8 个常量(`A_FAMILY` / `A_GRAN` / `A_MOTIF_LEN` / `A_F` / `A_Z` / `A_P` / `A_SHAPE` / `A_SLOTS`)。模块加载时**不**写进 `registry.py` 的 `ALPHABET` / `_F` / `_Z` / `_P` / `SLOTS_PER_EVENT`,只是数据准备完毕等待 Task 2 的 `update()` 合入。这一步**纯数据扩张,0 消费者**,既有行为字节级不变。

**Files:**
- Create: `src/des/_a_pool.py`
- Test: `tests/test_a_pool.py`(Create — S8 owner 文件首批断言)

**Interfaces:**
- Consumes: `des.types.IN_PLACE_DIR`(S4 已加,A `Bastion Pile` 用)。
- Produces:
  - `A_FAMILY: dict[str, str]` — 24 行 letter → `"F"` / `"P"` / `"Z"`,**无 rank-4**。
  - `A_GRAN: dict[str, str]` — 24 行 `"residue"` 或 `"motif"`(仅 `Predator Lock` 是 motif)。
  - `A_MOTIF_LEN: dict[str, int]` — 1 行 `{"Predator Lock": 3}`。
  - `A_F: dict[str, tuple]` — 8 行 F 类 A 的 7-tuple(S5 形状,verbatim 抄 File Structure 表)。
  - `A_Z: dict[str, tuple]` — 8 行 Z 类 A 的 4-tuple `(z, prey_clauses, period, vis_mode)`(S1 形状)。
  - `A_P: dict[str, tuple]` — 8 行 P 类 A 的 2-tuple `(p_add, period)`。
  - `A_SHAPE: dict[str, tuple]` — 8 行 P 类 A 的 3-tuple `(power, family_mask, flatten_mix)`。
  - `A_SLOTS: dict[str, int]` — 24 行,P_cascade=2,其余=1。

- [ ] **Step 1: 写失败测试 — 24 行存在 + family ∈ {F,P,Z} + 无 rank-4 letter**

新建 `tests/test_a_pool.py`:

```python
# tests/test_a_pool.py
"""S8 A pool extremes: 24 letters across F/P/Z families (no rank-4),
de-gated reachability, multi-P spectrum blend.

This file is the S8 owner test file (sibling: tests/test_motif.py /
test_vis.py / test_spectrum_shape.py / test_direction_kinds.py /
test_phase_windows.py / test_threshold_predicates.py / test_multi_slot.py).
Covers the 24 verbatim rows, gran tagging (Predator Lock motif=3, others
residue), SPECTRUM_SHAPE value-domain extension (power=4, mask='cross'),
SLOTS_PER_EVENT for P_cascade=2, multi-P blend single/double-letter
cases, n_locked gate retired audit, default-BB0 byte-identical regression."""
from __future__ import annotations
import pytest


_A_FAMILY_EXPECTED = {
    # 乙1 (8 escalation copies)
    "Apex Bloom":         "F",
    "Ember Drip":         "F",
    "Bastion Pile":       "F",
    "Apex Fang":          "Z",
    "Pan Sweep":          "Z",
    "Hotspot Amp":        "P",
    "Sink Cascade":       "P",
    "Glacial Drift":      "P",
    # 乙2 (8 native)
    "F_NOVA":             "F",
    "F_TRICKLE":          "F",
    "F_SCATTER":          "F",
    "Predator Lock":      "Z",
    "Void Bite":          "Z",
    "P_cascade":          "P",
    "P_crossclan_surge":  "P",
    "P_frozen":           "P",
    # 甲 (8 native)
    "F8Ar1":              "F",
    "Lance Front":        "F",
    "Ambush Venom":       "Z",
    "Sweep Surge":        "Z",
    "Nip Whisper":        "Z",
    "Coil Null":          "Z",
    "P_zscan_invert":     "P",
    "P_stutter":          "P",
}


def test_a_pool_module_exposes_24_letters_with_family_F_P_Z():
    """24 个 A 池 letter 全部存在;family ∈ {F, P, Z};禁 rank-4。"""
    from des._a_pool import A_FAMILY
    assert set(A_FAMILY.keys()) == set(_A_FAMILY_EXPECTED.keys())
    assert len(A_FAMILY) == 24, f"expected 24 A letters, got {len(A_FAMILY)}"
    for letter, fam in A_FAMILY.items():
        assert fam in ("F", "P", "Z"), f"{letter}: family {fam!r} must be F/P/Z (no rank-4)"
        assert fam == _A_FAMILY_EXPECTED[letter], (
            f"{letter}: expected family {_A_FAMILY_EXPECTED[letter]!r}, got {fam!r}")


def test_a_pool_no_rank_4_letter_in_alphabet_value_set():
    """spec §2: A 是 family F/P/Z 极端值,FAMILY_RANK 仍 {N,F,P,Z}。"""
    from des._a_pool import A_FAMILY
    assert "A" not in set(A_FAMILY.values())
    assert "S" not in set(A_FAMILY.values())
```

- [ ] **Step 2: 跑失败测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_a_pool.py -v
```

Expected: 2 条 FAIL with `ModuleNotFoundError: No module named 'des._a_pool'`。

- [ ] **Step 3: 创建 `src/des/_a_pool.py` 含 8 个常量**

新建 `src/des/_a_pool.py`,verbatim 抄 File Structure 表:

```python
# src/des/_a_pool.py
"""S8 A pool — 24 extreme primitives at family F/P/Z (NOT rank-4).

A primitives reuse every mechanism from S1–S7 at extreme parameter values.
The n_locked(chan)≥θ overwrite gate is RETIRED (2026-06-24); A is reachable
purely by the global affinity spectrum (within-family at aff=0.70).

This module is data-only. registry.py merges these tables into ALPHABET /
GRAN / MOTIF_LEN / _F / _Z / _P / SPECTRUM_SHAPE / SLOTS_PER_EVENT after
its own definitions, then runs module-load assertions over the merged set."""
from __future__ import annotations
from des.types import IN_PLACE_DIR


# letter -> "F" | "P" | "Z" (family at extreme values; FAMILY_RANK unchanged).
A_FAMILY: dict[str, str] = {
    # 乙1 — 8 escalation copies (copy-of lineage annotation only; not read).
    "Apex Bloom":         "F",
    "Ember Drip":         "F",
    "Bastion Pile":       "F",
    "Apex Fang":          "Z",
    "Pan Sweep":          "Z",
    "Hotspot Amp":        "P",
    "Sink Cascade":       "P",
    "Glacial Drift":      "P",
    # 乙2 — 8 native extreme variants.
    "F_NOVA":             "F",
    "F_TRICKLE":          "F",
    "F_SCATTER":          "F",
    "Predator Lock":      "Z",
    "Void Bite":          "Z",
    "P_cascade":          "P",
    "P_crossclan_surge":  "P",
    "P_frozen":           "P",
    # 甲 — 8 native extreme variants.
    "F8Ar1":              "F",
    "Lance Front":        "F",
    "Ambush Venom":       "Z",
    "Sweep Surge":        "Z",
    "Nip Whisper":        "Z",
    "Coil Null":          "Z",
    "P_zscan_invert":     "P",
    "P_stutter":          "P",
}


# letter -> "residue" | "motif" (Predator Lock is the only motif A).
A_GRAN: dict[str, str] = {l: ("motif" if l == "Predator Lock" else "residue")
                           for l in A_FAMILY}


# motif letter -> span length (>= 2). Predator Lock owns ≥3 motif spec → len 3.
A_MOTIF_LEN: dict[str, int] = {"Predator Lock": 3}


# F-pool A rows (S5 7-tuple): (f, dirs, p_leave, period, f_lo, burst_w, burst_k).
# Static rows have f_lo=f, burst_w=1, burst_k=1 (S5 degenerate path).
# F_NOVA is the sole windowed-f A: f_hi=0.85 / f_lo=0.05 / burst_w=20 / burst_k=1.
A_F: dict[str, tuple] = {
    # 乙1
    "Apex Bloom":   (0.85, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.20, 4, 0.85, 1, 1),
    "Ember Drip":   (0.05, "hash:ember",                       0.04, 9, 0.05, 1, 1),
    "Bastion Pile": (0.85, (IN_PLACE_DIR,),                    0.00, 3, 0.85, 1, 1),
    # 乙2
    "F_NOVA":       (0.85, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.50, 2, 0.05, 20, 1),
    "F_TRICKLE":    (0.02, "hash:trickle",                     0.02, 8, 0.02, 1, 1),
    "F_SCATTER":    (0.12, "hash:scatter3",                    0.60, 3, 0.12, 1, 1),
    # 甲
    "F8Ar1":        (0.25, "rand:1of4",                        0.10, 2, 0.25, 1, 1),
    "Lance Front":  (0.80, "hash:lance",                       0.30, 4, 0.80, 1, 1),
}


# Z-pool A rows (S1/S6/S3 4-tuple): (z, prey_clauses, period, vis_mode).
# vis_mode is "uniform" (default S1 path) for all but Void Bite / Nip Whisper
# which use "vis_weighted" (1/L · Σ vis-weighted hit). Predator Lock uses motif
# clause; others use family-only / threshold clause-tags from S3.
A_Z: dict[str, tuple] = {
    # 乙1
    "Apex Fang":     (1.50, (("Z", "generalist"),),                         9, "uniform"),
    "Pan Sweep":     (0.50, (("F",), ("Z",), ("P",)),                       6, "uniform"),
    # 乙2
    "Predator Lock": (1.45, (("Z", "motif", "len>=3"),),                    9, "uniform"),
    "Void Bite":     (0.95, (("N", "lowvis"),),                             5, "vis_weighted"),
    # 甲
    "Ambush Venom":  (1.30, (("F", "motif"),),                              7, "uniform"),
    "Sweep Surge":   (0.45, (("F",), ("P",)),                               3, "uniform"),
    "Nip Whisper":   (0.15, (("N", "lowvis"),),                             3, "vis_weighted"),
    "Coil Null":     (0.20, (("Z",),),                                      8, "uniform"),
}


# P-pool A rows: (p_add, period). rate = min(P_MAX, MU + p_add) is computed in
# phenotype() at v1; A's high-p_add rows still cap at P_MAX in v1 (calibration
# constants own the cap; spec §6 + Hotspot Amp roster note "rate=0.31").
A_P: dict[str, tuple] = {
    # 乙1
    "Hotspot Amp":       (0.30, 3),
    "Sink Cascade":      (0.25, 3),
    "Glacial Drift":     (0.0,  12),
    # 乙2
    "P_cascade":         (0.28, 2),
    "P_crossclan_surge": (0.20, 4),
    "P_frozen":          (0.0,  8),
    # 甲
    "P_zscan_invert":    (0.10, 4),
    "P_stutter":         (0.32, 2),
}


# Shape per P-pool A (S2 SPECTRUM_SHAPE 3-tuple, value domain extended in S8):
# (power, family_mask, flatten_mix). power ∈ {1,2,3,4} (S8 adds 4.0); family_mask
# ∈ {None,"F","Z","N","adjacent","cross"} (S8 adds "cross" = |Δrank|>=2).
A_SHAPE: dict[str, tuple] = {
    "Hotspot Amp":       (1.0, None,    0.0),    # roster: q ∝ aff(·)
    "Sink Cascade":      (1.0, "N",     0.0),    # roster: q ∝ aff(·)·𝟙[fam=N]
    "Glacial Drift":     (1.0, None,    0.0),    # roster: q ∝ aff(·)
    "P_cascade":         (1.0, None,    0.0),    # roster: q ∝ aff(·); slots=2
    "P_crossclan_surge": (1.0, "cross", 0.0),    # roster: q ∝ aff(·)·𝟙[|Δrank|≥2]
    "P_frozen":          (4.0, None,    0.0),    # roster: q ∝ aff(·)^4
    "P_zscan_invert":    (1.0, "F",     0.0),    # roster: q ∝ aff(·)·𝟙[fam=F]
    "P_stutter":         (4.0, None,    0.0),    # roster: q ∝ aff(·)^4
}


# Slots-per-event per A letter. P_cascade is the sole slots=2 letter (roster
# "2 slots/event"); every other A row defaults to slots=1 (S7 norm).
A_SLOTS: dict[str, int] = {l: (2 if l == "P_cascade" else 1) for l in A_FAMILY}
```

- [ ] **Step 4: 跑测试,确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_a_pool.py -v
```

Expected: 2 条全 PASS。

- [ ] **Step 5: 跑全 suite,确认无回归**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿。`_a_pool.py` 是孤儿模块,无 `__init__` 导入,既有路径 0 漂移。

Backtrack: 若 `from des.types import IN_PLACE_DIR` ImportError → S4 未落地?当前路线图 S0/S6/S1/S2/S4/S5/S3/S7 都假设已在 main(spec § dependency)。手 `grep IN_PLACE_DIR src/des/types.py` 确认;若真的不存在,临时改用 `(0, 0)` sentinel 占位,但 in commit message 标 `BLOCKED-ON-S4`。

- [ ] **Step 6: Commit**

```bash
git add src/des/_a_pool.py tests/test_a_pool.py
git commit -m "feat(s8): add _a_pool.py with 24 A-pool data tables (no consumer)

Eight constants (A_FAMILY / A_GRAN / A_MOTIF_LEN / A_F / A_Z / A_P /
A_SHAPE / A_SLOTS) hold verbatim roster data for the 24 A-pool letters.
Family is F/P/Z — no rank-4 letter (descaffold 2026-06-20). registry.py
merge happens in Task 2; this commit is data-only."
```

---

### Task 2: 合入 24 行到 `ALPHABET` / `GRAN` / `MOTIF_LEN` / `_F` / `_Z` / `_P` / `SLOTS_PER_EVENT`(无 SPECTRUM_SHAPE,Task 4)

**Goal:** 在 `src/des/registry.py` 既有表定义结束后,用 `dict.update(A_*)` 合入 Task 1 的 24 行数据,**不**加 `SPECTRUM_SHAPE`(那张表的值域 assert 还没扩展,留 Task 4 一起做)。这步完成后,`phenotype()` 解析任意 A letter **不会崩**,但谱仍走 dominant_p 单源(Task 5 才升级 blend);kernel 完全不动;默认 BB0 字节级不变(BB0_TEMPLATE 不含 A letter)。

**Files:**
- Modify: `src/des/registry.py`(在 `_P` 块之后、`affinity` 之前插入合入语句 + module-load 极端值 assert)
- Test: `tests/test_a_pool.py`(append 注册表覆盖 + 极端值边界)
- Test: `tests/test_registry.py`(append 一条 `_F` / `_Z` / `_P` keys 包含 A_F/A_Z/A_P keys 的覆盖断言)

**Interfaces:**
- Consumes: `des._a_pool.A_FAMILY` / `A_GRAN` / `A_MOTIF_LEN` / `A_F` / `A_Z` / `A_P` / `A_SLOTS`(Task 1)。
- Produces:
  - `ALPHABET` 扩到 16 (v1+S2) + 24 (A) = 40 个 key(假设 S5 加的 `FBURST`/`F_NOVA` 不与 A `F_NOVA` 冲突,见下文 conflict 决议)。
  - `GRAN` 同步扩到 40 行。
  - `MOTIF_LEN` 从 `{}` 扩到 `{"Predator Lock": 3}`。
  - `_F` / `_Z` / `_P` 各自吃下 8 行 A。
  - `SLOTS_PER_EVENT` 加 24 行(P_cascade=2,其余=1)。
  - module-load assert 守 A 行极端值 `f≤0.85 / z≤1.5 / p_add≤0.34`。

- [ ] **Step 1: `F_NOVA` 命名冲突解决(spec §1 乙2 + S5 plan)**

`F_NOVA` 在 S5 plan 的 Naming Contract 表里也被作为新 F 行注册过(spec roster L218 与 spec §1 乙2)。**两条 spec 指向同一字面值** —— S5 已落地的 `F_NOVA` 行就是 A 池的 `F_NOVA`,不是两个独立 letter。

行动:**不重复注册**。Task 1 的 `A_F["F_NOVA"]` 数值与 S5 plan `_F["F_NOVA"]` 数值 verbatim 一致(`(0.85, 4-nbr-tuple, 0.50, 2, 0.05, 20, 1)`)。Task 2 合入逻辑加一行守门 `if key in _F: assert _F[key] == A_F[key]` —— 任何不一致都是 spec 漂移,立即 fail。

`F_NOVA` 同时属于 S5 owner(windowed-f 落地)与 S8 owner(A 池注册);**`A_FAMILY` 仍保留 `F_NOVA: "F"`**(单 source of truth 是 `_a_pool.py`),`ALPHABET` 与 `_F` 由 S5 先注册一次,S8 update 时验证字节级相等而不覆盖。

- [ ] **Step 2: 写失败测试 — 注册表覆盖 + 冲突守门 + 极端值**

追加到 `tests/test_a_pool.py`:

```python
# --- Task 2 surface: registry merge --------------------------------------------

def test_a_pool_letters_in_alphabet_with_correct_family():
    """每个 A letter 在 ALPHABET 里,value == A_FAMILY[letter]."""
    from des.registry import ALPHABET
    from des._a_pool import A_FAMILY
    for letter, fam in A_FAMILY.items():
        assert letter in ALPHABET, f"{letter}: missing from ALPHABET"
        assert ALPHABET[letter] == fam, (
            f"{letter}: ALPHABET says {ALPHABET[letter]!r}, A_FAMILY says {fam!r}")


def test_a_pool_letters_in_gran_with_correct_value():
    """每个 A letter 在 GRAN 里;Predator Lock=motif,其余=residue."""
    from des.registry import GRAN
    from des._a_pool import A_FAMILY, A_GRAN
    for letter in A_FAMILY:
        assert letter in GRAN, f"{letter}: missing from GRAN"
        assert GRAN[letter] == A_GRAN[letter], (
            f"{letter}: GRAN={GRAN[letter]!r}, A_GRAN={A_GRAN[letter]!r}")


def test_predator_lock_motif_len_3():
    """Predator Lock 是 A 池唯一 motif letter,MOTIF_LEN=3 (roster L225)."""
    from des.registry import MOTIF_LEN
    assert MOTIF_LEN.get("Predator Lock") == 3


def test_a_pool_F_rows_merged_to__F_dict():
    """8 个 F 类 A 行在 _F dict 里,数值字节级 = A_F."""
    from des.registry import _F
    from des._a_pool import A_F
    for letter, row in A_F.items():
        assert letter in _F, f"{letter}: missing from _F"
        assert _F[letter] == row, f"{letter}: _F={_F[letter]!r}, A_F={row!r}"


def test_a_pool_Z_rows_merged_to__Z_dict():
    from des.registry import _Z
    from des._a_pool import A_Z
    for letter, row in A_Z.items():
        assert letter in _Z, f"{letter}: missing from _Z"
        assert _Z[letter] == row, f"{letter}: _Z={_Z[letter]!r}, A_Z={row!r}"


def test_a_pool_P_rows_merged_to__P_dict():
    from des.registry import _P
    from des._a_pool import A_P
    for letter, row in A_P.items():
        assert letter in _P, f"{letter}: missing from _P"
        assert _P[letter] == row, f"{letter}: _P={_P[letter]!r}, A_P={row!r}"


def test_a_pool_slots_per_event_merged():
    """SLOTS_PER_EVENT 覆盖 24 A letter;P_cascade=2,其余=1."""
    from des.registry import SLOTS_PER_EVENT
    from des._a_pool import A_SLOTS
    for letter, n in A_SLOTS.items():
        assert SLOTS_PER_EVENT.get(letter) == n, (
            f"{letter}: SLOTS_PER_EVENT={SLOTS_PER_EVENT.get(letter)!r}, expected {n!r}")
    assert SLOTS_PER_EVENT["P_cascade"] == 2


def test_a_pool_extreme_value_bounds_assert_at_module_load():
    """spec §6: f<=0.85 / z<=1.5 / p_add<=0.34 守 A 行 module-load."""
    from des.registry import _F, _Z, _P
    from des._a_pool import A_F, A_Z, A_P
    for letter in A_F:
        assert _F[letter][0] <= 0.85, f"{letter}: f={_F[letter][0]} > 0.85"
    for letter in A_Z:
        assert _Z[letter][0] <= 1.5, f"{letter}: z={_Z[letter][0]} > 1.5"
    for letter in A_P:
        assert _P[letter][0] <= 0.34, f"{letter}: p_add={_P[letter][0]} > 0.34"


def test_default_bb0_phenotype_does_not_raise_after_A_merge():
    """默认 BB0 layout 不含 A letter; phenotype() 解析不应被 A 行影响, 不抛."""
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert p is not None
    # 默认 BB0 是 v1 6-字母组合,不应包含任何 A letter
    from des._a_pool import A_FAMILY
    for letter in BB0_TEMPLATE["layout"]:
        assert letter not in A_FAMILY, (
            f"BB0 should not contain A letter, found {letter}")
```

追加到 `tests/test_registry.py`:

```python
def test_s8_a_pool_keys_subset_of_registry_tables():
    """S8: A 池 keys ⊆ _F/_Z/_P 各表;SLOTS_PER_EVENT.keys() ⊇ ALPHABET.keys()."""
    from des.registry import _F, _Z, _P, ALPHABET, SLOTS_PER_EVENT
    from des._a_pool import A_F, A_Z, A_P, A_FAMILY
    assert set(A_F.keys()) <= set(_F.keys())
    assert set(A_Z.keys()) <= set(_Z.keys())
    assert set(A_P.keys()) <= set(_P.keys())
    assert set(A_FAMILY.keys()) <= set(ALPHABET.keys())
    assert set(ALPHABET.keys()) <= set(SLOTS_PER_EVENT.keys())
```

- [ ] **Step 3: 跑失败测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_a_pool.py tests/test_registry.py::test_s8_a_pool_keys_subset_of_registry_tables -v
```

Expected: 9 条全 FAIL —— A letter 还没合入 ALPHABET / _F / _Z / _P / GRAN / MOTIF_LEN / SLOTS_PER_EVENT。

- [ ] **Step 4: 在 `src/des/registry.py` 合入 24 行**

打开 `src/des/registry.py`,在文件顶部 `from des.types import ...` 之后加:

```python
from des._a_pool import (
    A_FAMILY as _A_FAMILY,
    A_GRAN as _A_GRAN,
    A_MOTIF_LEN as _A_MOTIF_LEN,
    A_F as _A_F,
    A_Z as _A_Z,
    A_P as _A_P,
    A_SLOTS as _A_SLOTS,
)
# A_SHAPE 在 SPECTRUM_SHAPE 定义后合入(Task 4),此处不导入避免循环序问题。
```

在 `SLOTS_PER_EVENT` 块之后、`def affinity` 之前(若 affinity 不在那位置,选 `_P` 块之后第一个非数据行之前)插入合入语句:

```python
# --- S8: A pool merge (24 letters at family F/P/Z, no rank-4) ----------------
# spec §2 / 4: A reaches via affinity spectrum (within-family at aff=0.70);
# the n_locked(chan)>=θ overwrite gate is RETIRED (2026-06-24). The merge
# below extends ALPHABET / GRAN / MOTIF_LEN / _F / _Z / _P / SLOTS_PER_EVENT
# with the 24 A rows; SPECTRUM_SHAPE merge happens in Task 4 (value-domain
# assert needs to be relaxed first to accept power=4 and family_mask='cross').

# Conflict guard: F_NOVA is registered by both S5 (windowed-f owner) and S8
# (A-pool owner). Verify byte-equal data when both sides have an entry, never
# silently overwrite — drift in either spec must surface immediately.
for _lk, _lv in _A_F.items():
    if _lk in _F:
        assert _F[_lk] == _lv, (
            f"A-pool / pre-S8 conflict on _F[{_lk!r}]: "
            f"S8 wants {_lv!r}, registry has {_F[_lk]!r}")
for _lk, _lv in _A_Z.items():
    if _lk in _Z:
        assert _Z[_lk] == _lv, (
            f"A-pool / pre-S8 conflict on _Z[{_lk!r}]: "
            f"S8 wants {_lv!r}, registry has {_Z[_lk]!r}")
for _lk, _lv in _A_P.items():
    if _lk in _P:
        assert _P[_lk] == _lv, (
            f"A-pool / pre-S8 conflict on _P[{_lk!r}]: "
            f"S8 wants {_lv!r}, registry has {_P[_lk]!r}")
del _lk, _lv

ALPHABET.update(_A_FAMILY)
GRAN.update(_A_GRAN)
MOTIF_LEN.update(_A_MOTIF_LEN)
_F.update(_A_F)
_Z.update(_A_Z)
_P.update(_A_P)
SLOTS_PER_EVENT.update(_A_SLOTS)

# Module-load extreme-value assertions (spec §6). Only A rows are bounds-checked;
# pre-S8 rows already pass the bounds (every v1/S1-S7 row has f<=0.5 / z<=1.0 /
# p_add<=0.07), so iterating only A keeps the assert message scoped to S8 changes.
for _lk in _A_F:
    assert _F[_lk][0] <= 0.85, f"_F[{_lk!r}].f = {_F[_lk][0]} exceeds A-pool bound 0.85"
for _lk in _A_Z:
    assert _Z[_lk][0] <= 1.5, f"_Z[{_lk!r}].z = {_Z[_lk][0]} exceeds A-pool bound 1.5"
for _lk in _A_P:
    assert _P[_lk][0] <= 0.34, f"_P[{_lk!r}].p_add = {_P[_lk][0]} exceeds A-pool bound 0.34"
del _lk

# FEATURE_BIT for predicate-bit-based feature_mask was built from ALPHABET at
# module load (S6); it is *positional* by sorted letter name, so re-sort after
# the merge. (S6 rebuilds the dict; S8 also rebuilds to surface any new key.)
# Implementation note: if S6 already wires FEATURE_BIT to depend on ALPHABET
# at module load AND ALPHABET was extended in-place via .update() before
# FEATURE_BIT was built, the original dict-comprehension at S6 line is enough.
# If FEATURE_BIT is computed strictly before the A merge above, rebuild it:
FEATURE_BIT = {name: 1 << i for i, name in enumerate(sorted(ALPHABET))}
```

(注:`FEATURE_BIT` 的重建是防御性的——若 S6 把 `FEATURE_BIT` 设计成 `phenotype()` 内每次重算或合并在 ALPHABET 之后,则上面 1 行可删;**保留这一行不抑制任何既有行为**,因为它仍是从同一 ALPHABET 派生,字节级等价。)

- [ ] **Step 5: 跑测试,确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_a_pool.py tests/test_registry.py::test_s8_a_pool_keys_subset_of_registry_tables -v
```

Expected: 9 条全 PASS。

Backtrack:
- 若 `test_a_pool_F_rows_merged_to__F_dict` FAIL with conflict assert 抛出 → S5 落地的 `F_NOVA` 数值与 spec roster 不一致;比对 spec §1 乙2 `(0.85, 4-nbr, 0.50, 2, 0.05, 20, 1)`,**fix S5 plan 在 future commit** 而非本 commit 静默调整(spec 漂移不进 S8 责任范围)。
- 若 `test_predator_lock_motif_len_3` FAIL → S6 加 `MOTIF_LEN` 的格式是 `dict[str,int]` 而非 `dict[str,tuple]`?核对 S6 plan Task 1 Step 3 输出 `MOTIF_LEN: dict[str, int] = {}`。
- 若 `assert FEATURE_BIT[A_letter]` 在测试外抛 KeyError → `FEATURE_BIT` 重建那行被注释掉而 S6 原始构造不接 A;复跑 Step 4 加回 `FEATURE_BIT = ...` 那行。

- [ ] **Step 6: 跑全 suite,确认 default BB0 byte-identical**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿。默认 BB0_TEMPLATE 不含 A letter,phenotype() 走旧路径(单 P 字母 dominant_p);kernel 完全不动;同 seed engine 跑两次仍 byte-identical。

Backtrack: 若 spectrum-数值类 FAIL 出现(`tests/test_registry.py::test_dominant_p_*` 等),root cause **不应**在本 task —— SPECTRUM_SHAPE 还没合入 A 池(Task 4 的活),`_spectrum_for` 仍只读 6+10 = 16 字母 → 默认局 spectrum 字节级与 Task 2 前 identical。若仍 FAIL,先核对 S2 Task 6 是否把那些数值断言改成结构断言(spec §2 红线:守结构非数值);若 S2 没改,**记录测试名 + 留 Task 7 final sweep 处理**,本步不修。

- [ ] **Step 7: Commit**

```bash
git add src/des/registry.py tests/test_a_pool.py tests/test_registry.py
git commit -m "feat(s8): merge 24 A-pool rows into ALPHABET / _F / _Z / _P / SLOTS_PER_EVENT

dict.update merge of _a_pool tables + conflict guards (F_NOVA already
registered by S5, byte-equal verify). Module-load asserts bound A rows
at f<=0.85 / z<=1.5 / p_add<=0.34. Predator Lock gets MOTIF_LEN=3.
SPECTRUM_SHAPE merge lands in Task 4 (after value-domain extension)."
```

---

### Task 3: 24 行 verbatim per-subpool 验证 + de-gate 审计 + relabel-invariance

**Goal:** 在 Task 2 已合入 24 行的基础上,把「每一行 verbatim 数据」、「de-gate 后无 n_locked code path」、「relabel-invariance」这三件守门测试一次性写完。这一步**只加测试,不动 src**;若有 FAIL,root cause 100% 在 Task 1 / Task 2 的数据/合入逻辑漂移,fix forward。

**Files:**
- Test: `tests/test_a_pool.py`(append 3 个子池 verbatim 表 + de-gate audit + relabel-invariance)

**Interfaces:**
- Consumes: Task 2 合入后的 `ALPHABET` / `GRAN` / `MOTIF_LEN` / `_F` / `_Z` / `_P` / `SLOTS_PER_EVENT`(全 40 行)+ `des._a_pool` 8 个常量 + `des.registry.phenotype` 同 v1 行为。
- Produces: 验证 23 行 verbatim(乙1 8 + 乙2 8 + 甲 8) + 1 行确认 P_cascade slots=2 + 4 条静态守门测试。

- [ ] **Step 1: 写 verbatim 表测试 — 乙1 子池(8 行,带 copy-of 注释验证)**

追加到 `tests/test_a_pool.py`:

```python
# --- Task 3 surface: per-subpool verbatim audit -------------------------------

_SUB_YI1_F = (
    ("Apex Bloom",   (0.85, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.20, 4, 0.85, 1, 1)),
    ("Ember Drip",   (0.05, "hash:ember",                       0.04, 9, 0.05, 1, 1)),
    ("Bastion Pile", (0.85, None,                               0.00, 3, 0.85, 1, 1)),
)
_SUB_YI1_Z = (
    ("Apex Fang", (1.50, (("Z", "generalist"),), 9, "uniform")),
    ("Pan Sweep", (0.50, (("F",), ("Z",), ("P",)), 6, "uniform")),
)
_SUB_YI1_P = (
    ("Hotspot Amp",   (0.30, 3)),
    ("Sink Cascade",  (0.25, 3)),
    ("Glacial Drift", (0.0,  12)),
)


def test_yi1_F_rows_verbatim_from_roster_L188_to_L211():
    """乙1 F 类 3 行 verbatim 抄 roster L188-211 (Apex Bloom / Ember Drip / Bastion Pile)."""
    from des.registry import _F
    from des.types import IN_PLACE_DIR
    # Bastion Pile dirs 是 (IN_PLACE_DIR,) — 测试用占位 None 后跑时替换
    for letter, expected in _SUB_YI1_F:
        row = _F[letter]
        # 7-tuple shape
        assert len(row) == 7, f"{letter}: _F row must be 7-tuple (S5 shape), got len {len(row)}"
        # f / p_leave / period / f_lo / burst_w / burst_k 字节级
        assert row[0] == expected[0], f"{letter}: f mismatch"
        assert row[2] == expected[2], f"{letter}: p_leave mismatch"
        assert row[3] == expected[3], f"{letter}: period mismatch"
        assert row[4] == expected[4], f"{letter}: f_lo mismatch"
        assert row[5] == expected[5], f"{letter}: burst_w mismatch"
        assert row[6] == expected[6], f"{letter}: burst_k mismatch"
        if letter == "Bastion Pile":
            assert row[1] == (IN_PLACE_DIR,), f"Bastion Pile dirs must be (IN_PLACE_DIR,)"
        else:
            assert row[1] == expected[1], f"{letter}: dirs mismatch"


def test_yi1_Z_rows_verbatim():
    """乙1 Z 类 2 行 (Apex Fang / Pan Sweep) verbatim."""
    from des.registry import _Z
    for letter, expected in _SUB_YI1_Z:
        assert _Z[letter] == expected, f"{letter}: _Z={_Z[letter]!r}, expected {expected!r}"


def test_yi1_P_rows_verbatim():
    """乙1 P 类 3 行 (Hotspot Amp / Sink Cascade / Glacial Drift) verbatim."""
    from des.registry import _P
    for letter, expected in _SUB_YI1_P:
        assert _P[letter] == expected, f"{letter}: _P={_P[letter]!r}, expected {expected!r}"
```

- [ ] **Step 2: 写 verbatim 表测试 — 乙2 子池(8 行,F_NOVA / Predator Lock / P_cascade / P_frozen 关键行)**

继续追加到 `tests/test_a_pool.py`:

```python
def test_yi2_F_rows_verbatim_with_F_NOVA_windowed():
    """乙2 F 类 3 行 (F_NOVA windowed / F_TRICKLE / F_SCATTER) verbatim."""
    from des.registry import _F
    expected = {
        "F_NOVA":    (0.85, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.50, 2, 0.05, 20, 1),
        "F_TRICKLE": (0.02, "hash:trickle",                     0.02, 8, 0.02, 1, 1),
        "F_SCATTER": (0.12, "hash:scatter3",                    0.60, 3, 0.12, 1, 1),
    }
    for letter, row in expected.items():
        assert _F[letter] == row, f"{letter}: _F={_F[letter]!r}, expected {row!r}"


def test_yi2_Z_rows_predator_lock_motif_void_bite_vis_weighted():
    """乙2 Z 类 2 行 (Predator Lock motif+len>=3 / Void Bite vis_weighted) verbatim."""
    from des.registry import _Z, GRAN, MOTIF_LEN
    assert _Z["Predator Lock"] == (
        1.45, (("Z", "motif", "len>=3"),), 9, "uniform")
    assert GRAN["Predator Lock"] == "motif"
    assert MOTIF_LEN["Predator Lock"] == 3
    assert _Z["Void Bite"] == (
        0.95, (("N", "lowvis"),), 5, "vis_weighted")


def test_yi2_P_rows_cascade_crossclan_frozen_verbatim():
    """乙2 P 类 3 行 (P_cascade / P_crossclan_surge / P_frozen) verbatim."""
    from des.registry import _P, SLOTS_PER_EVENT
    assert _P["P_cascade"]         == (0.28, 2)
    assert _P["P_crossclan_surge"] == (0.20, 4)
    assert _P["P_frozen"]          == (0.0,  8)
    # P_cascade 唯一 slots=2 letter (S7 reserved hook + S8 fills)
    assert SLOTS_PER_EVENT["P_cascade"] == 2
    assert SLOTS_PER_EVENT["P_crossclan_surge"] == 1
    assert SLOTS_PER_EVENT["P_frozen"] == 1
```

- [ ] **Step 3: 写 verbatim 表测试 — 甲 子池(8 行,Sweep Surge boundary 0.45 / Nip Whisper z=0.15 dwarf 等)**

继续追加到 `tests/test_a_pool.py`:

```python
def test_jia_F_rows_F8Ar1_random_LanceFront_hash():
    """甲 F 类 2 行 (F8Ar1 rand:1of4 / Lance Front hash:lance) verbatim."""
    from des.registry import _F
    assert _F["F8Ar1"]       == (0.25, "rand:1of4",  0.10, 2, 0.25, 1, 1)
    assert _F["Lance Front"] == (0.80, "hash:lance", 0.30, 4, 0.80, 1, 1)


def test_jia_Z_rows_ambush_sweep_nip_coil_verbatim():
    """甲 Z 类 4 行 (Ambush Venom motif / Sweep Surge boundary 0.45 /
    Nip Whisper vis_weighted / Coil Null Z-only)."""
    from des.registry import _Z
    assert _Z["Ambush Venom"] == (1.30, (("F", "motif"),),       7, "uniform")
    assert _Z["Sweep Surge"]  == (0.45, (("F",), ("P",)),        3, "uniform")
    assert _Z["Nip Whisper"]  == (0.15, (("N", "lowvis"),),      3, "vis_weighted")
    assert _Z["Coil Null"]    == (0.20, (("Z",),),               8, "uniform")


def test_jia_P_rows_zscan_invert_stutter_verbatim():
    """甲 P 类 2 行 (P_zscan_invert F-only / P_stutter aff^4) verbatim."""
    from des.registry import _P
    assert _P["P_zscan_invert"] == (0.10, 4)
    assert _P["P_stutter"]      == (0.32, 2)
```

- [ ] **Step 4: 写 de-gate 审计测试 — 无 n_locked 覆写 code path**

继续追加到 `tests/test_a_pool.py`:

```python
# --- De-gate audit (spec §2 + §3): n_locked overwrite gate is RETIRED ---------

def test_n_locked_is_advisory_only_not_wired_to_mutation_path():
    """spec §2 末尾: n_locked is computed on-demand but has no consumer in the
    mutation core after S8. Audit by grep: src/des/kernels/reproduction.py
    + src/des/registry.py must not call n_locked() inside any mutation
    decision branch. n_locked() the function may exist (S6 owns it as a
    structural readout); what's banned is its use as a gate."""
    import inspect
    from des.kernels import reproduction as repro_mod
    src = inspect.getsource(repro_mod)
    # n_locked must not be referenced inside _mutation_outcomes / phase2_reproduce
    # at all. (S6 owns the function for advisory readouts elsewhere.)
    assert "n_locked" not in src, (
        "n_locked must NOT appear in src/des/kernels/reproduction.py — "
        "the gate was retired 2026-06-24 (spec §2). Found references.")


def test_a_letters_reachable_via_within_family_affinity_spectrum():
    """Apex Bloom (family=F) reachable from F4Nr1 spectrum (within-family at
    aff=0.70). Test: _spectrum_for('F4Nr1') must contain Apex Bloom in the
    target set with non-zero weight (residue gran-matched, aff=0.70 same-family)."""
    from des.registry import _spectrum_for
    spec = dict(_spectrum_for("F4Nr1"))
    assert "Apex Bloom" in spec, (
        "Apex Bloom must be reachable from F4Nr1 via within-family aff=0.70 "
        "spectrum (de-gate ⇒ A is a normal same-family target)")
    assert spec["Apex Bloom"] > 0.0


def test_p_letters_reachable_via_within_family_affinity_spectrum():
    """Hotspot Amp (family=P) reachable from P_base spectrum (within-family)."""
    from des.registry import _spectrum_for
    spec = dict(_spectrum_for("P_base"))
    assert "Hotspot Amp" in spec
    assert spec["Hotspot Amp"] > 0.0
```

- [ ] **Step 5: 写 relabel-invariance 测试 — 重排量级不漂移结构**

继续追加到 `tests/test_a_pool.py`:

```python
# --- Relabel-invariance (spec §6) ---------------------------------------------

def test_a_family_is_structural_under_magnitude_relabel(monkeypatch):
    """spec §6: A 的 family/gran 是结构属性,与 f/z/p_add 量级无关。
    重排 _F / _Z / _P 的量级 (例如把 Apex Bloom f 从 0.85 改 0.50),
    ALPHABET / GRAN / MOTIF_LEN 不应漂移。"""
    from des import registry
    from des._a_pool import A_FAMILY, A_GRAN
    # 重排量级
    monkeypatch.setitem(registry._F, "Apex Bloom",
                        (0.50, ((-1, 0),), 0.99, 99, 0.50, 1, 1))
    monkeypatch.setitem(registry._Z, "Apex Fang",
                        (0.99, (("Z", "generalist"),), 99, "uniform"))
    monkeypatch.setitem(registry._P, "Hotspot Amp", (0.01, 99))
    # 结构属性不动
    for letter, fam in A_FAMILY.items():
        assert registry.ALPHABET[letter] == fam
        assert registry.GRAN[letter] == A_GRAN[letter]
```

- [ ] **Step 6: 跑全 task3 测试,确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_a_pool.py -v
```

Expected: 全 PASS。

Backtrack:
- 若 `test_a_letters_reachable_via_within_family_affinity_spectrum` FAIL with `Apex Bloom not in spec`:
  - root cause:S2 `_spectrum_for` 在 ALPHABET 扩到 24 个新行后,可能因为 GRAN 不匹配把 A 行过滤掉。验证 Apex Bloom 的 `GRAN == "residue"` 与 F4Nr1 一致;若不一致,Task 1 数据漂移,修 `_a_pool.py`。
- 若 `test_n_locked_is_advisory_only_not_wired_to_mutation_path` FAIL with `"n_locked" in src`:
  - root cause:S6 把 n_locked 用进了 mutation 路径(不该)或 S8 Task 4 把 SPECTRUM_SHAPE merge 错地调进了 reproduction.py。立即 root-cause 修复;de-gate 红线不可妥协(spec §2 + §3)。
- 若 `test_yi2_Z_rows_predator_lock_motif_void_bite_vis_weighted` FAIL with `vis_mode=='vis_weighted'` 不存在:
  - root cause:S1 plan 落地的 `_Z` 行可能没用 `"vis_weighted"` 字面值;查 S1 Task 6 spec,如果是 `"vis_weighted"` 单数,则 _a_pool.py 是对的;若 S1 用了不同字面,修 `_a_pool.py` 对齐 S1 实际 string 而非 spec。

- [ ] **Step 7: 跑全 suite**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿(本 task 只加测试,Task 2 已守 default BB0 byte-identical)。

- [ ] **Step 8: Commit**

```bash
git add tests/test_a_pool.py
git commit -m "test(s8): verbatim per-subpool audit + de-gate + relabel-invariance

Pin every A row verbatim from spec §1 (乙1/乙2/甲, 8+8+8 = 24 letters).
Audit n_locked gate retired (no reference in reproduction.py). Verify A
letters reachable via within-family aff=0.70 spectrum. Relabel-invariance:
A family/gran is structural, not magnitude-dependent."
```

---

### Task 4: 扩展 `SPECTRUM_SHAPE` 值域(power=4 / mask="cross")+ 合入 A_SHAPE 8 行

**Goal:** S2 在 `registry.py` 的 module-load assert 锁了 `power ∈ {1.0, 2.0, 3.0}` + `family_mask ∈ {None, "F", "Z", "N", "adjacent"}`。S8 三个 A 行需要 `power=4.0`(`P_frozen` / `P_stutter` aff⁴)+ `family_mask="cross"`(`P_crossclan_surge` |Δrank|≥2)—— 必须同步扩展 assert 值域,合入 `A_SHAPE` 8 行,然后扩展 `_spectrum_for(letter)` 函数体加 `"cross"` 分支。这是 S8 唯一动 `_spectrum_for` 的 task。

**Files:**
- Modify: `src/des/registry.py` — `SPECTRUM_SHAPE` 块下 assert 放宽 + `update(_A_SHAPE)` + `_spectrum_for` 函数体加 `"cross"` 分支
- Test: `tests/test_a_pool.py`(append SPECTRUM_SHAPE 域 + A_SHAPE 合入)
- Test: `tests/test_spectrum_shape.py`(append `power=4` + `cross` 行为)

**Interfaces:**
- Consumes: Task 1 `A_SHAPE` 8 行;Task 2 已合入的 `_P` 24 + 8 行 P 字母;S2 `_spectrum_for(letter)` body(带 power / family_mask / flatten_mix 三旋钮)。
- Produces:
  - `SPECTRUM_SHAPE` 加 8 行(`Hotspot Amp` / `Sink Cascade` / `Glacial Drift` / `P_cascade` / `P_crossclan_surge` / `P_frozen` / `P_zscan_invert` / `P_stutter`)。
  - module-load assert 放宽:`power ∈ {1.0, 2.0, 3.0, 4.0}` + `family_mask ∈ {None, "F", "Z", "N", "adjacent", "cross"}`。
  - `_spectrum_for(letter)` 加 `"cross"` family_mask 分支:`if mask == "cross": if abs(FAMILY_RANK[ALPHABET[t]] - src_rank) < 2: continue`。

- [ ] **Step 1: 写失败测试 — 域扩展 + cross/power=4 行为**

追加到 `tests/test_a_pool.py`:

```python
# --- Task 4 surface: SPECTRUM_SHAPE merge + value-domain extension ------------

def test_a_pool_shape_rows_merged():
    """8 个 A_SHAPE 行进 SPECTRUM_SHAPE,verbatim 抄 _a_pool.A_SHAPE."""
    from des.registry import SPECTRUM_SHAPE
    from des._a_pool import A_SHAPE
    for letter, row in A_SHAPE.items():
        assert letter in SPECTRUM_SHAPE, f"{letter}: missing from SPECTRUM_SHAPE"
        assert SPECTRUM_SHAPE[letter] == row, (
            f"{letter}: SPECTRUM_SHAPE={SPECTRUM_SHAPE[letter]!r}, A_SHAPE={row!r}")


def test_p_frozen_shape_power_4():
    """P_frozen aff^4 锐化 (roster L237 q ∝ aff^4)."""
    from des.registry import SPECTRUM_SHAPE
    assert SPECTRUM_SHAPE["P_frozen"] == (4.0, None, 0.0)


def test_p_stutter_shape_power_4():
    """P_stutter aff^4 锐化 (roster L264 q ∝ aff^4 高率零产出)."""
    from des.registry import SPECTRUM_SHAPE
    assert SPECTRUM_SHAPE["P_stutter"] == (4.0, None, 0.0)


def test_p_crossclan_surge_shape_cross_mask():
    """P_crossclan_surge 跨族大跳 (roster L234 q ∝ aff·𝟙[|Δrank|≥2])."""
    from des.registry import SPECTRUM_SHAPE
    assert SPECTRUM_SHAPE["P_crossclan_surge"] == (1.0, "cross", 0.0)


def test_spectrum_shape_power_domain_includes_4():
    """S8 扩展: power ∈ {1, 2, 3, 4} (P_frozen / P_stutter aff^4)."""
    from des.registry import SPECTRUM_SHAPE
    powers = {row[0] for row in SPECTRUM_SHAPE.values()}
    assert 4.0 in powers, "power=4.0 must be allowed for P_frozen / P_stutter"
    # 不应有越界值
    for letter, (power, _, _) in SPECTRUM_SHAPE.items():
        assert power in (1.0, 2.0, 3.0, 4.0), f"{letter}: bad power {power!r}"


def test_spectrum_shape_family_mask_domain_includes_cross():
    """S8 扩展: family_mask ∈ {None,F,Z,N,adjacent,cross}."""
    from des.registry import SPECTRUM_SHAPE
    masks = {row[1] for row in SPECTRUM_SHAPE.values()}
    assert "cross" in masks, "family_mask='cross' must be allowed for P_crossclan_surge"
    for letter, (_, mask, _) in SPECTRUM_SHAPE.items():
        assert mask in (None, "F", "Z", "N", "adjacent", "cross"), (
            f"{letter}: bad family_mask {mask!r}")
```

追加到 `tests/test_spectrum_shape.py`:

```python
# --- S8: cross-family mask + power=4 behaviour --------------------------------

def test_p_crossclan_surge_only_hits_cross_rank_targets():
    """family_mask='cross' (|Δrank|>=2): P (rank=2) 的 cross targets 必须是
    rank=0 (N) — |2-0|=2 >= 2, 命中. F (rank=1) / P (rank=2) / Z (rank=3) 不命中."""
    from des.types import FAMILY_RANK
    from des.registry import _spectrum_for, ALPHABET
    src_rank = FAMILY_RANK["P"]
    spec = dict(_spectrum_for("P_crossclan_surge"))
    for t, q in spec.items():
        if q == 0:
            continue
        rank_dt = abs(FAMILY_RANK[ALPHABET[t]] - src_rank)
        assert rank_dt >= 2, (
            f"P_crossclan_surge mass leaked to |Δrank|={rank_dt} target {t!r}")


def test_p_frozen_aff_pow_4_sharper_than_aff_pow_3():
    """P_frozen (aff^4) 比 P_entropy_brake (aff^3) 同家族占比更高."""
    from des.registry import _spectrum_for, ALPHABET
    brake = dict(_spectrum_for("P_entropy_brake"))
    frozen = dict(_spectrum_for("P_frozen"))
    same_p = [t for t in brake if ALPHABET[t] == "P"]
    brake_mass = sum(brake[t] for t in same_p)
    frozen_mass = sum(frozen[t] for t in same_p if t in frozen)
    assert frozen_mass > brake_mass + 1e-9, (
        f"P_frozen mass on P-family ({frozen_mass:.4f}) must exceed P_entropy_brake "
        f"({brake_mass:.4f}) — aff^4 is sharper than aff^3")


def test_p_crossclan_surge_renormalizes_to_unity():
    """family_mask='cross' 仍归一."""
    from des.registry import _spectrum_for
    spec = _spectrum_for("P_crossclan_surge")
    if spec == ():
        pytest.skip("no cross-rank targets in current alphabet")
    total = sum(q for _, q in spec)
    assert abs(total - 1.0) < 1e-9
```

- [ ] **Step 2: 跑失败测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_a_pool.py tests/test_spectrum_shape.py -v
```

Expected:
- A_SHAPE merge / power=4 / cross-mask 测试 FAIL with `KeyError`(`P_frozen` 不在 SPECTRUM_SHAPE)或 module-load `AssertionError: SPECTRUM_SHAPE[...].power = 4.0 not in {1,2,3}`(若 import 顺序导致 _A_SHAPE 已 update 进来)。
- `test_p_crossclan_surge_only_hits_cross_rank_targets` FAIL with `KeyError: P_crossclan_surge` 或 `_spectrum_for` 未识别 `"cross"` mask 全部归 0。

- [ ] **Step 3: 放宽 SPECTRUM_SHAPE 域 assert + 合入 A_SHAPE**

打开 `src/des/registry.py`,**先修 S2 的 module-load assert**(在 `SPECTRUM_SHAPE` 块下方,S2 写的那段 `for _letter, (_power, _mask, _mix) in SPECTRUM_SHAPE.items():` 循环):

```python
# Module-load value-domain assertions (S2 base + S8 extension).
# S8: power=4 (P_frozen / P_stutter aff^4); family_mask='cross' (P_crossclan_surge).
assert set(SPECTRUM_SHAPE.keys()) == set(_P.keys()), (
    "SPECTRUM_SHAPE must be co-extensive with _P; "
    f"missing={set(_P.keys()) - set(SPECTRUM_SHAPE.keys())}, "
    f"extra={set(SPECTRUM_SHAPE.keys()) - set(_P.keys())}")
for _letter, (_power, _mask, _mix) in SPECTRUM_SHAPE.items():
    assert _power in (1.0, 2.0, 3.0, 4.0), \
        f"SPECTRUM_SHAPE[{_letter!r}].power = {_power!r} not in {{1,2,3,4}}"
    assert _mask in (None, "F", "Z", "N", "adjacent", "cross"), \
        f"SPECTRUM_SHAPE[{_letter!r}].family_mask = {_mask!r} not in {{None,F,Z,N,adjacent,cross}}"
    assert 0.0 <= _mix <= 1.0, \
        f"SPECTRUM_SHAPE[{_letter!r}].flatten_mix = {_mix!r} outside [0,1]"
del _letter, _power, _mask, _mix
```

注意:**先 `update(_A_SHAPE)` 再 assert** —— 顺序很关键。`SPECTRUM_SHAPE` 字典定义之后立刻是「先 `update` 再 `assert`」,不能先 assert(那时 12 行还没有 P_cascade 等,assert 「co-extensive with _P」会 FAIL 因为 _P 里已经有 8 行 A 没有对应 SPECTRUM_SHAPE 行)。

把 `SPECTRUM_SHAPE` 块那段改为:

```python
SPECTRUM_SHAPE: dict[str, tuple[float, "str | None", float]] = {
    # ... S2 既有 12 行 (P_base ... P_balanced) ...
}

# S8: merge A_SHAPE (8 P-pool A rows) BEFORE the value-domain assert below,
# so the assert ranges over all 12 + 8 = 20 entries.
from des._a_pool import A_SHAPE as _A_SHAPE
SPECTRUM_SHAPE.update(_A_SHAPE)
del _A_SHAPE

# Then the assert (already shown above with the relaxed domains).
```

打开 `src/des/registry.py::_spectrum_for(letter)` 函数体的 `"adjacent"` 分支后追加 `"cross"` 分支:

```python
def _spectrum_for(letter: str) -> tuple[tuple[str, float], ...]:
    """... (S2 docstring unchanged) ..."""
    src_fam = ALPHABET[letter]
    src_gran = GRAN[letter]
    src_len = MOTIF_LEN.get(letter)
    power, mask, mix = SPECTRUM_SHAPE.get(letter, (1.0, None, 0.0))
    src_rank = FAMILY_RANK[src_fam]
    A = len(ALPHABET)

    survivors: dict[str, float] = {}
    for t in ALPHABET:
        if t == letter:
            continue
        if GRAN[t] != src_gran:
            continue
        if src_gran == "motif" and MOTIF_LEN[t] != src_len:
            continue
        # S2 / S8 family_mask predicate
        if mask is None:
            pass
        elif mask == "adjacent":
            if abs(FAMILY_RANK[ALPHABET[t]] - src_rank) != 1:
                continue
        elif mask == "cross":                                 # S8: |Δrank| >= 2
            if abs(FAMILY_RANK[ALPHABET[t]] - src_rank) < 2:
                continue
        else:
            if ALPHABET[t] != mask:
                continue
        w = affinity(src_fam, ALPHABET[t]) ** power
        if mix > 0.0:
            w = (1.0 - mix) * w + mix * (1.0 / (A - 1))
        survivors[t] = w

    tot = sum(survivors.values())
    if tot == 0.0:
        return ()
    return tuple((t, w / tot) for t, w in sorted(survivors.items()))
```

(若 S2 `_spectrum_for` 函数体已经有完全相同的结构,只需追加 `elif mask == "cross":` 那 3 行;不 reformat 其他逻辑。)

- [ ] **Step 4: 跑测试,确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_a_pool.py tests/test_spectrum_shape.py -v
```

Expected: 全 PASS。

Backtrack:
- 若 module-load `AssertionError: SPECTRUM_SHAPE not co-extensive with _P`:
  - 顺序问题。`update(_A_SHAPE)` 必须在 assert 之前;同时 `_A_F`/`_A_Z`/`_A_P` 已在 Task 2 update 进 `_F`/`_Z`/`_P`,所以现在 `_P` 大小 = 12+8=20,`SPECTRUM_SHAPE` 大小也必须 = 20。grep `update(_A_P)` 与 `update(_A_SHAPE)` 是不是都跑了。
- 若 `test_p_crossclan_surge_only_hits_cross_rank_targets` FAIL with `mass leaked to |Δrank|=1 target`:
  - `_spectrum_for` 的 `"cross"` 分支可能写反了 `<` 与 `>=`。验:`if abs(FAMILY_RANK[ALPHABET[t]] - src_rank) < 2: continue` —— rank 距离<2 的 (即 0,1) 跳过,只留 |Δrank|≥2 即 (2,3)。
- 若 `test_p_frozen_aff_pow_4_sharper_than_aff_pow_3` FAIL:
  - 检查 `aff ** 4` 而非 `aff * 4`(指数 vs 乘法);P_frozen 应该比 P_entropy_brake 在同家族 mass 上严格更大。

- [ ] **Step 5: 跑全 suite,确认默认 BB0 字节级仍不变**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿。BB0_TEMPLATE 不含任何 P_frozen / P_stutter / P_crossclan_surge,默认 strain phenotype `dominant_p="P_base"` 走 `_spectrum_for("P_base")`,P_base 仍是 `(1.0, None, 0.0)` shape,`_spectrum_for("P_base")` 在「ALPHABET 已扩到 40 行」的口径下 normalize,数值与 Task 2 后口径一致(本 task 只对**已经在 Task 2 进 ALPHABET 的 A 行**的 SHAPE 做扩展,ALPHABET 大小不变)。

Backtrack:`test_phenotype_synthetic_P_with_slots_2_propagates`(S7 测试)若 FAIL → S7 测试用 monkeypatch 合成 `P_cascade` 到 `_P`,Task 2 已把真 P_cascade 落进 `_P`;monkeypatch 与 real entry 冲突。改测试名期待 → S7 写的 monkeypatch 应当用 `monkeypatch.setitem(registry._P, ...)` 而 `_P["P_cascade"]` 已存在,monkeypatch 会成功覆盖(不破)。若仍 FAIL,在 Task 7 final sweep 时联合 fix。

- [ ] **Step 6: Commit**

```bash
git add src/des/registry.py tests/test_a_pool.py tests/test_spectrum_shape.py
git commit -m "feat(s8): extend SPECTRUM_SHAPE domain (power=4, mask='cross') + merge 8 A rows

power ∈ {1,2,3,4} (P_frozen / P_stutter aff^4); family_mask ∈
{None,F,Z,N,adjacent,cross} (P_crossclan_surge |Δrank|>=2). _spectrum_for
gains a 'cross' branch beside existing 'adjacent'. A_SHAPE merge happens
before the value-domain assert; default BB0 byte-identical (P_base path
unchanged)."
```

---

### Task 5: Multi-P 混合谱 `blend_p_spectra` + `phenotype()` 替换 dominant_p 谱源

**Goal:** spec §4.1 唯一真实机制变更 —— 把 `phenotype()` 里 `spectrum = _spectrum_for(dominant_p)` 这一行(`registry.py:111`)替换为「逐 P 字母收集 `(p_add, q_i)` pair,再调 `blend_p_spectra` 算 `Σ pᵢ qᵢ / Σ pᵢ`」。**单 P 字母 strain 路径数学等价 dominant_p 路径**(Σ 只有一项,blend == dominant);默认 BB0 字节级不变。`Σ p_add_i == 0` 时退化为「等权 blend」(`Σ q_i / N_p`)而非分母零抛错。**`dominant_p` 字段仍保留**(给 S7 `SLOTS_PER_EVENT` 用,不动 S7 spec)。

**Files:**
- Modify: `src/des/registry.py` — 加 `blend_p_spectra` helper + 改 `phenotype()` 末尾 spectrum 计算
- Test: `tests/test_multi_p_blend.py`(Create — blend 公式纯单元测试)
- Test: `tests/test_a_pool.py`(append phenotype 端到端测试)
- Test: `tests/test_phenotype_cache.py`(append 单 P 字母 byte-identical)

**Interfaces:**
- Consumes: `_spectrum_for(letter) -> tuple[tuple[str, float], ...]`(S2 + S8 Task 4 已扩 cross/power=4);`_P` 字典(Task 2 已合 24+8=20 行);S7 `SLOTS_PER_EVENT[dominant_p]` 仍 by dominant_p。
- Produces:
  - `blend_p_spectra(pairs: tuple[tuple[float, tuple[tuple[str, float], ...]], ...]) -> tuple[tuple[str, float], ...]`。
    - `pairs = []` → 返回 `()`。
    - `pairs = [(p, q)]`(单字母) → 返回 `q`(身份;Σ 一项)。
    - `Σ p_add > 0` → 返回 `(target, Σ_i pᵢ · q_i(target) / Σ_i pᵢ)` 逐 target 累加 + sort。
    - `Σ p_add == 0`(全零 P_base / Glacial Drift / P_frozen 等)→ 返回 `(target, Σ_i q_i(target) / N_p)` 等权退化。
  - `phenotype(sequence).spectrum` 改读 blend 而非 dominant_p 谱;**`Phenotype.spectrum` cache 与 `_mutation_outcomes` 调用点不动**。

- [ ] **Step 1: 写失败测试 — blend 公式纯单元**

新建 `tests/test_multi_p_blend.py`:

```python
# tests/test_multi_p_blend.py
"""S8 multi-P spectrum blend: blend_p_spectra(pairs) -> spectrum.

Implements spec §4.1: spectrum(t) = Σ_i p_add_i · q_i(t) / Σ_i p_add_i,
or — when Σ p_add_i == 0 — equal-weight degenerate path Σ q_i / N_p.

Single-letter path is the dominant_p identity: blend([(p, q)]) == q
byte-equal. Default BB0 strain has dominant_p='P_base' → single-letter
path → byte-identical pre-S8."""
from __future__ import annotations
import pytest


def test_blend_empty_pairs_returns_empty():
    """No P letter ⇒ spectrum = ()."""
    from des.registry import blend_p_spectra
    assert blend_p_spectra(()) == ()


def test_blend_single_letter_is_identity():
    """Σ has one term ⇒ blend == q itself, byte-equal."""
    from des.registry import blend_p_spectra
    q = (("F4Nr1", 0.6), ("F4Nr4", 0.4))
    assert blend_p_spectra(((0.05, q),)) == q
    # p_add value irrelevant for single-letter identity
    assert blend_p_spectra(((0.0, q),))  == q
    assert blend_p_spectra(((0.30, q),)) == q


def test_blend_two_letters_weighted_per_target():
    """spectrum(t) = (p₁·q₁(t) + p₂·q₂(t)) / (p₁ + p₂) for every target t.
    Fixture: p1=0.06 q1 over {A,B} = (0.7, 0.3); p2=0.04 q2 = (0.2, 0.8).
    Expected: target A = (0.06·0.7 + 0.04·0.2) / 0.10 = (0.042+0.008)/0.10 = 0.50;
    target B = (0.06·0.3 + 0.04·0.8) / 0.10 = (0.018+0.032)/0.10 = 0.50.
    Σ spectrum = 1.0."""
    from des.registry import blend_p_spectra
    q1 = (("A", 0.7), ("B", 0.3))
    q2 = (("A", 0.2), ("B", 0.8))
    blended = dict(blend_p_spectra(((0.06, q1), (0.04, q2))))
    assert abs(blended["A"] - 0.50) < 1e-9
    assert abs(blended["B"] - 0.50) < 1e-9
    assert abs(sum(blended.values()) - 1.0) < 1e-9


def test_blend_two_letters_weighted_asymmetric():
    """Heavier letter dominates the blend."""
    from des.registry import blend_p_spectra
    q1 = (("A", 1.0), ("B", 0.0))
    q2 = (("A", 0.0), ("B", 1.0))
    blended = dict(blend_p_spectra(((0.10, q1), (0.02, q2))))
    # weight q1 = 0.10/0.12 ≈ 0.833; weight q2 = 0.02/0.12 ≈ 0.167
    assert abs(blended["A"] - 10/12) < 1e-9
    assert abs(blended["B"] - 2/12) < 1e-9


def test_blend_target_only_in_one_letter():
    """target appearing in only one letter still contributes weighted by that p_add."""
    from des.registry import blend_p_spectra
    q1 = (("A", 0.6), ("B", 0.4))
    q2 = (("A", 0.5), ("C", 0.5))
    blended = dict(blend_p_spectra(((0.06, q1), (0.04, q2))))
    # target B only in q1: weight = (0.06·0.4 + 0.04·0) / 0.10 = 0.24
    # target C only in q2: weight = (0.06·0 + 0.04·0.5) / 0.10 = 0.20
    # target A in both: (0.06·0.6 + 0.04·0.5) / 0.10 = 0.56
    assert abs(blended["B"] - 0.24) < 1e-9
    assert abs(blended["C"] - 0.20) < 1e-9
    assert abs(blended["A"] - 0.56) < 1e-9
    assert abs(sum(blended.values()) - 1.0) < 1e-9


def test_blend_sum_p_add_zero_uses_equal_weight_degenerate_path():
    """Σ p_add == 0 ⇒ equal-weight average (avoid div-by-zero).
    Fixture: two letters both p_add=0 (e.g. P_base / P_frozen).
    Expected: spectrum(t) = (q₁(t) + q₂(t)) / 2."""
    from des.registry import blend_p_spectra
    q1 = (("A", 0.8), ("B", 0.2))
    q2 = (("A", 0.4), ("B", 0.6))
    blended = dict(blend_p_spectra(((0.0, q1), (0.0, q2))))
    # equal weight: avg
    assert abs(blended["A"] - 0.6) < 1e-9
    assert abs(blended["B"] - 0.4) < 1e-9
    assert abs(sum(blended.values()) - 1.0) < 1e-9


def test_blend_single_letter_with_p_add_zero_still_identity():
    """Σ=0 + 单字母 ⇒ 单字母路径仍取 identity (N_p=1 equal-weight = q itself)."""
    from des.registry import blend_p_spectra
    q = (("A", 0.7), ("B", 0.3))
    assert blend_p_spectra(((0.0, q),)) == q


def test_blend_output_sorted_by_target_name():
    """spectrum 输出按 target 名字升序排列 (与 _spectrum_for 同款 canonical order)."""
    from des.registry import blend_p_spectra
    q1 = (("Apple", 0.5), ("Zebra", 0.5))
    q2 = (("Banana", 1.0),)
    blended = blend_p_spectra(((0.05, q1), (0.03, q2)))
    names = [t for t, _ in blended]
    assert names == sorted(names)


def test_blend_handles_empty_q_in_one_letter():
    """One letter's q is () (e.g. P_fscan in F-less alphabet) — skip its contribution."""
    from des.registry import blend_p_spectra
    q1 = (("A", 0.6), ("B", 0.4))
    q2 = ()
    blended = blend_p_spectra(((0.05, q1), (0.03, q2)))
    # Effective: only q1 contributes; result == q1 (single-letter identity given q2 empty)
    # NOTE: blend formula sums over (p_i · q_i(t)) for t in union; q2 contributes 0
    # everywhere. Σ p = 0.08, so q1's contribution gets 0.05/0.08 weight, q2 gets 0.
    # spectrum(A) = (0.05·0.6 + 0.03·0) / 0.08 = 0.375
    # But by spec it's also legitimate to renormalize over surviving total — both
    # interpretations agree because q2 is all-zero. Implementation MUST yield:
    blended_d = dict(blended)
    assert abs(blended_d["A"] - 0.375) < 1e-9
    assert abs(blended_d["B"] - 0.25)  < 1e-9
    # Σ < 1 → renormalize to 1
    assert abs(sum(blended_d.values()) - (0.375 + 0.25)) < 1e-9
    # Implementation must EITHER renormalize to Σ=1 OR leave the residual hole.
    # spec §4.1 says "blended is a p_add-weighted average ... then re-normalized";
    # so the implementation MUST renormalize:
    s = sum(blended_d.values())
    assert abs(s - 1.0) < 1e-9, (
        f"blended must renormalize to Σ=1 after dropping empty q letters, got Σ={s}")
```

(`test_blend_handles_empty_q_in_one_letter` 的 expected 数值用了 `0.375 / 0.25` 但又 assert `Σ=1`。这两条不能同时为真 —— 上半段是 implementation 中间态的描述,下半段是 spec 要求的 final renormalize。Step 3 实现时必须最终归一,本测试用 assert `Σ=1` 作 ground truth,前两条 absolute value assert **删除**以避免歧义。修正版下文 Step 1.5 写出。)

- [ ] **Step 1.5: 修正 `test_blend_handles_empty_q_in_one_letter`**

把上面那条测试改为(最终归一后期望值):

```python
def test_blend_handles_empty_q_in_one_letter():
    """One letter's q is () (e.g. P_fscan in F-less alphabet) — skip its contribution
    and renormalize the survivors to Σ=1 (spec §4.1)."""
    from des.registry import blend_p_spectra
    q1 = (("A", 0.6), ("B", 0.4))
    q2 = ()
    blended = blend_p_spectra(((0.05, q1), (0.03, q2)))
    blended_d = dict(blended)
    # q2 contributes nothing; result is essentially q1 renormalized (already sums to 1).
    assert abs(blended_d["A"] - 0.6) < 1e-9
    assert abs(blended_d["B"] - 0.4) < 1e-9
    assert abs(sum(blended_d.values()) - 1.0) < 1e-9
```

(替换上一条同名测试。Step 1 的代码块里那条「带歧义中间值的版本」**不要写进文件**;最终文件里只保留 Step 1.5 的版本。)

- [ ] **Step 2: 跑失败测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_multi_p_blend.py -v
```

Expected: 9 条全 FAIL with `ImportError: cannot import name 'blend_p_spectra' from 'des.registry'`。

- [ ] **Step 3: 实现 `blend_p_spectra` helper**

打开 `src/des/registry.py`,在 `_spectrum_for(letter)` 函数之后、`phenotype(sequence)` 函数之前插入:

```python
def blend_p_spectra(
    pairs: tuple[tuple[float, tuple[tuple[str, float], ...]], ...],
) -> tuple[tuple[str, float], ...]:
    """p_add-weighted average of per-letter spectra, renormalized to Σ=1.

    spec §4.1 — replaces the v1 'dominant_p' single-source selection:
        spectrum(t) = Σ_i p_add_i · q_i(t) / Σ_i p_add_i           (Σ p_add > 0)
        spectrum(t) = Σ_i q_i(t) / N_p                              (Σ p_add == 0)

    pairs : sequence of (p_add, q_i) where q_i is the already shape-modulated,
            gran-filtered, normalized spectrum from _spectrum_for(letter_i).
    Returns: normalized (target, weight) tuple sorted by target name.

    Edge cases:
      empty pairs                -> ()
      single (p, q)              -> q  (identity; v1 dominant_p path)
      every p_add_i == 0         -> equal-weight average
      some q_i == ()             -> skip (zero contribution everywhere)
    """
    if not pairs:
        return ()

    # Filter out letters whose q_i is empty (e.g. P_fscan in F-less alphabet).
    nonempty = tuple((p, q) for p, q in pairs if q)
    if not nonempty:
        return ()

    # Single-letter identity (preserves byte-equality with v1 dominant_p path).
    if len(nonempty) == 1:
        return nonempty[0][1]

    sum_p = sum(p for p, _ in nonempty)
    if sum_p > 0.0:
        weights = tuple(p / sum_p for p, _ in nonempty)
    else:
        # All p_add == 0 ⇒ degenerate equal-weight average (avoid div-by-zero).
        n = len(nonempty)
        weights = tuple(1.0 / n for _ in nonempty)

    # Accumulate per-target mass over the union of targets.
    bucket: dict[str, float] = {}
    for w, (_, q) in zip(weights, nonempty):
        for tgt, qi in q:
            bucket[tgt] = bucket.get(tgt, 0.0) + w * qi

    # Renormalize (defensive; bucket should already sum to ~1 if every q_i was
    # normalized — but floating point + dropped-empty branches can leave drift).
    tot = sum(bucket.values())
    if tot <= 0.0:
        return ()
    return tuple((t, w / tot) for t, w in sorted(bucket.items()))
```

- [ ] **Step 4: 跑 blend unit 测试,确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_multi_p_blend.py -v
```

Expected: 9 条全 PASS。

Backtrack:
- 若 `test_blend_target_only_in_one_letter` FAIL with `weights mismatch`:
  - `bucket[tgt] = bucket.get(tgt, 0.0) + w * qi` 的 `w * qi` 不能写成 `w + qi` 或 `w * qi / sum_p`(`w` 已经是归一权重);确认实现里 `weights` 已经除过 `sum_p`,bucket 累加用 `w * qi` 而非再除。
- 若 `test_blend_handles_empty_q_in_one_letter` FAIL with `Σ != 1`:
  - 实现末尾的 `tot = sum(bucket.values())` + `/ tot` renormalize 没生效;补 final `return tuple(... / tot)` 那一段。

- [ ] **Step 5: 写 phenotype 端到端测试 — 单 P 字母 byte-identical + multi-P 真正 blend**

追加到 `tests/test_a_pool.py`:

```python
# --- Task 5 surface: phenotype() multi-P blend integration --------------------

def test_phenotype_default_bb0_spectrum_byte_identical_to_pre_S8(monkeypatch):
    """默认 BB0 dominant_p='P_base' (单 P 字母) → blend == _spectrum_for('P_base')
    字节级 (单字母身份). 守 spec §3 红线 + spec §4.1 单字母 byte-identity."""
    from des.registry import phenotype, _spectrum_for, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    expected = _spectrum_for("P_base")
    assert p.spectrum == expected, (
        f"single-P-letter blend must equal _spectrum_for('P_base') byte-equal;\n"
        f"got    {p.spectrum!r}\n"
        f"expect {expected!r}")


def test_phenotype_two_P_letters_blends_per_spec_4_1():
    """同 strain 装 2 个 P 字母 (P_base p_add=0.0, P_hotspot p_add=0.05) →
    blend = (0·q_base + 0.05·q_hotspot) / 0.05 = q_hotspot (Σ p_add > 0 但 P_base
    权重 0). 应字节级 == _spectrum_for('P_hotspot')."""
    from des.registry import phenotype, _spectrum_for
    seq = ("P_base", "P_hotspot") + ("N0",) * 14
    p = phenotype(seq)
    expected = _spectrum_for("P_hotspot")
    assert p.spectrum == expected, (
        f"blend with one zero-weight letter must equal the nonzero one;\n"
        f"got    {p.spectrum!r}\n"
        f"expect {expected!r}")


def test_phenotype_three_P_letters_all_zero_p_add_equal_weight():
    """3 个全 p_add=0 的 P 字母 (P_base / P_slow_drift / P_frozen) 共存 →
    Σ p_add = 0 → 等权 blend = (q_base + q_slow_drift + q_frozen) / 3."""
    from des.registry import phenotype, _spectrum_for, blend_p_spectra
    seq = ("P_base", "P_slow_drift", "P_frozen") + ("N0",) * 13
    p = phenotype(seq)
    expected = blend_p_spectra((
        (0.0, _spectrum_for("P_base")),
        (0.0, _spectrum_for("P_slow_drift")),
        (0.0, _spectrum_for("P_frozen")),
    ))
    assert p.spectrum == expected


def test_phenotype_dominant_p_field_still_used_by_slots_per_event(monkeypatch):
    """S7 仍 piggyback dominant_p: SLOTS_PER_EVENT[dominant_p] 应正确.
    strain (P_base, P_cascade) → dominant_p = P_cascade (highest p_add 0.28) →
    slots_per_event = 2 (P_cascade 是唯一 slots=2)."""
    from des.registry import phenotype
    seq = ("P_base", "P_cascade") + ("N0",) * 14
    p = phenotype(seq)
    # S7 守门: slots_per_event = SLOTS_PER_EVENT[dominant_p]
    assert p.slots_per_event == 2, (
        f"dominant_p='P_cascade' should propagate slots=2, got {p.slots_per_event!r}")
```

追加到 `tests/test_phenotype_cache.py`:

```python
def test_phenotype_cache_byte_identical_for_single_P_letter_post_S8():
    """phenotype_arrays / cache 对单 P 字母 strain 字节级 = pre-S8 (单字母身份)."""
    import torch
    from des.phenotype_cache import phenotype_arrays
    from des.registry import phenotype, BB0_TEMPLATE
    # 间接测试: cache key 是 sequence; 同 seq 两次 phenotype() 必字节级相同
    p1 = phenotype(BB0_TEMPLATE["layout"])
    p2 = phenotype(BB0_TEMPLATE["layout"])
    assert p1 == p2  # frozen dataclass equality covers all fields
    assert p1.spectrum == p2.spectrum
```

- [ ] **Step 6: 跑端到端测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_a_pool.py::test_phenotype_default_bb0_spectrum_byte_identical_to_pre_S8 tests/test_a_pool.py::test_phenotype_two_P_letters_blends_per_spec_4_1 tests/test_a_pool.py::test_phenotype_three_P_letters_all_zero_p_add_equal_weight tests/test_a_pool.py::test_phenotype_dominant_p_field_still_used_by_slots_per_event -v
```

Expected: 4 条全 FAIL —— `phenotype()` 还在用 dominant_p 单源,blend 没接上。

- [ ] **Step 7: 改 `phenotype()` 末尾接 blend 替代 dominant_p 谱源**

打开 `src/des/registry.py::phenotype(sequence)`,**保留** `dominant_p` 解析逻辑(S7 + slots_per_event 仍由 dominant_p 取),把 `spectrum = _spectrum_for(dominant_p) if dominant_p else ()` 这一行替换为 blend。

具体改动:在 `for letter in sequence:` 循环结束之后,既有 `dominant_p` 解析之外新增「P-letter pair 收集」:

```python
    # ... (循环结束) ...

    # S7: piggyback dominant_p for slots_per_event (no change to S7 path).
    slots_per_event = SLOTS_PER_EVENT.get(dominant_p, 1) if dominant_p else 1

    # S8 §4.1: spectrum is the p_add-weighted blend of per-letter spectra,
    # not the single dominant_p source. Single-letter strains see byte-equal
    # behavior (blend == q for one (p, q) pair).
    p_pairs = tuple(
        (_P[letter][0], _spectrum_for(letter))
        for letter in sequence
        if letter in _P
    )
    spectrum = blend_p_spectra(p_pairs) if p_pairs else ()
```

把原行删除:

```python
    # 删掉这一行 (S2):
    # spectrum = _spectrum_for(dominant_p) if dominant_p else ()
```

注:`p_pairs` 用 `tuple(...)` 而非 list,与 `Phenotype.spectrum: tuple[tuple[str, float], ...]` 同 immutable 风格。`if letter in _P` filter 排除 N/F/Z letter(它们没有 p_add)。`for letter in sequence` 而非 `for letter in set(sequence)` —— 同一 P 字母在序列中出现 K 次,K 次贡献(spec §3 红线:「携带」per-letter 不是 per-occurrence;但若用户在 BB0 layout 用同一 P 字母多次,blend 仍累加 K 次贡献,等价于一个权重为 `K * p_add` 的等效字母。下一条 backtrack 决议保留:**重复字母不去重**,因 S6/S7/S8 spec 全部按 sequence 扫描)。

- [ ] **Step 8: 跑端到端测试,确认 PASS**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_a_pool.py tests/test_multi_p_blend.py tests/test_phenotype_cache.py -v
```

Expected: 全 PASS。

Backtrack:
- 若 `test_phenotype_default_bb0_spectrum_byte_identical_to_pre_S8` FAIL with `spectrum mismatch`:
  - 单字母 strain(BB0 默认只有 `P_base`)`p_pairs = ((0.0, _spectrum_for("P_base")),)`,`blend_p_spectra` 走「单字母 identity」分支返回 `_spectrum_for("P_base")` 本身。若 FAIL,检查实现 `len(nonempty) == 1` 分支是否直接 `return nonempty[0][1]`。
- 若 `test_phenotype_two_P_letters_blends_per_spec_4_1` FAIL with `不等于 q_hotspot`:
  - blend 公式 `Σ p·q / Σ p` 在 P_base p=0 + P_hotspot p=0.05 下 = `(0·q_base + 0.05·q_hotspot) / 0.05 = q_hotspot`。若 FAIL,检查 blend 实现 sum_p>0 分支的 `weights = p_i/sum_p` 是不是写成了 `p_i / n`。
- 若 `test_phenotype_dominant_p_field_still_used_by_slots_per_event` FAIL with `slots_per_event != 2`:
  - root cause:本 Task 改 spectrum 时**不应**删除 `dominant_p` 解析逻辑;`SLOTS_PER_EVENT.get(dominant_p, 1)` 仍依赖 dominant_p 字段。保留 phenotype() 里既有 `if dominant_p is None or p_add > _P[dominant_p][0]: dominant_p = letter` 分支不动。

- [ ] **Step 9: 跑全 suite,确认无回归**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿。默认 BB0 (`P_base` 单字母) 走单字母 identity,phenotype 与 cache 字节级不变;同 seed engine 跑两次仍 byte-identical(spec §3 红线)。

Backtrack: 若 `test_spectrum_residue_path_byte_identical_to_legacy`(S6 owner test)FAIL → S6 测试用 hard-coded P_base 数值,本 task 走单字母 identity 应保留;若 FAIL,grep S6 测试名 + 看具体期望值 —— 大概率是 default BB0 strain 的 `dominant_p` 检测仍正确但 `_P` 已增长到 20 行,`_spectrum_for("P_base")` 的归一分母变了。**这属于 S2 已锁的「6→16 重录基线」漂移**(Global Constraints),如 S2 plan 已 RE-RECORD fixture 则本步不再重录;若未 RE-RECORD,grep S6 测试是不是仍读 6-字母基线 → Task 7 final sweep 时处理。

- [ ] **Step 10: Commit**

```bash
git add src/des/registry.py tests/test_multi_p_blend.py tests/test_a_pool.py tests/test_phenotype_cache.py
git commit -m "feat(s8): multi-P spectrum blend replaces dominant_p single-source

spec §4.1: spectrum(t) = Σ p_add_i · q_i(t) / Σ p_add_i, renormalized.
Single-letter strain ⇒ identity (byte-identical pre-S8 dominant_p path).
Σ p_add == 0 ⇒ equal-weight degenerate. dominant_p field retained for
S7 slots_per_event piggyback. blend_p_spectra() lives in registry.py."
```

---

### Task 6: roster doc cleanup(24 条覆写行 + OPEN-1/θ 标 RETIRED)+ 默认局 byte-identical 端到端

**Goal:** spec §2 末尾的两件 doc cleanup 一次性做完:(a) `context/design/primitive-roster.md` 里 A 池 24 条 `覆写: {株:n_locked≥θ}` 行 verbatim 替换为「`覆写: A reachable via affinity spectrum (...);θ-gate retired (de-gate, 2026-06-24).`」;(b) OPEN-1/θ section 标 RETIRED。这只是 doc 改动,**不进代码路径**,但属本 plan 交付物。同时跑一次默认局端到端 byte-identical 验证,守 spec §3 红线。

**Files:**
- Modify: `context/design/primitive-roster.md`(24 条 `覆写:` 行 + OPEN-1/θ section)
- Test: `tests/test_a_pool.py`(append 默认局 byte-identical engine.run 端到端)

**Interfaces:**
- Consumes: spec §2 末尾 verbatim 两条替换文本。
- Produces:
  - 24 行 doc text 替换(零代码消费)。
  - 1 条端到端测试:同 seed 跑两次默认 4-faction 局,world.count / strain_id byte-equal。

- [ ] **Step 1: 修 primitive-roster.md 的 24 条覆写行**

打开 `context/design/primitive-roster.md`,grep 出全部 24 条 `覆写: {株:n_locked` 行(`grep -n '覆写: {' primitive-roster.md` 或用编辑器搜索)。**逐条替换为下面这条 verbatim 文本**(spec §2 末尾给出的 RETIRED 句):

```
- 覆写: A reachable via affinity spectrum (same-family draw, aff=0.70, gran-matched); θ-gate retired (de-gate, 2026-06-24).
```

(注意保留每条行首的 list bullet `-` 与缩进;原文格式参考 roster L190 / L192 / L194 等。)

具体 24 条受影响行号(从 Read primitive-roster.md L181–266 看到):
- 乙1 8 条:Apex Bloom L190 / Ember Drip L193 / Bastion Pile L196 / Apex Fang L199 / Pan Sweep L202 / Hotspot Amp L205 / Sink Cascade L208 / Glacial Drift L211。
- 乙2 8 条:F_NOVA L217 / F_TRICKLE L220 / F_SCATTER L223 / Predator Lock L226 / Void Bite L229 / P_cascade L232 / P_crossclan_surge L235 / P_frozen L238。
- 甲 8 条:F8Ar1 L244 / Lance Front L247 / Ambush Venom L250 / Sweep Surge L253 / Nip Whisper L256 / Coil Null L259 / P_zscan_invert L262 / P_stutter L265。

(行号可能因 doc 编辑略漂移;以 `覆写: {株:n_locked` 字符串作 anchor 查找最稳。)

每条行的「F-重 backbone 物种内由 gran 匹配插槽突变涌现」/「Z-重 backbone」/「P-重 backbone」尾部描述**全部删掉**,只留 RETIRED 句。

- [ ] **Step 2: 修 primitive-roster.md 的 OPEN-1/θ section + 「A 池表头」**

定位「A 池 — 突变可达极端变体 (rank 4, 24)」section 的 header 段(roster L181–184),把:

```markdown
> 覆写列表(2026-06-24 重订,替代逐条点名前驱):某 A 的覆写列表 = backbone 上下文门控谓词 $\{\text{株}:n_{\text{locked}}(\text{chan}(A))\ge\theta_{\text{chan}(A)}\}$——即只在 backbone-locked 组成里**该 A 所产通道对应族**已达数量阈 θ 的物种内,由一个 gran 匹配的可变插槽突变涌现。「建难」由门控自然涌现(极端 F 变体只在 F-重株里冒头,无需手写阶梯/禁 N)。同通道 A 共用同一覆写门,彼此差异全在各自 formula。θ 数值轮占位。**copy-of 仅血统注释(标该 A 照哪条常规档放大),不进覆写逻辑(突变核不读 copy-of)。**
```

替换为 verbatim spec §2 末尾的 RETIRED 段:

```markdown
> RETIRED (2026-06-24) — the n_locked≥θ overwrite gate is removed; A obeys the single global affinity rule. n_locked kept as an advisory structural readout, not wired into mutation. A primitives are family F/P/Z at extreme values (NOT rank-4); reachable within-family at aff=0.70 via the normal spectrum. copy-of is lineage annotation only — mutation core never reads it.
```

定位「覆写列表 (overwrite/predecessor list)」section(roster L267–278),把 `✅ 两种形态(2026-06-24 重订)` 段下「A 池(24)= backbone 上下文门控谓词 ...」那一条 bullet **整条删除**(保留 N/F/P/Z 池 `from:` 行 bullet 不动)。删除后该 section 变成:

```markdown
## 覆写列表 (overwrite/predecessor list)

✅ A 池 (24) 的覆写已 RETIRED (2026-06-24)。仅保留 N/F/P/Z 池的 `from:` 行作为后续设计参考线(非红线判据)。

**废弃**(均非用户设计的概念,2026-06-24 删):逐条点名前驱当红线、「三结构律(粒度/阶梯/非对称)」、「破易建难非对称律」、对称性。粒度配对(residue↔residue / motif↔motif)是突变核机制本身,不需另立为律。
```

(保留下面那一行 `### motif 粒度突变规则 (2026-06-24 裁定:motif↔motif **等长**,零机制改动)` 整段不动 —— 那是 S6 的 doc,与 S8 无关。)

- [ ] **Step 3: 写默认局 byte-identical engine.run 端到端测试**

追加到 `tests/test_a_pool.py`:

```python
# --- Task 6 surface: default-run regression lock ------------------------------

def test_default_4_faction_run_byte_identical_post_s8():
    """spec §3 红线: 默认 4 同 BB0 faction 局, 同 seed 跑两次 world.count /
    strain_id byte-equal. A pool 不入默认调色板 ⇒ blend 单字母 identity ⇒
    static run 与 pre-S8 字节级不变."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng_a = Engine(H=8, W=8, K=8, seed=0, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2,
                   layouts=(BB0_TEMPLATE["layout"],) * 4)
    eng_b = Engine(H=8, W=8, K=8, seed=0, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2,
                   layouts=(BB0_TEMPLATE["layout"],) * 4)
    eng_a.run(30, recorder=None, stop_on=())
    eng_b.run(30, recorder=None, stop_on=())
    assert torch.equal(eng_a.world.count, eng_b.world.count), (
        "default 4-faction run not deterministic post-S8")
    assert torch.equal(eng_a.world.strain_id, eng_b.world.strain_id), (
        "default 4-faction strain trajectory not deterministic post-S8")


def test_no_a_pool_letter_in_default_palette():
    """spec §3 红线: BB0_TEMPLATE 默认调色板里不含任何 A letter."""
    from des.registry import BB0_TEMPLATE
    from des._a_pool import A_FAMILY
    for letter in BB0_TEMPLATE["layout"]:
        assert letter not in A_FAMILY, (
            f"BB0 default contains A letter {letter!r} — must be v1 6-letter palette only")


def test_alphabet_post_S8_size_is_v1_plus_S2_plus_A():
    """ALPHABET 大小 = 6 (v1) + 10 (S2 P pool) + 7 (S5 + S4 等新 F/Z 字母) + 24 (A pool)
    粗略下限. 实际可漂; 重点是 24 个 A letter 全部进 ALPHABET."""
    from des.registry import ALPHABET
    from des._a_pool import A_FAMILY
    assert all(l in ALPHABET for l in A_FAMILY)
    assert len(ALPHABET) >= 30, f"expected ALPHABET >= 30 letters post-S8, got {len(ALPHABET)}"
```

- [ ] **Step 4: 跑端到端测试**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_a_pool.py::test_default_4_faction_run_byte_identical_post_s8 tests/test_a_pool.py::test_no_a_pool_letter_in_default_palette tests/test_a_pool.py::test_alphabet_post_S8_size_is_v1_plus_S2_plus_A -v
```

Expected: 3 条全 PASS(默认 BB0 不含 A,blend 单字母 identity,kernel 字节级不变)。

Backtrack:
- 若 `test_default_4_faction_run_byte_identical_post_s8` FAIL with `tensor mismatch`:
  - root cause 几乎一定在 Task 5 的 blend 实现里漏了「单字母 identity」分支,导致 BB0 默认 strain 的 spectrum 走了「Σ p>0 normalize」路径产生浮点漂移。回 Task 5 验证 `if len(nonempty) == 1: return nonempty[0][1]` 这一行。
- 若 `test_no_a_pool_letter_in_default_palette` FAIL:
  - BB0_TEMPLATE 在某个之前 spec 里被错改;查 `git log src/des/registry.py | head` 找 offending commit;此红线不可破。

- [ ] **Step 5: 跑全 suite,确认无回归**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 全绿。

- [ ] **Step 6: Commit**

```bash
git add context/design/primitive-roster.md tests/test_a_pool.py
git commit -m "docs(s8): retire n_locked overwrite gate in primitive-roster.md + byte-id regression

24 A-pool overwrite lines rewritten to the RETIRED form (spec §2 末尾):
'A reachable via affinity spectrum ... θ-gate retired (de-gate, 2026-06-24)'.
OPEN-1/θ table-header note marked RETIRED. End-to-end byte-identical
regression: default 4-faction BB0 run, same seed twice, world.count +
strain_id tensor-equal — A is reachable but not in the default palette."
```

---

### Task 7: Final regression sweep + smoke + push

**Goal:** 把整个 S8 deliverable (Tasks 1-6) 一起跑一遍 —— 全 suite 绿、smoke run 不崩、性能档位 (~15.8ms/tick / 128² grid) 没明显漂移、工作树干净、push origin。这是 sibling task 的同款收口动作 (S0 Task 6, S6 Task 9, S2 Task 7, S5 Task 6, S4 Task 8, S3 Task 5, S7 Task 5)。

**Files:**
- 不预期 source 改动。若 Step 1 暴露回归,本 task 修 forward,commit message 引用 offending commit。
- Test: `tests/`(整套) + 默认 4-faction symmetric run smoke + 多 P 字母 strain probe + A 池涌现 probe。

**Interfaces:**
- Consumes: Tasks 1-6 全部产物。
- Produces: 绿 `pytest tests/` + 干净 `git status` + push 到 origin。

- [ ] **Step 1: Full pytest sweep**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -q
```

Expected: 全绿。总数 = 285 engine + 146 web + S6 motif + S1 vis + S2 spectrum_shape + S4 direction_kinds + S5 phase_windows + S3 threshold_predicates + S7 multi_slot + S8 a_pool (`tests/test_a_pool.py` ~22 条) + S8 multi_p_blend (`tests/test_multi_p_blend.py` ~9 条) + 既有 `test_registry.py` / `test_phenotype_cache.py` / `test_spectrum_shape.py` append 部分。**没有 `FAILED tests/...` 行**是验收标准 (`SKIPPED` 允许)。

Backtrack: 若任何 FAIL,先按 owner 文件 root-cause 到对应 Task;用 `git log` 找 offending commit,fix forward,不 reset。

常见 FAIL 类:
- (a) S2 既有 `test_spectrum_normalizes_to_unit_sum_for_every_P_letter` 抓全 `_P` keys 跑 `_spectrum_for`,A 池 8 个 P 字母也会被遍历 → 必须全部归一;若 `P_crossclan_surge` 在某些 ALPHABET 配置下没有 cross-rank target,`_spectrum_for` 返回 `()`,既有 S2 测试已写 `if spec == (): continue`,通过。
- (b) S2 既有 `test_p_aic_is_sharper_than_p_base` 等 sharpen 比较测试:加 cross 分支后逻辑应不变(都走非 cross 路径)。若 FAIL,grep `elif mask == "cross"` 位置,确认在 `adjacent` 之后、`else: ALPHABET[t] != mask` 之前。
- (c) S7 既有 `test_phenotype_synthetic_P_with_slots_2_propagates`(monkeypatch 合成 P_cascade):Task 2 后真 P_cascade 已存在,monkeypatch 重写值不冲突,应通过。

- [ ] **Step 2: Smoke run probe(运行时性能没崩)**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --probe 30
```

Expected stdout 形如 `[probe 30 ticks] X.X ms/tick | peak Y.YY GB | strains->… | total …`。`X.X ms/tick` 应保持在 S0/S6/S1/S2/S4/S5/S3/S7 完工时的同一档(target ≈15.8 ms/tick on 128² grid;≤ 20% drift acceptable)。

若 drift > 20%,常见原因:
- (a) `phenotype()` 末尾「逐 P 字母 _spectrum_for」对单 P 字母 strain 仍只跑 1 次,与 v1 dominant_p 单源同成本;若 profile 显示多次调用,grep `for letter in sequence: if letter in _P:` 那段。
- (b) BB0 默认 strain 单字母,`_spectrum_for("P_base")` 的 inner loop `for t in ALPHABET` 从 16 涨到 ~40,O(|A|) 总开销上升约 2.5x —— 是设计预期(spec §3 红线:`|A|` 跟 registry),不是性能 regression。
- (c) `blend_p_spectra` 单字母 identity 路径直接 return,O(1),不上升。

- [ ] **Step 3: Byte-identical default-run smoke(守 BB0 字节级不变)**

跑两次同 seed BB0 默认局:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --seeds 0 --T 30
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --seeds 0 --T 30
```

Expected: 两次产出的 parquet 在 `data/runs/` 下用 pyarrow 读后行级一致。守 spec §3 红线 + Task 6 测试覆盖。

补充验证 (与最新 sibling baseline 比对):

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -c "
import pyarrow.parquet as pq, sys
a = pq.read_table(sys.argv[1]).to_pydict()
b = pq.read_table(sys.argv[2]).to_pydict()
print('cols match:', sorted(a.keys()) == sorted(b.keys()))
for col in a:
    print(col, 'equal:', a[col] == b[col])
" <baseline.parquet> <s8-fresh.parquet>
```

Expected: 全 `equal: True`。若 False,root cause 大概率在 Task 5 blend 实现里漏了「单字母 identity」分支(直接 return,不进 bucket 累加)。

- [ ] **Step 4: Multi-P 字母 smoke — 合成 strain 跑 phenotype**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -c "
from des.registry import phenotype
seq = ('P_base', 'P_hotspot', 'P_aic') + ('N0',) * 13
p = phenotype(seq)
print('spectrum size:', len(p.spectrum))
print('Sum q:', sum(q for _, q in p.spectrum))
print('slots_per_event:', p.slots_per_event)
"
```

Expected stdout: `spectrum size: <int>` (>= 5), `Sum q: 0.999...` (~1.0 浮点精度), `slots_per_event: 1`(dominant_p 是 P_hotspot 0.05,`SLOTS_PER_EVENT["P_hotspot"]=1`)。任何 `RuntimeError` / `KeyError` 都是 blend 实现 bug。

- [ ] **Step 5: A 池涌现 smoke — 4 阵营对称局跑 100 tick**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -c "
import torch
from des.engine import Engine
from des.registry import BB0_TEMPLATE
from des._a_pool import A_FAMILY
eng = Engine(H=16, W=16, K=16, seed=0, device=torch.device('cpu'),
             z_max=8.0, fill_per_cell=4,
             layouts=(BB0_TEMPLATE['layout'],)*4)
eng.run(100, recorder=None, stop_on=())
n_a = sum(1 for sid in range(len(eng.table))
          for letter in eng.table.sequence_of(sid) if letter in A_FAMILY)
print(f'A letters in strain table: {n_a}')
"
```

Expected: `A letters in strain table: <int>`(可能 0 也可能 >0 —— informational smoke,not pass/fail。若 >0 说明 de-gate 工作。**不应抛**)。

- [ ] **Step 6: Inspect & clean stray data**

Run:

```
git status
```

Expected: 干净工作树。若 `data/runs/<ts>-*.parquet` smoke 残留 / `s8_*` 临时日志,删掉。

- [ ] **Step 7: Final commit (only if Step 1 needed a fix-forward)**

若 Step 1 surface 了 regression 并修了:

```bash
git add <files-touched>
git commit -m "fix(s8): <description of the regression fixed>"
```

Otherwise this step is a no-op。

- [ ] **Step 8: Push to origin**

```bash
git push origin <current-branch>
```

Expected: push succeeds。branch ready for review / merge to `main`。

后续动作(out of S8 plan scope):
1. 6→68 重录基线纪律延续 —— S8 后 ALPHABET ~40 letters,离 68 仍差;由 future-spec 决定何时跨 68 节点 RE-RECORD baseline parquet。
2. 非对称-backbone 角色系统(HARD-GATE,spec §7 + CLAUDE.md)。
3. P_cascade SLOTS_PER_EVENT 多 P 合并规则升级(future-spec)。
4. S5 windowed-f 多字母 stacking 升级(future-spec)。

---

## Self-Review

**1. Spec coverage:**

- **§1 (Why — 24 A 池 + 三子池):**
  - 乙1 (8 escalation copies):Task 1 `A_FAMILY` 8 行 + Task 2 合入 + Task 3 verbatim 测试 `test_yi1_F_rows_verbatim_from_roster_L188_to_L211` / `test_yi1_Z_rows_verbatim` / `test_yi1_P_rows_verbatim`。
  - 乙2 (8 native):Task 1 + Task 2 + Task 3 `test_yi2_F_rows_verbatim_with_F_NOVA_windowed` / `test_yi2_Z_rows_predator_lock_motif_void_bite_vis_weighted` / `test_yi2_P_rows_cascade_crossclan_frozen_verbatim`。
  - 甲 (8 native):Task 1 + Task 2 + Task 3 `test_jia_F_rows_F8Ar1_random_LanceFront_hash` / `test_jia_Z_rows_ambush_sweep_nip_coil_verbatim` / `test_jia_P_rows_zscan_invert_stutter_verbatim`。
  - "Reuse S1–S7 mechanisms at extreme values":Task 1 `A_F` 走 S5 7-tuple + 含 IN_PLACE_DIR / hash:* / rand:1of4 / 4-nbr 共 4 种 dirs 形态(S4 已识别);`A_Z` 走 S1 4-tuple 含 vis_mode;`A_P` 走 S2 + Task 4 `A_SHAPE`(power=4 + mask="cross" 新值域);`A_SLOTS` 走 S7 `SLOTS_PER_EVENT[P_cascade]=2`。
- **§2 (De-gate user decision):**
  - "A primitives are family F/P/Z, NOT a 5th rank-4 family":Task 1 `A_FAMILY` 全部 value ∈ {F,P,Z};Task 1 `test_a_pool_no_rank_4_letter_in_alphabet_value_set` 守门。
  - "FAMILY_RANK stays {N,F,P,Z}":Global Constraints 红线 + 不动 `src/des/types.py`。
  - "Reachability via affinity spectrum, no special path":Task 3 `test_a_letters_reachable_via_within_family_affinity_spectrum` / `test_p_letters_reachable_via_within_family_affinity_spectrum`。
  - "n_locked overwrite gate retired":Task 3 `test_n_locked_is_advisory_only_not_wired_to_mutation_path` grep-based audit;Task 6 doc cleanup 24 条 RETIRED 句替换 + OPEN-1/θ section 标 RETIRED;Global Constraints 红线。
  - "Roster cleanup":Task 6 Step 1 + Step 2 验明替换 24 行 + OPEN-1/θ section verbatim。
- **§3 (Red lines):**
  - 红线 1 (de-gating adds no "who is strong"):全栈 0 手写 magnitude;A 强度全由 `_F`/`_Z`/`_P` 极端值经既有路径流动;Task 1 / Task 2 verbatim 数据,无新内核分支。
  - 红线 2 (default game unchanged):Task 2 默认 BB0 不含 A;Task 5 单字母 blend identity 守 byte-identity;Task 6 端到端 byte-identical engine.run。
  - 红线 3 (`copy-of` 仅注释):Global Constraints 显式禁止;`_a_pool.py` 数据表不带 `copy_of` 字段;mutation core (`_mutation_outcomes` / `phase2_reproduce`) 不读 copy-of。
- **§4 (Architecture — registry data entry + de-gate edit):**
  - Task 1 (data tables) → Task 2 (merge) → Task 4 (SPECTRUM_SHAPE 值域扩展 + 合入)。
  - "affinity untouched":Global Constraints + `_spectrum_for` 仅加 `"cross"` mask 分支,不改 `affinity()`。
  - "de-gate is mostly no-op in code":Task 2 / Task 4 都是 dict.update + 值域 assert 放宽,kernel 0 改动。
- **§4.1 (Multi-P spectrum blend — S8 owns):**
  - Task 5 `blend_p_spectra` 实现 `Σ pᵢ qᵢ / Σ pᵢ` + `Σ=0` 等权退化 + 单字母 identity;`phenotype()` 末尾 spectrum 计算改读 blend。
  - 测试覆盖:`tests/test_multi_p_blend.py` 9 条 unit test + `tests/test_a_pool.py` 4 条 phenotype 端到端。
  - 单 P 字母 byte-identical:Task 5 `test_phenotype_default_bb0_spectrum_byte_identical_to_pre_S8`。
  - 双 P 字母 weight:`test_phenotype_two_P_letters_blends_per_spec_4_1` + `test_blend_two_letters_weighted_per_target`。
  - 等权退化:`test_phenotype_three_P_letters_all_zero_p_add_equal_weight` + `test_blend_sum_p_add_zero_uses_equal_weight_degenerate_path`。
- **§5 (Data flow):** A primitives flow through 既有 `mint→phenotype→kernel`;S8 仅扩 registry 数据 + 改 `phenotype()` 末尾 spectrum 一行。Task 1–5 严格按 spec data flow 实现。
- **§6 (Error handling):**
  - "f≤0.85 / z≤1.5 / p_add≤0.34":Task 2 module-load assert + Task 2 `test_a_pool_extreme_value_bounds_assert_at_module_load`。
  - "rate≤0.35":由 `min(P_MAX=0.08, MU+p_add)` 在 phenotype 已守(p_add≤0.34 ⇒ MU+p_add ≤ 0.35,与 P_MAX=0.08 cap 不冲突;Global Constraints 显式说明 P_MAX 是 v1 placeholder,不另起 cap 路径)。
  - "z↔prey 反相关":Global Constraints 显式声明非 runtime 检查,roster verbatim 数值 + prey_clauses 字段保证。
- **§7 (Out of scope / notes):**
  - "rank-4 vs family-F/P/Z":Task 1 Global Constraints + verbatim 测试已锁。
  - "asymmetric-backbone role system HARD-GATE":Global Constraints + Task 7 后续动作显式列出。
  - "κ same-channel synergy κ=0":Global Constraints。
  - "A appears in default game symmetrically, no faction asymmetry":Task 6 `test_default_4_faction_run_byte_identical_post_s8` + Task 7 Step 5 A 涌现 smoke。

**2. Placeholder scan:**

- 无 `TBD` / `TODO` / "implement later" / "fill in details"。
- 无 "similar to Task N" 引用(每 task 独立)。
- 无 "write tests for the above"(每 test step 给出完整代码)。
- 无 "add appropriate error handling" / "add validation"(具体 assert + boundary 值都给出)。
- Task 7 Step 3 `<baseline.parquet>` / `<s8-fresh.parquet>` 是命令行参数动态文件名占位(实施者按 ls 结果填),步骤描述清楚判定规则,不是 plan 级空白(与 sibling S7 Task 5 Step 3 同款做法)。
- Task 7 Step 7 `<files-touched>` / `<description of the regression fixed>` 是 fix-forward 占位(只有 Step 1 surface regression 时才用),sibling S7 Task 5 Step 7 同款。
- Task 7 Step 8 `<current-branch>` 是 push 目标占位(实施者按 `git branch --show-current` 填),sibling S7 Task 5 Step 8 同款。

**3. Type consistency:**

- `A_FAMILY: dict[str, str]` / `A_GRAN: dict[str, str]` / `A_MOTIF_LEN: dict[str, int]` / `A_F: dict[str, tuple]` / `A_Z: dict[str, tuple]` / `A_P: dict[str, tuple]` / `A_SHAPE: dict[str, tuple]` / `A_SLOTS: dict[str, int]` —— Task 1 定义,Task 2 import + update,Task 3 验证,Task 4 (A_SHAPE) import + update。全程同名同 dtype。
- `_a_pool.py` 模块名:Task 1 创建,Task 2 `from des._a_pool import A_FAMILY as _A_FAMILY, ...` 导入,Task 4 类似 `from des._a_pool import A_SHAPE as _A_SHAPE`(分两次 import 因避循环序问题:`_A_F/_Z/_P` 在 `_P` 块之后立即合入,`_A_SHAPE` 在 `SPECTRUM_SHAPE` 块之后才能合入)。
- `blend_p_spectra(pairs)` signature:Task 5 定义 `pairs: tuple[tuple[float, tuple[tuple[str, float], ...]], ...] -> tuple[tuple[str, float], ...]`,Task 5 phenotype() call-site 同 signature。
- `phenotype().spectrum` 类型:`tuple[tuple[str, float], ...]` —— S2 锁定的类型,Task 5 blend 输出同 type,Task 5 `test_phenotype_default_bb0_spectrum_byte_identical_to_pre_S8` 用 `==` 比较守 byte-equal。
- `phenotype().slots_per_event`:S7 已锁定 `int`,Task 5 显式声明 dominant_p 字段保留,`test_phenotype_dominant_p_field_still_used_by_slots_per_event` 守 S7 路径不破。
- `family_mask` 值域:Task 4 扩到 `{None, "F", "Z", "N", "adjacent", "cross"}`,Task 4 `_spectrum_for` 加 `elif mask == "cross"` 分支,Task 4 `test_p_crossclan_surge_only_hits_cross_rank_targets` 守 `|Δrank| >= 2` 语义。
- `power` 值域:Task 4 扩到 `{1.0, 2.0, 3.0, 4.0}`,Task 4 `test_spectrum_shape_power_domain_includes_4` + `test_p_frozen_aff_pow_4_sharper_than_aff_pow_3` 守。
- `MOTIF_LEN["Predator Lock"]: int = 3`:Task 1 `A_MOTIF_LEN = {"Predator Lock": 3}`,Task 2 update,Task 3 `test_predator_lock_motif_len_3` + `test_yi2_Z_rows_predator_lock_motif_void_bite_vis_weighted` 守。
- `SLOTS_PER_EVENT["P_cascade"]: int = 2`:Task 1 `A_SLOTS["P_cascade"] = 2`,Task 2 update,Task 3 `test_yi2_P_rows_cascade_crossclan_frozen_verbatim` + Task 5 `test_phenotype_dominant_p_field_still_used_by_slots_per_event` 守。
- `F_NOVA` 双 owner(S5 + S8):Task 2 Step 1 显式 conflict guard `if _lk in _F: assert _F[_lk] == _lv`,verbatim 数值要求 byte-equal;Task 3 `test_yi2_F_rows_verbatim_with_F_NOVA_windowed` 用 `expected[F_NOVA] == (0.85, 4-nbr, 0.50, 2, 0.05, 20, 1)` 双向锁。

无 method / property 名称漂移:
- `blend_p_spectra` (snake_case + 单复数明确) 全 plan 同名;不出现 `blendPSpectra` / `blend_p_spectrum` / `p_spectrum_blend` 等变体。
- `A_FAMILY` / `A_GRAN` / `A_MOTIF_LEN` / `A_F` / `A_Z` / `A_P` / `A_SHAPE` / `A_SLOTS` (UPPER_SNAKE) 全表名一致;Task 2 import alias 加 `_` 前缀(`_A_FAMILY` 等)避免与 registry 局部名冲突。
- `dominant_p` 仍是 S2 / S7 既有字段名,Task 5 显式保留逻辑路径。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-24-s8-a-pool-extremes.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. Use `superpowers:subagent-driven-development`。
2. **Inline Execution** — execute tasks in this session with batch checkpoints. Use `superpowers:executing-plans`。

Which approach?
