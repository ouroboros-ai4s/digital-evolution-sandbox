# PHASE3 K-wall 仲裁性能优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 PHASE3 K墙仲裁的争用分支从"逐个体 `repeat_interleave`+`argsort`"换成"record 粒度的精确瓮抽样(Gumbel-max)",消除随世界占用爆炸的成本,同时逐字保全仲裁语义。

**Architecture:** 只改 `src/des/kernels/arbitration.py` 的 `phase3_arbitrate_vec` 争用分支(约 L63-90)。把该分支抽成模块私有 helper `_urn_draw_contested(...)`,返回与旧 `seated` 完全同形的 `[n_rec]` int64 张量;函数其余部分(Section 1 coalesce / Section 3 writeback)与引擎其它模块全部不动。先建立一个 MVHG 矩检验闸门(对当前精确代码即 PASS),swap 后必须保持绿 = 分布未被破坏的证明。

**Tech Stack:** PyTorch 2.10+cu128 (CUDA, RTX 5080),pytest。仅用 `torch.rand`,无新依赖。

## Global Constraints

- **解释器铁律:** 一切 python 命令用 `D:/anaconda3/envs/basic/python.exe`(torch 2.10+cu128 / CUDA True)。裸 `python` 是另一个 cpu-only torch 2.5.1 —— **永远用显式 basic 路径**。
- **imports:** 经 `PYTHONPATH=src`,从 repo root `G:/OUROBOROS-AI4S/digital-evolution-sandbox` 运行。**不 pip / conda install。**
- **引擎冻结:** 除 `arbitration.py` 争用分支外,不改任何引擎语义、阈值、结局常数、recorder、world、reproduction、antagonism。
- **必须保全的仲裁契约(逐字,design.md L168-175 / L201):** ① K墙精确抽样:`total > avail=K−占用` 时精确抽 `avail` 个,每个体存活概率 = `avail/total`(无放回多元超几何);② 等比例、无隐藏权重 —— 分配只依赖送达量,绝不依赖 `sid`/`faction`("加权重=手写φ=数据作废");③ faction-blind(红线 7-J);④ 蒸发不挤活体(只填空位,绝不驱逐活体);⑤ 硬上限 ≤ K / 同 sid 不同 faction 分槽 / 收敛同 `(sid,faction)` 合并;⑥ 枚举顺序无关(均值 pairwise ratio < 1.10);⑦ 给定种子确定且统计正确。property 测试是统计性的(非 bit-exact)。
- **测试纯净:** 套件在 `-W error` 下无 warning(既有约定)。

---

### Task 1: MVHG 矩检验闸门(强制 de-risk gate)

建立分布特征化测试 —— 对**当前精确代码**即应 PASS(证明测试正确捕获了"无放回多元超几何"这条 law),swap 后必须保持绿。这是 §5.1 强制非可选闸门,失败会静默污染数据(本项目被 B1/B2/B3 烧过),故先立闸。

**Files:**
- Create: `tests/test_arbitration_moments.py`

**Interfaces:**
- Consumes: `des.kernels.arbitration.phase3_arbitrate_vec`(现有公共函数,签名 `phase3_arbitrate_vec(live_sid, live_count, live_faction, arrivals, K, birth_tick, T, generator, MAXSID, NFAC=4)`);`des.kernels.reproduction.ArrivalBuffer`(`.add(ty,tx,sid,cnt,fac)` / `.tensors()`)。
- Produces: 一个永久回归闸门,断言争用格的 per-record 幸存计数服从闭式 MVHG 均值/方差。

- [ ] **Step 1: 写矩检验测试**

通过公共 `phase3_arbitrate_vec` 跑两类 synthetic 单争用格(空 residents):Case A 验"比例均值"(热区型,N≫avail),Case B 验"无放回精确性"(小 N,MVHG 方差 ≪ multinomial 方差,能识别近似回退)。

```python
# tests/test_arbitration_moments.py
"""MVHG moment gate for PHASE3 contested-cell allocation (spec 2026-06-21
arbitration-perf §5.1). PASSES on the exact current code; MUST stay green after
the urn-draw swap. A statistical bug here is silent -> it would poison the
recorded dataset, so this gate is mandatory, not optional."""
import torch
from des.kernels.arbitration import phase3_arbitrate_vec
from des.kernels.reproduction import ArrivalBuffer

DEV = torch.device("cpu")        # gate runs on CPU: deterministic, no GPU needed
MAXSID = 256
NFAC = 4


def _one_cell_arrivals(counts):
    """One contested cell at (0,0); record i = strain (i+1), faction 0, count c_i."""
    buf = ArrivalBuffer(DEV)
    for i, c in enumerate(counts):
        buf.add(torch.tensor([0]), torch.tensor([0]),
                torch.tensor([i + 1], dtype=torch.int32),
                torch.tensor([c], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int8))
    return buf.tensors()


def _survivors_per_record(counts, K, trials):
    """Run phase3 `trials` times into an empty K-slot cell; return [trials, R]
    survivor-count matrix keyed by the strain ids (i+1)."""
    R = len(counts)
    out = torch.zeros((trials, R), dtype=torch.float64)
    for seed in range(trials):
        sid = torch.zeros((1, 1, K), dtype=torch.int32)
        cnt = torch.zeros((1, 1, K), dtype=torch.int32)
        fac = torch.zeros((1, 1, K), dtype=torch.int8)
        birth = torch.zeros((1, 1, K), dtype=torch.int32)
        arr = _one_cell_arrivals(counts)
        g = torch.Generator(device=DEV); g.manual_seed(seed)
        nsid, ncnt, _, _ = phase3_arbitrate_vec(
            sid, cnt, fac, arr, K=K, birth_tick=birth, T=1, generator=g,
            MAXSID=MAXSID, NFAC=NFAC)
        seated = int(ncnt[0, 0].sum())
        assert seated == K, f"seed={seed}: seated {seated} != avail {K}"  # hard cap
        for i in range(R):
            m = nsid[0, 0] == (i + 1)
            out[seed, i] = int(ncnt[0, 0][m].sum())
    return out


def test_mvhg_proportional_means_hot_regime():
    # N=1000, avail=10. MVHG mean_i = avail * c_i / N = [1, 2, 7].
    counts = [100, 200, 700]; K = 10; N = sum(counts)
    surv = _survivors_per_record(counts, K, trials=10000)
    means = surv.mean(dim=0)
    expected = torch.tensor([K * c / N for c in counts], dtype=torch.float64)
    assert torch.allclose(means, expected, atol=0.15), \
        f"means {means.tolist()} != MVHG {expected.tolist()}"
    # every trial seats exactly avail across records
    assert torch.all(surv.sum(dim=1) == K)


def test_mvhg_without_replacement_variance_small_N():
    # N=9, avail=6, counts [3,3,3]. MVHG var_i = avail*p*(1-p)*(N-avail)/(N-1)
    #   = 6*(1/3)*(2/3)*(3/8) = 0.5  ;  multinomial (with-replacement) var = 1.333.
    # The exact draw must match 0.5, NOT 1.333 -> this discriminates exact vs approx.
    counts = [3, 3, 3]; K = 6; N = sum(counts)
    surv = _survivors_per_record(counts, K, trials=10000)
    var = surv.var(dim=0, unbiased=True)
    p = 1.0 / 3.0
    mvhg_var = K * p * (1 - p) * (N - K) / (N - 1)          # 0.5
    multinomial_var = K * p * (1 - p)                        # 1.333
    assert torch.all(var < 0.5 * (mvhg_var + multinomial_var)), \
        f"variance {var.tolist()} looks with-replacement (MVHG={mvhg_var:.3f}, " \
        f"multinomial={multinomial_var:.3f})"
    means = surv.mean(dim=0)
    assert torch.allclose(means, torch.full((3,), 2.0, dtype=torch.float64),
                          atol=0.1), f"means {means.tolist()} != [2,2,2]"
```

- [ ] **Step 2: 跑测试,对当前代码应 PASS**

Run: `cd /g/OUROBOROS-AI4S/digital-evolution-sandbox && PYTHONPATH=src D:/anaconda3/envs/basic/python.exe -m pytest tests/test_arbitration_moments.py -v`
Expected: 2 passed(当前 `phase3_arbitrate_vec` 已是精确 MVHG,故两条均 PASS。若不 PASS,说明测试或对当前 law 的理解有误 —— 先修测试,不要继续)。

- [ ] **Step 3: 在 `-W error` 下复跑确认无 warning**

Run: `cd /g/OUROBOROS-AI4S/digital-evolution-sandbox && PYTHONPATH=src D:/anaconda3/envs/basic/python.exe -m pytest tests/test_arbitration_moments.py -W error -q`
Expected: 2 passed,无 warning。

- [ ] **Step 4: Commit**

```bash
cd /g/OUROBOROS-AI4S/digital-evolution-sandbox
git add tests/test_arbitration_moments.py
git commit -m "test: MVHG moment gate for PHASE3 contested allocation (de-risk gate)

Characterizes the exact without-replacement law (proportional means +
finite-population variance discriminator). Passes on current exact code;
must stay green after the urn-draw swap. Mandatory per spec §5.1.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: 争用分支换成 record 粒度瓮抽样

把 `phase3_arbitrate_vec` 的争用分支(L63-90 的逐个体 `repeat_interleave`+`argsort`)替换为调用新私有 helper `_urn_draw_contested(...)`,后者按 record 粒度做精确无放回抽样,成本与个体数无关。

**Files:**
- Modify: `src/des/kernels/arbitration.py`(替换 L63-90 争用分支;新增模块私有 helper)

**Interfaces:**
- Consumes: 函数内 Section 2 已算好的 `merged` `[n_rec]` int64(per-record 计数)、`rec_cell` `[n_rec]` int64(每 record 的 cell 线性索引)、`avail_grid` `[H,W]` int(每格空位 = `K−占用`)、`contested` `[n_rec]` bool、`generator`、`dev`、`W`。
- Produces: helper 返回 `seated` `[n_rec]` int64(争用 record 的幸存计数,非争用位置为 0),与旧 `seated` 变量同形;最终 `survived = torch.where(contested, seated, survived)` 这行不变。Section 3 writeback 不受影响。

- [ ] **Step 1: 跑既有套件 + 矩闸门,建立绿基线**

Run: `cd /g/OUROBOROS-AI4S/digital-evolution-sandbox && PYTHONPATH=src D:/anaconda3/envs/basic/python.exe -m pytest tests/test_arbitration.py tests/test_arbitration_properties.py tests/test_arbitration_moments.py -q`
Expected: all passed(swap 前的基线)。

- [ ] **Step 2: 新增 helper `_urn_draw_contested`**

在 `arbitration.py` 顶部(`import torch` 之后、`def phase3_arbitrate_vec` 之前)插入。算法 = 逐空位 Gumbel-max 瓮抽样,权重 = 剩余计数,跨争用格向量化:

```python
def _urn_draw_contested(merged, rec_cell, avail_grid, contested, W, generator, dev):
    """Exact multivariate-hypergeometric survivor counts for contested records,
    drawn at RECORD granularity (no per-individual expansion).

    Per design.md L170-175/L201: draw exactly avail = K - occupied survivors from
    the pooled arriving individuals of each contested cell; per-individual survival
    prob = avail/total, weighted ONLY by delivered count -> faction-blind & sid-blind
    by construction. Cost ~ O(max_avail * n_contested_cells * R_max); max_avail <= K
    and ~0 when the world is full (the hot regime). Uses only torch.rand.

    Returns: seated [n_rec] int64 -- survivor count per contested record, 0 elsewhere.
    """
    n_rec = merged.shape[0]
    seated = torch.zeros(n_rec, dtype=torch.int64, device=dev)
    c_idx = torch.nonzero(contested, as_tuple=False).flatten()      # [m] global rec idx
    m = c_idx.numel()
    if m == 0:
        return seated

    c_cell = rec_cell[c_idx]                                         # [m] cell per rec
    cc, inv = torch.unique(c_cell, return_inverse=True)             # cc:[n_cc], inv:[m]
    n_cc = cc.numel()

    # within-cell column index per contested record (group by row=inv, enumerate)
    order = torch.argsort(inv, stable=True)
    inv_s = inv[order]
    seg = torch.ones(m, dtype=torch.bool, device=dev)
    seg[1:] = inv_s[1:] != inv_s[:-1]
    grp = torch.cumsum(seg.long(), 0) - 1
    start = torch.searchsorted(grp.contiguous(), grp.contiguous())
    col_s = torch.arange(m, device=dev) - start                     # col in sorted order
    col = torch.empty(m, dtype=torch.int64, device=dev)
    col[order] = col_s                                              # scatter back
    R_max = int(col.max().item()) + 1

    remaining = torch.zeros((n_cc, R_max), dtype=torch.float32, device=dev)
    remaining[inv, col] = merged[c_idx].to(torch.float32)          # per-record counts
    survivor = torch.zeros((n_cc, R_max), dtype=torch.int64, device=dev)
    cell_avail = avail_grid.flatten()[cc].to(torch.int64)          # [n_cc] avail per cell
    drawn = torch.zeros(n_cc, dtype=torch.int64, device=dev)
    rows = torch.arange(n_cc, device=dev)
    max_avail = int(cell_avail.max().item())
    tiny = 1e-20
    neg_inf = torch.tensor(float("-inf"), device=dev)

    for _ in range(max_avail):
        active = drawn < cell_avail                                 # [n_cc] bool
        u = torch.rand((n_cc, R_max), generator=generator, device=dev).clamp_(tiny, 1.0)
        gumbel = -torch.log(-torch.log(u))                          # standard Gumbel
        score = torch.where(remaining > 0,
                            torch.log(remaining.clamp(min=tiny)) + gumbel,
                            neg_inf.expand_as(remaining))           # mask empty records
        pick = score.argmax(dim=1)                                  # [n_cc] one rec/cell
        inc = active.to(torch.int64)
        survivor.index_put_((rows, pick), inc, accumulate=True)
        remaining.index_put_((rows, pick), (-inc).to(remaining.dtype), accumulate=True)
        drawn = drawn + inc

    seated[c_idx] = survivor[inv, col]                              # gather back
    return seated
```

- [ ] **Step 3: 替换争用分支为 helper 调用**

把 `phase3_arbitrate_vec` 中这段(现 L60-90):

```python
    survived = merged.clone()
    # cells that fit entirely (total <= avail): keep merged as-is.
    contested = rec_total > rec_avail                        # [n_rec] bool
    if contested.any():
        c_idx = torch.nonzero(contested, as_tuple=False).flatten()   # record indices
        c_counts = merged[c_idx]                              # [m]
        # expand contested records to individuals, tagged by local record index
        labels = torch.repeat_interleave(c_idx, c_counts)     # [n_ind] -> record idx
        ind_cell = rec_cell[labels]                           # [n_ind] cell per individual
        keys = torch.rand(labels.shape[0], generator=generator, device=dev)  # i.i.d.
        # sort by (cell, key): pack cell into the integer part, key into fractional.
        # cells are < H*W; key in [0,1) -> composite = cell + key is monotonic per cell.
        composite = ind_cell.to(torch.float64) + keys.to(torch.float64)
        order = torch.argsort(composite)
        sorted_cell = ind_cell[order]
        sorted_label = labels[order]
        # within-cell rank via segment start (sorted_cell is non-decreasing)
        n_ind = sorted_cell.shape[0]
        seg_start = torch.zeros(n_ind, dtype=torch.bool, device=dev)
        seg_start[0] = True
        seg_start[1:] = sorted_cell[1:] != sorted_cell[:-1]
        group_id = torch.cumsum(seg_start.long(), 0) - 1      # 0-based group per individual
        start_pos = torch.searchsorted(group_id.contiguous(), group_id.contiguous())
        rank = torch.arange(n_ind, device=dev) - start_pos    # within-cell rank
        # avail per individual (by its cell)
        ind_avail = avail_grid.flatten()[sorted_cell]         # [n_ind]
        keep = rank < ind_avail                               # [n_ind] bool
        kept_labels = sorted_label[keep]
        # survivors per contested record = count of kept individuals with that label
        seated = torch.bincount(kept_labels, minlength=n_rec) # [n_rec]; 0 for non-contested
        survived = torch.where(contested, seated, survived)
```

替换为:

```python
    survived = merged.clone()
    # cells that fit entirely (total <= avail): keep merged as-is.
    contested = rec_total > rec_avail                        # [n_rec] bool
    if contested.any():
        # record-granularity exact urn draw (no per-individual expansion).
        seated = _urn_draw_contested(merged, rec_cell, avail_grid, contested,
                                     W, generator, dev)
        survived = torch.where(contested, seated, survived)
```

- [ ] **Step 4: 跑矩闸门 —— 必须保持绿(swap 正确性的核心证明)**

Run: `cd /g/OUROBOROS-AI4S/digital-evolution-sandbox && PYTHONPATH=src D:/anaconda3/envs/basic/python.exe -m pytest tests/test_arbitration_moments.py -v`
Expected: 2 passed(均值仍 = MVHG,方差仍是无放回的,非 multinomial)。若任一失败 —— 分布被破坏,停下查 helper,**不要**继续。

- [ ] **Step 5: 跑既有仲裁套件 —— 全绿(语义未变)**

Run: `cd /g/OUROBOROS-AI4S/digital-evolution-sandbox && PYTHONPATH=src D:/anaconda3/envs/basic/python.exe -m pytest tests/test_arbitration.py tests/test_arbitration_properties.py -v`
Expected: all passed,尤其 `test_kwall_order_independent`(顺序无关 <1.10)、`test_hard_cap_never_exceeds_K`、`test_faction_blind_equal_arrivals`、`test_resident_not_evicted`、`test_same_sid_different_faction_kept_separate`。

- [ ] **Step 6: 跑全套件 + `-W error`(确认引擎其它部分零回归、无 warning)**

Run: `cd /g/OUROBOROS-AI4S/digital-evolution-sandbox && PYTHONPATH=src D:/anaconda3/envs/basic/python.exe -m pytest -W error -q`
Expected: 全部 passed,无 warning。

- [ ] **Step 7: Commit**

```bash
cd /g/OUROBOROS-AI4S/digital-evolution-sandbox
git add src/des/kernels/arbitration.py
git commit -m "perf: PHASE3 contested K-wall draw at record granularity (exact urn)

Replace per-individual repeat_interleave+argsort (cost ~ n_individuals,
unbounded as cells fill) with Gumbel-max urn draw over [n_cc, R_max] remaining
counts (cost ~ max_avail*n_cc*R_max, and max_avail~0 at world-fill). Exact
multivariate-hypergeometric, torch.rand only, faction/sid-blind by construction.
MVHG moment gate + order-independence + all property tests stay green.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: 性能验收 + 收尾

用既有 `scripts/run_batch.py --phase-probe` 在世界填满阶段实测仲裁耗时,确认从 ~1324ms 降到与 reproduction(~123ms)同量级或更低;清理一次性诊断脚本。

**Files:**
- (无代码改动;运行验证 + 删除 untracked 诊断脚本 `scripts/diag_recorder.py`)

**Interfaces:**
- Consumes: `scripts/run_batch.py` 现有 `--phase-probe N`(打印 `[phase-probe N ticks] anta X | repro Y | arb Z ms/tick`)。

- [ ] **Step 1: 实测 swap 后 per-phase 耗时(世界填满阶段)**

Run: `cd /g/OUROBOROS-AI4S/digital-evolution-sandbox && PYTHONPATH=src D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --phase-probe 240`
Expected: 打印 `[phase-probe 240 ticks] anta ~24 | repro ~123 | arb Z ms/tick`,其中 **Z 从 ~1324ms 降到 ≲123ms 量级**(swap 前 arb=1323.9)。记录实测三个数字到提交说明 / 汇报。

- [ ] **Step 2: 若 arb 仍显著 > repro,停下汇报**

判据(报数不判决):仅记录 anta/repro/arb 三数及 arb 相对 swap 前(1323.9ms)的下降倍数。若 arb 未降到 repro 同量级,**不要**自行加码改对抗或其它,停下把数字交用户决定下一步。

- [ ] **Step 3: 删除一次性诊断脚本**

`scripts/diag_recorder.py` 是诊断期一次性产物(untracked),可复用的 probe 已在 `run_batch.py --phase-probe`。删除以保持工作树干净。

Run: `cd /g/OUROBOROS-AI4S/digital-evolution-sandbox && rm -f scripts/diag_recorder.py && git status --short`
Expected: `diag_recorder.py` 不再出现在 untracked 列表。

- [ ] **Step 4: 确认工作树干净 + 全套件最终绿**

Run: `cd /g/OUROBOROS-AI4S/digital-evolution-sandbox && PYTHONPATH=src D:/anaconda3/envs/basic/python.exe -m pytest -q && git status --short`
Expected: 全部 passed;`git status` 仅显示已 commit 的改动(无遗留)。

---

## 完成标志

- 矩闸门 `test_arbitration_moments.py` 绿(分布精确),既有仲裁套件 + 全套件绿(语义零回归),`-W error` 无 warning。
- `--phase-probe` 实测 arb 从 ~1324ms 降到 repro 量级,T=450×4 跑批回到可行区间。
- 引擎除 `arbitration.py` 争用分支外未动;diag 脚本已清理。

**本计划范围到"引擎够快 + 全绿 + 性能验收"为止。** 重跑首批(四象限四阵营 128²/K64/T450×4)+ analyze_batch 出报告交用户判动力学 = 另一计划(`2026-06-21-first-batch-run-and-verify.md` 的 Task 7),不在此计划内。
