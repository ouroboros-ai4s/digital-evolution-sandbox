# Faction Expansion + Hot-Path Vectorization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the two engine hot-path kernels once to add the faction (team) axis (4-quadrant expansion, cross-faction-only antagonism, same-faction immunity) AND fully vectorize arbitration (P1) + reproduction (P2) + downgrade the stop-check (P3), killing the per-element GPU↔CPU sync that made the first batch 12.8 s/tick.

**Architecture:** Parallel `faction [H,W,K]` int8 tensor alongside `strain_id`/`count`. Dual-orthogonal identity: `sid` is the phenotype key (`phe[sid]` NEVER indexed by faction), `faction` is the team key, slot identity = `(sid, faction)`. Antagonism gates fight on `faction_i != faction_j`. Arbitration K-wall sampling becomes a random-key multivariate-hypergeometric draw (i.i.d. keys, never read sid/faction → fair by construction). Reproduction groups by direction (≤4 tensor ops, not per-strain) and batch-mints mutants once per tick.

**Tech Stack:** Python 3.12+, PyTorch 2.10 (cu128 GPU build), pyarrow 22, pytest 9 — all preinstalled in the `basic` conda env. Code is device-agnostic.

## Global Constraints

- **Spec authority:** `docs/superpowers/specs/2026-06-21-faction-and-vectorization-redesign.md`; upstream mechanism doc `context/2026-06-11-16-10-design.md` (chapter「世界初始化 · 阵营模型 · 目标宏观动力学」) is authoritative for mechanics. Conflict → design.md wins. Implement, don't redefine.
- **Environment — DO NOT MODIFY:** run everything inside the existing `basic` conda env. The real interpreter is `D:/anaconda3/envs/basic/python.exe` (torch 2.10+cu128, CUDA True, RTX 5080). The bare `python` on PATH is a different cpu-only torch 2.5.1 — **always use the explicit `basic` path**. **Never run `pip install`, `conda install`, or `pip install -e .`.** Import works purely via `pyproject.toml` `[tool.pytest.ini_options] pythonpath = ["src"]`.
- **Run tests with:** `D:/anaconda3/envs/basic/python.exe -m pytest <path> -v` from the repo root `G:/OUROBOROS-AI4S/digital-evolution-sandbox`.
- **私货红线 (project survival line), 11-item audit (spec §7):** outcome constants (`base/z/f/μ/z_max/δ/α₀/β_fold/κ`) are global scalars, **NEVER** faction-keyed. No 4×4 asymmetric matchup matrix. `phe[sid]` gather paths NEVER carry a faction index. K-wall sampling never reads sid/faction. faction appears in **exactly one** place: the antagonism fight-eligibility predicate. Same-faction immunity = **exactly** skip the kill, no bonus.
- **Dual-orthogonal identity:** slot identity = `(sid, faction)`. Same sequence + different faction share one phenotype cache row (StrainTable is UNTOUCHED). faction is inherited unchanged through mutation (mutation only changes `sid`).
- **Determinism:** all randomness from the explicit `torch.Generator` passed in; same seed → reproducible run. New sampling is NOT bit-identical to the old reference (that's intentional — old data was garbage; spec §1.12).
- **C = K:** per-cell slot capacity equals individual cap K; distinct `(sid,faction)` per cell ≤ K by pigeonhole. `strain_id == 0` is the empty sentinel.
- **v1 constants (baked in registry, not knobs):** `μ=0.01 / z_max=8.0 / δ=0.05 / p_max=0.08`, `α₀=β_fold=κ=0`. `NFAC=4`. Locked config: 128×128 / K=64 / fill=20 / **T=450** / seeds [0,1,2,3].
- **Direction universe (v1):** `ALL_DIRECTIONS = [(-1,0),(1,0),(0,-1),(0,1)]` — every v1 strain's directions are a subset (F4Nr1 = north only; F4Nr4 = all four). If a future F primitive adds a new direction, extend this list.
- **TDD + frequent commits:** every task is test-first and ends with a commit. Use `git add <explicit paths>`, never `git add .`.

---

## ⚠ Pre-flight open question (must resolve before the batch run; does NOT block Tasks 1–8)

While tracing the real code for this plan I found two facts the spec's §3 timeline does not account for. **Neither affects the faction or vectorization tasks** (they are correct either way), but both affect `T` and acceptance criterion #2:

1. **Strain firing period is `min` over all primitives, not the F-period.** `registry.phenotype()` sets `period = min(periods)`. BB0 contains `P_base` (period 1), so **BB0's period = 1**, not 5. Reproduction's `fires_this_tick` uses this single strain period → BB0 reproduces **every tick**, i.e. expansion is **1 cell/tick/axis**, not 1 cell/5 ticks. Real timeline on 128² (centers 64 apart): fronts meet **~tick 32**, world fills **~tick 64** — 5× faster than spec §3's 160/320.

2. **Latent reproduction bug (B4):** current `phase2_reproduce` rolls BOTH the offspring-count tensor and the target-coordinate tensor by the same `(dy,dx)`, which maps every source cell's offspring back onto the source cell → **offspring never actually move to neighbors.** Full-field homogeneous BB0 masked this; 4-quadrant expansion would expose it (nothing would ever expand). **Task 5 fixes this** (roll exactly one of {data, coordinate}); it is in-scope because vectorized reproduction rewrites this code path anyway.

**Decision (user, 2026-06-21): choice B.** Reproduction fires on the F-primitive period (=5), restoring the design's 异相位时钟 (per-phase firing clock). T=450 and the spec §3 160/320 timeline hold. **Task 5** computes `repro_period` (min over F periods) and `anta_period` (min over Z periods); **Task 6** consumes `repro_period` for reproduction; **Task 7** switches antagonism's firing clock to `anta_period`. Acceptance #2's tick numbers (meet ~160, fill ~320) are valid under choice B.

Tasks 1–6 are period-agnostic for the faction/vectorization changes; the per-phase clock is fully in effect after Task 7. Task 10 (run_batch T=450) depends on Tasks 7 + 9.

## ⚠ Test-sequencing note (pipeline rewrite — read before executing)

This plan rewires a 4-phase pipeline whose signatures all change. The **engine integrator (`engine.py`) is rewired once, last, in Task 9**. Consequently, between Task 2 and Task 9 the engine-level suites (`tests/test_engine.py`, the body-liveness + `test_full_run_*` tests in `tests/test_integration.py`) are **expectedly RED** — `snapshot()` becomes a 3-tuple in Task 2, kernel signatures gain `faction`, etc. This is normal for a pipeline rewrite.

**Each kernel task's gate runs ONLY its own listed test file(s)** — those must be green at that task's commit. Task 9 re-greens the whole suite. The red-until-Task-9 set is: `test_engine.py`, and `test_integration.py::{test_g10_*, test_mutants_appear, test_frequencies_move_not_frozen, test_not_instant_global_death, test_full_run_dumps_self_contained_frames}`. The pure-function / red-line integration tests (`test_phenotype_reads_only_sequence`, `test_kwall_equal_ratio_no_hidden_weight`) stay green throughout (Task 9 migrates the kwall one). Do NOT mark Task 9 complete until the full suite is green.

---

### Task 1: BB0 reproduction primitive F4Nr1 → F4Nr4 (B3 fix)

**Files:**
- Modify: `src/des/registry.py` (`_LOCKED` dict, symbol-located — currently `{1: "F4Nr1", 5: "BroadSweep", 7: "P_base"}`)
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `BB0_TEMPLATE["layout"][1] == "F4Nr4"` — every later task that mints BB0 now gets a 4-direction reproducer (f=0.50, period=5).

- [ ] **Step 1: Update the failing test**

In `tests/test_registry.py`, the existing `test_bb0_template_shape` asserts `BB0_TEMPLATE["layout"][1] == "F4Nr1"`. Change that one line to:

```python
    assert BB0_TEMPLATE["layout"][1] == "F4Nr4"   # B3 fix: 4-dir expansion (was F4Nr1, north-only)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_registry.py::test_bb0_template_shape -v`
Expected: FAIL — `assert 'F4Nr1' == 'F4Nr4'`

- [ ] **Step 3: Make the change**

In `src/des/registry.py`, change the `_LOCKED` dict:

```python
_LOCKED = {1: "F4Nr4", 5: "BroadSweep", 7: "P_base"}  # 0-indexed; F4Nr4 = 4-dir expansion (B3 fix)
```

- [ ] **Step 4: Run the full registry + reproduction + world suites**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_registry.py tests/test_reproduction.py tests/test_world.py -v`
Expected: PASS. (Note: `test_reproduction.py`'s `_LOCKED_SLOTS = {1,5,7}` is about *which positions* are locked, not which letters sit there — it still passes. `test_mutation_*` use `BB0_TEMPLATE["mutable"]` which is unchanged.)

- [ ] **Step 5: Commit**

```bash
git add src/des/registry.py tests/test_registry.py
git commit -m "fix(B3): BB0 reproducer F4Nr1 -> F4Nr4 (4-dir expansion)"
```

---

### Task 2: World faction tensor + four-quadrant seeding (B1 fix)

**Files:**
- Modify: `src/des/world.py` (`World.__init__`, `World.snapshot`, add `init_factions`)
- Test: `tests/test_world.py`

**Interfaces:**
- Consumes: `StrainTable.get_or_mint`, `BB0_TEMPLATE["layout"]`.
- Produces:
  - `World.faction: torch.Tensor` shape `[H,W,K]` int8, zero-init.
  - `World.snapshot() -> tuple[Tensor, Tensor, Tensor]` = `(strain_id, count, faction)` (was 2-tuple).
  - `init_factions(H, W, K, device, table, fill_per_cell, n_fac=4) -> World` — seeds BB0 at the four quadrant centers `(H//4,W//4)`, `(H//4,3*W//4)`, `(3*H//4,W//4)`, `(3*H//4,3*W//4)` with faction 0/1/2/3, all other cells empty.
- **Breaking change:** every caller of `snapshot()` (engine.py) and any test reading a 2-tuple must be updated. `init_bb0` stays (used by recorder/world tests) — only `snapshot` changes arity.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_world.py`:

```python
from des.world import init_factions

def test_world_has_faction_tensor():
    w = World(8, 8, 16, DEV)
    assert w.faction.shape == (8, 8, 16)
    assert w.faction.dtype == torch.int8
    assert w.faction.sum().item() == 0          # zero-init

def test_snapshot_returns_faction():
    w = World(4, 4, 8, DEV)
    w.count[0, 0, 0] = 5
    w.faction[0, 0, 0] = 2
    sid, cnt, fac = w.snapshot()                 # now a 3-tuple
    assert fac[0, 0, 0].item() == 2
    w.faction[0, 0, 0] = 9
    assert fac[0, 0, 0].item() == 2              # snapshot is an independent copy

def test_init_factions_seeds_four_quadrant_centers():
    t = StrainTable()
    H = W = 16; K = 32
    w = init_factions(H, W, K, DEV, t, fill_per_cell=10, n_fac=4)
    centers = [(H//4, W//4), (H//4, 3*W//4), (3*H//4, W//4), (3*H//4, 3*W//4)]
    # exactly 4 non-empty cells
    assert int((w.count.sum(dim=-1) > 0).sum()) == 4
    bb0 = t.get_or_mint(BB0_TEMPLATE["layout"])
    seen_factions = set()
    for (cy, cx) in centers:
        assert int(w.count[cy, cx, 0]) == 10
        assert int(w.strain_id[cy, cx, 0]) == bb0     # all four are the same BB0 sequence
        seen_factions.add(int(w.faction[cy, cx, 0]))
    assert seen_factions == {0, 1, 2, 3}              # one faction per center

def test_init_factions_d4_symmetric_orbit():
    # the four centers are the D4 orbit of one point. Concretely: reflecting the
    # center set about the mid-lines (rows about H/2, cols about W/2) must permute
    # the set onto itself — that is the no-faction-gets-a-geometric-edge guarantee.
    H = W = 16
    centers = {(H//4, W//4), (H//4, 3*W//4), (3*H//4, W//4), (3*H//4, 3*W//4)}
    refl_x = {(cy, (W - cx) % W) for (cy, cx) in centers}
    refl_y = {((H - cy) % H, cx) for (cy, cx) in centers}
    assert refl_x == centers and refl_y == centers
```

- [ ] **Step 2: Run to verify they fail**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_world.py -v`
Expected: FAIL — `init_factions` import error / `faction` attribute missing / snapshot returns 2-tuple.

- [ ] **Step 3: Implement**

Replace `src/des/world.py` body with:

```python
# src/des/world.py
from __future__ import annotations
import torch
from des.phenotype_cache import StrainTable
from des.registry import BB0_TEMPLATE


class World:
    def __init__(self, H: int, W: int, K: int, device: torch.device) -> None:
        self.H, self.W, self.K, self.device = H, W, K, device
        self.strain_id = torch.zeros((H, W, K), dtype=torch.int32, device=device)
        self.count = torch.zeros((H, W, K), dtype=torch.int32, device=device)
        self.faction = torch.zeros((H, W, K), dtype=torch.int8, device=device)

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.strain_id.clone(), self.count.clone(), self.faction.clone()

    def occupancy(self) -> torch.Tensor:
        return self.count.sum(dim=-1)

    def distinct_per_cell(self) -> torch.Tensor:
        return (self.count > 0).sum(dim=-1)


def init_bb0(H: int, W: int, K: int, device: torch.device,
             table: StrainTable, fill_per_cell: int) -> World:
    assert fill_per_cell <= K, "fill must fit in K slots"
    w = World(H, W, K, device)
    bb0 = table.get_or_mint(BB0_TEMPLATE["layout"])
    w.strain_id[:, :, 0] = bb0
    w.count[:, :, 0] = fill_per_cell
    return w


def init_factions(H: int, W: int, K: int, device: torch.device,
                  table: StrainTable, fill_per_cell: int, n_fac: int = 4) -> World:
    """Seed BB0 at the four quadrant centers, one faction each, everything else empty.
    The four centers are the D4-symmetric orbit of one point (equal to grid center,
    equal nearest-wall distance, pairwise-symmetric) → no faction gets a geometric edge."""
    assert fill_per_cell <= K, "fill must fit in K slots"
    assert n_fac == 4, "v1 seeds exactly 4 factions at the 4 quadrant centers"
    w = World(H, W, K, device)
    bb0 = table.get_or_mint(BB0_TEMPLATE["layout"])
    centers = [(H // 4, W // 4), (H // 4, 3 * W // 4),
               (3 * H // 4, W // 4), (3 * H // 4, 3 * W // 4)]
    for fac, (cy, cx) in enumerate(centers):
        w.strain_id[cy, cx, 0] = bb0
        w.count[cy, cx, 0] = fill_per_cell
        w.faction[cy, cx, 0] = fac
    return w
```

- [ ] **Step 4: Run to verify they pass**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_world.py -v`
Expected: PASS (all, including the unchanged `test_init_bb0_*`).

- [ ] **Step 5: Commit**

```bash
git add src/des/world.py tests/test_world.py
git commit -m "feat(B1): faction tensor + init_factions four-quadrant seeding"
```

---

### Task 3: Antagonism — fight on faction, not strain (B2 fix)

**Files:**
- Modify: `src/des/kernels/antagonism.py` (`phase1_antagonism` signature + the `diff_strain` predicate)
- Test: `tests/test_antagonism.py`

**Interfaces:**
- Consumes: `World.faction` slot tensor.
- Produces: `phase1_antagonism(strain_id, count, faction, phe, birth_tick, T, z_max, generator) -> Tensor` — faction is a NEW positional arg inserted after `count`. The fight predicate is now `faction_i != faction_j` (cross-faction only); same-faction (any sid) is immune.
- **Red-line:** faction enters at EXACTLY this one predicate. `phe[sid]` gathers (`z_raw`, `prey_mask`, `feature_mask`, `period`) are untouched — z and targeting stay pure functions of sequence. No 4x4 matrix; the predicate is the symmetric `!=`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_antagonism.py`:

```python
def test_same_faction_immune_even_when_different_strains():
    # two DIFFERENT strains (predator + its valid prey) on the SAME faction:
    # must NOT fight (faction gate), even though strain-targeting would allow it.
    t, pred, prey, phe = _setup()
    sid = torch.zeros((1, 1, 4), dtype=torch.int32)
    cnt = torch.zeros((1, 1, 4), dtype=torch.int32)
    fac = torch.zeros((1, 1, 4), dtype=torch.int8)
    sid[0, 0, 0] = pred; cnt[0, 0, 0] = 100; fac[0, 0, 0] = 1
    sid[0, 0, 1] = prey; cnt[0, 0, 1] = 100; fac[0, 0, 1] = 1   # same faction
    birth = torch.zeros((1, 1, 4), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    out = phase1_antagonism(sid, cnt, fac, phe, birth, T=5, z_max=8.0, generator=g)
    assert out[0, 0, 0].item() == 100 and out[0, 0, 1].item() == 100   # no kills

def test_cross_faction_fights_even_when_same_strain():
    # SAME strain (a self-targeting predator) on DIFFERENT factions: must fight.
    # BroadSweep preys on families F,Z; BroadSweep is itself Z-family -> self-prey.
    t = StrainTable()
    pred = t.get_or_mint(("BroadSweep",))
    phe = t.phenotype_arrays(DEV)
    sid = torch.zeros((1, 1, 4), dtype=torch.int32)
    cnt = torch.zeros((1, 1, 4), dtype=torch.int32)
    fac = torch.zeros((1, 1, 4), dtype=torch.int8)
    sid[0, 0, 0] = pred; cnt[0, 0, 0] = 100; fac[0, 0, 0] = 0
    sid[0, 0, 1] = pred; cnt[0, 0, 1] = 100; fac[0, 0, 1] = 3   # different faction, same strain
    birth = torch.zeros((1, 1, 4), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    out = phase1_antagonism(sid, cnt, fac, phe, birth, T=5, z_max=8.0, generator=g)
    assert out[0, 0, 0].item() < 100 and out[0, 0, 1].item() < 100   # both took losses
```

Also update the three EXISTING tests in this file (`test_same_strain_immunity_no_self_annihilation`, `test_predator_kills_prey_with_self_loss`, `test_no_hit_without_matching_feature`) to pass a faction tensor. For each, add a faction tensor and thread it into the call:

```python
    fac = torch.zeros((1, 1, 4), dtype=torch.int8)
    fac[0, 0, 0] = 0; fac[0, 0, 1] = 1     # different factions: faction gate passes
    out = phase1_antagonism(sid, cnt, fac, phe, birth, T=5, z_max=8.0, generator=g)
```

Note: `test_same_strain_immunity_no_self_annihilation` was a strain-immunity test; under the faction model it becomes a same-faction immunity test, so keep BOTH slots on faction 0 there (`fac` stays all-zero) — it must still show no self-annihilation. `test_predator_kills_prey_with_self_loss` and `test_no_hit_without_matching_feature` use DIFFERENT factions (0 and 1) so the faction gate passes and their existing assertions hold unchanged (the exact-value assert `out[0,0,1] == 100 - round(100*(8*0.4/8.4))` still passes).

- [ ] **Step 2: Run to verify the new tests fail**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_antagonism.py -v`
Expected: FAIL — `phase1_antagonism` got an unexpected positional / missing `faction`.

- [ ] **Step 3: Implement**

In `src/des/kernels/antagonism.py`, add `faction` to the signature (after `count`) and replace the strain predicate with the faction predicate. Two regions change.

Signature:

```python
def phase1_antagonism(
    strain_id: torch.Tensor,
    count: torch.Tensor,
    faction: torch.Tensor,
    phe: dict[str, torch.Tensor],
    birth_tick: torch.Tensor,
    T: int,
    z_max: float,
    generator: torch.Generator,
) -> torch.Tensor:
```

Predicate — replace the `diff_strain` block (the `sid_i = sid_long.unsqueeze(-1)` ... `diff_strain = sid_i != sid_j` lines) with:

```python
    # faction gate: fight iff attacker and prey are on DIFFERENT factions.
    # faction is a SLOT-level state, NOT gathered through sid (dual-orthogonal identity:
    # phe[sid] never sees faction). Same-faction (any sid) is immune — exactly skip the kill.
    fac_slot = faction.long()                        # [H, W, K]
    fac_i = fac_slot.unsqueeze(-1)                    # [H, W, K, 1]
    fac_j = fac_slot.unsqueeze(-2)                    # [H, W, 1, K]
    diff_faction = fac_i != fac_j                     # [H, W, K, K] bool
```

Then change the `valid` line from `valid = hit & diff_strain & fires_i & alive_j` to:

```python
    valid = hit & diff_faction & fires_i & alive_j    # [H, W, K, K] bool
```

(Also update the docstring line `strain_id[i] != strain_id[j]  (G10: same-strain immunity)` -> `faction[i] != faction[j]  (same-faction immunity)`.)

- [ ] **Step 4: Run to verify all pass**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_antagonism.py -v`
Expected: PASS (2 new + 3 updated existing).

- [ ] **Step 5: Commit**

```bash
git add src/des/kernels/antagonism.py tests/test_antagonism.py
git commit -m "fix(B2): antagonism fights on faction != faction, not strain"
```

---

### Task 4: Recorder — add faction column (6-col schema)

**Files:**
- Modify: `src/des/recorder.py` (`_SCHEMA`, `_run`, `dump`)
- Test: `tests/test_recorder.py`

**Interfaces:**
- Consumes: `World.faction`.
- Produces: parquet rows are now 6-col `(tick, cell_x, cell_y, strain, faction, count)`; every frame self-contained.

- [ ] **Step 1: Write the failing test**

In `tests/test_recorder.py`, update `test_dump_writes_nonempty_rows`'s column-set assertion:

```python
    assert set(tbl.columns) == {"tick", "cell_x", "cell_y", "strain", "faction", "count"}
```

Append a new test:

```python
def test_dump_records_faction(tmp_path):
    from des.world import init_factions
    t = StrainTable()
    w = init_factions(16, 16, 32, DEV, t, fill_per_cell=5, n_fac=4)
    path = str(tmp_path / "run.parquet")
    rec = Recorder(path, t); rec.dump(0, w); rec.close()
    tbl = pq.read_table(path).to_pandas()
    assert len(tbl) == 4                                  # 4 seeded cells
    assert set(tbl["faction"].tolist()) == {0, 1, 2, 3}   # one row per faction
    assert (tbl["count"] == 5).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_recorder.py -v`
Expected: FAIL — column set missing `faction` / `init_factions` import.

- [ ] **Step 3: Implement**

In `src/des/recorder.py`:

Schema:

```python
_SCHEMA = pa.schema([
    ("tick", pa.int32()), ("cell_x", pa.int32()), ("cell_y", pa.int32()),
    ("strain", pa.string()), ("faction", pa.int8()), ("count", pa.int32()),
])
```

`_run` — unpack a 6-field job and add the faction column to the record batch:

```python
                tick, ys, xs, sids, facs, cnts = job
                strains = [".".join(self._table.sequence_of(int(s))) for s in sids]
                batch = pa.record_batch([
                    pa.array([tick] * len(sids), pa.int32()),
                    pa.array(xs, pa.int32()),
                    pa.array(ys, pa.int32()),
                    pa.array(strains, pa.string()),
                    pa.array(facs, pa.int8()),
                    pa.array(cnts, pa.int32()),
                ], schema=_SCHEMA)
```

`dump` — gather faction at the same non-zero indices and enqueue it:

```python
    def dump(self, tick: int, world) -> None:
        self._check_thread()
        cnt = world.count.to("cpu")
        sid = world.strain_id.to("cpu")
        fac = world.faction.to("cpu")
        nz = torch.nonzero(cnt > 0, as_tuple=False)   # [M,3] = (y,x,k)
        ys = nz[:, 0].tolist(); xs = nz[:, 1].tolist()
        ks = nz[:, 2]
        sids = sid[nz[:, 0], nz[:, 1], ks].tolist()
        facs = fac[nz[:, 0], nz[:, 1], ks].tolist()
        cnts = cnt[nz[:, 0], nz[:, 1], ks].tolist()
        self._q.put((tick, ys, xs, sids, facs, cnts))
```

- [ ] **Step 4: Run to verify pass**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_recorder.py -v`
Expected: PASS (`test_writer_thread_error_surfaces` still works — a bad sid still raises in the worker).

- [ ] **Step 5: Commit**

```bash
git add src/des/recorder.py tests/test_recorder.py
git commit -m "feat: recorder 6-col schema with faction"
```

---

### Task 5: Phenotype direction bitmask + per-phase periods (precompute for vectorized reproduction)

**Files:**
- Modify: `src/des/phenotype_cache.py` (`phenotype_arrays` — add three gathered arrays)
- Modify: `src/des/registry.py` (`phenotype` — compute `repro_period`, `anta_period`, `dir_bits`)
- Modify: `src/des/types.py` (`Phenotype` — add `repro_period`, `anta_period`, `dir_bits` fields)
- Test: `tests/test_phenotype_cache.py`, `tests/test_registry.py`

**Interfaces:**
- Consumes: `ALL_DIRECTIONS = [(-1,0),(1,0),(0,-1),(0,1)]` (define in registry).
- Produces:
  - `Phenotype.dir_bits: int` — bit `d` set iff `ALL_DIRECTIONS[d]` is in this strain's `directions`.
  - `Phenotype.repro_period: int` — `min` over F-primitive periods (the reproduction firing clock); `1` if no F letter.
  - `Phenotype.anta_period: int` — `min` over Z-primitive periods (the antagonism firing clock); `1` if no Z letter.
  - `phenotype_arrays(device)` dict gains keys `dir_bits` (int64 `[n]`), `repro_period` (int64 `[n]`), `anta_period` (int64 `[n]`). EMPTY row (id 0) = 0 dir_bits, period 1.
- **Why:** Task 6 groups firing slots by direction via `(dir_bits[sid] >> d) & 1` — a gathered bitmask, NO per-strain Python loop. Task 7 uses `repro_period`/`anta_period` to give each phase its own firing clock (choice B). `period` (the old `min`-over-all) stays for backward compat but is no longer read by the kernels after Task 7.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_registry.py`:

```python
from des.registry import ALL_DIRECTIONS

def test_dir_bits_match_directions():
    # F4Nr1 = north only -> exactly the bit for (-1,0)
    north_bit = 1 << ALL_DIRECTIONS.index((-1, 0))
    assert phenotype(("F4Nr1",)).dir_bits == north_bit
    # F4Nr4 = all four -> all four bits
    assert phenotype(("F4Nr4",)).dir_bits == (1 << len(ALL_DIRECTIONS)) - 1
    # no F letter -> no directions -> 0
    assert phenotype(("N0",)).dir_bits == 0

def test_per_phase_periods_split():
    # BB0 has F4Nr4 (F, period 5), BroadSweep (Z, period 5), P_base (P, period 1).
    # OLD period = min(5,5,1) = 1.  NEW: repro_period = 5 (F only), anta_period = 5 (Z only).
    ph = phenotype(BB0_TEMPLATE["layout"])
    assert ph.period == 1            # old min-over-all unchanged (back-compat)
    assert ph.repro_period == 5      # F-primitive period
    assert ph.anta_period == 5       # Z-primitive period
    # a strain with no F letter -> repro_period defaults to 1
    assert phenotype(("BroadSweep",)).repro_period == 1
    # a strain with no Z letter -> anta_period defaults to 1
    assert phenotype(("F4Nr4",)).anta_period == 1
```

Append to `tests/test_phenotype_cache.py`:

```python
def test_phenotype_arrays_has_dir_and_periods():
    t = StrainTable()
    sid4 = t.get_or_mint(("F4Nr4",))
    arr = t.phenotype_arrays(torch.device("cpu"))
    assert set(arr.keys()) == {"f", "p_leave", "z_raw", "p_x", "prey_mask",
                               "feature_mask", "period", "dir_bits",
                               "repro_period", "anta_period"}
    assert arr["dir_bits"].dtype == torch.int64
    assert int(arr["dir_bits"][sid4]) == 0b1111      # all 4 directions
    assert int(arr["dir_bits"][0]) == 0              # EMPTY row
    assert int(arr["repro_period"][sid4]) == 5
    assert int(arr["repro_period"][0]) == 1          # EMPTY row period 1
```

Update the exact key-set guard in the existing `test_phenotype_arrays_indexed_by_id` (it asserts `set(arr.keys()) == {...7 keys...}`) to the new 10-key set:

```python
    assert set(arr.keys()) == {"f", "p_leave", "z_raw", "p_x", "prey_mask",
                               "feature_mask", "period", "dir_bits",
                               "repro_period", "anta_period"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_registry.py tests/test_phenotype_cache.py -v`
Expected: FAIL — `ALL_DIRECTIONS` import error / `dir_bits` attribute missing / key-set mismatch.

- [ ] **Step 3: Implement — types.py**

Add three fields to the `Phenotype` dataclass (after `period`):

```python
@dataclass(frozen=True)
class Phenotype:
    f: float
    directions: tuple[tuple[int, int], ...]
    p_leave: float
    z_raw: float
    prey_mask: int
    feature_mask: int
    p_x: float
    spectrum: tuple[tuple[str, float], ...]
    period: int
    repro_period: int
    anta_period: int
    dir_bits: int
    phase_type: PhaseType | None
    fold: tuple[frozenset[int], ...]
```

- [ ] **Step 4: Implement — registry.py**

Add the direction universe near the top (after the alphabet block):

```python
# v1 direction universe: every strain's directions are a subset of these.
# bit d (in dir_bits) <-> ALL_DIRECTIONS[d]. Extend if a future F primitive adds a direction.
ALL_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIR_BIT = {d: 1 << i for i, d in enumerate(ALL_DIRECTIONS)}
```

In `phenotype()`, track F-periods and Z-periods separately. Replace the single `periods: list[int] = []` accumulation with phase-specific lists and compute the three new values. The minimal diff:

- Add near the other accumulators:
  ```python
      f_periods: list[int] = []
      z_periods: list[int] = []
  ```
- In the `if letter in _F:` branch, after `periods.append(per)` add `f_periods.append(per)`.
- In the `elif letter in _Z:` branch, after `periods.append(per)` add `z_periods.append(per)`.
- Before the `return`, compute:
  ```python
      repro_period = min(f_periods) if f_periods else 1
      anta_period = min(z_periods) if z_periods else 1
      dir_bits = 0
      for d in directions:
          dir_bits |= _DIR_BIT.get(d, 0)
  ```
- Pass them into the `Phenotype(...)` constructor:
  ```python
      return Phenotype(
          f=f, directions=tuple(directions), p_leave=p_leave, z_raw=z_sum,
          prey_mask=prey_mask, feature_mask=feature_mask, p_x=p_x,
          spectrum=spectrum, period=period,
          repro_period=repro_period, anta_period=anta_period, dir_bits=dir_bits,
          phase_type=phase_type, fold=(),
      )
  ```

- [ ] **Step 5: Implement — phenotype_cache.py**

In `phenotype_arrays`, add three zero-init arrays and fill them in the loop. EMPTY row defaults: `dir_bits=0`, periods `1`.

```python
        dir_bits = torch.zeros(n, dtype=torch.int64, device=device)
        repro_period = torch.ones(n, dtype=torch.int64, device=device)
        anta_period = torch.ones(n, dtype=torch.int64, device=device)
```

Inside the `for sid in range(1, n):` loop, after `period[sid] = phe.period`:

```python
            dir_bits[sid] = phe.dir_bits
            repro_period[sid] = phe.repro_period
            anta_period[sid] = phe.anta_period
```

And extend the result dict:

```python
        result = {"f": f, "p_leave": p_leave, "z_raw": z_raw, "p_x": p_x,
                  "prey_mask": prey, "feature_mask": feat, "period": period,
                  "dir_bits": dir_bits, "repro_period": repro_period,
                  "anta_period": anta_period}
```

- [ ] **Step 6: Run to verify pass**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_registry.py tests/test_phenotype_cache.py -v`
Expected: PASS (all, including updated key-set guards).

- [ ] **Step 7: Commit**

```bash
git add src/des/types.py src/des/registry.py src/des/phenotype_cache.py tests/test_registry.py tests/test_phenotype_cache.py
git commit -m "feat: phenotype dir_bits + per-phase repro/anta periods (vectorization precompute)"
```

---

### Task 6: Vectorized reproduction + faction + B4 movement fix (P2)

**Files:**
- Modify: `src/des/kernels/reproduction.py` (`ArrivalBuffer`, `phase2_reproduce`)
- Test: `tests/test_reproduction.py`

**Interfaces:**
- Consumes: `phe["dir_bits"]`, `phe["f"]`, `phe["p_x"]`, `phe["repro_period"]` (Task 5/7); `World.faction`; `ALL_DIRECTIONS` (registry); `binom`, `fires_this_tick` (common).
- Produces:
  - `ArrivalBuffer.add(ty, tx, sid, cnt, fac)` — 5-arg (added `fac`).
  - `ArrivalBuffer.tensors() -> (ty, tx, sid, cnt, fac)` — 5-tuple; empty buffer returns 5 empty tensors.
  - `phase2_reproduce(world, snap_sid, snap_count, snap_faction, phe, table, birth_tick, T, generator) -> (ArrivalBuffer, live)` — added `snap_faction` after `snap_count`. `live` is the post-migration resident count `[H,W,K]` (unchanged semantics).
- **What changes vs current code (three things at once):**
  1. **faction:** group by `(sid, faction)` per slot, not just sid; offspring carry the parent's faction.
  2. **vectorize:** loop over the ≤4 directions (not over present strains). Per direction, one masked binom over the whole `[H,W,K]` slot grid, then `roll`. Mutants batch-minted once per tick.
  3. **B4 fix:** roll the offspring-count tensor by `(dy,dx)` to deposit at the neighbor; do NOT also roll a coordinate grid (the current code rolls both, cancelling the move). Targets are computed by rolling counts into place and reading the resident `(y,x)` index directly.

- **Design note (per-slot, faction-preserving):** reproduction is computed at the **slot** granularity (`[H,W,K]`), so each slot already carries one `(sid, faction)`. We never re-group by sid; faction rides along because it is a slot tensor. This is simpler than the spec's "(sid,faction) regroup" phrasing and strictly equivalent: a slot IS a single (sid,faction).

- [ ] **Step 1: Write the failing tests**

Rewrite `tests/test_reproduction.py`'s buffer + scatter tests to the 5-arg API and add the movement + faction assertions. Replace `test_arrival_buffer_accumulates`:

```python
def test_arrival_buffer_accumulates():
    buf = ArrivalBuffer(DEV)
    buf.add(torch.tensor([0]), torch.tensor([1]), torch.tensor([7]),
            torch.tensor([3]), torch.tensor([2], dtype=torch.int8))
    ty, tx, sid, cnt, fac = buf.tensors()
    assert ty.tolist() == [0] and tx.tolist() == [1]
    assert sid.tolist() == [7] and cnt.tolist() == [3]
    assert fac.tolist() == [2]
```

Add a movement test (the B4 regression gate) and a faction-inheritance test:

```python
def test_offspring_land_on_neighbors_not_source():
    # single occupied cell, pure F4Nr4 (4 neighbors). Offspring must arrive at the
    # 4 von-Neumann neighbors of the source, NOT back on the source cell.
    t = StrainTable()
    fid = t.get_or_mint(("F4Nr4",))
    w = World(5, 5, 64, DEV)
    w.strain_id[2, 2, 0] = fid; w.count[2, 2, 0] = 200; w.faction[2, 2, 0] = 1
    phe = t.phenotype_arrays(DEV)
    birth = torch.zeros((5, 5, 64), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    buf, live = phase2_reproduce(w, w.strain_id.clone(), w.count.clone(),
                                 w.faction.clone(), phe, t, birth, T=5, generator=g)
    ty, tx, sid, cnt, fac = buf.tensors()
    arrived = {(int(y), int(x)) for y, x, c in zip(ty, tx, cnt) if c > 0}
    neighbors = {(1, 2), (3, 2), (2, 1), (2, 3)}
    assert arrived <= neighbors, f"offspring landed off-neighbor: {arrived - neighbors}"
    assert (2, 2) not in arrived, "B4 regression: offspring deposited back on source cell"
    assert arrived, "no offspring scattered at all"

def test_offspring_inherit_parent_faction():
    t = StrainTable()
    fid = t.get_or_mint(("F4Nr4",))
    w = World(5, 5, 64, DEV)
    w.strain_id[2, 2, 0] = fid; w.count[2, 2, 0] = 200; w.faction[2, 2, 0] = 3
    phe = t.phenotype_arrays(DEV)
    birth = torch.zeros((5, 5, 64), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(1)
    buf, live = phase2_reproduce(w, w.strain_id.clone(), w.count.clone(),
                                 w.faction.clone(), phe, t, birth, T=5, generator=g)
    ty, tx, sid, cnt, fac = buf.tensors()
    assert (fac[cnt > 0] == 3).all(), "offspring did not inherit parent faction 3"
```

Update the two remaining existing tests (`test_reproduction_scatters_to_neighbor`, `test_no_reproduction_when_not_firing`) to pass `w.faction.clone()` as the new 4th arg and unpack a 5-tuple from `buf.tensors()`:

```python
    buf, live = phase2_reproduce(w, w.strain_id.clone(), w.count.clone(),
                                 w.faction.clone(), phe, t, birth, T=5, generator=g)
    ty, tx, sid, cnt, fac = buf.tensors()
```

(`test_no_reproduction_when_not_firing` keeps `T=1`; with F4Nr4 `repro_period=5`, `1 % 5 != 0` so nobody fires — still asserts `cnt.numel() == 0`. This test now depends on Task 7's per-phase clock for the period to be 5; under the pre-Task-7 `period` it is also 1→1%1==0 would fire. **Mark this single assertion as Task-7-dependent** in a comment; until Task 7 it may need `T=1` with the old min-period. To keep Task 6 self-contained, use a strain with NO P letter here: `fid = t.get_or_mint(("F4Nr4",))` already has period 5 under the OLD min-rule too, since F4Nr4's only period is 5. So this test passes under both rules — no Task-7 dependency. Leave as-is.)

- [ ] **Step 2: Run to verify they fail**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_reproduction.py -v`
Expected: FAIL — `add()` takes 4 positional args / `phase2_reproduce` missing `snap_faction` / `tensors()` returns 4-tuple.

- [ ] **Step 3: Implement — ArrivalBuffer (5-column)**

Replace the `ArrivalBuffer` class in `src/des/kernels/reproduction.py`:

```python
class ArrivalBuffer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._ty: list[torch.Tensor] = []
        self._tx: list[torch.Tensor] = []
        self._sid: list[torch.Tensor] = []
        self._cnt: list[torch.Tensor] = []
        self._fac: list[torch.Tensor] = []

    def add(self, ty, tx, sid, cnt, fac) -> None:
        m = cnt > 0
        if m.any():
            self._ty.append(ty[m]); self._tx.append(tx[m])
            self._sid.append(sid[m]); self._cnt.append(cnt[m])
            self._fac.append(fac[m])

    def tensors(self):
        if not self._cnt:
            z = torch.zeros(0, dtype=torch.int64, device=self.device)
            return (z, z, z.to(torch.int32), z.to(torch.int32),
                    z.to(torch.int8))
        return (torch.cat(self._ty), torch.cat(self._tx),
                torch.cat(self._sid), torch.cat(self._cnt),
                torch.cat(self._fac))
```

`_mutate_sequence` is UNCHANGED (it reads only the sequence — pure).

- [ ] **Step 4: Implement — vectorized `phase2_reproduce`**

Replace the entire `phase2_reproduce` function (the per-strain loop version) with this slot-level vectorized version. It reads `phe["repro_period"]` (added in Task 5) so reproduction already fires on the F-period; Task 7 only has to switch antagonism to `anta_period`.

```python
def phase2_reproduce(world, snap_sid, snap_count, snap_faction, phe, table,
                     birth_tick, T, generator):
    """Slot-level vectorized reproduction. Loops over the <=4 directions, NOT over
    strains. Offspring move to the neighbor (B4 fix: roll the data into place; read
    target coords from the static meshgrid -- never roll a coordinate grid). Offspring
    carry the parent slot's faction. Mutants are batch-minted once per tick."""
    from des.registry import ALL_DIRECTIONS
    H, W, K = snap_count.shape
    dev = world.device
    sid_long = snap_sid.long()

    f = phe["f"][sid_long]                       # [H,W,K]
    p_leave = phe["p_leave"][sid_long]
    p_x = phe["p_x"][sid_long]
    repro_period = phe["repro_period"][sid_long]
    dir_bits = phe["dir_bits"][sid_long]

    alive = snap_count > 0
    fires = fires_this_tick(birth_tick, repro_period, T) & alive & (f > 0)

    buf = ArrivalBuffer(dev)

    # static target-coordinate grids, broadcast to slot shape (NEVER rolled)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev),
                            torch.arange(W, device=dev), indexing="ij")
    ty = yy.unsqueeze(-1).expand(H, W, K).contiguous()   # [H,W,K]
    tx = xx.unsqueeze(-1).expand(H, W, K).contiguous()

    faction_long = snap_faction.to(torch.int8)

    # --- pass 1: per direction, compute non-mutant + mutant offspring, roll into place ---
    rolled = []                 # list of (rolled_non, rolled_mut, rolled_sid, rolled_fac)
    produced_mut = torch.zeros((H, W, K), dtype=torch.bool, device=dev)
    for (dy, dx) in ALL_DIRECTIONS:
        bit = ALL_DIRECTIONS.index((dy, dx))
        dir_mask = ((dir_bits >> bit) & 1).bool()        # slots that move this direction
        active = fires & dir_mask
        a = (snap_count * active).to(torch.int32)        # [H,W,K] firing counts
        scattered = binom(a, f, generator)               # offspring per source slot
        mut = binom(scattered, p_x, generator)           # mutant split
        non = scattered - mut
        produced_mut |= (mut > 0)
        # B4 fix: roll the DATA (counts, sid, faction) to the neighbor; coords stay static
        r_non = torch.roll(non, shifts=(dy, dx), dims=(0, 1))
        r_mut = torch.roll(mut, shifts=(dy, dx), dims=(0, 1))
        r_sid = torch.roll(snap_sid, shifts=(dy, dx), dims=(0, 1))
        r_fac = torch.roll(faction_long, shifts=(dy, dx), dims=(0, 1))
        rolled.append((r_non, r_mut, r_sid, r_fac))

    # --- batch-mint mutant children once per tick (deterministic: sorted parent order) ---
    n_before = len(table) + 1
    child_map = torch.arange(n_before, dtype=torch.int64, device=dev)  # parent->child sid
    mut_parents = torch.unique(sid_long[produced_mut])
    mutable = BB0_TEMPLATE["mutable"]
    for p in sorted(int(x) for x in mut_parents.tolist()):
        if p == 0:
            continue
        seq = table.sequence_of(p)
        spectrum = table.phenotype_of(p).spectrum
        child = table.get_or_mint(_mutate_sequence(seq, mutable, spectrum, generator))
        child_map[p] = child

    # --- pass 2: emit arrival records ---
    for (r_non, r_mut, r_sid, r_fac) in rolled:
        # non-mutant offspring keep the parent sid
        buf.add(ty.flatten(), tx.flatten(), r_sid.to(torch.int32).flatten(),
                r_non.flatten(), r_fac.flatten())
        # mutant offspring carry child sid (faction unchanged)
        child_sid = child_map[r_sid.long()].to(torch.int32)
        buf.add(ty.flatten(), tx.flatten(), child_sid.flatten(),
                r_mut.flatten(), r_fac.flatten())

    # --- migration out (unchanged: p_leave departers vanish, design line 115/309) ---
    leave = binom(snap_count, p_leave, generator)
    leave = torch.minimum(leave, snap_count)
    live = (world.count - leave).clamp(min=0)
    return buf, live
```

- [ ] **Step 5: Run to verify pass**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_reproduction.py -v`
Expected: PASS (5-arg buffer, movement-to-neighbor, faction inheritance, the two updated existing tests, and all `test_mutation_*` which call `_mutate_sequence` directly — unchanged).

**Determinism note for the reviewer:** generator consumption order changed (all direction binoms first, then one mint per mutant-parent in sorted sid order). This is NOT bit-identical to the old per-strain code — intentional per spec §1.12. Same seed still reproduces the same run because the order is fixed (sorted parents, fixed direction list). `child_map` indices are all parent sids `< len(table)+1` at entry, so no index overflow when new children get larger sids.

- [ ] **Step 6: Commit**

```bash
git add src/des/kernels/reproduction.py tests/test_reproduction.py
git commit -m "feat(P2+B4): vectorized faction-aware reproduction, offspring move to neighbors"
```

---

### Task 7: Antagonism fires on `anta_period` (per-phase clock, choice B)

**Files:**
- Modify: `src/des/kernels/antagonism.py` (the `fires_this_tick` call inside `phase1_antagonism`)
- Test: `tests/test_antagonism.py`

**Interfaces:**
- Consumes: `phe["anta_period"]` (Task 5).
- Produces: antagonism now fires on the Z-primitive period, not the old `min`-over-all `period`. For BB0 both are 5 vs old 1; the behavioural change is that a strain whose `min` period came from a P letter (period 1) no longer fights every tick.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_antagonism.py`. Build a strain whose Z-period (5) differs from its all-primitive min (1, from a P letter), and assert it does NOT fight on a tick that is not a multiple of 5:

```python
def test_antagonism_fires_on_anta_period_not_min_period():
    # ("BroadSweep","P_base"): BroadSweep period 5 (Z), P_base period 1 (P).
    # OLD min-period = 1 -> would fire every tick. NEW anta_period = 5 -> only T%5==0.
    t = StrainTable()
    pred = t.get_or_mint(("BroadSweep", "P_base"))
    prey = t.get_or_mint(("F4Nr4",))
    phe = t.phenotype_arrays(DEV)
    sid = torch.zeros((1, 1, 4), dtype=torch.int32)
    cnt = torch.zeros((1, 1, 4), dtype=torch.int32)
    fac = torch.zeros((1, 1, 4), dtype=torch.int8)
    sid[0, 0, 0] = pred; cnt[0, 0, 0] = 100; fac[0, 0, 0] = 0
    sid[0, 0, 1] = prey; cnt[0, 0, 1] = 100; fac[0, 0, 1] = 1
    birth = torch.zeros((1, 1, 4), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    # T=3 is NOT a multiple of anta_period 5 -> attacker does not fire -> no kills
    out3 = phase1_antagonism(sid, cnt, fac, phe, birth, T=3, z_max=8.0, generator=g)
    assert out3[0, 0, 1].item() == 100, "fired on a non-anta-period tick (still using min period?)"
    # T=5 IS a multiple -> fires -> prey takes losses
    out5 = phase1_antagonism(sid, cnt, fac, phe, birth, T=5, z_max=8.0, generator=g)
    assert out5[0, 0, 1].item() < 100, "did not fire on anta_period tick"
```

- [ ] **Step 2: Run to verify it fails**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_antagonism.py::test_antagonism_fires_on_anta_period_not_min_period -v`
Expected: FAIL — at T=3 the prey is killed because the kernel still uses `period` (min=1).

- [ ] **Step 3: Implement**

In `src/des/kernels/antagonism.py`, change the period gather + fire line. Replace:

```python
    period = phe["period"][sid_long]                 # [H, W, K]
```

with:

```python
    period = phe["anta_period"][sid_long]            # [H, W, K]  Z-primitive firing clock
```

(The `fires = fires_this_tick(birth_tick, period, T) & alive` line is unchanged — it now reads `anta_period`.)

- [ ] **Step 4: Run the antagonism suite**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_antagonism.py -v`
Expected: PASS. (The Task-3 tests used `T=5`, a multiple of 5, so they still fire and pass unchanged.)

- [ ] **Step 5: Commit**

```bash
git add src/des/kernels/antagonism.py tests/test_antagonism.py
git commit -m "feat(choice-B): antagonism fires on anta_period (Z-primitive clock)"
```

---

### Task 8: Vectorized arbitration K-wall (P1) — random-key sampling + faction keying

**Files:**
- Rewrite: `src/des/kernels/arbitration.py` (`phase3_arbitrate_vec` sections 1–2 vectorized + faction; DELETE reference `phase3_arbitrate`)
- Test: rewrite `tests/test_arbitration.py` (point at `_vec`, add faction), DELETE `tests/test_arbitration_vec_equiv.py`
- New test: `tests/test_arbitration_properties.py`

**Interfaces:**
- Consumes: arrivals 5-tuple `(a_ty, a_tx, a_sid, a_cnt, a_fac)` from `ArrivalBuffer.tensors()` (Task 6); `NFAC=4`.
- Produces: `phase3_arbitrate_vec(live_sid, live_count, live_faction, arrivals, K, birth_tick, T, generator, MAXSID, NFAC=4) -> (new_sid, new_count, new_faction, new_birth)` — added `live_faction` (3rd arg) and `new_faction` (3rd return). The per-cell Python loop with `int(tensor[...])` is GONE; the K-wall is a single random-key draw.
- **Red-line (spec §7-E/J):** the random keys are `torch.rand(...)` — i.i.d., NEVER read sid or faction. Survival prob = `avail/total` for every individual by construction. No per-faction quota.

- **Random-key K-wall theorem (spec §3.2):** "draw `avail` of `total` individuals without replacement" ⟺ "give each individual an i.i.d. uniform key, keep the `avail` smallest-key individuals per cell." Vectorized: expand contested records to individuals, assign keys, sort by `(cell, key)`, keep within-cell rank `< avail`.

- [ ] **Step 1: Write the property tests**

Create `tests/test_arbitration_properties.py`:

```python
# Property tests for vectorized arbitration (replaces the deleted bit-identical
# equivalence suite). Asserts the invariants the random-key K-wall must hold.
from __future__ import annotations
import torch
from des.kernels.arbitration import phase3_arbitrate_vec
from des.kernels.reproduction import ArrivalBuffer

DEV = torch.device("cpu")
MAXSID = 64
NFAC = 4


def _arrivals(events):
    """events: list of (y, x, sid, cnt, fac)."""
    buf = ArrivalBuffer(DEV)
    for (y, x, sid, cnt, fac) in events:
        buf.add(torch.tensor([y]), torch.tensor([x]),
                torch.tensor([sid], dtype=torch.int32),
                torch.tensor([cnt], dtype=torch.int32),
                torch.tensor([fac], dtype=torch.int8))
    return buf.tensors()


def _empty_state(H, W, K):
    z = lambda dt: torch.zeros((H, W, K), dtype=dt, device=DEV)
    return z(torch.int32), z(torch.int32), z(torch.int8), z(torch.int32)


def test_hard_cap_never_exceeds_K():
    K = 8
    sid, cnt, fac, birth = _empty_state(1, 1, K)
    arr = _arrivals([(0, 0, 5, 100, 0), (0, 0, 7, 100, 1)])  # 200 into empty cell
    g = torch.Generator(device=DEV); g.manual_seed(0)
    nsid, ncnt, nfac, nbirth = phase3_arbitrate_vec(
        sid, cnt, fac, arr, K, birth, T=1, generator=g, MAXSID=MAXSID, NFAC=NFAC)
    assert int(ncnt[0, 0].sum()) == K            # exactly avail=K seated

def test_conservation_uncontested_all_survive():
    K = 64
    sid, cnt, fac, birth = _empty_state(1, 1, K)
    arr = _arrivals([(0, 0, 5, 10, 0), (0, 0, 7, 20, 1)])   # 30 << 64, no thinning
    g = torch.Generator(device=DEV); g.manual_seed(0)
    nsid, ncnt, nfac, _ = phase3_arbitrate_vec(
        sid, cnt, fac, arr, K, birth, T=1, generator=g, MAXSID=MAXSID, NFAC=NFAC)
    seated = {(int(nsid[0,0,k]), int(nfac[0,0,k])): int(ncnt[0,0,k])
              for k in range(K) if ncnt[0,0,k] > 0}
    assert seated == {(5, 0): 10, (7, 1): 20}

def test_same_sid_different_faction_kept_separate():
    # red/blue BB0 (same sid, faction 0 vs 1) must NOT merge into one slot.
    K = 64
    sid, cnt, fac, birth = _empty_state(1, 1, K)
    arr = _arrivals([(0, 0, 5, 10, 0), (0, 0, 5, 15, 1)])   # same sid, diff faction
    g = torch.Generator(device=DEV); g.manual_seed(0)
    nsid, ncnt, nfac, _ = phase3_arbitrate_vec(
        sid, cnt, fac, arr, K, birth, T=1, generator=g, MAXSID=MAXSID, NFAC=NFAC)
    seated = {(int(nsid[0,0,k]), int(nfac[0,0,k])): int(ncnt[0,0,k])
              for k in range(K) if ncnt[0,0,k] > 0}
    assert seated == {(5, 0): 10, (5, 1): 15}    # two distinct slots

def test_faction_blind_equal_arrivals():
    # equal arrivals from two factions, heavily oversubscribed: survivors ~equal.
    K = 8
    accum = {0: 0, 1: 0}
    for seed in range(400):
        sid, cnt, fac, birth = _empty_state(1, 1, K)
        arr = _arrivals([(0, 0, 5, 100, 0), (0, 0, 7, 100, 1)])
        g = torch.Generator(device=DEV); g.manual_seed(seed)
        nsid, ncnt, nfac, _ = phase3_arbitrate_vec(
            sid, cnt, fac, arr, K, birth, T=1, generator=g, MAXSID=MAXSID, NFAC=NFAC)
        for k in range(K):
            if ncnt[0,0,k] > 0:
                accum[int(nfac[0,0,k])] += int(ncnt[0,0,k])
    ratio = accum[0] / max(1, accum[1])
    assert 0.85 < ratio < 1.18, f"faction-blind violated: {accum}, ratio={ratio:.3f}"

def test_resident_not_evicted():
    K = 8
    sid, cnt, fac, birth = _empty_state(1, 1, K)
    sid[0,0,0] = 9; cnt[0,0,0] = 5; fac[0,0,0] = 2          # resident: avail = 3
    arr = _arrivals([(0, 0, 5, 100, 0), (0, 0, 7, 100, 1)])
    g = torch.Generator(device=DEV); g.manual_seed(0)
    nsid, ncnt, nfac, _ = phase3_arbitrate_vec(
        sid, cnt, fac, arr, K, birth, T=1, generator=g, MAXSID=MAXSID, NFAC=NFAC)
    res = (nsid[0,0] == 9) & (nfac[0,0] == 2)
    assert int(ncnt[0,0][res].sum()) == 5                   # resident intact
    assert int(ncnt[0,0].sum()) == 8                        # 5 resident + 3 new
```

- [ ] **Step 2: Run to verify they fail**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_arbitration_properties.py -v`
Expected: FAIL — `phase3_arbitrate_vec` got an unexpected `live_faction` / returns a 3-tuple not 4-tuple.

- [ ] **Step 3: Implement — full rewrite of `arbitration.py`**

Replace the ENTIRE file `src/des/kernels/arbitration.py` with the version below. The reference `phase3_arbitrate` is deleted (it existed only for the now-deleted bit-identical equivalence suite). Sections 1 (coalesce) and 2 (K-wall) are now fully vectorized; section 3 (writeback) keeps the proven vectorized logic plus faction.

```python
# src/des/kernels/arbitration.py
from __future__ import annotations
import torch


def phase3_arbitrate_vec(live_sid, live_count, live_faction, arrivals, K,
                         birth_tick, T, generator, MAXSID, NFAC=4):
    """Fully vectorized K-wall arbitration with faction.

    Sections:
      1. coalesce arrivals by (cell, sid, faction)            -- tensor unique+scatter_add
      2. K-wall multivariate-hypergeometric draw via random keys (NO per-cell loop,
         NO .item() sync). Keys are i.i.d. uniform, never read sid/faction -> survival
         prob = avail/total for every individual (spec red-line 7-J, fair by construction).
      3. vectorized writeback to fixed-K slots, keyed on (sid, faction).
    """
    H, W, _ = live_count.shape
    dev = live_count.device
    a_ty, a_tx, a_sid, a_cnt, a_fac = arrivals

    # --- 1. coalesce arrivals by (cell, sid, faction) ---
    if a_cnt.numel() > 0:
        cell = a_ty.long() * W + a_tx.long()
        key = (cell * MAXSID + a_sid.long()) * NFAC + a_fac.long()
        uniq, inv = torch.unique(key, return_inverse=True)
        merged = torch.zeros(uniq.shape[0], dtype=torch.int64, device=dev)
        merged.scatter_add_(0, inv, a_cnt.long())
        u_fac = (uniq % NFAC).to(torch.int8)
        rest = uniq // NFAC
        u_sid = (rest % MAXSID).to(torch.int32)
        u_cell = rest // MAXSID
        u_y = (u_cell // W).long()
        u_x = (u_cell % W).long()
    else:
        u_y = torch.zeros(0, dtype=torch.long, device=dev)
        u_x = torch.zeros(0, dtype=torch.long, device=dev)
        u_sid = torch.zeros(0, dtype=torch.int32, device=dev)
        u_fac = torch.zeros(0, dtype=torch.int8, device=dev)
        merged = torch.zeros(0, dtype=torch.int64, device=dev)

    new_sid = live_sid.clone()
    new_cnt = live_count.clone()
    new_faction = live_faction.clone()
    new_birth = birth_tick.clone()
    if merged.numel() == 0:
        return new_sid, new_cnt, new_faction, new_birth

    # --- 2. K-wall: random-key multivariate hypergeometric (vectorized) ---
    n_rec = merged.shape[0]
    rec_cell = u_y * W + u_x                                  # [n_rec] cell of each record
    resident_occ = live_count.sum(dim=-1)                    # [H,W]
    avail_grid = (K - resident_occ).clamp(min=0)             # [H,W]
    # per-record avail and per-cell total (scatter over records sharing a cell)
    rec_avail = avail_grid.flatten()[rec_cell]               # [n_rec]
    # total individuals arriving per cell, broadcast back to each record
    cell_total = torch.zeros(H * W, dtype=torch.int64, device=dev)
    cell_total.scatter_add_(0, rec_cell, merged)
    rec_total = cell_total[rec_cell]                         # [n_rec]

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

    # --- 3. vectorized writeback, keyed on (sid, faction) ---
    mask = survived > 0
    if not mask.any():
        return new_sid, new_cnt, new_faction, new_birth
    v_y = u_y[mask]; v_x = u_x[mask]
    v_sid = u_sid[mask]; v_fac = u_fac[mask]
    v_cnt = survived[mask].to(torch.int32)
    N = v_y.shape[0]

    resident_sid = new_sid[v_y, v_x]                          # [N,K]
    resident_cnt = new_cnt[v_y, v_x]                          # [N,K]
    resident_fac = new_faction[v_y, v_x]                      # [N,K]
    # resident-hit: same sid AND same faction AND occupied
    hit = ((resident_sid == v_sid[:, None]) &
           (resident_fac == v_fac[:, None]) &
           (resident_cnt > 0))                                # [N,K]
    is_resident = hit.any(dim=1)
    is_new = ~is_resident

    if is_resident.any():
        r = is_resident
        r_k = hit[r].long().argmax(dim=1)
        new_cnt.index_put_((v_y[r], v_x[r], r_k), v_cnt[r], accumulate=True)

    if is_new.any():
        n = is_new
        n_y = v_y[n]; n_x = v_x[n]; n_sid = v_sid[n]; n_fac = v_fac[n]; n_cnt = v_cnt[n]
        Nnew = n_y.shape[0]
        cell_lin = n_y * W + n_x
        sort_ord = torch.argsort(cell_lin, stable=True)
        n_y = n_y[sort_ord]; n_x = n_x[sort_ord]; n_sid = n_sid[sort_ord]
        n_fac = n_fac[sort_ord]; n_cnt = n_cnt[sort_ord]; cell_lin = cell_lin[sort_ord]
        cell_change = torch.ones(Nnew, dtype=torch.bool, device=dev)
        cell_change[1:] = cell_lin[1:] != cell_lin[:-1]
        group_cumsum = cell_change.long().cumsum(0) - 1
        group_start = torch.searchsorted(group_cumsum.contiguous(),
                                         group_cumsum.contiguous())
        within_rank = torch.arange(Nnew, device=dev) - group_start
        cell_empty = (new_cnt[n_y, n_x] == 0)                 # [Nnew,K]
        empty_ord = cell_empty.long().cumsum(dim=1) - 1
        target_match = (empty_ord == within_rank[:, None]) & cell_empty
        assert target_match.any(dim=1).all(), \
            "C=K invariant violated: no empty slot for a new (sid,faction)"
        target_k = target_match.long().argmax(dim=1)
        new_sid.index_put_((n_y, n_x, target_k), n_sid, accumulate=False)
        new_cnt.index_put_((n_y, n_x, target_k), n_cnt.int(), accumulate=False)
        new_faction.index_put_((n_y, n_x, target_k), n_fac, accumulate=False)
        new_birth.index_put_((n_y, n_x, target_k),
                             torch.full((Nnew,), T, dtype=torch.int32, device=dev),
                             accumulate=False)

    return new_sid, new_cnt, new_faction, new_birth
```

- [ ] **Step 4: Migrate `test_arbitration.py` to the new API and delete the equivalence suite**

Delete the obsolete bit-identical equivalence suite (its reference function is gone):

```bash
git rm tests/test_arbitration_vec_equiv.py
```

Rewrite `tests/test_arbitration.py` to call `phase3_arbitrate_vec` with the faction arg. The existing tests (`test_arrivals_seat_into_empty_cell`, `test_convergent_arrivals_merge_same_strain`, `test_kwall_thins_when_over_capacity`, `test_kwall_order_independent`) keep their intent; update the helper and calls. Replace the top of the file:

```python
# tests/test_arbitration.py
import torch
from des.kernels.arbitration import phase3_arbitrate_vec
from des.kernels.reproduction import ArrivalBuffer

DEV = torch.device("cpu")
MAXSID = 64
NFAC = 4


def _arrivals(events):
    """events: list of (y, x, sid, cnt) — all faction 0 (single-faction K-wall tests)."""
    buf = ArrivalBuffer(DEV)
    for (y, x, sid, cnt) in events:
        buf.add(torch.tensor([y]), torch.tensor([x]),
                torch.tensor([sid], dtype=torch.int32),
                torch.tensor([cnt], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int8))
    return buf.tensors()
```

In each test, add a `fac` resident tensor and adapt the call + unpack. Pattern for every call site:

```python
    fac = torch.zeros((1, 1, K), dtype=torch.int8)
    nsid, ncnt, nfac, nbirth = phase3_arbitrate_vec(
        sid, cnt, fac, arr, K=K, birth_tick=birth, T=3, generator=g,
        MAXSID=MAXSID, NFAC=NFAC)
```

For `test_kwall_thins_when_over_capacity` the resident strain9 sits at `sid[0,0,0]=9` with `fac[0,0,0]=0`; arrivals are faction 0 too, so the strain-blind ratio check is unchanged in meaning (now faction-uniform, the K-wall is still sid/faction-blind). The assertions on counts (`new_total == 3`, resident intact, ratio within 15%) are unchanged.

- [ ] **Step 5: Run the arbitration suites**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_arbitration.py tests/test_arbitration_properties.py -v`
Expected: PASS (migrated legacy tests + 5 new property tests).

- [ ] **Step 6: Commit**

```bash
git add src/des/kernels/arbitration.py tests/test_arbitration.py tests/test_arbitration_properties.py
git rm tests/test_arbitration_vec_equiv.py
git commit -m "feat(P1): fully vectorized random-key K-wall arbitration with faction; drop reference impl + equiv suite"
```

---

### Task 9: Engine rewire + P3 stop-check downgrade (re-greens the whole suite)

**Files:**
- Modify: `src/des/engine.py` (`__init__`, `step`, `_fixated`, `run`, add `total_count`/extinction batching)
- Modify: `tests/test_engine.py`, `tests/test_integration.py` (migrate to faction)
- Migrate: `tests/test_integration.py::test_kwall_equal_ratio_no_hidden_weight` → call `_vec` with faction

**Interfaces:**
- Consumes: `init_factions` (Task 2), `phase1_antagonism(...,faction,...)` (Tasks 3/7), `phase2_reproduce(...,snap_faction,...)` (Task 6), `phase3_arbitrate_vec(...,live_faction,...,NFAC)` (Task 8).
- Produces: a working `Engine` whose tick threads faction through all three phases; `_fixated` = single-faction field; stop-check runs every `check_every` ticks.

- **P3 (spec §5):** `run()` currently calls `total_count()` (full sum+item) and `_fixated()` (full unique+item) EVERY tick — two GPU→CPU syncs per tick. Downgrade to every `check_every` (default 10) ticks. Overrunning a few ticks is harmless (data is dumped every tick regardless). `_fixated` uses the int8 `faction` tensor: `distinct = unique(faction[count>0])`, only ≤4 values, cheap.

- [ ] **Step 1: Write/adjust the failing tests**

Update `tests/test_engine.py` — the `Engine` now seeds four-quadrant factions, so add a fixation-semantics test and keep the determinism/liveness ones (they still hold). Append:

```python
def test_fixation_is_single_faction_field():
    # hand-build an engine state where only one faction survives -> _fixated True
    e = Engine(H=8, W=8, K=16, seed=0, device=DEV)
    e.world.count.zero_(); e.world.strain_id.zero_(); e.world.faction.zero_()
    e.world.strain_id[0, 0, 0] = 1; e.world.count[0, 0, 0] = 5; e.world.faction[0, 0, 0] = 2
    e.world.strain_id[1, 1, 0] = 1; e.world.count[1, 1, 0] = 5; e.world.faction[1, 1, 0] = 2
    assert e._fixated() is True            # all survivors faction 2
    e.world.faction[1, 1, 0] = 3           # introduce a second faction
    assert e._fixated() is False

def test_engine_seeds_four_factions():
    e = Engine(H=16, W=16, K=32, seed=0, device=DEV)
    live = e.world.count.sum(dim=-1) > 0
    assert int(live.sum()) == 4            # four quadrant centers seeded
    facs = e.world.faction[e.world.count > 0]
    assert set(facs.tolist()) == {0, 1, 2, 3}
```

`tests/test_integration.py`: `test_g10_no_self_annihilation_at_t0` used homogeneous BB0 (every cell same strain) to prove no t=0 self-kill. Under four-quadrant seeding the cells are isolated and same-faction within a cell, so the t=0-no-antagonism property still holds but for a different reason (no cross-faction contact yet). Replace its body:

```python
def test_g10_no_self_annihilation_at_t0():
    # four isolated faction seeds: no two factions share a cell at t=0 -> no antagonism
    # losses on the first tick. World count does not drop on tick 0.
    e = Engine(H=16, W=16, K=32, seed=0, device=DEV, fill_per_cell=20)
    before = e.total_count()
    e.step()
    assert e.total_count() >= before
```

Migrate `test_kwall_equal_ratio_no_hidden_weight` to the vectorized signature (it imported the deleted `phase3_arbitrate`):

```python
def test_kwall_equal_ratio_no_hidden_weight():
    from des.kernels.arbitration import phase3_arbitrate_vec
    from des.kernels.reproduction import ArrivalBuffer
    survivors = {5: 0, 7: 0}
    for s in range(200):
        sid = torch.zeros((1,1,16), dtype=torch.int32)
        cnt = torch.zeros((1,1,16), dtype=torch.int32)
        fac = torch.zeros((1,1,16), dtype=torch.int8)
        birth = torch.zeros((1,1,16), dtype=torch.int32)
        buf = ArrivalBuffer(DEV)
        buf.add(torch.tensor([0]),torch.tensor([0]),torch.tensor([5],dtype=torch.int32),
                torch.tensor([100],dtype=torch.int32),torch.tensor([0],dtype=torch.int8))
        buf.add(torch.tensor([0]),torch.tensor([0]),torch.tensor([7],dtype=torch.int32),
                torch.tensor([100],dtype=torch.int32),torch.tensor([1],dtype=torch.int8))
        g = torch.Generator(device=DEV); g.manual_seed(s)
        nsid, ncnt, nfac, _ = phase3_arbitrate_vec(sid, cnt, fac, buf.tensors(), K=16,
                                 birth_tick=birth, T=1, generator=g, MAXSID=16, NFAC=4)
        for k in range(16):
            if ncnt[0,0,k] > 0:
                survivors[int(nsid[0,0,k])] += int(ncnt[0,0,k])
    r = survivors[5] / max(1, survivors[7])
    assert 0.85 < r < 1.18, f"strain-blind violated: ratio={r}"
```

(`test_phenotype_reads_only_sequence` is UNCHANGED and must stay green — it guards the core invariant that `phenotype` takes only `sequence`.)

- [ ] **Step 2: Run to verify the engine suite fails**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_engine.py -v`
Expected: FAIL — `step()` unpacks a 2-tuple snapshot / kernels missing faction args.

- [ ] **Step 3: Implement — rewrite `engine.py`**

Replace the entire `src/des/engine.py`:

```python
# src/des/engine.py
from __future__ import annotations
import torch
from des.phenotype_cache import StrainTable
from des.world import init_factions
from des.kernels.antagonism import phase1_antagonism
from des.kernels.reproduction import phase2_reproduce
from des.kernels.arbitration import phase3_arbitrate_vec

NFAC = 4


class Engine:
    def __init__(self, H, W, K, seed, device, z_max=8.0, fill_per_cell=None,
                 check_every=10):
        self.H, self.W, self.K, self.device, self.z_max = H, W, K, device, z_max
        self.check_every = check_every
        self.table = StrainTable()
        fill = K // 2 if fill_per_cell is None else fill_per_cell
        self.world = init_factions(H, W, K, device, self.table,
                                   fill_per_cell=fill, n_fac=NFAC)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.birth = torch.zeros((H, W, K), dtype=torch.int32, device=device)
        self.T = 0
        self._phe = self.table.phenotype_arrays(device)
        self._phe_n = len(self.table)

    def _refresh_phe(self):
        if len(self.table) != self._phe_n:
            self._phe = self.table.phenotype_arrays(self.device)
            self._phe_n = len(self.table)

    def step(self) -> None:
        snap_sid, snap_count, snap_faction = self.world.snapshot()
        # PHASE1: antagonism (faction-gated) -> post-antagonism counts
        post_anta = phase1_antagonism(snap_sid, snap_count, snap_faction, self._phe,
                                      self.birth, self.T, self.z_max, self.gen)
        self.world.count = post_anta
        # PHASE2: reproduction — amounts from snapshot, space from post-antagonism world
        buf, live = phase2_reproduce(self.world, snap_sid, snap_count, snap_faction,
                                     self._phe, self.table, self.birth, self.T, self.gen)
        self.world.count = live
        self._refresh_phe()
        # PHASE3: arbitration (faction-keyed)
        arrivals = buf.tensors()
        nsid, ncnt, nfac, nbirth = phase3_arbitrate_vec(
            self.world.strain_id, self.world.count, self.world.faction, arrivals, self.K,
            self.birth, self.T, self.gen, MAXSID=len(self.table) + 1, NFAC=NFAC)
        self.world.strain_id, self.world.count = nsid, ncnt
        self.world.faction, self.birth = nfac, nbirth
        self.T += 1

    def total_count(self) -> int:
        return int(self.world.count.sum())

    def distinct_strains(self) -> int:
        present = self.world.strain_id[self.world.count > 0]
        return int(torch.unique(present).numel())

    def _fixated(self) -> bool:
        # single-faction field: only one faction among all living individuals
        facs = self.world.faction[self.world.count > 0]
        return facs.numel() > 0 and int(torch.unique(facs).numel()) == 1

    def run(self, ticks, recorder=None, stop_on=("fixation", "extinction")) -> int:
        ran = 0
        for _ in range(ticks):
            self.step()
            ran += 1
            if recorder is not None:
                recorder.dump(self.T, self.world)
            # P3: stop-check is GPU->CPU sync; run it only every check_every ticks
            if ran % self.check_every == 0:
                if "extinction" in stop_on and self.total_count() == 0:
                    break
                if "fixation" in stop_on and self._fixated() and ran > 1:
                    break
        return ran
```

- [ ] **Step 4: Run the WHOLE suite (this task re-greens everything)**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest -v`
Expected: PASS across the entire suite — engine, integration (migrated), world, registry, phenotype_cache, antagonism, reproduction, arbitration, arbitration_properties, recorder, types, kernel_common, smoke. No reference to deleted `phase3_arbitrate` or `test_arbitration_vec_equiv.py` remains.

- [ ] **Step 5: Commit**

```bash
git add src/des/engine.py tests/test_engine.py tests/test_integration.py
git commit -m "feat(P3): engine threads faction, single-faction fixation, batched stop-check"
```

---

### Task 10: run_batch → T=450, four-quadrant config, perf profiler ruler

**Files:**
- Modify: `scripts/run_batch.py` (T=450, comment update; `init_factions` is used via `Engine` already)
- No test file (script); validated by the `--probe` perf ruler in this task and Task 11's acceptance run.

**Interfaces:**
- Consumes: `Engine` (already seeds four-quadrant factions via Task 9).
- Produces: a batch runner with the locked config `128×128 / K=64 / fill=20 / T=450 / seeds [0,1,2,3]` and a `--probe` mode that prints per-phase ms/tick.

- [ ] **Step 1: Update the locked T and header comment**

In `scripts/run_batch.py`, change `T = 200` to:

```python
T = 450               # 2026-06-21: F4Nr4 repro_period=5 -> meet ~tick160, fill ~tick320,
                      # T covers front transient + meeting-band antagonism + post-fill red-queen
```

Update the module docstring's "T=200 ticks" to "T=450 ticks" and the seeding line to note four-quadrant faction seeding (Engine seeds via `init_factions`, not full-field `init_bb0`).

- [ ] **Step 2: Add a per-phase timing probe**

Add a `--phase-probe N` mode that times each phase separately with `torch.cuda.synchronize()` fences (proves the §0 static attribution that P1/P2 were the cost). Append to `run_batch.py`:

```python
def phase_probe(seed: int, ticks: int, device: torch.device) -> None:
    """Per-phase ms/tick on the locked grid, to verify P1/P2 were the hot spots."""
    import time
    from des.kernels.antagonism import phase1_antagonism
    from des.kernels.reproduction import phase2_reproduce
    from des.kernels.arbitration import phase3_arbitrate_vec
    e = Engine(H=H, W=W, K=K, seed=seed, device=device, z_max=Z_MAX, fill_per_cell=FILL)
    acc = {"anta": 0.0, "repro": 0.0, "arb": 0.0}
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    for _ in range(ticks):
        ssid, scnt, sfac = e.world.snapshot()
        sync(); t0 = time.perf_counter()
        post = phase1_antagonism(ssid, scnt, sfac, e._phe, e.birth, e.T, e.z_max, e.gen)
        sync(); t1 = time.perf_counter(); e.world.count = post
        buf, live = phase2_reproduce(e.world, ssid, scnt, sfac, e._phe, e.table,
                                     e.birth, e.T, e.gen)
        sync(); t2 = time.perf_counter(); e.world.count = live; e._refresh_phe()
        nsid, ncnt, nfac, nbirth = phase3_arbitrate_vec(
            e.world.strain_id, e.world.count, e.world.faction, buf.tensors(), e.K,
            e.birth, e.T, e.gen, MAXSID=len(e.table) + 1, NFAC=4)
        sync(); t3 = time.perf_counter()
        e.world.strain_id, e.world.count, e.world.faction, e.birth = nsid, ncnt, nfac, nbirth
        e.T += 1
        acc["anta"] += (t1 - t0); acc["repro"] += (t2 - t1); acc["arb"] += (t3 - t2)
    n = max(1, ticks)
    print(f"[phase-probe {ticks} ticks] "
          f"anta {1000*acc['anta']/n:.1f} | repro {1000*acc['repro']/n:.1f} | "
          f"arb {1000*acc['arb']/n:.1f} ms/tick (occupancy grows over the run)")
```

Wire it into `main()`: add the argument alongside the existing `--probe`/`--cpu` args, and add the dispatch after the existing `if args.probe:` block:

```python
    # add next to the existing ap.add_argument(...) lines:
    ap.add_argument("--phase-probe", type=int, default=0,
                    help="per-phase ms/tick timing on the locked grid")
```
```python
    # add right after the existing `if args.probe: ... return` block:
    if args.phase_probe:
        phase_probe(0, args.phase_probe, device)
        return
```

- [ ] **Step 3: Run the perf ruler (acceptance #10 evidence)**

Run a short probe (no parquet) to confirm the order-of-magnitude win and locate the new hot spot:

Run: `D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --probe 60`
Expected: `ms/tick` an order of magnitude below the first batch's 12.8 s/tick (target < 1000 ms/tick at 128/K64). Also run `--phase-probe 60` to see the per-phase split.

If the probe is still > 1 s/tick, do NOT proceed to the full batch — re-profile (the §6 perf-ruler clause): the remaining cost is either the dense `[H,W,K,K]` antagonism tensor (expected, deferred C1) or an unvectorized path that slipped through.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_batch.py
git commit -m "chore: run_batch T=450 four-quadrant config + per-phase perf probe"
```

---

### Task 11: End-to-end acceptance test (spec §8 criteria as assertions)

**Files:**
- New: `tests/test_acceptance.py`

**Interfaces:**
- Consumes: `Engine`, `Recorder`, `init_factions`, `phase3_arbitrate_vec`.
- Produces: a small-grid acceptance suite asserting the spec §8 macro-dynamics criteria that unit tests don't cover (seeding, monotone expansion, cross-faction contact → losses, faction-blind cross-seed win-rate, phenotype-no-faction grep guard). Runs on a small grid so it's CI-fast; the full 128² batch is Task 10's job, not a test.

- [ ] **Step 1: Write the acceptance tests**

Create `tests/test_acceptance.py`:

```python
# Spec §8 acceptance criteria as runnable assertions (small grid, CI-fast).
from __future__ import annotations
import inspect
import torch
from des.engine import Engine
from des.world import init_factions
from des.phenotype_cache import StrainTable

DEV = torch.device("cpu")


def test_acc1_seeding_exactly_four_cells():
    t = StrainTable()
    w = init_factions(32, 32, 64, DEV, t, fill_per_cell=20, n_fac=4)
    assert int((w.count.sum(dim=-1) > 0).sum()) == 4
    facs = w.faction[w.count > 0]
    assert set(facs.tolist()) == {0, 1, 2, 3}

def test_acc2_expansion_is_monotone_then_grows():
    # occupancy (number of non-empty cells) strictly grows over early ticks: from 4 up.
    e = Engine(H=32, W=32, K=64, seed=0, device=DEV, fill_per_cell=20)
    occ0 = int((e.world.count.sum(dim=-1) > 0).sum())
    assert occ0 == 4
    # repro_period=5 -> first expansion lands at T=5. Run 12 ticks: must exceed 4.
    e.run(12, stop_on=())
    occ1 = int((e.world.count.sum(dim=-1) > 0).sum())
    assert occ1 > occ0, f"world did not expand from 4 (got {occ1})"

def test_acc3_cross_faction_contact_causes_losses():
    # tiny grid so fronts meet fast; after enough ticks some cell holds >1 faction
    # and total count is not just monotonically growing (antagonism removes some).
    e = Engine(H=8, W=8, K=64, seed=1, device=DEV, fill_per_cell=20)
    e.run(30, stop_on=())
    # at least one cell now holds two different factions (contact happened)
    fac = e.world.faction; cnt = e.world.count
    multi = 0
    for y in range(8):
        for x in range(8):
            live = cnt[y, x] > 0
            if live.any() and torch.unique(fac[y, x][live]).numel() > 1:
                multi += 1
    assert multi > 0, "no cross-faction contact after 30 ticks on an 8x8 grid"

def test_acc4_same_faction_never_fights():
    # a whole grid of ONE faction (all same faction, mixed strains) -> antagonism
    # removes nobody, ever. Single-faction field => no kills.
    from des.kernels.antagonism import phase1_antagonism
    t = StrainTable()
    pred = t.get_or_mint(("BroadSweep",))
    prey = t.get_or_mint(("F4Nr4",))
    phe = t.phenotype_arrays(DEV)
    sid = torch.zeros((2, 2, 4), dtype=torch.int32)
    cnt = torch.zeros((2, 2, 4), dtype=torch.int32)
    fac = torch.zeros((2, 2, 4), dtype=torch.int8)   # ALL faction 0
    sid[..., 0] = pred; cnt[..., 0] = 50
    sid[..., 1] = prey; cnt[..., 1] = 50
    birth = torch.zeros((2, 2, 4), dtype=torch.int32)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    out = phase1_antagonism(sid, cnt, fac, phe, birth, T=5, z_max=8.0, generator=g)
    assert torch.equal(out, cnt), "same-faction field suffered antagonism losses"

def test_acc5_phenotype_path_has_no_faction_index():
    # red-line static guard: the phenotype function signature takes only `sequence`,
    # and the phenotype_arrays dict keys never include a faction axis.
    from des.registry import phenotype
    assert list(inspect.signature(phenotype).parameters) == ["sequence"]
    t = StrainTable(); t.get_or_mint(("F4Nr4",))
    arr = t.phenotype_arrays(DEV)
    assert "faction" not in arr and "fac" not in arr

def test_acc6_faction_blind_cross_seed_winrate():
    # symmetry-group sneak-goods self-check (spec §7 theorem, weak CI form): across
    # seeds, no faction should systematically dominate the EARLY expansion. Measure
    # occupied-cell share per faction at a fixed early tick; each faction's mean share
    # ~1/4. This is a coarse CI guard (the real 1/4 binomial CI check is the batch run).
    shares = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    SEEDS = 8
    for seed in range(SEEDS):
        e = Engine(H=24, W=24, K=64, seed=seed, device=DEV, fill_per_cell=20)
        e.run(10, stop_on=())
        cnt = e.world.count; fac = e.world.faction
        live = cnt > 0
        facs = fac[live]
        total = facs.numel()
        if total == 0:
            continue
        for fk in shares:
            shares[fk] += float((facs == fk).sum()) / total
    means = {k: v / SEEDS for k, v in shares.items()}
    # symmetric seeding -> each faction's mean occupied share within [0.15, 0.35] of 0.25
    for k, m in means.items():
        assert 0.15 < m < 0.35, f"faction {k} mean share {m:.3f} off symmetric 0.25: {means}"
```

- [ ] **Step 2: Run to verify (these should PASS once Tasks 1–9 are in)**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_acceptance.py -v`
Expected: PASS. If `test_acc6` is flaky at this small scale, widen the band to `[0.10, 0.40]` and note it — the authoritative 1/4-CI check is the full batch (Task 10), not this CI-fast proxy.

- [ ] **Step 3: Run the FULL suite one final time**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest -v`
Expected: all green. This is the spec §8 criterion #11 (test suite green; `phase3_arbitrate` reference + equiv suite gone).

- [ ] **Step 4: Commit**

```bash
git add tests/test_acceptance.py
git commit -m "test: spec §8 acceptance criteria as end-to-end assertions"
```

- [ ] **Step 5: (manual, not a code step) Run the batch & eyeball §8**

After the suite is green, run the real batch and verify the macro-dynamics by eye against spec §8 (#2 meet ~160 / fill ~320, #3 meeting-band losses, #8 fixation may or may not trigger — co-existence at T=450 is a legal outcome):

Run: `D:/anaconda3/envs/basic/python.exe scripts/run_batch.py`
This produces four parquet files under `data/runs/`. Inspect occupancy-over-tick and per-faction counts; this is the project's actual deliverable, not a unit test.

---

## Spec coverage map (self-review)

| Spec section | Task(s) |
|---|---|
| §1 decisions (faction immutable, neutralize-only, 4×BB0, four-quadrant, F4Nr4, torus, fixation=single-faction) | 1 (F4Nr4), 2 (seeding+faction), 3 (neutralize/immune), 9 (fixation) |
| §2.1 world.py (faction tensor, snapshot) | 2 |
| §2.2 antagonism predicate | 3 |
| §2.3 recorder 6-col | 4 |
| §2.4 StrainTable untouched | (invariant — no task modifies `phenotype_cache` identity logic; Task 5 only adds gathered arrays) |
| §3.1 coalesce key+faction | 8 |
| §3.2 random-key K-wall | 8 |
| §3.3 writeback (sid,faction) | 8 |
| §3.4 delete reference impl | 8 |
| §4.1 ArrivalBuffer 5-col | 6 |
| §4.2 per-strain loop elimination | 6 |
| §4.3 batch mutant minting | 6 |
| §4.4 migration unchanged | 6 |
| §5 P3 stop-check downfreq | 9 |
| §6 test suite rewrite (property + perf ruler) | 8 (property), 10 (perf ruler), 11 (acceptance) |
| §7 11-item sneak-goods audit | enforced as red-line notes in Tasks 3/5/6/8; asserted in 11 (acc5) |
| §8 acceptance #1–11 | 11 (acc1–6), 10 (#10 perf), 9 (#11 suite green) |
| §9 YAGNI (dense tensor C1, calibration, κ/α₀/β) | out of scope — not touched; antagonism stays dense (noted Task 10 Step 3) |
| choice-B per-phase clock | 5 (compute), 6 (repro uses), 7 (anta uses) |
| B4 movement bug | 6 |

**Not a separate task by design:** §2.4 (StrainTable phenotype identity) is a *don't-touch* invariant — same sequence + different faction must share one cache row. No task alters `get_or_mint`/`phenotype_of`; Task 5 only appends gathered arrays. The dual-orthogonal identity holds because faction never enters `phe[sid]`.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-21-faction-and-vectorization.md`.

**Dependency order (strict):** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11. Tasks 1–8 each gate on their OWN test file only (engine-level suites are expectedly red from Task 2 until Task 9 — see the Test-sequencing note). Task 9 re-greens the full suite; do not mark it done until `pytest -v` is fully green. Task 10's batch run is gated on the perf ruler being < 1 s/tick. Task 11 is the §8 acceptance gate.
