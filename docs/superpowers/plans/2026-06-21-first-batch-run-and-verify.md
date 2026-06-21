# First-Batch Run + Data-Verify Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the locked first batch (T=450, 4 seeds, no early stop) and add a report-only `analyze_batch.py` that prints per-seed + cross-seed dynamics metrics from the parquets and NEVER emits PASS/FAIL.

**Architecture:** One-line change to `scripts/run_batch.py` (pass `stop_on=()` so all 4 seeds run full length). New pure-function analysis module `scripts/analyze_batch.py`: `load → per_seed_metrics → cross_seed_metrics → render_text + dump_json`. Metric functions take a pandas DataFrame and return plain dicts (testable, no I/O, no verdicts). A synthetic-parquet unit test pins the metric math.

**Tech Stack:** Python, pandas 3.0.1, pyarrow 22, numpy 2.3.5. Engine already on main (frozen — NOT modified here). Interpreter: `D:/anaconda3/envs/basic/python.exe`, imports via `PYTHONPATH=src`.

## Global Constraints

- **报数不判决:** analysis code computes numbers only; NEVER prints/returns PASS/FAIL or fires a GATE. (spec §0.1)
- **反推量标 proxy:** parquet schema is exactly `(tick, cell_x, cell_y, strain, faction, count)` — 6 cols, only non-empty rows. "kills/减员" is NOT a column; any count-drop figure is a proxy conflating K-wall evaporation + p_leave migration + arbitration. Label it proxy, never "kills". (spec §0.2)
- **不脑补:** if data can't show it, the report says "不可从快照辨识" — no mechanistic guessing. (spec §0.3)
- **裁定是人:** the script presents numbers; the user makes the dynamics verdict. (spec §4)
- **earliest tick = 1:** `engine.run` dumps after `T` increments, so parquet has no tick 0. Seeding check uses `min(tick)`; `repro_period=5` means ticks 1–4 are still the 4 static seed cells. (verified against `src/des/engine.py:66-79`)
- **YAGNI (out of scope):** β<0 freq-dependence regression, ACF/PSD red-queen metrics, three-null-model thresholds, κ/α₀/β_fold effects, z_eff calibration, any GATE verdict. (spec §1)
- **Engine frozen:** do NOT modify `src/des/*`. Only `scripts/` + `tests/`.

## File Structure

- `scripts/run_batch.py` (MODIFY, 1 line) — pass `stop_on=()` to the full-batch run so all 4 seeds run T=450.
- `scripts/analyze_batch.py` (CREATE) — report-only analysis. Pure metric functions + text/JSON renderers + CLI. One file: small, cohesive, all the analysis lives together.
- `tests/test_analyze_batch.py` (CREATE) — synthetic-parquet unit tests pinning the metric math (no real batch, CI-fast).

Module layout of `analyze_batch.py` (functions defined in dependency order, used by Tasks 2–6):

```
load(path) -> pd.DataFrame                            # Task 2
survival_spatial_metrics(df, n_cells) -> dict         # Task 3
diversity_metrics(df) -> dict                         # Task 4
proxy_and_seeding_metrics(df) -> dict                 # Task 4
per_seed_metrics(df, n_cells) -> dict                 # Task 4 (aggregates 3+4)
cross_seed_metrics(per_seed_list) -> dict             # Task 5
render_text(per_seed_list, cross) -> str              # Task 6
dump_json(report, path) -> None                       # Task 6
main()                                                # Task 6 (CLI)
```

---

## Task 1: run_batch — disable early stop for the full batch

**Files:**
- Modify: `scripts/run_batch.py` (the `run_one` function's `e.run(...)` call)

**Interfaces:**
- Consumes: `Engine.run(ticks, recorder=None, stop_on=("fixation","extinction"))` (engine, frozen).
- Produces: nothing for later tasks (independent deliverable). After this, a real batch writes 4 parquets each with exactly T=450 ticks recorded.

- [ ] **Step 1: Make the edit**

In `scripts/run_batch.py`, find the line in `run_one`:

```python
    ran = e.run(ticks, recorder=rec)
```

Change it to:

```python
    ran = e.run(ticks, recorder=rec, stop_on=())   # spec §3.1: run full T, no early stop
```

With `stop_on=()` both `"extinction" in ()` and `"fixation" in ()` are False, so the loop always runs all `ticks`. fixation is still recoverable from data (distinct factions → 1) but the world keeps evolving so post-fixation dynamics are captured. (spec §3.1)

- [ ] **Step 2: Verify the probe path still runs (no real batch yet)**

Run:
```bash
PYTHONPATH=src D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --probe 20
```
Expected: prints one `[probe 20 ticks] ... ms/tick ... peak ... GB | strains-> ... | total ...` line and an `est full batch` line, exit 0. The `--probe` path calls `run_one(..., record=False)` using the same `e.run` call you edited, confirming the edit is valid and 20 ticks complete.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_batch.py
git commit -m "feat: run_batch full T=450, no early stop (spec §3.1)"
```

---

## Task 2: load() + synthetic-parquet test harness

**Files:**
- Create: `scripts/analyze_batch.py`
- Create: `tests/test_analyze_batch.py`

**Interfaces:**
- Produces: `load(path) -> pd.DataFrame` with columns `tick, cell_x, cell_y, strain, faction, count`. `_toy(path, rows)` test helper writing a schema-shaped parquet from a list of 6-tuples.

- [ ] **Step 1: Write the failing test**

`tests/test_analyze_batch.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import pandas as pd
import analyze_batch as ab

def _toy(path, rows):
    """rows: list of (tick, cell_x, cell_y, strain, faction, count)."""
    df = pd.DataFrame(rows, columns=["tick", "cell_x", "cell_y", "strain", "faction", "count"])
    df.to_parquet(path)
    return df

def test_load_roundtrips_schema(tmp_path):
    p = tmp_path / "t.parquet"
    _toy(p, [(1, 0, 0, "S0", 0, 10), (1, 1, 1, "S0", 1, 10)])
    df = ab.load(str(p))
    assert list(df.columns) == ["tick", "cell_x", "cell_y", "strain", "faction", "count"]
    assert len(df) == 2
    assert df["strain"].tolist() == ["S0", "S0"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src D:/anaconda3/envs/basic/python.exe -m pytest tests/test_analyze_batch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analyze_batch'`.

- [ ] **Step 3: Write minimal implementation**

`scripts/analyze_batch.py`:
```python
#!/usr/bin/env python
"""Report-only analysis of first-batch parquets. Computes per-seed + cross-seed
dynamics metrics and PRINTS them. NEVER emits PASS/FAIL — the human judges
(spec §0.1, §4). 'kills/减员' is not a column; count-drop figures are PROXIES
conflating K-wall evaporation + p_leave + arbitration (spec §0.2)."""
from __future__ import annotations
import pyarrow.parquet as pq
import pandas as pd


def load(path: str) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src D:/anaconda3/envs/basic/python.exe -m pytest tests/test_analyze_batch.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_batch.py tests/test_analyze_batch.py
git commit -m "feat: analyze_batch load() + synthetic-parquet test harness"
```

---

## Task 3: per-seed survival + spatial + faction metrics

**Files:**
- Modify: `scripts/analyze_batch.py`
- Modify: `tests/test_analyze_batch.py`

**Interfaces:**
- Consumes: `load`, `_toy` (Task 2).
- Produces: `survival_spatial_metrics(df, n_cells=16384) -> dict` with keys: `ticks` (sorted list), `total_count` ({tick:int}), `extinction_tick` (int|None), `distinct_factions` ({tick:int}), `fixation_tick` (int|None), `occupied_cells` ({tick:int}), `first_cross_faction_tick` (int|None), `fill_tick` (int|None), `faction_occupied` ({tick:{faction:int}}), `faction_share` ({tick:{faction:float}}), `winner_faction` (int|None).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analyze_batch.py`:
```python
def test_survival_spatial_known_answers(tmp_path):
    # 2x2 grid (n_cells=4). tick1: 4 separate cells, factions 0..3, strain S0.
    # tick2: faction1 invades cell(0,0) -> cross-faction at tick2.
    # tick3: only faction 0 remains anywhere -> fixation at tick3.
    rows = [
        (1, 0, 0, "S0", 0, 10), (1, 1, 0, "S0", 1, 10),
        (1, 0, 1, "S0", 2, 10), (1, 1, 1, "S0", 3, 10),
        (2, 0, 0, "S0", 0, 8), (2, 0, 0, "S0", 1, 5),   # cell(0,0) now 2 factions
        (2, 1, 0, "S0", 1, 10), (2, 0, 1, "S0", 2, 10), (2, 1, 1, "S0", 3, 10),
        (3, 0, 0, "S0", 0, 20), (3, 1, 0, "S0", 0, 5),  # only faction 0 left
    ]
    p = tmp_path / "s.parquet"; _toy(p, rows)
    m = ab.survival_spatial_metrics(ab.load(str(p)), n_cells=4)
    assert m["ticks"] == [1, 2, 3]
    assert m["total_count"] == {1: 40, 2: 43, 3: 25}
    assert m["extinction_tick"] is None
    assert m["distinct_factions"] == {1: 4, 2: 4, 3: 1}
    assert m["fixation_tick"] == 3
    assert m["occupied_cells"] == {1: 4, 2: 4, 3: 1}
    assert m["first_cross_faction_tick"] == 2
    assert m["fill_tick"] == 1            # 4 occupied cells == n_cells at tick 1
    assert m["winner_faction"] == 0
    assert abs(m["faction_share"][1][0] - 0.25) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src D:/anaconda3/envs/basic/python.exe -m pytest tests/test_analyze_batch.py::test_survival_spatial_known_answers -v`
Expected: FAIL — `AttributeError: module 'analyze_batch' has no attribute 'survival_spatial_metrics'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/analyze_batch.py`:
```python
def survival_spatial_metrics(df: pd.DataFrame, n_cells: int = 128 * 128) -> dict:
    ticks = sorted(int(t) for t in df["tick"].unique())
    full = list(range(min(ticks), max(ticks) + 1)) if ticks else []

    tot = df.groupby("tick")["count"].sum()
    total_count = {int(t): int(tot.get(t, 0)) for t in full}
    extinction_tick = next((t for t in full if total_count[t] == 0), None)

    live = df[df["count"] > 0]
    dfac = live.groupby("tick")["faction"].nunique()
    distinct_factions = {int(t): int(dfac.get(t, 0)) for t in ticks}
    fixation_tick = next((t for t in ticks if distinct_factions[t] == 1), None)

    def _occ(g):
        return g[["cell_x", "cell_y"]].drop_duplicates().shape[0]
    occ = live.groupby("tick")[["cell_x", "cell_y"]].apply(_occ)
    occupied_cells = {int(t): int(occ.get(t, 0)) for t in ticks}
    fill_tick = next((t for t in ticks if occupied_cells[t] >= n_cells), None)

    per_cell_fac = live.groupby(["tick", "cell_x", "cell_y"])["faction"].nunique()
    cross = per_cell_fac[per_cell_fac > 1]
    first_cross_faction_tick = (
        int(cross.index.get_level_values("tick").min()) if len(cross) else None)

    fac_occ = live.groupby(["tick", "faction"])[["cell_x", "cell_y"]].apply(_occ)
    faction_occupied = {int(t): {} for t in ticks}
    for (t, f), v in fac_occ.items():
        faction_occupied[int(t)][int(f)] = int(v)

    fac_cnt = live.groupby(["tick", "faction"])["count"].sum()
    faction_share = {int(t): {} for t in ticks}
    for (t, f), v in fac_cnt.items():
        denom = total_count.get(int(t), 0) or 1
        faction_share[int(t)][int(f)] = float(v) / denom

    last = ticks[-1] if ticks else None
    winner_faction = None
    if last is not None and faction_share[last]:
        winner_faction = int(max(faction_share[last], key=faction_share[last].get))

    return {
        "ticks": ticks, "total_count": total_count, "extinction_tick": extinction_tick,
        "distinct_factions": distinct_factions, "fixation_tick": fixation_tick,
        "occupied_cells": occupied_cells, "first_cross_faction_tick": first_cross_faction_tick,
        "fill_tick": fill_tick, "faction_occupied": faction_occupied,
        "faction_share": faction_share, "winner_faction": winner_faction,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src D:/anaconda3/envs/basic/python.exe -m pytest tests/test_analyze_batch.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_batch.py tests/test_analyze_batch.py
git commit -m "feat: per-seed survival/spatial/faction metrics"
```

---

## Task 4: diversity + oligopoly + proxy + seeding metrics, then per_seed aggregate

**Files:**
- Modify: `scripts/analyze_batch.py`
- Modify: `tests/test_analyze_batch.py`

**Interfaces:**
- Consumes: `load`, `_toy`, `survival_spatial_metrics` (Tasks 2–3).
- Produces:
  - `diversity_metrics(df, eps=0.01, lag=5) -> dict`: keys `distinct_strains` ({tick:int}), `n2` ({tick:float} Hill q=2 = 1/Σpₛ²), `new_strain_first_seen` ({strain:first_tick}), `d_max` ({tick:float} max single-strain global freq), `established_flux` ({tick:float} = ½Σ|Δpₛ| over strains, lag `lag`), `leader_changes` (int, # of argmax-strain switches over ticks).
  - `proxy_and_seeding_metrics(df) -> dict`: keys `seed_tick` (int = min tick), `seed_distinct_strains` (int, expect 1), `seed_distinct_factions` (int, expect 4), `net_decrease_proxy` ({tick:int} sum of negative per-(cell,faction,strain) count deltas vs previous tick — **PROXY, not kills**), `strain_faction_xtab` ({strain:{faction:int}} for strains present under ≥2 factions, last tick).
  - `per_seed_metrics(df, n_cells=128*128) -> dict`: merge of the three metric dicts plus `seed` (int|None, parsed by caller — here always None; the CLI fills it from filename).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analyze_batch.py`:
```python
def test_diversity_known_answers(tmp_path):
    # tick1: 2 strains equal freq (10 each) -> N2 = 1/(0.5^2+0.5^2) = 2.0
    # tick2: A=30 B=10 -> freqs .75/.25 -> N2 = 1/(.5625+.0625)=1.6
    # new strain C first seen tick2. leader: A both ticks -> 0 changes.
    rows = [
        (1, 0, 0, "A", 0, 10), (1, 1, 0, "B", 0, 10),
        (2, 0, 0, "A", 0, 30), (2, 1, 0, "B", 0, 10), (2, 2, 0, "C", 0, 0),
    ]
    p = tmp_path / "d.parquet"; _toy(p, rows)
    m = ab.diversity_metrics(ab.load(str(p)))
    assert m["distinct_strains"][1] == 2
    assert abs(m["n2"][1] - 2.0) < 1e-9
    assert abs(m["n2"][2] - 1.6) < 1e-9
    assert m["new_strain_first_seen"]["A"] == 1
    assert m["new_strain_first_seen"]["C"] == 2   # count 0 still records first-seen row
    assert abs(m["d_max"][2] - 0.75) < 1e-9
    assert m["leader_changes"] == 0

def test_proxy_and_seeding_known_answers(tmp_path):
    # seed tick = 1: 1 strain, 4 factions. tick2 cell(0,0) f0 drops 10->4 = -6 proxy.
    rows = [
        (1, 0, 0, "S0", 0, 10), (1, 1, 0, "S0", 1, 10),
        (1, 0, 1, "S0", 2, 10), (1, 1, 1, "S0", 3, 10),
        (2, 0, 0, "S0", 0, 4), (2, 1, 0, "S0", 1, 10),
        (2, 0, 1, "S0", 2, 10), (2, 1, 1, "S0", 3, 10),
    ]
    p = tmp_path / "s2.parquet"; _toy(p, rows)
    m = ab.proxy_and_seeding_metrics(ab.load(str(p)))
    assert m["seed_tick"] == 1
    assert m["seed_distinct_strains"] == 1
    assert m["seed_distinct_factions"] == 4
    assert m["net_decrease_proxy"][2] == 6     # only the -6 drop, summed as magnitude

def test_per_seed_merges_all(tmp_path):
    rows = [(1, 0, 0, "S0", 0, 10), (1, 1, 1, "S0", 1, 10)]
    p = tmp_path / "m.parquet"; _toy(p, rows)
    m = ab.per_seed_metrics(ab.load(str(p)), n_cells=4)
    # has keys from all three metric groups
    assert "fixation_tick" in m and "n2" in m and "seed_distinct_factions" in m
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src D:/anaconda3/envs/basic/python.exe -m pytest tests/test_analyze_batch.py::test_diversity_known_answers -v`
Expected: FAIL — `AttributeError: ... has no attribute 'diversity_metrics'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/analyze_batch.py`:
```python
def diversity_metrics(df: pd.DataFrame, eps: float = 0.01, lag: int = 5) -> dict:
    ticks = sorted(int(t) for t in df["tick"].unique())
    by_strain = df.groupby(["tick", "strain"])["count"].sum()
    distinct_strains, n2, d_max = {}, {}, {}
    freqs = {}  # tick -> {strain: freq}
    for t in ticks:
        s = by_strain.loc[t]
        s = s[s > 0]
        tot = float(s.sum()) or 1.0
        f = (s / tot)
        freqs[t] = {str(k): float(v) for k, v in f.items()}
        distinct_strains[t] = int((s > 0).sum())
        n2[t] = float(1.0 / (f ** 2).sum()) if len(f) else 0.0
        d_max[t] = float(f.max()) if len(f) else 0.0

    # first-seen tick per strain (any row, including count 0 — records emergence)
    first = df.groupby("strain")["tick"].min()
    new_strain_first_seen = {str(k): int(v) for k, v in first.items()}

    # established-set flux = 1/2 sum |p_s(t) - p_s(t-lag)| over union of strains
    established_flux = {}
    for i, t in enumerate(ticks):
        if i - lag < 0:
            continue
        t0 = ticks[i - lag]
        keys = set(freqs[t]) | set(freqs[t0])
        flux = 0.5 * sum(abs(freqs[t].get(k, 0.0) - freqs[t0].get(k, 0.0)) for k in keys)
        established_flux[t] = float(flux)

    # leader changes = # of times the argmax strain switches across ticks
    leaders = []
    for t in ticks:
        if freqs[t]:
            leaders.append(max(freqs[t], key=freqs[t].get))
    leader_changes = sum(1 for a, b in zip(leaders, leaders[1:]) if a != b)

    return {"distinct_strains": distinct_strains, "n2": n2,
            "new_strain_first_seen": new_strain_first_seen, "d_max": d_max,
            "established_flux": established_flux, "leader_changes": leader_changes}


def proxy_and_seeding_metrics(df: pd.DataFrame) -> dict:
    ticks = sorted(int(t) for t in df["tick"].unique())
    seed_tick = ticks[0] if ticks else None
    seed_df = df[df["tick"] == seed_tick] if seed_tick is not None else df.iloc[:0]
    seed_live = seed_df[seed_df["count"] > 0]
    seed_distinct_strains = int(seed_live["strain"].nunique())
    seed_distinct_factions = int(seed_live["faction"].nunique())

    # net-decrease PROXY: per (cell,faction,strain) count drop vs previous tick.
    # NOT kills -- conflates K-wall evaporation + p_leave + arbitration (spec §0.2).
    key = ["cell_x", "cell_y", "faction", "strain"]
    piv = df.pivot_table(index=key, columns="tick", values="count",
                         aggfunc="sum", fill_value=0)
    net_decrease_proxy = {}
    cols = list(piv.columns)
    for i, t in enumerate(cols):
        if i == 0:
            continue
        delta = piv[t] - piv[cols[i - 1]]
        net_decrease_proxy[int(t)] = int(-delta[delta < 0].sum())

    # strain x faction cross-tab at last tick, for strains under >=2 factions
    last = ticks[-1] if ticks else None
    xtab = {}
    if last is not None:
        ld = df[(df["tick"] == last) & (df["count"] > 0)]
        g = ld.groupby(["strain", "faction"])["count"].sum()
        for strain in ld["strain"].unique():
            row = g.loc[strain] if strain in g.index.get_level_values(0) else None
            if row is not None and row.index.nunique() >= 2:
                xtab[str(strain)] = {int(f): int(v) for f, v in row.items()}

    return {"seed_tick": seed_tick, "seed_distinct_strains": seed_distinct_strains,
            "seed_distinct_factions": seed_distinct_factions,
            "net_decrease_proxy": net_decrease_proxy, "strain_faction_xtab": xtab}


def per_seed_metrics(df: pd.DataFrame, n_cells: int = 128 * 128) -> dict:
    m = {"seed": None}
    m.update(survival_spatial_metrics(df, n_cells=n_cells))
    m.update(diversity_metrics(df))
    m.update(proxy_and_seeding_metrics(df))
    return m
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src D:/anaconda3/envs/basic/python.exe -m pytest tests/test_analyze_batch.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_batch.py tests/test_analyze_batch.py
git commit -m "feat: per-seed diversity/oligopoly/proxy/seeding metrics + aggregate"
```

---

## Task 5: cross-seed aggregate metrics (1/4 CI, D4 symmetry, GATE0, timeline)

**Files:**
- Modify: `scripts/analyze_batch.py`
- Modify: `tests/test_analyze_batch.py`

**Interfaces:**
- Consumes: a list of per-seed metric dicts (Task 4 output).
- Produces: `cross_seed_metrics(per_seed_list, steady_min_ticks=200) -> dict` with keys:
  - `n_seeds` (int), `winners` (list[int|None] per seed),
  - `win_counts` ({faction:int}),
  - `win_ci_note` (dict: `expected_share`=0.25, `n`, `per_faction_count`, `binom_2sigma_lo`, `binom_2sigma_hi` — the symmetric-expectation band 0.25·n ± 2·√(n·0.25·0.75); the script reports counts + band, **does not judge**),
  - `d4_symmetry_spread` ({tick:int} max−min of per-faction occupied-cell counts, averaged across seeds at shared ticks),
  - `gate0_short_run` (dict: `steady_window_ticks`, `required`, `note` string flag),
  - `timeline_reconciliation` (dict: per-seed `first_cross_faction_tick` + `fill_tick`, alongside `spec_meet=160`, `spec_fill=320`, `design_meet=32`, `design_fill=60` for the user to compare).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analyze_batch.py`:
```python
def test_cross_seed_winrate_and_flags(tmp_path):
    # 3 fake per-seed dicts with winners 0,1,0 and short steady windows
    per_seed = []
    for w, last in [(0, 50), (1, 50), (0, 50)]:
        per_seed.append({
            "seed": None, "winner_faction": w,
            "ticks": list(range(1, last + 1)),
            "fixation_tick": 40, "first_cross_faction_tick": 20, "fill_tick": 35,
            "faction_occupied": {1: {0: 1, 1: 1, 2: 1, 3: 1}},
        })
    c = ab.cross_seed_metrics(per_seed)
    assert c["n_seeds"] == 3
    assert c["winners"] == [0, 1, 0]
    assert c["win_counts"][0] == 2 and c["win_counts"][1] == 1
    assert abs(c["win_ci_note"]["expected_share"] - 0.25) < 1e-9
    # steady window = last_tick - fill_tick = 50 - 35 = 15 < 200 -> short-run flagged
    assert c["gate0_short_run"]["steady_window_ticks"] <= 15
    assert "200" in str(c["gate0_short_run"]["required"]) or c["gate0_short_run"]["required"] == 200
    assert c["timeline_reconciliation"]["spec_meet"] == 160
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src D:/anaconda3/envs/basic/python.exe -m pytest tests/test_analyze_batch.py::test_cross_seed_winrate_and_flags -v`
Expected: FAIL — `AttributeError: ... has no attribute 'cross_seed_metrics'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/analyze_batch.py`:
```python
import math


def cross_seed_metrics(per_seed_list: list, steady_min_ticks: int = 200) -> dict:
    n = len(per_seed_list)
    winners = [m.get("winner_faction") for m in per_seed_list]
    win_counts = {}
    for w in winners:
        if w is not None:
            win_counts[int(w)] = win_counts.get(int(w), 0) + 1

    p = 0.25
    sigma = math.sqrt(n * p * (1 - p)) if n else 0.0
    win_ci_note = {
        "expected_share": p, "n": n,
        "per_faction_count": win_counts,
        "binom_2sigma_lo": p * n - 2 * sigma,
        "binom_2sigma_hi": p * n + 2 * sigma,
    }

    # D4 symmetry: per-tick max-min of per-faction occupied counts, averaged across seeds
    spread_acc, spread_cnt = {}, {}
    for m in per_seed_list:
        for t, fac_occ in m.get("faction_occupied", {}).items():
            if fac_occ:
                vals = list(fac_occ.values())
                spread_acc[t] = spread_acc.get(t, 0) + (max(vals) - min(vals))
                spread_cnt[t] = spread_cnt.get(t, 0) + 1
    d4_symmetry_spread = {int(t): spread_acc[t] / spread_cnt[t] for t in spread_acc}

    # GATE0: steady window = last tick - fill tick (post-fill band). Use the min across
    # seeds (worst case). If a seed never filled, treat window as 0.
    windows = []
    for m in per_seed_list:
        ticks = m.get("ticks") or []
        last = ticks[-1] if ticks else 0
        fill = m.get("fill_tick")
        windows.append((last - fill) if fill is not None else 0)
    steady_window = min(windows) if windows else 0
    gate0_short_run = {
        "steady_window_ticks": steady_window, "required": steady_min_ticks,
        "note": ("steady window < required: red-queen freq-dependence (β<0) NOT computable "
                 "this batch (spec §1 / GATE0 NA-SHORT-RUN)")
                if steady_window < steady_min_ticks else "steady window adequate",
    }

    timeline_reconciliation = {
        "per_seed": [
            {"first_cross_faction_tick": m.get("first_cross_faction_tick"),
             "fill_tick": m.get("fill_tick")}
            for m in per_seed_list
        ],
        "spec_meet": 160, "spec_fill": 320, "design_meet": 32, "design_fill": 60,
    }

    return {"n_seeds": n, "winners": winners, "win_counts": win_counts,
            "win_ci_note": win_ci_note, "d4_symmetry_spread": d4_symmetry_spread,
            "gate0_short_run": gate0_short_run,
            "timeline_reconciliation": timeline_reconciliation}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src D:/anaconda3/envs/basic/python.exe -m pytest tests/test_analyze_batch.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_batch.py tests/test_analyze_batch.py
git commit -m "feat: cross-seed metrics (1/4 CI, D4 symmetry, GATE0, timeline)"
```

---

## Task 6: render_text + dump_json + CLI main

**Files:**
- Modify: `scripts/analyze_batch.py`
- Modify: `tests/test_analyze_batch.py`

**Interfaces:**
- Consumes: `load`, `per_seed_metrics`, `cross_seed_metrics`.
- Produces:
  - `render_text(per_seed_list, cross) -> str` — structured text report, no PASS/FAIL tokens. Must contain a literal `PROXY` label near the net-decrease line and the GATE0 note verbatim.
  - `dump_json(report, path) -> None` — writes `{"per_seed": [...], "cross_seed": {...}}` as JSON.
  - `main()` — CLI: `--runs-dir data/runs`, `--glob "*-seed*.parquet"`, `--n-cells 16384`. Globs parquets, parses `seedN` from each filename into `seed`, computes per-seed + cross-seed, prints `render_text`, writes `data/runs/analysis-{timestamp}.json`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analyze_batch.py`:
```python
import json

def test_render_text_no_verdicts_and_has_proxy_label(tmp_path):
    rows = [(1, 0, 0, "S0", 0, 10), (1, 1, 1, "S0", 1, 10)]
    p = tmp_path / "r.parquet"; df = _toy(p, rows)
    ps = ab.per_seed_metrics(ab.load(str(p)), n_cells=4)
    cross = ab.cross_seed_metrics([ps])
    txt = ab.render_text([ps], cross)
    assert isinstance(txt, str) and len(txt) > 0
    # report-only discipline: no PASS/FAIL tokens anywhere (spec §0.1)
    up = txt.upper()
    assert "PASS" not in up and "FAIL" not in up
    # proxy must be labeled (spec §0.2)
    assert "PROXY" in up

def test_dump_json_roundtrips(tmp_path):
    rows = [(1, 0, 0, "S0", 0, 10)]
    p = tmp_path / "j.parquet"; _toy(p, rows)
    ps = ab.per_seed_metrics(ab.load(str(p)), n_cells=4)
    cross = ab.cross_seed_metrics([ps])
    out = tmp_path / "a.json"
    ab.dump_json({"per_seed": [ps], "cross_seed": cross}, str(out))
    loaded = json.loads(out.read_text())
    assert "per_seed" in loaded and "cross_seed" in loaded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src D:/anaconda3/envs/basic/python.exe -m pytest tests/test_analyze_batch.py::test_render_text_no_verdicts_and_has_proxy_label -v`
Expected: FAIL — `AttributeError: ... has no attribute 'render_text'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/analyze_batch.py`:
```python
import argparse, glob, json, os, re, datetime


def _last(d):
    """last value of a {tick: val} dict by tick order, or None."""
    if not d:
        return None
    return d[max(d)]


def render_text(per_seed_list: list, cross: dict) -> str:
    L = []
    L.append("=" * 70)
    L.append("FIRST-BATCH ANALYSIS REPORT (numbers only — human makes the verdict)")
    L.append("=" * 70)
    for m in per_seed_list:
        L.append(f"\n--- seed {m.get('seed')} ---")
        ticks = m.get("ticks") or []
        span = f"{ticks[0]}..{ticks[-1]}" if ticks else "(empty)"
        L.append(f"  ticks recorded: {span}  ({len(ticks)} ticks)")
        L.append(f"  total_count last: {_last(m['total_count'])}   "
                 f"extinction_tick: {m['extinction_tick']}")
        L.append(f"  distinct_factions last: {_last(m['distinct_factions'])}   "
                 f"fixation_tick: {m['fixation_tick']}")
        L.append(f"  occupied_cells last: {_last(m['occupied_cells'])}   "
                 f"first_cross_faction_tick: {m['first_cross_faction_tick']}   "
                 f"fill_tick: {m['fill_tick']}")
        L.append(f"  winner_faction: {m['winner_faction']}   "
                 f"faction_share last: {m['faction_share'].get(max(m['faction_share'])) if m['faction_share'] else {}}")
        L.append(f"  distinct_strains last: {_last(m['distinct_strains'])}   "
                 f"N2 last: {_last(m['n2']):.2f}   d_max last: {_last(m['d_max']):.3f}")
        L.append(f"  total strains ever seen: {len(m['new_strain_first_seen'])}   "
                 f"leader_changes: {m['leader_changes']}")
        L.append(f"  established_flux last: {_last(m['established_flux'])}")
        L.append(f"  net_decrease_proxy last: {_last(m['net_decrease_proxy'])}  "
                 f"[PROXY — NOT kills; conflates K-wall+p_leave+arbitration]")
        L.append(f"  seeding: tick {m['seed_tick']} -> "
                 f"{m['seed_distinct_strains']} strain(s), {m['seed_distinct_factions']} faction(s) "
                 f"(expect 1 strain / 4 factions)")
        L.append(f"  strain×faction xtab (strains under ≥2 factions, last tick): "
                 f"{m['strain_faction_xtab']}")
    L.append("\n" + "=" * 70)
    L.append("CROSS-SEED AGGREGATE")
    L.append("=" * 70)
    L.append(f"  n_seeds: {cross['n_seeds']}   winners: {cross['winners']}")
    ci = cross["win_ci_note"]
    L.append(f"  win counts by faction: {ci['per_faction_count']}")
    L.append(f"  symmetric expectation: {ci['expected_share']} share; "
             f"2σ band on win-count = [{ci['binom_2sigma_lo']:.2f}, {ci['binom_2sigma_hi']:.2f}] "
             f"(deviation = possible sneak-goods leak; user judges)")
    L.append(f"  GATE0: steady_window={cross['gate0_short_run']['steady_window_ticks']} "
             f"required={cross['gate0_short_run']['required']} — "
             f"{cross['gate0_short_run']['note']}")
    tr = cross["timeline_reconciliation"]
    L.append(f"  timeline: per-seed {tr['per_seed']}")
    L.append(f"            spec expects meet~{tr['spec_meet']}/fill~{tr['spec_fill']}; "
             f"design(period-unfactored) meet~{tr['design_meet']}/fill~{tr['design_fill']} "
             f"— compare to observed above")
    return "\n".join(L)


def dump_json(report: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)


def main() -> None:
    ap = argparse.ArgumentParser(description="Report-only first-batch analysis (no verdicts).")
    ap.add_argument("--runs-dir", default=os.path.join("data", "runs"))
    ap.add_argument("--glob", default="*-seed*.parquet")
    ap.add_argument("--n-cells", type=int, default=128 * 128)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.runs_dir, args.glob)))
    if not paths:
        print(f"no parquets matched {args.glob} in {args.runs_dir}")
        return
    per_seed = []
    for p in paths:
        m = re.search(r"seed(\d+)", os.path.basename(p))
        ms = per_seed_metrics(load(p), n_cells=args.n_cells)
        ms["seed"] = int(m.group(1)) if m else None
        per_seed.append(ms)
    cross = cross_seed_metrics(per_seed)
    print(render_text(per_seed, cross))
    tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.runs_dir, f"analysis-{tag}.json")
    dump_json({"per_seed": per_seed, "cross_seed": cross}, out)
    print(f"\n-> wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full test suite**

Run: `PYTHONPATH=src D:/anaconda3/envs/basic/python.exe -m pytest tests/test_analyze_batch.py -v`
Expected: PASS (all tests).

Then confirm no regression in the engine suite:
Run: `PYTHONPATH=src D:/anaconda3/envs/basic/python.exe -m pytest tests/ -q`
Expected: all pass (85 engine tests + the new analyze tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_batch.py tests/test_analyze_batch.py
git commit -m "feat: analyze_batch render_text + dump_json + CLI"
```

---

## Task 7: Run the batch + analyze (the deliverable; human verdict)

**Files:** none (execution + report).

**Interfaces:** consumes Tasks 1–6.

> **Note:** This task BURNS COMPUTE (a real T=450 × 4-seed batch). Per spec §4, probe first, then run, then hand the report to the user — the user makes the dynamics verdict; do NOT self-judge "data looks right".

- [ ] **Step 1: Probe timing/VRAM first**

Run:
```bash
PYTHONPATH=src D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --probe 60
```
Expected: prints ms/tick + peak GB + an `est full batch (T=450 x 4 seeds)` seconds estimate. Sanity: peak GB well under the GPU's VRAM; estimate is the wall-clock budget. If peak GB is alarming, STOP and report to the user before the full run.

- [ ] **Step 2: Run the full batch**

Run:
```bash
PYTHONPATH=src D:/anaconda3/envs/basic/python.exe scripts/run_batch.py
```
Expected: 4 lines `seed N: ran 450 ticks ...` (note: **450**, not an early-stop number) + 4 parquet paths under `data/runs/`.

- [ ] **Step 3: Analyze**

Run:
```bash
PYTHONPATH=src D:/anaconda3/envs/basic/python.exe scripts/analyze_batch.py
```
Expected: the structured text report on stdout + `-> wrote data/runs/analysis-{timestamp}.json`.

- [ ] **Step 4: Hand off to the user**

Present the report. State plainly: this is numbers only; the dynamics verdict (does it match the intended red-queen expansion?) is the user's call (spec §4). Note the GATE0 short-window caveat: β<0 red-queen-rotation is NOT claimable from this batch. Do not over-interpret (prior-batch lesson).

---

## Self-Review

**Spec coverage (spec §→ task):**
- §0 discipline (no verdict / proxy label / no guessing / human judges) → Global Constraints + Task 6 (PROXY label + no-PASS/FAIL test) + Task 7 Step 4. ✓
- §1 scope: run_batch stop_on → Task 1; analyze_batch → Tasks 2–6; tests → every task; YAGNI exclusions → Global Constraints + Task 5 GATE0 note. ✓
- §2.1 per-seed metrics (survival/fixation/spatial/occ/diversity/N2/d_max/flux/share/proxy/seeding/xtab) → Tasks 3+4. ✓
- §2.2 cross-seed (1/4 CI / D4 symmetry / GATE0 flag / timeline reconciliation) → Task 5. ✓
- §3.1 run_batch edit → Task 1; §3.2 module structure → Tasks 2–6; §3.3 outputs (parquet + json + stdout) → Task 6 main; §3.4 synthetic-parquet test → Tasks 2–6 tests. ✓
- §4 flow (probe → run → analyze → human verdict) → Task 7. ✓
- §5 acceptance (4×450 parquet / report no PASS-FAIL / tests green / user verdict) → Tasks 1,6,7. ✓

**Placeholder scan:** no TBD/TODO; every code step shows full code; every run step shows command + expected output. ✓

**Type consistency:** `per_seed_metrics` returns a dict merged from `survival_spatial_metrics` + `diversity_metrics` + `proxy_and_seeding_metrics`; `render_text`/`main` read only keys those three define (`total_count`, `distinct_factions`, `occupied_cells`, `faction_share`, `n2`, `d_max`, `new_strain_first_seen`, `leader_changes`, `established_flux`, `net_decrease_proxy`, `seed_*`, `strain_faction_xtab`). `cross_seed_metrics` reads `winner_faction`, `ticks`, `fill_tick`, `faction_occupied`, `first_cross_faction_tick` — all present in per-seed output. ✓

