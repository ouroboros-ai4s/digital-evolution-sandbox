# S0 — Unify CLI Match Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the shared web/engine config core into `src/des/run.py` and add a headless `scripts/run_match.py` CLI front door so AI players can configure 4-faction matches and harvest data without the WebSocket layer.

**Architecture:** Three pieces. (1) Pure relocation: move `PALETTE`/`_DEFAULTS`/`layout_from_slots`/`build_engine_from_config` plus a behavior-preserving `pick_device` from `webapp/server.py` into `src/des/run.py`; `webapp/server.py` re-imports them. (2) New `scripts/run_match.py` consumes a JSON whose schema is identical to the WS `config` payload, validates a top-level key allow-list (red-line guard against outcome constants), runs the engine headless, writes one parquet to `data/runs/`. (3) Match result reuses `webapp/readouts.py:compute_readouts` on the final tick's in-memory records (NOT a parquet read-back) and prints JSON to stdout.

**Tech Stack:** Python 3 (`D:\anaconda3\envs\basic\python.exe`), PyTorch 2.10+cu128, pyarrow, aiohttp (web only), pytest. Windows host with `PYTHONPATH=src` discipline.

## Global Constraints

- **No hand-written "who is strong" / no private goods**: same template, same fixed G→P, no `f_baseline[species]`. Only the player picks legal slot configs.
- **Outcome constants stay locked in the registry**: `μ / z_max / δ / p_max / α / κ / β` are NOT CLI args. Only `{players, grid, K, fill, T, seed}` are legitimate CLI/JSON inputs.
- **CLI top-level key allow-list**: `{players, grid, K, fill, T, seed}` only. Any other top-level key (especially `z_max / mu / delta / p_max / alpha / kappa / beta`) → exit 1.
- **Playground isolation**: web live runs write `data/playground/` (unchanged). CLI match runs write `data/runs/`.
- **Single-source readouts**: match results MUST call `webapp.readouts.compute_readouts` verbatim. No second definition.
- **No win-judging in the sandbox**: emit per-faction `total / share / occupied_cells / distinct_strains / n2 / d_max`. No "winner" field. The win call belongs to the AI player.
- **Behavior-preserving lift**: 285 engine tests + 146 web tests stay green after the move. The lift is byte-equivalent semantics, not a rewrite.
- **Web async loop stays in `webapp/server.py`**: irreducibly different from sync batch path; do NOT introduce a shared loop.
- **`scripts/run_batch.py` stays as the canonical symmetric-default producer**: only its inline device line changes (→ `pick_device`). No other behavior change.
- **Out of scope**: 6→68 primitive expansion (S1–S8), engine/kernel/registry changes, multi-config sweeps, win-judging.

---

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `src/des/run.py` | **Create** | Shared engine-layer config core: `PALETTE`, `_DEFAULTS`, `layout_from_slots`, `build_engine_from_config`, `pick_device`, `compute_match_result`. Engine-layer ⇒ no `webapp` import (would invert dep direction). |
| `webapp/server.py` | **Modify** | Re-import `PALETTE / _DEFAULTS / layout_from_slots / build_engine_from_config` from `des.run`. Replace local `_device(device)` with `pick_device(device)`. Keep async WS loop, routes, frame encoding — those are web-only. |
| `scripts/run_batch.py` | **Modify** | Replace inline device line with `pick_device(force_cpu=args.cpu)`. No other behavior change. |
| `scripts/run_match.py` | **Create** | New CLI front door. Reads JSON, validates top-level allow-list `{players, grid, K, fill, T, seed}`, calls `pick_device`/`build_engine_from_config`/`Engine.run`, emits one parquet to `data/runs/`, prints match-result JSON to stdout. |
| `tests/test_server_config.py` | **Modify** | Update only the 4 import lines (`from webapp.server import …` → `from des.run import …` for the lifted symbols). No assertion changes. |
| `tests/test_run_config.py` | **Create** | Pure unit tests for the lifted core on the engine-layer path (`des.run`). Mirrors the relevant subset of `test_server_config.py`. |
| `tests/test_run_match.py` | **Create** | New: subprocess + in-process E2E tests for `scripts/run_match.py`. |

**Naming contract (locked, used by every task):**

```python
# src/des/run.py
PALETTE: list[str]                                                # ordered 6-letter list
_DEFAULTS: dict                                                   # grid/K/fill/T/seed/z_max
def layout_from_slots(slots: dict) -> tuple[str, ...]
def build_engine_from_config(cfg: dict, device) -> tuple[Engine, dict]
def pick_device(device=None, force_cpu: bool = False) -> torch.device
def compute_match_result(eng, parquet_path: str) -> dict
```

`compute_match_result` consumes `eng.world` + `eng.table` + `eng.T` and returns
`{"path": str, "ticks": int, "final": {<compute_readouts output>}}`.
The `final` block is `compute_readouts(...)`'s return value verbatim (single source).

---

### Task 1: Lift the shared config core into `src/des/run.py`

**Goal:** Pure relocation of `PALETTE`, `_DEFAULTS`, `layout_from_slots`, `build_engine_from_config` from `webapp/server.py` into a new engine-layer module `src/des/run.py`. Web continues to behave identically by re-importing.

**Files:**
- Create: `src/des/run.py`
- Modify: `webapp/server.py` (remove the lifted definitions; add imports from `des.run`)
- Test: existing `tests/test_server_config.py` (modify import lines only); no new test in this task — the regression lock IS the test.

**Interfaces:**
- Consumes: `des.engine.Engine`, `des.registry._SLOTS / _LOCKED / ALPHABET`.
- Produces (importable from `des.run`):
  - `PALETTE: list[str]`
  - `_DEFAULTS: dict`
  - `layout_from_slots(slots: dict) -> tuple[str, ...]`
  - `build_engine_from_config(cfg: dict, device) -> tuple[Engine, dict]`

- [ ] **Step 1: Create `src/des/run.py` with the lifted symbols (verbatim copy)**

```python
# src/des/run.py
"""Shared engine-layer config core. Imported by webapp/server.py and
scripts/run_match.py. No webapp imports here — engine layer must stay
free of aiohttp deps so headless callers don't pay the cost."""
from __future__ import annotations
from des.engine import Engine
from des.registry import _SLOTS, _LOCKED, ALPHABET

# 6 v1 primitives, ordered (front-end dropdown reads the same list via /config)
PALETTE = ["N0", "F4Nr1", "F4Nr4", "P_base", "P_hotspot", "BroadSweep"]

_DEFAULTS = {"grid": 128, "K": 64, "fill": 20, "T": 450, "seed": 0, "z_max": 8.0}


def layout_from_slots(slots: dict) -> tuple[str, ...]:
    """Assemble a 16-position BB0 layout from mutable-slot choices.
    locked positions := _LOCKED; slot positions := given (default 'N0');
    backbone positions := 'N0'. Rejects non-slot indices and off-palette letters."""
    out = []
    for i in range(16):
        if i in _LOCKED:
            out.append(_LOCKED[i])
        elif i in _SLOTS:
            letter = slots.get(i, "N0")
            if letter not in ALPHABET:
                raise ValueError(f"slot {i} = {letter!r} not in palette {PALETTE}")
            out.append(letter)
        else:
            out.append("N0")
    for k in slots:
        if k not in _SLOTS:
            raise ValueError(f"position {k} is not a mutable slot {sorted(_SLOTS)}")
    return tuple(out)


def build_engine_from_config(cfg: dict, device) -> tuple[Engine, dict]:
    """cfg keys: players (list of exactly 4 {slots:{int:str}}), grid, K, fill,
    T, seed, z_max (globals shared across all four factions). Returns (engine,
    resolved) where resolved has defaults filled, a 'players' echo, and a 4-tuple
    'layouts'. Each layout -> layout_from_slots -> validate_bb0_layout enforces the
    template structure (red-line 4: same template, slot choices may differ); only
    primitive letters + run knobs ever enter here (red-line 3)."""
    players = cfg.get("players")
    if not isinstance(players, list) or len(players) != 4:
        raise ValueError(f"config must have exactly 4 players, got {players!r}")
    layouts = tuple(
        layout_from_slots({int(k): v for k, v in (p.get("slots") or {}).items()})
        for p in players
    )
    resolved = dict(_DEFAULTS)
    for k in _DEFAULTS:
        if cfg.get(k) is not None:
            resolved[k] = cfg[k]
    resolved["players"] = players
    resolved["layouts"] = layouts
    g = int(resolved["grid"])
    eng = Engine(H=g, W=g, K=int(resolved["K"]), seed=int(resolved["seed"]),
                 device=device, z_max=float(resolved["z_max"]),
                 fill_per_cell=int(resolved["fill"]), layouts=layouts)
    return eng, resolved
```

- [ ] **Step 2: Add `pick_device` to `src/des/run.py` (behavior-preserving fold)**

Append to `src/des/run.py`:

```python
import torch


def pick_device(device=None, force_cpu: bool = False) -> torch.device:
    """Behavior-preserving fold of webapp/server.py:_device and the inline
    device line in scripts/run_batch.py. The server passes its launch device
    arg straight through (`device=...`), preserving today's `cuda:N` selection;
    `force_cpu=True` forces CPU (matches run_batch.py --cpu); `device=None`
    auto-selects cuda-if-available."""
    if force_cpu:
        return torch.device("cpu")
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- [ ] **Step 3: Strip the lifted definitions out of `webapp/server.py` and re-import**

Edit `webapp/server.py`. Remove lines 11 (the `from des.registry import ...` line that imports `_SLOTS`, `_LOCKED`, `ALPHABET` — only used by `layout_from_slots`, now lifted), the `PALETTE` constant (line 21), the `_DEFAULTS` constant (line 23), the `layout_from_slots` def (lines 26–44), the `build_engine_from_config` def (lines 47–71), and the `_device` def (lines 74–77).

Replace with this import block right after the existing `from des.recorder import Recorder` line:

```python
from des.run import (
    PALETTE, _DEFAULTS, layout_from_slots, build_engine_from_config, pick_device,
)
```

In `make_app`, change:

```python
app[_DEVICE_KEY] = _device(device)
```

to:

```python
app[_DEVICE_KEY] = pick_device(device)
```

Everything else in `webapp/server.py` (async WS loop, route registration, `_jsonable`, `make_app`, `main`) is unchanged.

- [ ] **Step 4: Update import paths in `tests/test_server_config.py`**

The existing test file imports the lifted symbols from `webapp.server`. Change the top of `tests/test_server_config.py`:

```python
import pytest
import torch
from des.run import PALETTE, layout_from_slots, build_engine_from_config
from webapp.server import _jsonable
from des.registry import BB0_TEMPLATE, _SLOTS, _LOCKED
```

(`_jsonable` stays in `webapp/server.py` because it serializes the WS-payload shape — engine-layer code has no reason to touch JSON-payload formatting.) No assertion changes.

- [ ] **Step 5: Run the regression lock (the move's only acceptance gate)**

Run from repo root:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: **all tests pass** (285 engine + 146 web = 431; exact count may vary by ±1 if the suite has grown, but every previously-passing test must still pass). The test that proves the lift worked is `tests/test_server_config.py`: it now imports from `des.run` and still passes byte-equivalently.

Backtrack: if any previously-green test fails, revert and verify the strip in step 3 didn't drop a line or break an import.

- [ ] **Step 6: Commit**

```
git add src/des/run.py webapp/server.py tests/test_server_config.py
git commit -m "feat(s0): lift shared config core into src/des/run.py

Move PALETTE / _DEFAULTS / layout_from_slots / build_engine_from_config
out of webapp/server.py and add a behavior-preserving pick_device.
webapp/server.py now imports from des.run; existing tests prove
behavior-equivalence. Foundation for the CLI match-runner front door."
```

---

### Task 2: Wire `pick_device` into `scripts/run_batch.py`

**Goal:** Replace the inline device line with `pick_device(force_cpu=args.cpu)` so the symmetric-default producer shares the same device selection logic. No other behavior change — `run_batch.py` remains the canonical reproducible anchor.

**Files:**
- Modify: `scripts/run_batch.py:117` (the single inline device-selection line)
- Test: `tests/test_smoke.py` (existing) — the batch runner's existing smoke path still passes; no new test in this task.

**Interfaces:**
- Consumes: `des.run.pick_device(device=None, force_cpu: bool = False) -> torch.device` (Task 1).
- Produces: none new.

- [ ] **Step 1: Edit the device line in `scripts/run_batch.py`**

At the top of the file, after the existing `import torch` line, add:

```python
from des.run import pick_device
```

In `main()`, replace line 117:

```python
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
```

with:

```python
    device = pick_device(force_cpu=args.cpu)
```

Everything else (`H`, `W`, `K`, `FILL`, `T`, `SEEDS`, `Z_MAX`, `run_one`, `phase_probe`, the `--probe` / `--phase-probe` arg paths) is unchanged.

- [ ] **Step 2: Smoke-run `run_batch.py --probe` to prove behavior is preserved**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --probe 3
```

Expected: prints a single line of the form `device=cuda ...` (or `device=cpu ...` on a no-GPU box), then `[probe 3 ticks] X.X ms/tick | peak Y.YY GB | strains->… | total …`, exits 0, **no parquet written under `data/runs/`** (record=False on probe path).

Backtrack: if the import fails, recheck Task 1 step 1 created `src/des/run.py` at the right path and that `PYTHONPATH=src` is set.

- [ ] **Step 3: Run the full pytest suite again**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: all previously-green tests still pass (regression lock holds).

- [ ] **Step 4: Commit**

```
git add scripts/run_batch.py
git commit -m "refactor(s0): use shared pick_device in run_batch.py

Replace the inline device-selection line with des.run.pick_device.
No behavior change; both --cpu and the default cuda-if-available
path are byte-equivalent. Aligns the batch runner with the lifted
shared core so the new run_match.py CLI uses the same primitive."
```

---

### Task 3: Add `compute_match_result` to `src/des/run.py`

**Goal:** Wrap `webapp/readouts.py:compute_readouts` so callers get the documented match-result envelope without re-deriving the readouts payload. Single-source: `compute_readouts` is the only place that defines what `total / share / n2 / d_max` mean.

**Files:**
- Modify: `src/des/run.py` (append the new function)
- Test: `tests/test_run_config.py` (Create)

**Interfaces:**
- Consumes:
  - `webapp.readouts.compute_readouts(cell_x, cell_y, strain, faction, count) -> dict` (existing)
  - `eng.world.count`, `eng.world.strain_id`, `eng.world.faction` (torch tensors), `eng.table.sequence_of(int) -> tuple[str,...]`, `eng.T` (int)
- Produces: `compute_match_result(eng, parquet_path: str) -> dict` returning `{"path": str, "ticks": int, "final": <compute_readouts return>}`. The `final` block contains keys `total / occupied_cells / distinct_strains / n2 / d_max / faction_share`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_config.py`:

```python
import pytest
import torch
from des.run import (
    PALETTE, _DEFAULTS, layout_from_slots, build_engine_from_config,
    pick_device, compute_match_result,
)
from webapp.readouts import compute_readouts

DEV = torch.device("cpu")


def test_palette_lifted_unchanged():
    assert PALETTE == ["N0", "F4Nr1", "F4Nr4", "P_base", "P_hotspot", "BroadSweep"]


def test_pick_device_force_cpu_overrides_cuda_arg():
    assert pick_device(device=torch.device("cuda"), force_cpu=True) == torch.device("cpu")


def test_pick_device_passthrough():
    d = torch.device("cpu")
    assert pick_device(device=d) is d


def test_compute_match_result_shape_and_single_source():
    cfg = {"players": [{"slots": {}} for _ in range(4)],
           "grid": 16, "K": 8, "fill": 4, "T": 3, "seed": 0}
    eng, _ = build_engine_from_config(cfg, DEV)
    eng.run(3, recorder=None, stop_on=())
    res = compute_match_result(eng, "data/runs/fake.parquet")
    assert res["path"] == "data/runs/fake.parquet"
    assert res["ticks"] == 3
    final = res["final"]
    for k in ("total", "occupied_cells", "distinct_strains", "n2", "d_max", "faction_share"):
        assert k in final
    # single-source: faction_share matches a direct compute_readouts call
    cnt = eng.world.count.cpu(); sid = eng.world.strain_id.cpu(); fac = eng.world.faction.cpu()
    nz = torch.nonzero(cnt > 0, as_tuple=False)
    ys = nz[:, 0].tolist(); xs = nz[:, 1].tolist(); ks = nz[:, 2]
    sids = sid[nz[:, 0], nz[:, 1], ks].tolist()
    facs = fac[nz[:, 0], nz[:, 1], ks].tolist()
    cnts = cnt[nz[:, 0], nz[:, 1], ks].tolist()
    strains = [".".join(eng.table.sequence_of(int(s))) for s in sids]
    direct = compute_readouts(xs, ys, strains, facs, cnts)
    assert final["faction_share"] == direct["faction_share"]
    assert final["total"] == direct["total"]
```

- [ ] **Step 2: Run the failing test**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_run_config.py -v
```

Expected: `test_compute_match_result_shape_and_single_source` FAILs with `ImportError: cannot import name 'compute_match_result' from 'des.run'`.

- [ ] **Step 3: Implement `compute_match_result` in `src/des/run.py`**

Append to `src/des/run.py`:

```python
import torch
from webapp.readouts import compute_readouts


def compute_match_result(eng, parquet_path: str) -> dict:
    """Final-tick match result. Reads the engine's in-memory final world
    snapshot (NOT a parquet read-back — the parquet footer isn't written
    until Recorder.close()). Reuses webapp.readouts.compute_readouts as
    the single source for total/share/n2/d_max definitions.
    Returns: {"path": str, "ticks": int, "final": <compute_readouts return>}.
    No 'winner' field — the sandbox stays goal-free; the AI player decides."""
    cnt = eng.world.count.cpu()
    sid = eng.world.strain_id.cpu()
    fac = eng.world.faction.cpu()
    nz = torch.nonzero(cnt > 0, as_tuple=False)
    ys = nz[:, 0].tolist()
    xs = nz[:, 1].tolist()
    ks = nz[:, 2]
    sids = sid[nz[:, 0], nz[:, 1], ks].tolist()
    facs = fac[nz[:, 0], nz[:, 1], ks].tolist()
    cnts = cnt[nz[:, 0], nz[:, 1], ks].tolist()
    strains = [".".join(eng.table.sequence_of(int(s))) for s in sids]
    final = compute_readouts(xs, ys, strains, facs, cnts)
    return {"path": parquet_path, "ticks": int(eng.T), "final": final}
```

Note: this introduces `webapp.readouts` as an import from `des.run`. That is **deliberate and acceptable** despite the "engine layer must not import webapp" rule because `webapp/readouts.py` is itself a stdlib-only pure module (no aiohttp, no torch — verified by reading its file header). It is the single-source readouts definition; importing it is what makes the single-source guarantee enforceable.

- [ ] **Step 4: Run the test to verify it passes**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_run_config.py -v
```

Expected: all 4 tests in `tests/test_run_config.py` PASS, including `test_compute_match_result_shape_and_single_source`.

- [ ] **Step 5: Re-run the full suite to confirm no regression**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: every previously-green test plus the 4 new `tests/test_run_config.py` tests pass.

- [ ] **Step 6: Commit**

```
git add src/des/run.py tests/test_run_config.py
git commit -m "feat(s0): add compute_match_result helper in des.run

Single-source match-result envelope: reuses webapp.readouts.compute_readouts
on the engine's in-memory final world (parquet footer is only written on
Recorder.close, so a read-back mid-finalize would fail). No 'winner' field —
sandbox stays goal-free."
```

---

### Task 4: Create `scripts/run_match.py` skeleton with arg parsing + key allow-list guard

**Goal:** Build the CLI front door scaffold with the **red-line key allow-list guard** wired in first. The guard is the single most important piece — it is what enforces "outcome constants stay locked in the registry" against the JSON front end. We deliver it as the first thing tested so red-line violations fail loudly.

**Files:**
- Create: `scripts/run_match.py` (skeleton with `--config / --cpu / --out`, JSON load + key validation, exit 1 on violations)
- Test: `tests/test_run_match.py` (Create — first batch of subprocess tests)

**Interfaces:**
- Consumes:
  - `des.run.pick_device`, `des.run.build_engine_from_config`, `des.run.compute_match_result` (Tasks 1, 3)
  - `des.recorder.Recorder`
- Produces:
  - CLI: `python scripts/run_match.py --config <path> [--cpu] [--out <dir>]`
  - Module-level constants: `ALLOWED_KEYS = frozenset({"players", "grid", "K", "fill", "T", "seed"})`, `DEFAULT_OUT = "data/runs"`
  - Functions: `validate_config_keys(cfg: dict) -> None` (raises `ValueError` on disallowed keys); `main(argv: list[str] | None = None) -> int` (returns process exit code).

- [ ] **Step 1: Write the failing tests for arg parsing + key allow-list guard**

Create `tests/test_run_match.py`:

```python
"""End-to-end CLI tests for scripts/run_match.py.

These tests drive the script as a subprocess so the actual entry-point
behavior (argparse, JSON load, exit code, stdout payload) is exercised."""
from __future__ import annotations
import json, os, subprocess, sys
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PY = sys.executable
SCRIPT = os.path.join(REPO, "scripts", "run_match.py")


def _run(args, env_extra=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(REPO, "src")
    if env_extra:
        env.update(env_extra)
    return subprocess.run([PY, SCRIPT, *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)


def _write(tmp_path, name, payload):
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return str(p)


def _empty_players():
    return [{"slots": {}} for _ in range(4)]


def test_missing_config_arg_exits_nonzero():
    r = _run([])
    assert r.returncode != 0


def test_config_file_not_found_exits_1(tmp_path):
    r = _run(["--config", str(tmp_path / "missing.json")])
    assert r.returncode == 1
    assert "missing.json" in (r.stderr + r.stdout)


def test_config_malformed_json_exits_1(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ this is not json ")
    r = _run(["--config", str(p)])
    assert r.returncode == 1


def test_disallowed_top_level_key_z_max_rejected(tmp_path):
    cfg = {"players": _empty_players(), "grid": 16, "K": 8, "fill": 4,
           "T": 2, "seed": 0, "z_max": 20}
    r = _run(["--config", _write(tmp_path, "c.json", cfg), "--cpu",
              "--out", str(tmp_path / "runs")])
    assert r.returncode == 1
    assert "z_max" in (r.stderr + r.stdout)


@pytest.mark.parametrize("k", ["mu", "delta", "p_max", "alpha", "kappa", "beta"])
def test_disallowed_outcome_constants_rejected(tmp_path, k):
    cfg = {"players": _empty_players(), "grid": 16, "K": 8, "fill": 4,
           "T": 2, "seed": 0, k: 0.5}
    r = _run(["--config", _write(tmp_path, "c.json", cfg), "--cpu",
              "--out", str(tmp_path / "runs")])
    assert r.returncode == 1
    assert k in (r.stderr + r.stdout)
```

- [ ] **Step 2: Run the new tests to confirm they fail**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_run_match.py -v
```

Expected: every test FAILs because `scripts/run_match.py` does not exist yet (the subprocess will exit with `can't open file` errors).

- [ ] **Step 3: Create `scripts/run_match.py` skeleton (arg parser + key guard + JSON load)**

Create `scripts/run_match.py`:

```python
#!/usr/bin/env python
"""Headless CLI front door for the DES engine. AI players configure
4 factions' starting strains in a JSON file and run a match producing
one parquet under data/runs/.

Usage:
    PYTHONPATH=src python scripts/run_match.py --config match.json [--cpu] [--out data/runs]

The JSON schema is identical to the WebSocket `config` payload:
    {"players": [{"slots": {"0": "F4Nr4", "2": "P_hotspot"}}, {…}, {…}, {…}],
     "grid": 128, "K": 64, "fill": 20, "T": 450, "seed": 0}

Exactly 4 players. Only top-level keys allowed: players, grid, K, fill, T,
seed. Any other key (z_max / mu / delta / p_max / alpha / kappa / beta) is
a red-line violation and exits 1 — outcome constants stay locked in the
registry (spec §2 red-line 2)."""
from __future__ import annotations
import argparse, datetime, json, os, sys

ALLOWED_KEYS = frozenset({"players", "grid", "K", "fill", "T", "seed"})
DEFAULT_OUT = os.path.join("data", "runs")


def validate_config_keys(cfg: dict) -> None:
    """Top-level allow-list. Rejects outcome-constant keys (z_max / mu /
    delta / p_max / alpha / kappa / beta) and any other unknown key.
    This is the CLI's own guard — the web path fixes z_max from _DEFAULTS
    and never exposes it; this closes the same door for the JSON front end."""
    extra = set(cfg.keys()) - ALLOWED_KEYS
    if extra:
        raise ValueError(
            f"disallowed top-level config keys: {sorted(extra)}; "
            f"allowed = {sorted(ALLOWED_KEYS)} "
            f"(outcome constants stay in the registry)")


def _now_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="DES headless match runner")
    ap.add_argument("--config", required=True, help="path to match config JSON")
    ap.add_argument("--cpu", action="store_true", help="force CPU device")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output dir for parquet")
    return ap.parse_args(argv)


def _load_config(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        cfg = _load_config(args.config)
        validate_config_keys(cfg)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"malformed JSON in {args.config}: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    # engine assembly + run + result emission added in Task 5
    print(json.dumps({"event": "config_validated", "path": args.config}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the Task-4 tests to confirm they pass**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_run_match.py -v
```

Expected: `test_missing_config_arg_exits_nonzero`, `test_config_file_not_found_exits_1`, `test_config_malformed_json_exits_1`, `test_disallowed_top_level_key_z_max_rejected`, and all 6 parametrized `test_disallowed_outcome_constants_rejected` cases PASS.

- [ ] **Step 5: Commit**

```
git add scripts/run_match.py tests/test_run_match.py
git commit -m "feat(s0): run_match.py skeleton + key allow-list guard

Add the CLI front door with arg parsing, JSON load, and the red-line
top-level key allow-list (players/grid/K/fill/T/seed only). Outcome
constants (z_max/mu/delta/p_max/alpha/kappa/beta) are rejected with
exit 1. Engine assembly + match-result emission land in Task 5."
```

---

### Task 5: Wire engine assembly + match-result emission into `scripts/run_match.py`

**Goal:** Take the validated config, build the engine via the shared core, run headless to completion, write one parquet under `data/runs/`, and print the match-result JSON to stdout. Errors from `build_engine_from_config` (4-player check, illegal slot, off-palette letter) → exit 1; `Recorder` writer-thread death propagates (data loss must not be swallowed, per spec §5).

**Files:**
- Modify: `scripts/run_match.py` (replace the placeholder line in `main` with the full pipeline)
- Test: `tests/test_run_match.py` (append happy-path + invalid-config + result-shape tests)

**Interfaces:**
- Consumes:
  - `des.run.pick_device(device=None, force_cpu: bool = False) -> torch.device`
  - `des.run.build_engine_from_config(cfg, device) -> (Engine, dict)`
  - `des.run.compute_match_result(eng, parquet_path) -> dict`
  - `des.recorder.Recorder(path, table)` with `.dump(tick, world)` and `.close()`
  - CLI guard `validate_config_keys` (Task 4)
- Produces: stdout JSON `{"path": "data/runs/<ts>-match.parquet", "ticks": <T>, "final": {…}}`; exit 0 on success, 1 on validation/IO error.

- [ ] **Step 1: Append the new tests to `tests/test_run_match.py`**

Append at the end of `tests/test_run_match.py`:

```python
def test_valid_4_player_run_writes_parquet_and_prints_result(tmp_path):
    cfg = {"players": _empty_players(), "grid": 16, "K": 8, "fill": 4,
           "T": 3, "seed": 0}
    out_dir = tmp_path / "runs"
    r = _run(["--config", _write(tmp_path, "ok.json", cfg), "--cpu",
              "--out", str(out_dir)])
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout.strip().splitlines()[-1])
    assert payload["ticks"] == 3
    assert os.path.isfile(payload["path"])
    assert payload["path"].startswith(str(out_dir))
    final = payload["final"]
    for k in ("total", "occupied_cells", "distinct_strains",
              "n2", "d_max", "faction_share"):
        assert k in final
    # faction_share is a dict[str|int, float]; sums to ~1.0 on a non-empty world
    if final["total"] > 0:
        assert abs(sum(final["faction_share"].values()) - 1.0) < 1e-6


def test_three_players_exits_nonzero_no_parquet(tmp_path):
    cfg = {"players": [{"slots": {}} for _ in range(3)],
           "grid": 16, "K": 8, "fill": 4, "T": 2, "seed": 0}
    out_dir = tmp_path / "runs"
    r = _run(["--config", _write(tmp_path, "bad.json", cfg), "--cpu",
              "--out", str(out_dir)])
    assert r.returncode == 1
    assert "4 players" in (r.stderr + r.stdout)
    assert not out_dir.exists() or len(list(out_dir.iterdir())) == 0


def test_off_palette_letter_exits_nonzero_no_parquet(tmp_path):
    cfg = {"players": [{"slots": {0: "NOPE"}}, {"slots": {}}, {"slots": {}}, {"slots": {}}],
           "grid": 16, "K": 8, "fill": 4, "T": 2, "seed": 0}
    out_dir = tmp_path / "runs"
    r = _run(["--config", _write(tmp_path, "bad.json", cfg), "--cpu",
              "--out", str(out_dir)])
    assert r.returncode == 1
    assert "palette" in (r.stderr + r.stdout)
    assert not out_dir.exists() or len(list(out_dir.iterdir())) == 0


def test_non_slot_index_exits_nonzero_no_parquet(tmp_path):
    cfg = {"players": [{"slots": {4: "F4Nr1"}}, {"slots": {}}, {"slots": {}}, {"slots": {}}],
           "grid": 16, "K": 8, "fill": 4, "T": 2, "seed": 0}
    out_dir = tmp_path / "runs"
    r = _run(["--config", _write(tmp_path, "bad.json", cfg), "--cpu",
              "--out", str(out_dir)])
    assert r.returncode == 1
    assert "slot" in (r.stderr + r.stdout)


def test_web_config_interchangeable_with_cli_config(tmp_path):
    """Single-schema invariant: a config the WS path accepts must also be
    accepted by run_match (and vice-versa). Both use build_engine_from_config."""
    import torch
    from des.run import build_engine_from_config
    cfg = {"players": [{"slots": {0: "F4Nr1"}}, {"slots": {0: "P_hotspot"}},
                       {"slots": {0: "P_base"}}, {"slots": {0: "BroadSweep"}}],
           "grid": 16, "K": 8, "fill": 4, "T": 2, "seed": 0}
    # WS path equivalent
    eng, _ = build_engine_from_config(cfg, torch.device("cpu"))
    assert eng.H == 16
    # CLI path
    r = _run(["--config", _write(tmp_path, "ws.json", cfg), "--cpu",
              "--out", str(tmp_path / "runs")])
    assert r.returncode == 0, r.stderr
```

- [ ] **Step 2: Run the new tests to confirm they fail**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_run_match.py -v
```

Expected: the new tests in `tests/test_run_match.py` (`test_valid_4_player_run_writes_parquet_and_prints_result`, `test_three_players_exits_nonzero_no_parquet`, `test_off_palette_letter_exits_nonzero_no_parquet`, `test_non_slot_index_exits_nonzero_no_parquet`, `test_web_config_interchangeable_with_cli_config`) FAIL — the script's main() still prints `{"event": "config_validated", …}` and never builds an engine or writes a parquet. (The earlier Task-4 tests still pass.)

- [ ] **Step 3: Replace the placeholder block in `scripts/run_match.py` with the full pipeline**

In `scripts/run_match.py`, replace the body of `main()` from the line `print(json.dumps({"event": "config_validated", "path": args.config}))` (and the preceding placeholder comment) with the real pipeline. The full new `main()` reads:

```python
def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        cfg = _load_config(args.config)
        validate_config_keys(cfg)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"malformed JSON in {args.config}: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    # imports placed inside main so --help / config-validation errors don't pay
    # the torch import cost (saves ~1s on cold start; matters for AI orchestrators
    # that fan out many config-validate-only invocations).
    from des.run import pick_device, build_engine_from_config, compute_match_result
    from des.recorder import Recorder
    try:
        device = pick_device(force_cpu=args.cpu)
        eng, resolved = build_engine_from_config(cfg, device)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)
    parquet_path = os.path.join(args.out, f"{_now_tag()}-match.parquet")
    rec = Recorder(parquet_path, eng.table)
    try:
        eng.run(int(resolved["T"]), recorder=rec, stop_on=())   # match: run full T
    finally:
        rec.close()   # propagates writer-thread death (data loss must not be swallowed)
    result = compute_match_result(eng, parquet_path)
    print(json.dumps(result))
    return 0
```

Key choices encoded above:
- Import torch-loading modules inside `main` so the validation-only fast path stays cheap.
- `stop_on=()` mirrors `scripts/run_batch.py:run_one`: a match runs the full requested `T` (no early-exit on fixation/extinction).
- `rec.close()` is in a `finally` so a torch error mid-run still closes the parquet writer cleanly. `Recorder._check_thread` (called by `close`) re-raises a writer-thread death.

- [ ] **Step 4: Run the full pytest suite to confirm the pipeline works end-to-end**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_run_match.py tests/test_run_config.py tests/test_server_config.py -v
```

Expected: all tests in those three files pass. The happy-path test produces a real parquet under `tmp_path/runs/` and the CLI prints a single JSON line that parses back to a dict with the documented shape.

- [ ] **Step 5: Run the full repo suite to confirm no regression**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: all 285 engine + 146 web tests + the new tests pass.

- [ ] **Step 6: Manual end-to-end smoke (operator confidence check)**

From the repo root:

```powershell
$env:PYTHONPATH='src'
'{"players":[{"slots":{}},{"slots":{}},{"slots":{}},{"slots":{}}],"grid":16,"K":8,"fill":4,"T":3,"seed":0}' | Out-File -Encoding utf8 _smoke_match.json
D:/anaconda3/envs/basic/python.exe scripts/run_match.py --config _smoke_match.json --cpu --out data/runs
Remove-Item _smoke_match.json
```

Expected stdout: a single JSON line of shape `{"path": "data/runs/<ts>-match.parquet", "ticks": 3, "final": {"total": …, "occupied_cells": …, …, "faction_share": {"0": …, "1": …, "2": …, "3": …}}}`. Exit code 0. A parquet file appears under `data/runs/`.

- [ ] **Step 7: Commit**

```
git add scripts/run_match.py tests/test_run_match.py
git commit -m "feat(s0): wire engine assembly + match-result emission

run_match.py now builds the engine via the shared core, runs headless
to completion, writes one parquet to data/runs/, and prints the match
result envelope to stdout. Errors from build_engine_from_config exit 1
without writing a parquet; Recorder writer-thread death propagates so
data-loss surfaces. Closes S0."
```

---

### Task 6: Final regression sweep + tidy

**Goal:** Prove the whole S0 deliverable is green together: lifted core + run_batch device fold + run_match CLI + new tests + unchanged web + unchanged engine. Clean up any incidental clutter (smoke files, stray data) so the working tree is publishable.

**Files:**
- No source modifications expected. If a regression slips through, fix it in this task and reference the offending commit in the message.
- Test: `tests/` (the entire suite)

**Interfaces:**
- Consumes: every artifact produced by Tasks 1–5.
- Produces: a green `pytest tests/` and a clean `git status`.

- [ ] **Step 1: Full pytest sweep**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -q
```

Expected: every test passes. Total count = 285 engine + 146 web + 4 (`tests/test_run_config.py`) + ~10 (`tests/test_run_match.py`); the exact total may differ, but no previously-green test should now fail.

Backtrack: if a test fails, identify the task that introduced the regression by reverting commits one at a time until the suite is green again, then fix forward in a new commit.

- [ ] **Step 2: Web smoke (manual, optional but recommended)**

Build the front end and start the server to confirm the lift didn't break the live path:

```
cd webapp/frontend; npm run build; cd ../..
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m webapp.server
```

Open `http://localhost:8000`, click `start` with the default 4 empty-slots config, watch a few ticks tick, then `Ctrl+C`. Expected: live frames stream, no exception in the terminal. (If `npm run build` was already done recently the rebuild may be a no-op.)

- [ ] **Step 3: Inspect & clean stray data**

```
git status
```

Expected output: clean working tree. If `data/runs/<ts>-match.parquet` files left over from manual smoke tests appear, decide per-file: keep if you want to inspect, otherwise `rm` them. Do NOT commit smoke parquets — they are not test fixtures.

- [ ] **Step 4: Final commit (only if step 1 needed a fix-forward)**

If step 1 surfaced a regression that you fixed:

```
git add <files-touched>
git commit -m "fix(s0): <description of the regression fixed>"
```

Otherwise this step is a no-op.

- [ ] **Step 5: Push to origin**

```
git push origin <current-branch>
```

Expected: push succeeds. The branch is ready for review / merge to `main`.

---

## Self-Review

**1. Spec coverage:**
- §3.1 (lift core): Tasks 1, 2 — `PALETTE`, `_DEFAULTS`, `layout_from_slots`, `build_engine_from_config`, `pick_device` all in `src/des/run.py`; `webapp/server.py` re-imports; `run_batch.py` uses `pick_device`.
- §3.2 (CLI front door): Tasks 4, 5 — `scripts/run_match.py` accepts the WS-equivalent JSON, validates allow-list, builds engine, runs headless, writes parquet, emits stdout JSON.
- §3.2 key allow-list (red-line #2): Task 4 — `validate_config_keys` rejects `z_max / mu / delta / p_max / alpha / kappa / beta` with exit 1. Tested via parametrized cases.
- §3.3 (match result reuse `compute_readouts` on in-memory final tick): Task 3 — `compute_match_result` reads `eng.world` directly (NOT parquet read-back), wraps `compute_readouts`. Tested in `tests/test_run_config.py::test_compute_match_result_shape_and_single_source`.
- §3.4 (`run_batch.py` stays as-is except device line): Task 2 — single-line replacement with `pick_device(force_cpu=args.cpu)`; smoke run preserved.
- §4 (data flow): Tasks 4 + 5 implement `match.json → run_match.py → pick_device → build_engine_from_config → Engine.run → parquet + stdout JSON`. Web path data-flow unchanged because `webapp/server.py` only swaps imports.
- §5 (error handling): Task 4 catches `FileNotFoundError`, `json.JSONDecodeError`, `ValueError` → exit 1; Task 5 lets `Recorder._check_thread` propagate via `rec.close()` in `finally`.
- §6 (testing): regression lock = full suite green after Task 1; new behavior = `tests/test_run_match.py` (Tasks 4, 5) covering valid run, bad configs, result-shape, web↔CLI interchangeability.
- §7 (out of scope): no engine/kernel/registry change in any task; no multi-config sweep; no win-judging; no shared sync/async loop.
- Red-lines (§2): no hand-written "who is strong" (no faction-specific knobs introduced); outcome constants explicitly rejected in Task 4 guard; `data/playground/` vs `data/runs/` isolation preserved (web writes the former, CLI writes the latter); single-source readouts (`compute_match_result` calls `compute_readouts` verbatim); no "winner" field in the result envelope.

**2. Placeholder scan:** No `TBD`, `TODO`, "implement later", "similar to Task N", or vague "add validation" / "add error handling" remain. Every code step shows the actual code; every command step shows the actual command and expected output; every backtrack condition is concrete.

**3. Type consistency:**
- `pick_device(device=None, force_cpu: bool = False) -> torch.device` — same signature in Task 1 step 2, Task 2 step 1, Task 5 step 3.
- `build_engine_from_config(cfg: dict, device) -> tuple[Engine, dict]` — same signature in Task 1, Task 3 (consumed), Task 5 (consumed).
- `compute_match_result(eng, parquet_path: str) -> dict` returning `{"path", "ticks", "final"}` — defined Task 3, consumed Task 5, asserted in Task 5 happy-path test.
- `validate_config_keys(cfg: dict) -> None` raising `ValueError` — defined Task 4, used in Task 4 / 5 main pipeline.
- `ALLOWED_KEYS = frozenset({"players", "grid", "K", "fill", "T", "seed"})` — only defined once, in Task 4.
- Result envelope keys (`total / occupied_cells / distinct_strains / n2 / d_max / faction_share`) consistent across Task 3 docstring, Task 5 happy-path assertions, and `webapp/readouts.py:compute_readouts` (the single source).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-24-s0-unify-cli-match-runner.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. Use `superpowers:subagent-driven-development`.
2. **Inline Execution** — execute tasks in this session with batch checkpoints. Use `superpowers:executing-plans`.

Which approach?
