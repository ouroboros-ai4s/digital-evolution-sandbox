# Phenotype Cache Rebuild Performance Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the ~1080ms/tick `StrainTable.phenotype_arrays` rebuild cost by replacing `10n` scalar GPU writes with one bulk host→device transfer per field.

**Architecture:** `phenotype_arrays` currently rebuilds the per-field device tensors with a Python `for sid in range(1, n)` loop that does ~10 scalar assignments into CUDA tensors (`f[sid] = phe.f` …). Each scalar store is a CPU→GPU sync. The fix builds 10 plain Python lists on the CPU first (no syncs), then constructs each device tensor in a single `torch.tensor(list, device=…)` call — collapsing `10n` syncs into 10 bulk transfers. The dirty-flag cache, key-set, dtypes, and EMPTY-sentinel-row defaults are preserved exactly, so this is a behavior-neutral refactor verified against the existing test suite.

**Tech Stack:** Python 3.12, PyTorch 2.10 (cu128), pytest 9. Conda env `basic`, interpreter `D:/anaconda3/envs/basic/python.exe`.

## Global Constraints

- Target interpreter: `D:/anaconda3/envs/basic/python.exe` (conda env `basic`). Do not install or add dependencies — torch/pyarrow/pytest are already present.
- `pyproject.toml` sets `pythonpath = ["src"]`, so `pytest` resolves `des.*` without env setup. Standalone scripts under `scripts/` need `PYTHONPATH=src` (bash) or `$env:PYTHONPATH='src'` (PowerShell).
- `phenotype_arrays(device)` MUST keep its exact public contract: return a dict with key-set `{"f", "p_leave", "z_raw", "p_x", "prey_mask", "feature_mask", "period", "dir_bits", "repro_period", "anta_period"}`; float fields dtype `torch.float32`, int fields dtype `torch.int64`; shape `len(table) + 1` per field (EMPTY row at index 0).
- EMPTY sentinel row (index 0): `f/p_leave/z_raw/p_x/prey_mask/feature_mask/dir_bits = 0`, and `period/repro_period/anta_period = 1` (M1: avoids modulo-by-zero in the firing clock).
- Dirty-flag cache contract (existing test `test_phenotype_arrays_cached_until_mint`): repeated calls with no new mint return the SAME dict object; a `get_or_mint` of a new strain must invalidate it so the next call rebuilds.
- This is a performance refactor: observable behavior (values, dtypes, keys, cache identity) is unchanged. Tests added here are characterization/guard tests — they pass both before and after the refactor.

---

## File Structure

- `src/des/phenotype_cache.py` — **modify** `StrainTable.phenotype_arrays` (lines 44-83). This is the only production-code change. The dirty-flag cache machinery (`__init__` lines 14-17, `get_or_mint` dirty mark line 28, cache short-circuit lines 46-47) stays untouched; only the rebuild body that allocates and fills the tensors changes from scalar GPU writes to list-build + bulk transfer.
- `tests/test_phenotype_cache.py` — **modify** (append tests). Add a characterization test that locks the exact per-strain values/dtypes the refactor must preserve. Existing tests (`test_phenotype_arrays_indexed_by_id`, `test_phenotype_arrays_cached_until_mint`, `test_phenotype_arrays_has_dir_and_periods`) already guard key-set, cache identity, EMPTY row, and dir/period fields; the new test guards full-array equality across all 10 fields.
- `scripts/run_batch.py` — **modify** `phase_probe` (the per-phase timing helper, lines ~83-101). Move `_refresh_phe()` out of the `arb` bucket into its own `phe` bucket so the printout stops misattributing ~1080ms/tick of phenotype-rebuild cost to arbitration (spec §1.1, the 20× gap). Measurement-only; no engine behavior changes.

**Note on TDD discipline:** This is a behavior-preserving refactor. The honest discipline is *characterization-first* (green-first): write a test that pins down the current output, confirm it PASSES on the current code (it now documents the contract the refactor must keep), perform the refactor, confirm it STILL passes. There is no genuine red phase to fake — the output is identical by design. The guard against regression is the existing suite plus the new equivalence test.

---

### Task 1: Bulk-transfer rebuild in `phenotype_arrays`

**Files:**
- Modify: `src/des/phenotype_cache.py:44-83`
- Test: `tests/test_phenotype_cache.py` (append)

**Interfaces:**
- Consumes: `StrainTable.get_or_mint(sequence: tuple[str, ...]) -> int`, `StrainTable._id_to_phe: list[Phenotype | None]`, `StrainTable._next: int`, `Phenotype` fields `f, p_leave, z_raw, p_x, prey_mask, feature_mask, period, dir_bits, repro_period, anta_period`.
- Produces: unchanged public contract — `phenotype_arrays(device: torch.device) -> dict[str, torch.Tensor]` with key-set `{"f","p_leave","z_raw","p_x","prey_mask","feature_mask","period","dir_bits","repro_period","anta_period"}`, float fields `float32`, int fields `int64`, shape `len(table)+1`, EMPTY row at index 0 (zeros except `period/repro_period/anta_period = 1`), and the same dirty-flag cache identity behavior.

- [ ] **Step 1: Write the characterization test**

Append to `tests/test_phenotype_cache.py`:

```python
# ---------------------------------------------------------------------------
# Task-1 (perf) test: bulk-transfer rebuild must equal per-strain phenotype
# values across every field. Locks the contract the refactor preserves.
# ---------------------------------------------------------------------------

_FIELD_ATTRS = ("f", "p_leave", "z_raw", "p_x", "prey_mask", "feature_mask",
                "period", "dir_bits", "repro_period", "anta_period")


def test_phenotype_arrays_bulk_matches_per_strain():
    t = StrainTable()
    sids = [
        t.get_or_mint(("F4Nr1",)),
        t.get_or_mint(("F4Nr4", "BroadSweep")),
        t.get_or_mint(("P_hotspot", "N0")),
        t.get_or_mint(("BroadSweep", "F4Nr1", "P_base")),
    ]
    arr = t.phenotype_arrays(torch.device("cpu"))

    # shape + EMPTY row
    for key in _FIELD_ATTRS:
        assert arr[key].shape[0] == len(t) + 1, f"{key} wrong length"
    for key in ("f", "p_leave", "z_raw", "p_x", "prey_mask", "feature_mask", "dir_bits"):
        assert arr[key][EMPTY_ID].item() == 0, f"{key} EMPTY row must be 0"
    for key in ("period", "repro_period", "anta_period"):
        assert arr[key][EMPTY_ID].item() == 1, f"{key} EMPTY row must be 1"

    # dtypes
    for key in ("f", "p_leave", "z_raw", "p_x"):
        assert arr[key].dtype == torch.float32, f"{key} must be float32"
    for key in ("prey_mask", "feature_mask", "period", "dir_bits",
                "repro_period", "anta_period"):
        assert arr[key].dtype == torch.int64, f"{key} must be int64"

    # every minted strain's row equals its Phenotype field, field by field
    for sid in sids:
        phe = t.phenotype_of(sid)
        for key in _FIELD_ATTRS:
            got = arr[key][sid].item()
            want = getattr(phe, key)
            if key in ("f", "p_leave", "z_raw", "p_x"):
                assert abs(got - want) < 1e-6, f"{key}[{sid}]={got} != {want}"
            else:
                assert got == want, f"{key}[{sid}]={got} != {want}"
```

- [ ] **Step 2: Run the test to confirm it PASSES on current code**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_phenotype_cache.py::test_phenotype_arrays_bulk_matches_per_strain -v`
Expected: PASS. (Green-first — this characterizes the behavior the refactor must preserve; there is no red phase for a behavior-neutral refactor.)

- [ ] **Step 3: Refactor `phenotype_arrays` to list-build + bulk transfer**

Replace the body of `phenotype_arrays` in `src/des/phenotype_cache.py` (currently lines 44-83) with:

```python
    def phenotype_arrays(self, device: torch.device) -> dict[str, torch.Tensor]:
        # I1: return cached tensors when not dirty and same device
        if not self._arrays_dirty and self._cached_device == device and self._cached_arrays is not None:
            return self._cached_arrays
        n = self._next
        # Build per-field CPU lists in one pass, then do ONE bulk host->device
        # transfer per field. The old loop assigned each f[sid]=scalar straight
        # into a CUDA tensor -> 10n CPU->GPU syncs/rebuild (~1080ms/tick at n~1e4).
        # ponytail: list-build + 10 torch.tensor() calls = 10 transfers, not 10n.
        # Index 0 = EMPTY sentinel: zeros, except periods=1 (M1: modulo-by-zero
        # guard in the (T-birth)%period firing clock; id 0 never fires anyway).
        f = [0.0] * n
        p_leave = [0.0] * n
        z_raw = [0.0] * n
        p_x = [0.0] * n
        prey = [0] * n
        feat = [0] * n
        dir_bits = [0] * n
        period = [1] * n
        repro_period = [1] * n
        anta_period = [1] * n
        for sid in range(1, n):
            phe = self._id_to_phe[sid]
            if phe is None:
                raise KeyError(f"strain id {sid} has no phenotype (internal error)")
            f[sid] = phe.f
            p_leave[sid] = phe.p_leave
            z_raw[sid] = phe.z_raw
            p_x[sid] = phe.p_x
            prey[sid] = phe.prey_mask
            feat[sid] = phe.feature_mask
            period[sid] = phe.period
            dir_bits[sid] = phe.dir_bits
            repro_period[sid] = phe.repro_period
            anta_period[sid] = phe.anta_period
        result = {
            "f": torch.tensor(f, dtype=torch.float32, device=device),
            "p_leave": torch.tensor(p_leave, dtype=torch.float32, device=device),
            "z_raw": torch.tensor(z_raw, dtype=torch.float32, device=device),
            "p_x": torch.tensor(p_x, dtype=torch.float32, device=device),
            "prey_mask": torch.tensor(prey, dtype=torch.int64, device=device),
            "feature_mask": torch.tensor(feat, dtype=torch.int64, device=device),
            "period": torch.tensor(period, dtype=torch.int64, device=device),
            "dir_bits": torch.tensor(dir_bits, dtype=torch.int64, device=device),
            "repro_period": torch.tensor(repro_period, dtype=torch.int64, device=device),
            "anta_period": torch.tensor(anta_period, dtype=torch.int64, device=device),
        }
        # store cache and clear dirty flag
        self._cached_arrays = result
        self._cached_device = device
        self._arrays_dirty = False
        return result
```

- [ ] **Step 4: Run the new test plus the full cache suite to confirm still green**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest tests/test_phenotype_cache.py -v`
Expected: PASS — all tests including `test_phenotype_arrays_bulk_matches_per_strain`, `test_phenotype_arrays_cached_until_mint`, `test_phenotype_arrays_indexed_by_id`, and `test_phenotype_arrays_has_dir_and_periods`.

- [ ] **Step 5: Commit**

```bash
git add src/des/phenotype_cache.py tests/test_phenotype_cache.py
git commit -m "perf: bulk-transfer phenotype_arrays rebuild (10n GPU syncs -> 10)"
```

---

### Task 2: Fix phase-probe misattribution (`_refresh_phe` charged to `arb`)

**Files:**
- Modify: `scripts/run_batch.py:72-101` (the `phase_probe` function)

**Interfaces:**
- Consumes: `Engine._refresh_phe()`, `Engine.world`, `phase1_antagonism`, `phase2_reproduce`, `phase3_arbitrate_vec` (all already imported inside `phase_probe`).
- Produces: nothing importable — `phase_probe` is a CLI-only diagnostic that prints a timing line. After this task the printed line gains a `phe` bucket and the `arb` bucket no longer includes phenotype-rebuild time.

**Why no unit test:** `phase_probe` is a print-only diagnostic (no return value, no importable behavior). Per YAGNI it does not earn a pytest harness. Verification is running the probe and reading the printed buckets. The correctness check is structural: `_refresh_phe()` must sit between a `sync()/perf_counter()` pair so its cost lands in its own bucket, not `arb`.

- [ ] **Step 1: Add a `phe` bucket and time `_refresh_phe()` into it**

In `scripts/run_batch.py`, replace the `acc` initializer (line 79):

```python
    acc = {"anta": 0.0, "repro": 0.0, "arb": 0.0}
```

with:

```python
    acc = {"anta": 0.0, "repro": 0.0, "phe": 0.0, "arb": 0.0}
```

- [ ] **Step 2: Split `_refresh_phe()` into its own timed section**

Replace the loop body (lines 83-97) so `_refresh_phe()` is bracketed by its own `sync()/perf_counter()` and `arb` starts only after it. New loop body:

```python
    for _ in range(ticks):
        ssid, scnt, sfac = e.world.snapshot()
        sync(); t0 = time.perf_counter()
        post = phase1_antagonism(ssid, scnt, sfac, e._phe, e.birth, e.T, e.z_max, e.gen)
        sync(); t1 = time.perf_counter(); e.world.count = post
        buf, live = phase2_reproduce(e.world, ssid, scnt, sfac, e._phe, e.table,
                                     e.birth, e.T, e.gen)
        sync(); t2 = time.perf_counter(); e.world.count = live
        e._refresh_phe()
        sync(); t3 = time.perf_counter()
        nsid, ncnt, nfac, nbirth = phase3_arbitrate_vec(
            e.world.strain_id, e.world.count, e.world.faction, buf.tensors(), e.K,
            e.birth, e.T, e.gen, MAXSID=len(e.table) + 1, NFAC=4)
        sync(); t4 = time.perf_counter()
        e.world.strain_id, e.world.count, e.world.faction, e.birth = nsid, ncnt, nfac, nbirth
        e.T += 1
        acc["anta"] += (t1 - t0); acc["repro"] += (t2 - t1)
        acc["phe"] += (t3 - t2); acc["arb"] += (t4 - t3)
```

- [ ] **Step 3: Update the printout to include the `phe` bucket**

Replace the print statement (lines 99-101):

```python
    print(f"[phase-probe {ticks} ticks] "
          f"anta {1000*acc['anta']/n:.1f} | repro {1000*acc['repro']/n:.1f} | "
          f"arb {1000*acc['arb']/n:.1f} ms/tick (occupancy grows over the run)")
```

with:

```python
    print(f"[phase-probe {ticks} ticks] "
          f"anta {1000*acc['anta']/n:.1f} | repro {1000*acc['repro']/n:.1f} | "
          f"phe {1000*acc['phe']/n:.1f} | arb {1000*acc['arb']/n:.1f} "
          f"ms/tick (occupancy grows over the run)")
```

- [ ] **Step 4: Run the probe and confirm `phe` is now small (post-Task-1) and broken out**

Run (PowerShell): `$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --phase-probe 200`
Expected: a line of the form `[phase-probe 200 ticks] anta … | repro … | phe … | arb … ms/tick`. After Task 1 the `phe` bucket should be small (single-digit to low-tens ms/tick, not ~1080), and `arb` should match the standalone curve (`diag_arb_curve.py`, tens of ms/tick) rather than the old ~1129ms/tick. This confirms both the misattribution fix and that Task 1 removed the rebuild cost.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_batch.py
git commit -m "diag: give _refresh_phe its own phase-probe bucket (was charged to arb)"
```

---

### Task 3: Full-suite regression + end-to-end timing verification

**Files:**
- No code changes. This task is the integration gate: confirm nothing regressed and the original symptom (T=450×4 batch blocked by phenotype rebuild) is resolved.

**Interfaces:**
- Consumes: the whole `des` package and `tests/`. Nothing produced.

- [ ] **Step 1: Run the full test suite**

Run: `D:/anaconda3/envs/basic/python.exe -m pytest -q`
Expected: PASS — entire suite green (no behavior change from a perf refactor). Pay attention to `test_phenotype_cache.py`, `test_engine.py`, `test_integration.py`, and `test_acceptance.py`, which exercise `phenotype_arrays` through `Engine._refresh_phe`.

- [ ] **Step 2: Confirm the rebuild cost is gone via the standalone op-level diagnostic**

Run (PowerShell): `$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/diag_arb_sections.py 200`
Expected: the printed `phase3 full-world: … ms/call` stays in the tens-of-ms range (arbitration was already fast per spec §1.1). This diagnostic isolates phase3 and does not itself call `phenotype_arrays` in its timed section, so it serves as the unchanged-arb baseline confirming the fix did not perturb arbitration.

- [ ] **Step 3: Confirm end-to-end the batch is no longer rebuild-bound**

Run (PowerShell): `$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --phase-probe 450`
Expected: `phe` bucket small (single-to-low-tens ms/tick) at tick 450 with a ~10k-strain table — versus the pre-fix ~1080ms/tick. The full 450-tick probe completing in well under a minute (rather than stalling) is the end-to-end confirmation that the T=450×4 run is unblocked.

- [ ] **Step 4: Commit (only if a measurement note or doc was added)**

No code changes in this task; nothing to commit unless Step 1-3 surfaced a regression. If a regression appears, STOP — do not paper over it; diagnose whether the refactor changed a dtype/value (compare against the Task 1 characterization test) and fix in `phenotype_cache.py`.

---

## Self-Review

**1. Spec coverage.** The spec (`2026-06-22-phenotype-cache-perf-design.md`) is a measured diagnosis ending at §1.1's "core waste" observation — there is no separate proposed-solution section to cover. Mapping its claims to tasks:
- §1.1 "10n scalar GPU writes / per-tick full rebuild" is the bottleneck → **Task 1** replaces the scalar-store loop with list-build + 10 bulk transfers.
- §1.1 "phase-probe arb bucket reports 1129ms/tick, ~20× the real arb because it counts `_refresh_phe()`" → **Task 2** breaks `_refresh_phe()` into its own `phe` bucket.
- §1.1 "strain ids append monotonically; `_id_to_phe[sid]` is immutable after mint; Phenotype objects are already CPU-cached" — this is the *why the fix is safe* premise (the rebuild only moves already-computed values to GPU), validated by **Task 1**'s field-by-field equivalence test and **Task 3**'s full suite. No gap.

**2. Placeholder scan.** No TBD/TODO/"handle edge cases"/"similar to Task N". Every code step shows the full replacement code. Commands are exact with expected output. Clear.

**3. Type consistency.** `phenotype_arrays(device)` key-set and dtypes are stated identically in Global Constraints, Task 1 Interfaces, the Task 1 test (`_FIELD_ATTRS` tuple), and the refactored dict — all 10 keys match: `f, p_leave, z_raw, p_x, prey_mask, feature_mask, period, dir_bits, repro_period, anta_period`. Float fields (`f, p_leave, z_raw, p_x`) → `float32`; the other six → `int64`. EMPTY-row defaults (`period/repro_period/anta_period = 1`, rest `0`) are consistent across constraints, refactor code, and test. `phase_probe`'s `acc` dict keys (`anta, repro, phe, arb`) match between Task 2 Steps 1, 2, and 3. Consistent.

**Honesty note carried into execution:** Tasks 1-2 are behavior-neutral; their tests are green-first characterization, not red→green TDD. This is the correct discipline for a perf refactor and is stated explicitly so the executing agent does not fabricate a failing phase.
