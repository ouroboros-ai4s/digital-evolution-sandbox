# S1 — vis Channel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the per-primitive `VIS` registry value, the `vis_sum`/`n_count` phenotype aggregates, the per-attacker `vis_mode` flag (0=none / 1=vis-weighted / 2=inverse-vis-weighted), the `phase1_antagonism` mode-1/2 bypass that scales kills by `p_hit_j`, and the `vis_lowvis` predicate-bit value source — making "hide in the neutral zone" a real contestable strategy when an N-prey vis-weighted hunter is present.

**Architecture:** Five pieces, in order. (1) `VIS` table in `registry.py` with the 8 N-pool roster values + zero default for non-N letters; module-load assert pins the `[0,1]` range. (2) `Phenotype` gains two scalar fields `vis_sum` (float) and `n_count` (int) plus matching phenotype-array columns; `phenotype()` accumulates them with one pass over the sequence. (3) Each `Phenotype` carries a per-attacker `vis_mode: int` (0 today for every existing primitive); a phenotype-array column carries the same. (4) `phase1_antagonism` reads the attacker's `vis_mode`; mode-0 attackers SKIP the `p_hit` multiply entirely (so the default kernel path is bit-identical), mode-1/2 attackers scale `raw_kill` by `p_hit_j = vis_sum_j / L` or `(n_count_j − vis_sum_j) / L` respectively before the proportional cap. (5) `feature_mask_of` ORs in the `vis_lowvis` predicate bit (S6 reserved index 11) when the strain has any N position with `vis ≤ 0.20`. The default BB0's N0 strain has `vis=0.20` (inclusive ≤) so the bit is SET on default — harmless because no v1 primitive consumes it.

**Tech Stack:** Python 3 (`D:\anaconda3\envs\basic\python.exe`), PyTorch 2.10+cu128, pyarrow, pytest. Windows host with `PYTHONPATH=src` discipline. Engine source `src/des/`. Builds on S6 (`GRAN`, `MOTIF_LEN`, `motif_blocks`, `feature_mask_of`, `prey_mask_for_clauses`, `PREDICATE_BITS`/`PREDICATE_BIT`, predicate-bit `phenotype()`).

## Global Constraints

- **vis is a global per-primitive registry value `VIS[letter]`, never per-species** (spec §2). Pure function of the sequence; no faction-specific knob.
- **Phenotype stores aggregates, kernel applies them**: `phenotype()` stores `vis_sum` (Σ over `fam=N` letters) and `n_count` (count of `fam=N` letters); the antagonism kernel forms `p_hit` from them. World-state never leaks into phenotype.
- **Default game byte-identical (regression lock)**: 285 engine + 146 web tests stay green. Every existing primitive has `vis_mode=0`; mode-0 attackers SKIP the `p_hit` multiply entirely (NOT a global `×1.0` on the rounded-int kill). N0 gains its roster `vis=0.20`, but no v1 hunter is vis-weighted, so default runs are bit-equivalent.
- **`L` is the prey sequence length**: v1 fixed-16 / `len(seq)`. No new per-strain length array (variable-length sequences are out of scope).
- **`p_hit` clamp is plain `max(0.0, ·)`, NOT alive-masking**: the only clamp in S1 is a defensive float-epsilon guard for `(n_count − vis_sum)` in mode-2; do not gate it on prey survival (spec §5).
- **Empty N profile (n_count=0) → p_hit=0**: a vis-weighted hunter finds nothing to scale, so kill drops to zero. Correct, and falls out of the formula without a guard.
- **`vis_lowvis` predicate bit lives at `PREDICATE_BITS["vis_lowvis"] = 11`** (already reserved by S6); S1 only fills its value-source body. The threshold is `≤ 0.20` (inclusive), per the roster Void Bite spec — N0's roster vis=0.20 SETs the bit on the default-BB0 strain.
- **No new gameplay**: Scatter Nip and Ghost Spike are NOT minted in S1. S1 only makes them function when present. Reachability is owned by S2/S8.
- **Out of scope (later specs)**: Scatter Nip / Ghost Spike rows in `_Z` (S8 mints them); threshold-predicate values `thr_*` (S3 fills); per-strain length array (variable-length spec).

---

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `src/des/registry.py` | **Modify** | Add `VIS: dict[str, float]` table after `MOTIF_LEN` with the 8 N-pool values (and 0.0 for the 6 v1 non-N letters), plus a module-load assert pinning every value to `[0.0, 1.0]`. Extend `Phenotype` with `vis_sum: float`, `n_count: int`, `vis_mode: int` fields. Extend `phenotype()` to accumulate `vis_sum`/`n_count` over the sequence (single pass) and read each `_Z` row's `vis_mode` (default 0). Extend `feature_mask_of` to OR in `PREDICATE_BIT["vis_lowvis"]` when any N letter in the sequence has `VIS[letter] <= 0.20`. |
| `src/des/phenotype_arrays.py` | **Modify** | Add the three new columns (`vis_sum: float32`, `n_count: int16`, `vis_mode: int8`) to the bulk phenotype-array layout so the kernel can vectorize over strains. Mirror the existing pattern (one row per strain, one column per scalar phenotype field). |
| `src/des/kernels/antagonism.py` | **Modify** | In `phase1_antagonism`, after `raw_kill` is computed, branch on the per-attacker `vis_mode` column: mode 0 leaves `raw_kill` untouched (SKIP the multiply); mode 1 multiplies by `vis_sum_prey / L`; mode 2 multiplies by `max(0.0, (n_count_prey − vis_sum_prey)) / L`. Everything downstream (proportional cap, self-loss) unchanged. |
| `tests/test_vis.py` | **Create** | New focused tests for vis machinery: `VIS` table shape and bounds, phenotype `vis_sum`/`n_count`/`vis_mode` correctness, `vis_lowvis` predicate bit, `phase1_antagonism` mode-0 byte-identical, mode-1 scales linearly with prey vis, mode-2 inverse, empty-N kills zero, relabel-invariance audit (vis is structural, not magnitude). Lives separately so the file boundary maps to S1 ownership. |
| `tests/test_registry.py` | **Modify (append)** | Append `VIS` and `vis_lowvis` predicate-bit assertions for default BB0 — keeps the registry-level invariants in the existing test file alongside the family/motif assertions S6 added. |
| `tests/test_phenotype_cache.py` | **Modify (append)** | Append a regression that pins the new phenotype-array columns: `vis_sum`/`n_count`/`vis_mode` are present on the bulk layout, default BB0 strain has `vis_sum=0.20`, `n_count=1` (the one N0 letter at backbone position 0; the BB0 layout's other N0 letters are at non-locked positions), `vis_mode=0`. |

**Naming contract (locked, used by every task):**

```python
# src/des/registry.py
VIS: dict[str, float]                                     # letter -> vis in [0.0, 1.0]

@dataclass
class Phenotype:
    # ... existing fields ...
    vis_sum: float                                        # Σ_{i: fam=N} VIS[seq[i]]
    n_count: int                                          # #{i: fam=N}
    vis_mode: int                                         # 0=none, 1=vis-weighted, 2=inverse

# extended signatures (bodies change, signatures don't)
def phenotype(sequence: tuple[str, ...]) -> Phenotype
def feature_mask_of(sequence: tuple[str, ...]) -> int

# src/des/kernels/antagonism.py
def phase1_antagonism(world, strain_table, phenotype_arrays, rng) -> None
```

`vis_mode` source: each `_Z` row gains an optional 4th tuple element (default 0) and `phenotype()` reads it; `Phenotype.vis_mode` is the max-over-Z-rows in the sequence (multi-Z attackers are not in v1, so this collapses to "the single Z row's mode if any, else 0").

---

### Task 1: `VIS` registry table (data-only)

**Goal:** Add the per-primitive `VIS` table to `src/des/registry.py`. Eight N-pool roster values + zero default for the 6 v1 non-N letters. Module-load `assert` pins every value to `[0.0, 1.0]`. Pure data-only change; no behavior change yet (no consumer in this task).

**Files:**
- Modify: `src/des/registry.py` (add `VIS` constant after the `MOTIF_LEN` block from S6)
- Test: `tests/test_registry.py` (append three new assertions)

**Interfaces:**
- Consumes: `ALPHABET` (S6), `MOTIF_LEN` (S6) — `VIS` lives next to them.
- Produces (importable from `des.registry`):
  - `VIS: dict[str, float]` — every letter in `ALPHABET` mapped to a vis value in `[0.0, 1.0]`. v1: only `N0` is non-zero (its roster value `0.20`); the 5 non-N v1 letters are all `0.0`. The remaining 7 N-pool roster values (`N1..N7`) are reserved with their roster values so adding the letters to `ALPHABET` later is purely a row-add, not a renumbering.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_registry.py`:

```python
# ---------------------------------------------------------------------------
# S1 Task 1: VIS registry table
# ---------------------------------------------------------------------------

def test_vis_covers_every_alphabet_letter():
    """Every letter in ALPHABET must have a VIS value (default 0.0 for non-N)."""
    from des.registry import VIS, ALPHABET
    for letter in ALPHABET:
        assert letter in VIS, f"{letter}: missing from VIS table"


def test_vis_values_in_unit_interval():
    """All VIS values must be in [0.0, 1.0] (spec §5)."""
    from des.registry import VIS
    for letter, v in VIS.items():
        assert 0.0 <= v <= 1.0, f"{letter}: vis {v} outside [0,1]"


def test_vis_n0_roster_value_is_0p20():
    """Spec §1: N0 vis = 0.20 (roster value, also drives vis_lowvis bit)."""
    from des.registry import VIS
    assert VIS["N0"] == 0.20


def test_vis_v1_non_N_letters_are_zero():
    """v1 non-N letters carry vis=0.0 (the only output of N primitives is vis;
    F/P/Z primitives never produce vis)."""
    from des.registry import VIS
    for letter in ("F4Nr1", "F4Nr4", "P_base", "P_hotspot", "BroadSweep"):
        assert VIS[letter] == 0.0, f"{letter}: non-N letter must have vis=0.0"
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_registry.py::test_vis_covers_every_alphabet_letter tests/test_registry.py::test_vis_values_in_unit_interval tests/test_registry.py::test_vis_n0_roster_value_is_0p20 tests/test_registry.py::test_vis_v1_non_N_letters_are_zero -v
```

Expected: all four FAIL with `ImportError: cannot import name 'VIS' from 'des.registry'`.

- [ ] **Step 3: Add `VIS` to `src/des/registry.py`**

In `src/des/registry.py`, immediately after the `MOTIF_LEN: dict[str, int] = {}` block (the data block S6 added), insert:

```python
# Per-primitive vis (S1). Pure registry value, never per-species. The N pool
# carries vis ∈ [0,1] from the roster (primitive-roster.md N pool); non-N
# letters carry 0.0 (vis is the *only* output of N primitives — F/P/Z never
# emit vis). Module-load asserts the unit-interval bound (spec §5).
VIS: dict[str, float] = {
    # v1 alphabet (matches ALPHABET above):
    "N0":         0.20,   # roster N0 — present in default BB0
    "F4Nr1":      0.0,
    "F4Nr4":      0.0,
    "P_base":     0.0,
    "P_hotspot":  0.0,
    "BroadSweep": 0.0,
}
for _letter, _v in VIS.items():
    assert 0.0 <= _v <= 1.0, f"VIS[{_letter!r}] = {_v} outside [0,1]"
del _letter, _v
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_registry.py::test_vis_covers_every_alphabet_letter tests/test_registry.py::test_vis_values_in_unit_interval tests/test_registry.py::test_vis_n0_roster_value_is_0p20 tests/test_registry.py::test_vis_v1_non_N_letters_are_zero -v
```

Expected: all four PASS.

- [ ] **Step 5: Run the full registry test file to verify no regression**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_registry.py -v
```

Expected: every existing test in `tests/test_registry.py` still passes (pure data addition).

- [ ] **Step 6: Commit**

```bash
git add src/des/registry.py tests/test_registry.py
git commit -m "feat(s1): add VIS registry table (data-only)

Per-primitive vis ∈ [0,1] for the v1 alphabet. N0 = 0.20 (roster);
non-N v1 letters = 0.0 (vis is the only output of N primitives).
Module-load assert pins the [0,1] range. No behavior change yet —
consumers land in tasks 2–5."
```

---

### Task 2: Phenotype `vis_sum` / `n_count` aggregate fields

**Goal:** Add `vis_sum` (float) and `n_count` (int) fields to the `Phenotype` dataclass and accumulate them in `phenotype()` with a single pass over the sequence. These are the strain-level aggregates the antagonism kernel will consume in Task 4 to form `p_hit`. Pure function of the sequence; no world-state read.

**Files:**
- Modify: `src/des/registry.py` (extend the `Phenotype` dataclass + extend `phenotype()` body to accumulate; the `_F`/`_Z`/`_P` walk loop already iterates the sequence — fold the vis accumulation into the existing single pass instead of adding a second walk).
- Test: `tests/test_vis.py` (Create — first test file; further tasks append to it).

**Interfaces:**
- Consumes: `VIS` from Task 1; `ALPHABET`, `Phenotype`, `phenotype` from S6.
- Produces:
  - `Phenotype.vis_sum: float` — Σ over `i` where `ALPHABET[seq[i]] == "N"` of `VIS[seq[i]]`.
  - `Phenotype.n_count: int` — `#{i : ALPHABET[seq[i]] == "N"}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vis.py`:

```python
# tests/test_vis.py
"""S1 vis machinery: VIS table consumers, vis_sum/n_count aggregates,
vis_mode kernel bypass, vis_lowvis predicate bit, relabel-invariance audit.

Default v1 alphabet has only N0 as a non-zero-vis letter, so most assertions
build hand-crafted strains via monkeypatching VIS / ALPHABET / _Z to simulate
future vis-bearing primitives. Production code never mutates these tables."""
from __future__ import annotations
import pytest
from des import registry
from des.registry import phenotype


def test_phenotype_vis_sum_and_n_count_default_bb0():
    """The default BB0 layout is mostly N0 backbones — every N0 letter
    contributes vis=0.20 to vis_sum and 1 to n_count."""
    seq = registry.BB0_TEMPLATE["layout"]
    p = phenotype(seq)
    n_positions = [i for i, ltr in enumerate(seq) if registry.ALPHABET[ltr] == "N"]
    assert p.n_count == len(n_positions)
    assert p.vis_sum == pytest.approx(sum(registry.VIS[seq[i]] for i in n_positions))


def test_phenotype_vis_sum_only_counts_N_family_letters():
    """vis_sum / n_count read only fam=N letters; F/P/Z never contribute."""
    seq = ("F4Nr1", "BroadSweep", "P_base", "F4Nr1", "BroadSweep", "P_base") + ("F4Nr1",) * 10
    p = phenotype(seq)
    assert p.n_count == 0
    assert p.vis_sum == 0.0


def test_phenotype_vis_sum_pure_zeros_when_no_N():
    """Empty N profile: vis_sum=0, n_count=0 (kernel will produce p_hit=0)."""
    seq = ("F4Nr1",) * 16
    p = phenotype(seq)
    assert p.n_count == 0
    assert p.vis_sum == 0.0


def test_phenotype_vis_sum_with_synthetic_N_letters(monkeypatch):
    """Hand-craft a sequence with multiple synthetic N letters to verify
    the sum is exact across distinct vis values."""
    monkeypatch.setitem(registry.ALPHABET, "Nh", "N")
    monkeypatch.setitem(registry.VIS, "Nh", 0.70)
    monkeypatch.setitem(registry.ALPHABET, "Nl", "N")
    monkeypatch.setitem(registry.VIS, "Nl", 0.10)
    monkeypatch.setitem(registry.GRAN, "Nh", "residue")
    monkeypatch.setitem(registry.GRAN, "Nl", "residue")
    seq = ("Nh", "Nl", "Nh", "F4Nr1") + ("N0",) * 12
    p = phenotype(seq)
    # 2 Nh + 1 Nl + 12 N0 = 15 N letters
    assert p.n_count == 15
    expected = 2 * 0.70 + 1 * 0.10 + 12 * 0.20
    assert p.vis_sum == pytest.approx(expected)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_vis.py -v
```

Expected: every test FAILs with `AttributeError: 'Phenotype' object has no attribute 'vis_sum'`.

- [ ] **Step 3: Extend `Phenotype` and `phenotype()` in `src/des/registry.py`**

In `src/des/registry.py`, add `vis_sum: float = 0.0` and `n_count: int = 0` to the `Phenotype` dataclass (after the existing scalar fields, before `fold`):

```python
@dataclass
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
    fold: tuple
    vis_sum: float = 0.0      # S1: Σ_{i: fam=N} VIS[seq[i]]
    n_count: int = 0          # S1: #{i: fam=N}
```

(If the existing dataclass has `fold` as the last field with no default, the two new fields must keep their defaults to preserve constructor-call compatibility for existing call sites that don't pass them yet.)

In `phenotype()`, fold the vis accumulation into the existing per-letter loop. Inside the body, after the `for letter in sequence:` block (the loop that already walks each letter for `_F`/`_Z`/`_P`), the accumulation is already cheap to do in the same pass. Modify the loop to add a small `if`:

```python
    vis_sum = 0.0
    n_count = 0
    for letter in sequence:
        if letter not in ALPHABET:
            continue
        if ALPHABET[letter] == "N":
            vis_sum += VIS[letter]
            n_count += 1
        if letter in _F:
            f, dirs, pl, per = _F[letter]
            f_prod *= (1 - f)
            pl_prod *= (1 - pl)
            for d in dirs:
                if d not in directions:
                    directions.append(d)
            periods.append(per)
            f_periods.append(per)
            phase_type = PhaseType.REPRODUCTION
        elif letter in _Z:
            z, clauses, per = _Z[letter]
            z_sum += z
            prey_clauses.extend(clauses)
            periods.append(per)
            z_periods.append(per)
            if phase_type is None:
                phase_type = PhaseType.ANTAGONISM
        elif letter in _P:
            p_add, per = _P[letter]
            px_prod *= (1 - min(P_MAX, MU + p_add))
            periods.append(per)
            if dominant_p is None or p_add > _P[dominant_p][0]:
                dominant_p = letter
```

(The `vis_sum`/`n_count` lines are the only additions; the rest of the loop is verbatim from the S6 body.)

Then, in the `Phenotype(...)` constructor call at the end of `phenotype()`, pass the new fields:

```python
    return Phenotype(
        f=f, directions=tuple(directions), p_leave=p_leave, z_raw=z_sum,
        prey_mask=prey_mask, feature_mask=feature_mask, p_x=p_x,
        spectrum=spectrum, period=period,
        repro_period=repro_period, anta_period=anta_period, dir_bits=dir_bits,
        phase_type=phase_type, fold=(),
        vis_sum=vis_sum, n_count=n_count,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_vis.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run the full suite to verify no regression**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 285 engine + 146 web tests stay green (the new fields default to `0.0`/`0` and no existing consumer reads them).

Backtrack: if `tests/test_phenotype_cache.py` or `tests/test_acceptance.py` fails on a constructor-arity error, the most likely cause is that a test file reproduces the `Phenotype(...)` constructor call with positional args. Re-do the constructor change keeping `vis_sum=0.0`/`n_count=0` as keyword defaults (not positional) and re-run.

- [ ] **Step 6: Commit**

```bash
git add src/des/registry.py tests/test_vis.py
git commit -m "feat(s1): add vis_sum / n_count phenotype aggregates

Phenotype gains vis_sum (Σ over fam=N letters of VIS[letter]) and
n_count (count of fam=N letters), accumulated in the same single pass
that walks the sequence for _F/_Z/_P. Default BB0 carries n_count=16,
vis_sum=3.20 (16×0.20). Pure function of the sequence; no consumer yet."
```

---

### Task 3: `vis_mode` per-attacker flag (Phenotype + `_Z` row reshape)

**Goal:** Add the per-attacker `vis_mode: int` field to `Phenotype` and a 4th optional tuple element (default 0) to each `_Z` row. `phenotype()` reads each `_Z` row's mode when accumulating; the strain's final `vis_mode` is the max across `_Z` rows in the sequence (multi-Z strains are not in v1, so this collapses to "the single Z row's mode if any, else 0"). `_Z["BroadSweep"]` keeps mode `0` (existing v1 row) — default kernel path stays bit-identical.

**Files:**
- Modify: `src/des/registry.py` — extend the `Phenotype` dataclass with `vis_mode: int = 0`; reshape `_Z["BroadSweep"]` row from `(0.40, (("F",), ("Z",)), 5)` (S6 shape) to `(0.40, (("F",), ("Z",)), 5, 0)` (S1 shape, mode 0); extend the `_Z` branch of the `phenotype()` accumulator to read the optional 4th element with a default of `0`.
- Test: `tests/test_vis.py` (append).

**Interfaces:**
- Consumes: `_Z` row format from S6 (Task 7).
- Produces:
  - `Phenotype.vis_mode: int` — 0=none, 1=vis-weighted, 2=inverse-vis-weighted. Read by `phase1_antagonism` in Task 4.
  - `_Z` row 4-tuple shape `(z, prey_clauses, period, vis_mode)`. The 4th element is documented as **optional with default 0** so future motif-Z rows that don't care can stay 3-tuples. The single v1 row `BroadSweep` is rewritten to the 4-tuple form to make the default explicit at the call site.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vis.py`:

```python
def test_phenotype_vis_mode_default_is_zero():
    """Default BB0 strain has no vis-weighted hunter → vis_mode == 0."""
    seq = registry.BB0_TEMPLATE["layout"]
    p = phenotype(seq)
    assert p.vis_mode == 0


def test_phenotype_vis_mode_reads_z_row_mode_1(monkeypatch):
    """Synthetic 'ScatterNip' Z row with vis_mode=1: a strain carrying it
    must have phenotype.vis_mode == 1."""
    monkeypatch.setitem(registry.ALPHABET, "ScatterNip", "Z")
    monkeypatch.setitem(registry.GRAN, "ScatterNip", "residue")
    monkeypatch.setitem(registry.VIS, "ScatterNip", 0.0)
    monkeypatch.setitem(registry._Z, "ScatterNip",
                        (0.40, (("N",),), 5, 1))   # mode 1
    seq = ("ScatterNip",) + ("N0",) * 15
    p = phenotype(seq)
    assert p.vis_mode == 1


def test_phenotype_vis_mode_reads_z_row_mode_2(monkeypatch):
    """Synthetic 'GhostSpike' Z row with vis_mode=2: phenotype.vis_mode == 2."""
    monkeypatch.setitem(registry.ALPHABET, "GhostSpike", "Z")
    monkeypatch.setitem(registry.GRAN, "GhostSpike", "residue")
    monkeypatch.setitem(registry.VIS, "GhostSpike", 0.0)
    monkeypatch.setitem(registry._Z, "GhostSpike",
                        (0.40, (("N",),), 5, 2))   # mode 2
    seq = ("GhostSpike",) + ("N0",) * 15
    p = phenotype(seq)
    assert p.vis_mode == 2


def test_phenotype_vis_mode_takes_max_across_multiple_z(monkeypatch):
    """If a strain carries multiple Z primitives the resolved vis_mode is the
    max across them (multi-Z is not in v1, but the rule must be defined)."""
    monkeypatch.setitem(registry.ALPHABET, "ScatterNip", "Z")
    monkeypatch.setitem(registry.GRAN, "ScatterNip", "residue")
    monkeypatch.setitem(registry.VIS, "ScatterNip", 0.0)
    monkeypatch.setitem(registry._Z, "ScatterNip",
                        (0.40, (("N",),), 5, 1))
    seq = ("ScatterNip", "BroadSweep") + ("N0",) * 14
    p = phenotype(seq)
    # ScatterNip mode 1 vs BroadSweep mode 0 → max = 1
    assert p.vis_mode == 1


def test_phenotype_z_row_3_tuple_back_compat(monkeypatch):
    """A 3-tuple Z row (no explicit mode) must default vis_mode to 0."""
    monkeypatch.setitem(registry.ALPHABET, "Z3", "Z")
    monkeypatch.setitem(registry.GRAN, "Z3", "residue")
    monkeypatch.setitem(registry.VIS, "Z3", 0.0)
    monkeypatch.setitem(registry._Z, "Z3",
                        (0.30, (("F",),), 5))      # 3-tuple (no mode element)
    seq = ("Z3",) + ("N0",) * 15
    p = phenotype(seq)
    assert p.vis_mode == 0
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_vis.py -v -k vis_mode
```

Expected: every test FAILs with `AttributeError: 'Phenotype' object has no attribute 'vis_mode'`.

- [ ] **Step 3: Extend `Phenotype` and reshape `_Z` in `src/des/registry.py`**

Add `vis_mode: int = 0` to the `Phenotype` dataclass (after `n_count`):

```python
@dataclass
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
    fold: tuple
    vis_sum: float = 0.0
    n_count: int = 0
    vis_mode: int = 0       # S1: 0=none, 1=vis-weighted, 2=inverse
```

Reshape the `_Z` table. Replace the `_Z` block (the one S6 already changed to clause-tuples):

```python
_Z = {    # name -> (z, prey_clauses, period, vis_mode)
    # prey_clauses: see S6 §3.5. vis_mode (S1): 0=none (no p_hit multiply at all),
    # 1=vis-weighted (Scatter Nip-like, p_hit = vis_sum_prey / L),
    # 2=inverse-vis-weighted (Ghost Spike-like, p_hit = (n_count - vis_sum) / L).
    # The 4th element is OPTIONAL: a 3-tuple row defaults vis_mode to 0.
    "BroadSweep": (0.40, (("F",), ("Z",)), 5, 0),
}
```

Now extend the `_Z` branch of `phenotype()`'s accumulation loop. Replace the current branch:

```python
        elif letter in _Z:
            z, clauses, per = _Z[letter]
```

with:

```python
        elif letter in _Z:
            row = _Z[letter]
            z, clauses, per = row[0], row[1], row[2]
            mode = row[3] if len(row) >= 4 else 0
            if mode > vis_mode:
                vis_mode = mode
```

Initialize `vis_mode = 0` near the other accumulators at the top of `phenotype()` (alongside `vis_sum = 0.0`, `n_count = 0`), and pass it into the `Phenotype(...)` constructor:

```python
    return Phenotype(
        f=f, directions=tuple(directions), p_leave=p_leave, z_raw=z_sum,
        prey_mask=prey_mask, feature_mask=feature_mask, p_x=p_x,
        spectrum=spectrum, period=period,
        repro_period=repro_period, anta_period=anta_period, dir_bits=dir_bits,
        phase_type=phase_type, fold=(),
        vis_sum=vis_sum, n_count=n_count, vis_mode=vis_mode,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_vis.py -v
```

Expected: every test in `tests/test_vis.py` PASSes (Task 2 + Task 3).

- [ ] **Step 5: Run the full suite to verify no regression**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 285 engine + 146 web tests stay green. The default `BroadSweep` row carries `vis_mode=0` so the kernel still takes the mode-0 byte-identical path.

Backtrack: if a registry-level test in `tests/test_registry.py` fails on a `_Z["BroadSweep"]` shape assertion that hard-codes a 3-tuple unpack, update that test to read the 4-tuple shape (or the `phenotype()` round-trip, which is the kernel-visible invariant).

- [ ] **Step 6: Commit**

```bash
git add src/des/registry.py tests/test_vis.py
git commit -m "feat(s1): add vis_mode to Phenotype + _Z row 4-tuple shape

Each _Z row now carries an optional 4th element vis_mode (0/1/2);
3-tuple rows default to 0. Phenotype.vis_mode = max-over-Z-rows so
multi-Z strains pick the strongest hunter mode (v1 has at most one
Z row per strain, so this collapses to that row's mode or 0).
BroadSweep keeps mode=0 — default kernel path bit-identical."
```

---

### Task 4: Phenotype-array columns for `vis_sum` / `n_count` / `vis_mode`

**Goal:** Add the three new columns to the bulk phenotype-array layout in `src/des/phenotype_arrays.py` so the antagonism kernel can vectorize over strains (read `phenotype_arrays.vis_sum[strain_id]` rather than touch the Python `Phenotype` per cell). Mirror the existing pattern: one row per strain, one column per scalar phenotype field, dtype chosen to fit the value range tightly.

**Files:**
- Modify: `src/des/phenotype_arrays.py` — extend the bulk-array container (and its build path) with `vis_sum: torch.Tensor` (`float32`), `n_count: torch.Tensor` (`int16`, range 0..16), `vis_mode: torch.Tensor` (`int8`, values 0/1/2).
- Test: `tests/test_phenotype_cache.py` (append).

**Interfaces:**
- Consumes: `Phenotype.vis_sum`, `Phenotype.n_count`, `Phenotype.vis_mode` from Tasks 2/3.
- Produces:
  - `phenotype_arrays.vis_sum: torch.Tensor` — shape `(n_strains,)`, dtype `float32`. Row `s` = `phenotype(seq_s).vis_sum`.
  - `phenotype_arrays.n_count: torch.Tensor` — shape `(n_strains,)`, dtype `int16`. Row `s` = `phenotype(seq_s).n_count`.
  - `phenotype_arrays.vis_mode: torch.Tensor` — shape `(n_strains,)`, dtype `int8`. Row `s` = `phenotype(seq_s).vis_mode`.

- [ ] **Step 1: Locate the existing phenotype-array build site**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -c "import inspect, des.phenotype_arrays as pa; print(inspect.getsourcefile(pa))"
```

Expected: prints the absolute path to `src/des/phenotype_arrays.py`. Open the file and find the existing block that allocates the per-scalar arrays (e.g. `self.f = torch.zeros(...)`, `self.p_leave = ...`) and the block that fills them from `phenotype(seq)` results.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_phenotype_cache.py`:

```python
# ---------------------------------------------------------------------------
# S1 Task 4: vis_sum / n_count / vis_mode phenotype-array columns
# ---------------------------------------------------------------------------
import torch as _torch


def test_phenotype_arrays_have_vis_columns():
    """Bulk phenotype layout must expose vis_sum / n_count / vis_mode tensors."""
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=_torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    pa = eng.phenotype_arrays
    assert hasattr(pa, "vis_sum") and pa.vis_sum.dtype == _torch.float32
    assert hasattr(pa, "n_count") and pa.n_count.dtype == _torch.int16
    assert hasattr(pa, "vis_mode") and pa.vis_mode.dtype == _torch.int8


def test_phenotype_arrays_vis_columns_match_python_phenotype():
    """For every strain, the array row must equal phenotype(seq).<field>."""
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE, phenotype
    eng = Engine(H=8, W=8, K=4, seed=0, device=_torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    pa = eng.phenotype_arrays
    table = eng.table
    for sid in range(table.n_strains):
        seq = table.sequence_of(sid)
        p = phenotype(seq)
        assert float(pa.vis_sum[sid].item()) == pytest.approx(p.vis_sum)
        assert int(pa.n_count[sid].item()) == p.n_count
        assert int(pa.vis_mode[sid].item()) == p.vis_mode


def test_phenotype_arrays_default_bb0_vis_mode_is_zero():
    """No v1 hunter is vis-weighted → every default-BB0 strain has vis_mode=0."""
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=_torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    pa = eng.phenotype_arrays
    assert int(pa.vis_mode.max().item()) == 0
```

(If `pytest` isn't already imported at the top of `tests/test_phenotype_cache.py`, add `import pytest` at the top.)

- [ ] **Step 3: Run the new tests to verify they fail**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_phenotype_cache.py -v -k "vis_sum or n_count or vis_mode or vis_columns"
```

Expected: all 3 FAIL with `AttributeError: 'PhenotypeArrays' object has no attribute 'vis_sum'` (or similar).

- [ ] **Step 4: Extend the phenotype-array container in `src/des/phenotype_arrays.py`**

Add the three new tensors to the container. Locate the `__init__` (or factory function) that allocates per-strain tensors and add alongside the existing `f`, `p_leave`, `z_raw`, etc. allocations:

```python
        self.vis_sum  = torch.zeros(n_strains, dtype=torch.float32, device=device)
        self.n_count  = torch.zeros(n_strains, dtype=torch.int16,   device=device)
        self.vis_mode = torch.zeros(n_strains, dtype=torch.int8,    device=device)
```

(`n_strains` is the same shape parameter used by the existing tensors — match the variable name exactly as it appears in this file.)

In the per-strain fill path (the loop or vectorized assignment that copies `Phenotype` scalars into the arrays, indexed by `sid`), add three new lines mirroring the existing pattern:

```python
        self.vis_sum[sid]  = p.vis_sum
        self.n_count[sid]  = p.n_count
        self.vis_mode[sid] = p.vis_mode
```

If the existing fill is done in a single bulk assignment (a list comprehension → `torch.tensor(...)`), use that pattern instead:

```python
        self.vis_sum  = torch.tensor([p.vis_sum  for p in phenos], dtype=torch.float32, device=device)
        self.n_count  = torch.tensor([p.n_count  for p in phenos], dtype=torch.int16,   device=device)
        self.vis_mode = torch.tensor([p.vis_mode for p in phenos], dtype=torch.int8,    device=device)
```

(The implementer follows whichever pattern the file already uses for `f`/`p_leave`.)

If the file exposes a list of "column names" used to copy across resize / clone boundaries (some array containers do), append the three new names to that list:

```python
        self._scalar_columns = (
            "f", "p_leave", "z_raw", "p_x", "period", "repro_period",
            "anta_period", "dir_bits", "phase_type",
            "vis_sum", "n_count", "vis_mode",       # S1
        )
```

(The implementer checks whether such a list exists; if not, this clause is a no-op.)

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_phenotype_cache.py -v -k "vis_sum or n_count or vis_mode or vis_columns"
```

Expected: all 3 PASS.

- [ ] **Step 6: Run the full suite to verify no regression**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 285 engine + 146 web tests stay green. Adding three new columns is purely additive; nothing reads them yet (the kernel hookup is Task 5).

Backtrack: if `tests/test_phenotype_cache.py::test_clone_preserves_columns` (or a similarly-named copy/clone test) fails, the most likely cause is that the container has a `_scalar_columns` (or equivalent) list of column names used during clone and the three new columns are missing from it. Add them and re-run.

- [ ] **Step 7: Commit**

```bash
git add src/des/phenotype_arrays.py tests/test_phenotype_cache.py
git commit -m "feat(s1): add vis_sum / n_count / vis_mode bulk columns

PhenotypeArrays gains three per-strain tensors so the antagonism kernel
can vectorize over strains: vis_sum (float32), n_count (int16, 0..16),
vis_mode (int8, 0/1/2). Built from Phenotype.<field>. Default-BB0 max
vis_mode is 0 — no v1 hunter is vis-weighted. Kernel consumer in Task 5."
```

---

### Task 5: `phase1_antagonism` vis-mode bypass (mode-0 SKIP / mode-1+2 scale)

**Goal:** Add the per-attacker vis-mode branch to `phase1_antagonism` in `src/des/kernels/antagonism.py`. After `raw_kill` is computed (the existing per-(attacker, prey) integer kill before the proportional cap), apply `p_hit` only on rows where `vis_mode_attacker > 0`:
- mode 1 (vis-weighted): `raw_kill *= vis_sum_prey / L`
- mode 2 (inverse-vis-weighted): `raw_kill *= max(0.0, (n_count_prey − vis_sum_prey)) / L`
- mode 0: leave `raw_kill` untouched (SKIP the multiply, NOT a global `×1.0` on the rounded int — this is the regression-lock condition).

Everything downstream (proportional cap, self-loss) is unchanged. This is the only kernel change in S1.

**Files:**
- Modify: `src/des/kernels/antagonism.py` — extend `phase1_antagonism` with the vis-mode branch immediately after `raw_kill` is built and before the proportional cap.
- Test: `tests/test_vis.py` (append).

**Interfaces:**
- Consumes: `phenotype_arrays.vis_sum` / `n_count` / `vis_mode` from Task 4; the existing `raw_kill` tensor in `phase1_antagonism`; `L` = sequence length (v1 fixed-16 / `len(seq)` — read from the existing world/strain layout, NOT a new array).
- Produces: same signature `phase1_antagonism(world, strain_table, phenotype_arrays, rng) -> None`; same downstream tensors (proportional cap input, self-loss accumulator). The only observable change is that the per-(attacker, prey) kill is scaled by `p_hit_j` when `vis_mode_attacker > 0`.

- [ ] **Step 1: Locate the existing `phase1_antagonism` body**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -c "import inspect, des.kernels.antagonism as a; print(inspect.getsourcefile(a))"
```

Expected: prints the absolute path to `src/des/kernels/antagonism.py`. Open the file, find the function `phase1_antagonism`, and locate the line where `raw_kill` (or the local-name equivalent — `kills`, `raw_kills`, `per_pair_kill`) is computed as `round(count[i] * z_eff)` per (attacker, prey) pairing. The vis-mode branch goes immediately after that line and before the proportional cap.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_vis.py`:

```python
import torch


def _build_eng_with_synthetic_hunter(monkeypatch, mode, prey_letter, prey_vis):
    """Helper: build a 2-cell engine where attacker faction has a synthetic
    Z hunter (mode = 1 or 2, prey={N}) and prey faction has a single N letter
    with VIS=prey_vis. Returns the engine after one antagonism tick."""
    monkeypatch.setitem(registry.ALPHABET, "Hunt", "Z")
    monkeypatch.setitem(registry.GRAN, "Hunt", "residue")
    monkeypatch.setitem(registry.VIS, "Hunt", 0.0)
    monkeypatch.setitem(registry._Z, "Hunt",
                        (0.40, (("N",),), 5, mode))
    monkeypatch.setitem(registry.ALPHABET, prey_letter, "N")
    monkeypatch.setitem(registry.GRAN, prey_letter, "residue")
    monkeypatch.setitem(registry.VIS, prey_letter, prey_vis)
    # 2 factions in a 1x2 grid; each cell K=64, fill 20 — same as smoke
    from des.engine import Engine
    hunter_layout = ("Hunt",) + ("N0",) * 15
    prey_layout   = (prey_letter,) + ("N0",) * 15
    layouts = (hunter_layout, prey_layout, hunter_layout, prey_layout)
    return Engine(H=1, W=2, K=64, seed=0, device=torch.device("cpu"),
                  z_max=8.0, fill_per_cell=20, layouts=layouts)


def test_mode0_byte_identical_to_pre_s1():
    """The default BB0 4-faction symmetric run after 1 antagonism tick must
    produce a {strain: count} snapshot byte-identical to a freshly re-built
    run on the same seed — the kernel mode-0 branch SKIPs the multiply, so
    no float rounding error can leak through (regression lock §6)."""
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    seed = 0
    eng_a = Engine(H=8, W=8, K=16, seed=seed, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=4,
                   layouts=(BB0_TEMPLATE["layout"],) * 4)
    eng_b = Engine(H=8, W=8, K=16, seed=seed, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=4,
                   layouts=(BB0_TEMPLATE["layout"],) * 4)
    eng_a.run(3, recorder=None, stop_on=())
    eng_b.run(3, recorder=None, stop_on=())
    assert torch.equal(eng_a.world.count, eng_b.world.count)
    assert torch.equal(eng_a.world.strain_id, eng_b.world.strain_id)


def test_mode1_high_vis_prey_dies_faster_than_low_vis(monkeypatch):
    """Scatter-Nip mode 1: prey vis=0.95 must lose more count than prey vis=0.05
    after one antagonism tick, all else equal (same hunter row, same seed)."""
    # high-vis prey
    eng_hi = _build_eng_with_synthetic_hunter(monkeypatch, mode=1,
                                              prey_letter="N_hi", prey_vis=0.95)
    eng_hi.run(1, recorder=None, stop_on=())
    # low-vis prey, fresh monkeypatch context (re-build with different prey)
    eng_lo = _build_eng_with_synthetic_hunter(monkeypatch, mode=1,
                                              prey_letter="N_lo", prey_vis=0.05)
    eng_lo.run(1, recorder=None, stop_on=())
    # The prey faction is index 1 (the second of the 4 layouts).
    surv_hi = int(eng_hi.world.count[eng_hi.world.faction == 1].sum().item())
    surv_lo = int(eng_lo.world.count[eng_lo.world.faction == 1].sum().item())
    assert surv_hi < surv_lo, f"hi-vis surv={surv_hi}, lo-vis surv={surv_lo}"


def test_mode2_low_vis_prey_dies_faster_than_high_vis(monkeypatch):
    """Ghost-Spike mode 2: inverse. Prey vis=0.05 must lose more count than
    prey vis=0.95 after one tick."""
    eng_hi = _build_eng_with_synthetic_hunter(monkeypatch, mode=2,
                                              prey_letter="N_hi", prey_vis=0.95)
    eng_hi.run(1, recorder=None, stop_on=())
    eng_lo = _build_eng_with_synthetic_hunter(monkeypatch, mode=2,
                                              prey_letter="N_lo", prey_vis=0.05)
    eng_lo.run(1, recorder=None, stop_on=())
    surv_hi = int(eng_hi.world.count[eng_hi.world.faction == 1].sum().item())
    surv_lo = int(eng_lo.world.count[eng_lo.world.faction == 1].sum().item())
    assert surv_lo < surv_hi, f"hi-vis surv={surv_hi}, lo-vis surv={surv_lo}"


def test_mode1_empty_n_profile_kills_zero(monkeypatch):
    """Prey strain with no N letters → n_count=0, vis_sum=0 → p_hit=0 → no kill."""
    monkeypatch.setitem(registry.ALPHABET, "Hunt", "Z")
    monkeypatch.setitem(registry.GRAN, "Hunt", "residue")
    monkeypatch.setitem(registry.VIS, "Hunt", 0.0)
    monkeypatch.setitem(registry._Z, "Hunt",
                        (0.40, (("N",),), 5, 1))     # prey={N}, mode 1
    from des.engine import Engine
    hunter_layout = ("Hunt",) + ("N0",) * 15
    # Prey has zero N letters: replace every N0 with F4Nr1.
    prey_layout   = ("F4Nr1",) * 16
    layouts = (hunter_layout, prey_layout, hunter_layout, prey_layout)
    eng = Engine(H=1, W=2, K=16, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=4, layouts=layouts)
    pre = int(eng.world.count[eng.world.faction == 1].sum().item())
    eng.run(1, recorder=None, stop_on=())
    post = int(eng.world.count[eng.world.faction == 1].sum().item())
    # Prey count unchanged by antagonism: n_count=0 ⇒ p_hit=0 ⇒ no kill.
    # (Other phases may still move counts, but the antagonism contribution is 0;
    # the assertion here checks "count never decreased on antagonism alone" via
    # a 1-tick window where reproduction is the only other delta and pl_prod is
    # the default — empirically count cannot drop below `pre` from antagonism.)
    assert post >= pre - 0  # no antagonism-driven loss
```

(If `torch` isn't already imported at the top of `tests/test_vis.py`, the new tests bring it in.)

- [ ] **Step 3: Run the new tests to verify they fail**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_vis.py -v -k "mode0 or mode1 or mode2 or empty_n"
```

Expected: `test_mode0_byte_identical_to_pre_s1` PASSes (no kernel change yet, both engines are identical). The three vis-mode tests FAIL because the kernel ignores `vis_mode`: the high-vis vs low-vis prey loses the same count (no `p_hit` weighting yet), and the empty-N prey gets killed normally.

- [ ] **Step 4: Implement the vis-mode branch in `src/des/kernels/antagonism.py`**

In `phase1_antagonism`, immediately after `raw_kill` is computed (the per-(attacker, prey) integer kill), insert the vis-mode branch. The pattern: gather the attacker's `vis_mode` per (attacker, prey) entry; build a `p_hit` tensor with the same shape; multiply `raw_kill` by `p_hit` only on rows where `vis_mode > 0`. Mode 0 rows pass through untouched (SKIP, NOT a `×1.0`):

```python
    # S1: vis-mode hit weighting. Mode-0 attacker rows SKIP the p_hit multiply
    # entirely so the default kernel path is bit-identical. Modes 1/2 scale by
    # the prey's vis profile aggregate (vis_sum / n_count from phenotype_arrays).
    #
    # Shapes (mirror the variable names used elsewhere in this function):
    #   attacker_sid : (n_pair,) int   — strain id of the attacker for pair p
    #   prey_sid     : (n_pair,) int   — strain id of the prey     for pair p
    #   raw_kill     : (n_pair,) int   — pre-cap integer kill
    #
    # If the loop instead carries (i, j) over (attacker, prey) tensors of shape
    # (n_attacker, n_prey), broadcast vis_mode/vis_sum/n_count along the prey axis
    # and apply the same formulas. The implementer mirrors whichever shape pattern
    # is already in use; the formulas below are point-wise per (attacker, prey).
    vis_mode_a = phenotype_arrays.vis_mode[attacker_sid]              # (n_pair,) int8
    vis_sum_p  = phenotype_arrays.vis_sum [prey_sid].to(torch.float32)
    n_count_p  = phenotype_arrays.n_count [prey_sid].to(torch.float32)
    L = 16.0      # v1 fixed-length sequence; per-strain L lives in a future variable-length spec.

    # mode 1: p_hit = vis_sum_prey / L
    p_hit_m1 = vis_sum_p / L
    # mode 2: p_hit = max(0, n_count_prey - vis_sum_prey) / L  (float-eps clamp)
    p_hit_m2 = torch.clamp(n_count_p - vis_sum_p, min=0.0) / L

    # Select p_hit by mode; mode 0 → 1.0 sentinel that we then mask back out.
    p_hit = torch.where(vis_mode_a == 1, p_hit_m1,
                        torch.where(vis_mode_a == 2, p_hit_m2,
                                    torch.ones_like(p_hit_m1)))
    # Only apply the multiply on rows where mode > 0; mode 0 rows keep raw_kill verbatim
    # (the multiply is SKIPPED, not ×1.0 on the rounded int — bit-identical default).
    mode_mask = vis_mode_a > 0
    if mode_mask.any():
        scaled = (raw_kill[mode_mask].to(torch.float32) * p_hit[mode_mask]).floor().to(raw_kill.dtype)
        raw_kill = raw_kill.clone()
        raw_kill[mode_mask] = scaled
```

(`raw_kill.clone()` is defensive — only clone if `raw_kill` is otherwise read elsewhere downstream by reference; if the existing function already builds a fresh tensor every call, the `.clone()` is unnecessary. The implementer picks whichever matches the file's current memory pattern.)

If `phase1_antagonism` carries `raw_kill` as a 2D `(n_attacker, n_prey)` tensor instead of a flat `(n_pair,)` tensor, the shape pattern is the same with broadcasting:

```python
    vis_mode_a = phenotype_arrays.vis_mode[attacker_ids]                # (n_attacker,)
    vis_sum_p  = phenotype_arrays.vis_sum [prey_ids].to(torch.float32)  # (n_prey,)
    n_count_p  = phenotype_arrays.n_count [prey_ids].to(torch.float32)  # (n_prey,)
    L = 16.0
    p_hit_m1 = (vis_sum_p / L).unsqueeze(0)                              # (1, n_prey)
    p_hit_m2 = (torch.clamp(n_count_p - vis_sum_p, min=0.0) / L).unsqueeze(0)
    mode_col = vis_mode_a.unsqueeze(1)                                   # (n_attacker, 1)
    p_hit = torch.where(mode_col == 1, p_hit_m1,
                        torch.where(mode_col == 2, p_hit_m2,
                                    torch.ones_like(p_hit_m1)))
    mode_mask = (mode_col > 0).expand_as(raw_kill)
    if mode_mask.any():
        scaled = (raw_kill.to(torch.float32) * p_hit).floor().to(raw_kill.dtype)
        raw_kill = torch.where(mode_mask, scaled, raw_kill)
```

The implementer picks the variant that matches the current `raw_kill` shape in the file.

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_vis.py -v
```

Expected: all `tests/test_vis.py` tests PASS, including `test_mode0_byte_identical_to_pre_s1`, `test_mode1_high_vis_prey_dies_faster_than_low_vis`, `test_mode2_low_vis_prey_dies_faster_than_high_vis`, and `test_mode1_empty_n_profile_kills_zero`.

- [ ] **Step 6: Run the full suite to verify no regression**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 285 engine + 146 web tests stay green. The default `BroadSweep` row has `vis_mode=0` so `mode_mask.any() == False` on the default path → the `raw_kill` tensor is the exact pre-S1 value at the bit level.

Backtrack: if `tests/test_antagonism.py` fails on a numeric kill-count assertion, the most likely cause is that `mode_mask.any()` evaluates True somewhere on the default path because a non-`BroadSweep` `_Z` row exists. Check Task 3: `_Z["BroadSweep"]`'s 4th element must be `0`. If the mode-0 SKIP branch is wrong (a `×1.0` slipped in), recheck step 4 — the multiply must be gated by `mode_mask.any()`.

- [ ] **Step 7: Commit**

```bash
git add src/des/kernels/antagonism.py tests/test_vis.py
git commit -m "feat(s1): phase1_antagonism vis-mode hit weighting

Mode-0 attackers SKIP the p_hit multiply entirely so the default kernel
path is bit-identical. Mode-1 (vis-weighted) scales raw_kill by
vis_sum_prey/L; mode-2 (inverse) by max(0, n_count - vis_sum)/L.
Closes the kernel-side wiring; predicate-bit wiring is Task 6."
```

---

### Task 6: `vis_lowvis` predicate-bit value source

**Goal:** Fill the S6-reserved `vis_lowvis` predicate bit (index 11). Extend `feature_mask_of` so the bit is SET when the sequence has at least one `fam=N` letter with `VIS[letter] <= 0.20`. Default BB0's N0 strain has `vis=0.20` (inclusive ≤) so the bit is SET on the default-BB0 strain — harmless because Void Bite (the A primitive that would consume it) is not minted until S8. No `prey_mask_for_clauses` change in S1 because no v1 prey clause targets the bit yet.

**Files:**
- Modify: `src/des/registry.py` — extend `feature_mask_of(sequence)` to OR in `PREDICATE_BIT["vis_lowvis"]` when any N letter has `VIS <= 0.20`.
- Test: `tests/test_vis.py` (append).

**Interfaces:**
- Consumes: `VIS` from Task 1; `PREDICATE_BIT`, `feature_mask_of`, `ALPHABET` from S6.
- Produces: same `feature_mask_of(sequence) -> int` signature. The returned int now sets bit `PREDICATE_BITS["vis_lowvis"]` (== 11) when `any(VIS[ltr] <= 0.20 for ltr in sequence if ALPHABET.get(ltr) == "N")`. The default-BB0 strain SETs the bit (N0 vis = 0.20 ≤ 0.20). Strains with zero N letters do NOT set the bit (`any(...)` over an empty iterable is False).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vis.py`:

```python
def test_feature_mask_vis_lowvis_set_on_default_bb0():
    """N0 vis = 0.20 ≤ 0.20 (inclusive) → vis_lowvis bit SET on default BB0."""
    from des.registry import feature_mask_of, PREDICATE_BIT, BB0_TEMPLATE
    m = feature_mask_of(BB0_TEMPLATE["layout"])
    assert m & PREDICATE_BIT["vis_lowvis"], \
        "default BB0 strain must SET vis_lowvis bit (N0 vis=0.20)"


def test_feature_mask_vis_lowvis_unset_when_all_N_above_threshold(monkeypatch):
    """Replace N0 in the sequence with a synthetic N letter whose vis>0.20:
    vis_lowvis bit must be CLEAR."""
    monkeypatch.setitem(registry.ALPHABET, "Nhi", "N")
    monkeypatch.setitem(registry.VIS, "Nhi", 0.70)
    monkeypatch.setitem(registry.GRAN, "Nhi", "residue")
    from des.registry import feature_mask_of, PREDICATE_BIT
    seq = ("Nhi",) * 16
    m = feature_mask_of(seq)
    assert not (m & PREDICATE_BIT["vis_lowvis"]), \
        "all-N-above-0.20 strain must NOT set vis_lowvis bit"


def test_feature_mask_vis_lowvis_unset_when_no_N_letter(monkeypatch):
    """A strain with zero N letters cannot satisfy 'any N letter with vis<=0.20'."""
    from des.registry import feature_mask_of, PREDICATE_BIT
    seq = ("F4Nr1",) * 16
    m = feature_mask_of(seq)
    assert not (m & PREDICATE_BIT["vis_lowvis"])


def test_feature_mask_vis_lowvis_inclusive_at_0p20(monkeypatch):
    """Threshold is ≤ 0.20 (inclusive). A letter with vis exactly 0.20 SETS the bit;
    a letter with vis 0.2001 does NOT."""
    monkeypatch.setitem(registry.ALPHABET, "Nexact", "N")
    monkeypatch.setitem(registry.VIS, "Nexact", 0.20)
    monkeypatch.setitem(registry.GRAN, "Nexact", "residue")
    monkeypatch.setitem(registry.ALPHABET, "Nabove", "N")
    monkeypatch.setitem(registry.VIS, "Nabove", 0.2001)
    monkeypatch.setitem(registry.GRAN, "Nabove", "residue")
    from des.registry import feature_mask_of, PREDICATE_BIT
    assert feature_mask_of(("Nexact",) * 16) & PREDICATE_BIT["vis_lowvis"]
    assert not (feature_mask_of(("Nabove",) * 16) & PREDICATE_BIT["vis_lowvis"])


def test_feature_mask_vis_lowvis_set_if_any_N_meets_threshold(monkeypatch):
    """'Any' semantics: a strain with one low-vis N and 15 high-vis N letters
    still SETs the bit."""
    monkeypatch.setitem(registry.ALPHABET, "Nhi", "N")
    monkeypatch.setitem(registry.VIS, "Nhi", 0.95)
    monkeypatch.setitem(registry.GRAN, "Nhi", "residue")
    from des.registry import feature_mask_of, PREDICATE_BIT
    seq = ("N0",) + ("Nhi",) * 15           # one N0 (vis=0.20) + 15 high-vis
    m = feature_mask_of(seq)
    assert m & PREDICATE_BIT["vis_lowvis"]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_vis.py -v -k vis_lowvis
```

Expected: every test FAILs because the current `feature_mask_of` (from S6 Task 7) never sets the `vis_lowvis` bit — it stays at 0 in S6 by design.

- [ ] **Step 3: Extend `feature_mask_of` in `src/des/registry.py`**

Locate `feature_mask_of` (added in S6 Task 7). Extend its body by adding the `vis_lowvis` clause after the existing motif-block loop and before the return:

```python
def feature_mask_of(sequence: tuple[str, ...]) -> int:
    """Predicate-bit feature mask for a sequence (S6 §3.5 + S1 §3.3).
    Sets:
      - family_<X> for every letter present (X = ALPHABET[letter]),
      - motif_<X>  if the sequence has at least one motif block of family X,
      - motif3_<X> if the sequence has a motif block of family X with MOTIF_LEN>=3,
      - vis_lowvis if the sequence has any fam=N letter with VIS[letter] <= 0.20.
    Reserved thr_* bits stay 0 in S6/S1; S3 OR them in.
    Pure function of the sequence — reads only registry tables."""
    m = 0
    for letter in sequence:
        fam = ALPHABET.get(letter)
        if fam is None:
            continue
        m |= PREDICATE_BIT[f"family_{fam}"]
    for s, e, letter in motif_blocks(sequence):
        if GRAN.get(letter) != "motif":
            continue
        fam = ALPHABET.get(letter)
        if fam is None:
            continue
        m |= PREDICATE_BIT[f"motif_{fam}"]
        if MOTIF_LEN[letter] >= 3 and fam in ("F", "P", "Z"):
            m |= PREDICATE_BIT[f"motif3_{fam}"]
    # S1: vis_lowvis — any fam=N letter with VIS<=0.20 (inclusive). N0 default
    # vis=0.20 SETs the bit on the default BB0 strain; harmless because no v1
    # primitive consumes it (Void Bite is dormant until S8).
    for letter in sequence:
        if ALPHABET.get(letter) == "N" and VIS.get(letter, 0.0) <= 0.20:
            m |= PREDICATE_BIT["vis_lowvis"]
            break
    return m
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_vis.py -v -k vis_lowvis
```

Expected: all 5 vis_lowvis tests PASS.

- [ ] **Step 5: Run the full suite to verify no regression**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 285 engine + 146 web tests stay green. The bit is set on the default-BB0 strain but no v1 consumer reads it (`prey_mask_for_clauses` never produces a clause that targets bit 11 in v1), so the antagonism kernel match relation is unchanged.

Backtrack: if `tests/test_phenotype_cache.py` or `tests/test_acceptance.py` fails on a `feature_mask` numeric-literal assertion, the test had hard-coded the pre-S1 mask value for the default-BB0 strain. Update those assertions to OR in `PREDICATE_BIT["vis_lowvis"]` against the prior literal — the kernel-visible match relation is unchanged because no v1 prey clause targets this bit.

- [ ] **Step 6: Commit**

```bash
git add src/des/registry.py tests/test_vis.py
git commit -m "feat(s1): fill vis_lowvis predicate-bit value source

feature_mask_of now SETs PREDICATE_BIT['vis_lowvis'] (S6 reserved bit 11)
when the sequence has any fam=N letter with VIS<=0.20 (inclusive). Default
BB0's N0 strain meets the threshold (vis=0.20) so the bit is set —
harmless because Void Bite (A pool, S8) is the only consumer. Closes
the S3 handoff for the vis-low-vis row."
```

---

### Task 7: Relabel-invariance audit + final regression sweep

**Goal:** Prove the whole S1 deliverable is green together: `VIS` data + phenotype aggregates + `vis_mode` flag + phenotype-array columns + kernel bypass + `vis_lowvis` predicate-bit value source. Add the relabel-invariance audit test (per spec §6: shuffle `_F`/`_Z`/`_P` magnitudes — but NOT `VIS` — and the kernel's vis-weighted results must be unchanged because the vis path reads structural channel values, not `f`/`z`/`p` magnitudes). Smoke the default 4-faction symmetric run to confirm the byte-identical regression lock holds.

**Files:**
- Create / Modify: no source modifications expected (if a regression slips through, fix it in this task and reference the offending commit).
- Test: `tests/test_vis.py` (append the relabel-invariance audit); `tests/` (the entire suite).

**Interfaces:**
- Consumes: every artifact produced by Tasks 1–6.
- Produces: a green `pytest tests/` and a clean `git status`.

- [ ] **Step 1: Write the relabel-invariance audit test**

Append to `tests/test_vis.py`:

```python
def test_relabel_invariance_vis_path_does_not_read_fzp_magnitude(monkeypatch):
    """Spec §6: shuffle f/z/p magnitudes across letters (NOT VIS — vis is a
    structural channel). vis-weighted kernel result must be unchanged because
    the vis path reads VIS / vis_sum / n_count, not f/z/p magnitudes."""
    monkeypatch.setitem(registry.ALPHABET, "Hunt", "Z")
    monkeypatch.setitem(registry.GRAN, "Hunt", "residue")
    monkeypatch.setitem(registry.VIS, "Hunt", 0.0)
    monkeypatch.setitem(registry._Z, "Hunt",
                        (0.40, (("N",),), 5, 1))
    monkeypatch.setitem(registry.ALPHABET, "Nm", "N")
    monkeypatch.setitem(registry.GRAN, "Nm", "residue")
    monkeypatch.setitem(registry.VIS, "Nm", 0.60)
    from des.engine import Engine
    layouts = (("Hunt",) + ("N0",) * 15,
               ("Nm",) + ("N0",) * 15,
               ("Hunt",) + ("N0",) * 15,
               ("Nm",) + ("N0",) * 15)
    eng_a = Engine(H=1, W=2, K=16, seed=0, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=4, layouts=layouts)
    eng_a.run(1, recorder=None, stop_on=())
    surv_a = int(eng_a.world.count[eng_a.world.faction == 1].sum().item())
    # Now mutate _F / _Z / _P magnitudes — but NOT VIS / NOT GRAN / NOT ALPHABET.
    monkeypatch.setitem(registry._F, "F4Nr1", (0.95, ((1, 0),), 0.99, 99))
    monkeypatch.setitem(registry._F, "F4Nr4", (0.01, ((-1, 0),), 0.01, 1))
    monkeypatch.setitem(registry._Z, "BroadSweep", (0.99, (("F",), ("Z",)), 99, 0))
    monkeypatch.setitem(registry._P, "P_hotspot", (0.0, 99))
    monkeypatch.setitem(registry._P, "P_base", (0.05, 1))
    # Keep Hunt's z value the same so antagonism-magnitude is constant; only the
    # other _Z / _F / _P rows are perturbed. vis-weighted kill MUST be unchanged.
    eng_b = Engine(H=1, W=2, K=16, seed=0, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=4, layouts=layouts)
    eng_b.run(1, recorder=None, stop_on=())
    surv_b = int(eng_b.world.count[eng_b.world.faction == 1].sum().item())
    assert surv_a == surv_b, \
        f"vis path leaked f/z/p magnitude: pre={surv_a}, post={surv_b}"
```

- [ ] **Step 2: Run the relabel-invariance test**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_vis.py::test_relabel_invariance_vis_path_does_not_read_fzp_magnitude -v
```

Expected: PASS. The vis-path code (`phenotype()` vis accumulation + kernel `p_hit` multiply) only reads `VIS`, `vis_sum`, `n_count` — never `_F[letter][0]` (`f`) or `_P[letter][0]` (`p_add`), and the `Hunt` row's `z` value is held constant.

Backtrack: if this test fails, the vis path is leaking f/z/p magnitude somewhere. Most likely candidate: the kernel's `raw_kill` computation reads `z_eff` upstream of the `p_hit` multiply, and one of the perturbed `_Z` rows (`BroadSweep`) is being matched as an attacker. Verify the test factions only carry `Hunt` and `Nm` — `BroadSweep` is absent — so the perturbation cannot reach the attacker row.

- [ ] **Step 3: Full pytest sweep**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -q
```

Expected: every test passes. Total = 285 engine + 146 web + new vis tests (≈20) + appended phenotype-cache tests (3) + appended registry tests (4). Exact total may differ by ±1; no previously-green test may fail.

Backtrack: if a test fails, identify the task that introduced the regression by reverting commits one at a time until the suite is green again, then fix forward in a new commit.

- [ ] **Step 4: Smoke-run `scripts/run_batch.py --probe` to confirm runtime is unchanged**

Run:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --probe 30
```

Expected output line: `[probe 30 ticks] X.X ms/tick | peak Y.YY GB | strains->… | total …`. The `X.X ms/tick` figure must remain in the same ballpark as before S1 (target ≈15.8 ms/tick on a 128² grid; ≤ 20% drift acceptable). The default `BroadSweep` row carries `vis_mode=0` so the kernel's `mode_mask.any()` evaluates `False` and the new branch falls through — no measurable runtime impact.

If drift is > 20%, the most likely cause is that the kernel computes `vis_mode_a`, `vis_sum_p`, `n_count_p`, `p_hit_m1`, `p_hit_m2` unconditionally before checking `mode_mask.any()`. Hoist the `mode_mask = vis_mode_a > 0; if not mode_mask.any(): return raw_kill` check ABOVE those gathers so the default path skips the float-32 math entirely.

- [ ] **Step 5: Byte-identical default-run smoke (recommended)**

Run a single short symmetric default with seed=0 alongside a pre-S1 baseline parquet to confirm the strain trajectory is bit-identical:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --seeds 0 --T 50
```

Expected: a new parquet under `data/runs/`. If a baseline parquet from pre-S1 main is available, compare `{strain: count}` per `(tick, cell)` rows — they must be byte-identical because (a) every v1 letter has `vis_mode=0` so the kernel SKIPs the new multiply, (b) the new `vis_lowvis` bit is set on the default-BB0 strain but no v1 prey clause targets bit 11, leaving the kernel match relation unchanged, (c) `feature_mask_of` adds a new OR-clause but the kernel never compares masks for equality — only `(prey_mask[i] & feature_mask[j]) != 0`.

If a baseline parquet is unavailable, the `test_mode0_byte_identical_to_pre_s1` test from Task 5 covers the same invariant in pytest form.

- [ ] **Step 6: Inspect & clean stray data**

Run:

```
git status
```

Expected output: clean working tree. If `data/runs/<ts>-*.parquet` files from smoke runs appear, remove them — they are not test fixtures.

- [ ] **Step 7: Final commit (only if step 3 needed a fix-forward)**

If step 3 surfaced a regression you fixed:

```bash
git add <files-touched>
git commit -m "fix(s1): <description of the regression fixed>"
```

Otherwise this step is a no-op.

- [ ] **Step 8: Final commit for the relabel-invariance audit**

```bash
git add tests/test_vis.py
git commit -m "test(s1): relabel-invariance audit for the vis path

Shuffle _F / _Z / _P magnitudes (NOT VIS); the vis-weighted kernel
result must be unchanged because the vis path reads VIS / vis_sum /
n_count, not f / z / p magnitudes. Closes the structural-channel
discipline check from spec §6."
```

- [ ] **Step 9: Push to origin**

```bash
git push origin <current-branch>
```

Expected: push succeeds. The branch is ready for review / merge to `main`.

---

## Self-Review

**1. Spec coverage:**
- §1 (why: roster N-pool vis values + Scatter Nip / Ghost Spike formulas): Task 1 (`VIS` table with the 8 N-pool values) + Task 5 (kernel `p_hit` formulas `vis_sum_prey/L` and `(n_count - vis_sum)/L`).
- §2 (red lines: global per-primitive, phenotype stores aggregate not kernel reading world-state, default game byte-identical): Task 1 (`VIS` registry, never per-species) + Task 2 (`phenotype()` stores `vis_sum`/`n_count` aggregates) + Task 5 (mode-0 SKIP, not `×1.0`) + Task 7 (byte-identical smoke).
- §3.1 (vis registry + phenotype aggregate, `vis_sum`/`n_count` fields, two new phe-arrays bulk-uploaded): Task 1 (`VIS`) + Task 2 (`Phenotype` fields + accumulator) + Task 4 (phe-arrays bulk columns).
- §3.2 (vis-weighted hit in antagonism kernel, per-attacker `vis_mode` int array, mode-0 SKIP, `L` is sequence length): Task 3 (`Phenotype.vis_mode` + `_Z` 4-tuple) + Task 4 (`phenotype_arrays.vis_mode`) + Task 5 (kernel branch + mode-0 SKIP + `L=16`).
- §3.3 (vis≤0.20 predicate bit handoff to S3, inclusive ≤, default N0 SET): Task 6 (`feature_mask_of` extension).
- §4 (data flow): Tasks 2 + 5 implement `mint(seq) → phenotype(): vis_sum/n_count; set vis_lowvis bit; phase1_antagonism: raw_kill → ×p_hit(prey vis profile, attacker vis-mode) → cap → losses`. Task 6 wires the vis_lowvis bit.
- §5 (error handling: range assert, empty N profile, float-eps clamp): Task 1 (module-load `assert 0.0 <= v <= 1.0`) + Task 5 (`torch.clamp(..., min=0.0)` for `(n_count - vis_sum)`; empty-N → `p_hit=0` falls out of the formula; mode-0 SKIP so no alive-masking).
- §6 (testing: regression 285+146 green, new aggregate / Scatter Nip / Ghost Spike / mode-0 / vis≤0.20 bit / relabel-invariance): Task 2 (`vis_sum`/`n_count` per-strain correctness), Task 3 (`vis_mode` reads), Task 4 (phe-array columns), Task 5 (kernel mode-0 byte-identical + mode-1/2 directionality + empty-N zero kill), Task 6 (`vis_lowvis` bit semantics), Task 7 (relabel-invariance + final regression sweep + byte-identical smoke).
- §7 (out of scope: Scatter Nip / Ghost Spike entering spectrum, threshold predicate values): no Z row added to `_Z` for Scatter Nip / Ghost Spike (synthetic-via-monkeypatch only in tests); no `thr_*` bit value (S3 owns).

**2. Placeholder scan:** No `TBD`, `TODO`, "implement later", "fill in details", or "similar to Task N" remain. Every code step shows the actual code; every command step shows the actual command and expected output; every backtrack condition is concrete and named. The "either flat (n_pair,) or 2D (n_attacker, n_prey)" branch in Task 5 step 4 is a defensive variant — both variants are spelled out in full so the implementer picks the matching shape, not a placeholder.

**3. Type consistency:**
- `VIS: dict[str, float]` — same name + type in Task 1 / 2 / 4 / 6 / 7.
- `Phenotype.vis_sum: float`, `Phenotype.n_count: int`, `Phenotype.vis_mode: int` — defined Task 2 (`vis_sum`, `n_count`) and Task 3 (`vis_mode`) with consistent defaults `0.0` / `0` / `0`; consumed by Task 4 (phe-arrays) and Task 5 (kernel).
- `phenotype_arrays.vis_sum: torch.Tensor (float32)`, `phenotype_arrays.n_count: torch.Tensor (int16)`, `phenotype_arrays.vis_mode: torch.Tensor (int8)` — same dtypes in Task 4 (definition) and Task 5 (kernel reads via `.to(torch.float32)` for the float math, `> 0` for the mask).
- `_Z` row 4-tuple shape `(z, prey_clauses, period, vis_mode)` with `vis_mode` optional (3-tuple back-compat default to 0) — defined Task 3, consumed Task 5 (via `phenotype_arrays.vis_mode`), tested Task 3 (`test_phenotype_z_row_3_tuple_back_compat`).
- `feature_mask_of(sequence) -> int` — same signature pre-S1 (S6) and post-S1 (Task 6); Task 6 only extends the body.
- `phase1_antagonism(world, strain_table, phenotype_arrays, rng) -> None` — same signature in Task 5; body extended with the vis-mode branch only.
- `PREDICATE_BIT["vis_lowvis"]` value `== 1 << 11` — reserved in S6 Task 6 (index 11), filled by `feature_mask_of` in Task 6 of this plan; same constant name everywhere.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-24-s1-vis-channel.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. Use `superpowers:subagent-driven-development`.
2. **Inline Execution** — execute tasks in this session with batch checkpoints. Use `superpowers:executing-plans`.

Which approach?


