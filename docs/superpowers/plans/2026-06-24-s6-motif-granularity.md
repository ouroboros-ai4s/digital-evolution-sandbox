# S6 — Motif Granularity (Cross-Cutting Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `gran` (residue/motif) registry property + derived `motif_blocks` decomposition + gran-matched equal-length spectrum pre-filter + atomic block-overwrite mutation core + on-demand `n_locked(chan)` readout + predicate-bit `feature_mask`/`prey_mask` encoding scheme — the cross-cutting foundation S2/S3/S7/S8 build on, with the default all-residue BB0 byte-identical to today.

**Architecture:** Six pieces, in order. (1) `GRAN`/`MOTIF_LEN` registry tables (default residue for every current letter; data-only). (2) Pure helper `motif_blocks(layout) -> tuple[(start, end, letter), ...]` that runs over the flat-16 `tuple[str]` layout — derived state, never stored. (3) `_spectrum_for(letter)` gains a gran-match + equal-length pre-filter (two predicate terms; residue-only path is byte-identical because every current letter is residue). (4) `_mutation_outcomes(seq, mutable, spectrum, blocks)` overwrites the **whole motif block** atomically when the chosen mutable slot lies inside one (residue path unchanged). (5) `n_locked(layout, chan) -> int` helper, recomputed on demand from `motif_blocks` over the backbone-locked positions; not stored on `Phenotype`/`StrainTable`. (6) Predicate-bit scheme: `PREDICATE_BITS = {name: bit}` enumeration with the 11 predicates S6 owns + 4 reserved positions for S1 (vis) and S3 (thresholds); `phenotype()` switches `feature_mask`/`prey_mask` to `OR` of predicate bits. The antagonism kernel match expression `(prey_mask[i] & feature_mask[j]) != 0` is unchanged; only what the bits *mean* changes.

**Tech Stack:** Python 3 (`D:\anaconda3\envs\basic\python.exe`), PyTorch 2.10+cu128, pyarrow, pytest. Windows host with `PYTHONPATH=src` discipline. Engine source `src/des/`.

## Global Constraints

- **No smuggled bias**: `gran` is a global per-primitive registry property `GRAN[letter]`, never `GRAN[species][letter]`. Strength still flows only through `_F`/`_Z`/`_P`. Granularity pairing is a structural "where can mutation go" rule, not a "who is strong" judgment.
- **Phenotype stays a pure function of the sequence**: motif block structure is *derived* from the layout via `motif_blocks(layout)`, not stored as mutable world-state, not stored on `Phenotype`, not stored on `StrainTable`.
- **Default game unchanged (regression lock)**: the default BB0 is all-residue (6 v1 letters; no repeated motif letters). Every motif mechanism added here is **dormant** until an asymmetric backbone places a motif primitive — byte-identical default runs are the regression lock. 285 engine + 146 web tests must stay green.
- **Equal-length motif↔motif mutation only**: an N-position motif mutates only to another N-position motif (`primitive-roster.md` line 277, design.md L322 `P_loopswap_lite`). Strain length is fixed within a lineage; the 16-position layout invariant is preserved.
- **gran pairing**: mutation happens only within the same gran — residue↔residue, motif↔motif (`primitive-roster.md` line 273). This is the mutation-core mechanism itself; no separate "gran law".
- **Antagonism kernel unchanged**: the match expression in `src/des/kernels/antagonism.py` stays `(prey_mask[i] & feature_mask[j]) != 0`. S6 only changes what each bit means — the kernel is not touched.
- **Predicate vocabulary fits int64**: assert at module import that `len(PREDICATE_BITS) <= 63` (highest bit reserved by sign). Today S6 owns 11 bits + reserves 4 (vis lowvis, 3 threshold) = 15. Vocabulary growth is bounded.
- **`n_locked` not stored**: `n_locked(layout, chan)` is a backbone-fixed species constant computed on demand from `motif_blocks` over the 16 locked positions. No field on `Phenotype`. No tensor in `phenotype_arrays`.
- **Out of scope (later specs)**: vis predicate values (S1), threshold predicate values (S3), `_F`/`_Z`/`_P` rows for the other 62 primitives (each later spec adds its own), `A`-pool gate consumption of `n_locked` (S8 — A is de-gated, `n_locked` is advisory only).

---

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `src/des/registry.py` | **Modify** | Add `GRAN`/`MOTIF_LEN` tables + `PREDICATE_BITS` + `motif_blocks(layout)` + `n_locked(layout, chan)` + predicate-bit helpers. Switch `phenotype()` to use predicate bits for `feature_mask`/`prey_mask`. Extend `_spectrum_for(letter)` with the gran-match + equal-length pre-filter. Extend `validate_bb0_layout` with the motif contiguity check. |
| `src/des/kernels/reproduction.py:34-48,134` | **Modify** | `_mutation_outcomes(seq, mutable, spectrum, blocks)` — add `blocks` parameter; when chosen mutable slot lies inside a motif block, overwrite the whole block atomically (equal-length guarantee preserves the 16-position layout). Update the call site at line 134 to pass `motif_blocks(seq)`. |
| `tests/test_registry.py` | **Modify** | Update the existing predicate-bit-affected assertions (`test_feature_mask_is_or_of_letter_bits`, `test_broadsweep_prey_targets_F_and_Z_families`) to read predicate-bit semantics. Append the new S6 test cases (gran property, motif_blocks, spectrum pre-filter, n_locked, predicate vocabulary). |
| `tests/test_motif.py` | **Create** | New focused tests for motif machinery: hand-built motif layouts, block overwrite outcomes, equal-length filter, antagonism-match invariance, relabel-invariance audit. Lives separately so the file boundary maps to S6 ownership. |
| `tests/test_reproduction.py` | **Modify (append)** | Append a residue-mutation regression that pins the existing all-residue mutation behavior byte-equivalently after the `_mutation_outcomes` signature change. |

**Naming contract (locked, used by every task):**

```python
# src/des/registry.py
GRAN: dict[str, str]                                     # letter -> "residue" | "motif"
MOTIF_LEN: dict[str, int]                                # motif letter -> span length (>= 2)
PREDICATE_BITS: dict[str, int]                           # predicate name -> bit index (0..62)
PREDICATE_BIT: dict[str, int]                            # predicate name -> 1 << bit index

def motif_blocks(layout: tuple[str, ...]) -> tuple[tuple[int, int, str], ...]
def n_locked(layout: tuple[str, ...], chan: str) -> int
def feature_mask_of(sequence: tuple[str, ...]) -> int
def prey_mask_for_clauses(prey_clauses: tuple[tuple[str, ...], ...]) -> int

# extended signatures
def _spectrum_for(letter: str) -> tuple[tuple[str, float], ...]   # now gran-matched + equal-len
def validate_bb0_layout(layout: tuple[str, ...]) -> None          # now also enforces motif contiguity

# src/des/kernels/reproduction.py
def _mutation_outcomes(
    seq: tuple[str, ...],
    mutable: tuple[bool, ...],
    spectrum: tuple[tuple[str, float], ...],
    blocks: tuple[tuple[int, int, str], ...],
) -> tuple[list[tuple[str, ...]], list[float]]
```

`motif_blocks` returns ascending-by-`start` blocks; residue letters appear as
singletons `(i, i+1, letter)`; motif runs collapse into one block of length
`MOTIF_LEN[letter]`. `n_locked(layout, "F" | "P" | "Z")` counts blocks whose
family equals `chan`, motif-or-residue counted as 1 block each (per spec §3.4
and `primitive-roster.md` OPEN-1 ②). `N` is never counted.

**Predicate vocabulary (spec §3.5, locked at module load):** S6 owns the
following 11 bits; S1 reserves 1 vis bit, S3 reserves 3 threshold bits.
Total 15, well under the int64 limit of 63.

| bit name | predicate | filled by |
| --- | --- | --- |
| `family_N` | `letter.family == "N"` | S6 |
| `family_F` | `letter.family == "F"` | S6 |
| `family_P` | `letter.family == "P"` | S6 |
| `family_Z` | `letter.family == "Z"` | S6 |
| `motif_F` | sequence contains a motif block of family F | S6 |
| `motif_P` | sequence contains a motif block of family P | S6 |
| `motif_Z` | sequence contains a motif block of family Z | S6 |
| `motif_N` | sequence contains a motif block of family N | S6 |
| `motif3_F` | sequence contains a motif block of family F with `MOTIF_LEN >= 3` | S6 |
| `motif3_P` | sequence contains a motif block of family P with `MOTIF_LEN >= 3` | S6 |
| `motif3_Z` | sequence contains a motif block of family Z with `MOTIF_LEN >= 3` | S6 |
| `vis_lowvis` | `gran=="residue" AND vis<=0.20` (Void Bite) | reserved → S1+S3 |
| `thr_crest` | `family=="F" AND f>=0.5` (Crest Bite) | reserved → S3 |
| `thr_hotspot` | `family=="P" AND p_add>=0.05` (Hotspot Snipe) | reserved → S3 |
| `thr_mirror` | `family=="Z" AND z<=0.45 AND \|prey\|>=2` (Mirror Fang) | reserved → S3 |

S6 reserves the names with a stable bit index (so adding the values later
is purely a bit-set, not a renumbering). The bit indices live in
`PREDICATE_BITS` as a single `dict[str, int]`; `assert max(PREDICATE_BITS.values()) < 63`
runs at import time.

---

### Task 1: `GRAN`/`MOTIF_LEN` registry tables (data-only)

**Goal:** Add the per-primitive `gran` and `MOTIF_LEN` tables to `src/des/registry.py`. Every existing letter (`N0`, `F4Nr1`, `F4Nr4`, `P_base`, `P_hotspot`, `BroadSweep`) gets `gran="residue"`; `MOTIF_LEN` is empty (no motif primitives exist yet — they arrive in later specs). Pure data-only change; no behavior change yet.

**Files:**
- Modify: `src/des/registry.py` (append `GRAN` and `MOTIF_LEN` constants after `ALPHABET` at line 18)
- Test: `tests/test_registry.py` (append two new tests)

**Interfaces:**
- Consumes: nothing new.
- Produces (importable from `des.registry`):
  - `GRAN: dict[str, str]` — every letter in `ALPHABET` mapped to `"residue"` or `"motif"`. v1: every letter is `"residue"`.
  - `MOTIF_LEN: dict[str, int]` — motif letter -> span length (`>= 2`). v1: empty.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_registry.py`:

```python
# ---------------------------------------------------------------------------
# S6 Task 1: GRAN / MOTIF_LEN tables
# ---------------------------------------------------------------------------

def test_gran_covers_every_alphabet_letter():
    """GRAN must have one entry per letter in ALPHABET, value in {residue, motif}."""
    from des.registry import GRAN, ALPHABET
    assert set(GRAN.keys()) == set(ALPHABET.keys())
    for letter, gran in GRAN.items():
        assert gran in ("residue", "motif"), f"{letter}: bad gran {gran!r}"


def test_v1_alphabet_is_all_residue_motif_len_empty():
    """v1 has no motif primitives yet — every letter is residue, MOTIF_LEN is empty."""
    from des.registry import GRAN, MOTIF_LEN
    for letter, gran in GRAN.items():
        assert gran == "residue", f"{letter}: v1 must be residue, got {gran!r}"
    assert MOTIF_LEN == {}, f"v1 has no motif primitives, got MOTIF_LEN={MOTIF_LEN!r}"
```

- [ ] **Step 2: Run the new tests to verify they fail**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_registry.py::test_gran_covers_every_alphabet_letter tests/test_registry.py::test_v1_alphabet_is_all_residue_motif_len_empty -v
```

Expected: both FAIL with `ImportError: cannot import name 'GRAN' from 'des.registry'`.

- [ ] **Step 3: Add `GRAN` and `MOTIF_LEN` to `src/des/registry.py`**

In `src/des/registry.py`, immediately after the `ALPHABET = {...}` block (around line 18, before `FEATURE_BIT = ...`), add:

```python
# Granularity per primitive (S6). residue = single position; motif = N consecutive
# positions of the SAME letter. Roster tags `gran` explicitly only for N0–N7; every
# other letter is residue by single-position occupancy. v1 alphabet is all-residue.
GRAN: dict[str, str] = {
    "N0":         "residue",
    "F4Nr1":      "residue",
    "F4Nr4":      "residue",
    "P_base":     "residue",
    "P_hotspot":  "residue",
    "BroadSweep": "residue",
}

# Span length per motif primitive. residue letters MUST NOT appear here.
# v1: empty (no motif primitives — each later spec adds its own rows).
MOTIF_LEN: dict[str, int] = {}
```

- [ ] **Step 4: Run the tests to verify they pass**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_registry.py::test_gran_covers_every_alphabet_letter tests/test_registry.py::test_v1_alphabet_is_all_residue_motif_len_empty -v
```

Expected: both PASS.

- [ ] **Step 5: Run the full registry test file to verify no regression**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_registry.py -v
```

Expected: every existing test in `tests/test_registry.py` still passes (no behavior change).

- [ ] **Step 6: Commit**

```
git add src/des/registry.py tests/test_registry.py
git commit -m "feat(s6): add GRAN/MOTIF_LEN registry tables (data-only)

Per-primitive granularity tag (residue|motif) and span-length table for
motif primitives. v1 alphabet is all-residue; MOTIF_LEN is empty (motif
primitives arrive in later specs). No behavior change."
```

---

### Task 2: `motif_blocks(layout)` derived helper

**Goal:** Add the pure decomposition helper that groups runs of the same `gran=="motif"` letter into `(start, end, letter)` blocks. Residue letters always come out as singletons. Default all-residue layouts therefore produce 16 singletons; this is the basis for byte-identical regression.

**Files:**
- Modify: `src/des/registry.py` (add `motif_blocks` function after `_spectrum_for`, before `phenotype`)
- Test: `tests/test_motif.py` (Create)

**Interfaces:**
- Consumes: `GRAN`, `MOTIF_LEN`, `ALPHABET` from Task 1.
- Produces: `motif_blocks(layout: tuple[str, ...]) -> tuple[tuple[int, int, str], ...]` — ascending-by-`start` blocks; residues are length-1 blocks; motif runs collapse into one length-`MOTIF_LEN[letter]` block. Letters not in `ALPHABET` (defensive: should never happen post-`validate_bb0_layout`) are treated as residues so the function is total.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_motif.py`:

```python
# tests/test_motif.py
"""S6 motif machinery: block decomposition, gran-matched mutation, n_locked,
predicate-bit feature/prey masks, antagonism-match invariance.

Default v1 alphabet is all-residue, so most assertions here build hand-crafted
layouts via monkeypatching GRAN/MOTIF_LEN to simulate a future motif primitive.
This is the only test file allowed to mutate registry tables — production code
never does."""
from __future__ import annotations
import pytest
from des import registry
from des.registry import motif_blocks


def test_all_residue_layout_yields_16_singletons():
    layout = ("N0",) * 16
    blocks = motif_blocks(layout)
    assert len(blocks) == 16
    for i, (s, e, ltr) in enumerate(blocks):
        assert (s, e, ltr) == (i, i + 1, "N0")


def test_default_bb0_layout_yields_16_singletons():
    """The default BB0 backbone is all-residue → motif_blocks must agree
    with the trivial decomposition."""
    blocks = motif_blocks(registry.BB0_TEMPLATE["layout"])
    assert len(blocks) == 16
    for i, (s, e, _) in enumerate(blocks):
        assert (s, e) == (i, i + 1)


def test_motif_block_groups_consecutive_repeated_motif_letter(monkeypatch):
    """Hand-craft a 16-position layout with a length-3 motif letter M3 and
    verify motif_blocks groups the 3 consecutive Ms into one block."""
    monkeypatch.setitem(registry.GRAN, "M3", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M3", 3)
    monkeypatch.setitem(registry.ALPHABET, "M3", "F")
    layout = ("N0", "M3", "M3", "M3", "N0", "BroadSweep", "N0", "P_base",
              "N0", "N0", "N0", "N0", "N0", "N0", "N0", "N0")
    blocks = motif_blocks(layout)
    # 1 N0 + 1 M3 (length 3) + 1 N0 + 1 BroadSweep + 1 N0 + 1 P_base + 9 N0 = 15
    assert len(blocks) == 15
    assert (1, 4, "M3") in blocks
    # all other positions are singletons
    for s, e, ltr in blocks:
        if ltr != "M3":
            assert e - s == 1


def test_two_separated_motif_blocks_of_same_letter_stay_separated(monkeypatch):
    """Two non-contiguous runs of the same motif letter are TWO blocks (the
    separator breaks the run)."""
    monkeypatch.setitem(registry.GRAN, "M2", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M2", 2)
    monkeypatch.setitem(registry.ALPHABET, "M2", "Z")
    layout = ("M2", "M2", "N0", "M2", "M2") + ("N0",) * 11
    blocks = motif_blocks(layout)
    m2_blocks = [b for b in blocks if b[2] == "M2"]
    assert len(m2_blocks) == 2
    assert (0, 2, "M2") in m2_blocks
    assert (3, 5, "M2") in m2_blocks
```

- [ ] **Step 2: Run the new tests to verify they fail**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_motif.py -v
```

Expected: every test FAILs with `ImportError: cannot import name 'motif_blocks' from 'des.registry'`.

- [ ] **Step 3: Implement `motif_blocks` in `src/des/registry.py`**

In `src/des/registry.py`, between `_spectrum_for` and `phenotype` (around line 54), insert:

```python
def motif_blocks(layout: tuple[str, ...]) -> tuple[tuple[int, int, str], ...]:
    """Decompose a flat-16 layout into `(start, end, letter)` blocks. Residue
    letters appear as singletons `(i, i+1, letter)`. Runs of the same
    `gran=="motif"` letter collapse into one block of length `MOTIF_LEN[letter]`.
    Pure function of the layout; reads only the registry tables. Default
    all-residue layouts yield 16 singletons (regression-lock invariant)."""
    blocks: list[tuple[int, int, str]] = []
    i = 0
    n = len(layout)
    while i < n:
        letter = layout[i]
        if GRAN.get(letter) == "motif":
            length = MOTIF_LEN[letter]
            blocks.append((i, i + length, letter))
            # Walk forward only while the same motif letter repeats.
            j = i + 1
            while j < i + length and j < n and layout[j] == letter:
                j += 1
            i = j
        else:
            blocks.append((i, i + 1, letter))
            i += 1
    return tuple(blocks)
```

- [ ] **Step 4: Run the tests to verify they pass**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_motif.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run the full suite to verify no regression**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: every previously-green test still passes; new motif tests pass.

- [ ] **Step 6: Commit**

```
git add src/des/registry.py tests/test_motif.py
git commit -m "feat(s6): add motif_blocks(layout) derived helper

Pure decomposition: runs of the same gran=motif letter collapse into one
block; residues appear as singletons. Default all-residue layouts yield
16 singletons — regression-lock invariant. Pure function of the layout
(reads GRAN/MOTIF_LEN), no world-state stored."
```

---

### Task 3: `_spectrum_for` gran-match + equal-length pre-filter

**Goal:** Extend `_spectrum_for(letter)` with the two-predicate pre-filter from spec §3.3: targets must share gran with `letter`, and motif targets must additionally match `MOTIF_LEN`. The residue-only path is byte-identical because every existing letter is residue and no motif letters exist yet in `ALPHABET` — `gran_match` is `True` for all current targets.

**Files:**
- Modify: `src/des/registry.py:45-53` (`_spectrum_for`)
- Test: `tests/test_motif.py` (append)

**Interfaces:**
- Consumes: `GRAN`, `MOTIF_LEN`, `ALPHABET`, `affinity` from prior tasks.
- Produces: same signature `_spectrum_for(letter: str) -> tuple[tuple[str, float], ...]`. Targets now restricted to `{t : t != letter AND GRAN[t] == GRAN[letter] AND (GRAN[letter] == "residue" OR MOTIF_LEN[t] == MOTIF_LEN[letter])}`. Renormalized to `Σq=1` across the surviving set.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_motif.py`:

```python
def test_spectrum_residue_path_byte_identical_to_legacy():
    """Default v1 alphabet is all-residue: the spectrum for every letter must
    survive the gran-match filter exactly as before (regression lock)."""
    from des.registry import _spectrum_for, ALPHABET
    # Reproduce the legacy formula directly to compare.
    from des.registry import affinity, ALPHABET as A
    def legacy(letter):
        src_fam = A[letter]
        weights = {t: affinity(src_fam, A[t]) for t in A if t != letter}
        tot = sum(weights.values())
        if tot == 0:
            return ()
        return tuple((t, w / tot) for t, w in sorted(weights.items()))
    for letter in ALPHABET:
        assert _spectrum_for(letter) == legacy(letter), \
            f"residue spectrum changed for {letter}"


def test_spectrum_motif_excludes_cross_gran_targets(monkeypatch):
    """A motif source letter must not produce residue targets in its spectrum."""
    monkeypatch.setitem(registry.GRAN, "M2", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M2", 2)
    monkeypatch.setitem(registry.ALPHABET, "M2", "F")
    monkeypatch.setitem(registry.GRAN, "M2b", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M2b", 2)
    monkeypatch.setitem(registry.ALPHABET, "M2b", "Z")
    monkeypatch.setitem(registry.GRAN, "M3", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M3", 3)
    monkeypatch.setitem(registry.ALPHABET, "M3", "F")
    spec = registry._spectrum_for("M2")
    targets = {t for t, _ in spec}
    # No residue letters in the M2 spectrum.
    assert "F4Nr1" not in targets and "P_base" not in targets and "N0" not in targets
    # M3 excluded by equal-length predicate (different MOTIF_LEN).
    assert "M3" not in targets
    # M2b survives (same gran, same length, different family).
    assert "M2b" in targets


def test_spectrum_motif_renormalizes_to_unit_sum(monkeypatch):
    monkeypatch.setitem(registry.GRAN, "M2", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M2", 2)
    monkeypatch.setitem(registry.ALPHABET, "M2", "F")
    monkeypatch.setitem(registry.GRAN, "M2b", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M2b", 2)
    monkeypatch.setitem(registry.ALPHABET, "M2b", "Z")
    monkeypatch.setitem(registry.GRAN, "M2c", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M2c", 2)
    monkeypatch.setitem(registry.ALPHABET, "M2c", "P")
    spec = registry._spectrum_for("M2")
    total = sum(q for _, q in spec)
    assert abs(total - 1.0) < 1e-12, f"spectrum did not renormalize: total={total}"


def test_spectrum_empty_when_no_compatible_target(monkeypatch):
    """If the gran-matched + equal-length filter leaves zero candidates,
    _spectrum_for must return ()."""
    monkeypatch.setitem(registry.GRAN, "M_lonely", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M_lonely", 7)
    monkeypatch.setitem(registry.ALPHABET, "M_lonely", "F")
    spec = registry._spectrum_for("M_lonely")
    assert spec == ()
```

- [ ] **Step 2: Run the new tests to verify they fail**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_motif.py -v -k spectrum
```

Expected: `test_spectrum_residue_path_byte_identical_to_legacy` PASSes (no change yet); the three motif-path tests FAIL because the current `_spectrum_for` does not filter by gran (it returns the residue + motif targets indiscriminately, so e.g. `M3` shows up in `M2`'s spectrum).

- [ ] **Step 3: Update `_spectrum_for` in `src/des/registry.py`**

Replace the body of `_spectrum_for` at lines 45–53 with the gran-matched version:

```python
def _spectrum_for(letter: str) -> tuple[tuple[str, float], ...]:
    """Family-distance spectrum filtered by gran-match + equal-length predicate
    (S6 §3.3). Targets must share gran with `letter`, and motif targets must
    additionally match MOTIF_LEN. q(target) ∝ affinity(family(letter), family(target));
    renormalized to Σq=1 across the surviving set. Pure function of the alphabet.
    Residue source + residue-only alphabet (v1 default) ≡ legacy behavior."""
    src_fam = ALPHABET[letter]
    src_gran = GRAN[letter]
    src_len = MOTIF_LEN.get(letter)
    survivors = {}
    for t in ALPHABET:
        if t == letter:
            continue
        if GRAN[t] != src_gran:
            continue
        if src_gran == "motif" and MOTIF_LEN[t] != src_len:
            continue
        survivors[t] = affinity(src_fam, ALPHABET[t])
    tot = sum(survivors.values())
    if tot == 0:
        return ()
    return tuple((t, w / tot) for t, w in sorted(survivors.items()))
```

- [ ] **Step 4: Run the motif tests to verify they pass**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_motif.py -v
```

Expected: every test in `tests/test_motif.py` PASSes — including `test_spectrum_residue_path_byte_identical_to_legacy` (the v1 alphabet is all-residue so the filter is a no-op).

- [ ] **Step 5: Run the full suite to verify no regression**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: all 285 engine + 146 web tests stay green; the spectrum-related tests in `tests/test_registry.py` (`test_dominant_p_is_position_independent`, `test_dominant_p_picks_higher_padd`) still pass byte-identically.

Backtrack: if `test_dominant_p_picks_higher_padd` fails, you probably forgot to import `GRAN` and `MOTIF_LEN` into the function scope or you broke the `tot == 0` empty-spectrum path. Recheck the body matches step 3 exactly.

- [ ] **Step 6: Commit**

```
git add src/des/registry.py tests/test_motif.py
git commit -m "feat(s6): gran-match + equal-length spectrum pre-filter

_spectrum_for now restricts targets to the same gran as the source, and
motif targets must additionally match MOTIF_LEN. Residue-only v1 alphabet
is byte-identical. Foundation for motif↔motif equal-length mutation."
```

---

### Task 4: `_mutation_outcomes` block-overwrite (gran-aware mutation core)

**Goal:** Extend `_mutation_outcomes` in `src/des/kernels/reproduction.py` so when the chosen mutable slot lies inside a motif block, the outcome overwrites the **whole block** atomically with the new (equal-length) motif letter. Residue path is byte-identical because each residue is a singleton block. Add a `blocks` parameter (the caller passes `motif_blocks(seq)`).

**Files:**
- Modify: `src/des/kernels/reproduction.py:34-48` (`_mutation_outcomes` body + signature) and line 134 (call site)
- Test: `tests/test_motif.py` (append) + `tests/test_reproduction.py` (append regression)

**Interfaces:**
- Consumes: `motif_blocks(layout)` from Task 2; `_spectrum_for(letter)` filter from Task 3.
- Produces: `_mutation_outcomes(seq, mutable, spectrum, blocks) -> (child_sequences, weights)` with `weights` summing to 1 across the full `mutable_slot x spectrum` product; each child sequence has length 16. Residue slot → singleton overwrite. Motif slot → whole-block overwrite with the equal-length target letter.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_motif.py`:

```python
def test_mutation_outcomes_residue_only_path_byte_identical(monkeypatch):
    """The legacy single-slot overwrite must still produce identical
    (children, weights) for an all-residue parent — regression lock."""
    from des.kernels.reproduction import _mutation_outcomes
    from des.registry import _spectrum_for, BB0_TEMPLATE, motif_blocks
    seq = BB0_TEMPLATE["layout"]
    mutable = BB0_TEMPLATE["mutable"]
    # use the BB0 dominant_p ("P_base") spectrum to get a real one
    spectrum = _spectrum_for("P_base")
    blocks = motif_blocks(seq)
    children, weights = _mutation_outcomes(seq, mutable, spectrum, blocks)
    # all-residue path: outcomes count == |slots| * |spectrum|
    n_slots = sum(mutable)
    assert len(children) == n_slots * len(spectrum)
    assert abs(sum(weights) - 1.0) < 1e-9
    # every child differs from seq in exactly ONE position (singleton overwrite)
    for child in children:
        diff = sum(1 for a, b in zip(seq, child) if a != b)
        assert diff in (0, 1)  # 0 = self-loop; 1 = single residue swap


def test_mutation_outcomes_motif_slot_overwrites_whole_block(monkeypatch):
    """Hand-craft a parent containing a length-3 motif and pick a mutable
    slot inside it. The outcome must overwrite all 3 positions of the
    motif with the target letter, not just one."""
    monkeypatch.setitem(registry.GRAN, "M3a", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M3a", 3)
    monkeypatch.setitem(registry.ALPHABET, "M3a", "F")
    monkeypatch.setitem(registry.GRAN, "M3b", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "M3b", 3)
    monkeypatch.setitem(registry.ALPHABET, "M3b", "Z")
    from des.kernels.reproduction import _mutation_outcomes
    from des.registry import motif_blocks
    # parent: M3a M3a M3a at indices 0,1,2; all 3 positions mutable
    seq = ("M3a", "M3a", "M3a") + ("N0",) * 13
    mutable = (True, True, True) + (False,) * 13
    spectrum = (("M3b", 1.0),)              # single target, equal length
    blocks = motif_blocks(seq)
    children, weights = _mutation_outcomes(seq, mutable, spectrum, blocks)
    # 3 mutable slots × 1 target = 3 outcomes, all identical (whole-block overwrite)
    assert len(children) == 3
    # every outcome replaces positions 0..2 with M3b
    expected = ("M3b", "M3b", "M3b") + ("N0",) * 13
    for child in children:
        assert child == expected
    # weights uniformly 1/3
    assert all(abs(w - 1 / 3) < 1e-9 for w in weights)


def test_mutation_outcomes_motif_outcome_preserves_layout_length():
    """Length-fixed invariant: every outcome layout MUST be exactly 16 positions."""
    import des.registry as reg
    reg.GRAN["M2x"] = "motif"; reg.MOTIF_LEN["M2x"] = 2; reg.ALPHABET["M2x"] = "F"
    reg.GRAN["M2y"] = "motif"; reg.MOTIF_LEN["M2y"] = 2; reg.ALPHABET["M2y"] = "Z"
    try:
        from des.kernels.reproduction import _mutation_outcomes
        from des.registry import motif_blocks
        seq = ("M2x", "M2x") + ("N0",) * 14
        mutable = (True, True) + (False,) * 14
        spectrum = (("M2y", 1.0),)
        children, _ = _mutation_outcomes(seq, mutable, spectrum, motif_blocks(seq))
        for child in children:
            assert len(child) == 16
    finally:
        del reg.GRAN["M2x"]; del reg.MOTIF_LEN["M2x"]; del reg.ALPHABET["M2x"]
        del reg.GRAN["M2y"]; del reg.MOTIF_LEN["M2y"]; del reg.ALPHABET["M2y"]
```

Append to `tests/test_reproduction.py`:

```python
def test_mutation_outcomes_signature_takes_blocks():
    """Regression: post-S6 _mutation_outcomes accepts a 4th positional `blocks` arg.
    Calling with the legacy 3-arg form must raise."""
    from des.kernels.reproduction import _mutation_outcomes
    from des.registry import _spectrum_for, BB0_TEMPLATE
    seq = BB0_TEMPLATE["layout"]
    mutable = BB0_TEMPLATE["mutable"]
    spectrum = _spectrum_for("P_base")
    with pytest.raises(TypeError):
        _mutation_outcomes(seq, mutable, spectrum)  # missing blocks arg
```

(If `pytest` isn't already imported at the top of `tests/test_reproduction.py`, add `import pytest` at the top.)

- [ ] **Step 2: Run the new tests to verify they fail**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_motif.py tests/test_reproduction.py::test_mutation_outcomes_signature_takes_blocks -v
```

Expected: the three new motif tests FAIL (current `_mutation_outcomes` has 3 args, not 4; and even with `blocks` added, the legacy body overwrites only one position so the motif outcome doesn't span 3 positions). `test_mutation_outcomes_signature_takes_blocks` FAILs because the current 3-arg signature still accepts the legacy call.

- [ ] **Step 3: Update `_mutation_outcomes` in `src/des/kernels/reproduction.py`**

Replace the body of `_mutation_outcomes` at lines 34–48 with:

```python
def _mutation_outcomes(seq, mutable, spectrum, blocks):
    """Per-parent mutation categorical (design L246: uniform mutable slot x spectrum
    letter). Returns (child_sequences, weights) over the full slot x spectrum product;
    weights sum to 1. Self-loops (letter == current) yield child == parent. Pure fn
    of the sequence + its spectrum + its motif-block decomposition (S6).
    Residue slot → singleton overwrite. Motif slot → whole-block overwrite with the
    equal-length target letter (block boundaries come from `blocks`)."""
    slot_idx = [i for i, ok in enumerate(mutable) if ok]
    if not slot_idx or not spectrum:
        return [], []
    # index -> (start, end) of the block covering position i.
    # residue letters are length-1 blocks; this map is total over 0..len(seq)-1.
    cover: dict[int, tuple[int, int]] = {}
    for s, e, _ in blocks:
        for k in range(s, e):
            cover[k] = (s, e)
    children, weights = [], []
    for s in slot_idx:                      # ascending: canonical order
        for letter, q in spectrum:          # spectrum already sorted in _spectrum_for
            start, end = cover[s]
            new = list(seq)
            for k in range(start, end):      # overwrite the whole covering block
                new[k] = letter
            children.append(tuple(new))
            weights.append(q / len(slot_idx))
    return children, weights
```

Then update the call site at line 134 (inside `phase2_reproduce`). Replace:

```python
            children, weights = _mutation_outcomes(seq, BB0_TEMPLATE["mutable"], spectrum)
```

with:

```python
            children, weights = _mutation_outcomes(
                seq, BB0_TEMPLATE["mutable"], spectrum, motif_blocks(seq))
```

And add `motif_blocks` to the existing `from des.registry import BB0_TEMPLATE` line near the top of the file (line 5):

```python
from des.registry import BB0_TEMPLATE, motif_blocks
```

- [ ] **Step 4: Run the tests to verify they pass**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_motif.py tests/test_reproduction.py -v
```

Expected: all motif tests PASS; all reproduction tests (existing + new signature test) PASS.

- [ ] **Step 5: Run the full suite to verify no regression**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 285 engine + 146 web tests stay green. The residue-only path produces byte-identical outcomes because every block is a singleton `(i, i+1, letter)` and `range(i, i+1) = [i]` overwrites exactly one position — identical to the legacy `new[s] = letter`.

Backtrack: if you see a reproduction regression, the most likely cause is that you forgot to recompute `motif_blocks(seq)` per parent in the loop (line 134 call site). The `blocks` argument must be derived from `seq` (the parent's sequence), not a stale value.

- [ ] **Step 6: Commit**

```
git add src/des/kernels/reproduction.py tests/test_motif.py tests/test_reproduction.py
git commit -m "feat(s6): _mutation_outcomes overwrites whole motif block atomically

When a mutable slot lies inside a motif block, the outcome overwrites all
positions of the block with the equal-length target letter. Residue slots
remain singleton overwrites (byte-identical default path). New 4th arg
`blocks` is motif_blocks(seq); spectrum is already gran-matched + equal-len."
```

---

### Task 5: `n_locked(layout, chan)` on-demand readout

**Goal:** Add the `n_locked` helper that counts locked-position blocks per channel `F`/`P`/`Z` (motif or residue counted as 1 block each, `N` never counted). Recomputed on demand from `motif_blocks` over the 16 backbone-locked positions; not stored on `Phenotype`, not in `phenotype_arrays`. Spec §3.4: this is advisory only (S8 retired the A-pool gate that used to consume it); it remains as a structural readout for the relabel-invariance audit and a future asymmetric-backbone role system.

**Files:**
- Modify: `src/des/registry.py` (add `n_locked` after `motif_blocks`)
- Test: `tests/test_motif.py` (append)

**Interfaces:**
- Consumes: `motif_blocks`, `_LOCKED`, `ALPHABET`, `FAMILY_RANK` (only `F`/`P`/`Z`/`N` keys are consulted indirectly via `ALPHABET[letter]`).
- Produces: `n_locked(layout: tuple[str, ...], chan: str) -> int` where `chan in {"F", "P", "Z"}`. Counts blocks whose first letter's family equals `chan` AND whose entire span lies within `_LOCKED.keys()`. (A motif spanning a locked + non-locked range does not count — the spec is "blocks over the backbone-locked positions".) For default BB0 returns `F:1`, `P:1`, `Z:1`, `N:0` (default has 3 locked F/P/Z and the rest backbone N0; N is unsupported → raise on `chan=="N"`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_motif.py`:

```python
def test_n_locked_default_bb0_FPZ_counts_1_each():
    """Default BB0 has F4Nr4 at index 1, BroadSweep at index 5, P_base at index 7
    — three residue letters at locked positions → F:1 P:1 Z:1."""
    from des.registry import n_locked, BB0_TEMPLATE
    layout = BB0_TEMPLATE["layout"]
    assert n_locked(layout, "F") == 1
    assert n_locked(layout, "P") == 1
    assert n_locked(layout, "Z") == 1


def test_n_locked_rejects_N_channel():
    """N never counts (spec §3.4)."""
    from des.registry import n_locked, BB0_TEMPLATE
    with pytest.raises(ValueError):
        n_locked(BB0_TEMPLATE["layout"], "N")


def test_n_locked_rejects_unknown_channel():
    from des.registry import n_locked, BB0_TEMPLATE
    with pytest.raises(ValueError):
        n_locked(BB0_TEMPLATE["layout"], "X")


def test_n_locked_counts_motif_block_as_one(monkeypatch):
    """A locked motif of family F (length 3) counts as 1 F-block, not 3."""
    monkeypatch.setitem(registry.GRAN, "MF3", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "MF3", 3)
    monkeypatch.setitem(registry.ALPHABET, "MF3", "F")
    # Patch _LOCKED to require positions 0,1,2 to be the MF3 motif so the block
    # lies entirely inside locked positions. Original _LOCKED is restored by
    # monkeypatch's tearDown.
    monkeypatch.setattr(registry, "_LOCKED", {0: "MF3", 1: "MF3", 2: "MF3",
                                              5: "BroadSweep", 7: "P_base"})
    layout = ("MF3", "MF3", "MF3", "N0", "N0", "BroadSweep",
              "N0", "P_base") + ("N0",) * 8
    assert registry.n_locked(layout, "F") == 1
    assert registry.n_locked(layout, "P") == 1
    assert registry.n_locked(layout, "Z") == 1


def test_n_locked_excludes_block_partially_outside_locked_set(monkeypatch):
    """A motif block that straddles a locked position and a non-locked position
    is NOT a locked block — n_locked counts only blocks that lie entirely
    inside _LOCKED."""
    monkeypatch.setitem(registry.GRAN, "MF2", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "MF2", 2)
    monkeypatch.setitem(registry.ALPHABET, "MF2", "F")
    # _LOCKED only contains position 1 (not 2). The MF2 block spans 1..3 → not all locked.
    monkeypatch.setattr(registry, "_LOCKED", {1: "MF2", 5: "BroadSweep", 7: "P_base"})
    layout = ("N0", "MF2", "MF2", "N0", "N0", "BroadSweep",
              "N0", "P_base") + ("N0",) * 8
    # The single MF2 block (1, 3, 'MF2') is not fully inside _LOCKED={1,5,7}
    # → it does NOT contribute to n_locked("F"); the count is 0 for F.
    assert registry.n_locked(layout, "F") == 0
    assert registry.n_locked(layout, "Z") == 1
    assert registry.n_locked(layout, "P") == 1
```

(If `pytest` isn't already imported at the top of `tests/test_motif.py`, add `import pytest` at the top.)

- [ ] **Step 2: Run the new tests to verify they fail**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_motif.py -v -k n_locked
```

Expected: every test FAILs with `ImportError: cannot import name 'n_locked' from 'des.registry'`.

- [ ] **Step 3: Implement `n_locked` in `src/des/registry.py`**

In `src/des/registry.py`, immediately after `motif_blocks` (and before `phenotype`), insert:

```python
def n_locked(layout: tuple[str, ...], chan: str) -> int:
    """Count locked-position blocks whose family equals `chan`. Motif and
    residue blocks each count as 1 (per primitive-roster.md OPEN-1 ②).
    Only blocks whose entire span lies inside `_LOCKED.keys()` are counted —
    a motif straddling locked and non-locked positions is excluded.
    `chan` must be one of {"F", "P", "Z"}; "N" is never counted (spec §3.4)."""
    if chan not in ("F", "P", "Z"):
        raise ValueError(
            f"n_locked: chan must be one of F/P/Z, got {chan!r} (N never counts)")
    locked_positions = set(_LOCKED.keys())
    count = 0
    for s, e, letter in motif_blocks(layout):
        if not all(k in locked_positions for k in range(s, e)):
            continue
        if ALPHABET.get(letter) == chan:
            count += 1
    return count
```

- [ ] **Step 4: Run the tests to verify they pass**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_motif.py -v -k n_locked
```

Expected: all 5 `n_locked` tests PASS.

- [ ] **Step 5: Run the full suite to verify no regression**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 285 engine + 146 web tests stay green; new motif tests pass.

- [ ] **Step 6: Commit**

```
git add src/des/registry.py tests/test_motif.py
git commit -m "feat(s6): add n_locked(layout, chan) on-demand readout

Counts locked-position blocks per channel F/P/Z; motif and residue each
count as 1; blocks straddling locked + non-locked positions excluded.
Not stored on Phenotype/StrainTable — recomputed on demand. Advisory only
(S8 retired the A-pool gate that used to consume this); kept for the
relabel-invariance audit and the future asymmetric-backbone role system."
```

---

### Task 6: `PREDICATE_BITS` vocabulary + import-time int64 assertion

**Goal:** Define the predicate-bit vocabulary as a stable `dict[str, int]` (predicate name -> bit index 0..62) plus a derived `PREDICATE_BIT` (name -> 1 << index) for cheap OR-ing. Assert at module import that the highest bit fits int64. S6 owns the 11 family/motif/motif3 bits + reserves 4 names for S1/S3 with stable indices.

**Files:**
- Modify: `src/des/registry.py` (add the dicts after `MOTIF_LEN`, before `FEATURE_BIT`)
- Test: `tests/test_motif.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `PREDICATE_BITS: dict[str, int]` — name -> bit index, bits `0..14` assigned for the 15 names below.
  - `PREDICATE_BIT: dict[str, int]` — name -> `1 << PREDICATE_BITS[name]`.
  - Module-level assertion: `assert max(PREDICATE_BITS.values()) < 63, "predicate vocabulary overflows int64"`.
  - The reserved names `vis_lowvis`, `thr_crest`, `thr_hotspot`, `thr_mirror` are present with stable indices but no semantic body yet (S1/S3 fill them).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_motif.py`:

```python
def test_predicate_bits_present_and_distinct():
    """11 S6 predicates + 4 reserved (S1/S3) = 15 names; bit indices distinct."""
    from des.registry import PREDICATE_BITS
    expected_names = {
        "family_N", "family_F", "family_P", "family_Z",
        "motif_F", "motif_P", "motif_Z", "motif_N",
        "motif3_F", "motif3_P", "motif3_Z",
        "vis_lowvis", "thr_crest", "thr_hotspot", "thr_mirror",
    }
    assert set(PREDICATE_BITS.keys()) == expected_names
    indices = list(PREDICATE_BITS.values())
    assert len(indices) == len(set(indices)), "duplicate bit indices"
    for idx in indices:
        assert 0 <= idx < 63


def test_predicate_bit_is_shift_of_predicate_bits():
    from des.registry import PREDICATE_BITS, PREDICATE_BIT
    for name, idx in PREDICATE_BITS.items():
        assert PREDICATE_BIT[name] == 1 << idx


def test_predicate_vocabulary_fits_int64():
    """Module-level assertion: highest bit must fit signed int64."""
    from des.registry import PREDICATE_BITS
    assert max(PREDICATE_BITS.values()) < 63
```

- [ ] **Step 2: Run the new tests to verify they fail**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_motif.py -v -k predicate
```

Expected: all 3 FAIL with `ImportError: cannot import name 'PREDICATE_BITS' from 'des.registry'`.

- [ ] **Step 3: Add `PREDICATE_BITS` and `PREDICATE_BIT` to `src/des/registry.py`**

In `src/des/registry.py`, immediately after the `MOTIF_LEN` block (Task 1), add:

```python
# Predicate-bit vocabulary (S6 §3.5). Each bit = a structural predicate, not a
# letter. Stable indices: S1/S3 will populate the reserved names without
# renumbering. Total 15 of the 63 available signed-int64 bits.
PREDICATE_BITS: dict[str, int] = {
    # family-of-letter (4 bits): set per-letter at mint.
    "family_N":     0,
    "family_F":     1,
    "family_P":     2,
    "family_Z":     3,
    # sequence has at least one motif block of family fam (4 bits): set per-strain at mint.
    "motif_F":      4,
    "motif_P":      5,
    "motif_Z":      6,
    "motif_N":      7,
    # sequence has at least one motif block of family fam with MOTIF_LEN >= 3 (3 bits).
    "motif3_F":     8,
    "motif3_P":     9,
    "motif3_Z":    10,
    # Reserved (S1 / S3 fill the values; bit indices stable so adding the values
    # later is a bit-set, not a renumbering).
    "vis_lowvis":  11,   # gran=='residue' AND vis<=0.20    (S1 fills vis bit, S3 wires)
    "thr_crest":   12,   # family=='F' AND f>=0.5            (S3)
    "thr_hotspot": 13,   # family=='P' AND p_add>=0.05       (S3)
    "thr_mirror":  14,   # family=='Z' AND z<=0.45 AND |prey|>=2  (S3)
}
assert max(PREDICATE_BITS.values()) < 63, \
    "predicate vocabulary overflows int64; halt before phenotype() is built"

PREDICATE_BIT: dict[str, int] = {name: 1 << idx for name, idx in PREDICATE_BITS.items()}
```

- [ ] **Step 4: Run the tests to verify they pass**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_motif.py -v -k predicate
```

Expected: all 3 PASS.

- [ ] **Step 5: Run the full suite to verify no regression**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 285 engine + 146 web tests stay green. Adding the `PREDICATE_BITS` dict is a pure data addition; nothing reads it yet (Task 7 wires it in).

- [ ] **Step 6: Commit**

```
git add src/des/registry.py tests/test_motif.py
git commit -m "feat(s6): add PREDICATE_BITS vocabulary (15 names, int64-safe)

11 S6 predicates (family/motif/motif3) + 4 reserved for S1/S3. Stable bit
indices so later specs add values without renumbering. Import-time assert
keeps max bit < 63 (signed int64 safe). Pure data addition; phenotype()
wiring lands in Task 7."
```

---

### Task 7: Predicate-bit `feature_mask`/`prey_mask` helpers + `phenotype()` rewiring

**Goal:** Switch `phenotype()` from per-letter `FEATURE_BIT` masks to predicate-bit masks. Add two pure helpers, `feature_mask_of(sequence)` and `prey_mask_for_clauses(prey_clauses)`, that compute predicate-bit ORs. Update the per-letter `_Z` table so each `Z` row carries a tuple of prey-family clauses (today just `("F",)` or `("Z",)` etc.). The antagonism kernel match expression `(prey_mask[i] & feature_mask[j]) != 0` is **unchanged** — only what each bit means changes.

This task is the riskiest because it rewires the meaning of bits that other tests already assert. We therefore update the affected `tests/test_registry.py` assertions in the same task to read predicate-bit semantics, and add a "match-invariance on a known case" test in `tests/test_motif.py` to prove the antagonism kernel still picks the same pairs.

**Files:**
- Modify: `src/des/registry.py` (add helpers; rewrite the `feature_mask`/`prey_mask` accumulation block inside `phenotype()`; reshape `_Z` rows' prey field from `("F", "Z")`-style to `(("F",), ("Z",))`-style clause tuples — see Step 3)
- Modify: `tests/test_registry.py` (rewrite `test_feature_mask_is_or_of_letter_bits` and `test_broadsweep_prey_targets_F_and_Z_families` to predicate-bit semantics)
- Test: `tests/test_motif.py` (append predicate-mask + antagonism-match invariance tests)

**Interfaces:**
- Consumes: `PREDICATE_BITS`, `PREDICATE_BIT`, `GRAN`, `MOTIF_LEN`, `ALPHABET`, `motif_blocks` from Tasks 1–6.
- Produces:
  - `feature_mask_of(sequence: tuple[str, ...]) -> int` — OR of every predicate bit the sequence satisfies. Sets `family_<X>` for every letter present; sets `motif_<F|P|Z|N>` if the sequence has a motif block of that family; sets `motif3_<F|P|Z>` if the sequence has a motif block of family X with `MOTIF_LEN >= 3`. Reserved bits (`vis_lowvis`, `thr_*`) stay 0 in S6 (S1/S3 will OR them in).
  - `prey_mask_for_clauses(prey_clauses: tuple[tuple[str, ...], ...]) -> int` — OR over clauses; each clause `("F",)` or `("F", "motif")` or `("F", "motif", "len>=3")` translates to the matching `family_*`/`motif_*`/`motif3_*` bit. v1 every clause is a single-element family tuple, so the existing semantics are preserved bit-for-bit on the kernel match.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_motif.py`:

```python
def test_feature_mask_of_sets_family_bit_per_letter():
    """A v1 all-residue sequence sets only family_* bits + no motif_* / motif3_* bits."""
    from des.registry import feature_mask_of, PREDICATE_BIT
    seq = ("F4Nr1", "N0", "BroadSweep")
    m = feature_mask_of(seq)
    assert m & PREDICATE_BIT["family_F"]
    assert m & PREDICATE_BIT["family_N"]
    assert m & PREDICATE_BIT["family_Z"]
    # no P letter → no family_P bit set
    assert not (m & PREDICATE_BIT["family_P"])
    # no motif blocks → motif_* / motif3_* all clear
    for k in ("motif_F", "motif_P", "motif_Z", "motif_N",
              "motif3_F", "motif3_P", "motif3_Z"):
        assert not (m & PREDICATE_BIT[k]), f"unexpected {k} bit set on residue-only seq"


def test_feature_mask_of_sets_motif_and_motif3_bits(monkeypatch):
    """A length-3 motif of family F sets family_F + motif_F + motif3_F."""
    monkeypatch.setitem(registry.GRAN, "MF3", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "MF3", 3)
    monkeypatch.setitem(registry.ALPHABET, "MF3", "F")
    seq = ("MF3", "MF3", "MF3", "N0", "N0", "BroadSweep", "N0", "P_base") + ("N0",) * 8
    from des.registry import feature_mask_of, PREDICATE_BIT
    m = feature_mask_of(seq)
    assert m & PREDICATE_BIT["family_F"]
    assert m & PREDICATE_BIT["motif_F"]
    assert m & PREDICATE_BIT["motif3_F"]


def test_feature_mask_of_length2_motif_no_motif3_bit(monkeypatch):
    """A length-2 motif of family F sets motif_F but NOT motif3_F."""
    monkeypatch.setitem(registry.GRAN, "MF2", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "MF2", 2)
    monkeypatch.setitem(registry.ALPHABET, "MF2", "F")
    seq = ("MF2", "MF2") + ("N0",) * 14
    from des.registry import feature_mask_of, PREDICATE_BIT
    m = feature_mask_of(seq)
    assert m & PREDICATE_BIT["motif_F"]
    assert not (m & PREDICATE_BIT["motif3_F"])


def test_prey_mask_for_clauses_family_only_singletons():
    """v1 prey clauses are single-element family tuples; prey_mask = OR of family_* bits."""
    from des.registry import prey_mask_for_clauses, PREDICATE_BIT
    pm = prey_mask_for_clauses((("F",), ("Z",)))
    assert pm == (PREDICATE_BIT["family_F"] | PREDICATE_BIT["family_Z"])


def test_prey_mask_for_clauses_motif_clause():
    """A clause ('F', 'motif') targets motif_F bit only, not family_F."""
    from des.registry import prey_mask_for_clauses, PREDICATE_BIT
    pm = prey_mask_for_clauses((("F", "motif"),))
    assert pm == PREDICATE_BIT["motif_F"]


def test_prey_mask_for_clauses_motif3_clause():
    """A clause ('Z', 'motif', 'len>=3') targets motif3_Z bit only."""
    from des.registry import prey_mask_for_clauses, PREDICATE_BIT
    pm = prey_mask_for_clauses((("Z", "motif", "len>=3"),))
    assert pm == PREDICATE_BIT["motif3_Z"]


def test_antagonism_match_invariant_under_predicate_rewire():
    """The kernel match expression (prey_mask[i] & feature_mask[j]) != 0 must
    still pick the same (attacker, prey) pairs on the v1 alphabet — only what
    each bit means changes, not the match outcome."""
    from des.registry import phenotype
    # BroadSweep preys on F-family and Z-family. Build phenotypes for an
    # F-only prey, Z-only prey, P-only prey, N-only prey, and BroadSweep itself.
    p_bs   = phenotype(("BroadSweep",))
    p_f    = phenotype(("F4Nr1",))
    p_z    = phenotype(("BroadSweep",))
    p_p    = phenotype(("P_base",))
    p_n    = phenotype(("N0",))
    # BroadSweep attacks F-prey and Z-prey, not P, not N
    assert (p_bs.prey_mask & p_f.feature_mask) != 0
    assert (p_bs.prey_mask & p_z.feature_mask) != 0
    assert (p_bs.prey_mask & p_p.feature_mask) == 0
    assert (p_bs.prey_mask & p_n.feature_mask) == 0
```

Rewrite the existing assertions in `tests/test_registry.py`. Replace
`test_feature_mask_is_or_of_letter_bits` (lines 32–34) with:

```python
def test_feature_mask_is_or_of_family_predicate_bits():
    """Post-S6: feature_mask is OR of family_<X> predicate bits, not letter bits."""
    from des.registry import PREDICATE_BIT
    p = phenotype(("F4Nr1", "BroadSweep"))
    assert p.feature_mask & PREDICATE_BIT["family_F"]
    assert p.feature_mask & PREDICATE_BIT["family_Z"]
    # no N letter, no P letter
    assert not (p.feature_mask & PREDICATE_BIT["family_N"])
    assert not (p.feature_mask & PREDICATE_BIT["family_P"])
```

And replace `test_broadsweep_prey_targets_F_and_Z_families` (lines 36–42) with:

```python
def test_broadsweep_prey_targets_F_and_Z_families():
    """Post-S6: prey_mask is OR of family_<X> predicate bits selected by clauses."""
    from des.registry import PREDICATE_BIT
    p = phenotype(("BroadSweep",))
    assert p.prey_mask & PREDICATE_BIT["family_F"]
    assert p.prey_mask & PREDICATE_BIT["family_Z"]
    # BroadSweep's clauses select F and Z families only.
    assert not (p.prey_mask & PREDICATE_BIT["family_P"])
    assert not (p.prey_mask & PREDICATE_BIT["family_N"])
```

- [ ] **Step 2: Run the new tests to verify they fail**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_motif.py tests/test_registry.py -v
```

Expected: every new motif test FAILs with `ImportError: cannot import name 'feature_mask_of' from 'des.registry'`. The two rewritten tests in `test_registry.py` also FAIL because the current `phenotype()` still uses `FEATURE_BIT` (per-letter), not `PREDICATE_BIT`.

- [ ] **Step 3: Reshape `_Z` rows + add helpers + rewire `phenotype()` in `src/des/registry.py`**

First, reshape `_Z` so each row's prey field is a tuple of clause-tuples. Replace lines 31–33 (`_Z = { ... }`) with:

```python
_Z = {    # name -> (z, prey_clauses, period)
    # prey_clauses: tuple of clause-tuples. Each clause selects ONE predicate
    # bit; the prey_mask is the OR over clauses. v1 clauses are single-element
    # family tuples → identical kernel-match outcomes to the pre-S6 family code.
    # Future motif-specialist Z rows will use multi-element clauses like
    # ("F", "motif") or ("Z", "motif", "len>=3").
    "BroadSweep": (0.40, (("F",), ("Z",)), 5),
}
```

Then, after `_spectrum_for` and before `motif_blocks` (i.e. roughly at line 55), add:

```python
def feature_mask_of(sequence: tuple[str, ...]) -> int:
    """Predicate-bit feature mask for a sequence (S6 §3.5). Sets:
      - family_<X> for every letter present (X = ALPHABET[letter]),
      - motif_<X> if the sequence has at least one motif block of family X,
      - motif3_<X> if the sequence has a motif block of family X with MOTIF_LEN>=3.
    Reserved bits (vis_lowvis, thr_*) stay 0 in S6; S1/S3 OR them in.
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
    return m


def prey_mask_for_clauses(prey_clauses: tuple[tuple[str, ...], ...]) -> int:
    """Predicate-bit prey mask for a Z row's clause list (S6 §3.5).
    Each clause is a tuple whose first element is the family ('F'|'P'|'Z'|'N')
    and whose optional further elements specialize the predicate:
      ('F',)                → family_F bit
      ('F', 'motif')        → motif_F bit
      ('F', 'motif','len>=3') → motif3_F bit
    OR the selected bits to form prey_mask. Pure function of the clause list."""
    m = 0
    for clause in prey_clauses:
        if not clause:
            continue
        fam = clause[0]
        tags = clause[1:]
        if "motif" in tags and "len>=3" in tags:
            if fam in ("F", "P", "Z"):
                m |= PREDICATE_BIT[f"motif3_{fam}"]
        elif "motif" in tags:
            m |= PREDICATE_BIT[f"motif_{fam}"]
        else:
            m |= PREDICATE_BIT[f"family_{fam}"]
    return m
```

Now rewire the `feature_mask`/`prey_mask` accumulation block inside `phenotype()`. In `src/des/registry.py`, replace lines 56–125 (the body of `phenotype`) with the predicate-bit version below. The only changes from the existing body are: (a) drop `feature_mask |= FEATURE_BIT[letter]` from the per-letter loop, (b) drop the `for fam in fams: for t,bit in FEATURE_BIT.items(): ...` mass-OR block from the `_Z` branch, (c) compute `feature_mask = feature_mask_of(sequence)` once after the loop, and (d) compute `prey_mask` by accumulating clauses and calling `prey_mask_for_clauses` once after the loop.

```python
def phenotype(sequence: tuple[str, ...]) -> Phenotype:
    """Pure function of the sequence only. No world-state, no neighbors, no tick.
    κ=0 in v1 — no self-coordination neighbor scan. S6: feature_mask and prey_mask
    are predicate-bit ORs (not per-letter ORs); the antagonism kernel match
    expression is unchanged."""
    f_prod = 1.0          # accumulate Π(1-fᵢ)
    pl_prod = 1.0
    px_prod = 1.0
    z_sum = 0.0
    prey_clauses: list[tuple[str, ...]] = []
    directions: list[tuple[int, int]] = []
    periods: list[int] = []
    f_periods: list[int] = []
    z_periods: list[int] = []
    phase_type: PhaseType | None = None
    dominant_p: str | None = None

    for letter in sequence:
        if letter not in ALPHABET:
            continue
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

    f = 1 - f_prod
    p_leave = 1 - pl_prod
    p_x = max(MU, 1 - px_prod)
    spectrum = _spectrum_for(dominant_p) if dominant_p else ()
    period = min(periods) if periods else 1
    repro_period = min(f_periods) if f_periods else 1
    anta_period = min(z_periods) if z_periods else 1
    dir_bits = 0
    for d in directions:
        dir_bits |= _DIR_BIT.get(d, 0)

    # S6: predicate-bit masks. Antagonism kernel match expression unchanged.
    feature_mask = feature_mask_of(sequence)
    prey_mask = prey_mask_for_clauses(tuple(prey_clauses))

    return Phenotype(
        f=f, directions=tuple(directions), p_leave=p_leave, z_raw=z_sum,
        prey_mask=prey_mask, feature_mask=feature_mask, p_x=p_x,
        spectrum=spectrum, period=period,
        repro_period=repro_period, anta_period=anta_period, dir_bits=dir_bits,
        phase_type=phase_type, fold=(),
    )
```

Finally, `FEATURE_BIT` (line 19) is no longer used by `phenotype()`. Keep the constant in the module (tests in `tests/test_registry.py` still import it for historical reference and an existing test reads `FEATURE_BIT` to verify a letter-level invariant in earlier rungs). No removal in this task — pure data, no semantic role.

- [ ] **Step 4: Run the affected test files to verify they pass**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_motif.py tests/test_registry.py tests/test_antagonism.py -v
```

Expected: all motif tests + the rewritten registry tests + the existing antagonism tests pass. The antagonism kernel sees no change in match outcomes because the v1 prey clauses (`("F",)`, `("Z",)`) translate to the same kernel-visible `prey_mask` (just stored in different bit positions, and `feature_mask` of the prey lights up the matching `family_X` bit at the same predicate index — match relation preserved).

- [ ] **Step 5: Run the full suite to verify no regression**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 285 engine + 146 web tests stay green. If a test outside `tests/test_registry.py` reads `feature_mask` / `prey_mask` raw integer values directly (rather than going through the kernel match expression), that test must be updated to read predicate-bit semantics — investigate the failure and patch in this commit.

Backtrack: if `tests/test_phenotype_cache.py` fails, the most likely cause is that some test there hard-codes a numeric mask value derived from `FEATURE_BIT`. Replace those hard-codes with `PREDICATE_BIT["family_F"] | …` reads. If `tests/test_antagonism.py` fails on a `feature_mask`/`prey_mask` literal, do the same fix there.

- [ ] **Step 6: Commit**

```
git add src/des/registry.py tests/test_registry.py tests/test_motif.py
git commit -m "feat(s6): predicate-bit feature_mask / prey_mask

Switch phenotype() to feature_mask = OR(family_*/motif_*/motif3_* predicate
bits the sequence satisfies); _Z prey rows reshape to clause tuples
((family,), (family,'motif'), (family,'motif','len>=3')); add helpers
feature_mask_of(seq) and prey_mask_for_clauses(clauses). Antagonism kernel
match expression unchanged — only what each bit means changes. v1 alphabet
match outcomes byte-identical (single-element family clauses translate to
the same kernel-visible relation)."
```

---

### Task 8: `validate_bb0_layout` motif contiguity check + relabel-invariance audit

**Goal:** Extend `validate_bb0_layout` to raise on broken motif spans (non-contiguous, or length≠`MOTIF_LEN`) when any motif letter is present. No-op for all-residue layouts (regression lock). Then add the relabel-invariance audit test: shuffling `_F`/`_Z`/`_P` magnitudes across letters while keeping structural columns (gran/family/motif length) fixed must leave `motif_blocks`/`n_locked`/`feature_mask` unchanged — they read structure, not magnitude.

**Files:**
- Modify: `src/des/registry.py:140-159` (extend `validate_bb0_layout`)
- Test: `tests/test_motif.py` (append)

**Interfaces:**
- Consumes: `motif_blocks`, `GRAN`, `MOTIF_LEN` from earlier tasks.
- Produces: same signature `validate_bb0_layout(layout: tuple[str, ...]) -> None`. Additional raise condition: any motif letter present whose span is broken (the run of identical letters is shorter than `MOTIF_LEN[letter]` OR the letter recurs at a non-contiguous span of the wrong length).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_motif.py`:

```python
def test_validate_bb0_layout_all_residue_unchanged():
    """No motif letter present → validate behaves identically to pre-S6."""
    from des.registry import validate_bb0_layout, BB0_TEMPLATE
    validate_bb0_layout(BB0_TEMPLATE["layout"])   # must not raise


def test_validate_bb0_layout_broken_motif_span_raises(monkeypatch):
    """A length-3 motif placed at positions 0,1 (only 2 copies) must raise."""
    monkeypatch.setitem(registry.GRAN, "MF3", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "MF3", 3)
    monkeypatch.setitem(registry.ALPHABET, "MF3", "F")
    # Allow MF3 at the locked positions for this test by patching _LOCKED.
    monkeypatch.setattr(registry, "_LOCKED", {0: "MF3", 1: "MF3",
                                              5: "BroadSweep", 7: "P_base"})
    # Only 2 copies of MF3, but MOTIF_LEN is 3 → broken span.
    bad = ("MF3", "MF3", "N0", "N0", "N0", "BroadSweep",
           "N0", "P_base") + ("N0",) * 8
    with pytest.raises(ValueError, match="motif"):
        registry.validate_bb0_layout(bad)


def test_validate_bb0_layout_motif_correct_span_ok(monkeypatch):
    monkeypatch.setitem(registry.GRAN, "MF3", "motif")
    monkeypatch.setitem(registry.MOTIF_LEN, "MF3", 3)
    monkeypatch.setitem(registry.ALPHABET, "MF3", "F")
    monkeypatch.setattr(registry, "_LOCKED", {0: "MF3", 1: "MF3", 2: "MF3",
                                              5: "BroadSweep", 7: "P_base"})
    good = ("MF3", "MF3", "MF3", "N0", "N0", "BroadSweep",
            "N0", "P_base") + ("N0",) * 8
    registry.validate_bb0_layout(good)   # must not raise


def test_relabel_invariance_motif_n_locked_feature_mask(monkeypatch):
    """Shuffle f/z/p magnitudes across letters; fix structural columns
    (gran/family/MOTIF_LEN). motif_blocks / n_locked / feature_mask must
    be byte-identical because they read structure, not magnitude.

    This is the §6 relabel-invariance audit translated into a single test."""
    from des.registry import motif_blocks, n_locked, feature_mask_of, BB0_TEMPLATE
    layout = BB0_TEMPLATE["layout"]
    pre_blocks = motif_blocks(layout)
    pre_n = (n_locked(layout, "F"), n_locked(layout, "P"), n_locked(layout, "Z"))
    pre_mask = feature_mask_of(layout)
    # mutate _F / _Z / _P magnitudes (NOT gran / NOT family / NOT MOTIF_LEN)
    monkeypatch.setitem(registry._F, "F4Nr1",
                        (0.95, ((1, 0),), 0.99, 99))                # change f, p_leave, period
    monkeypatch.setitem(registry._F, "F4Nr4",
                        (0.01, ((-1, 0),), 0.01, 1))
    monkeypatch.setitem(registry._Z, "BroadSweep",
                        (0.99, (("F",), ("Z",)), 99))                # change z, period
    monkeypatch.setitem(registry._P, "P_hotspot", (0.0, 99))
    monkeypatch.setitem(registry._P, "P_base", (0.05, 1))
    # structural readouts MUST be unchanged
    assert motif_blocks(layout) == pre_blocks
    assert (n_locked(layout, "F"), n_locked(layout, "P"), n_locked(layout, "Z")) == pre_n
    assert feature_mask_of(layout) == pre_mask
```

- [ ] **Step 2: Run the new tests to verify they fail**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_motif.py -v -k "validate_bb0_layout or relabel_invariance"
```

Expected: `test_validate_bb0_layout_all_residue_unchanged` PASSes (already true). `test_validate_bb0_layout_broken_motif_span_raises` FAILs because the current `validate_bb0_layout` does not look at motif spans. `test_validate_bb0_layout_motif_correct_span_ok` may FAIL because the existing per-position `_LOCKED` check sees a locked `"MF3"` and demands a *current-`ALPHABET`* palette letter — adjust by walking through the new motif-aware path first (see step 3). `test_relabel_invariance_motif_n_locked_feature_mask` PASSes (the implementations from Tasks 2/5/7 already only read structure).

- [ ] **Step 3: Extend `validate_bb0_layout` in `src/des/registry.py`**

Replace the body of `validate_bb0_layout` at lines 140–159 with:

```python
def validate_bb0_layout(layout: tuple[str, ...]) -> None:
    """Enforce the BB0 symmetry invariant (viz spec §5 / red-line 4) + S6
    motif contiguity. locked positions must equal _LOCKED; backbone (non-locked,
    non-slot) positions must stay "N0"; only _SLOTS positions may vary, and
    only to a primitive in the current palette. If any motif letter is present,
    its blocks must be contiguous and exactly MOTIF_LEN positions long.
    Raises ValueError on any violation."""
    if len(layout) != 16:
        raise ValueError(f"BB0 layout must have 16 positions, got {len(layout)}")
    # S6: motif contiguity check (no-op for all-residue layouts).
    has_motif = any(GRAN.get(ltr) == "motif" for ltr in layout)
    if has_motif:
        # Walk left-to-right; whenever we see a motif letter, ensure exactly
        # MOTIF_LEN[letter] consecutive copies starting here, then jump past.
        i = 0
        n = len(layout)
        while i < n:
            ltr = layout[i]
            if GRAN.get(ltr) == "motif":
                need = MOTIF_LEN[ltr]
                end = i + need
                if end > n or any(layout[k] != ltr for k in range(i, end)):
                    raise ValueError(
                        f"motif {ltr!r} at position {i} requires {need} contiguous "
                        f"copies; got layout[{i}:{end}] = {layout[i:end]}")
                i = end
            else:
                i += 1
    # legacy per-position invariant (residues + locked + backbone)
    for i, letter in enumerate(layout):
        if i in _LOCKED:
            if letter != _LOCKED[i]:
                raise ValueError(
                    f"position {i} is locked to {_LOCKED[i]!r}, got {letter!r}")
        elif i in _SLOTS:
            if letter not in ALPHABET:
                raise ValueError(
                    f"slot {i} = {letter!r} not in palette {sorted(ALPHABET)}")
        else:
            if letter != "N0":
                raise ValueError(
                    f"position {i} is backbone-fixed to 'N0', got {letter!r}")
```

- [ ] **Step 4: Run the tests to verify they pass**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/test_motif.py -v
```

Expected: every test in `tests/test_motif.py` passes (motif contiguity raise, motif-correct OK, relabel-invariance).

- [ ] **Step 5: Run the full suite to verify no regression**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -x -q
```

Expected: 285 engine + 146 web tests stay green; new motif tests pass. Default all-residue BB0 takes the `has_motif == False` path and the function behaves byte-identically to pre-S6.

- [ ] **Step 6: Commit**

```
git add src/des/registry.py tests/test_motif.py
git commit -m "feat(s6): validate_bb0_layout motif contiguity + relabel audit

Extend validate_bb0_layout to raise on broken motif spans (non-contiguous
or length != MOTIF_LEN[letter]). No-op for all-residue layouts (regression
lock holds). Add relabel-invariance test: shuffling _F/_Z/_P magnitudes
must leave motif_blocks / n_locked / feature_mask unchanged because they
read structure, not magnitude (spec §6)."
```

---

### Task 9: Final regression sweep + smoke run

**Goal:** Prove the whole S6 deliverable is green together: data tables + `motif_blocks` + gran-matched spectrum + block-overwrite mutation + `n_locked` + predicate-bit masks + `validate_bb0_layout` + relabel-invariance audit. Smoke the default 4-faction symmetric run to confirm the all-residue regression lock holds byte-for-byte (no observable behavior change vs. main).

**Files:**
- No source modifications expected. If a regression slips through, fix it in this task and reference the offending commit in the message.
- Test: `tests/` (the entire suite) + `scripts/run_batch.py --probe` smoke

**Interfaces:**
- Consumes: every artifact produced by Tasks 1–8.
- Produces: a green `pytest tests/` and a clean `git status`.

- [ ] **Step 1: Full pytest sweep**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m pytest tests/ -q
```

Expected: every test passes. Total = 285 engine + 146 web + the new motif tests (≈25) + the new registry tests (≈3) + the appended reproduction test (1). Exact total may differ by ±1; no previously-green test may fail.

Backtrack: if a test fails, identify the task that introduced the regression by reverting commits one at a time until the suite is green again, then fix forward in a new commit.

- [ ] **Step 2: Smoke-run `scripts/run_batch.py --probe` to confirm runtime is unchanged**

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --probe 30
```

Expected output line: `[probe 30 ticks] X.X ms/tick | peak Y.YY GB | strains->… | total …`. The `X.X ms/tick` figure must remain in the same ballpark as before S6 (the spec target is ≈15.8 ms/tick on a 128² grid; ≤ 20% drift is acceptable for the probe). Exit 0; no parquet written under `data/runs/` (probe path is record-free).

If the drift is > 20%, the most likely cause is that `motif_blocks(seq)` is being recomputed inside a hot loop in `phase2_reproduce` for every individual rather than per parent. Confirm line 134 calls `motif_blocks(seq)` exactly once per `p in sorted(set(...))` iteration (the per-parent loop), not inside the inner `for letter, q in spectrum` enumeration.

- [ ] **Step 3: Byte-identical default-run smoke (optional but recommended)**

Drop a 1-faction symmetric-default seed run alongside a pre-S6 baseline parquet to confirm the strain trajectory is identical:

```
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe scripts/run_batch.py --seeds 0 --T 50
```

Expected: a new parquet under `data/runs/`. Compare to the matching pre-S6 baseline parquet (if one was kept). The `{strain: count}` per `(tick, cell)` rows must be byte-identical because: (a) the v1 alphabet is all-residue → `motif_blocks` returns 16 singletons → block-overwrite reduces to singleton overwrite → mutation outcomes byte-identical, (b) `_spectrum_for` filter is a no-op on the residue-only alphabet → spectrum byte-identical, (c) `feature_mask` / `prey_mask` predicate-bit encoding preserves the kernel-visible match relation byte-identically.

If a baseline parquet is unavailable, the test_smoke and test_acceptance tests in `tests/` cover the same invariant in pytest form.

- [ ] **Step 4: Inspect & clean stray data**

```
git status
```

Expected output: clean working tree. If `data/runs/<ts>-*.parquet` files from smoke runs appear, remove them — they are not test fixtures.

- [ ] **Step 5: Final commit (only if step 1 needed a fix-forward)**

If step 1 surfaced a regression you fixed:

```
git add <files-touched>
git commit -m "fix(s6): <description of the regression fixed>"
```

Otherwise this step is a no-op.

- [ ] **Step 6: Push to origin**

```
git push origin <current-branch>
```

Expected: push succeeds. The branch is ready for review / merge to `main`.

---

## Self-Review

**1. Spec coverage:**
- §3.1 (gran registry property): Task 1 — `GRAN[letter]` and `MOTIF_LEN[letter]` data tables; v1 alphabet all-residue, `MOTIF_LEN` empty.
- §3.2 (motif encoding: repeated-letter + derived blocks, layout stays `tuple[str]` of 16, no stored group-map): Task 2 — `motif_blocks(layout)` is the derived helper; nothing stored on `Phenotype` or `StrainTable`.
- §3.3 (mutation core respects gran — spectrum pre-filter + block overwrite): Task 3 (gran-match + equal-length pre-filter in `_spectrum_for`) + Task 4 (`_mutation_outcomes` block overwrite, equal-length guarantee preserves the 16-position layout).
- §3.4 (`n_locked(chan)` structural readout, computed on demand, not stored): Task 5 — `n_locked(layout, chan)`; raises on N; advisory only (no consumer in S6 since A is de-gated per S8 roadmap).
- §3.5 (predicate-bit encoding scheme, full vocabulary, 64-bit overflow resolved): Task 6 (vocabulary + import-time int64 assertion) + Task 7 (`feature_mask_of`, `prey_mask_for_clauses`, `phenotype()` rewiring, `_Z` clause reshape, antagonism kernel match unchanged).
- §4 (data flow): Tasks 2 + 3 + 4 + 7 — `mint(seq) → phenotype() → motif_blocks → spectrum pre-filter → Phenotype.spectrum`; `feature_mask = OR predicate bits`; `mutate → block overwrite via motif_blocks`. `n_locked` (Task 5) is the on-demand structural readout.
- §5 (error handling: motif span broken raises, predicate vocabulary fits int64): Task 8 (motif contiguity check in `validate_bb0_layout`) + Task 6 (`assert max(PREDICATE_BITS.values()) < 63`).
- §6 (testing: regression lock + gran-matched spectrum + block overwrite + n_locked + predicate bits + antagonism match unchanged + relabel-invariance): Tasks 1–8 each ship their own pytest + the regression lock is reproven at the end of every task; Task 8 ships the relabel-invariance audit; Task 9 is the final all-suite sweep + smoke run.
- §7 (out of scope): no S1 vis predicate values introduced (bit `vis_lowvis` reserved with no body), no S3 threshold values (3 thresh bits reserved), no S8 A-pool gate consumer, no κ same-channel synergy span-awareness, no full 68-letter `_F`/`_Z`/`_P` tables.
- Red lines (§2): no per-species gran knob (only `GRAN[letter]` global); strength still flows only via `_F`/`_Z`/`_P` (Task 7's `_Z` reshape is structural, not magnitudinal); motif structure is derived (Task 2), not stored; default game byte-identical (every task closes with the suite-green check + Task 9 byte-identical smoke against pre-S6 baseline).

**2. Placeholder scan:** No `TBD`, `TODO`, "implement later", "fill in details", or "similar to Task N" remain. Every code step shows the actual code; every command step shows the actual command and expected output; every backtrack condition is concrete and named.

**3. Type consistency:**
- `GRAN: dict[str, str]` and `MOTIF_LEN: dict[str, int]` — same names + types in Task 1 / 2 / 3 / 4 / 5 / 7 / 8.
- `motif_blocks(layout: tuple[str, ...]) -> tuple[tuple[int, int, str], ...]` — same signature + return tuple-of-tuples in Tasks 2 / 4 / 5 / 7 / 8.
- `_spectrum_for(letter: str) -> tuple[tuple[str, float], ...]` — same signature pre-S6 and post-S6 (Task 3 changes the body, not the contract).
- `_mutation_outcomes(seq, mutable, spectrum, blocks) -> (children, weights)` — 4-arg signature consistent across Task 4 implementation, Task 4 call-site update at `phase2_reproduce` line 134, and the `tests/test_reproduction.py::test_mutation_outcomes_signature_takes_blocks` regression test.
- `n_locked(layout: tuple[str, ...], chan: str) -> int` — same signature in Task 5 implementation + Task 8 relabel-invariance test.
- `PREDICATE_BITS: dict[str, int]` (15 names) and `PREDICATE_BIT: dict[str, int]` (`{name: 1 << idx}`) — defined once in Task 6, consumed by `feature_mask_of` and `prey_mask_for_clauses` in Task 7, asserted by tests in Tasks 6 / 7 / 8.
- `feature_mask_of(sequence) -> int` and `prey_mask_for_clauses(prey_clauses) -> int` — defined Task 7, consumed in `phenotype()` (Task 7) and the relabel-invariance test (Task 8).
- `_Z` row shape `(z, prey_clauses, period)` with `prey_clauses: tuple[tuple[str, ...], ...]` — reshaped in Task 7 step 3; `phenotype()` consumes the new shape in the same step; `BroadSweep` clause `(("F",), ("Z",))` is the only v1 row.
- `validate_bb0_layout(layout) -> None` — same signature pre-S6 and post-S6 (Task 8 extends the body with the `has_motif` branch).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-24-s6-motif-granularity.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. Use `superpowers:subagent-driven-development`.
2. **Inline Execution** — execute tasks in this session with batch checkpoints. Use `superpowers:executing-plans`.

Which approach?
