# S6 — motif 粒度 (cross-cutting foundation)

> Created: 2026-06-24 · Roadmap sub-project 2/9 (order S0→**S6**→S1→S2→S4→S5→S3→S7→S8).
> Foundation spec: establishes the representations S2/S3/S7/S8 build on. Faithful translation of the roster — no new gameplay.

## 1. Why

The roster has two granularities (`primitive-roster.md` N pool +末尾「motif 粒度突变规则」): **residue** (single position, e.g. N0–N3/N5/N6, all current 6 v1 letters) and **motif** (a primitive spanning multiple consecutive positions, e.g. N4/N7, and every Z motif-specialist's prey). The current code is residue-only: `layout` is a flat `tuple[str]` of 16 single letters, `mutable` a per-position bool tuple, and `_mutation_outcomes` overwrites one slot. S6 adds the motif representation so later specs (motif prey predicates, n_locked, A pool) have a foundation.

Two settled rulings this spec encodes (both already in the roster/design, not invented here):
- **motif↔motif mutation is equal-length** (`primitive-roster.md` line 277, design.md L322 `P_loopswap_lite`): an N-position motif mutates only to another N-position motif; strain length is fixed within a lineage.
- **granularity pairing** (`primitive-roster.md` line 273): mutation happens only within the same gran — residue↔residue, motif↔motif. This is the mutation-core mechanism itself, not a separate law.

## 2. Red lines

- **No smuggled bias**: `gran` is a global per-primitive registry property (`gran[letter]`), never `gran[species][letter]`. Strength still flows only through `_F/_Z/_P`. Granularity pairing is a structural "where can mutation go" rule, not a "who is strong" judgment.
- **Phenotype stays a pure function of the sequence**: motif block structure is *derived* from the layout, not stored as mutable world-state.
- **Default game unchanged**: the default BB0 is all-residue (6 v1 letters, no repeated motif letters). Every motif mechanism added here is **dormant** until an asymmetric backbone places a motif primitive — byte-identical default runs are the regression lock.

## 3. Architecture

### 3.1 `gran` registry property
Add `GRAN = {letter: "residue" | "motif"}` to `registry.py`, one entry per primitive (the roster declares it per-entry). v1's 6 letters are all `residue`. A motif letter additionally declares its span length `MOTIF_LEN = {letter: N}` (intrinsic to the primitive — the roster's `{motif}` vs `{ℓ≥3 motif}` distinction is a length difference between *different* motif primitives, not a mutable property).

### 3.2 Motif encoding: repeated-letter + derived blocks (ponytail-chosen, HOW-3)
A motif occupying N positions = the **same motif-letter repeated across N consecutive positions** in the flat-16 layout. The layout stays `tuple[str]` of 16; `mutable`/`fold`/`validate_bb0_layout`/world geometry are byte-identical. A derived pass `motif_blocks(layout) -> [(start, end, letter)]` groups runs of the same `gran=="motif"` letter into blocks; residue letters (incl. N0) are always singletons. Nothing stored in the frozen Phenotype; nothing to keep in sync.

> Rejected (over-engineering): a stored group-map (derivable → don't store); a variable-length layout tuple (breaks position-indexed `fold` sets + the `len==16` invariant).

### 3.3 Mutation core respects gran (the foundational change to `_mutation_outcomes`)
- **Spectrum pre-filter**: in `_spectrum_for(letter)`, restrict targets to `gran(target) == gran(letter)` (residue sources → residue targets; motif → equal-length motif targets). One mask line.
- **Block overwrite**: in `_mutation_outcomes`, when the chosen mutable slot lies inside a motif block, the outcome overwrites the **whole block** atomically to the new (equal-length) motif letter; a residue slot overwrites singly. Equal-length guarantees the 16-position layout is preserved.

### 3.4 `n_locked(chan)` counting (structural readout)
Compute at mint, in `phenotype()` or a registry helper: iterate `motif_blocks` over the backbone-locked positions, count **blocks** whose primitive family == chan (F/P/Z), a motif block counting as 1 regardless of span (`primitive-roster.md` OPEN-1 ②). N never counts. Store as 3 small CPU-side ints on the strain. For the default BB0 this is F:1/P:1/Z:1 (unchanged, already cross-referenced in design.md).

> **Scope honesty**: n_locked was originally the input to the A-pool overwrite gate. The roadmap **de-gates A** (S8: A reachable by pure affinity spectrum, gate retired) → n_locked has **no mutation consumer**. S6 still computes it as a cheap structural readout (3 ints, used by the relabel-invariance audit and available for the future asymmetric-backbone role system), but does NOT wire it into any gate. If you'd rather not compute an unused value, n_locked can be dropped from S6 entirely with zero impact on the default game — flagged as a trim option for the plan phase.

### 3.5 Predicate-bit encoding scheme (resolves the 64-bit overflow; foundation for S1/S3/S8)
`feature_mask`/`prey_mask` are currently **per-letter** bits. 68 letters > 64-bit budget → S6 switches the scheme to **predicate bits**: each bit = a structural predicate, not a letter. S6 defines the *scheme + the full vocabulary* (enumerated from the roster's Z/A prey clauses); S1 populates the vis predicate, S3 populates the threshold predicates. The vocabulary (verbatim from the roster, this is a read not a decision):

| predicate | source clauses | filled by |
|---|---|---|
| family ∈ {N,F,P,Z} (4 bits) | `{F,Z}`,`{N}`,`{Z,P}`,`{Z}`,`{P}`,`{F,P}`,`{F,Z,P}` | S6 |
| motif∋fam, fam∈{F,P,Z,N} (4 bits) | Ambush Coil/Burst Leech/Clade Snare/Frame Pincer/Ambush Venom | S6 |
| ℓ≥3 motif∋fam, fam∈{F,P,Z} (3 bits) | Lineage Reaper/Coil Cinch/Idiotype Lance/Predator Lock | S6 |
| fam=F ∧ f≥0.5 | Crest Bite | S3 |
| fam=P ∧ p_add≥0.05 | Hotspot Snipe | S3 |
| fam=Z ∧ z≤0.45 ∧ \|prey\|≥2 | Mirror Fang | S3 |
| fam=N ∧ vis≤0.20 | Void Bite (A pool) | S3 (vis bit from S1) |

A strain's `feature_mask` = OR of every predicate bit it satisfies (computed at mint, pure function of sequence). A predator's `prey_mask` = OR of the predicate bits its prey clause selects. Antagonism match stays exactly `(prey_mask[i] & feature_mask[j]) != 0` — **the antagonism kernel does not change**, only what the bits mean.

## 4. Data flow
```
mint(seq) ─► phenotype():
   gran/motif_blocks(seq) ──► spectrum pre-filter (gran-matched) ──► Phenotype.spectrum
                          └─► n_locked_F/P/Z (blocks in locked positions)
                          └─► feature_mask = OR(predicate bits seq satisfies)   [S6 family/motif/ℓ≥3 bits; S1/S3 fill rest]
mutate: _mutation_outcomes ─► slot in motif block? overwrite whole block (equal-len) : overwrite single
```

## 5. Error handling
- A motif layout where a declared span is broken (non-contiguous, or length≠MOTIF_LEN): `validate_bb0_layout` raises — extend it to check motif contiguity/length when any motif letter is present (no-op for all-residue layouts).
- Predicate vocabulary > 64 bits: it is not (≤ ~15 bits). Assert the bit count fits int64 at module load.

## 6. Testing
- **Regression lock**: 285 engine + 146 web tests green (default all-residue path byte-identical; `motif_blocks` returns all-singletons → no behavior change).
- **New**: `motif_blocks` groups a hand-built motif layout correctly; gran-matched spectrum excludes cross-gran targets; block overwrite preserves length and replaces the whole block; n_locked counts a locked motif as 1; `feature_mask` predicate bits set correctly for sample residue + motif strains; antagonism match unchanged on a known-answer case.
- **relabel-invariance**: shuffle f/z/p across letters, fix structural columns (gran/family/motif), recompute → motif/n_locked/predicate-bit results must be identical (reads structure, not magnitude).

## 7. Out of scope
- vis predicate values (S1), threshold predicate values (S3), A-pool gate consumption of n_locked (S8 — note: with A de-gated per the roadmap, n_locked may become advisory; S8 settles it).
- κ same-channel synergy span-awareness (κ=0 in v1; motifs dormant in default).
- The full 68-letter `_F/_Z/_P` tables (each later spec adds its own primitives' rows).
