# S7 — 多位点突变 (multi-slot mutation)

> Created: 2026-06-24 · Roadmap sub-project 8/9 (S0→S6→S1→S2→S4→S5→S3→**S7**→S8). Depends on S2 (spectrum) + S6 (gran).
> Faithful translation of P_cascade. Small. No new gameplay.

## 1. Why
One A-pool primitive mutates **2 slots in a single event** (SHM 连发):
- **P_cascade** (`primitive-roster.md` line 230): `p_add=0.28, rate=0.29, q∝aff(·), 2 slots/event, T=2`.

Every other mutation primitive does 1 slot/event (the current `_mutation_outcomes` behavior). S7 generalizes the mutation core to N slots/event, defaulting N=1.

## 2. Red lines
- `slots_per_event` is a global per-primitive registry int — structural, not per-species. P_cascade's "2" is verbatim from the roster.
- Pure mutation-core change; no kernel-physics change beyond drawing 2 slot×spectrum outcomes instead of 1.
- Default: every v1 primitive has slots_per_event=1 → byte-identical. P_cascade is an A primitive, dormant until S8 mints it.

## 3. Architecture
Single locus: **`_mutation_outcomes(seq, mutable, spectrum)` in `reproduction.py`**, plus a `slots_per_event` field on the spectrum-source primitive.

Current `_mutation_outcomes` builds the per-event categorical over (mutable slot × spectrum letter) for **1** slot. For N slots/event:
- Draw N **distinct** mutable slots (uniform without replacement among mutable slots), each independently drawing a letter from the spectrum, applied to the child sequence in one event.
- N is `slots_per_event` of the strain's dominant-P primitive (the one already selected for the spectrum). Default 1 → identical to current single-slot path.
- gran pairing (S6) applies per chosen slot: a motif slot draws an equal-length motif letter; a residue slot a residue letter. Each of the N slots resolves independently under its own gran.

> Ponytail ceiling: with N=2 and ~6 mutable slots, the outcome space is the product (slot-pair × letter²) — small. The current per-individual multinomial scatter (reproduction.py:108–149) extends by drawing 2 slot indices per mutant individual. # ponytail: per-individual 2-slot draw; if a future primitive wants N≫2 across many slots the product explodes → switch to sequential single-slot application N times. Not before.

## 4. Data flow
```
mint(seq) ─► phenotype(): slots_per_event = N (dominant-P primitive; default 1)
phase2 mutation: per mutant individual ─► draw N distinct mutable slots ─► each draws spectrum letter (gran-matched) ─► one child
```

## 5. Error handling
- N > number of mutable slots: clamp N to #mutable (can't mutate more slots than exist). For default BB0 (6 mutable) and N=2, fine.
- N distinct slots: sample without replacement; if #mutable < N, use all mutable slots.
- Same-sequence children still merge via `get_or_mint` (existing).

## 6. Testing
- Regression: 285+146 green (N=1 default → current single-slot path unchanged; verify the refactor produces identical outcomes for N=1 against the existing mutation tests).
- New: P_cascade (N=2) produces children differing from parent at exactly 2 mutable slots (when both draws are non-self-loop); the 2 slots are distinct; gran respected per slot; N clamped when #mutable<N; same-sequence cascade children merge.
- relabel-invariance: slots_per_event is structural; shuffling f/z/p doesn't change which slots/letters are drawn (drives off spectrum + mutable, both structural).

## 7. Out of scope
- P_cascade's rate/spectrum (S2 handles rate via p_add; spectrum is plain aff). The A-pool gating/minting of P_cascade — S8.
