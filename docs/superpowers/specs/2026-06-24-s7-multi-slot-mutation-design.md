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
- Default: every v1 primitive has slots_per_event=1 → byte-identical **by construction** (same enumeration order, same RNG call count — not merely the same distribution; see §3). P_cascade is an A primitive, dormant until S8 mints it.
- For all of S0–S7, every active primitive has slots_per_event=1, so N=1 always and the default game is byte-identical regardless of any aggregation. P_cascade is the **only** primitive in the 68-roster with slots≠1 (roster L230), minted in S8.

## 3. Architecture
Single locus: **`_mutation_outcomes(seq, mutable, spectrum)` in `reproduction.py`**, plus a `slots_per_event` field on the spectrum-source primitive.

Current `_mutation_outcomes` builds the per-event categorical over (mutable slot × spectrum letter) for **1** slot. For N slots/event:
- **N==1 → current code verbatim.** The single combined `(slot × letter)` categorical + one `torch.multinomial` per distinct parent, enumeration order = slot-ascending × `_spectrum_for` order, weight `q/|slots|` (reproduction.py:43–47, 140–143). N=1 is byte-identical **by construction** — same enumeration order, same RNG call count — not just distributionally equal.
- **N≥2 → joint-enumeration path.** Enumerate unordered slot-sets S of size N (distinct mutable slots) × per-slot spectrum letters. Weight of a `(slot-set S, letters)` outcome = `(1 / C(m,N)) · ∏_{s∈S} q(letter_s)`, where m = #mutable. At N=1 this reduces to `q/C(m,1) = q/m = q/|slots|` — i.e. the formula is continuous with the current path. One `torch.multinomial` over the enumerated weights per distinct parent (same draw machinery as N=1).
- N is `slots_per_event`, read from **the same selected primitive the spectrum is sourced from** (the existing v1 spectrum-source: highest `p_add`, ties by first occurrence — `registry.py:101–105`, `dominant_p`). It piggybacks S2's spectrum-source selection; it is **not** a new selection rule. Default 1 → identical to current single-slot path.
- gran pairing (S6) applies per chosen slot: a motif slot draws an equal-length motif letter; a residue slot a residue letter. Each of the N slots resolves independently under its own gran.

> Why enumerate (not sequential) at N=2: enumerate reuses the existing `_mutation_outcomes` build + single-`multinomial` draw machinery → less new code, and P_cascade is dormant until S8 so the enumeration cost never runs hot. Sequential is the deferred upgrade path only (see ceiling note).
> **slots_per_event aggregation across stacked P is DEFERRED to S8.** For all of S0–S7 every active primitive has slots_per_event=1, so N=1 always (P_cascade is the sole slots=2 primitive and cannot coexist with another P until S8 mints it). The rule for combining slots_per_event when multiple P stack is the same v1-single-source-vs-design-blend question S2 owns for the spectrum (design.md L223 blend vs the v1 single-dominant simplification) — not decided here.

> Ponytail ceiling: with N=2 and ~6 mutable slots, the outcome space is the product (slot-pair × letter²) — magnitude ≈ `C(6,2)·|spectrum|²` per parent (~3840 for m=6, |spectrum|≈16), tractable. The current per-individual multinomial scatter (reproduction.py:108–149) extends by drawing 2 slot indices per mutant individual. # ponytail: per-individual 2-slot draw; at N≥3 the `C(m,N)·|spectrum|^N` product explodes → switch to sequential single-slot application N times (each step excluding already-mutated slots, to keep slots distinct). Not before.

## 4. Data flow
```
mint(seq) ─► phenotype(): slots_per_event = N (read from the same primitive the spectrum is sourced from; default 1)
phase2 mutation: per mutant individual ─► draw N distinct mutable slots ─► each draws spectrum letter (gran-matched) ─► one child
```

## 5. Error handling
- N > number of mutable slots: clamp N to #mutable (can't mutate more slots than exist). For default BB0 (6 mutable) and N=2, fine.
- N distinct slots: sample without replacement; if #mutable < N, use all mutable slots.
- Same-sequence children still merge via `get_or_mint` (existing).

## 6. Testing
- Regression: 285+146 green (N=1 default → current single-slot path unchanged; verify the refactor produces identical outcomes for N=1 against the existing mutation tests).
- New: P_cascade (N=2) produces children differing from parent at exactly 2 mutable slots (when both draws are non-self-loop); the 2 slots are distinct; gran respected per slot; N clamped when #mutable<N; same-sequence cascade children merge.
- New (N=2 distribution): over many draws, the empirical `(slot-set, letters)` frequencies match the joint mass `(1/C(m,N))·∏ q(letter_s)` — confirms the weight formula and per-slot independence.
- relabel-invariance: slots_per_event is structural; shuffling f/z/p doesn't change which slots/letters are drawn (drives off spectrum + mutable, both structural).
- Deferred to S8 (with the stacked-P aggregation, §3): a test that slots_per_event resolves correctly when P_cascade coexists with another P.

## 7. Out of scope
- P_cascade's rate/spectrum (S2 handles rate via p_add; spectrum is plain aff). The A-pool gating/minting of P_cascade — S8.
