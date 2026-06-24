# S8 — A 池 24 极端变体 (de-gated)

> Created: 2026-06-24 · Roadmap sub-project 9/9 (S0→S6→S1→S2→S4→S5→S3→S7→**S8**). Depends on S1/S2/S3/S4/S5/S6/S7 (A primitives reuse all prior mechanisms).
> Faithful translation of the A pool + the **de-gate** decision (user, 2026-06-24). No new gameplay.

## 1. Why
The A pool (`primitive-roster.md` lines 180–264) is 24 extreme/edge variants, all producing only f/p/z. They reuse every mechanism from S1–S7 at extreme parameter values:
- **乙1 (8 escalation copies)**: Apex Bloom (f=0.85)/Ember Drip/Bastion Pile/Apex Fang (z=1.50)/Pan Sweep (prey={F,Z,P})/Hotspot Amp (p_add=0.30)/Sink Cascade/Glacial Drift.
- **乙2 (8 native)**: F_NOVA (phase, S5)/F_TRICKLE/F_SCATTER (hash dirs, S4)/Predator Lock (ℓ≥3 motif∋Z, S3/S6)/Void Bite (vis≤0.20 + inverse-vis hit, S1/S3)/P_cascade (2 slots, S7)/P_crossclan_surge (|Δrank|≥2 mask, S2)/P_frozen (aff⁴, S2).
- **甲 (8 native)**: F8Ar1/Lance Front/Ambush Venom (motif∋F)/Sweep Surge (prey={F,P})/Nip Whisper (vis-weighted N)/Coil Null (prey={Z})/P_zscan_invert (𝟙[fam=F])/P_stutter (aff⁴).

The A pool is the **collection point**: it adds no new mechanism, only new registry rows at extreme values. The only S8-specific decision is reachability.

## 2. The de-gate (user decision, 2026-06-24)
A primitives are **de-gated**: reachable purely by the global affinity spectrum, exactly like any other primitive. The `n_locked(chan)≥θ` overwrite gate is **retired**.
- **A primitives are family F/P/Z at extreme values, NOT a 5th rank-4 family.** design.md line 339 abolished the rank-4 family in the 2026-06-20 descaffold: "字母族 S 族(rank4)… **作废** → 族降 4 档 `N<F<P<Z`". So Apex Bloom is family F, Apex Fang is family Z, Hotspot Amp is family P. "A 池 / rank 4" is an **organizational tier**, not a mutation-family. `FAMILY_RANK` stays {N,F,P,Z}; there is no rank-4 letter.
- **Reachability**: an extreme-F variant is reached from other F primitives **within family F at aff=0.70** (common), gran-matched — the normal spectrum, no special path. (Layering: 0.70/0.25/0.05 are the `affinity()` weights — same-family / adjacent |Δrank|=1 / cross |Δrank|≥2, `FAMILY_RANK` {N:0,F:1,P:2,Z:3} per S2; `_spectrum_for` normalizes them over the alphabet into the actual draw probabilities.) This is *more* reachable than the retired-gate framing assumed; A variants are ordinary same-family spectrum targets that happen to sit at extreme parameter values.
- Rationale for de-gating (holds, rationale corrected): the gate was an *extra* restriction layered on top of the spectrum, and it had to read backbone composition (`n_locked`) to decide reachability — a smuggled-bias surface. Removing it = A obeys the single global affinity rule, **fewer bias surfaces**.
- Cost (accepted): the "建难" property (A only emerges in specialized species) is gone; A is reachable as a normal same-family spectrum target.
- Roster cleanup (part of S8): the 24 `覆写: {株:n_locked≥θ}` lines + OPEN-1/θ section become **dead**. S8 rewrites each `覆写:` line verbatim to:
  > `覆写: A reachable via affinity spectrum (same-family draw, aff=0.70, gran-matched); θ-gate retired (de-gate, 2026-06-24).`
  and marks the OPEN-1/θ section:
  > `RETIRED (2026-06-24) — the n_locked≥θ overwrite gate is removed; A obeys the single global affinity rule. n_locked kept as an advisory structural readout, not wired into mutation.`
  **`n_locked` itself**: S6 computes it on demand; with the gate retired it has no consumer → advisory/unused (a structural readout, not wired into mutation).

## 3. Red lines
- De-gating removes a gate; it adds no "who is strong". All A strengths flow through `_F/_Z/_P` (extreme values, but global tables). The extreme ranges are roster-declared (f≤0.85 / z≤1.5 with narrow prey holding the z↔list anti-correlation / p_add≤0.34, rate cap 0.35).
- **Default game unchanged**: A primitives are rank-4, gran-matched, spectrum-reachable — but the default BB0's mutable slots are residue and the affinity path to rank-4 from a residue F/P/Z slot is the normal 0.25/0.05 spectrum. With four identical BB0 factions, A emergence is symmetric across factions → no faction asymmetry, no selection signal (the asymmetric-backbone role system is still HARD-GATE). De-gating lets A *appear* in the default game (rarely, symmetrically); it does NOT create roles.
- `copy-of` is lineage annotation only — the mutation core never reads it (roster line 183).

## 4. Architecture
S8 is mostly **registry data entry** + the de-gate edit. No new mechanism code.
- Add 24 A rows to `_F/_Z/_P` + `ALPHABET` (each tagged its true family F/P/Z, NOT rank-4) + `GRAN`/`MOTIF_LEN` (S6) + `SPECTRUM_SHAPE` (S2, for the P variants) + window params (S5, F_NOVA) + `rand_dir`/hash dirs (S4) + `slots_per_event` (S7, P_cascade) + `PREY_CLAUSE` (S3, the Z/motif/vis predators) + vis (n/a for A — A produces only f/p/z).
- **affinity is untouched**: A primitives are family F/P/Z, so they slot into the existing families and are reached within-family at aff=0.70 (extreme F from normal F, etc.). `FAMILY_RANK` stays {N,F,P,Z}; no rank-4 letter, no affinity change.
- The de-gate edit is mostly a **no-op in code**: the current `_mutation_outcomes` has no n_locked gate (the gate only ever existed on paper). De-gating = a **roster doc cleanup** (§2) + confirming no gate logic is added by S6/S8.

### 4.1 Multi-P spectrum blend (user ruling 2026-06-24 — S8 owns it)
The one real mechanism change S8 makes. v1 picks a single `dominant_p` (highest p_add, ties by first occurrence) as the spectrum source (`registry.py:101-105`); design.md L223 specifies **blending** the stacked-P spectra weighted by p_add:

```
spectrum(t) = Σ_i p_add_i · q_i(t)  /  Σ_i p_add_i      # over the strain's P letters i
```

This replaces the `dominant_p` selection in `phenotype()`. It lands in S8 because S8's A-pool rows are exactly the multi-slot P variants that exercise it: **P_cascade** (2 slots/event, S7), **P_frozen**/**P_stutter** (aff⁴ sharpening). Dormant in the default game: one P letter → `Σ` has a single term → blend == dominant_p == that letter's spectrum, byte-identical to v1's single-letter path. Each `q_i` is the already-shaped, gran-filtered, normalized spectrum from S2's `_spectrum_for`; the blend is a p_add-weighted average over the (already cached) per-letter spectra, then re-normalized.

## 5. Data flow
A primitives flow through the exact same mint→phenotype→kernel paths S1–S7 built; S8 only populates their registry rows. No new data path.

## 6. Error handling
- Extreme values must stay in roster bounds: assert f≤0.85, z≤1.5, p_add≤0.34 (rate≤0.35) at module load for A rows.
- z↔prey anti-correlation (high z must have narrow prey): not a code constraint, a design invariant already baked into each A's roster formula; no runtime check (the values are fixed data).

## 7. Out of scope / notes
- **Resolved (was a flag):** "Is an A primitive rank-4 or family-F/P/Z?" — settled by design.md line 339 (rank-4 family abolished, 族降 4 档 N<F<P<Z). A primitives are family F/P/Z at extreme values, reached within-family at aff=0.70. No 数值 call needed; the design already decided it.
- The asymmetric-backbone role system (per-faction K/rate/mechanics) remains HARD-GATE, untouched.
- κ same-channel synergy (κ=0 in v1).
- A given the within-family aff=0.70 reachability, A variants appear in the default game more readily than the retired gate implied — but still **symmetrically across the four identical BB0 factions**, so no faction asymmetry / no selection signal. Roles still require the HARD-GATE'd asymmetric-backbone system.
