# S2 — 塑形突变谱 (shaped mutation spectrum)

> Created: 2026-06-24 · Roadmap sub-project 4/9 (S0→S6→S1→**S2**→S4→S5→S3→S7→S8).
> Faithful translation of the P pool's spectrum-shaping variants. Touches ONE function (`_spectrum_for`). No new gameplay.

## 1. Why
The P pool (`primitive-roster.md` lines 89–124) is 12 mutation primitives differing in **rate** (`p_add`) and **spectrum shape** `q(t)` — the categorical over what a mutation becomes. Current code has only `P_base` (`q ∝ aff`) and `P_hotspot` (same shape, higher rate). S2 adds the 10 remaining P primitives, which are all variations of the spectrum-shaping function (verbatim from roster):

| primitive | p_add | q(t) ∝ |
|---|---|---|
| P_base | 0 | aff(fam(x),fam(t)) |
| P_hotspot | 0.05 | aff(·) |
| P_aic | 0.03 | aff(·)² (sharpen) |
| P_ep | 0.04 | ½·aff(·) + ½·1/(\|A\|−1) (flatten) |
| P_fscan | 0.02 | aff(·)·𝟙[fam(t)=F] |
| P_zscan | 0.02 | aff(·)·𝟙[fam(t)=Z] |
| P_entropy_brake | 0.01 | aff(·)³ |
| P_loopswap_lite | 0.03 | aff(·)·𝟙[\|Δrank\|=1] |
| P_neutral_sink | 0.02 | aff(·)·𝟙[fam(t)=N] |
| P_slow_drift | 0 | aff(·) |
| P_burst_lite | 0.07 | aff(·) |
| P_balanced | 0.04 | aff(·) |

`rate = min(p_max, μ + p_add)`, already implemented. T (period) per roster.

## 2. Red lines
- Every spectrum shape is a global function of `(fam(x), fam(t), rank)` — structural, no per-species magnitude. The shapes bias *direction* of mutation (toward F / toward N / sharpen / flatten), never hand-write "this species mutates better".
- `|A|` (alphabet size) in P_ep's flatten term = the live registry size; pure structural.
- Default: P_base/P_hotspot keep their exact current shape (`aff`, `aff` — power 1, no mask) → byte-identical. The 10 new primitives are dormant until minted.

## 3. Architecture
Single locus: **`_spectrum_for(letter)` in `registry.py`**. Currently hardcodes `q ∝ aff(src_fam, dst_fam)`. Generalize to a per-primitive **shape descriptor** read from a table:

```
SPECTRUM_SHAPE = {
  letter: (power, family_mask, flatten_mix)
}
# power: exponent on aff (1 default, 2 aic, 3 entropy_brake)
# family_mask: None (all) | {"F"} | {"Z"} | {"N"} | "adjacent" (|Δrank|=1)
# flatten_mix: 0.0 default | 0.5 for P_ep (½ aff + ½ uniform)
```
`_spectrum_for` computes per target t:
```
w(t) = (aff(src,fam(t)) ** power) · 𝟙[mask(t)]
w(t) = (1-mix)·w(t) + mix·(1/(|A|-1))    # P_ep only
```
then gran-filter (S6: `gran(t)==gran(letter)`), normalize Σ=1. Pure function, already cached in `Phenotype.spectrum`. **No kernel change** — the reproduction kernel consumes `spectrum` unchanged.

> Lazy note: the four scan/sink variants are the *same* code path (a family mask); aic/entropy_brake are the *same* path (a power); P_ep is the one with the flatten term. Three knobs cover all 12. No per-primitive special-casing.

## 4. Data flow
```
mint(seq) ─► phenotype() ─► dominant P letter ─► _spectrum_for(letter):
   aff^power · family_mask · flatten_mix ─► gran-filter (S6) ─► normalize ─► Phenotype.spectrum
reproduction kernel: consumes spectrum unchanged
```

## 5. Error handling
- A family_mask that excludes all targets (e.g. P_fscan in a registry with no F letters): Σw=0 → `_spectrum_for` returns `()` (already handled: empty spectrum → no mutation). Guard the normalize divide-by-zero (existing `if tot==0: return ()`).
- power/mix out of expected range: assert shape-table values at module load.
- Multi-P strain (more than one P letter): the existing `dominant_p` rule (highest p_add, ties by first occurrence) picks the shaping primitive — unchanged from current code; a principled multi-P blend stays deferred (already noted in registry.py:103).

## 6. Testing
- Regression: 285+146 green (P_base/P_hotspot shape identical).
- New: each of the 10 shapes produces the roster-specified bias — P_fscan mass only on F targets; P_aic sharper than P_base (higher mass on same-family); P_ep flatter (mass spread toward uniform); P_entropy_brake sharper than P_aic; P_loopswap_lite mass only on |Δrank|=1; P_neutral_sink mass only on N. Each normalizes to Σ=1. gran-filter respected (motif source → motif targets only).
- relabel-invariance: shuffle f/z/p magnitudes, fix family/rank/gran → every spectrum identical (shapes read structure only).

## 7. Out of scope
- Multi-slot mutation (S7) and A-pool spectrum gating (S8) consume `spectrum` but don't change its shape.
- P_burst_lite's phase modulation is just period-gating (no f-window); its spectrum is plain `aff` — S2 handles it fully; S5 owns the f-window primitives, not this.
