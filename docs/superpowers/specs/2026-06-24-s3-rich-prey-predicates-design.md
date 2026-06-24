# S3 — 富猎物谓词 (rich prey predicates)

> Created: 2026-06-24 · Roadmap sub-project 7/9 (S0→S6→S1→S2→S4→S5→**S3**→S7→S8). Depends on S6 (predicate-bit scheme + motif blocks) and S1 (vis bit value).
> Faithful translation of the Z pool's non-family prey clauses. No new gameplay.

## 1. Why
S6 established the predicate-bit *scheme* and the motif/family bits; S3 fills the remaining **threshold predicates** the Z pool's narrow specialists hunt by (`primitive-roster.md` Z pool):

| predator | prey clause (verbatim) |
|---|---|
| Crest Bite | `{s: fam(s)=F ∧ f_s≥0.5}` |
| Hotspot Snipe | `{s: fam(s)=P ∧ p_add,s≥0.05}` |
| Mirror Fang | `{s: fam(s)=Z ∧ z_s≤0.45 ∧ \|prey_s\|≥2}` |
| Void Bite (A) | `{s: fam(s)=N ∧ vis_s≤0.20}` |

Plus the motif clauses S6 already encoded as bits but S3 verifies population: `{motif∋F/P/Z/N}` (Ambush Coil/Burst Leech/Clade Snare/Frame Pincer), `{ℓ≥3 motif∋F/P/Z}` (Lineage Reaper/Coil Cinch/Idiotype Lance/Predator Lock). Current code: family-only bitmask, no thresholds.

## 2. Red lines
- The thresholds (f≥0.5, p_add≥0.05, z≤0.45, |prey|≥2, vis≤0.20) are **prey-selection predicates the predator declares** — "what do I hunt", structural. They are NOT "who is strong" — a predator hunting high-f prey doesn't make high-f prey weaker by fiat; it creates a frequency-dependent pressure that the dynamics resolve. The threshold values are verbatim from the roster, not chosen here.
- A strain's `feature_mask` is a pure function of its own phenotype (its f, p_add, z, prey-count, vis are all already computed). No world-state.
- Antagonism kernel **unchanged**: still `(prey_mask[i] & feature_mask[j]) != 0`. S3 only sets more bits.
- Default: no v1 predator uses these clauses → byte-identical. All threshold-hunters dormant until minted.

## 3. Architecture
All work is in **`phenotype()`** (set `feature_mask` bits on the prey side) and the **prey-clause→`prey_mask`** mapping (predator side). No kernel change.

### 3.1 feature_mask population (prey side)
In `phenotype()`, after computing f/p_x/z_raw/prey/vis, set the strain's threshold feature bits:
- `FEAT_F_HI` if the strain has an F letter with `f_s ≥ 0.5` (per-letter, from `_F` table — note: the strain's *stacked* f vs a *letter's* f. Roster says `fam(s)=F ∧ f_s≥0.5` where s is a **primitive**, so the bit reflects "carries an F primitive with f≥0.5", per-letter, not stacked). 
- `FEAT_P_HI` if it carries a P letter with `p_add ≥ 0.05`.
- `FEAT_Z_GENERALIST` if it carries a Z letter with `z_s ≤ 0.45 ∧ |prey_s| ≥ 2` (BroadSweep z=0.40 prey={F,Z} qualifies — Mirror Fang's intended target).
- `FEAT_N_LOWVIS` if it carries an N letter with `vis_s ≤ 0.20` (value source = S1's vis; S6 reserved the bit).
- motif bits (`FEAT_MOTIF_F/P/Z/N`, `FEAT_L3MOTIF_F/P/Z`) — S6 already sets these from `motif_blocks` + family; S3 confirms ℓ≥3 uses the block span length.

> Per-letter vs stacked: the roster's threshold clauses read `fam(s)` for a primitive s, so the feature bit = "carries such a primitive". This is unambiguous in the roster and matches the antagonism model (a predator detects a prey *carrying* the targeted feature). Documented to avoid a stacked-value misread.

### 3.2 prey_mask population (predator side)
Each predator's prey clause → the predicate bit(s) it matches. `PREY_CLAUSE = {predator_letter: bitmask}`. E.g. Crest Bite → `FEAT_F_HI`; Mirror Fang → `FEAT_Z_GENERALIST`; Ambush Coil → `FEAT_MOTIF_F`. Family-prey predators (BroadSweep `{F,Z}`) → OR of family bits, as S6.

## 4. Data flow
```
mint(seq) ─► phenotype():
   prey side:    set FEAT_* bits from own f/p_add/z/|prey|/vis/motif_blocks
   predator side: prey clause ─► PREY_CLAUSE[letter] ─► prey_mask
phase1_antagonism: (prey_mask[i] & feature_mask[j]) != 0   [UNCHANGED]
```

## 5. Error handling
- Total predicate-bit count must fit int64 (S6 asserts; S3 adds ~5 bits → still ≤ ~15). Re-assert.
- `|prey_s|` (prey-clause cardinality of a Z letter) — derivable from the registry `_Z` prey-family tuple length; compute at module load per Z letter, no runtime cost.
- A strain carrying multiple qualifying letters: bits are OR'd (carries ≥1 → bit set). Correct.

## 6. Testing
- Regression: 285+146 green (no v1 predator uses threshold clauses; feature bits added but unmatched → no kill change).
- New: a strain with F4Nr4 (f=0.50) sets FEAT_F_HI (0.50≥0.5 boundary — test the ≥); F4Nr1 (f=0.30) does not; P_hotspot (p_add=0.05) sets FEAT_P_HI (boundary); BroadSweep sets FEAT_Z_GENERALIST (z=0.40≤0.45, |prey|=2≥2); a low-vis N strain sets FEAT_N_LOWVIS; Crest Bite's prey_mask matches FEAT_F_HI strains only; Mirror Fang matches generalist-Z only. Antagonism known-answer unchanged for family-only predators.
- relabel-invariance: the thresholds read the strain's *own structural* f/p_add/z (which come from the registry tables, not per-species) — shuffling the table reassigns which letters qualify, but a fixed strain's bits follow its letters' registry values deterministically. Test: fix the registry, the bit assignment is a pure function of the sequence.

## 7. Out of scope
- The A-pool predators that use these clauses (Predator Lock, Ambush Venom, Void Bite, Sweep Surge, Coil Null) — minted in S8; S3 makes their clauses *matchable* now.
- vis aggregate computation (S1 owns it; S3 reads the per-letter vis≤0.20 fact).
