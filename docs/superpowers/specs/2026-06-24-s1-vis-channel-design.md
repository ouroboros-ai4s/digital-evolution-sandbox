# S1 — vis 可见度通道

> Created: 2026-06-24 · Roadmap sub-project 3/9 (S0→S6→**S1**→S2→S4→S5→S3→S7→S8).
> Faithful translation of the roster's N-pool vis column + the vis-weighted N-hunters. No new gameplay.

## 1. Why
The roster's entire N pool carries a `vis ∈ [0,1]` (被猎可见性): N0=0.20, N1=0.40, N2=0.70, N3=0.15, N4=0.35, N5=0.00, N6=1.00, N7=0.10 (`primitive-roster.md` N pool). vis is the only output of N primitives (`f=z=p_add=0`). It does nothing on its own — but two Z hunters target N with hit probability **weighted by vis**, making "hide in the neutral zone" a real, contestable strategy:
- **Scatter Nip**: `prey={N}`, `p_hit = (1/L)·Σ_{i∈N} vis_i` (high-vis N gets culled).
- **Ghost Spike**: `prey={N}`, `p_hit = (1/L)·Σ_{i∈N} (1−vis_i)` (mirror: low-vis N gets culled).

Current code has no vis field and N0's only role is inert filler.

## 2. Red lines
- vis is a global per-primitive registry value (`VIS[letter]`), never per-species. Pure function of the sequence.
- The kernel multiplying kills by `p_hit` reads the prey strain's vis profile (a phenotype value) — not world-state leaking into phenotype. The phenotype stores the vis aggregate; the kernel applies it.
- Default game: the 6 v1 letters are unaffected except N0 gains `vis=0.20` (already its roster value); since no v1 Z hunter has `prey={N}` weighted by vis, default runs are byte-identical. Scatter Nip/Ghost Spike are dormant until minted.

## 3. Architecture
### 3.1 vis registry + phenotype aggregate
- `VIS = {letter: float}` in `registry.py` (8 N values above; non-N letters vis=0, unused).
- `phenotype()` computes the strain's **vis profile aggregate** the hunters need: `vis_sum = Σ_{i: fam=N} vis_i` and `n_count = #{i: fam=N}` (so a hunter can form `(1/L)Σ vis` and `(1/L)Σ(1−vis) = (n_count − vis_sum)/L` over the prey's N positions; L = the already-known sequence length, v1 fixed-16 / `len(seq)`). Store `vis_sum` (float) + `n_count` (int) as Phenotype fields → two new phe-arrays bulk-uploaded.

### 3.2 vis-weighted hit in antagonism kernel
Currently every valid (attacker i, prey j) pairing kills `round(count[i]·z_eff)`. For a vis-weighted hunter, scale that by `p_hit` computed from prey j's vis profile:
- Add a per-attacker **vis-mode** flag as a small int array (an attacker-side behavior flag, distinct from and not part of S6's predicate-bit encoding scheme): `0` = none (all current primitives), `1` = vis-weighted (Scatter Nip), `2` = inverse-vis-weighted (Ghost Spike).
- In `phase1_antagonism`, after `raw_kill`, scale ONLY vis-mode attacker rows by `p_hit_j` computed from prey j: mode 1 → `vis_sum_j / L`; mode 2 → `(n_count_j − vis_sum_j) / L`. Mode-0 rows are left untouched — they SKIP the `p_hit` multiply entirely (no global `×1.0` on the rounded-int kill), so the default kernel path is bit-identical. Everything downstream (proportional cap, self-loss) unchanged.
- `L` (prey sequence length) = the already-known v1 fixed-16 / `len(seq)`. No new array; a per-strain length array belongs to the future variable-length spec, not here.

### 3.3 vis≤0.20 predicate bit (handoff to S3)
S6 reserved the `fam=N ∧ vis≤0.20` predicate bit for Void Bite (A pool). S1 fills its **value source**: `phenotype()` sets that feature bit if the strain has any N position with vis≤0.20. (Void Bite itself is an A primitive minted in S8; S1 just makes the bit computable now.)
- **Default N0:** N0's roster vis=0.20 meets the inclusive `≤0.20` (the `≤` matches roster Void Bite's `vis_s≤0.20`, not `<`), so this bit is SET for the default-BB0 N0 strain. Harmless: no default primitive consumes it (Void Bite is dormant until S8).

## 4. Data flow
```
mint(seq) ─► phenotype(): vis_sum=Σ_{N}vis, n_count=#N; set vis≤0.20 feature bit
phase1_antagonism: raw_kill ─► ×p_hit(prey vis profile, attacker vis-mode) ─► cap ─► losses
```

## 5. Error handling
- vis outside [0,1] in the registry: assert at module load.
- Empty N profile (n_count=0): `p_hit` denominator is L>0; `vis_sum=0` → mode-1 p_hit=0 (no N to hit, no kill), mode-2 p_hit=0 too (n_count=0). Correct (a hunter of N finds nothing). `(n_count − vis_sum)` is mathematically ≥0 (every N vis ≤ 1.0, max N6=1.00), so no guard is needed; if float epsilon is a concern, clamp with a plain `max(0.0, ·)` — not alive-masking.

## 6. Testing
- Regression: 285+146 green (only N0 gains its roster vis=0.20; no behavior change since no default hunter is vis-weighted).
- New: `phenotype()` vis_sum/n_count correct for sample N-bearing strains; Scatter Nip kills scale with prey vis (high-vis prey dies faster); Ghost Spike scales inverse; mode-0 attacker identical to current kernel; vis≤0.20 feature bit set correctly.
- relabel-invariance: shuffle f/z/p (not vis — vis is a structural channel here); vis-weighted results unchanged → confirms vis path doesn't read f/z/p magnitude.

## 7. Out of scope
- Scatter Nip/Ghost Spike entering the spectrum as mutable targets (they're Z primitives; S8 and the motif/spectrum specs own reachability). S1 only makes them *function* when present.
- The threshold predicate *values* f≥0.5 etc. (S3).
