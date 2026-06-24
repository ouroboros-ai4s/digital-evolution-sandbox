# S5 — 相位窗 f (phase windows)

> Created: 2026-06-24 · Roadmap sub-project 6/9 (S0→S6→S1→S2→S4→**S5**→S3→S7→S8).
> Faithful translation of the phase-modulated reproduction primitives. No new gameplay.

## 1. Why
Three roster primitives modulate f by a **time window** relative to birth, creating burst/rest rhythm (博弈节奏):
- **FBURST** (`primitive-roster.md` lines 81–82): `(T−birth) mod 12 < 2 ⇒ f=0.55, else f=0.05`; dirs=4-nbr, p_leave=0.20, T=2.
- **F_NOVA** (A pool, lines 215–216): `(T−birth) mod 20 < 1 ⇒ f=0.85, else 0.05`; dirs=4-nbr, p_leave=0.50, T=2.
- **P_burst_lite** (line 120): period-gated pulse — but its *f* isn't windowed (it's a P primitive, no f); its "相位窗突变爆发" = the existing period clock. **No new mechanism for P_burst_lite** — it's plain `aff` spectrum + slow T, fully handled by S2. Listed here only to disclaim it.

So S5 is really FBURST + F_NOVA: f that depends on `(T − birth_tick) mod W < k`.

## 2. Red lines
- The window params (W, k, f_hi, f_lo) are global per-primitive registry values — pure function of the sequence. The phenotype stores only the params.
- **The kernel computes the live f from `birth_tick` + `T`** — identical in shape to the existing `fires_this_tick((T−birth)%period==0)`. This is the kernel reading its clock, NOT the phenotype reading world-state. Red line holds (confirmed HOW-1).
- Default: all current primitives are static → window defaults make them byte-identical (see §3).

## 3. Architecture (HOW-1, ponytail-minimal)
The phase window is the *same shape* as `fires_this_tick`. Phenotype stores 4 window params; the kernel does one vectorized `where`.

- Add phe-arrays `f_hi, f_lo, burst_w, burst_k`. **Static primitives store `f_hi=f_lo=f`, `burst_w=1, burst_k=1`** → `(T−birth)%1 = 0 < 1` always true → `f = f_hi = f`. The 99% static case is byte-identical, no branch.
- In `phase2_reproduce`, replace `f = phe["f"][sid_long]` with:
  ```
  on = ((T - birth_tick) % burst_w) < burst_k          # [H,W,K], same form as fires_this_tick
  f  = torch.where(on, f_hi, f_lo)
  ```
  Everything downstream (binom offspring, mutation split, roll) unchanged. The old `f` array has consumers beyond this kernel (`phenotype_cache.py`, `kernels/reproduction.py`), so **keep `f` as an alias of `f_hi`** rather than dropping it — the static-default identity `f == f_hi` keeps every existing reader byte-identical.

No per-tick cache invalidation, no per-individual phenotype — the cache stores params, the kernel applies the clock.

## 4. Data flow
```
mint(seq) ─► phenotype(): f_hi/f_lo/burst_w/burst_k (static: f_hi=f_lo=f, w=k=1)
phase2_reproduce: on = (T-birth)%burst_w < burst_k ; f = where(on, f_hi, f_lo) ─► binom offspring (unchanged)
```

## 5. Error handling
- burst_w=0 would divide-by-zero in `%`; `burst_w` is a NEW array, so the existing `clamp(min=1)` that guards `fires_this_tick`'s `period` does NOT cover it automatically — apply the same `clamp(min=1)` explicitly to `burst_w`. The static default is already 1.
- Multi-F strain with mixed windowed + static F letters: f stacks via `1−Π(1−f_i)` as today, but each f_i is now its own windowed value. Compute each letter's live f (windowed), then stack. **This is the one subtlety**: the stack must happen on *post-window* f, per letter — so the window resolution moves into the per-letter accumulation in `phenotype()`'s consumer, or the kernel stacks per-letter. Lazy correct: since phenotype already collapses multi-F into one stacked `f`, S5 resolves at the **dominant-F** level (mirroring the existing `dominant_p` analogue): pick the windowed params of the dominant F letter (highest f, ties by first occurrence). **Default: with exactly one windowed F letter present, that letter's params apply; when two or more windowed F letters coexist, the dominant-F letter's window params apply to the stacked f (approximation, flagged); full per-letter windowed stacking deferred.** Motif/multi-F is dormant in default BB0, so this never fires in the symmetric game.

## 6. Testing
- Regression: 285+146 green (static defaults f_hi=f_lo=f, w=k=1 → identical f every tick).
- New: FBURST f=0.55 for ticks where `(T−birth)%12<2`, else 0.05 (test several birth offsets); F_NOVA f=0.85 in the 1-tick window per 20, else 0.05; a static strain's f constant across ticks; the `where` produces the old value when w=k=1.
- relabel-invariance: window params are structural; shuffling other primitives' f/z/p doesn't change FBURST's window.

## 7. Out of scope
- FBURST/F_NOVA dirs (S4 — 4-nbr static) and p_leave (already supported).
- P_burst_lite spectrum (S2). Full per-letter windowed-f stacking for multi-windowed-F strains (deferred; dormant in default).
