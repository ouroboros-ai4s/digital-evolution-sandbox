# S4 — 动态方向 (dynamic directions)

> Created: 2026-06-24 · Roadmap sub-project 5/9 (S0→S6→S1→S2→**S4**→S5→S3→S7→S8).
> Faithful translation of the F pool's direction variants. No new gameplay.

## 1. Why
The F pool (`primitive-roster.md` lines 59–82) defines 8 reproduction primitives differing in **dirs** (which neighbor cells offspring scatter to). Current code has F4Nr1 (1 fixed dir) and F4Nr4 (4-dir), both static `dir_bits`. S4 adds the rest, which fall into three direction kinds:

| primitive | dirs | kind |
|---|---|---|
| F4Nr4 | 4-nbr (all) | static |
| FSTACK | {(0,0)} in-place | static (deposits in source cell — needs new kernel branch, see §3.2) |
| FCLUMP | one axis {±x} or {±y} **by seq-hash** | hash-locked |
| FFRONT | 1 fixed dir **by hash** | hash-locked |
| F4Nr3 | 3 of 4-nbr **by seq-hash** (gap by hash) | hash-locked |
| F4Nr1 | 1 of 4-nbr **by seq-hash** | hash-locked (see §3.3) |
| FDRIFT | 1 rand of 4-nbr **/tick** | per-tick random |

(F8Ar1/Lance Front/Ember Drip/F_TRICKLE/F_SCATTER in the A pool reuse the same kinds — S8.)

## 2. Red lines
- Direction choice is a **deterministic function of the sequence** (hash-locked) or an **honest per-tick random draw** (drift) — neither hand-writes "who expands better". The hash maps sequence→direction-set structurally.
- **Determinism trap (caught by ponytail HOW-2)**: Python's built-in `hash()` is salted per process (`PYTHONHASHSEED`) → different directions each run → breaks reproducibility, fatal for a data-generation sandbox. **Fix: `zlib.crc32("\x1f".join(seq).encode())`** — stdlib, deterministic across runs/machines, lightweight (we need a uniform int to select dirs, not crypto). The `\x1f` (unit separator, not in the alphabet) delimiter prevents multi-char tokens from ambiguously concatenating (`"N0"+"F4Nr4"` can't collide with another grouping).
- Default: F4Nr4 keeps current dirs. F4Nr1 changes from the v1 placeholder `((-1,0),)` to hash-locked 1-of-4 (RESOLVED — see §3.3); this re-baselines the default game, accepted by user 2026-06-24.

## 3. Architecture
### 3.1 Hash-locked dirs computed at mint (no kernel change)
For FCLUMP/FFRONT/F4Nr3, `phenotype()` computes `h = zlib.crc32("\x1f".join(seq).encode())` and derives the direction set deterministically:
- FFRONT: `ALL_DIRECTIONS[h % 4]` (1 fixed dir).
- FCLUMP: `h % 2` → x-axis `{(-1,0),(1,0)}` or y-axis `{(0,-1),(0,1)}`.
- F4Nr3: drop direction `h % 4`, keep the other 3.
The result OR's into the **existing `dir_bits`** field. The reproduction kernel's `for (dy,dx) in ALL_DIRECTIONS` loop already reads `dir_bits` — **zero kernel change** for all hash-locked primitives.

### 3.2 New kernel logic: per-tick random (FDRIFT) + in-place deposit (FSTACK)
Two F primitives need behavior the current kernel does **not** have. The kernel iterates `for (dy,dx) in ALL_DIRECTIONS` (the 4 neighbors only) and emits offspring only for set direction bits — so `dir_bits==0` today emits **nothing**, it does NOT deposit in the source cell. Both new paths add a bool to the **frozen `Phenotype` dataclass** (`types.py`) and a corresponding array in the phenotype-cache build; both are then read in `phase2_reproduce`.

**FDRIFT — per-tick random.** Draws a fresh 1-of-4 each firing tick. Add `rand_dir` (bool). In `phase2_reproduce`, for slots with `rand_dir` set, draw a direction index per firing from the kernel `generator` instead of reading `dir_bits`.
- The per-tick draw uses the kernel's `generator` (world RNG) — kernel's job, not phenotype reading world-state. Red line holds.

**FSTACK — in-place deposit.** FSTACK (`f=0.60`, `dirs={(0,0)}`) means "stack offspring in the source cell" (high local reproduction, no dispersal). `(0,0)` is not in `ALL_DIRECTIONS`, so the existing direction loop never emits for it → FSTACK currently fires but disperses nowhere (zero offspring), a silent bug. Add `in_place` (bool) and an explicit no-roll emit branch: deposit `scattered`/`mut` into the source cell (no `torch.roll`).
- **Do NOT overload `dir_bits==0`** for either case — `dir_bits==0` currently means "emit nowhere"; overloading it for "random" would silently turn stackers into drifters, and overloading it for "in-place" is ambiguous with FDRIFT. The `rand_dir` and `in_place` flags are separate, explicit bools.

### 3.3 F4Nr1 "1 of 4-nbr by seq-hash" — RESOLVED (2026-06-24)
F4Nr1 = **hash-locked 1-of-4-neighbors via the same crc32 seq-hash as §3.1** (`ALL_DIRECTIONS[h % 4]`, one fixed direction, minted once per strain, constant for the whole lineage). Cross-strain this looks random ("四邻随机选一"); same-strain it is constant, which makes design.md's "单向渗透" correct. It is NOT per-tick random (only FDRIFT is per-tick, marked `/tick`). No special case — F4Nr1 ORs its hash-derived single direction into `dir_bits` at mint exactly like FFRONT.

This resolves an earlier ambiguity: the roster previously carried a literal `{(-1,0)}` (an engine v1 placeholder) alongside the description. Both truth docs now agree — roster L59-62 reads `{d_hash}(1 of 4-nbr, by seq-hash)` with a disambiguation note citing design L276 ("4N随机1") + L291 ("方向由序列hash定、出生锁死") + the "单向渗透" description; the v1 `{(-1,0)}` north-only is retired.

> **Baseline impact (accepted by user 2026-06-24).** Changing F4Nr1 from the v1 placeholder `{(-1,0)}` to hash-locked 1-of-4 **changes default-game dynamics** → the first 837MB baseline is no longer byte-reproducible and must be re-baselined; the 285 tests that lock F4Nr1 to north-only must be updated. The user accepted the re-baseline. Affected regression tests to be updated in the implementation phase.

### 3.4 Register the F-pool rows S4 owns
Per S6 §7 ("each later spec adds its own primitives' rows"), S4 adds these 5 primitives to `_F` + `ALPHABET` (family=`F`) + the `gran` table. Values are verbatim from `primitive-roster.md` L60-78; gran cross-checked against design.md F-table L285-289 (roster and design agree). Each row's direction logic (§3.1/§3.2) ships **with** its row — rows and direction handling are one deliverable, wired together:

| primitive | f | p_leave | period | gran | direction spec |
|---|---|---|---|---|---|
| FSTACK | 0.60 | 0.00 | 3 | residue | in_place (§3.2) |
| FCLUMP | 0.45 | 0.10 | 6 | motif | hash-locked axis (§3.1) |
| FFRONT | 0.50 | 0.25 | 4 | motif | hash-locked 1-fixed (§3.1) |
| F4Nr3 | 0.40 | 0.12 | 5 | residue | hash-locked 3-of-4 (§3.1) |
| FDRIFT | 0.15 | 0.30 | 2 | residue | per-tick rand_dir (§3.2) |

(F4Nr1/F4Nr4 already in `_F`; F4Nr1's dirs change to hash-locked per §3.3. FBURST is NOT here — deferred entirely to S5 per §7.)

## 4. Data flow
```
mint(seq) ─► phenotype(): hash-locked F primitive (FCLUMP/FFRONT/F4Nr3/F4Nr1)? crc32(seq) → dir set → dir_bits
                          per-tick-random F primitive (FDRIFT)? set rand_dir flag
                          in-place F primitive (FSTACK)? set in_place flag
phase2_reproduce: in_place slot? emit to source cell (no roll)
                  rand_dir slot? draw 1-of-4 from generator
                  else? use dir_bits (existing loop)
```

## 5. Error handling
- crc32 always returns a valid uint32 → `% 4` / `% 2` always valid index. No empty dir set for hash-locked (always ≥1 dir).
- FSTACK `{(0,0)}` in-place deposit is specified in §3.2 (explicit `in_place` flag + no-roll emit branch); it must NOT rely on `dir_bits==0`, which emits nothing.

## 6. Testing
- Regression: F4Nr4 unchanged. F4Nr1 changes from the v1 placeholder `(-1,0)` to hash-locked 1-of-4 — **this is a behavior change that re-baselines the default game** (accepted by user 2026-06-24, see §3.3): the 285 tests locking F4Nr1 to north-only and any byte-identical-baseline assertions must be updated to the hash-derived dir.
- New: crc32 hash is identical across two processes (determinism — run in subprocess, compare); FFRONT picks 1 stable dir per strain; FCLUMP picks a stable axis; F4Nr3 keeps 3; FDRIFT direction varies across ticks for the same strain (per-tick random); FSTACK stays in-place; `rand_dir` does not collide with FSTACK's dir_bits=0.
- relabel-invariance: the hash reads the *letter sequence* (structural identity), not f/z/p magnitudes — shuffling f/z/p leaves dirs unchanged.

## 7. Out of scope
- Phase-windowed f (FBURST/F_NOVA) — **S5, entirely**. FBURST has no static f (only S5's phase-window f), so an S4 row with dirs-but-no-f would be a half-built, non-functional primitive. Its trivial static 4-nbr dirs land in S5 alongside its f-window, not here.
- A-pool direction variants (F8Ar1 etc.) — S8 (reuse this machinery).
