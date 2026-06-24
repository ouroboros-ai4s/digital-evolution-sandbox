# S4 — 动态方向 (dynamic directions)

> Created: 2026-06-24 · Roadmap sub-project 5/9 (S0→S6→S1→S2→**S4**→S5→S3→S7→S8).
> Faithful translation of the F pool's direction variants. No new gameplay.

## 1. Why
The F pool (`primitive-roster.md` lines 59–82) defines 8 reproduction primitives differing in **dirs** (which neighbor cells offspring scatter to). Current code has F4Nr1 (1 fixed dir) and F4Nr4 (4-dir), both static `dir_bits`. S4 adds the rest, which fall into three direction kinds:

| primitive | dirs | kind |
|---|---|---|
| F4Nr4 | 4-nbr (all) | static |
| FSTACK | {(0,0)} in-place | static (no move) |
| FCLUMP | one axis {±x} or {±y} **by seq-hash** | hash-locked |
| FFRONT | 1 fixed dir **by hash** | hash-locked |
| F4Nr3 | 3 of 4-nbr **by seq-hash** (gap by hash) | hash-locked |
| F4Nr1 | 1 rand of 4-nbr | *roster says "1 rand"* — see §3.3 |
| FDRIFT | 1 rand of 4-nbr **/tick** | per-tick random |

(F8Ar1/Lance Front/Ember Drip/F_TRICKLE/F_SCATTER in the A pool reuse the same kinds — S8.)

## 2. Red lines
- Direction choice is a **deterministic function of the sequence** (hash-locked) or an **honest per-tick random draw** (drift) — neither hand-writes "who expands better". The hash maps sequence→direction-set structurally.
- **Determinism trap (caught by ponytail HOW-2)**: Python's built-in `hash()` is salted per process (`PYTHONHASHSEED`) → different directions each run → breaks reproducibility, fatal for a data-generation sandbox. **Fix: `zlib.crc32("".join(seq).encode())`** — stdlib, deterministic across runs/machines, lightweight (we need a uniform int to select dirs, not crypto).
- Default: F4Nr1/F4Nr4 keep current dirs. F4Nr1's roster "1 rand of 4-nbr" — see §3.3 for the faithful reading (currently it's hardcoded `((-1,0),)`).

## 3. Architecture
### 3.1 Hash-locked dirs computed at mint (no kernel change)
For FCLUMP/FFRONT/F4Nr3, `phenotype()` computes `h = zlib.crc32("".join(seq).encode())` and derives the direction set deterministically:
- FFRONT: `ALL_DIRECTIONS[h % 4]` (1 fixed dir).
- FCLUMP: `h % 2` → x-axis `{(-1,0),(1,0)}` or y-axis `{(0,-1),(0,1)}`.
- F4Nr3: drop direction `h % 4`, keep the other 3.
The result OR's into the **existing `dir_bits`** field. The reproduction kernel's `for (dy,dx) in ALL_DIRECTIONS` loop already reads `dir_bits` — **zero kernel change** for all hash-locked primitives.

### 3.2 Per-tick random (FDRIFT) — the only new kernel logic
FDRIFT draws a fresh 1-of-4 each firing tick. Add a phe-array flag `rand_dir` (bool). In `phase2_reproduce`, for slots with `rand_dir` set, draw a direction index per firing from the kernel `generator` instead of reading `dir_bits`.
- **Do NOT overload `dir_bits==0` for "random"** — `dir_bits==0` already means *no movement* (FSTACK's `{(0,0)}`); the collision would silently turn stackers into drifters. The `rand_dir` flag is separate.
- The per-tick draw uses the kernel's `generator` (world RNG) — kernel's job, not phenotype reading world-state. Red line holds.

### 3.3 F4Nr1 "1 rand of 4-nbr" — faithful reading
The roster writes F4Nr1 dirs as `{(-1,0)} (1 rand of 4-nbr)`. Two readings: (a) hash-locked 1-of-4 (deterministic per strain), (b) per-tick random 1-of-4 like FDRIFT. The current code hardcodes `((-1,0),)` (neither). **Ponytail/技术 default: hash-locked 1-of-4** (same machinery as FFRONT, deterministic, reproducible) — matches "BB0 可选起点" needing a stable identity, and FFRONT's `from: {F4Nr1, FDRIFT}` reads as "F4Nr1 (hash-locked) → FFRONT (hash-locked)". This is a HOW detail, not gameplay; flagged here for review, defaulting to hash-locked. If you want per-tick-random instead, it's a one-flag change.

## 4. Data flow
```
mint(seq) ─► phenotype(): hash-locked F primitive? crc32(seq) → dir set → dir_bits
                          per-tick-random F primitive? set rand_dir flag
phase2_reproduce: rand_dir slot? draw 1-of-4 from generator : use dir_bits (existing loop)
```

## 5. Error handling
- crc32 always returns a valid uint32 → `% 4` / `% 2` always valid index. No empty dir set for hash-locked (always ≥1 dir).
- FSTACK `{(0,0)}` is not in `ALL_DIRECTIONS` (the 4 nbrs); it currently means dir_bits=0 = stay. Confirm the `(0,0)` in-place offspring path emits to the same cell (no roll) — verify against the kernel's roll logic; if `(0,0)` needs explicit handling, add it (one branch: dy=dx=0 → no roll).

## 6. Testing
- Regression: 285+146 green (F4Nr4 unchanged; F4Nr1 changes from hardcoded `(-1,0)` to hash-locked 1-of-4 — **this is a behavior change**; update F4Nr1's known-answer test to the hash-derived dir, document it).
- New: crc32 hash is identical across two processes (determinism — run in subprocess, compare); FFRONT picks 1 stable dir per strain; FCLUMP picks a stable axis; F4Nr3 keeps 3; FDRIFT direction varies across ticks for the same strain (per-tick random); FSTACK stays in-place; `rand_dir` does not collide with FSTACK's dir_bits=0.
- relabel-invariance: the hash reads the *letter sequence* (structural identity), not f/z/p magnitudes — shuffling f/z/p leaves dirs unchanged.

## 7. Out of scope
- Phase-windowed f (FBURST/F_NOVA) — S5. FBURST has static 4-nbr dirs; only its f varies. S4 gives FBURST its dirs; S5 gives its f-window.
- A-pool direction variants (F8Ar1 etc.) — S8 (reuse this machinery).
