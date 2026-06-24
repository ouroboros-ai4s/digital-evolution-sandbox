# S0 — 统一系统 + CLI 对局产数据工具

> Created: 2026-06-24
> Sub-project S0 of the 9-spec "implement-the-new-roster" roadmap (S0→S6→S1→S2→S4→S5→S3→S7→S8).
> Scope this spec: **unify the two entry points into one system and give AI a headless front door to configure-and-play matches.** The engine primitive set is NOT touched here (still the 6-primitive v1 subset); the 6→68 expansion is S1–S8.

## 1. Why (use-need, verbatim from user)

The sandbox has two usage scenarios that are **the same action** — configure 4 factions' starting strains → run → harvest data:

- **人类手动查看效果 (web)**: click 4 players' slots in dropdowns → WebSocket live run. *Already delivered* (viz v2, PR #2, 146 tests green).
- **AI 自动化批量产数据 (CLI)**: "4 个 subagent 以真的赢下游戏为目的, 现场配置他们的株" — four AI agents each configure their own faction's strain aiming to **win the match**, run headless, produce data.

The web's `build_engine_from_config(players:[{slots}×4], grid, K, …)` is *already exactly the config surface AI needs*. The only gap: AI has no headless front door — `scripts/run_batch.py` bakes a fixed symmetric config (128²/K64/fill20/T450/seeds[0–3]) and cannot take per-faction strain choices. This spec adds that door and lifts the shared config core so both doors use one schema.

## 2. Red lines (unchanged — this spec touches none of them)

- **无私货 / no hand-written "who is strong"**: same template, same fixed G→P, no `f_baseline[species]`. The *player* (human or AI) picks legal slot configs to set their own goal — exactly as a chess player does. The player is external; it never edits the rules. This is **异起点 (red-line 4, opened in v2: different genotype, same template)**, NOT **异角色系统 (per-faction asymmetric K / mutation-rate / mechanics — still HARD-GATE)**. No new gate.
- **Outcome constants stay locked in the registry**: μ / z_max / δ / p_max / α / κ / β are NOT CLI args. Run *dimensions* (grid / K / fill / T / seed) are legitimate inputs; outcome constants are not. The CLI exposes only player slot choices + run dimensions.
- **Playground isolation**: web live runs still write `data/playground/`; CLI match runs write `data/runs/`. Unchanged.
- **Single-source readouts**: match results reuse `webapp/readouts.py:compute_readouts` verbatim. No second definition of "what total/share/n2 mean".
- **No win-judging baked into the sandbox**: the runner reports pure observational aggregates (per-faction total / share / occupied cells / distinct strains). It does NOT compute "who won" — that would be an internal goal = red-line violation. The win call is left to the AI player reading the data.

## 3. Architecture

Three pieces. Two are moves, one is new.

### 3.1 Lift the shared config core (pure relocation)

Move from `webapp/server.py` into a new engine-layer module **`src/des/run.py`**:

- `PALETTE` (the ordered 6-letter list)
- `_DEFAULTS` (grid/K/fill/T/seed/z_max)
- `layout_from_slots(slots) -> tuple[str,…]`
- `build_engine_from_config(cfg, device) -> (Engine, resolved)`
- `pick_device(force_cpu=False) -> torch.device` (the one true byte-identical dup; folds `server.py:_device` and `run_batch.py`'s inline device line)

`webapp/server.py` then **imports** these instead of defining them. Behavior byte-identical → the 146 web tests are the regression lock. `webapp/server.py` keeps its own async WS loop (irreducibly different: `await ws.send_json` + `if ws.closed: break` — confirmed by ponytail audit; a shared loop would async-infect the sync batch path for zero gain).

> Why engine-layer and not `webapp/`: the CLI runner must not import `webapp` (aiohttp dep, and `webapp` already absolute-imports the engine — a back-import would invert the dependency direction). `src/des/run.py` is the natural shared home; both `webapp/server.py` and `scripts/run_match.py` depend on the engine, never on each other.

### 3.2 New CLI: `scripts/run_match.py` (AI player front door)

```
PYTHONPATH=src python scripts/run_match.py --config match.json [--cpu] [--out data/runs]
```

- Reads one JSON file whose schema is **identical to the web WS `config` payload**:
  ```json
  {"players": [{"slots": {"0": "F4Nr4", "2": "P_hotspot"}}, {…}, {…}, {…}],
   "grid": 128, "K": 64, "fill": 20, "T": 450, "seed": 0}
  ```
  (Exactly 4 players; slot keys are mutable-slot indices as strings; values from `PALETTE`. Same validation as web: each layout → `layout_from_slots` → `validate_bb0_layout`; a bad config exits non-zero with the ValueError message, mirroring the web `{"event":"error"}` path.)
- `pick_device` → `build_engine_from_config` → `Engine.run(T, recorder=rec)` headless → one parquet under `data/runs/`.
- Prints the match result (§3.3) to stdout as JSON so an orchestrating AI can parse it.

One JSON schema, two front doors: WS for humans, JSON+CLI for AI. web↔CLI configs are interchangeable by construction (an AI can hand a human's config to the CLI and vice-versa).

### 3.3 Match result (pure aggregation, reuses readouts)

After the run, call `compute_readouts` on the final tick's non-empty records (same inputs the recorder already materializes). Emit:

```json
{"path": "data/runs/…parquet", "ticks": 450,
 "final": {"total": …, "occupied_cells": …, "distinct_strains": …,
           "n2": …, "d_max": …, "faction_share": {"0": …, "1": …, "2": …, "3": …}}}
```

`faction_share` is the per-faction outcome signal the AI player reads to judge win/loss. No "winner" field — the sandbox stays goal-free. `final.per_faction[f]` = `compute_readouts` called on faction `f`'s subset of the same final-tick records (one group-by, no new definition) → gives the AI finer per-faction signal (each faction's own total/occupied/distinct). Reuses the single readouts definition; adds no new metric.

### 3.4 `scripts/run_batch.py` stays as-is

It is the **canonical symmetric-default data producer** (four identical BB0 + seed sweep) — a reproducible anchor, still useful, untouched. `run_match.py` is the asymmetric AI-player door. Both are thin wrappers over the same shared core (`src/des/run.py` + `Engine` + `Recorder`); only the device line in `run_batch.py` changes (→ `pick_device`).

## 4. Data flow

```
match.json ──► run_match.py ──► pick_device ─┐
                                             ├─► build_engine_from_config ─► Engine.run(T, rec) ─► data/runs/*.parquet
              (validate: 4 layouts ──────────┘                                      │
               layout_from_slots → validate_bb0_layout)                             └─► final tick ─► compute_readouts ─► stdout JSON

web (unchanged): WS config ─► [same build_engine_from_config] ─► async loop ─► data/playground/ + live frames
```

## 5. Error handling

- Config missing / not 4 players / bad slot index / off-palette letter / both no-slots: same `ValueError`s already raised by `layout_from_slots` + `build_engine_from_config`; `run_match.py` catches, prints to stderr, exits 1.
- File not found / malformed JSON: exit 1 with a clear message.
- Recorder writer-thread death: already surfaced by `Recorder._check_thread`; the runner lets it propagate (a crashed writer = data loss, must not be swallowed).

## 6. Testing

- **Regression lock (the move)**: existing 146 web tests + 285 engine tests stay green after lifting the core — proves byte-identical behavior. `tests/test_server_config.py` (which exercises `layout_from_slots`/`build_engine_from_config`) updated only for the new import path, not for behavior.
- **New `tests/test_run_match.py`**:
  - valid 4-player JSON → engine assembled with 4 distinct layouts; headless run of a few ticks produces a parquet with >0 rows.
  - bad config (3 players / off-palette / non-slot index) → exits non-zero, no parquet written.
  - match result JSON has the documented keys; `faction_share` sums to ~1.0; `compute_readouts` output matches a direct call on the same final records (single-source check).
  - web config ↔ CLI config interchangeability: a config dict accepted by the WS path is accepted by `run_match` and vice-versa (shared `build_engine_from_config`).

## 7. Out of scope (explicit)

- Primitive-set expansion 6→68 (that's S1–S8).
- Any engine/kernel/registry change.
- Config-file batch sweeps / multi-config-per-invocation (Unix philosophy: one config per call; sweep with an outer shell loop if ever needed — YAGNI until asked).
- A shared sync/async loop (rejected by ponytail audit).
- Win-judging logic (red-line: stays with the player).
