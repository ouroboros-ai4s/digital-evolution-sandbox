"""Shared engine-layer config core. Imported by webapp/server.py and
scripts/run_match.py. No webapp imports here — engine layer must stay
free of aiohttp deps so headless callers don't pay the cost."""
from __future__ import annotations
import torch
from des.engine import Engine
from des.registry import _SLOTS, _LOCKED, ALPHABET

# 6 v1 primitives, ordered (front-end dropdown reads the same list via /config)
PALETTE = ["N0", "F4Nr1", "F4Nr4", "P_base", "P_hotspot", "BroadSweep"]

_DEFAULTS = {"grid": 128, "K": 64, "fill": 20, "T": 450, "seed": 0, "z_max": 8.0}


def layout_from_slots(slots: dict) -> tuple[str, ...]:
    """Assemble a 16-position BB0 layout from mutable-slot choices.
    locked positions := _LOCKED; slot positions := given (default 'N0');
    backbone positions := 'N0'. Rejects non-slot indices and off-palette letters."""
    out = []
    for i in range(16):
        if i in _LOCKED:
            out.append(_LOCKED[i])
        elif i in _SLOTS:
            letter = slots.get(i, "N0")
            if letter not in ALPHABET:
                raise ValueError(f"slot {i} = {letter!r} not in palette {PALETTE}")
            out.append(letter)
        else:
            out.append("N0")
    for k in slots:
        if k not in _SLOTS:
            raise ValueError(f"position {k} is not a mutable slot {sorted(_SLOTS)}")
    return tuple(out)


def build_engine_from_config(cfg: dict, device) -> tuple[Engine, dict]:
    """cfg keys: players (list of exactly 4 {slots:{int:str}}), grid, K, fill,
    T, seed, z_max (globals shared across all four factions). Returns (engine,
    resolved) where resolved has defaults filled, a 'players' echo, and a 4-tuple
    'layouts'. Each layout -> layout_from_slots -> validate_bb0_layout enforces the
    template structure (red-line 4: same template, slot choices may differ); only
    primitive letters + run knobs ever enter here (red-line 3)."""
    players = cfg.get("players")
    if not isinstance(players, list) or len(players) != 4:
        raise ValueError(f"config must have exactly 4 players, got {players!r}")
    layouts = tuple(
        layout_from_slots({int(k): v for k, v in (p.get("slots") or {}).items()})
        for p in players
    )
    resolved = dict(_DEFAULTS)
    for k in _DEFAULTS:
        if cfg.get(k) is not None:
            resolved[k] = cfg[k]
    resolved["players"] = players
    resolved["layouts"] = layouts
    g = int(resolved["grid"])
    eng = Engine(H=g, W=g, K=int(resolved["K"]), seed=int(resolved["seed"]),
                 device=device, z_max=float(resolved["z_max"]),
                 fill_per_cell=int(resolved["fill"]), layouts=layouts)
    return eng, resolved


def pick_device(device=None, force_cpu: bool = False) -> torch.device:
    """Behavior-preserving fold of webapp/server.py:_device and the inline
    device line in scripts/run_batch.py. The server passes its launch device
    arg straight through (`device=...`), preserving today's `cuda:N` selection;
    `force_cpu=True` forces CPU (matches run_batch.py --cpu); `device=None`
    auto-selects cuda-if-available."""
    if force_cpu:
        return torch.device("cpu")
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_match_result(eng, parquet_path: str) -> dict:
    """Final-tick match result. Reads the engine's in-memory final world
    snapshot (NOT a parquet read-back — the parquet footer isn't written
    until Recorder.close()). Reuses webapp.readouts.compute_readouts as
    the single source for total/share/n2/d_max definitions.
    Returns: {"path": str, "ticks": int, "final": <compute_readouts return>}.
    No 'winner' field — the sandbox stays goal-free; the AI player decides."""
    from webapp.readouts import compute_readouts

    cnt = eng.world.count.cpu()
    sid = eng.world.strain_id.cpu()
    fac = eng.world.faction.cpu()
    nz = torch.nonzero(cnt > 0, as_tuple=False)
    ys = nz[:, 0].tolist()
    xs = nz[:, 1].tolist()
    ks = nz[:, 2]
    sids = sid[nz[:, 0], nz[:, 1], ks].tolist()
    facs = fac[nz[:, 0], nz[:, 1], ks].tolist()
    cnts = cnt[nz[:, 0], nz[:, 1], ks].tolist()
    strains = [".".join(eng.table.sequence_of(int(s))) for s in sids]
    final = compute_readouts(xs, ys, strains, facs, cnts)
    return {"path": parquet_path, "ticks": int(eng.T), "final": final}
