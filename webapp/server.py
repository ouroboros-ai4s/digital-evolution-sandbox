"""aiohttp viz server for the DES engine. This module's pure helpers
(layout_from_slots / build_engine_from_config) are unit-tested without the
event loop; the aiohttp app + WebSocket live loop are added on top."""
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
    """cfg keys (all optional): slots{int:str}, grid, K, fill, T, seed, z_max.
    Returns (engine, resolved_cfg) where resolved_cfg has defaults filled and a
    'layout' tuple. Engine -> init_factions -> validate_bb0_layout enforces
    red-line 4; only primitive letters + run knobs ever enter here (red-line 3)."""
    slots = {int(k): v for k, v in (cfg.get("slots") or {}).items()}
    layout = layout_from_slots(slots)
    resolved = dict(_DEFAULTS)
    for k in _DEFAULTS:
        if cfg.get(k) is not None:
            resolved[k] = cfg[k]
    resolved["slots"] = slots
    resolved["layout"] = layout
    g = int(resolved["grid"])
    eng = Engine(H=g, W=g, K=int(resolved["K"]), seed=int(resolved["seed"]),
                 device=device, z_max=float(resolved["z_max"]),
                 fill_per_cell=int(resolved["fill"]), layout=layout)
    return eng, resolved
