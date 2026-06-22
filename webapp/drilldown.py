# webapp/drilldown.py
"""Replay drilldown over recorded playground parquet (viz spec §6).
Tier A (frame_at_tick / cell_at_tick): cheap single-frame reads.
Tier B (strain_trajectory): predicate-pushdown filtered read, no prebuilt index.
Reads recorded truth only — never rewrites, never beautifies (red-lines 1/2)."""
from __future__ import annotations
import pyarrow.parquet as pq
import pyarrow.compute as pc
from webapp.readouts import compute_readouts

NFAC = 4


def _rows(path: str, filt=None):
    tbl = pq.read_table(path, filters=filt) if filt else pq.read_table(path)
    return tbl.to_pydict()


def frame_at_tick(path: str, tick: int) -> dict:
    d = _rows(path, filt=[("tick", "==", tick)])
    n = len(d["tick"])
    by_cell: dict = {}
    for i in range(n):
        y, x = d["cell_y"][i], d["cell_x"][i]
        row = by_cell.setdefault((y, x), [0, 0, 0, 0])
        row[d["faction"][i]] += d["count"][i]
    cells = [[y, x, *row] for (y, x), row in by_cell.items()]
    readouts = compute_readouts(d["cell_x"], d["cell_y"], d["strain"],
                                d["faction"], d["count"])
    return {"tick": int(tick), "cells": cells, "readouts": readouts}


def cell_at_tick(path: str, tick: int, y: int, x: int) -> dict:
    d = _rows(path, filt=[("tick", "==", tick), ("cell_y", "==", y), ("cell_x", "==", x)])
    n = len(d["tick"])
    strains = [{"strain": d["strain"][i], "faction": int(d["faction"][i]),
                "count": int(d["count"][i])} for i in range(n)]
    return {"tick": int(tick), "y": int(y), "x": int(x), "strains": strains}


def strain_trajectory(path: str, strain: str) -> list:
    # ponytail: predicate pushdown, no prebuilt index; lazy-build a
    # strain->row_group index only if profiling proves this too slow.
    d = _rows(path, filt=[("strain", "==", strain)])
    n = len(d["tick"])
    per_tick: dict = {}
    for i in range(n):
        t = int(d["tick"][i])
        e = per_tick.setdefault(t, {"tick": t, "total_count": 0, "_cells": set()})
        e["total_count"] += int(d["count"][i])
        e["_cells"].add((d["cell_y"][i], d["cell_x"][i]))
    out = []
    for t in sorted(per_tick):
        e = per_tick[t]
        out.append({"tick": t, "total_count": e["total_count"],
                    "occupied_cells": len(e["_cells"])})
    return out
