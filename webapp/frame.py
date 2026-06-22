# webapp/frame.py
"""world tensor -> JSON frame (viz spec §2/§6). Faithful to the tensor: only
non-empty cells, per-faction count sums, NO smoothing/interpolation (red-line 2).
Readouts reuse the single-source pure function (red-line 5)."""
from __future__ import annotations
import torch
from webapp.readouts import compute_readouts

NFAC = 4


def encode_frame(world, table, tick: int, H: int, W: int, top_n: int = 5) -> dict:
    cnt = world.count.to("cpu")
    sid = world.strain_id.to("cpu")
    fac = world.faction.to("cpu")
    nz = torch.nonzero(cnt > 0, as_tuple=False)        # [M,3] = (y,x,k)
    ys = nz[:, 0].tolist(); xs = nz[:, 1].tolist(); ks = nz[:, 2]
    facs = fac[nz[:, 0], nz[:, 1], ks].tolist()
    cnts = cnt[nz[:, 0], nz[:, 1], ks].tolist()
    sids = sid[nz[:, 0], nz[:, 1], ks].tolist()
    # per-cell per-faction count sum
    by_cell: dict = {}
    for y, x, f, c in zip(ys, xs, facs, cnts):
        row = by_cell.setdefault((y, x), [0, 0, 0, 0])
        row[f] += c
    cells = [[y, x, *row] for (y, x), row in by_cell.items()]
    strains = [".".join(table.sequence_of(int(s))) for s in sids]
    readouts = compute_readouts(xs, ys, strains, facs, cnts)
    leaderboard = _leaderboard(strains, facs, cnts, readouts["total"], top_n)
    return {"tick": int(tick), "H": int(H), "W": int(W),
            "cells": cells, "readouts": readouts, "leaderboard": leaderboard}


def _leaderboard(strains, facs, cnts, total: int, top_n: int) -> list:
    """Dominant-strain ranking (spec §4). live-only — NOT in compute_readouts,
    so analyze_batch / red-line 5 stay untouched. Per strain: total count, share
    (count/total), and dominant faction (the faction holding the most of it)."""
    agg: dict = {}   # strain -> {"count": int, "fac": {f: c}}
    for s, f, c in zip(strains, facs, cnts):
        e = agg.setdefault(s, {"count": 0, "fac": {}})
        e["count"] += c
        e["fac"][f] = e["fac"].get(f, 0) + c
    denom = total or 1
    ranked = sorted(agg.items(), key=lambda kv: kv[1]["count"], reverse=True)[:top_n]
    out = []
    for s, e in ranked:
        dom_fac = max(e["fac"], key=e["fac"].get)
        out.append({"strain": s, "faction": int(dom_fac),
                    "count": int(e["count"]), "share": e["count"] / denom})
    return out


def cell_detail(world, table, y: int, x: int) -> dict:
    cnt = world.count[y, x].to("cpu")
    sid = world.strain_id[y, x].to("cpu")
    fac = world.faction[y, x].to("cpu")
    out = []
    for k in range(cnt.shape[0]):
        c = int(cnt[k])
        if c > 0:
            out.append({"strain": ".".join(table.sequence_of(int(sid[k]))),
                        "faction": int(fac[k]), "count": c})
    return {"y": int(y), "x": int(x), "strains": out}
