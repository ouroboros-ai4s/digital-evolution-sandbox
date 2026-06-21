#!/usr/bin/env python
"""Report-only analysis of first-batch parquets. Computes per-seed + cross-seed
dynamics metrics and PRINTS them. NEVER emits PASS/FAIL — the human judges
(spec §0.1, §4). 'kills/减员' is not a column; count-drop figures are PROXIES
conflating K-wall evaporation + p_leave + arbitration (spec §0.2)."""
from __future__ import annotations
import pyarrow.parquet as pq
import pandas as pd


def load(path: str) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def survival_spatial_metrics(df: pd.DataFrame, n_cells: int = 128 * 128) -> dict:
    ticks = sorted(int(t) for t in df["tick"].unique())
    full = list(range(min(ticks), max(ticks) + 1)) if ticks else []

    tot = df.groupby("tick")["count"].sum()
    total_count = {int(t): int(tot.get(t, 0)) for t in full}
    extinction_tick = next((t for t in full if total_count[t] == 0), None)

    live = df[df["count"] > 0]
    dfac = live.groupby("tick")["faction"].nunique()
    distinct_factions = {int(t): int(dfac.get(t, 0)) for t in ticks}
    fixation_tick = next((t for t in ticks if distinct_factions[t] == 1), None)

    def _occ(g):
        return g[["cell_x", "cell_y"]].drop_duplicates().shape[0]
    occ = live.groupby("tick", group_keys=False)[["cell_x", "cell_y"]].apply(_occ)
    occupied_cells = {int(t): int(occ.get(t, 0)) for t in ticks}
    fill_tick = next((t for t in ticks if occupied_cells[t] >= n_cells), None)

    per_cell_fac = live.groupby(["tick", "cell_x", "cell_y"])["faction"].nunique()
    cross = per_cell_fac[per_cell_fac > 1]
    first_cross_faction_tick = (
        int(cross.index.get_level_values("tick").min()) if len(cross) else None)

    fac_occ = live.groupby(["tick", "faction"], group_keys=False)[["cell_x", "cell_y"]].apply(_occ)
    faction_occupied = {int(t): {} for t in ticks}
    for (t, f), v in fac_occ.items():
        faction_occupied[int(t)][int(f)] = int(v)

    fac_cnt = live.groupby(["tick", "faction"])["count"].sum()
    faction_share = {int(t): {} for t in ticks}
    for (t, f), v in fac_cnt.items():
        denom = total_count.get(int(t), 0) or 1
        faction_share[int(t)][int(f)] = float(v) / denom

    last = ticks[-1] if ticks else None
    winner_faction = None
    if last is not None and faction_share[last]:
        winner_faction = int(max(faction_share[last], key=faction_share[last].get))

    return {
        "ticks": ticks, "total_count": total_count, "extinction_tick": extinction_tick,
        "distinct_factions": distinct_factions, "fixation_tick": fixation_tick,
        "occupied_cells": occupied_cells, "first_cross_faction_tick": first_cross_faction_tick,
        "fill_tick": fill_tick, "faction_occupied": faction_occupied,
        "faction_share": faction_share, "winner_faction": winner_faction,
    }
