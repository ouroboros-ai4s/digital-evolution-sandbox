#!/usr/bin/env python
"""Report-only analysis of first-batch parquets. Computes per-seed + cross-seed
dynamics metrics and PRINTS them. NEVER emits PASS/FAIL — the human judges
(spec §0.1, §4). 'kills/减员' is not a column; count-drop figures are PROXIES
conflating K-wall evaporation + p_leave + arbitration (spec §0.2)."""
from __future__ import annotations
import argparse
import datetime
import glob
import json
import math
import os
import re
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


def diversity_metrics(df: pd.DataFrame, eps: float = 0.01, lag: int = 5) -> dict:
    ticks = sorted(int(t) for t in df["tick"].unique())
    by_strain = df.groupby(["tick", "strain"])["count"].sum()
    distinct_strains, n2, d_max = {}, {}, {}
    freqs = {}  # tick -> {strain: freq}
    for t in ticks:
        s = by_strain.loc[t]
        s = s[s > 0]
        tot = float(s.sum()) or 1.0
        f = (s / tot)
        freqs[t] = {str(k): float(v) for k, v in f.items()}
        distinct_strains[t] = int((s > 0).sum())
        n2[t] = float(1.0 / (f ** 2).sum()) if len(f) else 0.0
        d_max[t] = float(f.max()) if len(f) else 0.0

    # first-seen tick per strain (any row, including count 0 — records emergence)
    first = df.groupby("strain")["tick"].min()
    new_strain_first_seen = {str(k): int(v) for k, v in first.items()}

    # established-set flux = 1/2 sum |p_s(t) - p_s(t-lag)| over union of strains
    established_flux = {}
    for i, t in enumerate(ticks):
        if i - lag < 0:
            continue
        t0 = ticks[i - lag]
        keys = {k for k in (set(freqs[t]) | set(freqs[t0]))
                if freqs[t].get(k, 0.0) >= eps or freqs[t0].get(k, 0.0) >= eps}
        flux = 0.5 * sum(abs(freqs[t].get(k, 0.0) - freqs[t0].get(k, 0.0)) for k in keys)
        established_flux[t] = float(flux)

    # leader changes = # of times the argmax strain switches across ticks
    leaders = []
    for t in ticks:
        if freqs[t]:
            leaders.append(max(freqs[t], key=freqs[t].get))
    leader_changes = sum(1 for a, b in zip(leaders, leaders[1:]) if a != b)

    return {"distinct_strains": distinct_strains, "n2": n2,
            "new_strain_first_seen": new_strain_first_seen, "d_max": d_max,
            "established_flux": established_flux, "leader_changes": leader_changes}


def proxy_and_seeding_metrics(df: pd.DataFrame) -> dict:
    ticks = sorted(int(t) for t in df["tick"].unique())
    seed_tick = ticks[0] if ticks else None
    seed_df = df[df["tick"] == seed_tick] if seed_tick is not None else df.iloc[:0]
    seed_live = seed_df[seed_df["count"] > 0]
    seed_distinct_strains = int(seed_live["strain"].nunique())
    seed_distinct_factions = int(seed_live["faction"].nunique())

    # net-decrease PROXY: per (cell,faction,strain) count drop vs previous tick.
    # NOT kills -- conflates K-wall evaporation + p_leave + arbitration (spec §0.2).
    key = ["cell_x", "cell_y", "faction", "strain"]
    piv = df.pivot_table(index=key, columns="tick", values="count",
                         aggfunc="sum", fill_value=0)
    net_decrease_proxy = {}
    cols = list(piv.columns)
    for i, t in enumerate(cols):
        if i == 0:
            continue
        delta = piv[t] - piv[cols[i - 1]]
        net_decrease_proxy[int(t)] = int(-delta[delta < 0].sum())

    # strain x faction cross-tab at last tick, for strains under >=2 factions
    last = ticks[-1] if ticks else None
    xtab = {}
    if last is not None:
        ld = df[(df["tick"] == last) & (df["count"] > 0)]
        g = ld.groupby(["strain", "faction"])["count"].sum()
        for strain, sub in g.groupby(level="strain"):
            fac_counts = {int(f): int(v) for (_s, f), v in sub.items()}
            if len(fac_counts) >= 2:
                xtab[str(strain)] = fac_counts

    return {"seed_tick": seed_tick, "seed_distinct_strains": seed_distinct_strains,
            "seed_distinct_factions": seed_distinct_factions,
            "net_decrease_proxy": net_decrease_proxy, "strain_faction_xtab": xtab}


def per_seed_metrics(df: pd.DataFrame, n_cells: int = 128 * 128) -> dict:
    m = {"seed": None}
    m.update(survival_spatial_metrics(df, n_cells=n_cells))
    m.update(diversity_metrics(df))
    m.update(proxy_and_seeding_metrics(df))
    return m



def cross_seed_metrics(per_seed_list: list, steady_min_ticks: int = 200) -> dict:
    n = len(per_seed_list)
    winners = [m.get("winner_faction") for m in per_seed_list]
    win_counts = {}
    for w in winners:
        if w is not None:
            win_counts[int(w)] = win_counts.get(int(w), 0) + 1

    p = 0.25
    sigma = math.sqrt(n * p * (1 - p)) if n else 0.0
    win_ci_note = {
        "expected_share": p, "n": n,
        "per_faction_count": win_counts,
        "binom_2sigma_lo": p * n - 2 * sigma,
        "binom_2sigma_hi": p * n + 2 * sigma,
    }

    # D4 symmetry: per-tick max-min of per-faction occupied counts, averaged across seeds
    spread_acc, spread_cnt = {}, {}
    for m in per_seed_list:
        for t, fac_occ in m.get("faction_occupied", {}).items():
            if fac_occ:
                vals = list(fac_occ.values())
                spread_acc[t] = spread_acc.get(t, 0) + (max(vals) - min(vals))
                spread_cnt[t] = spread_cnt.get(t, 0) + 1
    d4_symmetry_spread = {int(t): spread_acc[t] / spread_cnt[t] for t in spread_acc}

    # GATE0: steady window = last tick - fill tick (post-fill band). Use the min across
    # seeds (worst case). If a seed never filled, treat window as 0.
    windows = []
    for m in per_seed_list:
        ticks = m.get("ticks") or []
        last = ticks[-1] if ticks else 0
        fill = m.get("fill_tick")
        windows.append((last - fill) if fill is not None else 0)
    steady_window = min(windows) if windows else 0
    gate0_short_run = {
        "steady_window_ticks": steady_window, "required": steady_min_ticks,
        "note": ("steady window < required: red-queen freq-dependence (β<0) NOT computable "
                 "this batch (spec §1 / GATE0 NA-SHORT-RUN)")
                if steady_window < steady_min_ticks else "steady window adequate",
    }

    timeline_reconciliation = {
        "per_seed": [
            {"first_cross_faction_tick": m.get("first_cross_faction_tick"),
             "fill_tick": m.get("fill_tick")}
            for m in per_seed_list
        ],
        "spec_meet": 160, "spec_fill": 320, "design_meet": 32, "design_fill": 60,
    }

    return {"n_seeds": n, "winners": winners, "win_counts": win_counts,
            "win_ci_note": win_ci_note, "d4_symmetry_spread": d4_symmetry_spread,
            "gate0_short_run": gate0_short_run,
            "timeline_reconciliation": timeline_reconciliation}



def _last(d):
    """last value of a {tick: val} dict by tick order, or None."""
    if not d:
        return None
    return d[max(d)]


def _fmt(v, spec):
    """Format v with spec, or return 'n/a' if v is None."""
    return format(v, spec) if v is not None else "n/a"


def render_text(per_seed_list: list, cross: dict) -> str:
    L = []
    L.append("=" * 70)
    L.append("FIRST-BATCH ANALYSIS REPORT (numbers only — human makes the verdict)")
    L.append("=" * 70)
    for m in per_seed_list:
        L.append(f"\n--- seed {m.get('seed')} ---")
        ticks = m.get("ticks") or []
        span = f"{ticks[0]}..{ticks[-1]}" if ticks else "(empty)"
        L.append(f"  ticks recorded: {span}  ({len(ticks)} ticks)")
        L.append(f"  total_count last: {_last(m['total_count'])}   "
                 f"extinction_tick: {m['extinction_tick']}")
        L.append(f"  distinct_factions last: {_last(m['distinct_factions'])}   "
                 f"fixation_tick: {m['fixation_tick']}")
        L.append(f"  occupied_cells last: {_last(m['occupied_cells'])}   "
                 f"first_cross_faction_tick: {m['first_cross_faction_tick']}   "
                 f"fill_tick: {m['fill_tick']}")
        L.append(f"  winner_faction: {m['winner_faction']}   "
                 f"faction_share last: {m['faction_share'].get(max(m['faction_share'])) if m['faction_share'] else {}}")
        L.append(f"  distinct_strains last: {_last(m['distinct_strains'])}   "
                 f"N2 last: {_fmt(_last(m['n2']), '.2f')}   d_max last: {_fmt(_last(m['d_max']), '.3f')}")
        L.append(f"  total strains ever seen: {len(m['new_strain_first_seen'])}   "
                 f"leader_changes: {m['leader_changes']}")
        L.append(f"  established_flux last: {_last(m['established_flux'])}")
        L.append(f"  net_decrease_proxy last: {_last(m['net_decrease_proxy'])}  "
                 f"[PROXY — NOT kills; conflates K-wall+p_leave+arbitration]")
        L.append(f"  seeding: tick {m['seed_tick']} -> "
                 f"{m['seed_distinct_strains']} strain(s), {m['seed_distinct_factions']} faction(s) "
                 f"(expect 1 strain / 4 factions)")
        L.append(f"  strain×faction xtab (strains under ≥2 factions, last tick): "
                 f"{m['strain_faction_xtab']}")
    L.append("\n" + "=" * 70)
    L.append("CROSS-SEED AGGREGATE")
    L.append("=" * 70)
    L.append(f"  n_seeds: {cross['n_seeds']}   winners: {cross['winners']}")
    ci = cross["win_ci_note"]
    L.append(f"  win counts by faction: {ci['per_faction_count']}")
    L.append(f"  symmetric expectation: {ci['expected_share']} share; "
             f"2σ band on win-count = [{ci['binom_2sigma_lo']:.2f}, {ci['binom_2sigma_hi']:.2f}] "
             f"(deviation = possible sneak-goods leak; user judges)")
    L.append(f"  GATE0: steady_window={cross['gate0_short_run']['steady_window_ticks']} "
             f"required={cross['gate0_short_run']['required']} — "
             f"{cross['gate0_short_run']['note']}")
    tr = cross["timeline_reconciliation"]
    L.append(f"  timeline: per-seed {tr['per_seed']}")
    L.append(f"            spec expects meet~{tr['spec_meet']}/fill~{tr['spec_fill']}; "
             f"design(period-unfactored) meet~{tr['design_meet']}/fill~{tr['design_fill']} "
             f"— compare to observed above")
    return "\n".join(L)


def dump_json(report: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)


def main() -> None:
    ap = argparse.ArgumentParser(description="Report-only first-batch analysis (no verdicts).")
    ap.add_argument("--runs-dir", default=os.path.join("data", "runs"))
    ap.add_argument("--glob", default="*-seed*.parquet")
    ap.add_argument("--n-cells", type=int, default=128 * 128)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.runs_dir, args.glob)))
    if not paths:
        print(f"no parquets matched {args.glob} in {args.runs_dir}")
        return
    per_seed = []
    for p in paths:
        df = load(p)
        if df.empty:
            print(f"[skip] {p} — empty parquet (zero rows), skipping")
            continue
        m = re.search(r"seed(\d+)", os.path.basename(p))
        ms = per_seed_metrics(df, n_cells=args.n_cells)
        ms["seed"] = int(m.group(1)) if m else None
        per_seed.append(ms)
    cross = cross_seed_metrics(per_seed)
    print(render_text(per_seed, cross))
    tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.runs_dir, f"analysis-{tag}.json")
    dump_json({"per_seed": per_seed, "cross_seed": cross}, out)
    print(f"\n-> wrote {out}")


if __name__ == "__main__":
    main()
