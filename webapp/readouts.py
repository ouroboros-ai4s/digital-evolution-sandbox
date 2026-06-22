# webapp/readouts.py
"""Single-source per-tick readouts (viz spec red-line 5). Pure: stdlib only,
no pandas/torch import. The live path (webapp/frame.py) feeds world-tensor
records; scripts/analyze_batch.py feeds parquet-df records. One definition,
shared, so the live acceptance picture and the offline report never drift."""
from __future__ import annotations


def compute_readouts(cell_x, cell_y, strain, faction, count) -> dict:
    """All five sequences are equal length; one entry per non-empty
    (cell, slot) record of ONE tick. Returns:
      total            int   = Σ count
      occupied_cells   int   = # distinct (cell_x, cell_y)
      distinct_strains int   = # distinct strain present
      n2               float = 1 / Σ p_s²   (p_s = strain_total / total); 0 if empty
      d_max            float = max_s p_s; 0 if empty
      faction_share    dict[int,float] = Σcount_f / total per faction
    """
    total = 0
    by_strain: dict = {}
    by_faction: dict = {}
    occ = set()
    for i in range(len(count)):
        c = count[i]
        total += c
        by_strain[strain[i]] = by_strain.get(strain[i], 0) + c
        by_faction[faction[i]] = by_faction.get(faction[i], 0) + c
        occ.add((cell_x[i], cell_y[i]))
    n2 = 0.0
    d_max = 0.0
    if total:
        sumsq = sum((v / total) ** 2 for v in by_strain.values())
        n2 = 1.0 / sumsq if sumsq else 0.0
        d_max = max(v / total for v in by_strain.values())
    return {
        "total": int(total),
        "occupied_cells": len(occ),
        "distinct_strains": len(by_strain),
        "n2": float(n2),
        "d_max": float(d_max),
        "faction_share": {int(f): v / total for f, v in by_faction.items()},
    }
