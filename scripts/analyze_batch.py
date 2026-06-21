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
