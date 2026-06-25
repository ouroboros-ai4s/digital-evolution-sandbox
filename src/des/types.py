# src/des/types.py
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum

EMPTY_ID = 0
FAMILY_RANK = {"N": 0, "F": 1, "P": 2, "Z": 3}

class PhaseType(IntEnum):
    ANTAGONISM = 1
    REPRODUCTION = 2

@dataclass(frozen=True)
class Phenotype:
    f: float
    directions: tuple[tuple[int, int], ...]
    p_leave: float
    z_raw: float
    prey_mask: int
    feature_mask: int
    p_x: float
    spectrum: tuple[tuple[str, float], ...]
    period: int
    repro_period: int
    anta_period: int
    dir_bits: int
    phase_type: PhaseType | None
    fold: tuple[frozenset[int], ...]
    vis_sum: float = 0.0      # S1: Σ_{i: fam=N} VIS[seq[i]]
    n_count: int = 0          # S1: #{i: fam=N}
    vis_mode: int = 0         # S1: 0=none, 1=vis-weighted, 2=inverse-vis-weighted
    in_place: bool = False    # S4: FSTACK — emit in source cell, kernel skips direction roll
    rand_dir: bool = False    # S4: FDRIFT — draw 1-of-4 each firing tick from world RNG
    f_hi: float = 0.0          # S5: stacked on-window f (== Phenotype.f)
    f_lo: float = 0.0          # S5: stacked off-window f (static default = f_hi)
    burst_w: int = 1           # S5: window period (static default 1 → always on)
    burst_k: int = 1           # S5: on-window length (static default 1 → 100% duty)
