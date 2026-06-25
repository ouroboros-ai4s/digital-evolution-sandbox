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
