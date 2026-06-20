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
    phase_type: PhaseType | None
    fold: tuple[frozenset[int], ...]
