# src/des/registry.py
from __future__ import annotations
from des.types import Phenotype, PhaseType, FAMILY_RANK

MU = 0.01          # baseline mutation floor μ (v1 constant; calibration owns it later)
_DELTA = 0.05      # v1 mutation add-on for P_hotspot

# letter -> family. v1 subset: BB0 locked set + 2 mutation-ladder rungs + N filler.
ALPHABET = {
    "N0": "N",
    "F4Nr1": "F", "F4Nr4": "F",
    "P_base": "P", "P_hotspot": "P",
    "BroadSweep": "Z",
}
FEATURE_BIT = {name: 1 << i for i, name in enumerate(sorted(ALPHABET))}

# per-letter raw outputs (design tables; numbers are formula anchors, not calibrated knobs)
_F = {    # name -> (f, directions, p_leave, period)
    "F4Nr1": (0.30, ((-1, 0),), 0.05, 4),
    "F4Nr4": (0.50, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.15, 5),
}
_Z = {    # name -> (z, prey_families, period)
    "BroadSweep": (0.40, ("F", "Z"), 5),
}
_P = {    # name -> (p_add, period); effective rate = min(p_max, μ + p_add)
    "P_base": (0.0, 1),
    "P_hotspot": (_DELTA, 3),
}


def affinity(src_family: str, dst_family: str) -> float:
    d = abs(FAMILY_RANK[src_family] - FAMILY_RANK[dst_family])
    return 0.70 if d == 0 else 0.25 if d == 1 else 0.05


def _spectrum_for(letter: str) -> tuple[tuple[str, float], ...]:
    """Family-distance spectrum: q(target) ∝ affinity(letter_family, target_family),
    excluding the letter itself, normalized to Σq=1. Pure function of the alphabet."""
    src_fam = ALPHABET[letter]
    weights = {t: affinity(src_fam, ALPHABET[t]) for t in ALPHABET if t != letter}
    tot = sum(weights.values())
    if tot == 0:
        return ()
    return tuple((t, w / tot) for t, w in sorted(weights.items()))


def phenotype(sequence: tuple[str, ...]) -> Phenotype:
    """Pure function of the sequence only. No world-state, no neighbors, no tick.
    κ=0 in v1 — no self-coordination neighbor scan."""
    f_prod = 1.0          # accumulate Π(1-fᵢ)
    pl_prod = 1.0
    px_prod = 1.0
    z_sum = 0.0
    prey_mask = 0
    feature_mask = 0
    directions: list[tuple[int, int]] = []
    periods: list[int] = []
    phase_type: PhaseType | None = None
    dominant_p: str | None = None

    for letter in sequence:
        if letter not in ALPHABET:
            continue
        feature_mask |= FEATURE_BIT[letter]
        if letter in _F:
            f, dirs, pl, per = _F[letter]
            f_prod *= (1 - f)
            pl_prod *= (1 - pl)
            for d in dirs:
                if d not in directions:
                    directions.append(d)
            periods.append(per)
            phase_type = PhaseType.REPRODUCTION
        elif letter in _Z:
            z, fams, per = _Z[letter]
            z_sum += z
            for fam in fams:
                for t, bit in FEATURE_BIT.items():
                    if ALPHABET[t] == fam:
                        prey_mask |= bit
            periods.append(per)
            if phase_type is None:
                phase_type = PhaseType.ANTAGONISM
        elif letter in _P:
            p_add, per = _P[letter]
            p_max = 0.35
            px_prod *= (1 - min(p_max, MU + p_add))
            periods.append(per)
            dominant_p = letter

    f = 1 - f_prod
    p_leave = 1 - pl_prod
    # p(x): max(μ, 1 - Π(1-pᵢ)); with no P letter px_prod==1 → 0 → floor μ
    p_x = max(MU, 1 - px_prod)
    spectrum = _spectrum_for(dominant_p) if dominant_p else ()
    period = min(periods) if periods else 1

    return Phenotype(
        f=f, directions=tuple(directions), p_leave=p_leave, z_raw=z_sum,
        prey_mask=prey_mask, feature_mask=feature_mask, p_x=p_x,
        spectrum=spectrum, period=period, phase_type=phase_type, fold=(),
    )


# BB0: 16 positions. Slots (mutable) at design 1-indexed {1,3,4,10,11,14} = 0-idx {0,2,3,9,10,13}.
_SLOTS = {0, 2, 3, 9, 10, 13}
_LOCKED = {1: "F4Nr1", 5: "BroadSweep", 7: "P_base"}  # 0-indexed
BB0_TEMPLATE = {
    "layout": tuple(
        _LOCKED.get(i, "N0") for i in range(16)
    ),
    "mutable": tuple(i in _SLOTS for i in range(16)),
    "fold": (frozenset({0, 2, 3, 4}), frozenset({9, 13, 15})),
}
