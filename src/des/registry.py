# src/des/registry.py
from __future__ import annotations
from des.types import Phenotype, PhaseType, FAMILY_RANK

MU = 0.01          # baseline mutation floor μ (v1 constant; calibration owns it later)
_DELTA = 0.05      # v1 mutation add-on for P_hotspot
# v1 placeholder. p_max is an outcome-driven CALIBRATION constant
# (numerical-round protocol range [0.04, 0.15], init 0.08).
# Do NOT freeze at this value without running the calibration sweep.
P_MAX = 0.08

# letter -> family. v1 subset: BB0 locked set + 2 mutation-ladder rungs + N filler.
ALPHABET = {
    "N0": "N",
    "F4Nr1": "F", "F4Nr4": "F",
    "P_base": "P", "P_hotspot": "P",
    "BroadSweep": "Z",
}

# Granularity per primitive (S6). residue = single position; motif = N consecutive
# positions of the SAME letter. Roster tags `gran` explicitly only for N0–N7; every
# other letter is residue by single-position occupancy. v1 alphabet is all-residue.
GRAN: dict[str, str] = {
    "N0":         "residue",
    "F4Nr1":      "residue",
    "F4Nr4":      "residue",
    "P_base":     "residue",
    "P_hotspot":  "residue",
    "BroadSweep": "residue",
}

# Span length per motif primitive. residue letters MUST NOT appear here.
# v1: empty (no motif primitives — each later spec adds its own rows).
MOTIF_LEN: dict[str, int] = {}

FEATURE_BIT = {name: 1 << i for i, name in enumerate(sorted(ALPHABET))}

# v1 direction universe: every strain's directions are a subset of these.
# bit d (in dir_bits) <-> ALL_DIRECTIONS[d]. Extend if a future F primitive adds a direction.
ALL_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIR_BIT = {d: 1 << i for i, d in enumerate(ALL_DIRECTIONS)}

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


def motif_blocks(layout: tuple[str, ...]) -> tuple[tuple[int, int, str], ...]:
    """Decompose a flat-16 layout into `(start, end, letter)` blocks. Residue
    letters appear as singletons `(i, i+1, letter)`. Runs of the same
    `gran=="motif"` letter collapse into one block of length `MOTIF_LEN[letter]`.
    Pure function of the layout; reads only the registry tables. Default
    all-residue layouts yield 16 singletons (regression-lock invariant)."""
    blocks: list[tuple[int, int, str]] = []
    i = 0
    n = len(layout)
    while i < n:
        letter = layout[i]
        if GRAN.get(letter) == "motif":
            length = MOTIF_LEN[letter]
            blocks.append((i, i + length, letter))
            i += length
        else:
            blocks.append((i, i + 1, letter))
            i += 1
    return tuple(blocks)


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
    f_periods: list[int] = []
    z_periods: list[int] = []
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
            f_periods.append(per)
            phase_type = PhaseType.REPRODUCTION
        elif letter in _Z:
            z, fams, per = _Z[letter]
            z_sum += z
            for fam in fams:
                for t, bit in FEATURE_BIT.items():
                    if ALPHABET[t] == fam:
                        prey_mask |= bit
            periods.append(per)
            z_periods.append(per)
            if phase_type is None:
                phase_type = PhaseType.ANTAGONISM
        elif letter in _P:
            p_add, per = _P[letter]
            px_prod *= (1 - min(P_MAX, MU + p_add))
            periods.append(per)
            # Pick the P letter with the highest p_add as dominant; break ties by
            # first occurrence (sequence order). A fully principled multi-P rule
            # (p_add-weighted spectrum blend) is deferred to the spec, out of v1 scope.
            if dominant_p is None or p_add > _P[dominant_p][0]:
                dominant_p = letter

    f = 1 - f_prod
    p_leave = 1 - pl_prod
    # p(x): max(μ, 1 - Π(1-pᵢ)); with no P letter px_prod==1 → 0 → floor μ
    p_x = max(MU, 1 - px_prod)
    spectrum = _spectrum_for(dominant_p) if dominant_p else ()
    period = min(periods) if periods else 1
    repro_period = min(f_periods) if f_periods else 1
    anta_period = min(z_periods) if z_periods else 1
    dir_bits = 0
    for d in directions:
        dir_bits |= _DIR_BIT.get(d, 0)

    return Phenotype(
        f=f, directions=tuple(directions), p_leave=p_leave, z_raw=z_sum,
        prey_mask=prey_mask, feature_mask=feature_mask, p_x=p_x,
        spectrum=spectrum, period=period,
        repro_period=repro_period, anta_period=anta_period, dir_bits=dir_bits,
        phase_type=phase_type, fold=(),
    )


# BB0: 16 positions. Slots (mutable) at design 1-indexed {1,3,4,10,11,14} = 0-idx {0,2,3,9,10,13}.
_SLOTS = {0, 2, 3, 9, 10, 13}
_LOCKED = {1: "F4Nr4", 5: "BroadSweep", 7: "P_base"}  # 0-indexed; F4Nr4 = 4-dir expansion (B3 fix)
BB0_TEMPLATE = {
    "layout": tuple(
        _LOCKED.get(i, "N0") for i in range(16)
    ),
    "mutable": tuple(i in _SLOTS for i in range(16)),
    "fold": (frozenset({0, 2, 3, 4}), frozenset({9, 13, 15})),
}


def validate_bb0_layout(layout: tuple[str, ...]) -> None:
    """Enforce the BB0 symmetry invariant (viz spec §5 / red-line 4).
    locked positions must equal _LOCKED; backbone (non-locked, non-slot)
    positions must stay "N0"; only _SLOTS positions may vary, and only to a
    primitive in the 6-letter palette. Raises ValueError on any violation."""
    if len(layout) != 16:
        raise ValueError(f"BB0 layout must have 16 positions, got {len(layout)}")
    for i, letter in enumerate(layout):
        if i in _LOCKED:
            if letter != _LOCKED[i]:
                raise ValueError(
                    f"position {i} is locked to {_LOCKED[i]!r}, got {letter!r}")
        elif i in _SLOTS:
            if letter not in ALPHABET:
                raise ValueError(
                    f"slot {i} = {letter!r} not in palette {sorted(ALPHABET)}")
        else:
            if letter != "N0":
                raise ValueError(
                    f"position {i} is backbone-fixed to 'N0', got {letter!r}")
