# src/des/registry.py
from __future__ import annotations
import zlib
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
    # S2: P-pool expansion
    "P_aic":            "P",
    "P_ep":             "P",
    "P_fscan":          "P",
    "P_zscan":          "P",
    "P_entropy_brake":  "P",
    "P_loopswap_lite":  "P",
    "P_neutral_sink":   "P",
    "P_slow_drift":     "P",
    "P_burst_lite":     "P",
    "P_balanced":       "P",
    # S4: F-pool dynamic-direction primitives (spec §3.4)
    "FSTACK":  "F",
    "FCLUMP":  "F",
    "FFRONT":  "F",
    "F4Nr3":   "F",
    "FDRIFT":  "F",
    # S5: F-pool phase-window primitives (spec §3.2)
    "FBURST":  "F",
    "F_NOVA":  "F",
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
    # S2: P-pool expansion (all residue)
    "P_aic":            "residue",
    "P_ep":             "residue",
    "P_fscan":          "residue",
    "P_zscan":          "residue",
    "P_entropy_brake":  "residue",
    "P_loopswap_lite":  "residue",
    "P_neutral_sink":   "residue",
    "P_slow_drift":     "residue",
    "P_burst_lite":     "residue",
    "P_balanced":       "residue",
    # S4: F-pool dynamic directions
    "FSTACK":  "residue",
    "FCLUMP":  "motif",
    "FFRONT":  "motif",
    "F4Nr3":   "residue",
    "FDRIFT":  "residue",
    # S5: F-pool phase-window primitives
    "FBURST":  "residue",
    "F_NOVA":  "residue",
}

# Span length per motif primitive. residue letters MUST NOT appear here.
# v1: empty (no motif primitives — each later spec adds its own rows).
MOTIF_LEN: dict[str, int] = {
    # S4: motif F primitives (spec §3.4)
    "FCLUMP": 2,
    "FFRONT": 2,
}

# Per-primitive vis (S1). Pure registry value, never per-species. The N pool
# carries vis ∈ [0,1] from the roster (primitive-roster.md N pool); non-N
# letters carry 0.0 (vis is the *only* output of N primitives — F/P/Z never
# emit vis). Module-load asserts the unit-interval bound (spec §5).
VIS: dict[str, float] = {
    # v1 alphabet (matches ALPHABET above):
    "N0":         0.20,   # roster n0 — present in default BB0
    "F4Nr1":      0.0,
    "F4Nr4":      0.0,
    "P_base":     0.0,
    "P_hotspot":  0.0,
    "BroadSweep": 0.0,
    # S2: P-pool expansion (all non-N letters carry 0.0)
    "P_aic":            0.0,
    "P_ep":             0.0,
    "P_fscan":          0.0,
    "P_zscan":          0.0,
    "P_entropy_brake":  0.0,
    "P_loopswap_lite":  0.0,
    "P_neutral_sink":   0.0,
    "P_slow_drift":     0.0,
    "P_burst_lite":     0.0,
    "P_balanced":       0.0,
    # S4: F-pool dynamic-direction primitives (all non-N letters carry 0.0)
    "FSTACK":  0.0,
    "FCLUMP":  0.0,
    "FFRONT":  0.0,
    "F4Nr3":   0.0,
    "FDRIFT":  0.0,
    # S5: F-pool phase-window primitives (all non-N letters carry 0.0)
    "FBURST":  0.0,
    "F_NOVA":  0.0,
}
for _letter, _v in VIS.items():
    assert 0.0 <= _v <= 1.0, f"VIS[{_letter!r}] = {_v} outside [0,1]"
del _letter, _v

# Predicate-bit vocabulary (S6 §3.5). Each bit = a structural predicate, not a
# letter. Stable indices: S1/S3 will populate the reserved names without
# renumbering. Total 15 of the 63 available signed-int64 bits.
PREDICATE_BITS: dict[str, int] = {
    # family-of-letter (4 bits): set per-letter at mint.
    "family_N":     0,
    "family_F":     1,
    "family_P":     2,
    "family_Z":     3,
    # sequence has at least one motif block of family fam (4 bits): set per-strain at mint.
    "motif_F":      4,
    "motif_P":      5,
    "motif_Z":      6,
    "motif_N":      7,
    # sequence has at least one motif block of family fam with MOTIF_LEN >= 3 (3 bits).
    "motif3_F":     8,
    "motif3_P":     9,
    "motif3_Z":    10,
    # Reserved (S1 / S3 fill the values; bit indices stable so adding the values
    # later is a bit-set, not a renumbering).
    "vis_lowvis":  11,   # gran=='residue' AND vis<=0.20    (S1 fills vis bit, S3 wires)
    "thr_crest":   12,   # family=='F' AND f>=0.5            (S3)
    "thr_hotspot": 13,   # family=='P' AND p_add>=0.05       (S3)
    "thr_mirror":  14,   # family=='Z' AND z<=0.45 AND |prey|>=2  (S3)
}
assert max(PREDICATE_BITS.values()) < 63, \
    "predicate vocabulary overflows int64; halt before phenotype() is built"

PREDICATE_BIT: dict[str, int] = {name: 1 << idx for name, idx in PREDICATE_BITS.items()}

FEATURE_BIT = {name: 1 << i for i, name in enumerate(sorted(ALPHABET))}

# v1 direction universe: every strain's directions are a subset of these.
# bit d (in dir_bits) <-> ALL_DIRECTIONS[d]. Extend if a future F primitive adds a direction.
ALL_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIR_BIT = {d: 1 << i for i, d in enumerate(ALL_DIRECTIONS)}

# S4: in-place 方向 sentinel. 不入 ALL_DIRECTIONS, 不入 _DIR_BIT —— FSTACK
# 走 Phenotype.in_place 的独立内核分支, 与四邻 roll 路径正交.
IN_PLACE_DIR: tuple[int, int] = (0, 0)


def _hash_dirs(seq: tuple[str, ...], kind: str) -> tuple[tuple[int, int], ...]:
    """S4 hash-locked direction selection. Pure function of the sequence.

    Determinism: stdlib zlib.crc32(\\x1f-joined utf-8 bytes); the \\x1f (unit
    separator, not in the alphabet) prevents multi-char token concat ambiguity.
    Python's built-in hash() is salted per process (PYTHONHASHSEED) →不可复现, 致命
    for a data-generation sandbox; crc32 is byte-identical cross-process / cross-machine.

    kind:
      "ffront" | "f4nr1" -> ( ALL_DIRECTIONS[h % 4], )                       1 方向
      "fclump"           -> ( (-1,0), (1,0) ) or ( (0,-1), (0,1) ) per h % 2  一根轴
      "f4nr3"            -> ALL_DIRECTIONS minus ALL_DIRECTIONS[h % 4]        3 邻
    """
    h = zlib.crc32("\x1f".join(seq).encode())
    if kind in ("ffront", "f4nr1"):
        return (ALL_DIRECTIONS[h % 4],)
    if kind == "fclump":
        if h % 2 == 0:
            return ((-1, 0), (1, 0))
        return ((0, -1), (0, 1))
    if kind == "f4nr3":
        drop = h % 4
        return tuple(d for i, d in enumerate(ALL_DIRECTIONS) if i != drop)
    raise ValueError(f"_hash_dirs: unknown kind {kind!r}; "
                     "expected one of {'ffront','f4nr1','fclump','f4nr3'}")


# per-letter raw outputs (design tables; numbers are formula anchors, not calibrated knobs)
_F = {    # name -> (f, directions, p_leave, period, f_lo, burst_w, burst_k)
    # S4: F4Nr1 由 v1 占位 ((-1, 0),) 重底定为 hash-locked 1-of-4 (spec §3.3).
    # S5: static defaults f_lo=f, burst_w=1, burst_k=1 (no phase window yet).
    "F4Nr1": (0.30, "hash:f4nr1",                      0.05, 4, 0.30, 1, 1),
    "F4Nr4": (0.50, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.15, 5, 0.50, 1, 1),
    # S4: 5 new F primitives (spec §3.4 verbatim); S5 static defaults.
    "FSTACK":  (0.60, (IN_PLACE_DIR,), 0.00, 3, 0.60, 1, 1),
    "FCLUMP":  (0.45, "hash:fclump",   0.10, 6, 0.45, 1, 1),
    "FFRONT":  (0.50, "hash:ffront",   0.25, 4, 0.50, 1, 1),
    "F4Nr3":   (0.40, "hash:f4nr3",    0.12, 5, 0.40, 1, 1),
    "FDRIFT":  (0.15, "rand:1of4",     0.30, 2, 0.15, 1, 1),
    # S5: 2 new phase-window F primitives (spec §3.2 verbatim)
    "FBURST":  (0.55, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.20, 2, 0.05, 12, 2),
    "F_NOVA":  (0.85, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.50, 2, 0.05, 20, 1),
}
_Z = {    # name -> (z, prey_clauses, period, vis_mode)
    # prey_clauses: tuple of clause-tuples. Each clause selects ONE predicate
    # bit; the prey_mask is the OR over clauses. v1 clauses are single-element
    # family tuples → identical kernel-match outcomes to the pre-S6 family code.
    # Future motif-specialist Z rows will use multi-element clauses like
    # ("F", "motif") or ("Z", "motif", "len>=3").
    # vis_mode (S1): 0=none, 1=vis-weighted, 2=inverse-vis-weighted.
    # The 4th element is OPTIONAL: a 3-tuple row defaults vis_mode to 0.
    "BroadSweep": (0.40, (("F",), ("Z",)), 5, 0),
}

# S3: prey-clause cardinality (|prey_s| in roster Mirror Fang spec §1).
# Module-load derived from _Z[letter][1] (the prey_clauses tuple) so the
# feature_mask_of hot path reads O(1) per Z letter, never iterates clauses
# at mint time. Co-extensive with _Z; adding a new _Z row REQUIRES this
# dict be re-derived (it is, on every module import).
_Z_PREY_CARD: dict[str, int] = {name: len(row[1]) for name, row in _Z.items()}

_P = {    # name -> (p_add, period); effective rate = min(p_max, μ + p_add)
    "P_base": (0.0, 1),
    "P_hotspot": (_DELTA, 3),
    # S2: 10 new P primitives (verbatim from primitive-roster §P pool)
    "P_aic":            (0.03, 3),
    "P_ep":             (0.04, 3),
    "P_fscan":          (0.02, 5),
    "P_zscan":          (0.02, 5),
    "P_entropy_brake":  (0.01, 7),
    "P_loopswap_lite":  (0.03, 4),
    "P_neutral_sink":   (0.02, 5),
    "P_slow_drift":     (0.0,  9),
    "P_burst_lite":     (0.07, 2),
    "P_balanced":       (0.04, 3),
}

# Mutation spectrum shape per P primitive (S2). Three knobs cover all 12 P
# rows — no per-primitive special path. (power, family_mask, flatten_mix):
#   power       : 1=aff, 2=sharpen (P_aic), 3=super-sharpen (P_entropy_brake)
#   family_mask : None=all, "F"|"Z"|"N"=single family,
#                 "adjacent"=|Δrank|=1 (P_loopswap_lite)
#   flatten_mix : 0.0 default, 0.5 (P_ep — ½·aff + ½·1/(|A|-1))
SPECTRUM_SHAPE: dict[str, tuple[float, "str | None", float]] = {
    "P_base":           (1.0, None,       0.0),
    "P_hotspot":        (1.0, None,       0.0),
    "P_aic":            (2.0, None,       0.0),
    "P_ep":             (1.0, None,       0.5),
    "P_fscan":          (1.0, "F",        0.0),
    "P_zscan":          (1.0, "Z",        0.0),
    "P_entropy_brake":  (3.0, None,       0.0),
    "P_loopswap_lite":  (1.0, "adjacent", 0.0),
    "P_neutral_sink":   (1.0, "N",        0.0),
    "P_slow_drift":     (1.0, None,       0.0),
    "P_burst_lite":     (1.0, None,       0.0),
    "P_balanced":       (1.0, None,       0.0),
}

assert set(SPECTRUM_SHAPE.keys()) == set(_P.keys()), (
    "SPECTRUM_SHAPE must be co-extensive with _P; "
    f"missing={set(_P.keys()) - set(SPECTRUM_SHAPE.keys())}, "
    f"extra={set(SPECTRUM_SHAPE.keys()) - set(_P.keys())}")
for _letter, (_power, _mask, _mix) in SPECTRUM_SHAPE.items():
    assert _power in (1.0, 2.0, 3.0), \
        f"SPECTRUM_SHAPE[{_letter!r}].power = {_power!r} not in {{1,2,3}}"
    assert _mask in (None, "F", "Z", "N", "adjacent"), \
        f"SPECTRUM_SHAPE[{_letter!r}].family_mask = {_mask!r} not in {{None,F,Z,N,adjacent}}"
    assert 0.0 <= _mix <= 1.0, \
        f"SPECTRUM_SHAPE[{_letter!r}].flatten_mix = {_mix!r} outside [0,1]"
del _letter, _power, _mask, _mix


def affinity(src_family: str, dst_family: str) -> float:
    d = abs(FAMILY_RANK[src_family] - FAMILY_RANK[dst_family])
    return 0.70 if d == 0 else 0.25 if d == 1 else 0.05


def _spectrum_for(letter: str) -> tuple[tuple[str, float], ...]:
    """Family-distance spectrum, gran-matched + equal-length pre-filtered (S6),
    shape-modulated by SPECTRUM_SHAPE (S2). Pure function of the alphabet.

    Three-knob shape from SPECTRUM_SHAPE.get(letter, (1.0, None, 0.0)):
      w(t) = aff(fam(letter), fam(t)) ** power
      w(t) = (1 - mix) * w(t) + mix * 1/(|A| - 1)   # only when mix > 0
    Renormalized to Σq=1; empty pre-filter → ()."""
    src_fam = ALPHABET[letter]
    src_gran = GRAN[letter]
    src_len = MOTIF_LEN.get(letter)
    power, mask, mix = SPECTRUM_SHAPE.get(letter, (1.0, None, 0.0))
    src_rank = FAMILY_RANK[src_fam]
    A = len(ALPHABET)

    survivors: dict[str, float] = {}
    for t in ALPHABET:
        if t == letter:
            continue
        if GRAN[t] != src_gran:
            continue
        if src_gran == "motif" and MOTIF_LEN[t] != src_len:
            continue
        if mask is None:
            pass
        elif mask == "adjacent":
            if abs(FAMILY_RANK[ALPHABET[t]] - src_rank) != 1:
                continue
        else:
            if ALPHABET[t] != mask:
                continue
        w = affinity(src_fam, ALPHABET[t]) ** power
        if mix > 0.0:
            w = (1.0 - mix) * w + mix * (1.0 / (A - 1))
        survivors[t] = w

    tot = sum(survivors.values())
    if tot == 0.0:
        return ()
    return tuple((t, w / tot) for t, w in sorted(survivors.items()))


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


def feature_mask_of(sequence: tuple[str, ...]) -> int:
    """Predicate-bit feature mask for a sequence (S6 §3.5 + S1 §3.3).
    Sets:
      - family_<X> for every letter present (X = ALPHABET[letter]),
      - motif_<X> if the sequence has at least one motif block of family X,
      - motif3_<X> if the sequence has a motif block of family X with MOTIF_LEN>=3,
      - vis_lowvis if the sequence has any fam=N letter with VIS[letter] <= 0.20,
      - thr_crest if any F letter has f >= 0.5 (S3),
      - thr_hotspot if any P letter has p_add >= 0.05 (S3),
      - thr_mirror if any Z letter has z <= 0.45 and |prey| >= 2 (S3).
    Reserved bits initially 0; all conditions are OR'd.
    Pure function of the sequence — reads only registry tables."""
    m = 0
    vis_lowvis_found = False
    thr_crest_found = False
    thr_hotspot_found = False
    thr_mirror_found = False

    # Single pass over sequence for family bits and threshold conditions
    for letter in sequence:
        fam = ALPHABET.get(letter)
        if fam is None:
            continue
        m |= PREDICATE_BIT[f"family_{fam}"]

        # S1: vis_lowvis — fam=N and VIS<=0.20
        if not vis_lowvis_found and fam == "N" and VIS.get(letter, 0.0) <= 0.20:
            m |= PREDICATE_BIT["vis_lowvis"]
            vis_lowvis_found = True

        # S3: thr_crest — F letter with f >= 0.5
        if not thr_crest_found and letter in _F and _F[letter][0] >= 0.5:
            m |= PREDICATE_BIT["thr_crest"]
            thr_crest_found = True

        # S3: thr_hotspot — P letter with p_add >= 0.05
        if not thr_hotspot_found and letter in _P and _P[letter][0] >= 0.05:
            m |= PREDICATE_BIT["thr_hotspot"]
            thr_hotspot_found = True

        # S3: thr_mirror — Z letter with z <= 0.45 and |prey| >= 2
        if (not thr_mirror_found and letter in _Z
                and _Z[letter][0] <= 0.45
                and _Z_PREY_CARD[letter] >= 2):
            m |= PREDICATE_BIT["thr_mirror"]
            thr_mirror_found = True

    # Motif block processing (requires motif_blocks scan)
    for s, e, letter in motif_blocks(sequence):
        if GRAN.get(letter) != "motif":
            continue
        fam = ALPHABET.get(letter)
        if fam is None:
            continue
        m |= PREDICATE_BIT[f"motif_{fam}"]
        if MOTIF_LEN[letter] >= 3 and fam in ("F", "P", "Z"):
            m |= PREDICATE_BIT[f"motif3_{fam}"]

    return m


def prey_mask_for_clauses(prey_clauses: tuple[tuple[str, ...], ...]) -> int:
    """Predicate-bit prey mask for a Z row's clause list (S6 §3.5 + S3 §3.2).
    Each clause is a tuple whose first element is the family ('F'|'P'|'Z'|'N')
    and whose optional further elements specialize the predicate:
      ('F',)                     → family_F bit                      (S6)
      ('F', 'motif')             → motif_F bit                       (S6)
      ('F', 'motif', 'len>=3')   → motif3_F bit                      (S6)
      ('F', 'f_hi')              → thr_crest bit                     (S3 — Crest Bite)
      ('P', 'p_hi')              → thr_hotspot bit                   (S3 — Hotspot Snipe)
      ('Z', 'generalist')        → thr_mirror bit                    (S3 — Mirror Fang)
      ('N', 'lowvis')            → vis_lowvis bit                    (S3 — Void Bite; S1 reserved bit 11)
      ('F', '<unknown_tag>')     → family_F bit (fall through, forward-compat)
    OR the selected bits to form prey_mask. Pure function of the clause list."""
    m = 0
    for clause in prey_clauses:
        if not clause:
            continue
        fam = clause[0]
        tags = clause[1:]
        if "motif" in tags and "len>=3" in tags:
            if fam in ("F", "P", "Z"):
                m |= PREDICATE_BIT[f"motif3_{fam}"]
        elif "motif" in tags:
            m |= PREDICATE_BIT[f"motif_{fam}"]
        elif "f_hi" in tags and fam == "F":
            m |= PREDICATE_BIT["thr_crest"]
        elif "p_hi" in tags and fam == "P":
            m |= PREDICATE_BIT["thr_hotspot"]
        elif "generalist" in tags and fam == "Z":
            m |= PREDICATE_BIT["thr_mirror"]
        elif "lowvis" in tags and fam == "N":
            m |= PREDICATE_BIT["vis_lowvis"]
        else:
            # forward-compat fallback: unknown tag (or no tag) → family bit
            m |= PREDICATE_BIT[f"family_{fam}"]
    return m


def n_locked(layout: tuple[str, ...], chan: str) -> int:
    """Count locked-position blocks whose family equals `chan`. Motif and
    residue blocks each count as 1 (per primitive-roster.md OPEN-1 ②).
    Only blocks whose entire span lies inside `_LOCKED.keys()` are counted —
    a motif straddling locked and non-locked positions is excluded.
    `chan` must be one of {"F", "P", "Z"}; "N" is never counted (spec §3.4)."""
    if chan not in ("F", "P", "Z"):
        raise ValueError(
            f"n_locked: chan must be one of F/P/Z, got {chan!r} (N never counts)")
    locked_positions = set(_LOCKED.keys())
    count = 0
    for s, e, letter in motif_blocks(layout):
        if not all(k in locked_positions for k in range(s, e)):
            continue
        if ALPHABET.get(letter) == chan:
            count += 1
    return count


def phenotype(sequence: tuple[str, ...]) -> Phenotype:
    """Pure function of the sequence only. No world-state, no neighbors, no tick.
    κ=0 in v1 — no self-coordination neighbor scan. S6: feature_mask and prey_mask
    are predicate-bit ORs (not per-letter ORs); the antagonism kernel match
    expression is unchanged."""
    f_prod = 1.0          # accumulate Π(1-fᵢ)
    pl_prod = 1.0
    px_prod = 1.0
    z_sum = 0.0
    vis_sum = 0.0
    n_count = 0
    vis_mode = 0          # S1: max-over-Z-rows vis_mode
    prey_clauses: list[tuple[str, ...]] = []
    directions: list[tuple[int, int]] = []
    periods: list[int] = []
    f_periods: list[int] = []
    z_periods: list[int] = []
    phase_type: PhaseType | None = None
    dominant_p: str | None = None
    in_place = False           # S4: FSTACK 标志
    rand_dir = False           # S4: FDRIFT 标志
    # S5: dominant-F tracking for phase-window resolution
    dom_f_value: float = -1.0
    dom_f_lo: float = 0.0
    dom_burst_w: int = 1
    dom_burst_k: int = 1
    f_each: list[tuple[str, float, float]] = []  # (letter, f_val, f_lo_row)

    for letter in sequence:
        if letter not in ALPHABET:
            continue
        if ALPHABET.get(letter) == "N":
            vis_sum += VIS[letter]
            n_count += 1
        if letter in _F:
            f_val, dirs_spec, pl, per, f_lo_row, b_w, b_k = _F[letter]
            f_prod *= (1 - f_val)
            pl_prod *= (1 - pl)
            # S5: record for dominant-F tracking
            f_each.append((letter, f_val, f_lo_row))
            if f_val > dom_f_value:
                dom_f_value = f_val
                dom_f_lo = f_lo_row
                dom_burst_w = b_w
                dom_burst_k = b_k
            # S4: dirs_spec 三态. tuple → 字面方向 (旧路径, OR 进 directions/dir_bits);
            # "hash:<kind>" → mint 时调 _hash_dirs(sequence, kind) → 同样 OR 进;
            # "rand:1of4" → 不预写方向, 设 rand_dir=True, kernel 每 tick 现抽.
            if isinstance(dirs_spec, str):
                if dirs_spec == "rand:1of4":
                    rand_dir = True
                elif dirs_spec.startswith("hash:"):
                    kind = dirs_spec[len("hash:"):]
                    for d in _hash_dirs(sequence, kind):
                        if d not in directions:
                            directions.append(d)
                else:
                    raise ValueError(
                        f"_F[{letter!r}].directions: unknown spec {dirs_spec!r}; "
                        "expected tuple, 'hash:<kind>', or 'rand:1of4'")
            else:
                # 字面 tuple. (IN_PLACE_DIR,) 即 ((0, 0),) → in_place=True;
                # 其他字面方向 OR 进 directions 列表 (旧路径).
                if dirs_spec == (IN_PLACE_DIR,):
                    in_place = True
                else:
                    for d in dirs_spec:
                        if d not in directions:
                            directions.append(d)
            periods.append(per)
            f_periods.append(per)
            phase_type = PhaseType.REPRODUCTION
        elif letter in _Z:
            row = _Z[letter]
            z, clauses, per = row[0], row[1], row[2]
            mode = row[3] if len(row) >= 4 else 0
            if mode > vis_mode:
                vis_mode = mode
            z_sum += z
            prey_clauses.extend(clauses)
            periods.append(per)
            z_periods.append(per)
            if phase_type is None:
                phase_type = PhaseType.ANTAGONISM
        elif letter in _P:
            p_add, per = _P[letter]
            px_prod *= (1 - min(P_MAX, MU + p_add))
            periods.append(per)
            if dominant_p is None or p_add > _P[dominant_p][0]:
                dominant_p = letter

    f = 1 - f_prod
    p_leave = 1 - pl_prod
    p_x = max(MU, 1 - px_prod)
    spectrum = _spectrum_for(dominant_p) if dominant_p else ()
    period = min(periods) if periods else 1
    repro_period = min(f_periods) if f_periods else 1
    anta_period = min(z_periods) if z_periods else 1
    dir_bits = 0
    for d in directions:
        dir_bits |= _DIR_BIT.get(d, 0)

    # S5: stacked f_lo = 1 - (1 - dom_f_lo) * Π(1 - f_i) for all non-dominant F letters.
    # dominant = highest f_val, first occurrence on tie (max() is stable for first-max).
    if not f_each:
        f_lo_stacked = 0.0
        burst_w_out = 1
        burst_k_out = 1
    else:
        dom_idx = max(range(len(f_each)), key=lambda i: f_each[i][1])
        non_dom_prod = 1.0
        for i, (_, f_i, _) in enumerate(f_each):
            if i == dom_idx:
                continue
            non_dom_prod *= (1 - f_i)
        f_lo_stacked = 1.0 - (1.0 - dom_f_lo) * non_dom_prod
        burst_w_out = int(dom_burst_w)
        burst_k_out = int(dom_burst_k)

    # S6: predicate-bit masks. Antagonism kernel match expression unchanged.
    feature_mask = feature_mask_of(sequence)
    prey_mask = prey_mask_for_clauses(tuple(prey_clauses))

    return Phenotype(
        f=f, directions=tuple(directions), p_leave=p_leave, z_raw=z_sum,
        prey_mask=prey_mask, feature_mask=feature_mask, p_x=p_x,
        spectrum=spectrum, period=period,
        repro_period=repro_period, anta_period=anta_period, dir_bits=dir_bits,
        phase_type=phase_type, fold=(),
        vis_sum=vis_sum, n_count=n_count, vis_mode=vis_mode,
        in_place=in_place, rand_dir=rand_dir,
        f_hi=f, f_lo=f_lo_stacked, burst_w=burst_w_out, burst_k=burst_k_out,
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
    """Enforce the BB0 symmetry invariant (viz spec §5 / red-line 4) + S6
    motif contiguity. locked positions must equal _LOCKED; backbone (non-locked,
    non-slot) positions must stay "N0"; only _SLOTS positions may vary, and
    only to a primitive in the current palette. If any motif letter is present,
    its blocks must be contiguous and exactly MOTIF_LEN positions long.
    Raises ValueError on any violation."""
    if len(layout) != 16:
        raise ValueError(f"BB0 layout must have 16 positions, got {len(layout)}")
    # S6: motif contiguity check (no-op for all-residue layouts).
    has_motif = any(GRAN.get(ltr) == "motif" for ltr in layout)
    if has_motif:
        # Walk left-to-right; whenever we see a motif letter, ensure exactly
        # MOTIF_LEN[letter] consecutive copies starting here, then jump past.
        i = 0
        n = len(layout)
        while i < n:
            ltr = layout[i]
            if GRAN.get(ltr) == "motif":
                need = MOTIF_LEN[ltr]
                end = i + need
                if end > n or any(layout[k] != ltr for k in range(i, end)):
                    raise ValueError(
                        f"motif {ltr!r} at position {i} requires {need} contiguous "
                        f"copies; got layout[{i}:{end}] = {layout[i:end]}")
                i = end
            else:
                i += 1
    # legacy per-position invariant (residues + locked + backbone)
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
