"""S8 A-pool 24 extreme primitives across F/P/Z families.

Pure data module: 8 constants for A-pool extremes (de-gated reachability,
multi-P spectrum blend). Task 1 of S8 — data-only, no consumers yet.
Task 2 will update() these into registry tables.

Spec: docs/superpowers/specs/2026-06-24-s8-*.md
"""
from __future__ import annotations

from des.types import IN_PLACE_DIR


# 24 A-pool letters: 3 groups (乙1/乙2/甲) × 8 letters each
A_FAMILY: dict[str, str] = {
    # 乙1 — 8 escalation copies
    "Apex Bloom":         "F",
    "Ember Drip":         "F",
    "Bastion Pile":       "F",
    "Apex Fang":          "Z",
    "Pan Sweep":          "Z",
    "Hotspot Amp":        "P",
    "Sink Cascade":       "P",
    "Glacial Drift":      "P",
    # 乙2 — 8 native extreme variants
    "F_NOVA":             "F",
    "F_TRICKLE":          "F",
    "F_SCATTER":          "F",
    "Predator Lock":      "Z",
    "Void Bite":          "Z",
    "P_cascade":          "P",
    "P_crossclan_surge":  "P",
    "P_frozen":           "P",
    # 甲 — 8 native extreme variants
    "F8Ar1":              "F",
    "Lance Front":        "F",
    "Ambush Venom":       "Z",
    "Sweep Surge":        "Z",
    "Nip Whisper":        "Z",
    "Coil Null":          "Z",
    "P_zscan_invert":     "P",
    "P_stutter":          "P",
}

# Derive A_GRAN: all residue except Predator Lock (motif)
A_GRAN: dict[str, str] = {
    l: ("motif" if l == "Predator Lock" else "residue")
    for l in A_FAMILY
}

# A_MOTIF_LEN: only Predator Lock is motif-granule
A_MOTIF_LEN: dict[str, int] = {"Predator Lock": 3}

# A_F: 8 F-class rows, S5 7-tuple (f, dirs, p_leave, period, f_lo, burst_w, burst_k)
# Static rows: f_lo=f, burst_w=1, burst_k=1 (degenerate S5 path)
# F_NOVA: windowed — f_lo=0.05, burst_w=20, burst_k=1
A_F: dict[str, tuple] = {
    # 乙1
    "Apex Bloom":   (0.85, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.20, 4, 0.85, 1, 1),
    "Ember Drip":   (0.05, "hash:ember",                       0.04, 9, 0.05, 1, 1),
    "Bastion Pile": (0.85, (IN_PLACE_DIR,),                    0.00, 3, 0.85, 1, 1),
    # 乙2
    "F_NOVA":       (0.85, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.50, 2, 0.05, 20, 1),
    "F_TRICKLE":    (0.02, "hash:trickle",                     0.02, 8, 0.02, 1, 1),
    "F_SCATTER":    (0.12, "hash:scatter3",                    0.60, 3, 0.12, 1, 1),
    # 甲
    "F8Ar1":        (0.25, "rand:1of4",                        0.10, 2, 0.25, 1, 1),
    "Lance Front":  (0.80, "hash:lance",                       0.30, 4, 0.80, 1, 1),
}

# A_Z: 8 Z-class rows, S1 4-tuple (z, prey_clauses, period, vis_mode)
# vis_mode integers: 0=none/uniform, 1=vis-weighted, 2=inverse-vis-weighted
A_Z: dict[str, tuple] = {
    # 乙1
    "Apex Fang":     (1.50, (("Z", "generalist"),),          9, 0),
    "Pan Sweep":     (0.50, (("F",), ("Z",), ("P",)),        6, 0),
    # 乙2
    "Predator Lock": (1.45, (("Z", "motif", "len>=3"),),     9, 0),
    "Void Bite":     (0.95, (("N", "lowvis"),),              5, 1),
    # 甲
    "Ambush Venom":  (1.30, (("F", "motif"),),               7, 0),
    "Sweep Surge":   (0.45, (("F",), ("P",)),                3, 0),
    "Nip Whisper":   (0.15, (("N", "lowvis"),),              3, 1),
    "Coil Null":     (0.20, (("Z",),),                       8, 0),
}

# A_P: 8 P-class rows, 2-tuple (p_add, period)
A_P: dict[str, tuple] = {
    # 乙1
    "Hotspot Amp":       (0.30, 3),
    "Sink Cascade":      (0.25, 3),
    "Glacial Drift":     (0.0,  12),
    # 乙2
    "P_cascade":         (0.28, 2),
    "P_crossclan_surge": (0.20, 4),
    "P_frozen":          (0.0,  8),
    # 甲
    "P_zscan_invert":    (0.10, 4),
    "P_stutter":         (0.32, 2),
}

# A_SHAPE: 8 P-class rows, 3-tuple (power, family_mask, flatten_mix)
# Note: P_frozen and P_stutter use power=4.0; P_crossclan_surge uses family_mask="cross"
A_SHAPE: dict[str, tuple] = {
    "Hotspot Amp":       (1.0, None,    0.0),
    "Sink Cascade":      (1.0, "N",     0.0),
    "Glacial Drift":     (1.0, None,    0.0),
    "P_cascade":         (1.0, None,    0.0),
    "P_crossclan_surge": (1.0, "cross", 0.0),
    "P_frozen":          (4.0, None,    0.0),
    "P_zscan_invert":    (1.0, "F",     0.0),
    "P_stutter":         (4.0, None,    0.0),
}

# A_SLOTS: 24 rows (P_cascade=2, all others=1)
A_SLOTS: dict[str, int] = {
    letter: (2 if letter == "P_cascade" else 1)
    for letter in A_FAMILY
}
