# tests/test_readouts.py
from webapp.readouts import compute_readouts


def test_empty_is_all_zero():
    r = compute_readouts([], [], [], [], [])
    assert r == {"total": 0, "occupied_cells": 0, "distinct_strains": 0,
                 "n2": 0.0, "d_max": 0.0, "faction_share": {}}


def test_two_equal_strains_one_cell_each():
    # two strains 10/10 -> freqs .5/.5 -> N2 = 1/(.25+.25) = 2.0, d_max = .5
    cx = [0, 1]; cy = [0, 0]; strain = ["A", "B"]; fac = [0, 0]; cnt = [10, 10]
    r = compute_readouts(cx, cy, strain, fac, cnt)
    assert r["total"] == 20
    assert r["occupied_cells"] == 2
    assert r["distinct_strains"] == 2
    assert abs(r["n2"] - 2.0) < 1e-9
    assert abs(r["d_max"] - 0.5) < 1e-9
    assert r["faction_share"] == {0: 1.0}


def test_faction_share_quarter_each():
    cx = [0, 1, 2, 3]; cy = [0, 0, 0, 0]
    strain = ["S", "S", "S", "S"]; fac = [0, 1, 2, 3]; cnt = [10, 10, 10, 10]
    r = compute_readouts(cx, cy, strain, fac, cnt)
    for f in (0, 1, 2, 3):
        assert abs(r["faction_share"][f] - 0.25) < 1e-9
    assert r["distinct_strains"] == 1        # all the same strain
    assert abs(r["d_max"] - 1.0) < 1e-9       # one strain owns everything


def test_skewed_strain_freqs():
    # A=30 B=10 -> .75/.25 -> N2 = 1/(.5625+.0625)=1.6, d_max=.75
    cx = [0, 1]; cy = [0, 0]; strain = ["A", "B"]; fac = [0, 0]; cnt = [30, 10]
    r = compute_readouts(cx, cy, strain, fac, cnt)
    assert r["total"] == 40
    assert r["occupied_cells"] == 2
    assert abs(r["n2"] - 1.6) < 1e-9
    assert abs(r["d_max"] - 0.75) < 1e-9


def test_same_cell_multiple_slots_counts_once_as_occupied():
    # two records in the same cell (different slots) -> 1 occupied cell
    cx = [5, 5]; cy = [7, 7]; strain = ["A", "B"]; fac = [0, 1]; cnt = [3, 4]
    r = compute_readouts(cx, cy, strain, fac, cnt)
    assert r["occupied_cells"] == 1
    assert r["total"] == 7
