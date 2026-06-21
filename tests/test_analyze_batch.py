import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import pandas as pd
import analyze_batch as ab

def _toy(path, rows):
    """rows: list of (tick, cell_x, cell_y, strain, faction, count)."""
    df = pd.DataFrame(rows, columns=["tick", "cell_x", "cell_y", "strain", "faction", "count"])
    df.to_parquet(path)
    return df

def test_load_roundtrips_schema(tmp_path):
    p = tmp_path / "t.parquet"
    _toy(p, [(1, 0, 0, "S0", 0, 10), (1, 1, 1, "S0", 1, 10)])
    df = ab.load(str(p))
    assert list(df.columns) == ["tick", "cell_x", "cell_y", "strain", "faction", "count"]
    assert len(df) == 2
    assert df["strain"].tolist() == ["S0", "S0"]


def test_survival_spatial_known_answers(tmp_path):
    # 2x2 grid (n_cells=4). tick1: 4 separate cells, factions 0..3, strain S0.
    # tick2: faction1 invades cell(0,0) -> cross-faction at tick2.
    # tick3: only faction 0 remains anywhere -> fixation at tick3.
    rows = [
        (1, 0, 0, "S0", 0, 10), (1, 1, 0, "S0", 1, 10),
        (1, 0, 1, "S0", 2, 10), (1, 1, 1, "S0", 3, 10),
        (2, 0, 0, "S0", 0, 8), (2, 0, 0, "S0", 1, 5),   # cell(0,0) now 2 factions
        (2, 1, 0, "S0", 1, 10), (2, 0, 1, "S0", 2, 10), (2, 1, 1, "S0", 3, 10),
        (3, 0, 0, "S0", 0, 20), (3, 1, 0, "S0", 0, 5),  # only faction 0 left
    ]
    p = tmp_path / "s.parquet"; _toy(p, rows)
    m = ab.survival_spatial_metrics(ab.load(str(p)), n_cells=4)
    assert m["ticks"] == [1, 2, 3]
    assert m["total_count"] == {1: 40, 2: 43, 3: 25}
    assert m["extinction_tick"] is None
    assert m["distinct_factions"] == {1: 4, 2: 4, 3: 1}
    assert m["fixation_tick"] == 3
    assert m["occupied_cells"] == {1: 4, 2: 4, 3: 2}
    assert m["first_cross_faction_tick"] == 2
    assert m["fill_tick"] == 1            # 4 occupied cells == n_cells at tick 1
    assert m["winner_faction"] == 0
    assert abs(m["faction_share"][1][0] - 0.25) < 1e-9


def test_diversity_known_answers(tmp_path):
    # tick1: 2 strains equal freq (10 each) -> N2 = 1/(0.5^2+0.5^2) = 2.0
    # tick2: A=30 B=10 -> freqs .75/.25 -> N2 = 1/(.5625+.0625)=1.6
    # new strain C first seen tick2. leader: A both ticks -> 0 changes.
    rows = [
        (1, 0, 0, "A", 0, 10), (1, 1, 0, "B", 0, 10),
        (2, 0, 0, "A", 0, 30), (2, 1, 0, "B", 0, 10), (2, 2, 0, "C", 0, 0),
    ]
    p = tmp_path / "d.parquet"; _toy(p, rows)
    m = ab.diversity_metrics(ab.load(str(p)))
    assert m["distinct_strains"][1] == 2
    assert abs(m["n2"][1] - 2.0) < 1e-9
    assert abs(m["n2"][2] - 1.6) < 1e-9
    assert m["new_strain_first_seen"]["A"] == 1
    assert m["new_strain_first_seen"]["C"] == 2   # count 0 still records first-seen row
    assert abs(m["d_max"][2] - 0.75) < 1e-9
    assert m["leader_changes"] == 0

def test_proxy_and_seeding_known_answers(tmp_path):
    # seed tick = 1: 1 strain, 4 factions. tick2 cell(0,0) f0 drops 10->4 = -6 proxy.
    rows = [
        (1, 0, 0, "S0", 0, 10), (1, 1, 0, "S0", 1, 10),
        (1, 0, 1, "S0", 2, 10), (1, 1, 1, "S0", 3, 10),
        (2, 0, 0, "S0", 0, 4), (2, 1, 0, "S0", 1, 10),
        (2, 0, 1, "S0", 2, 10), (2, 1, 1, "S0", 3, 10),
    ]
    p = tmp_path / "s2.parquet"; _toy(p, rows)
    m = ab.proxy_and_seeding_metrics(ab.load(str(p)))
    assert m["seed_tick"] == 1
    assert m["seed_distinct_strains"] == 1
    assert m["seed_distinct_factions"] == 4
    assert m["net_decrease_proxy"][2] == 6     # only the -6 drop, summed as magnitude

def test_per_seed_merges_all(tmp_path):
    rows = [(1, 0, 0, "S0", 0, 10), (1, 1, 1, "S0", 1, 10)]
    p = tmp_path / "m.parquet"; _toy(p, rows)
    m = ab.per_seed_metrics(ab.load(str(p)), n_cells=4)
    # has keys from all three metric groups
    assert "fixation_tick" in m and "n2" in m and "seed_distinct_factions" in m

def test_xtab_single_faction_strain_excluded_and_two_faction_included(tmp_path):
    """Test xtab with strains under 1 vs 2+ factions at last tick.
    SOLO: single-faction -> must be excluded (used to crash on scalar collapse).
    MIX: two-faction -> must be included with correct faction->count mapping."""
    rows = [
        (1, 0, 0, "SOLO", 0, 5), (1, 1, 0, "MIX", 0, 5),
        (2, 0, 0, "SOLO", 0, 7),
        (2, 1, 0, "MIX", 0, 4), (2, 2, 0, "MIX", 1, 6),
    ]
    p = tmp_path / "x.parquet"
    _toy(p, rows)
    m = ab.proxy_and_seeding_metrics(ab.load(str(p)))
    xt = m["strain_faction_xtab"]
    assert "SOLO" not in xt, "single-faction strain must be excluded"
    assert xt["MIX"] == {0: 4, 1: 6}, f"two-faction strain must have correct counts, got {xt.get('MIX')}"


def test_cross_seed_winrate_and_flags(tmp_path):
    # 3 fake per-seed dicts with winners 0,1,0 and short steady windows
    per_seed = []
    for w, last in [(0, 50), (1, 50), (0, 50)]:
        per_seed.append({
            "seed": None, "winner_faction": w,
            "ticks": list(range(1, last + 1)),
            "fixation_tick": 40, "first_cross_faction_tick": 20, "fill_tick": 35,
            "faction_occupied": {1: {0: 1, 1: 1, 2: 1, 3: 1}},
        })
    c = ab.cross_seed_metrics(per_seed)
    assert c["n_seeds"] == 3
    assert c["winners"] == [0, 1, 0]
    assert c["win_counts"][0] == 2 and c["win_counts"][1] == 1
    assert abs(c["win_ci_note"]["expected_share"] - 0.25) < 1e-9
    # steady window = last_tick - fill_tick = 50 - 35 = 15 < 200 -> short-run flagged
    assert c["gate0_short_run"]["steady_window_ticks"] <= 15
    assert "200" in str(c["gate0_short_run"]["required"]) or c["gate0_short_run"]["required"] == 200
    assert c["timeline_reconciliation"]["spec_meet"] == 160


import json

def test_render_text_no_verdicts_and_has_proxy_label(tmp_path):
    rows = [(1, 0, 0, "S0", 0, 10), (1, 1, 1, "S0", 1, 10)]
    p = tmp_path / "r.parquet"; df = _toy(p, rows)
    ps = ab.per_seed_metrics(ab.load(str(p)), n_cells=4)
    cross = ab.cross_seed_metrics([ps])
    txt = ab.render_text([ps], cross)
    assert isinstance(txt, str) and len(txt) > 0
    # report-only discipline: no PASS/FAIL tokens anywhere (spec §0.1)
    up = txt.upper()
    assert "PASS" not in up and "FAIL" not in up
    # proxy must be labeled (spec §0.2)
    assert "PROXY" in up

def test_dump_json_roundtrips(tmp_path):
    rows = [(1, 0, 0, "S0", 0, 10)]
    p = tmp_path / "j.parquet"; _toy(p, rows)
    ps = ab.per_seed_metrics(ab.load(str(p)), n_cells=4)
    cross = ab.cross_seed_metrics([ps])
    out = tmp_path / "a.json"
    ab.dump_json({"per_seed": [ps], "cross_seed": cross}, str(out))
    loaded = json.loads(out.read_text())
    assert "per_seed" in loaded and "cross_seed" in loaded
