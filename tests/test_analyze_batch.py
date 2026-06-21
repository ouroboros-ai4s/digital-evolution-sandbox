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
