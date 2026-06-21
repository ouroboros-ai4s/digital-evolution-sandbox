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
