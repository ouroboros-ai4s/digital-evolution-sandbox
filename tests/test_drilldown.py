# tests/test_drilldown.py
import pandas as pd
from webapp.drilldown import frame_at_tick, cell_at_tick, strain_trajectory


def _toy(path):
    # asymmetric cells pin the (y,x) ordering. row = (tick, cell_x, cell_y, strain, faction, count).
    # tick1: cell cx=1,cy=3 -> f0 A=5; cell cx=2,cy=2 -> f1 A=3, f2 B=2.  tick2: cx=1,cy=3 f0 A=7.
    rows = [
        (1, 1, 3, "A", 0, 5), (1, 2, 2, "A", 1, 3), (1, 2, 2, "B", 2, 2),
        (2, 1, 3, "A", 0, 7),
    ]
    df = pd.DataFrame(rows, columns=["tick", "cell_x", "cell_y", "strain", "faction", "count"])
    df.to_parquet(str(path))


def test_frame_at_tick_aggregates_cells(tmp_path):
    p = tmp_path / "r.parquet"; _toy(p)
    fr = frame_at_tick(str(p), tick=1)
    assert fr["tick"] == 1
    # cells rows are [y, x, c0..c3] (same order as encode_frame) -> key by (y, x)
    cells = {(c[0], c[1]): c[2:] for c in fr["cells"]}
    assert sorted(cells.keys()) == sorted([(3, 1), (2, 2)])   # (cy,cx): (3,1) and (2,2)
    assert cells[(2, 2)] == [0, 3, 2, 0]
    assert fr["readouts"]["total"] == 10


def test_frame_at_tick_second_tick(tmp_path):
    p = tmp_path / "r.parquet"; _toy(p)
    fr = frame_at_tick(str(p), tick=2)
    assert len(fr["cells"]) == 1
    assert fr["cells"][0][0] == 3 and fr["cells"][0][1] == 1   # y=3, x=1 ordering pinned
    assert fr["readouts"]["total"] == 7


def test_cell_at_tick_lists_strains(tmp_path):
    p = tmp_path / "r.parquet"; _toy(p)
    d = cell_at_tick(str(p), tick=1, y=2, x=2)
    assert {"strain": "A", "faction": 1, "count": 3} in d["strains"]
    assert {"strain": "B", "faction": 2, "count": 2} in d["strains"]
    assert len(d["strains"]) == 2


def test_cell_at_tick_empty(tmp_path):
    p = tmp_path / "r.parquet"; _toy(p)
    d = cell_at_tick(str(p), tick=2, y=2, x=2)
    assert d["strains"] == []


def test_strain_trajectory_predicate_pushdown(tmp_path):
    p = tmp_path / "r.parquet"; _toy(p)
    traj = strain_trajectory(str(p), strain="A")
    by_tick = {t["tick"]: t for t in traj}
    assert by_tick[1]["total_count"] == 8       # 5 + 3
    assert by_tick[2]["total_count"] == 7
    assert by_tick[1]["occupied_cells"] == 2    # A at (cy3,cx1) + (cy2,cx2)
    # strain B never appears at tick2
    trajB = strain_trajectory(str(p), strain="B")
    assert [t["tick"] for t in trajB] == [1]
