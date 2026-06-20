# tests/test_recorder.py
import torch, pyarrow.parquet as pq
from des.world import World, init_bb0
from des.phenotype_cache import StrainTable
from des.recorder import Recorder

DEV = torch.device("cpu")

def test_dump_writes_nonempty_rows(tmp_path):
    t = StrainTable()
    w = init_bb0(4, 4, 16, DEV, t, fill_per_cell=7)
    path = str(tmp_path / "run.parquet")
    rec = Recorder(path, t)
    rec.dump(0, w)
    rec.close()
    tbl = pq.read_table(path).to_pandas()
    assert set(tbl.columns) == {"tick", "cell_x", "cell_y", "strain", "count"}
    assert len(tbl) == 16                      # 4x4 cells, one strain each, count>0
    assert (tbl["count"] == 7).all()
    assert (tbl["tick"] == 0).all()
    # strain stored as full sequence string, not id 0
    assert tbl["strain"].str.len().min() > 0

def test_empty_cells_not_written(tmp_path):
    t = StrainTable()
    w = World(4, 4, 16, DEV)                    # entirely empty
    sid = t.get_or_mint(("F4Nr1",))
    w.strain_id[0, 0, 0] = sid; w.count[0, 0, 0] = 3
    path = str(tmp_path / "run.parquet")
    rec = Recorder(path, t); rec.dump(5, w); rec.close()
    tbl = pq.read_table(path).to_pandas()
    assert len(tbl) == 1                        # only the one non-empty slot
    assert tbl.iloc[0]["cell_y"] == 0 and tbl.iloc[0]["cell_x"] == 0
    assert tbl.iloc[0]["count"] == 3
