# tests/test_recorder.py
import torch, pyarrow.parquet as pq
from des.world import World, init_bb0, init_factions
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
    assert set(tbl.columns) == {"tick", "cell_x", "cell_y", "strain", "faction", "count"}
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

def test_close_is_idempotent(tmp_path):
    t = StrainTable()
    w = init_bb0(2, 2, 16, DEV, t, fill_per_cell=5)
    path = str(tmp_path / "run.parquet")
    rec = Recorder(path, t)
    rec.dump(0, w)
    rec.close()
    rec.close()                                 # second call must be a no-op, not raise
    tbl = pq.read_table(path).to_pandas()
    assert len(tbl) == 4

def test_writer_thread_error_surfaces(tmp_path):
    # a sid not in the table makes the worker's sequence_of raise -> must surface, not deadlock
    t = StrainTable()
    w = World(2, 2, 16, DEV)
    w.strain_id[0, 0, 0] = 999999               # never minted -> KeyError in worker
    w.count[0, 0, 0] = 1
    path = str(tmp_path / "run.parquet")
    rec = Recorder(path, t)
    rec.dump(0, w)
    # the failure surfaces on close() (or a later dump()), never silently swallowed
    import pytest
    with pytest.raises(RuntimeError, match="writer thread died"):
        rec.close()

def test_dump_records_faction(tmp_path):
    from des.world import init_factions
    t = StrainTable()
    w = init_factions(16, 16, 32, DEV, t, fill_per_cell=5, n_fac=4)
    path = str(tmp_path / "run.parquet")
    rec = Recorder(path, t); rec.dump(0, w); rec.close()
    tbl = pq.read_table(path).to_pandas()
    assert len(tbl) == 4                                  # 4 seeded cells
    assert set(tbl["faction"].tolist()) == {0, 1, 2, 3}   # one row per faction
    assert (tbl["count"] == 5).all()
