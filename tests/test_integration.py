# tests/test_integration.py
import torch, pyarrow.parquet as pq
from des.engine import Engine
from des.recorder import Recorder
from des.registry import phenotype, BB0_TEMPLATE
from des.phenotype_cache import StrainTable

DEV = torch.device("cpu")

# --- red-line regression (project survival line) ---

def test_g10_no_self_annihilation_at_t0():
    # four isolated faction seeds: no two factions share a cell at t=0 -> no antagonism
    # losses on the first tick. World count does not drop on tick 0.
    e = Engine(H=16, W=16, K=32, seed=0, device=DEV, fill_per_cell=20)
    before = e.total_count()
    e.step()
    assert e.total_count() >= before

def test_phenotype_reads_only_sequence():
    # same sequence -> identical phenotype regardless of any world state
    seq = ("F4Nr1", "N0", "BroadSweep")
    assert phenotype(seq) == phenotype(seq)
    # phenotype signature takes ONLY the sequence (no count/opponent/tick/occupancy arg)
    import inspect
    params = list(inspect.signature(phenotype).parameters)
    assert params == ["sequence"]

def test_kwall_equal_ratio_no_hidden_weight():
    from des.kernels.arbitration import phase3_arbitrate_vec
    from des.kernels.reproduction import ArrivalBuffer
    survivors = {5: 0, 7: 0}
    for s in range(200):
        sid = torch.zeros((1,1,16), dtype=torch.int32)
        cnt = torch.zeros((1,1,16), dtype=torch.int32)
        fac = torch.zeros((1,1,16), dtype=torch.int8)
        birth = torch.zeros((1,1,16), dtype=torch.int32)
        buf = ArrivalBuffer(DEV)
        buf.add(torch.tensor([0]),torch.tensor([0]),torch.tensor([5],dtype=torch.int32),
                torch.tensor([100],dtype=torch.int32),torch.tensor([0],dtype=torch.int8))
        buf.add(torch.tensor([0]),torch.tensor([0]),torch.tensor([7],dtype=torch.int32),
                torch.tensor([100],dtype=torch.int32),torch.tensor([1],dtype=torch.int8))
        g = torch.Generator(device=DEV); g.manual_seed(s)
        nsid, ncnt, nfac, _ = phase3_arbitrate_vec(sid, cnt, fac, buf.tensors(), K=16,
                                 birth_tick=birth, T=1, generator=g, MAXSID=16, NFAC=4)
        for k in range(16):
            if ncnt[0,0,k] > 0:
                survivors[int(nsid[0,0,k])] += int(ncnt[0,0,k])
    r = survivors[5] / max(1, survivors[7])
    assert 0.85 < r < 1.18, f"strain-blind violated: ratio={r}"

# --- body-liveness acceptance (the eyeball gate as assertions) ---

def test_mutants_appear():
    e = Engine(H=32, W=32, K=32, seed=3, device=DEV)
    start = len(e.table)
    e.run(60)
    assert len(e.table) > start              # mutation minted new strains

def test_frequencies_move_not_frozen():
    e = Engine(H=32, W=32, K=32, seed=3, device=DEV)
    snap0 = e.world.count.clone()
    e.run(40)
    assert not torch.equal(snap0, e.world.count)   # world is not frozen

def test_not_instant_global_death():
    e = Engine(H=32, W=32, K=32, seed=3, device=DEV)
    e.run(40)
    assert e.total_count() > 0               # not a one-step wipeout

def test_full_run_dumps_self_contained_frames(tmp_path):
    t_path = str(tmp_path / "20260620.parquet")
    e = Engine(H=16, W=16, K=16, seed=9, device=DEV)
    rec = Recorder(t_path, e.table)
    e.run(20, recorder=rec)
    rec.close()
    df = pq.read_table(t_path).to_pandas()
    # every dumped tick is a full self-contained frame: reconstruct one tick's world
    ticks = sorted(df["tick"].unique())
    assert len(ticks) >= 1
    frame = df[df["tick"] == ticks[-1]]
    assert (frame["count"] > 0).all()        # only non-empty rows
    assert frame["strain"].str.len().min() > 0
