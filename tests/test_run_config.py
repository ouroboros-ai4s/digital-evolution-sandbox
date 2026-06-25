import pytest
import torch
from des.run import (
    PALETTE, _DEFAULTS, layout_from_slots, build_engine_from_config,
    pick_device, compute_match_result,
)
from webapp.readouts import compute_readouts

DEV = torch.device("cpu")


def test_palette_lifted_unchanged():
    assert PALETTE == ["N0", "F4Nr1", "F4Nr4", "P_base", "P_hotspot", "BroadSweep"]


def test_pick_device_force_cpu_overrides_cuda_arg():
    assert pick_device(device=torch.device("cuda"), force_cpu=True) == torch.device("cpu")


def test_pick_device_passthrough():
    d = torch.device("cpu")
    assert pick_device(device=d) is d


def test_compute_match_result_shape_and_single_source():
    cfg = {"players": [{"slots": {}} for _ in range(4)],
           "grid": 16, "K": 8, "fill": 4, "T": 3, "seed": 0}
    eng, _ = build_engine_from_config(cfg, DEV)
    eng.run(3, recorder=None, stop_on=())
    res = compute_match_result(eng, "data/runs/fake.parquet")
    assert res["path"] == "data/runs/fake.parquet"
    assert res["ticks"] == 3
    final = res["final"]
    for k in ("total", "occupied_cells", "distinct_strains", "n2", "d_max", "faction_share"):
        assert k in final
    # single-source: faction_share matches a direct compute_readouts call
    cnt = eng.world.count.cpu(); sid = eng.world.strain_id.cpu(); fac = eng.world.faction.cpu()
    nz = torch.nonzero(cnt > 0, as_tuple=False)
    ys = nz[:, 0].tolist(); xs = nz[:, 1].tolist(); ks = nz[:, 2]
    sids = sid[nz[:, 0], nz[:, 1], ks].tolist()
    facs = fac[nz[:, 0], nz[:, 1], ks].tolist()
    cnts = cnt[nz[:, 0], nz[:, 1], ks].tolist()
    strains = [".".join(eng.table.sequence_of(int(s))) for s in sids]
    direct = compute_readouts(xs, ys, strains, facs, cnts)
    assert final["faction_share"] == direct["faction_share"]
    assert final["total"] == direct["total"]
