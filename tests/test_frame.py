# tests/test_frame.py
import json
import torch
from des.world import World
from des.phenotype_cache import StrainTable
from des.registry import BB0_TEMPLATE
from webapp.frame import encode_frame, cell_detail

DEV = torch.device("cpu")


def _world_with(table):
    # 4x4 world, K=8. cell (1,1): faction0 BB0 count5. cell (2,2): faction1 BB0 count3 + faction2 BB0 count2.
    w = World(4, 4, 8, DEV)
    bb0 = table.get_or_mint(BB0_TEMPLATE["layout"])
    w.strain_id[1, 1, 0] = bb0; w.count[1, 1, 0] = 5; w.faction[1, 1, 0] = 0
    w.strain_id[2, 2, 0] = bb0; w.count[2, 2, 0] = 3; w.faction[2, 2, 0] = 1
    w.strain_id[2, 2, 1] = bb0; w.count[2, 2, 1] = 2; w.faction[2, 2, 1] = 2
    return w


def test_encode_frame_only_nonempty_cells():
    t = StrainTable(); w = _world_with(t)
    fr = encode_frame(w, t, tick=7, H=4, W=4)
    assert fr["tick"] == 7 and fr["H"] == 4 and fr["W"] == 4
    assert len(fr["cells"]) == 2                      # only the 2 occupied cells
    # frame must be JSON-serializable (no torch scalars leaking through)
    json.dumps(fr)


def test_encode_frame_faction_counts_per_cell():
    t = StrainTable(); w = _world_with(t)
    fr = encode_frame(w, t, tick=0, H=4, W=4)
    cells = {(c[0], c[1]): c[2:] for c in fr["cells"]}
    assert cells[(1, 1)] == [5, 0, 0, 0]              # faction0 = 5
    assert cells[(2, 2)] == [0, 3, 2, 0]              # faction1 = 3, faction2 = 2


def test_encode_frame_readouts_match_shared_fn():
    t = StrainTable(); w = _world_with(t)
    fr = encode_frame(w, t, tick=0, H=4, W=4)
    r = fr["readouts"]
    assert r["total"] == 10                           # 5 + 3 + 2
    assert r["occupied_cells"] == 2
    assert abs(r["faction_share"][0] - 0.5) < 1e-9    # 5/10


def test_cell_detail_lists_strains_in_cell():
    t = StrainTable(); w = _world_with(t)
    d = cell_detail(w, t, y=2, x=2)
    assert d["y"] == 2 and d["x"] == 2
    seq = ".".join(BB0_TEMPLATE["layout"])
    assert {"strain": seq, "faction": 1, "count": 3} in d["strains"]
    assert {"strain": seq, "faction": 2, "count": 2} in d["strains"]
    assert len(d["strains"]) == 2


def test_cell_detail_empty_cell():
    t = StrainTable(); w = _world_with(t)
    d = cell_detail(w, t, y=0, x=0)
    assert d["strains"] == []


def test_leaderboard_ranks_strains_by_total_count():
    # add a second distinct strain so the leaderboard has >1 entry to rank
    t = StrainTable(); w = _world_with(t)
    lay2 = list(BB0_TEMPLATE["layout"]); lay2[0] = "P_hotspot"
    s2 = t.get_or_mint(tuple(lay2))
    w.strain_id[0, 0, 0] = s2; w.count[0, 0, 0] = 100; w.faction[0, 0, 0] = 3
    fr = encode_frame(w, t, tick=0, H=4, W=4, top_n=5)
    lb = fr["leaderboard"]
    # BB0 total = 5+3+2 = 10; s2 total = 100 -> s2 ranks first
    assert lb[0]["strain"] == ".".join(lay2) and lb[0]["count"] == 100
    assert lb[0]["faction"] == 3                       # dominant faction of s2
    assert abs(lb[0]["share"] - 100 / 110) < 1e-9      # total world = 110
    assert lb[1]["count"] == 10                        # BB0 second


def test_leaderboard_top_n_caps_length():
    t = StrainTable(); w = _world_with(t)
    fr = encode_frame(w, t, tick=0, H=4, W=4, top_n=1)
    assert len(fr["leaderboard"]) == 1
