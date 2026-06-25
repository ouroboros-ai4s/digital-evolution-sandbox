import pytest
import torch
from des.run import PALETTE, layout_from_slots, build_engine_from_config
from webapp.server import _jsonable
from des.registry import BB0_TEMPLATE, _SLOTS, _LOCKED

DEV = torch.device("cpu")


def test_palette_is_the_six_v1_primitives():
    assert PALETTE == ["N0", "F4Nr1", "F4Nr4", "P_base", "P_hotspot", "BroadSweep"]


def test_layout_from_empty_slots_is_canonical_bb0():
    assert layout_from_slots({}) == BB0_TEMPLATE["layout"]


def test_layout_from_slots_fills_mutable_positions():
    lay = layout_from_slots({0: "P_hotspot", 13: "F4Nr1"})
    assert lay[0] == "P_hotspot" and lay[13] == "F4Nr1"
    assert lay[1] == _LOCKED[1]                # locked untouched
    assert lay[4] == "N0"                       # backbone untouched


def test_layout_from_slots_rejects_non_slot_index():
    with pytest.raises(ValueError, match="slot"):
        layout_from_slots({4: "F4Nr1"})         # 4 is backbone, not a slot


def test_layout_from_slots_rejects_unknown_primitive():
    with pytest.raises(ValueError, match="palette"):
        layout_from_slots({0: "NOPE"})


def test_build_engine_defaults():
    eng, cfg = build_engine_from_config(
        {"players": [{"slots": {}} for _ in range(4)]}, DEV)
    assert cfg["grid"] == 128 and cfg["K"] == 64 and cfg["fill"] == 20
    assert cfg["T"] == 450 and cfg["seed"] == 0 and cfg["z_max"] == 8.0
    assert all(lay == BB0_TEMPLATE["layout"] for lay in cfg["layouts"])
    assert eng.H == 128 and eng.W == 128 and eng.K == 64


def test_build_engine_custom_slot_reaches_world():
    cfg_in = {"players": [{"slots": {2: "F4Nr1"}} for _ in range(4)],
              "grid": 16, "K": 8, "fill": 4, "seed": 1}
    eng, cfg = build_engine_from_config(cfg_in, DEV)
    expect = eng.table.get_or_mint(cfg["layouts"][0])
    assert int(eng.world.strain_id[4, 4, 0]) == expect   # quadrant center of 16-grid


def _four_players(slots_list):
    return {"players": [{"slots": s} for s in slots_list]}


def test_build_engine_four_distinct_players():
    cfg_in = {**_four_players([{0: "N0"}, {0: "F4Nr1"}, {0: "P_base"}, {0: "P_hotspot"}]),
              "grid": 16, "K": 8, "fill": 4, "seed": 1}
    eng, cfg = build_engine_from_config(cfg_in, DEV)
    assert len(cfg["layouts"]) == 4
    centers = [(4, 4), (4, 12), (12, 4), (12, 12)]      # quadrant centers of 16-grid
    seen = set()
    for fac, (cy, cx) in enumerate(centers):
        assert int(eng.world.strain_id[cy, cx, 0]) == eng.table.get_or_mint(cfg["layouts"][fac])
        seen.add(int(eng.world.strain_id[cy, cx, 0]))
    assert len(seen) == 4                               # four genuinely different seeds


def test_build_engine_requires_exactly_four_players():
    with pytest.raises(ValueError, match="4 players"):
        build_engine_from_config(_four_players([{}, {}, {}]), DEV)   # only 3


def test_build_engine_rejects_illegal_slot_in_a_player():
    cfg_in = _four_players([{}, {}, {4: "F4Nr1"}, {}])   # pos 4 is backbone, not a slot
    with pytest.raises(ValueError, match="slot"):
        build_engine_from_config(cfg_in, DEV)


def test_build_engine_global_params_shared_not_per_player():
    eng, cfg = build_engine_from_config(
        {**_four_players([{}, {}, {}, {}]), "grid": 32, "K": 16, "fill": 4}, DEV)
    assert cfg["grid"] == 32 and cfg["K"] == 16          # single global value
    assert eng.H == 32 and eng.K == 16


def test_jsonable_serializes_players_and_layouts():
    _, cfg = build_engine_from_config(_four_players([{}, {}, {}, {}]), DEV)
    j = _jsonable(cfg)
    assert isinstance(j["layouts"], list) and len(j["layouts"]) == 4
    assert all(isinstance(lay, list) for lay in j["layouts"])
    assert isinstance(j["players"], list) and len(j["players"]) == 4


def test_make_app_registers_routes():
    from webapp.server import make_app, PLAYGROUND_DIR
    app = make_app(device=torch.device("cpu"))
    paths = {r.resource.canonical for r in app.router.routes() if r.resource is not None}
    assert {"/", "/config", "/ws", "/api/frame_at_tick", "/api/cell", "/api/trajectory"} <= paths
    assert "playground" in PLAYGROUND_DIR and "runs" not in PLAYGROUND_DIR
