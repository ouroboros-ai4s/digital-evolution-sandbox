import pytest
import torch
from webapp.server import PALETTE, layout_from_slots, build_engine_from_config
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
    eng, cfg = build_engine_from_config({}, DEV)
    assert cfg["grid"] == 128 and cfg["K"] == 64 and cfg["fill"] == 20
    assert cfg["T"] == 450 and cfg["seed"] == 0 and cfg["z_max"] == 8.0
    assert cfg["layout"] == BB0_TEMPLATE["layout"]
    assert eng.H == 128 and eng.W == 128 and eng.K == 64


def test_build_engine_custom_slot_reaches_world():
    cfg_in = {"slots": {2: "F4Nr1"}, "grid": 16, "K": 8, "fill": 4, "seed": 1}
    eng, cfg = build_engine_from_config(cfg_in, DEV)
    expect = eng.table.get_or_mint(cfg["layout"])
    assert int(eng.world.strain_id[4, 4, 0]) == expect   # quadrant center of 16-grid


def test_make_app_registers_routes():
    from webapp.server import make_app, PLAYGROUND_DIR
    app = make_app(device=torch.device("cpu"))
    paths = {r.resource.canonical for r in app.router.routes() if r.resource is not None}
    assert "/" in paths
    assert "/config" in paths
    assert "/ws" in paths
    assert "/api/frame_at_tick" in paths
    assert "/api/cell" in paths
    assert "/api/trajectory" in paths
    # red-line 1: playground dir is isolated, never data/runs
    assert "playground" in PLAYGROUND_DIR
    assert "runs" not in PLAYGROUND_DIR
