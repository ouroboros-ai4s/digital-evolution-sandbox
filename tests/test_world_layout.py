# tests/test_world_layout.py
import pytest
import torch
from des.registry import BB0_TEMPLATE, validate_bb0_layout, _SLOTS, _LOCKED
from des.world import init_factions
from des.phenotype_cache import StrainTable
from des.engine import Engine

DEV = torch.device("cpu")


def _canonical():
    return list(BB0_TEMPLATE["layout"])


def test_canonical_bb0_passes():
    # default BB0 (all slots N0) is the canonical symmetric genotype
    assert validate_bb0_layout(BB0_TEMPLATE["layout"]) is None


def test_slot_change_passes():
    lay = _canonical()
    lay[0] = "P_hotspot"          # slot 0 is mutable
    lay[13] = "F4Nr1"             # slot 13 is mutable
    assert validate_bb0_layout(tuple(lay)) is None


def test_tampered_locked_position_rejected():
    lay = _canonical()
    lay[1] = "N0"                 # position 1 must stay F4Nr4
    with pytest.raises(ValueError, match="locked"):
        validate_bb0_layout(tuple(lay))


def test_tampered_backbone_position_rejected():
    lay = _canonical()
    lay[4] = "BroadSweep"         # position 4 is backbone-fixed N0, not a slot
    with pytest.raises(ValueError, match="backbone"):
        validate_bb0_layout(tuple(lay))


def test_unknown_primitive_in_slot_rejected():
    lay = _canonical()
    lay[2] = "ZZZ_not_a_primitive"
    with pytest.raises(ValueError, match="palette"):
        validate_bb0_layout(tuple(lay))


def test_wrong_length_rejected():
    with pytest.raises(ValueError, match="16"):
        validate_bb0_layout(("N0",) * 15)


def test_init_factions_default_layout_unchanged():
    # layout=None must reproduce the canonical BB0 seeding (behavior-neutral)
    t = StrainTable()
    w = init_factions(8, 8, 16, DEV, t, fill_per_cell=10, n_fac=4)
    bb0 = t.get_or_mint(BB0_TEMPLATE["layout"])
    assert int((w.count.sum(dim=-1) > 0).sum()) == 4   # 4 seeded cells
    centers = [(2, 2), (2, 6), (6, 2), (6, 6)]
    for (cy, cx) in centers:
        assert int(w.strain_id[cy, cx, 0]) == bb0       # same BB0 everywhere


def test_init_factions_custom_layout_minted_for_all_four():
    t = StrainTable()
    lay = list(BB0_TEMPLATE["layout"]); lay[0] = "P_hotspot"   # legal slot change
    custom = tuple(lay)
    w = init_factions(8, 8, 16, DEV, t, fill_per_cell=10, n_fac=4, layout=custom)
    expect = t.get_or_mint(custom)
    centers = [(2, 2), (2, 6), (6, 2), (6, 6)]
    seen = set()
    for (cy, cx) in centers:
        assert int(w.strain_id[cy, cx, 0]) == expect      # all four = same custom layout
        seen.add(int(w.faction[cy, cx, 0]))
    assert seen == {0, 1, 2, 3}


def test_init_factions_rejects_tampered_layout():
    t = StrainTable()
    bad = list(BB0_TEMPLATE["layout"]); bad[1] = "N0"      # tampered locked position
    with pytest.raises(ValueError, match="locked"):
        init_factions(8, 8, 16, DEV, t, fill_per_cell=10, n_fac=4, layout=tuple(bad))


def test_engine_passes_layout_through():
    lay = list(BB0_TEMPLATE["layout"]); lay[2] = "F4Nr1"
    e = Engine(H=8, W=8, K=16, seed=0, device=DEV, fill_per_cell=10, layout=tuple(lay))
    expect = e.table.get_or_mint(tuple(lay))
    # the seeded strain at a quadrant center is the custom layout
    assert int(e.world.strain_id[2, 2, 0]) == expect


def test_engine_default_layout_unchanged():
    e = Engine(H=8, W=8, K=16, seed=0, device=DEV, fill_per_cell=10)
    bb0 = e.table.get_or_mint(BB0_TEMPLATE["layout"])
    assert int(e.world.strain_id[2, 2, 0]) == bb0


def _four_distinct():
    """Four layouts differing only in slot 0 (a legal mutable slot)."""
    out = []
    for letter in ("N0", "F4Nr1", "P_base", "P_hotspot"):
        lay = _canonical(); lay[0] = letter
        out.append(tuple(lay))
    return out


def test_init_factions_four_distinct_layouts():
    t = StrainTable()
    layouts = _four_distinct()
    w = init_factions(8, 8, 16, DEV, t, fill_per_cell=10, n_fac=4, layouts=layouts)
    centers = [(2, 2), (2, 6), (6, 2), (6, 6)]
    seen_strains = set()
    for fac, (cy, cx) in enumerate(centers):
        expect = t.get_or_mint(layouts[fac])
        assert int(w.strain_id[cy, cx, 0]) == expect      # fac-th center = fac-th layout
        assert int(w.faction[cy, cx, 0]) == fac
        seen_strains.add(int(w.strain_id[cy, cx, 0]))
    assert len(seen_strains) == 4                          # four genuinely different strains


def test_init_factions_layout_and_layouts_mutually_exclusive():
    t = StrainTable()
    with pytest.raises(ValueError, match="not both"):
        init_factions(8, 8, 16, DEV, t, fill_per_cell=10,
                      layout=BB0_TEMPLATE["layout"], layouts=_four_distinct())


def test_init_factions_layouts_must_be_exactly_four():
    t = StrainTable()
    three = _four_distinct()[:3]
    with pytest.raises(ValueError, match="exactly 4"):
        init_factions(8, 8, 16, DEV, t, fill_per_cell=10, layouts=three)


def test_init_factions_rejects_tampered_nth_layout():
    t = StrainTable()
    layouts = _four_distinct()
    bad = list(layouts[2]); bad[1] = "N0"        # tamper faction-2's locked position
    layouts = list(layouts)
    layouts[2] = tuple(bad)
    layouts = tuple(layouts)
    with pytest.raises(ValueError, match="locked"):
        init_factions(8, 8, 16, DEV, t, fill_per_cell=10, layouts=layouts)


def test_engine_passes_layouts_through():
    layouts = _four_distinct()
    e = Engine(H=8, W=8, K=16, seed=0, device=DEV, fill_per_cell=10, layouts=layouts)
    centers = [(2, 2), (2, 6), (6, 2), (6, 6)]
    for fac, (cy, cx) in enumerate(centers):
        assert int(e.world.strain_id[cy, cx, 0]) == e.table.get_or_mint(layouts[fac])
