# tests/test_world_layout.py
import pytest
from des.registry import BB0_TEMPLATE, validate_bb0_layout, _SLOTS, _LOCKED


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
