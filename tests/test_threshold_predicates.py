"""S3 rich-prey predicates: thr_crest / thr_hotspot / thr_mirror feature bits
+ 4 new prey-clause tags ("f_hi" / "p_hi" / "generalist" / "lowvis").

This file is the S3 owner test file (sibling: tests/test_motif.py /
test_vis.py / test_spectrum_shape.py / test_direction_kinds.py /
test_phase_windows.py / test_hash_dirs.py). vis_lowvis bit (S6 reserved,
S1 owner-filled) is tested in tests/test_vis.py — S3 only audits it
in Task 5 to make sure S1's behavior is still live."""
from __future__ import annotations
import pytest
from des import registry


def test_z_prey_card_exists_and_covers_every_Z_row():
    """_Z_PREY_CARD 必须覆盖 _Z 的全部 key, 与 _Z.keys() 同集合."""
    from des.registry import _Z, _Z_PREY_CARD
    assert set(_Z_PREY_CARD.keys()) == set(_Z.keys()), (
        f"_Z_PREY_CARD keys {set(_Z_PREY_CARD)} != _Z keys {set(_Z)}")


def test_z_prey_card_broadsweep_is_two():
    """BroadSweep prey_clauses = (("F",), ("Z",)) → cardinality 2."""
    from des.registry import _Z_PREY_CARD
    assert _Z_PREY_CARD["BroadSweep"] == 2


def test_z_prey_card_values_match_len_of_prey_clauses():
    """每行 _Z_PREY_CARD[name] 必须 == len(_Z[name][1])."""
    from des.registry import _Z, _Z_PREY_CARD
    for name, row in _Z.items():
        clauses = row[1]
        assert _Z_PREY_CARD[name] == len(clauses), (
            f"{name}: card={_Z_PREY_CARD[name]} but len(prey_clauses)={len(clauses)}")
