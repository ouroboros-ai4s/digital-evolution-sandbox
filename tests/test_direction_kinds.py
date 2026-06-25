# tests/test_direction_kinds.py
"""S4 dynamic-direction primitives + kernel branching.

This file is the S4 owner test file (sibling: tests/test_motif.py / test_vis.py /
test_spectrum_shape.py). Covers the 5 new F primitives' registry rows, phenotype
fields, phenotype-array columns, and end-to-end kernel branching."""
from __future__ import annotations
import pytest


def test_phenotype_has_in_place_default_false():
    """Phenotype.in_place 默认 False —— 既有所有 strain 都不走当格沉积."""
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert hasattr(p, "in_place")
    assert p.in_place is False


def test_phenotype_has_rand_dir_default_false():
    """Phenotype.rand_dir 默认 False —— 既有所有 strain 都不走每 tick 抽."""
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert hasattr(p, "rand_dir")
    assert p.rand_dir is False


def test_phenotype_is_still_frozen():
    """加字段后 dataclass 仍 frozen, 不可改: 守不可变契约."""
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    with pytest.raises(Exception):
        p.in_place = True       # FrozenInstanceError under @dataclass(frozen=True)
