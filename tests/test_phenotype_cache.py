# tests/test_phenotype_cache.py
import torch
from des.phenotype_cache import StrainTable
from des.types import EMPTY_ID

def test_mint_starts_at_one_and_is_stable():
    t = StrainTable()
    a = t.get_or_mint(("F4Nr1", "N0"))
    b = t.get_or_mint(("F4Nr1", "N0"))   # convergent → same id
    c = t.get_or_mint(("N0", "F4Nr1"))   # different order → different strain
    assert a == 1
    assert a == b
    assert c != a
    assert len(t) == 2

def test_empty_id_reserved():
    t = StrainTable()
    sid = t.get_or_mint(("N0",))
    assert sid != EMPTY_ID

def test_phenotype_cached_and_matches_registry():
    from des.registry import phenotype
    t = StrainTable()
    sid = t.get_or_mint(("F4Nr1", "BroadSweep"))
    assert t.phenotype_of(sid) == phenotype(("F4Nr1", "BroadSweep"))
    assert t.sequence_of(sid) == ("F4Nr1", "BroadSweep")

def test_phenotype_arrays_indexed_by_id():
    t = StrainTable()
    sid = t.get_or_mint(("F4Nr1",))
    arr = t.phenotype_arrays(torch.device("cpu"))
    assert arr["f"].shape[0] == len(t) + 1            # +1 for EMPTY row
    assert arr["f"][EMPTY_ID].item() == 0.0           # empty row zeroed
    assert abs(arr["f"][sid].item() - 0.30) < 1e-6
    assert arr["prey_mask"].dtype == torch.int64
    # M2: exact key-set guard for downstream kernels
    assert set(arr.keys()) == {"f", "p_leave", "z_raw", "p_x", "prey_mask", "feature_mask", "period"}


DEV = torch.device("cpu")


def test_phenotype_arrays_cached_until_mint():
    t = StrainTable()
    t.get_or_mint(("F4Nr1",))
    arr1 = t.phenotype_arrays(DEV)
    arr2 = t.phenotype_arrays(DEV)
    # I1: second call with no new mint must return the SAME object (cache hit)
    assert arr1 is arr2, "phenotype_arrays should be cached when no new strain minted"

    # After minting a new strain the cache must be invalidated
    sid2 = t.get_or_mint(("BroadSweep",))
    arr3 = t.phenotype_arrays(DEV)
    assert arr3 is not arr1, "phenotype_arrays must rebuild after a new strain is minted"
    # new strain's row must be present
    assert arr3["f"].shape[0] == len(t) + 1
    assert arr3["f"][sid2].item() >= 0.0  # value exists (non-sentinel)


def test_phenotype_of_empty_raises():
    import pytest
    t = StrainTable()
    # I2: sentinel id 0 must raise KeyError, not AssertionError
    with pytest.raises(KeyError):
        t.phenotype_of(0)
