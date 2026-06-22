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
    assert set(arr.keys()) == {"f", "p_leave", "z_raw", "p_x", "prey_mask",
                               "feature_mask", "period", "dir_bits",
                               "repro_period", "anta_period"}


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


# ---------------------------------------------------------------------------
# Task-5 tests: dir_bits + per-phase periods in phenotype_arrays
# ---------------------------------------------------------------------------

def test_phenotype_arrays_has_dir_and_periods():
    t = StrainTable()
    sid4 = t.get_or_mint(("F4Nr4",))
    arr = t.phenotype_arrays(torch.device("cpu"))
    assert set(arr.keys()) == {"f", "p_leave", "z_raw", "p_x", "prey_mask",
                               "feature_mask", "period", "dir_bits",
                               "repro_period", "anta_period"}
    assert arr["dir_bits"].dtype == torch.int64
    assert int(arr["dir_bits"][sid4]) == 0b1111      # all 4 directions
    assert int(arr["dir_bits"][0]) == 0              # EMPTY row
    assert int(arr["repro_period"][sid4]) == 5
    assert int(arr["repro_period"][0]) == 1          # EMPTY row period 1


# ---------------------------------------------------------------------------
# Task-1 (perf) test: bulk-transfer rebuild must equal per-strain phenotype
# values across every field. Locks the contract the refactor preserves.
# ---------------------------------------------------------------------------

_FIELD_ATTRS = ("f", "p_leave", "z_raw", "p_x", "prey_mask", "feature_mask",
                "period", "dir_bits", "repro_period", "anta_period")


def test_phenotype_arrays_bulk_matches_per_strain():
    t = StrainTable()
    sids = [
        t.get_or_mint(("F4Nr1",)),
        t.get_or_mint(("F4Nr4", "BroadSweep")),
        t.get_or_mint(("P_hotspot", "N0")),
        t.get_or_mint(("BroadSweep", "F4Nr1", "P_base")),
    ]
    arr = t.phenotype_arrays(torch.device("cpu"))

    # shape + EMPTY row
    for key in _FIELD_ATTRS:
        assert arr[key].shape[0] == len(t) + 1, f"{key} wrong length"
    for key in ("f", "p_leave", "z_raw", "p_x", "prey_mask", "feature_mask", "dir_bits"):
        assert arr[key][EMPTY_ID].item() == 0, f"{key} EMPTY row must be 0"
    for key in ("period", "repro_period", "anta_period"):
        assert arr[key][EMPTY_ID].item() == 1, f"{key} EMPTY row must be 1"

    # dtypes
    for key in ("f", "p_leave", "z_raw", "p_x"):
        assert arr[key].dtype == torch.float32, f"{key} must be float32"
    for key in ("prey_mask", "feature_mask", "period", "dir_bits",
                "repro_period", "anta_period"):
        assert arr[key].dtype == torch.int64, f"{key} must be int64"

    # every minted strain's row equals its Phenotype field, field by field
    for sid in sids:
        phe = t.phenotype_of(sid)
        for key in _FIELD_ATTRS:
            got = arr[key][sid].item()
            want = getattr(phe, key)
            if key in ("f", "p_leave", "z_raw", "p_x"):
                assert abs(got - want) < 1e-6, f"{key}[{sid}]={got} != {want}"
            else:
                assert got == want, f"{key}[{sid}]={got} != {want}"
