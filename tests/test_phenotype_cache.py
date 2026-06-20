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
