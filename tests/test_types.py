# tests/test_types.py
from des.types import Phenotype, PhaseType, FAMILY_RANK, EMPTY_ID

def test_family_rank_ordering():
    assert FAMILY_RANK["N"] < FAMILY_RANK["F"] < FAMILY_RANK["P"] < FAMILY_RANK["Z"]

def test_phenotype_is_frozen():
    p = Phenotype(f=0.3, directions=((-1, 0),), p_leave=0.05, z_raw=0.0,
                  prey_mask=0, feature_mask=0, p_x=0.01, spectrum=(),
                  period=4, phase_type=PhaseType.REPRODUCTION, fold=())
    assert p.f == 0.3 and p.period == 4
    try:
        p.f = 0.9  # type: ignore
        assert False, "should be frozen"
    except Exception:
        pass

def test_empty_id_is_zero():
    assert EMPTY_ID == 0
