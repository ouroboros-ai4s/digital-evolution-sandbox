# tests/test_phase_windows.py
"""S5 phase-window f primitives + kernel where-on-window branch."""
from __future__ import annotations
import pytest


def test_phenotype_has_f_hi_field():
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert hasattr(p, "f_hi")
    assert isinstance(p.f_hi, float)


def test_phenotype_has_f_lo_field():
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert hasattr(p, "f_lo")
    assert isinstance(p.f_lo, float)


def test_phenotype_has_burst_w_default_one():
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert hasattr(p, "burst_w")
    assert isinstance(p.burst_w, int)
    assert p.burst_w >= 1


def test_phenotype_has_burst_k_default_one():
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert hasattr(p, "burst_k")
    assert isinstance(p.burst_k, int)
    assert p.burst_k >= 1


def test_phenotype_is_still_frozen_after_s5_fields():
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    with pytest.raises(Exception):
        p.f_hi = 0.99


def test_existing_F_rows_static_default_means_f_hi_eq_f_lo_eq_f():
    from des.registry import phenotype
    p_f4nr1 = phenotype(("F4Nr1",) + ("N0",) * 15)
    assert p_f4nr1.f_hi == p_f4nr1.f
    assert p_f4nr1.f_lo == p_f4nr1.f
    assert p_f4nr1.burst_w == 1
    assert p_f4nr1.burst_k == 1
    p_f4nr4 = phenotype(("F4Nr4",) + ("N0",) * 15)
    assert p_f4nr4.f_hi == p_f4nr4.f
    assert p_f4nr4.f_lo == p_f4nr4.f
    assert p_f4nr4.burst_w == 1
    assert p_f4nr4.burst_k == 1


def test_default_bb0_layout_phenotype_static_default():
    from des.registry import phenotype, BB0_TEMPLATE
    p = phenotype(BB0_TEMPLATE["layout"])
    assert p.f_hi == p.f
    assert p.f_lo == p.f
    assert p.burst_w == 1
    assert p.burst_k == 1


def test_multi_F_static_strain_stacks_f_via_one_minus_prod():
    from des.registry import phenotype
    seq = ("F4Nr4", "F4Nr1") + ("N0",) * 14
    p = phenotype(seq)
    expected_f = 1 - (1 - 0.50) * (1 - 0.30)
    assert abs(p.f - expected_f) < 1e-9
    assert p.f_hi == p.f
    assert abs(p.f_lo - expected_f) < 1e-9   # static: f_lo == f
    assert p.burst_w == 1
    assert p.burst_k == 1


def test_F_row_is_7_tuple_for_existing_letters():
    from des.registry import _F
    for letter in ("F4Nr1", "F4Nr4"):
        row = _F[letter]
        assert len(row) == 7, f"{letter}: expected 7-tuple, got len={len(row)}"
        f_val, dirs, p_leave, period, f_lo, burst_w, burst_k = row
        assert f_lo == f_val
        assert burst_w == 1
        assert burst_k == 1


def test_phenotype_f_field_is_alias_of_f_hi():
    from des.registry import phenotype
    for seq in (("F4Nr1",) + ("N0",)*15, ("F4Nr4",) + ("N0",)*15,
                ("F4Nr4", "F4Nr1") + ("N0",)*14):
        p = phenotype(seq)
        assert p.f == p.f_hi, f"seq={seq!r}: f {p.f} != f_hi {p.f_hi}"


def test_s5_FBURST_present_in_alphabet_with_family_F():
    from des.registry import ALPHABET
    assert ALPHABET.get("FBURST") == "F"


def test_s5_F_NOVA_present_in_alphabet_with_family_F():
    from des.registry import ALPHABET
    assert ALPHABET.get("F_NOVA") == "F"


def test_s5_FBURST_and_F_NOVA_have_gran_residue():
    from des.registry import GRAN
    assert GRAN["FBURST"] == "residue"
    assert GRAN["F_NOVA"] == "residue"


def test_s5_FBURST_row_verbatim():
    from des.registry import _F
    expected = (0.55, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.20, 2, 0.05, 12, 2)
    assert _F["FBURST"] == expected


def test_s5_F_NOVA_row_verbatim():
    from des.registry import _F
    expected = (0.85, ((-1, 0), (1, 0), (0, -1), (0, 1)), 0.50, 2, 0.05, 20, 1)
    assert _F["F_NOVA"] == expected


def test_phenotype_FBURST_single_letter_has_correct_window_params():
    from des.registry import phenotype, ALL_DIRECTIONS
    p = phenotype(("FBURST",) + ("N0",) * 15)
    assert abs(p.f_hi - 0.55) < 1e-9
    assert abs(p.f_lo - 0.05) < 1e-9
    assert p.burst_w == 12
    assert p.burst_k == 2
    assert p.dir_bits == (1 << len(ALL_DIRECTIONS)) - 1
    assert p.f == p.f_hi


def test_phenotype_F_NOVA_single_letter_has_correct_window_params():
    from des.registry import phenotype, ALL_DIRECTIONS
    p = phenotype(("F_NOVA",) + ("N0",) * 15)
    assert abs(p.f_hi - 0.85) < 1e-9
    assert abs(p.f_lo - 0.05) < 1e-9
    assert p.burst_w == 20
    assert p.burst_k == 1
    assert p.dir_bits == (1 << len(ALL_DIRECTIONS)) - 1
    assert p.f == p.f_hi


def test_phenotype_FBURST_plus_static_F_dominant_is_FBURST():
    from des.registry import phenotype
    seq = ("FBURST", "F4Nr1") + ("N0",) * 14
    p = phenotype(seq)
    expected_f_hi = 1 - (1 - 0.55) * (1 - 0.30)
    expected_f_lo = 1 - (1 - 0.05) * (1 - 0.30)
    assert abs(p.f_hi - expected_f_hi) < 1e-9
    assert abs(p.f_lo - expected_f_lo) < 1e-9
    assert p.burst_w == 12
    assert p.burst_k == 2


def test_phenotype_static_F_plus_FBURST_dominant_still_FBURST():
    from des.registry import phenotype
    seq = ("F4Nr1", "FBURST") + ("N0",) * 14
    p = phenotype(seq)
    assert p.burst_w == 12
    assert p.burst_k == 2


# ---------------------------------------------------------------------------
# S5 Task 4: phenotype_arrays tensor columns for f_hi / f_lo / burst_w / burst_k
# ---------------------------------------------------------------------------

def test_phenotype_arrays_has_f_hi_column():
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    assert "f_hi" in phe
    assert phe["f_hi"].dtype == torch.float32
    assert float(phe["f_hi"][0].item()) == 0.0


def test_phenotype_arrays_has_f_lo_column():
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    assert "f_lo" in phe
    assert phe["f_lo"].dtype == torch.float32
    assert float(phe["f_lo"][0].item()) == 0.0


def test_phenotype_arrays_has_burst_w_column_default_one():
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    assert "burst_w" in phe
    assert phe["burst_w"].dtype == torch.int64
    assert int(phe["burst_w"][0].item()) == 1


def test_phenotype_arrays_has_burst_k_column_default_one():
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    assert "burst_k" in phe
    assert phe["burst_k"].dtype == torch.int64
    assert int(phe["burst_k"][0].item()) == 1


def test_phenotype_arrays_columns_match_python_phenotype():
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    for sid in range(1, len(eng.table) + 1):
        p = eng.table.phenotype_of(sid)
        assert abs(float(phe["f_hi"][sid].item()) - p.f_hi) < 1e-6
        assert abs(float(phe["f_lo"][sid].item()) - p.f_lo) < 1e-6
        assert int(phe["burst_w"][sid].item()) == p.burst_w
        assert int(phe["burst_k"][sid].item()) == p.burst_k


def test_phenotype_arrays_default_bb0_window_columns_degenerate():
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    eng.run(5, recorder=None, stop_on=())
    phe2 = eng.table.phenotype_arrays(torch.device("cpu"))
    n = len(eng.table)
    assert n >= 1
    for sid in range(1, n + 1):
        f_hi = float(phe2["f_hi"][sid].item())
        f_lo = float(phe2["f_lo"][sid].item())
        assert abs(f_hi - f_lo) < 1e-6, f"sid={sid}: f_hi={f_hi} != f_lo={f_lo}"
        assert int(phe2["burst_w"][sid].item()) == 1
        assert int(phe2["burst_k"][sid].item()) == 1


def test_default_bb0_same_seed_byte_identical_post_s5():
    """S5 changed kernel line, but default strains degenerate statically -> byte-identical."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng_a = Engine(H=8, W=8, K=8, seed=0, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2,
                   layouts=(BB0_TEMPLATE["layout"],) * 4)
    eng_b = Engine(H=8, W=8, K=8, seed=0, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=2,
                   layouts=(BB0_TEMPLATE["layout"],) * 4)
    eng_a.run(30, recorder=None, stop_on=())
    eng_b.run(30, recorder=None, stop_on=())
    assert torch.equal(eng_a.world.count, eng_b.world.count)
    assert torch.equal(eng_a.world.strain_id, eng_b.world.strain_id)


def test_fburst_seed_reproducible_across_runs():
    """FBURST strain same seed 2 runs -> bit-identical."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    base = list(BB0_TEMPLATE["layout"])
    base[0] = "FBURST"  # slot 0 is mutable
    fburst_layout = tuple(base)
    eng_a = Engine(H=4, W=4, K=16, seed=42, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=4,
                   layouts=(fburst_layout,) * 4)
    eng_b = Engine(H=4, W=4, K=16, seed=42, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=4,
                   layouts=(fburst_layout,) * 4)
    eng_a.run(15, recorder=None, stop_on=())
    eng_b.run(15, recorder=None, stop_on=())
    assert torch.equal(eng_a.world.count, eng_b.world.count)
    assert torch.equal(eng_a.world.strain_id, eng_b.world.strain_id)


def test_f_nova_seed_reproducible_across_runs():
    """F_NOVA strain same seed 2 runs -> bit-identical."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    base = list(BB0_TEMPLATE["layout"])
    base[0] = "F_NOVA"  # slot 0 is mutable
    fnova_layout = tuple(base)
    eng_a = Engine(H=4, W=4, K=16, seed=42, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=4,
                   layouts=(fnova_layout,) * 4)
    eng_b = Engine(H=4, W=4, K=16, seed=42, device=torch.device("cpu"),
                   z_max=8.0, fill_per_cell=4,
                   layouts=(fnova_layout,) * 4)
    eng_a.run(22, recorder=None, stop_on=())
    eng_b.run(22, recorder=None, stop_on=())
    assert torch.equal(eng_a.world.count, eng_b.world.count)
    assert torch.equal(eng_a.world.strain_id, eng_b.world.strain_id)


def test_phase_window_on_mask_formula_directly():
    """Directly verify on = ((T - birth_tick) % burst_w) < burst_k formula."""
    import torch
    burst_w = torch.tensor([12, 12, 12, 12, 12, 12, 20, 20, 20], dtype=torch.int64)
    burst_k = torch.tensor([2, 2, 2, 2, 2, 2, 1, 1, 1], dtype=torch.int64)
    birth_tick = torch.zeros(9, dtype=torch.int64)
    Ts = torch.tensor([0, 1, 2, 11, 12, 24, 0, 1, 20], dtype=torch.int64)
    on = ((Ts - birth_tick) % burst_w.clamp(min=1)) < burst_k
    expected = torch.tensor([True, True, False, False, True, True,
                             True, False, True])
    assert torch.equal(on, expected), f"on={on.tolist()}, expected={expected.tolist()}"


def test_fburst_offspring_window_on_vs_off():
    """FBURST run 24 ticks: system-level sanity - total count > 0."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    base = list(BB0_TEMPLATE["layout"])
    base[0] = "FBURST"  # slot 0 is mutable
    fburst_layout = tuple(base)
    layouts = (fburst_layout,) * 4
    eng = Engine(H=8, W=8, K=32, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=16, layouts=layouts)
    eng.run(24, recorder=None, stop_on=())
    total = int(eng.world.count.sum().item())
    assert total > 0, "FBURST strain should produce offspring across windows"
