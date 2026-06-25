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


def test_f4nr1_dir_bits_determined_by_crc32_of_sequence():
    """F4Nr1 的方向位由 crc32 决定, 但跨进程 / 跨调用恒定.
    同一 sequence 调两次 phenotype, dir_bits 必相等."""
    from des.registry import phenotype
    a = phenotype(("F4Nr1",)).dir_bits
    b = phenotype(("F4Nr1",)).dir_bits
    assert a == b
    # popcount 仍是 1
    assert bin(a).count("1") == 1


def test_phenotype_recognizes_hash_locked_directions_field(monkeypatch):
    """模拟一个 hash-locked F 字母, _F[...].directions = 'hash:fclump';
    phenotype() 必须把它翻译为 _hash_dirs(seq, 'fclump') 的 2 方向, OR 进 dir_bits."""
    import des.registry as reg
    monkeypatch.setitem(reg.ALPHABET, "FCLUMP_TEST", "F")
    monkeypatch.setitem(reg.GRAN, "FCLUMP_TEST", "motif")
    monkeypatch.setitem(reg.MOTIF_LEN, "FCLUMP_TEST", 2)
    monkeypatch.setitem(reg._F, "FCLUMP_TEST", (0.45, "hash:fclump", 0.10, 6))
    seq = ("FCLUMP_TEST", "FCLUMP_TEST") + ("N0",) * 14
    p = reg.phenotype(seq)
    # fclump => 2 dirs (一根轴); 全在 ALL_DIRECTIONS 中
    assert bin(p.dir_bits).count("1") == 2
    assert p.in_place is False
    assert p.rand_dir is False


def test_phenotype_recognizes_rand_dir_directions_field(monkeypatch):
    """_F[...].directions = 'rand:1of4' → phenotype 写 rand_dir=True, dir_bits=0."""
    import des.registry as reg
    monkeypatch.setitem(reg.ALPHABET, "FDRIFT_TEST", "F")
    monkeypatch.setitem(reg.GRAN, "FDRIFT_TEST", "residue")
    monkeypatch.setitem(reg._F, "FDRIFT_TEST", (0.15, "rand:1of4", 0.30, 2))
    seq = ("FDRIFT_TEST",) + ("N0",) * 15
    p = reg.phenotype(seq)
    assert p.rand_dir is True
    assert p.in_place is False
    # rand_dir 不预写 dir_bits, kernel 现抽
    assert p.dir_bits == 0


def test_phenotype_recognizes_in_place_directions_field(monkeypatch):
    """_F[...].directions = (IN_PLACE_DIR,) → phenotype 写 in_place=True, dir_bits=0."""
    import des.registry as reg
    monkeypatch.setitem(reg.ALPHABET, "FSTACK_TEST", "F")
    monkeypatch.setitem(reg.GRAN, "FSTACK_TEST", "residue")
    monkeypatch.setitem(reg._F, "FSTACK_TEST", (0.60, (reg.IN_PLACE_DIR,), 0.00, 3))
    seq = ("FSTACK_TEST",) + ("N0",) * 15
    p = reg.phenotype(seq)
    assert p.in_place is True
    assert p.rand_dir is False
    # in_place 不预写 dir_bits, kernel 走当格分支
    assert p.dir_bits == 0


def test_phenotype_mixed_hash_and_static_F_letters_or_dir_bits(monkeypatch):
    """同 strain 含 F4Nr4 (字面 4 邻) + 一个 hash-locked F 字母 → dir_bits 应是
    两者 OR (字面四邻全开 + hash 的额外方向重复, OR 不变化 4-bit 全开)."""
    import des.registry as reg
    monkeypatch.setitem(reg.ALPHABET, "FFRONT_TEST", "F")
    monkeypatch.setitem(reg.GRAN, "FFRONT_TEST", "motif")
    monkeypatch.setitem(reg.MOTIF_LEN, "FFRONT_TEST", 2)
    monkeypatch.setitem(reg._F, "FFRONT_TEST", (0.50, "hash:ffront", 0.25, 4))
    seq = ("F4Nr4", "FFRONT_TEST", "FFRONT_TEST") + ("N0",) * 13
    p = reg.phenotype(seq)
    # F4Nr4 全 4 邻 OR-into-dir_bits 后, hash-locked 单方向只会重复一个 bit
    # → dir_bits 仍是 (1 << 4) - 1 = 15
    assert p.dir_bits == (1 << len(reg.ALL_DIRECTIONS)) - 1


def test_phenotype_arrays_has_in_place_column():
    """phe['in_place'] 是 int8 张量, idx 0 (EMPTY) 是 0."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    assert "in_place" in phe
    assert phe["in_place"].dtype == torch.int8
    assert int(phe["in_place"][0].item()) == 0


def test_phenotype_arrays_has_rand_dir_column():
    """phe['rand_dir'] 是 int8 张量, idx 0 (EMPTY) 是 0."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    assert "rand_dir" in phe
    assert phe["rand_dir"].dtype == torch.int8
    assert int(phe["rand_dir"][0].item()) == 0


def test_phenotype_arrays_columns_match_python_phenotype():
    """对每个 strain, phe[<col>][sid] 必须 == int(Phenotype.<field>)."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    for sid in range(1, len(eng.table) + 1):
        seq = eng.table.sequence_of(sid)
        p_obj = eng.table.phenotype_of(sid)
        assert int(phe["in_place"][sid].item()) == int(p_obj.in_place)
        assert int(phe["rand_dir"][sid].item()) == int(p_obj.rand_dir)


def test_phenotype_arrays_default_bb0_all_zero_for_in_place_and_rand_dir():
    """默认 BB0 alphabet 里没有 FSTACK / FDRIFT, 所以 in_place / rand_dir
    全是 0 (kernel 永远不会因这两列误走新分支)."""
    import torch
    from des.engine import Engine
    from des.registry import BB0_TEMPLATE
    eng = Engine(H=8, W=8, K=4, seed=0, device=torch.device("cpu"),
                 z_max=8.0, fill_per_cell=2,
                 layouts=(BB0_TEMPLATE["layout"],) * 4)
    phe = eng.table.phenotype_arrays(torch.device("cpu"))
    assert int(phe["in_place"].max().item()) == 0
    assert int(phe["rand_dir"].max().item()) == 0


def test_s4_new_F_letters_present_in_alphabet_with_family_F():
    """5 个新 F 字母 family 全 'F'."""
    from des.registry import ALPHABET
    for letter in ("FSTACK", "FCLUMP", "FFRONT", "F4Nr3", "FDRIFT"):
        assert ALPHABET.get(letter) == "F", f"{letter}: {ALPHABET.get(letter)!r}"


def test_s4_new_F_letters_have_correct_gran():
    """FSTACK / F4Nr3 / FDRIFT = residue; FCLUMP / FFRONT = motif (spec §3.4)."""
    from des.registry import GRAN
    assert GRAN["FSTACK"] == "residue"
    assert GRAN["F4Nr3"] == "residue"
    assert GRAN["FDRIFT"] == "residue"
    assert GRAN["FCLUMP"] == "motif"
    assert GRAN["FFRONT"] == "motif"


def test_s4_motif_F_letters_have_motif_len_2():
    """FCLUMP / FFRONT 是长度 2 的 motif (spec §3.4)."""
    from des.registry import MOTIF_LEN
    assert MOTIF_LEN["FCLUMP"] == 2
    assert MOTIF_LEN["FFRONT"] == 2


def test_s4_new_F_rows_have_exact_values():
    """每行 (f, directions, p_leave, period) 与 spec §3.4 表 verbatim 一致."""
    from des.registry import _F, IN_PLACE_DIR
    assert _F["FSTACK"] == (0.60, (IN_PLACE_DIR,), 0.00, 3)
    assert _F["FCLUMP"] == (0.45, "hash:fclump",   0.10, 6)
    assert _F["FFRONT"] == (0.50, "hash:ffront",   0.25, 4)
    assert _F["F4Nr3"]  == (0.40, "hash:f4nr3",    0.12, 5)
    assert _F["FDRIFT"] == (0.15, "rand:1of4",     0.30, 2)


def test_phenotype_fstack_strain_has_in_place_true():
    from des.registry import phenotype
    p = phenotype(("FSTACK", "F4Nr4", "P_base", "BroadSweep") + ("N0",) * 12)
    assert p.in_place is True
    assert p.rand_dir is False


def test_phenotype_fdrift_strain_has_rand_dir_true():
    from des.registry import phenotype
    p = phenotype(("FDRIFT", "F4Nr4", "P_base", "BroadSweep") + ("N0",) * 12)
    assert p.rand_dir is True
    assert p.in_place is False


def test_phenotype_fclump_strain_has_two_dir_bits():
    """FCLUMP hash-locked 一根轴 (2 个 bit). 仍 motif=2 只能成对出现."""
    from des.registry import phenotype
    p = phenotype(("FCLUMP", "FCLUMP") + ("N0",) * 14)
    assert bin(p.dir_bits).count("1") == 2
    assert p.in_place is False
    assert p.rand_dir is False


def test_phenotype_f4nr3_strain_has_three_dir_bits():
    """F4Nr3 hash-locked 三邻 (3 个 bit)."""
    from des.registry import phenotype
    p = phenotype(("F4Nr3",) + ("N0",) * 15)
    assert bin(p.dir_bits).count("1") == 3
