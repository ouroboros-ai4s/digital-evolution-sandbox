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
