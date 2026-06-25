# tests/test_hash_dirs.py
"""S4 hash-locked direction selection. crc32(\\x1f.join(seq)) -> 各 kind 的方向集.

Determinism 是这一档的生命线: PYTHONHASHSEED 加盐的 hash() 跨进程不一致, 数据生成
沙盒立崩. 用 stdlib zlib.crc32 是为了 cross-process / cross-machine 一致 +
lightweight (we only need a uniform int to select dirs, not crypto)."""
from __future__ import annotations
import subprocess
import sys
import textwrap
import pytest
import zlib

from des.registry import _hash_dirs, IN_PLACE_DIR, ALL_DIRECTIONS


def test_in_place_dir_is_zero_zero_and_not_in_all_directions():
    """IN_PLACE_DIR = (0, 0); 严格不入 ALL_DIRECTIONS, 与四邻完全正交."""
    assert IN_PLACE_DIR == (0, 0)
    assert IN_PLACE_DIR not in ALL_DIRECTIONS


def test_ffront_returns_single_direction_in_all_directions():
    """ffront kind 返回 长度 1 的方向元组, 方向在 ALL_DIRECTIONS 中."""
    dirs = _hash_dirs(("F4Nr1", "N0", "N0"), "ffront")
    assert len(dirs) == 1
    assert dirs[0] in ALL_DIRECTIONS


def test_f4nr1_returns_single_direction_in_all_directions():
    """f4nr1 kind 与 ffront 同表 (spec §3.3): 单方向, ALL_DIRECTIONS[h%4]."""
    dirs = _hash_dirs(("F4Nr1", "N0", "N0"), "f4nr1")
    assert len(dirs) == 1
    assert dirs[0] in ALL_DIRECTIONS


def test_fclump_returns_axis_pair():
    """fclump kind 返回 长度 2 的方向元组, 是一根轴 (x 或 y)."""
    dirs = _hash_dirs(("FCLUMP", "N0", "N0"), "fclump")
    assert len(dirs) == 2
    axis_x = {(-1, 0), (1, 0)}
    axis_y = {(0, -1), (0, 1)}
    assert set(dirs) == axis_x or set(dirs) == axis_y


def test_f4nr3_returns_three_of_four_neighbors():
    """f4nr3 kind 返回 长度 3 的方向元组, 是 ALL_DIRECTIONS 的 3 元子集."""
    dirs = _hash_dirs(("F4Nr3", "N0", "N0"), "f4nr3")
    assert len(dirs) == 3
    for d in dirs:
        assert d in ALL_DIRECTIONS
    assert len(set(dirs)) == 3        # 无重复


def test_unknown_kind_raises_value_error():
    with pytest.raises(ValueError):
        _hash_dirs(("N0",), "not_a_kind")


def test_same_sequence_same_kind_yields_same_dirs():
    """同 seq + 同 kind → 同方向 (mint 时一次性算, 整个 lineage 锁死)."""
    seq = ("FFRONT", "N0", "F4Nr4", "P_hotspot")
    a = _hash_dirs(seq, "ffront")
    b = _hash_dirs(seq, "ffront")
    assert a == b


def test_unit_separator_prevents_concat_ambiguity():
    """`\\x1f` 分隔符使多字符字母 token 不会歧义拼接:
    ("N0", "F4Nr4") 必须 != ("N0F4Nr4",) 也 != ("N", "0F4Nr4"),
    crc32 算出来 hash 不一样 → 方向不一样 (一般情况)."""
    h_split = zlib.crc32("\x1f".join(("N0", "F4Nr4")).encode())
    h_concat = zlib.crc32("\x1f".join(("N0F4Nr4",)).encode())
    h_misjoin = zlib.crc32("\x1f".join(("N", "0F4Nr4")).encode())
    assert h_split != h_concat
    assert h_split != h_misjoin


def test_hash_is_deterministic_across_subprocesses():
    """开一个子进程, 用同 seq 调 _hash_dirs, 必须得到与父进程相同的方向集.
    PYTHONHASHSEED 加盐的 hash() 在这条断言上必失败 (spec §2)."""
    code = textwrap.dedent('''
        import sys
        sys.path.insert(0, r"{src}")
        from des.registry import _hash_dirs
        seq = ("FFRONT", "N0", "F4Nr4", "P_hotspot")
        print(_hash_dirs(seq, "ffront"))
    ''').format(src=str(__import__("os").path.abspath(__import__("os").path.join(
        __import__("os").path.dirname(__file__), "..", "src"))))
    parent = _hash_dirs(("FFRONT", "N0", "F4Nr4", "P_hotspot"), "ffront")
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert str(parent) == out, f"parent={parent!r}, child={out!r}"


def test_relabel_invariance_hash_reads_letter_sequence_only(monkeypatch):
    """重排 _F / _Z / _P 的量级 (f / z / p_add / period) 不影响 _hash_dirs
    输出 —— hash 读的是字面字母, 与 magnitude 解耦 (spec §6)."""
    import des.registry as reg
    seq = ("FFRONT", "F4Nr4", "P_hotspot")
    pre = _hash_dirs(seq, "ffront")
    monkeypatch.setitem(reg._F, "F4Nr4", (0.01, ((1, 0),), 0.99, 99))
    monkeypatch.setitem(reg._P, "P_hotspot", (0.0, 99))
    post = _hash_dirs(seq, "ffront")
    assert pre == post
