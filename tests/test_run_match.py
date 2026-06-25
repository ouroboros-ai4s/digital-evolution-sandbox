"""End-to-end CLI tests for scripts/run_match.py.

These tests drive the script as a subprocess so the actual entry-point
behavior (argparse, JSON load, exit code, stdout payload) is exercised."""
from __future__ import annotations
import json, os, subprocess, sys
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PY = sys.executable
SCRIPT = os.path.join(REPO, "scripts", "run_match.py")


def _run(args, env_extra=None):
    env = os.environ.copy()
    # Include both src (for des) and repo root (for webapp)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(REPO, "src"), REPO])
    if env_extra:
        env.update(env_extra)
    return subprocess.run([PY, SCRIPT, *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)


def _write(tmp_path, name, payload):
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return str(p)


def _empty_players():
    return [{"slots": {}} for _ in range(4)]


def test_missing_config_arg_exits_nonzero():
    r = _run([])
    assert r.returncode != 0


def test_config_file_not_found_exits_1(tmp_path):
    r = _run(["--config", str(tmp_path / "missing.json")])
    assert r.returncode == 1
    assert "missing.json" in (r.stderr + r.stdout)


def test_config_malformed_json_exits_1(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ this is not json ")
    r = _run(["--config", str(p)])
    assert r.returncode == 1


def test_disallowed_top_level_key_z_max_rejected(tmp_path):
    cfg = {"players": _empty_players(), "grid": 16, "K": 8, "fill": 4,
           "T": 2, "seed": 0, "z_max": 20}
    r = _run(["--config", _write(tmp_path, "c.json", cfg), "--cpu",
              "--out", str(tmp_path / "runs")])
    assert r.returncode == 1
    assert "z_max" in (r.stderr + r.stdout)


@pytest.mark.parametrize("k", ["mu", "delta", "p_max", "alpha", "kappa", "beta"])
def test_disallowed_outcome_constants_rejected(tmp_path, k):
    cfg = {"players": _empty_players(), "grid": 16, "K": 8, "fill": 4,
           "T": 2, "seed": 0, k: 0.5}
    r = _run(["--config", _write(tmp_path, "c.json", cfg), "--cpu",
              "--out", str(tmp_path / "runs")])
    assert r.returncode == 1
    assert k in (r.stderr + r.stdout)


def test_valid_4_player_run_writes_parquet_and_prints_result(tmp_path):
    cfg = {"players": _empty_players(), "grid": 16, "K": 8, "fill": 4,
           "T": 3, "seed": 0}
    out_dir = tmp_path / "runs"
    r = _run(["--config", _write(tmp_path, "ok.json", cfg), "--cpu",
              "--out", str(out_dir)])
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout.strip().splitlines()[-1])
    assert payload["ticks"] == 3
    assert os.path.isfile(payload["path"])
    assert payload["path"].startswith(str(out_dir))
    final = payload["final"]
    for k in ("total", "occupied_cells", "distinct_strains",
              "n2", "d_max", "faction_share"):
        assert k in final
    # faction_share is a dict[str|int, float]; sums to ~1.0 on a non-empty world
    if final["total"] > 0:
        assert abs(sum(final["faction_share"].values()) - 1.0) < 1e-6


def test_three_players_exits_nonzero_no_parquet(tmp_path):
    cfg = {"players": [{"slots": {}} for _ in range(3)],
           "grid": 16, "K": 8, "fill": 4, "T": 2, "seed": 0}
    out_dir = tmp_path / "runs"
    r = _run(["--config", _write(tmp_path, "bad.json", cfg), "--cpu",
              "--out", str(out_dir)])
    assert r.returncode == 1
    assert "4 players" in (r.stderr + r.stdout)
    assert not out_dir.exists() or len(list(out_dir.iterdir())) == 0


def test_off_palette_letter_exits_nonzero_no_parquet(tmp_path):
    cfg = {"players": [{"slots": {0: "NOPE"}}, {"slots": {}}, {"slots": {}}, {"slots": {}}],
           "grid": 16, "K": 8, "fill": 4, "T": 2, "seed": 0}
    out_dir = tmp_path / "runs"
    r = _run(["--config", _write(tmp_path, "bad.json", cfg), "--cpu",
              "--out", str(out_dir)])
    assert r.returncode == 1
    assert "palette" in (r.stderr + r.stdout)
    assert not out_dir.exists() or len(list(out_dir.iterdir())) == 0


def test_non_slot_index_exits_nonzero_no_parquet(tmp_path):
    cfg = {"players": [{"slots": {4: "F4Nr1"}}, {"slots": {}}, {"slots": {}}, {"slots": {}}],
           "grid": 16, "K": 8, "fill": 4, "T": 2, "seed": 0}
    out_dir = tmp_path / "runs"
    r = _run(["--config", _write(tmp_path, "bad.json", cfg), "--cpu",
              "--out", str(out_dir)])
    assert r.returncode == 1
    assert "slot" in (r.stderr + r.stdout)


def test_web_config_interchangeable_with_cli_config(tmp_path):
    """Single-schema invariant: a config the WS path accepts must also be
    accepted by run_match (and vice-versa). Both use build_engine_from_config."""
    import torch
    from des.run import build_engine_from_config
    cfg = {"players": [{"slots": {0: "F4Nr1"}}, {"slots": {0: "P_hotspot"}},
                       {"slots": {0: "P_base"}}, {"slots": {0: "BroadSweep"}}],
           "grid": 16, "K": 8, "fill": 4, "T": 2, "seed": 0}
    # WS path equivalent
    eng, _ = build_engine_from_config(cfg, torch.device("cpu"))
    assert eng.H == 16
    # CLI path
    r = _run(["--config", _write(tmp_path, "ws.json", cfg), "--cpu",
              "--out", str(tmp_path / "runs")])
    assert r.returncode == 0, r.stderr
