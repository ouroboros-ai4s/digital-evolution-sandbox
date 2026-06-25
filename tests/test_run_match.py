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
    env["PYTHONPATH"] = os.path.join(REPO, "src")
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
