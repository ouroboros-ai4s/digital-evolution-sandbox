#!/usr/bin/env python
"""Headless CLI front door for the DES engine. AI players configure
4 factions' starting strains in a JSON file and run a match producing
one parquet under data/runs/.

Usage:
    PYTHONPATH=src python scripts/run_match.py --config match.json [--cpu] [--out data/runs]

The JSON schema is identical to the WebSocket `config` payload:
    {"players": [{"slots": {"0": "F4Nr4", "2": "P_hotspot"}}, {…}, {…}, {…}],
     "grid": 128, "K": 64, "fill": 20, "T": 450, "seed": 0}

Exactly 4 players. Only top-level keys allowed: players, grid, K, fill, T,
seed. Any other key (z_max / mu / delta / p_max / alpha / kappa / beta) is
a red-line violation and exits 1 — outcome constants stay locked in the
registry (spec §2 red-line 2)."""
from __future__ import annotations
import argparse, datetime, json, os, sys

ALLOWED_KEYS = frozenset({"players", "grid", "K", "fill", "T", "seed"})
DEFAULT_OUT = os.path.join("data", "runs")


def validate_config_keys(cfg: dict) -> None:
    """Top-level allow-list. Rejects outcome-constant keys (z_max / mu /
    delta / p_max / alpha / kappa / beta) and any other unknown key.
    This is the CLI's own guard — the web path fixes z_max from _DEFAULTS
    and never exposes it; this closes the same door for the JSON front end."""
    extra = set(cfg.keys()) - ALLOWED_KEYS
    if extra:
        raise ValueError(
            f"disallowed top-level config keys: {sorted(extra)}; "
            f"allowed = {sorted(ALLOWED_KEYS)} "
            f"(outcome constants stay in the registry)")


def _now_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="DES headless match runner")
    ap.add_argument("--config", required=True, help="path to match config JSON")
    ap.add_argument("--cpu", action="store_true", help="force CPU device")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output dir for parquet")
    return ap.parse_args(argv)


def _load_config(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        cfg = _load_config(args.config)
        validate_config_keys(cfg)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"malformed JSON in {args.config}: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    # engine assembly + run + result emission added in Task 5
    print(json.dumps({"event": "config_validated", "path": args.config}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
