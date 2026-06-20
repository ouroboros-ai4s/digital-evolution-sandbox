#!/usr/bin/env python
"""First-batch runner for the Digital Evolution Sandbox engine.

Locked first-batch config (2026-06-20, user-approved): 128x128 grid, K=64,
fill_per_cell=20, T=200 ticks, seeds [0,1,2,3], outcome constants at protocol
init (mu=0.01 / z_max=8.0 / delta=0.05 / p_max=0.08), v1 stubs off
(alpha0=beta_fold=kappa=0 -- baked into the registry, not knobs here).

Each seed -> one timestamped parquet under data/runs/. Per spec section 7:
filename = timestamp, long format (tick, cell_x, cell_y, strain, count),
only non-empty rows, every tick a self-contained full snapshot.

Usage:
    PYTHONPATH=src D:/anaconda3/envs/basic/python.exe scripts/run_batch.py
    # optional: --probe N   run only N ticks of seed 0 as a timing/VRAM probe
"""
from __future__ import annotations
import argparse, os, time, datetime
import torch
from des.engine import Engine
from des.recorder import Recorder

# --- locked config ---
H = W = 128
K = 64
FILL = 20
T = 200
SEEDS = [0, 1, 2, 3]
Z_MAX = 8.0           # mu/delta/p_max live in the registry (outcome constants, not run args)

OUT_DIR = os.path.join("data", "runs")


def _now_tag() -> str:
    # explicit timestamp (scripts may run where bare datetime.now is fine; engine core forbids it)
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def run_one(seed: int, ticks: int, device: torch.device, out_dir: str,
            record: bool = True) -> dict:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    e = Engine(H=H, W=W, K=K, seed=seed, device=device, z_max=Z_MAX, fill_per_cell=FILL)
    rec = None
    path = None
    if record:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{_now_tag()}-seed{seed}.parquet")
        rec = Recorder(path, e.table)
    t0 = time.perf_counter()
    ran = e.run(ticks, recorder=rec)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    if rec is not None:
        rec.close()
    peak_gb = (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else None
    return {
        "seed": seed, "ran": ran, "secs": dt, "tick_ms": 1000 * dt / max(1, ran),
        "strains": len(e.table), "total": e.total_count(),
        "distinct_live": e.distinct_strains(), "peak_gb": peak_gb, "path": path,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=0,
                    help="run only N ticks of seed 0 (no parquet) as a timing/VRAM probe")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device} | torch={torch.__version__} | "
          f"{torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")

    if args.probe:
        r = run_one(0, args.probe, device, OUT_DIR, record=False)
        print(f"[probe {args.probe} ticks] {r['tick_ms']:.1f} ms/tick | "
              f"peak {r['peak_gb']:.2f} GB | strains->{r['strains']} | total {r['total']}")
        est = r["tick_ms"] * T * len(SEEDS) / 1000
        print(f"  -> est full batch (T={T} x {len(SEEDS)} seeds): ~{est:.0f} s "
              f"({est/60:.1f} min), excl. dump")
        return

    print(f"=== first batch: {H}x{W} K={K} fill={FILL} T={T} seeds={SEEDS} ===")
    rows = []
    for s in SEEDS:
        r = run_one(s, T, device, OUT_DIR, record=True)
        pk = f"{r['peak_gb']:.2f}GB" if r["peak_gb"] is not None else "n/a"
        print(f"seed {s}: ran {r['ran']} ticks in {r['secs']:.1f}s "
              f"({r['tick_ms']:.1f} ms/tick) | strains {r['strains']} | "
              f"live-distinct {r['distinct_live']} | total {r['total']} | peak {pk}")
        print(f"         -> {r['path']}")
        rows.append(r)
    tot = sum(r["secs"] for r in rows)
    print(f"=== batch done: {len(rows)} runs, {tot:.1f}s total ===")


if __name__ == "__main__":
    main()
