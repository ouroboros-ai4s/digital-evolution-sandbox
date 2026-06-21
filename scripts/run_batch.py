#!/usr/bin/env python
"""First-batch runner for the Digital Evolution Sandbox engine.

Locked first-batch config (2026-06-20, user-approved): 128x128 grid, K=64,
fill_per_cell=20, T=450 ticks, seeds [0,1,2,3], outcome constants at protocol
init (mu=0.01 / z_max=8.0 / delta=0.05 / p_max=0.08), v1 stubs off
(alpha0=beta_fold=kappa=0 -- baked into the registry, not knobs here).

Engine seeds four-quadrant factions via init_factions (Task 9); four factions
placed at quadrant centres (32,32)/(32,96)/(96,32)/(96,96), NOT full-field
init_bb0.  F4Nr4 repro_period=5 -> fronts meet ~tick160, world fills ~tick320;
T=450 covers front transient + meeting-band antagonism + post-fill red-queen.

Each seed -> one timestamped parquet under data/runs/. Per spec section 7:
filename = timestamp, long format (tick, cell_x, cell_y, strain, faction,
count), only non-empty rows, every tick a self-contained full snapshot.

Usage:
    PYTHONPATH=src D:/anaconda3/envs/basic/python.exe scripts/run_batch.py
    # optional: --probe N        run only N ticks of seed 0 as a timing/VRAM probe
    # optional: --phase-probe N  per-phase ms/tick (proves P1/P2 were the cost)
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
T = 450               # 2026-06-21: F4Nr4 repro_period=5 -> meet ~tick160, fill ~tick320,
                      # T covers front transient + meeting-band antagonism + post-fill red-queen
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
    ran = e.run(ticks, recorder=rec, stop_on=())   # spec §3.1: run full T, no early stop
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


def phase_probe(seed: int, ticks: int, device: torch.device) -> None:
    """Per-phase ms/tick on the locked grid, to verify P1/P2 were the hot spots."""
    import time
    from des.kernels.antagonism import phase1_antagonism
    from des.kernels.reproduction import phase2_reproduce
    from des.kernels.arbitration import phase3_arbitrate_vec
    e = Engine(H=H, W=W, K=K, seed=seed, device=device, z_max=Z_MAX, fill_per_cell=FILL)
    acc = {"anta": 0.0, "repro": 0.0, "arb": 0.0}
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    for _ in range(ticks):
        ssid, scnt, sfac = e.world.snapshot()
        sync(); t0 = time.perf_counter()
        post = phase1_antagonism(ssid, scnt, sfac, e._phe, e.birth, e.T, e.z_max, e.gen)
        sync(); t1 = time.perf_counter(); e.world.count = post
        buf, live = phase2_reproduce(e.world, ssid, scnt, sfac, e._phe, e.table,
                                     e.birth, e.T, e.gen)
        sync(); t2 = time.perf_counter(); e.world.count = live; e._refresh_phe()
        nsid, ncnt, nfac, nbirth = phase3_arbitrate_vec(
            e.world.strain_id, e.world.count, e.world.faction, buf.tensors(), e.K,
            e.birth, e.T, e.gen, MAXSID=len(e.table) + 1, NFAC=4)
        sync(); t3 = time.perf_counter()
        e.world.strain_id, e.world.count, e.world.faction, e.birth = nsid, ncnt, nfac, nbirth
        e.T += 1
        acc["anta"] += (t1 - t0); acc["repro"] += (t2 - t1); acc["arb"] += (t3 - t2)
    n = max(1, ticks)
    print(f"[phase-probe {ticks} ticks] "
          f"anta {1000*acc['anta']/n:.1f} | repro {1000*acc['repro']/n:.1f} | "
          f"arb {1000*acc['arb']/n:.1f} ms/tick (occupancy grows over the run)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=0,
                    help="run only N ticks of seed 0 (no parquet) as a timing/VRAM probe")
    ap.add_argument("--phase-probe", type=int, default=0,
                    help="per-phase ms/tick timing on the locked grid")
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

    if args.phase_probe:
        phase_probe(0, args.phase_probe, device)
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
