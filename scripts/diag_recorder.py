#!/usr/bin/env python
"""Diagnostic: where does dump() time go on the locked first-batch config?

Measures, as the world fills:
  - rows emitted per tick (non-empty slots)
  - dump() wall time (INCLUDES blocking on the bounded queue = true writer backpressure)
  - compute-only tick time (no recorder)
  - writer-thread string-build + write_batch time, isolated

Not a unit test; a one-off profiler. Run:
  PYTHONPATH=src D:/anaconda3/envs/basic/python.exe scripts/diag_recorder.py --ticks 340
"""
from __future__ import annotations
import argparse, time, tempfile, os
import torch
from des.engine import Engine
from des.recorder import Recorder, _SCHEMA
import pyarrow as pa

H = W = 128
K = 64
FILL = 20
Z_MAX = 8.0


def rows_at_tick(world) -> int:
    return int((world.count.to("cpu") > 0).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=340)
    ap.add_argument("--sample-every", type=int, default=20)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}")

    # --- Pass A: compute-only (no recorder), measure tick time + rows/occupancy growth ---
    e = Engine(H=H, W=W, K=K, seed=0, device=dev, z_max=Z_MAX, fill_per_cell=FILL)
    print("\n[pass A: compute-only, rows/tick growth]")
    print("tick  rows   occupied_cells")
    samples = []
    for t in range(1, args.ticks + 1):
        e.step()
        if t % args.sample_every == 0 or t == 1:
            cnt = e.world.count.to("cpu")
            rows = int((cnt > 0).sum())
            occ = int((cnt.sum(dim=-1) > 0).sum())
            samples.append((t, rows))
            print(f"{t:4d}  {rows:6d}  {occ:6d}")
    final_rows = samples[-1][1]

    # --- Pass B: isolate writer-thread cost — build strains list + record_batch for final state ---
    cnt = e.world.count.to("cpu"); sid = e.world.strain_id.to("cpu"); fac = e.world.faction.to("cpu")
    nz = torch.nonzero(cnt > 0, as_tuple=False)
    ys = nz[:, 0].tolist(); xs = nz[:, 1].tolist(); ks = nz[:, 2]
    sids = sid[nz[:, 0], nz[:, 1], ks].tolist()
    facs = fac[nz[:, 0], nz[:, 1], ks].tolist()
    cnts = cnt[nz[:, 0], nz[:, 1], ks].tolist()
    print(f"\n[pass B: writer-thread cost for one full-state dump, {len(sids)} rows]")
    t0 = time.perf_counter()
    strains = [".".join(e.table.sequence_of(int(s))) for s in sids]
    t1 = time.perf_counter()
    batch = pa.record_batch([
        pa.array([args.ticks] * len(sids), pa.int32()),
        pa.array(xs, pa.int32()), pa.array(ys, pa.int32()),
        pa.array(strains, pa.string()), pa.array(facs, pa.int8()),
        pa.array(cnts, pa.int32()),
    ], schema=_SCHEMA)
    t2 = time.perf_counter()
    print(f"  string-build (list of {len(sids)} seq-joins): {1000*(t1-t0):.1f} ms")
    print(f"  record_batch build:                          {1000*(t2-t1):.1f} ms")
    print(f"  est per-tick writer cost at full world:      {1000*(t2-t0):.1f} ms")
    print(f"  -> over {args.ticks} ticks (if all near-full): ~{(t2-t0)*args.ticks:.0f} s just string+batch")

    # --- Pass C: end-to-end with recorder, measure dump() blocking on bounded queue ---
    e2 = Engine(H=H, W=W, K=K, seed=0, device=dev, z_max=Z_MAX, fill_per_cell=FILL)
    tmp = os.path.join(tempfile.gettempdir(), "diag_seed0.parquet")
    rec = Recorder(tmp, e2.table)
    print(f"\n[pass C: end-to-end with recorder -> {tmp}]")
    dump_acc = 0.0; step_acc = 0.0
    for t in range(1, args.ticks + 1):
        ts = time.perf_counter()
        e2.step()
        if dev.type == "cuda": torch.cuda.synchronize(dev)
        td = time.perf_counter()
        rec.dump(e2.T, e2.world)   # includes block-on-full-queue
        te = time.perf_counter()
        step_acc += (td - ts); dump_acc += (te - td)
    print(f"  cumulative step():  {step_acc:.1f} s ({1000*step_acc/args.ticks:.1f} ms/tick)")
    print(f"  cumulative dump():  {dump_acc:.1f} s ({1000*dump_acc/args.ticks:.1f} ms/tick incl. queue block)")
    tc = time.perf_counter()
    rec.close()   # drains the queue — the real tail
    print(f"  rec.close() drain:  {time.perf_counter()-tc:.1f} s")
    sz = os.path.getsize(tmp) / 1e6
    print(f"  parquet size: {sz:.0f} MB  ({final_rows} rows/tick near end x {args.ticks} ticks)")
    os.remove(tmp)


if __name__ == "__main__":
    main()
