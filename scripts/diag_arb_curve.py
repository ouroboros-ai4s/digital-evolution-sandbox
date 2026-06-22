#!/usr/bin/env python
"""Per-tick arbitration cost curve over a full T=450 run (diag, read-only).

The 50ms full-world figure vs 1129ms phase-probe average means the cost lives in
the FILLING phase, not at full occupancy -- opposite to the original assumption.
This logs arb ms per tick across 0..T plus two suspects:
  occ%      = cell occupancy (front expansion vs full)
  max_avail = max over contested cells of (K - occupied)  [the urn loop trip count]
If arb spikes track high max_avail at mid-fill, the urn loop's worst-cell trip
count is the hot spot. Also prints cumulative arb seconds -> answers feasibility
of T=450 x 4 seeds directly. No kernel edits.

Usage: PYTHONPATH=src D:/anaconda3/envs/basic/python.exe scripts/diag_arb_curve.py [T]
"""
from __future__ import annotations
import sys, time
import torch
from des.engine import Engine
from des.kernels.antagonism import phase1_antagonism
from des.kernels.reproduction import phase2_reproduce
from des.kernels.arbitration import phase3_arbitrate_vec

H = W = 128; K = 64; FILL = 20; Z_MAX = 8.0
T = int(sys.argv[1]) if len(sys.argv) > 1 else 450

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={dev} | torch={torch.__version__} | T={T}")
e = Engine(H=H, W=W, K=K, seed=0, device=dev, z_max=Z_MAX, fill_per_cell=FILL)


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


cum_arb = 0.0
print(f"{'tick':>5} {'arb_ms':>9} {'occ%':>6} {'max_avail':>9} {'n_contest':>9} {'cum_arb_s':>9}")
for t in range(T):
    ssid, scnt, sfac = e.world.snapshot()
    post = phase1_antagonism(ssid, scnt, sfac, e._phe, e.birth, e.T, e.z_max, e.gen)
    e.world.count = post
    buf, live = phase2_reproduce(e.world, ssid, scnt, sfac, e._phe, e.table,
                                 e.birth, e.T, e.gen)
    e.world.count = live; e._refresh_phe()

    # suspects: occupancy + the urn loop's trip count (max avail over contested cells)
    occ = (e.world.count.sum(-1) > 0).float().mean().item()
    a_ty, a_tx, a_sid, a_cnt, a_fac = buf.tensors()
    max_avail = -1; n_contest = -1
    if a_cnt.numel() > 0:
        cell = a_ty.long() * W + a_tx.long()
        cell_total = torch.zeros(H * W, dtype=torch.int64, device=dev)
        cell_total.scatter_add_(0, cell, a_cnt.long())
        avail = (K - e.world.count.sum(-1)).clamp(min=0).flatten()
        contested = cell_total > avail
        n_contest = int(contested.sum().item())
        if n_contest > 0:
            max_avail = int(avail[contested].max().item())

    sync(); t0 = time.perf_counter()
    nsid, ncnt, nfac, nbirth = phase3_arbitrate_vec(
        e.world.strain_id, e.world.count, e.world.faction, buf.tensors(), e.K,
        e.birth, e.T, e.gen, MAXSID=len(e.table) + 1, NFAC=4)
    sync(); arb = time.perf_counter() - t0
    cum_arb += arb
    e.world.strain_id, e.world.count, e.world.faction, e.birth = nsid, ncnt, nfac, nbirth
    e.T += 1

    if t % 30 == 0 or t == T - 1:
        print(f"{t:>5} {1000*arb:>9.1f} {occ*100:>6.1f} {max_avail:>9} "
              f"{n_contest:>9} {cum_arb:>9.1f}")

print(f"\ntotal arb over {T} ticks (seed 0): {cum_arb:.1f}s "
      f"-> est {T} x 4 seeds arb-only: ~{cum_arb*4/60:.1f} min")
