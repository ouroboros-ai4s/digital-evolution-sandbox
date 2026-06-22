#!/usr/bin/env python
"""Read-only attribution of PHASE3 cost on a FULL world (diag, not engine).

phase_probe averages empty->full so it blends the cheap early ticks with the
expensive full-world ones. This drives the world to ~full occupancy first, then
profiles a single phase3 call there with torch.profiler -- op-level breakdown
names the real hot spot without editing the frozen kernel:
  sort/argsort/argmax/log  -> coalesce + urn-draw sections
  eq/expand/index_put       -> Section-3 writeback ([N,K] resident-hit matrices)

Usage: PYTHONPATH=src D:/anaconda3/envs/basic/python.exe scripts/diag_arb_sections.py [fill_ticks]
"""
from __future__ import annotations
import sys, time
import torch
from des.engine import Engine
from des.kernels.antagonism import phase1_antagonism
from des.kernels.reproduction import phase2_reproduce
from des.kernels.arbitration import phase3_arbitrate_vec

H = W = 128; K = 64; FILL = 20; Z_MAX = 8.0
FILL_TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 340

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={dev} | torch={torch.__version__}")
e = Engine(H=H, W=W, K=K, seed=0, device=dev, z_max=Z_MAX, fill_per_cell=FILL)


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def step():
    ssid, scnt, sfac = e.world.snapshot()
    post = phase1_antagonism(ssid, scnt, sfac, e._phe, e.birth, e.T, e.z_max, e.gen)
    e.world.count = post
    buf, live = phase2_reproduce(e.world, ssid, scnt, sfac, e._phe, e.table,
                                 e.birth, e.T, e.gen)
    e.world.count = live; e._refresh_phe()
    nsid, ncnt, nfac, nbirth = phase3_arbitrate_vec(
        e.world.strain_id, e.world.count, e.world.faction, buf.tensors(), e.K,
        e.birth, e.T, e.gen, MAXSID=len(e.table) + 1, NFAC=4)
    e.world.strain_id, e.world.count, e.world.faction, e.birth = nsid, ncnt, nfac, nbirth
    e.T += 1


# --- drive to full ---
for _ in range(FILL_TICKS):
    step()
occ = (e.world.count.sum(-1) > 0).float().mean().item()
nonempty = int((e.world.count > 0).sum().item())
print(f"after {FILL_TICKS} ticks: cell-occupancy {occ*100:.1f}% | nonempty slots {nonempty}")

# --- time the full phase3 a few times on the full world (steady-state arb cost) ---
ssid, scnt, sfac = e.world.snapshot()
post = phase1_antagonism(ssid, scnt, sfac, e._phe, e.birth, e.T, e.z_max, e.gen)
e.world.count = post
buf, live = phase2_reproduce(e.world, ssid, scnt, sfac, e._phe, e.table,
                             e.birth, e.T, e.gen)
e.world.count = live; e._refresh_phe()
args = (e.world.strain_id, e.world.count, e.world.faction, buf.tensors(), e.K,
        e.birth, e.T, e.gen)
kw = dict(MAXSID=len(e.table) + 1, NFAC=4)
for _ in range(2):  # warmup
    phase3_arbitrate_vec(*args, **kw)
sync(); t0 = time.perf_counter()
REP = 10
for _ in range(REP):
    phase3_arbitrate_vec(*args, **kw)
sync()
print(f"phase3 full-world: {1000*(time.perf_counter()-t0)/REP:.1f} ms/call (avg of {REP})")

# --- op-level attribution ---
from torch.profiler import profile, ProfilerActivity
acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
with profile(activities=acts) as prof:
    for _ in range(REP):
        phase3_arbitrate_vec(*args, **kw)
    sync()
key = "self_cuda_time_total" if dev.type == "cuda" else "self_cpu_time_total"
print(prof.key_averages().table(sort_by=key, row_limit=15))
