# src/des/engine.py
from __future__ import annotations
import torch
from des.phenotype_cache import StrainTable
from des.world import init_bb0
from des.kernels.antagonism import phase1_antagonism
from des.kernels.reproduction import phase2_reproduce
from des.kernels.arbitration import phase3_arbitrate, phase3_arbitrate_vec


class Engine:
    def __init__(self, H, W, K, seed, device, z_max=8.0, fill_per_cell=None):
        self.H, self.W, self.K, self.device, self.z_max = H, W, K, device, z_max
        self.table = StrainTable()
        fill = K // 2 if fill_per_cell is None else fill_per_cell
        self.world = init_bb0(H, W, K, device, self.table, fill_per_cell=fill)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.birth = torch.zeros((H, W, K), dtype=torch.int32, device=device)
        self.T = 0
        self._phe = self.table.phenotype_arrays(device)
        self._phe_n = len(self.table)

    def _refresh_phe(self):
        if len(self.table) != self._phe_n:
            self._phe = self.table.phenotype_arrays(self.device)
            self._phe_n = len(self.table)

    def step(self) -> None:
        snap_sid, snap_count = self.world.snapshot()
        # PHASE1: antagonism — returns post-antagonism counts
        post_anta = phase1_antagonism(snap_sid, snap_count, self._phe, self.birth,
                                      self.T, self.z_max, self.gen)
        self.world.count = post_anta
        # PHASE2: reproduction — amounts from snapshot, space from post-antagonism world
        buf, live = phase2_reproduce(self.world, snap_sid, snap_count, self._phe,
                                     self.table, self.birth, self.T, self.gen)
        # phase2 already returns post-antagonism residents minus migration (spec PHASE 2);
        # no extra clamp needed here.
        self.world.count = live
        self._refresh_phe()
        # PHASE3: arbitration
        arrivals = buf.tensors()
        nsid, ncnt, nbirth = phase3_arbitrate_vec(
            self.world.strain_id, self.world.count, arrivals, self.K,
            self.birth, self.T, self.gen, MAXSID=len(self.table) + 1)
        self.world.strain_id, self.world.count, self.birth = nsid, ncnt, nbirth
        self.T += 1

    def total_count(self) -> int:
        return int(self.world.count.sum())

    def distinct_strains(self) -> int:
        present = self.world.strain_id[self.world.count > 0]
        return int(torch.unique(present).numel())

    def _fixated(self) -> bool:
        present = self.world.strain_id[self.world.count > 0]
        return present.numel() > 0 and int(torch.unique(present).numel()) == 1

    def run(self, ticks, recorder=None, stop_on=("fixation", "extinction")) -> int:
        ran = 0
        for _ in range(ticks):
            self.step()
            ran += 1
            if recorder is not None:
                recorder.dump(self.T, self.world)
            if "extinction" in stop_on and self.total_count() == 0:
                break
            if "fixation" in stop_on and self._fixated() and ran > 1:
                break
        return ran
