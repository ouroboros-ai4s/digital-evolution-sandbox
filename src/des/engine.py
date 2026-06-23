# src/des/engine.py
from __future__ import annotations
import torch
from des.phenotype_cache import StrainTable
from des.world import init_factions
from des.kernels.antagonism import phase1_antagonism
from des.kernels.reproduction import phase2_reproduce
from des.kernels.arbitration import phase3_arbitrate_vec

NFAC = 4


class Engine:
    def __init__(self, H, W, K, seed, device, z_max=8.0, fill_per_cell=None,
                 check_every=10, layout=None, layouts=None):
        self.H, self.W, self.K, self.device, self.z_max = H, W, K, device, z_max
        self.check_every = check_every
        self.table = StrainTable()
        fill = K // 2 if fill_per_cell is None else fill_per_cell
        self.world = init_factions(H, W, K, device, self.table,
                                   fill_per_cell=fill, n_fac=NFAC,
                                   layout=layout, layouts=layouts)
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
        snap_sid, snap_count, snap_faction = self.world.snapshot()
        # PHASE1: antagonism (faction-gated) -> post-antagonism counts
        post_anta = phase1_antagonism(snap_sid, snap_count, snap_faction, self._phe,
                                      self.birth, self.T, self.z_max, self.gen)
        self.world.count = post_anta
        # PHASE2: reproduction — amounts from snapshot, space from post-antagonism world
        buf, live = phase2_reproduce(self.world, snap_sid, snap_count, snap_faction,
                                     self._phe, self.table, self.birth, self.T, self.gen)
        self.world.count = live
        self._refresh_phe()
        # PHASE3: arbitration (faction-keyed)
        arrivals = buf.tensors()
        nsid, ncnt, nfac, nbirth = phase3_arbitrate_vec(
            self.world.strain_id, self.world.count, self.world.faction, arrivals, self.K,
            self.birth, self.T, self.gen, MAXSID=len(self.table) + 1, NFAC=NFAC)
        self.world.strain_id, self.world.count = nsid, ncnt
        self.world.faction, self.birth = nfac, nbirth
        self.T += 1

    def total_count(self) -> int:
        return int(self.world.count.sum())

    def distinct_strains(self) -> int:
        present = self.world.strain_id[self.world.count > 0]
        return int(torch.unique(present).numel())

    def _fixated(self) -> bool:
        # single-faction field: only one faction among all living individuals
        facs = self.world.faction[self.world.count > 0]
        return facs.numel() > 0 and int(torch.unique(facs).numel()) == 1

    def run(self, ticks, recorder=None, stop_on=("fixation", "extinction")) -> int:
        ran = 0
        for _ in range(ticks):
            self.step()
            ran += 1
            if recorder is not None:
                recorder.dump(self.T, self.world)
            # P3: stop-check is GPU->CPU sync; run it only every check_every ticks
            if ran % self.check_every == 0:
                if "extinction" in stop_on and self.total_count() == 0:
                    break
                if "fixation" in stop_on and self._fixated() and ran > 1:
                    break
        return ran
