# src/des/phenotype_cache.py
from __future__ import annotations
import torch
from des.types import EMPTY_ID, Phenotype
from des.registry import phenotype as parse_phenotype


class StrainTable:
    def __init__(self) -> None:
        self._seq_to_id: dict[tuple[str, ...], int] = {}
        self._id_to_seq: list[tuple[str, ...]] = [()]      # index 0 = EMPTY
        self._id_to_phe: list[Phenotype | None] = [None]   # index 0 = EMPTY
        self._next = 1
        # I1: dirty-flag cache for phenotype_arrays
        self._arrays_dirty: bool = True
        self._cached_arrays: dict[str, torch.Tensor] | None = None
        self._cached_device: torch.device | None = None

    def get_or_mint(self, sequence: tuple[str, ...]) -> int:
        sid = self._seq_to_id.get(sequence)
        if sid is not None:
            return sid          # idempotent re-hit: do NOT mark dirty
        sid = self._next
        self._next += 1
        self._seq_to_id[sequence] = sid
        self._id_to_seq.append(sequence)
        self._id_to_phe.append(parse_phenotype(sequence))
        self._arrays_dirty = True   # new strain minted: invalidate cache
        return sid

    def sequence_of(self, sid: int) -> tuple[str, ...]:
        return self._id_to_seq[sid]

    def phenotype_of(self, sid: int) -> Phenotype:
        phe = self._id_to_phe[sid]
        if phe is None:
            # I2: explicit raise instead of assert (assert stripped under python -O)
            raise KeyError("strain id 0 is the EMPTY sentinel; it has no phenotype")
        return phe

    def __len__(self) -> int:
        return self._next - 1

    def phenotype_arrays(self, device: torch.device) -> dict[str, torch.Tensor]:
        # I1: return cached tensors when not dirty and same device
        if not self._arrays_dirty and self._cached_device == device and self._cached_arrays is not None:
            return self._cached_arrays
        n = self._next
        f = torch.zeros(n, dtype=torch.float32, device=device)
        p_leave = torch.zeros(n, dtype=torch.float32, device=device)
        z_raw = torch.zeros(n, dtype=torch.float32, device=device)
        p_x = torch.zeros(n, dtype=torch.float32, device=device)
        prey = torch.zeros(n, dtype=torch.int64, device=device)
        feat = torch.zeros(n, dtype=torch.int64, device=device)
        # M1: period[0]=1 (not 0) for the EMPTY sentinel row: avoids modulo-by-zero
        # in the (T-birth)%period firing clock; id 0 never fires anyway (count 0).
        period = torch.ones(n, dtype=torch.int64, device=device)
        for sid in range(1, n):
            phe = self._id_to_phe[sid]
            if phe is None:
                raise KeyError(f"strain id {sid} has no phenotype (internal error)")
            f[sid] = phe.f
            p_leave[sid] = phe.p_leave
            z_raw[sid] = phe.z_raw
            p_x[sid] = phe.p_x
            prey[sid] = phe.prey_mask
            feat[sid] = phe.feature_mask
            period[sid] = phe.period
        result = {"f": f, "p_leave": p_leave, "z_raw": z_raw, "p_x": p_x,
                  "prey_mask": prey, "feature_mask": feat, "period": period}
        # store cache and clear dirty flag
        self._cached_arrays = result
        self._cached_device = device
        self._arrays_dirty = False
        return result
