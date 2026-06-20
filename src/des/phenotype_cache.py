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

    def get_or_mint(self, sequence: tuple[str, ...]) -> int:
        sid = self._seq_to_id.get(sequence)
        if sid is not None:
            return sid
        sid = self._next
        self._next += 1
        self._seq_to_id[sequence] = sid
        self._id_to_seq.append(sequence)
        self._id_to_phe.append(parse_phenotype(sequence))
        return sid

    def sequence_of(self, sid: int) -> tuple[str, ...]:
        return self._id_to_seq[sid]

    def phenotype_of(self, sid: int) -> Phenotype:
        phe = self._id_to_phe[sid]
        assert phe is not None, "EMPTY id has no phenotype"
        return phe

    def __len__(self) -> int:
        return self._next - 1

    def phenotype_arrays(self, device: torch.device) -> dict[str, torch.Tensor]:
        n = self._next
        f = torch.zeros(n, dtype=torch.float32, device=device)
        p_leave = torch.zeros(n, dtype=torch.float32, device=device)
        z_raw = torch.zeros(n, dtype=torch.float32, device=device)
        p_x = torch.zeros(n, dtype=torch.float32, device=device)
        prey = torch.zeros(n, dtype=torch.int64, device=device)
        feat = torch.zeros(n, dtype=torch.int64, device=device)
        period = torch.ones(n, dtype=torch.int64, device=device)
        for sid in range(1, n):
            phe = self._id_to_phe[sid]
            assert phe is not None
            f[sid] = phe.f
            p_leave[sid] = phe.p_leave
            z_raw[sid] = phe.z_raw
            p_x[sid] = phe.p_x
            prey[sid] = phe.prey_mask
            feat[sid] = phe.feature_mask
            period[sid] = phe.period
        return {"f": f, "p_leave": p_leave, "z_raw": z_raw, "p_x": p_x,
                "prey_mask": prey, "feature_mask": feat, "period": period}
