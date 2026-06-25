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
        # Build per-field CPU lists in one pass, then do ONE bulk host->device
        # transfer per field. The old loop assigned each f[sid]=scalar straight
        # into a CUDA tensor -> 10n CPU->GPU syncs/rebuild (~1080ms/tick at n~1e4).
        # ponytail: list-build + 10 torch.tensor() calls = 10 transfers, not 10n.
        # Index 0 = EMPTY sentinel: zeros, except periods=1 (M1: modulo-by-zero
        # guard in the (T-birth)%period firing clock; id 0 never fires anyway).
        f = [0.0] * n
        p_leave = [0.0] * n
        z_raw = [0.0] * n
        p_x = [0.0] * n
        prey = [0] * n
        feat = [0] * n
        dir_bits = [0] * n
        period = [1] * n
        repro_period = [1] * n
        anta_period = [1] * n
        vis_sum = [0.0] * n      # S1: Σ_{i: fam=N} VIS[seq[i]]
        n_count = [0] * n        # S1: #{i: fam=N}
        vis_mode_l = [0] * n     # S1: 0=none, 1=vis-weighted, 2=inverse-vis-weighted
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
            dir_bits[sid] = phe.dir_bits
            repro_period[sid] = phe.repro_period
            anta_period[sid] = phe.anta_period
            vis_sum[sid] = phe.vis_sum
            n_count[sid] = phe.n_count
            vis_mode_l[sid] = phe.vis_mode
        result = {
            "f": torch.tensor(f, dtype=torch.float32, device=device),
            "p_leave": torch.tensor(p_leave, dtype=torch.float32, device=device),
            "z_raw": torch.tensor(z_raw, dtype=torch.float32, device=device),
            "p_x": torch.tensor(p_x, dtype=torch.float32, device=device),
            "prey_mask": torch.tensor(prey, dtype=torch.int64, device=device),
            "feature_mask": torch.tensor(feat, dtype=torch.int64, device=device),
            "period": torch.tensor(period, dtype=torch.int64, device=device),
            "dir_bits": torch.tensor(dir_bits, dtype=torch.int64, device=device),
            "repro_period": torch.tensor(repro_period, dtype=torch.int64, device=device),
            "anta_period": torch.tensor(anta_period, dtype=torch.int64, device=device),
            "vis_sum": torch.tensor(vis_sum, dtype=torch.float32, device=device),   # S1
            "n_count": torch.tensor(n_count, dtype=torch.int16, device=device),     # S1
            "vis_mode": torch.tensor(vis_mode_l, dtype=torch.int8, device=device),  # S1
        }
        # store cache and clear dirty flag
        self._cached_arrays = result
        self._cached_device = device
        self._arrays_dirty = False
        return result
