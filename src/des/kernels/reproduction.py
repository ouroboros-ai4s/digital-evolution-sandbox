# src/des/kernels/reproduction.py
from __future__ import annotations
import torch
from des.kernels.common import fires_this_tick, binom
from des.registry import BB0_TEMPLATE


class ArrivalBuffer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._ty: list[torch.Tensor] = []
        self._tx: list[torch.Tensor] = []
        self._sid: list[torch.Tensor] = []
        self._cnt: list[torch.Tensor] = []

    def add(self, ty, tx, sid, cnt) -> None:
        m = cnt > 0
        if m.any():
            self._ty.append(ty[m]); self._tx.append(tx[m])
            self._sid.append(sid[m]); self._cnt.append(cnt[m])

    def tensors(self):
        if not self._cnt:
            z = torch.zeros(0, dtype=torch.int64, device=self.device)
            return z, z, z.to(torch.int32), z.to(torch.int32)
        return (torch.cat(self._ty), torch.cat(self._tx),
                torch.cat(self._sid), torch.cat(self._cnt))


def _mutate_sequence(seq: tuple[str, ...], mutable_mask, spectrum, generator):
    """Replace one mutable slot's letter via Gumbel-argmax over `spectrum`.
    spectrum: tuple[(letter, q)]. Returns new sequence tuple. Reads only the sequence."""
    if not spectrum:
        return seq
    # choose first mutable slot whose current letter participates (v1 simple rule)
    idxs = [i for i, m in enumerate(mutable_mask) if m]
    if not idxs:
        return seq
    pick_idx = int(torch.randint(len(idxs), (1,), generator=generator).item())
    slot = idxs[pick_idx]
    letters = [t for t, _ in spectrum]
    logq = torch.log(torch.tensor([q for _, q in spectrum]) + 1e-12)
    gumbel = -torch.log(-torch.log(torch.rand(len(letters), generator=generator) + 1e-12) + 1e-12)
    pick = int(torch.argmax(logq + gumbel).item())
    new = list(seq); new[slot] = letters[pick]
    return tuple(new)


def phase2_reproduce(world, snap_sid, snap_count, phe, table, birth_tick, T, generator):
    H, W, K = snap_count.shape
    dev = world.device
    sid_long = snap_sid.long()
    f = phe["f"][sid_long]
    p_leave = phe["p_leave"][sid_long]
    p_x = phe["p_x"][sid_long]
    period = phe["period"][sid_long]
    alive = snap_count > 0
    fires = fires_this_tick(birth_tick, period, T) & alive & (f > 0)

    buf = ArrivalBuffer(dev)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")

    # group firing slots by strain to fetch directions (mutation rare → loop over present strains)
    present = torch.unique(snap_sid[fires])
    for sid_val in present.tolist():
        if sid_val == 0:
            continue
        phe_obj = table.phenotype_of(sid_val)
        dirs = phe_obj.directions
        if not dirs:
            continue
        seq = table.sequence_of(sid_val)
        mutable = BB0_TEMPLATE["mutable"]  # backbone-locked positions (F4Nr1@1, BroadSweep@5, P_base@7) never mutate
        # cells where THIS strain fires (any slot)
        slotmask = (snap_sid == sid_val) & fires
        cell_count = (snap_count * slotmask).sum(dim=-1)  # [H,W] count of this strain in firing slots
        cell_fires = cell_count > 0
        if not cell_fires.any():
            continue
        a = cell_count.to(torch.int32)
        f_val = float(phe_obj.f)
        px_val = float(phe_obj.p_x)
        for (dy, dx) in dirs:
            scattered = binom(a, torch.full_like(a, f_val, dtype=torch.float32), generator)
            scattered = torch.roll(scattered, shifts=(dy, dx), dims=(0, 1))
            ty = torch.roll(yy, shifts=(dy, dx), dims=(0, 1))
            tx = torch.roll(xx, shifts=(dy, dx), dims=(0, 1))
            # mutation split at landing
            mut = binom(scattered, torch.full_like(scattered, px_val, dtype=torch.float32), generator)
            non = scattered - mut
            buf.add(ty.flatten(), tx.flatten(),
                    torch.full((H * W,), sid_val, dtype=torch.int32, device=dev),
                    non.flatten())
            if mut.sum() > 0:
                child_seq = _mutate_sequence(seq, mutable, phe_obj.spectrum, generator)
                child_id = table.get_or_mint(child_seq)
                buf.add(ty.flatten(), tx.flatten(),
                        torch.full((H * W,), child_id, dtype=torch.int32, device=dev),
                        mut.flatten())

    # migration out (spec PHASE 2 line 136): world[g][s] -= min(a_snap*p_leave, a_snap),
    # subtracted from the POST-antagonism live world. leave AMOUNT is drawn from the
    # snapshot (a_snap), but applied to world.count (already = post-antagonism count here).
    # p_leave departers vanish from the source cell by design: design doc line 115/309 defines
    # p_leave as the reproduction migration COST (teng kong ming e rang wei), NOT a relocation
    # to neighbors -- they are not re-added as arrivals anywhere. This is intentional, not a
    # missing feature.
    leave = binom(snap_count, p_leave, generator)
    leave = torch.minimum(leave, snap_count)
    live = (world.count - leave).clamp(min=0)
    return buf, live
