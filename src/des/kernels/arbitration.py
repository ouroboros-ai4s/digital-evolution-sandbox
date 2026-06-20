# src/des/kernels/arbitration.py
from __future__ import annotations
import torch
from des.kernels.common import binom


def phase3_arbitrate(live_sid, live_count, arrivals, K, birth_tick, T, generator, MAXSID):
    H, W, _ = live_count.shape
    dev = live_count.device
    a_ty, a_tx, a_sid, a_cnt = arrivals

    # --- 1. coalesce arrivals by (cell, strain) ---
    if a_cnt.numel() > 0:
        cell = (a_ty.long() * W + a_tx.long())
        key = cell * MAXSID + a_sid.long()
        uniq, inv = torch.unique(key, return_inverse=True)
        merged = torch.zeros(uniq.shape[0], dtype=torch.int64, device=dev)
        merged.scatter_add_(0, inv, a_cnt.long())
        u_cell = uniq // MAXSID
        u_sid = (uniq % MAXSID).to(torch.int32)
        u_y = (u_cell // W).long()
        u_x = (u_cell % W).long()
    else:
        u_y = torch.zeros(0, dtype=torch.long, device=dev)
        u_x = torch.zeros(0, dtype=torch.long, device=dev)
        u_sid = torch.zeros(0, dtype=torch.int32, device=dev)
        merged = torch.zeros(0, dtype=torch.int64, device=dev)

    # --- 2. K-wall: per-cell individual cap ---
    # resident_occ = total individuals already seated (K is individual cap, not slot cap)
    resident_occ = live_count.sum(dim=-1)              # [H,W]
    available = (K - resident_occ).clamp(min=0)        # [H,W] remaining individual slots

    # Group coalesced arrivals by cell for sequential thinning.
    # Build a dict: (y,x) -> list of (idx_into_merged, sid, count)
    # Then draw sequentially per strain: binom(cnt_i, avail_remaining / total_remaining)
    # This is the hypergeometric-equivalent sequential draw — exact hard cap, strain-blind
    # (equal per-capita because ratio avail/total is identical across all strains in the cell
    # at the start, and updated symmetrically regardless of processing order).
    survived = merged.clone()
    if merged.numel() > 0:
        # group indices by cell
        cell_indices: dict[tuple[int, int], list[int]] = {}
        for idx in range(len(u_y)):
            key2 = (int(u_y[idx]), int(u_x[idx]))
            cell_indices.setdefault(key2, []).append(idx)

        for (cy, cx), idxs in cell_indices.items():
            avail_rem = int(available[cy, cx])
            total_rem = sum(int(merged[i]) for i in idxs)
            for idx in idxs:
                cnt_i = int(merged[idx])
                if cnt_i <= 0 or avail_rem <= 0:
                    survived[idx] = 0
                    continue
                if total_rem <= avail_rem:
                    # no thinning needed — all fit
                    s = cnt_i
                else:
                    # draw binom(cnt_i, avail_rem / total_rem): equal per-capita, strain-blind
                    p = torch.tensor(avail_rem / total_rem, dtype=torch.float32, device=dev)
                    s = int(binom(
                        torch.tensor([cnt_i], dtype=torch.int32, device=dev),
                        p.unsqueeze(0),
                        generator
                    )[0].item())
                    # hard clamp: s can't exceed remaining available (guards float rounding)
                    s = min(s, avail_rem)
                survived[idx] = s
                avail_rem -= s
                total_rem -= cnt_i

    # --- 3. write back to fixed-K slots ---
    new_sid = live_sid.clone()
    new_cnt = live_count.clone()
    new_birth = birth_tick.clone()
    # python-level seat per cell (v1 correctness-first; vectorize later)
    for idx in range(survived.shape[0]):
        c = int(survived[idx])
        if c <= 0:
            continue
        y = int(u_y[idx]); x = int(u_x[idx]); s = int(u_sid[idx])
        slots_sid = new_sid[y, x]; slots_cnt = new_cnt[y, x]
        existing = (slots_sid == s) & (slots_cnt > 0)
        if existing.any():
            k = int(torch.nonzero(existing)[0])
            slots_cnt[k] += c
        else:
            empty = torch.nonzero(slots_cnt == 0)
            assert empty.numel() > 0, "C=K invariant violated: no empty slot"
            k = int(empty[0])
            slots_sid[k] = s
            slots_cnt[k] = c
            new_birth[y, x, k] = T
    return new_sid, new_cnt, new_birth
