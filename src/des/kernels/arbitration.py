# src/des/kernels/arbitration.py
from __future__ import annotations
import torch


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

    # Multivariate-hypergeometric draw via uniform random permutation (torch.randperm).
    # Pool all arriving individuals (each tagged by its strain index), draw exactly
    # `available` of them WITHOUT replacement, count per strain.
    #
    # Guarantees (three locked constraints):
    #   1. Equal per-capita / strain-blind: every individual has survival prob
    #      `available/total` regardless of strain — by symmetry of the uniform permutation.
    #   2. Hard cap (prob 1): exactly `available` seated, every draw — never exceeds K.
    #   3. Order-independent: result distribution is symmetric in strains; enumeration
    #      order cannot bias it.
    #
    # Cells are processed in fixed row-major order over contested cells; the same
    # generator is threaded sequentially, giving bit-reproducible results across runs.
    survived = merged.clone()
    if merged.numel() > 0:
        # group indices by cell — fixed sort order: ascending (y,x) = row-major
        cell_indices: dict[tuple[int, int], list[int]] = {}
        for idx in range(len(u_y)):
            key2 = (int(u_y[idx]), int(u_x[idx]))
            cell_indices.setdefault(key2, []).append(idx)

        for (cy, cx), idxs in sorted(cell_indices.items()):
            avail = int(available[cy, cx])
            counts = [int(merged[i]) for i in idxs]
            total = sum(counts)

            # short-circuit: nothing to seat or no thinning needed
            if total == 0 or avail <= 0:
                for i in idxs:
                    survived[i] = 0
                continue
            if total <= avail:
                # all fit — no draw needed
                continue  # survived already == merged (cloned above)

            # single strain over capacity: seat exactly `avail` of it
            # (the randperm formula handles this, but short-circuit avoids the randperm call)
            if len(idxs) == 1:
                survived[idxs[0]] = avail
                continue

            # --- multivariate hypergeometric via randperm ---
            # Build label vector: each individual tagged by its local strain index (0..n-1)
            counts_t = torch.tensor(counts, dtype=torch.int64, device=dev)
            labels = torch.repeat_interleave(
                torch.arange(len(idxs), dtype=torch.int64, device=dev),
                counts_t
            )                                                  # length = total
            perm = torch.randperm(total, generator=generator, device=dev)
            selected = labels[perm[:avail]]                    # draw exactly `avail`
            seated = torch.bincount(selected, minlength=len(idxs))  # sums to exactly `avail`

            for local_i, idx in enumerate(idxs):
                survived[idx] = int(seated[local_i].item())

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
