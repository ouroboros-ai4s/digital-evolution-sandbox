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


def phase3_arbitrate_vec(live_sid, live_count, arrivals, K, birth_tick, T, generator, MAXSID):
    """Vectorized writeback (section 3 only).  Sections 1-2 are byte-identical to
    phase3_arbitrate so the generator is consumed identically -> survivor counts match."""
    H, W, _ = live_count.shape
    dev = live_count.device
    a_ty, a_tx, a_sid, a_cnt = arrivals

    # --- 1. coalesce arrivals by (cell, strain) ---  (VERBATIM from reference)
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

    # --- 2. K-wall: per-cell individual cap ---  (VERBATIM from reference)
    resident_occ = live_count.sum(dim=-1)              # [H,W]
    available = (K - resident_occ).clamp(min=0)        # [H,W]

    survived = merged.clone()
    if merged.numel() > 0:
        cell_indices: dict[tuple[int, int], list[int]] = {}
        for idx in range(len(u_y)):
            key2 = (int(u_y[idx]), int(u_x[idx]))
            cell_indices.setdefault(key2, []).append(idx)

        for (cy, cx), idxs in sorted(cell_indices.items()):
            avail = int(available[cy, cx])
            counts = [int(merged[i]) for i in idxs]
            total = sum(counts)

            if total == 0 or avail <= 0:
                for i in idxs:
                    survived[i] = 0
                continue
            if total <= avail:
                continue
            if len(idxs) == 1:
                survived[idxs[0]] = avail
                continue

            counts_t = torch.tensor(counts, dtype=torch.int64, device=dev)
            labels = torch.repeat_interleave(
                torch.arange(len(idxs), dtype=torch.int64, device=dev),
                counts_t
            )
            perm = torch.randperm(total, generator=generator, device=dev)
            selected = labels[perm[:avail]]
            seated = torch.bincount(selected, minlength=len(idxs))

            for local_i, idx in enumerate(idxs):
                survived[idx] = int(seated[local_i].item())

    # --- 3. vectorized writeback ---
    new_sid   = live_sid.clone()
    new_cnt   = live_count.clone()
    new_birth = birth_tick.clone()

    # filter to records that actually seat at least one individual
    mask = survived > 0
    if not mask.any():
        return new_sid, new_cnt, new_birth

    v_y   = u_y[mask]                    # [N]
    v_x   = u_x[mask]                    # [N]
    v_sid = u_sid[mask]                  # [N]  int32
    v_cnt = survived[mask].to(torch.int32)  # [N]

    N = v_y.shape[0]

    # resident slots for each survivor record: [N, K]
    resident_sid = new_sid[v_y, v_x]     # [N, K]
    resident_cnt = new_cnt[v_y, v_x]     # [N, K]

    # classify: resident-hit when the same sid already has count > 0
    hit        = (resident_sid == v_sid[:, None]) & (resident_cnt > 0)  # [N, K] bool
    is_resident = hit.any(dim=1)                                          # [N] bool
    is_new      = ~is_resident                                            # [N] bool

    # --- CASE (a): residents -- add count into the existing slot ---
    if is_resident.any():
        r_mask  = is_resident
        r_y     = v_y[r_mask]
        r_x     = v_x[r_mask]
        r_cnt   = v_cnt[r_mask]
        # argmax on bool gives index of first True (== the matching slot)
        r_k     = hit[r_mask].long().argmax(dim=1)   # [Nr]
        new_cnt.index_put_((r_y, r_x, r_k), r_cnt.int(), accumulate=True)
        # sid and birth unchanged for residents

    # --- CASE (b): new strains -- find a distinct empty slot per record ---
    if is_new.any():
        n_mask = is_new
        n_y    = v_y[n_mask]
        n_x    = v_x[n_mask]
        n_sid  = v_sid[n_mask]
        n_cnt  = v_cnt[n_mask]
        Nnew   = n_y.shape[0]

        # sort by cell linear index (stable) so records in the same cell are contiguous
        cell_lin  = n_y * W + n_x                              # [Nnew]
        sort_ord  = torch.argsort(cell_lin, stable=True)
        n_y   = n_y[sort_ord]
        n_x   = n_x[sort_ord]
        n_sid = n_sid[sort_ord]
        n_cnt = n_cnt[sort_ord]
        cell_lin = cell_lin[sort_ord]

        # within-cell rank: 0,1,2,... for successive records in the same cell
        cell_change = torch.ones(Nnew, dtype=torch.bool, device=dev)
        cell_change[1:] = cell_lin[1:] != cell_lin[:-1]
        group_start_idx = torch.zeros(Nnew, dtype=torch.long, device=dev)
        # cumsum of cell_change gives 1-based group id; use to find each record's group start
        group_cumsum = cell_change.long().cumsum(0) - 1        # 0-based group id [Nnew]
        # searchsorted: for each record find the position of its group_id in group_cumsum
        # since group_cumsum is non-decreasing and cell_change marks new groups,
        # the start of group g is the first index where group_cumsum == g
        # equivalent: group_start[i] = first index j where cell_lin[j] == cell_lin[i]
        group_start_idx = torch.searchsorted(group_cumsum.contiguous(),
                                             group_cumsum.contiguous())
        within_rank = torch.arange(Nnew, device=dev) - group_start_idx  # [Nnew]

        # empty slots in new_cnt at the target cells after resident writes (use updated new_cnt)
        cell_empty  = (new_cnt[n_y, n_x] == 0)                # [Nnew, K] bool
        empty_ord   = cell_empty.long().cumsum(dim=1) - 1      # 0-based ordinal among empties
        target_match = (empty_ord == within_rank[:, None]) & cell_empty  # [Nnew, K]

        # production guard: C=K invariant -- enough empty slots for all new strains
        assert target_match.any(dim=1).all(), \
            "C=K invariant violated in vec writeback: no empty slot for a new strain"

        target_k = target_match.long().argmax(dim=1)           # [Nnew]

        new_sid.index_put_(   (n_y, n_x, target_k), n_sid,                    accumulate=False)
        new_cnt.index_put_(   (n_y, n_x, target_k), n_cnt.int(),              accumulate=False)
        new_birth.index_put_( (n_y, n_x, target_k),
                               torch.full((Nnew,), T, dtype=torch.int32, device=dev),
                               accumulate=False)

    return new_sid, new_cnt, new_birth
