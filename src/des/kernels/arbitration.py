# src/des/kernels/arbitration.py
from __future__ import annotations
import torch


def _urn_draw_contested(merged, rec_cell, avail_grid, contested, W, generator, dev):
    """Exact multivariate-hypergeometric survivor counts for contested records,
    drawn at RECORD granularity (no per-individual expansion).

    Per design.md L170-175/L201: draw exactly avail = K - occupied survivors from
    the pooled arriving individuals of each contested cell; per-individual survival
    prob = avail/total, weighted ONLY by delivered count -> faction-blind & sid-blind
    by construction. Cost ~ O(max_avail * n_contested_cells * R_max); max_avail <= K
    and ~0 when the world is full (the hot regime). Uses only torch.rand.

    Returns: seated [n_rec] int64 -- survivor count per contested record, 0 elsewhere.
    """
    n_rec = merged.shape[0]
    seated = torch.zeros(n_rec, dtype=torch.int64, device=dev)
    c_idx = torch.nonzero(contested, as_tuple=False).flatten()      # [m] global rec idx
    m = c_idx.numel()
    if m == 0:
        return seated

    c_cell = rec_cell[c_idx]                                         # [m] cell per rec
    cc, inv = torch.unique(c_cell, return_inverse=True)             # cc:[n_cc], inv:[m]
    n_cc = cc.numel()

    # within-cell column index per contested record (group by row=inv, enumerate)
    order = torch.argsort(inv, stable=True)
    inv_s = inv[order]
    seg = torch.ones(m, dtype=torch.bool, device=dev)
    seg[1:] = inv_s[1:] != inv_s[:-1]
    grp = torch.cumsum(seg.long(), 0) - 1
    start = torch.searchsorted(grp.contiguous(), grp.contiguous())
    col_s = torch.arange(m, device=dev) - start                     # col in sorted order
    col = torch.empty(m, dtype=torch.int64, device=dev)
    col[order] = col_s                                              # scatter back
    R_max = int(col.max().item()) + 1

    remaining = torch.zeros((n_cc, R_max), dtype=torch.float32, device=dev)
    remaining[inv, col] = merged[c_idx].to(torch.float32)          # per-record counts
    survivor = torch.zeros((n_cc, R_max), dtype=torch.int64, device=dev)
    cell_avail = avail_grid.flatten()[cc].to(torch.int64)          # [n_cc] avail per cell
    drawn = torch.zeros(n_cc, dtype=torch.int64, device=dev)
    rows = torch.arange(n_cc, device=dev)
    max_avail = int(cell_avail.max().item())
    tiny = 1e-20
    neg_inf = torch.tensor(float("-inf"), device=dev)

    for _ in range(max_avail):
        active = drawn < cell_avail                                 # [n_cc] bool
        u = torch.rand((n_cc, R_max), generator=generator, device=dev).clamp_(tiny, 1.0)
        gumbel = -torch.log(-torch.log(u))                          # standard Gumbel
        score = torch.where(remaining > 0,
                            torch.log(remaining.clamp(min=tiny)) + gumbel,
                            neg_inf.expand_as(remaining))           # mask empty records
        pick = score.argmax(dim=1)                                  # [n_cc] one rec/cell
        inc = active.to(torch.int64)
        survivor.index_put_((rows, pick), inc, accumulate=True)
        remaining.index_put_((rows, pick), (-inc).to(remaining.dtype), accumulate=True)
        drawn = drawn + inc

    seated[c_idx] = survivor[inv, col]                              # gather back
    return seated


def phase3_arbitrate_vec(live_sid, live_count, live_faction, arrivals, K,
                         birth_tick, T, generator, MAXSID, NFAC=4):
    """Fully vectorized K-wall arbitration with faction.

    Sections:
      1. coalesce arrivals by (cell, sid, faction)            -- tensor unique+scatter_add
      2. K-wall multivariate-hypergeometric draw via random keys (NO per-cell loop,
         NO .item() sync). Keys are i.i.d. uniform, never read sid/faction -> survival
         prob = avail/total for every individual (spec red-line 7-J, fair by construction).
      3. vectorized writeback to fixed-K slots, keyed on (sid, faction).
    """
    H, W, _ = live_count.shape
    dev = live_count.device
    a_ty, a_tx, a_sid, a_cnt, a_fac = arrivals

    # --- 1. coalesce arrivals by (cell, sid, faction) ---
    if a_cnt.numel() > 0:
        cell = a_ty.long() * W + a_tx.long()
        key = (cell * MAXSID + a_sid.long()) * NFAC + a_fac.long()
        uniq, inv = torch.unique(key, return_inverse=True)
        merged = torch.zeros(uniq.shape[0], dtype=torch.int64, device=dev)
        merged.scatter_add_(0, inv, a_cnt.long())
        u_fac = (uniq % NFAC).to(torch.int8)
        rest = uniq // NFAC
        u_sid = (rest % MAXSID).to(torch.int32)
        u_cell = rest // MAXSID
        u_y = (u_cell // W).long()
        u_x = (u_cell % W).long()
    else:
        u_y = torch.zeros(0, dtype=torch.long, device=dev)
        u_x = torch.zeros(0, dtype=torch.long, device=dev)
        u_sid = torch.zeros(0, dtype=torch.int32, device=dev)
        u_fac = torch.zeros(0, dtype=torch.int8, device=dev)
        merged = torch.zeros(0, dtype=torch.int64, device=dev)

    new_sid = live_sid.clone()
    new_cnt = live_count.clone()
    new_faction = live_faction.clone()
    new_birth = birth_tick.clone()
    if merged.numel() == 0:
        return new_sid, new_cnt, new_faction, new_birth

    # --- 2. K-wall: random-key multivariate hypergeometric (vectorized) ---
    n_rec = merged.shape[0]
    rec_cell = u_y * W + u_x                                  # [n_rec] cell of each record
    resident_occ = live_count.sum(dim=-1)                    # [H,W]
    avail_grid = (K - resident_occ).clamp(min=0)             # [H,W]
    # per-record avail and per-cell total (scatter over records sharing a cell)
    rec_avail = avail_grid.flatten()[rec_cell]               # [n_rec]
    # total individuals arriving per cell, broadcast back to each record
    cell_total = torch.zeros(H * W, dtype=torch.int64, device=dev)
    cell_total.scatter_add_(0, rec_cell, merged)
    rec_total = cell_total[rec_cell]                         # [n_rec]

    survived = merged.clone()
    # cells that fit entirely (total <= avail): keep merged as-is.
    contested = rec_total > rec_avail                        # [n_rec] bool
    if contested.any():
        # record-granularity exact urn draw (no per-individual expansion).
        seated = _urn_draw_contested(merged, rec_cell, avail_grid, contested,
                                     W, generator, dev)
        survived = torch.where(contested, seated, survived)

    # --- 3. vectorized writeback, keyed on (sid, faction) ---
    mask = survived > 0
    if not mask.any():
        return new_sid, new_cnt, new_faction, new_birth
    v_y = u_y[mask]; v_x = u_x[mask]
    v_sid = u_sid[mask]; v_fac = u_fac[mask]
    v_cnt = survived[mask].to(torch.int32)
    N = v_y.shape[0]

    resident_sid = new_sid[v_y, v_x]                          # [N,K]
    resident_cnt = new_cnt[v_y, v_x]                          # [N,K]
    resident_fac = new_faction[v_y, v_x]                      # [N,K]
    # resident-hit: same sid AND same faction AND occupied
    hit = ((resident_sid == v_sid[:, None]) &
           (resident_fac == v_fac[:, None]) &
           (resident_cnt > 0))                                # [N,K]
    is_resident = hit.any(dim=1)
    is_new = ~is_resident

    if is_resident.any():
        r = is_resident
        r_k = hit[r].long().argmax(dim=1)
        new_cnt.index_put_((v_y[r], v_x[r], r_k), v_cnt[r], accumulate=True)

    if is_new.any():
        n = is_new
        n_y = v_y[n]; n_x = v_x[n]; n_sid = v_sid[n]; n_fac = v_fac[n]; n_cnt = v_cnt[n]
        Nnew = n_y.shape[0]
        cell_lin = n_y * W + n_x
        sort_ord = torch.argsort(cell_lin, stable=True)
        n_y = n_y[sort_ord]; n_x = n_x[sort_ord]; n_sid = n_sid[sort_ord]
        n_fac = n_fac[sort_ord]; n_cnt = n_cnt[sort_ord]; cell_lin = cell_lin[sort_ord]
        cell_change = torch.ones(Nnew, dtype=torch.bool, device=dev)
        cell_change[1:] = cell_lin[1:] != cell_lin[:-1]
        group_cumsum = cell_change.long().cumsum(0) - 1
        group_start = torch.searchsorted(group_cumsum.contiguous(),
                                         group_cumsum.contiguous())
        within_rank = torch.arange(Nnew, device=dev) - group_start
        cell_empty = (new_cnt[n_y, n_x] == 0)                 # [Nnew,K]
        empty_ord = cell_empty.long().cumsum(dim=1) - 1
        target_match = (empty_ord == within_rank[:, None]) & cell_empty
        assert target_match.any(dim=1).all(), \
            "C=K invariant violated: no empty slot for a new (sid,faction)"
        target_k = target_match.long().argmax(dim=1)
        new_sid.index_put_((n_y, n_x, target_k), n_sid, accumulate=False)
        new_cnt.index_put_((n_y, n_x, target_k), n_cnt.int(), accumulate=False)
        new_faction.index_put_((n_y, n_x, target_k), n_fac, accumulate=False)
        new_birth.index_put_((n_y, n_x, target_k),
                             torch.full((Nnew,), T, dtype=torch.int32, device=dev),
                             accumulate=False)

    return new_sid, new_cnt, new_faction, new_birth
