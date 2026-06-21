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
        self._fac: list[torch.Tensor] = []

    def add(self, ty, tx, sid, cnt, fac) -> None:
        m = cnt > 0
        if m.any():
            self._ty.append(ty[m]); self._tx.append(tx[m])
            self._sid.append(sid[m]); self._cnt.append(cnt[m])
            self._fac.append(fac[m])

    def tensors(self):
        if not self._cnt:
            z = torch.zeros(0, dtype=torch.int64, device=self.device)
            return (z, z, z.to(torch.int32), z.to(torch.int32),
                    z.to(torch.int8))
        return (torch.cat(self._ty), torch.cat(self._tx),
                torch.cat(self._sid), torch.cat(self._cnt),
                torch.cat(self._fac))


def _mutation_outcomes(seq, mutable, spectrum):
    """Per-parent mutation categorical (design L246: uniform mutable slot x spectrum
    letter). Returns (child_sequences, weights) over the full slot x spectrum product;
    weights sum to 1. Self-loops (letter == current) yield child == parent. Pure fn of
    the sequence + its spectrum -- reads no world state."""
    slot_idx = [i for i, ok in enumerate(mutable) if ok]
    if not slot_idx or not spectrum:
        return [], []
    children, weights = [], []
    for s in slot_idx:                      # ascending: canonical order
        for letter, q in spectrum:          # spectrum already sorted in _spectrum_for
            new = list(seq); new[s] = letter
            children.append(tuple(new))
            weights.append(q / len(slot_idx))
    return children, weights


def phase2_reproduce(world, snap_sid, snap_count, snap_faction, phe, table,
                     birth_tick, T, generator):
    """Slot-level vectorized reproduction. Loops over the <=4 directions, NOT over
    strains. Offspring move to the neighbor (B4 fix: roll the data into place; read
    target coords from the static meshgrid -- never roll a coordinate grid). Offspring
    carry the parent slot's faction. Mutation is per-individual (design L243): each of
    the n*p(x) mutant individuals draws its outcome independently.
    # ponytail: per-individual expand + per-parent multinomial scatter. Ceiling: total
    # mutant headcount is small (p_x ~ mu). If a high-p_x regime makes it explode,
    # upgrade to per-cell Multinomial(mut, w) batched over cells -- not before."""
    from des.registry import ALL_DIRECTIONS
    H, W, K = snap_count.shape
    dev = world.device
    sid_long = snap_sid.long()

    f = phe["f"][sid_long]                       # [H,W,K]
    p_leave = phe["p_leave"][sid_long]
    p_x = phe["p_x"][sid_long]
    repro_period = phe["repro_period"][sid_long]
    dir_bits = phe["dir_bits"][sid_long]

    alive = snap_count > 0
    fires = fires_this_tick(birth_tick, repro_period, T) & alive & (f > 0)

    buf = ArrivalBuffer(dev)

    # static target-coordinate grids, broadcast to slot shape (NEVER rolled)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev),
                            torch.arange(W, device=dev), indexing="ij")
    ty = yy.unsqueeze(-1).expand(H, W, K).contiguous()   # [H,W,K]
    tx = xx.unsqueeze(-1).expand(H, W, K).contiguous()

    faction_long = snap_faction.to(torch.int8)

    # --- pass 1: per direction, compute non-mutant + mutant offspring, roll into place ---
    rolled = []                 # list of (rolled_non, rolled_mut, rolled_sid, rolled_fac)
    for (dy, dx) in ALL_DIRECTIONS:
        bit = ALL_DIRECTIONS.index((dy, dx))
        dir_mask = ((dir_bits >> bit) & 1).bool()        # slots that move this direction
        active = fires & dir_mask
        a = (snap_count * active).to(torch.int32)        # [H,W,K] firing counts
        scattered = binom(a, f, generator)               # offspring per source slot
        mut = binom(scattered, p_x, generator)           # mutant split
        non = scattered - mut
        # B4 fix: roll the DATA (counts, sid, faction) to the neighbor; coords stay static
        r_non = torch.roll(non, shifts=(dy, dx), dims=(0, 1))
        r_mut = torch.roll(mut, shifts=(dy, dx), dims=(0, 1))
        r_sid = torch.roll(snap_sid, shifts=(dy, dx), dims=(0, 1))
        r_fac = torch.roll(faction_long, shifts=(dy, dx), dims=(0, 1))
        rolled.append((r_non, r_mut, r_sid, r_fac))

    # --- pass 2a: emit non-mutant offspring (keep parent sid; faction unchanged) ---
    fy = ty.flatten(); fx = tx.flatten()
    for (r_non, r_mut, r_sid, r_fac) in rolled:
        buf.add(fy, fx, r_sid.to(torch.int32).flatten(),
                r_non.flatten(), r_fac.flatten())

    # --- pass 2b: per-INDIVIDUAL mutation (design L243: n·p(x) individuals each draw
    # independently; L246: pick a mutable slot uniformly, then a letter from spectrum;
    # L201: same-sequence mutants merge by key automatically via get_or_mint). ---
    # Build a flat stream of mutant individuals, each carrying (cell, faction, parent sid).
    cell_flat = (fy * W + fx)                              # [H*W*K]
    cells, facs, parents = [], [], []
    for (_r_non, r_mut, r_sid, r_fac) in rolled:
        m = r_mut.flatten() > 0
        if not m.any():
            continue
        reps = r_mut.flatten()[m]
        cells.append(torch.repeat_interleave(cell_flat[m], reps))
        facs.append(torch.repeat_interleave(r_fac.flatten()[m], reps))
        parents.append(torch.repeat_interleave(r_sid.flatten()[m].long(), reps))

    if cells:
        ind_cell = torch.cat(cells)
        ind_fac = torch.cat(facs)
        ind_parent = torch.cat(parents)
        child_sid_i = ind_parent.clone()                  # default: self (parent)
        # loop over DISTINCT mutant parents (same cardinality the old code paid)
        for p in sorted(set(int(x) for x in ind_parent.tolist())):
            if p == 0:
                continue
            seq = table.sequence_of(p)
            spectrum = table.phenotype_of(p).spectrum
            children, weights = _mutation_outcomes(seq, BB0_TEMPLATE["mutable"], spectrum)
            if not children:
                continue                                  # no mutation possible -> stays parent
            out_sid = [table.get_or_mint(c) for c in children]  # self-loop -> parent sid
            sel = (ind_parent == p)
            n_p = int(sel.sum())
            w = torch.tensor(weights, dtype=torch.float32, device=dev)
            draws = torch.multinomial(w, n_p, replacement=True, generator=generator)
            osid = torch.tensor(out_sid, dtype=torch.int64, device=dev)
            child_sid_i[sel] = osid[draws]
        # aggregate individuals -> (cell, faction, child sid) counts, then emit
        key = torch.stack([ind_cell, ind_fac.long(), child_sid_i], dim=1)
        uniq, cnt = torch.unique(key, dim=0, return_counts=True)
        buf.add(uniq[:, 0] // W, uniq[:, 0] % W,
                uniq[:, 2].to(torch.int32), cnt.to(torch.int32),
                uniq[:, 1].to(torch.int8))

    # --- migration out (unchanged: p_leave departers vanish, design line 115/309) ---
    leave = binom(snap_count, p_leave, generator)
    leave = torch.minimum(leave, snap_count)
    live = (world.count - leave).clamp(min=0)
    return buf, live
