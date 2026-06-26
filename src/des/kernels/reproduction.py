# src/des/kernels/reproduction.py
from __future__ import annotations
import torch
from des.kernels.common import fires_this_tick, binom
from des.registry import BB0_TEMPLATE, motif_blocks


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


def _mutation_outcomes(seq, mutable, spectrum, blocks, slots_per_event=1):
    """Per-parent mutation categorical. N=1 (default) keeps the legacy single-slot
    overwrite path verbatim (same enumeration order, same weights, same RNG
    call count — spec §3 red line 'by construction byte-identical, not merely
    distributionally equal'). N>=2 enumerates unordered slot-sets of size N
    (clamped to #mutable) × per-slot spectrum letters; weight of (slot-set S,
    letters) = (1/C(m,N)) * prod(q(letter_s) for s in S). At N=1 the joint
    formula reduces to q/C(m,1) = q/m — continuous with the legacy path; N=1
    nonetheless takes the verbatim legacy branch to guarantee byte-identity.

    Returns (child_sequences, weights) over the full enumeration; weights
    sum to 1. Self-loops (letter == current) yield child == parent. Pure fn
    of (sequence, spectrum, motif blocks, N) — reads no world state."""
    slot_idx = [i for i, ok in enumerate(mutable) if ok]
    if not slot_idx or not spectrum:
        return [], []
    # index -> (start, end) of the block covering position i (S6 motif overwrite).
    cover: dict[int, tuple[int, int]] = {}
    for s, e, _ in blocks:
        for k in range(s, e):
            cover[k] = (s, e)

    if slots_per_event == 1:
        # ---------------------------------------------------------------
        # N=1: legacy verbatim path (pre-S7). DO NOT REFACTOR INTO JOINT
        # ENUMERATION — spec §3 red line requires the byte-identical
        # enumeration order + RNG call count, not just the same weights.
        # ---------------------------------------------------------------
        children, weights = [], []
        for s in slot_idx:                      # ascending: canonical order
            for letter, q in spectrum:          # spectrum already sorted in _spectrum_for
                start, end = cover[s]
                new = list(seq)
                for k in range(start, end):      # S6: overwrite the whole covering block
                    new[k] = letter
                children.append(tuple(new))
                weights.append(q / len(slot_idx))
        return children, weights

    # N>=2: joint enumeration path lands in Task 4.
    raise NotImplementedError(
        f"_mutation_outcomes slots_per_event>=2 lands in S7 Task 4; "
        f"got slots_per_event={slots_per_event}. "
        f"(P_cascade is the sole roster letter with slots=2 and is minted in S8.)"
    )


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

    # S5: windowed live f. (T - birth_tick) % burst_w < burst_k → on; on→f_hi, off→f_lo.
    # burst_w.clamp(min=1) is defensive (defaults are 1; FBURST=12, F_NOVA=20).
    f_hi    = phe["f_hi"][sid_long]                           # [H,W,K] float32
    f_lo    = phe["f_lo"][sid_long]                           # [H,W,K] float32
    burst_w = phe["burst_w"][sid_long].clamp(min=1)           # [H,W,K] int64
    burst_k = phe["burst_k"][sid_long]                        # [H,W,K] int64
    on = ((T - birth_tick) % burst_w) < burst_k               # [H,W,K] bool
    f  = torch.where(on, f_hi, f_lo)                          # [H,W,K] float32
    p_leave = phe["p_leave"][sid_long]
    p_x = phe["p_x"][sid_long]
    repro_period = phe["repro_period"][sid_long]
    dir_bits = phe["dir_bits"][sid_long]
    in_place_mask = phe["in_place"][sid_long].bool()      # [H,W,K]
    rand_dir_mask = phe["rand_dir"][sid_long].bool()      # [H,W,K]

    alive = snap_count > 0
    fires = fires_this_tick(birth_tick, repro_period, T) & alive & (f > 0)

    buf = ArrivalBuffer(dev)

    # static target-coordinate grids, broadcast to slot shape (NEVER rolled)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev),
                            torch.arange(W, device=dev), indexing="ij")
    ty = yy.unsqueeze(-1).expand(H, W, K).contiguous()   # [H,W,K]
    tx = xx.unsqueeze(-1).expand(H, W, K).contiguous()

    faction_long = snap_faction.to(torch.int8)

    # --- pass 1: 静态 dir_bits 路径 (排除 in_place / rand_dir slot) ---
    static_mask = (~in_place_mask) & (~rand_dir_mask)
    rolled = []                 # list of (rolled_non, rolled_mut, rolled_sid, rolled_fac)
    for (dy, dx) in ALL_DIRECTIONS:
        bit = ALL_DIRECTIONS.index((dy, dx))
        dir_mask = ((dir_bits >> bit) & 1).bool() & static_mask
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

    # --- S4: in-place path (FSTACK) — deposit in source cell, no roll ---
    in_place_active = fires & in_place_mask
    if in_place_active.any():
        a_ip = (snap_count * in_place_active).to(torch.int32)
        scattered_ip = binom(a_ip, f, generator)
        mut_ip = binom(scattered_ip, p_x, generator)
        non_ip = scattered_ip - mut_ip
        rolled.append((non_ip, mut_ip, snap_sid, faction_long))

    # --- S4: rand-dir path (FDRIFT) — draw 1-of-4 per firing slot from world RNG ---
    rand_active = fires & rand_dir_mask
    if rand_active.any():
        a_rd = (snap_count * rand_active).to(torch.int32)
        scattered_rd = binom(a_rd, f, generator)
        mut_rd = binom(scattered_rd, p_x, generator)
        non_rd = scattered_rd - mut_rd
        dir_idx = torch.randint(0, 4, (H, W, K), generator=generator,
                                device=dev, dtype=torch.int64)
        for d in range(4):
            dy, dx = ALL_DIRECTIONS[d]
            sel = (dir_idx == d) & rand_active
            non_d = (non_rd * sel).to(non_rd.dtype)
            mut_d = (mut_rd * sel).to(mut_rd.dtype)
            r_non = torch.roll(non_d, shifts=(dy, dx), dims=(0, 1))
            r_mut = torch.roll(mut_d, shifts=(dy, dx), dims=(0, 1))
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
            phe_obj = table.phenotype_of(p)
            spectrum = phe_obj.spectrum
            children, weights = _mutation_outcomes(
                seq, BB0_TEMPLATE["mutable"], spectrum, motif_blocks(seq),
                slots_per_event=phe_obj.slots_per_event)
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
