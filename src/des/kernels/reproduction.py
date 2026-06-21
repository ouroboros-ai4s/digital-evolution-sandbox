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


def _mutate_sequence(seq: tuple[str, ...], mutable_mask, spectrum, generator):
    """Replace one mutable slot's letter via Gumbel-argmax over `spectrum`.
    spectrum: tuple[(letter, q)]. Returns new sequence tuple. Reads only the sequence."""
    if not spectrum:
        return seq
    # choose first mutable slot whose current letter participates (v1 simple rule)
    idxs = [i for i, m in enumerate(mutable_mask) if m]
    if not idxs:
        return seq
    # device-agnostic (D6): all draws must live on the generator's device, else a CUDA
    # generator driving CPU tensors raises at the first mutation.
    dev = generator.device
    pick_idx = int(torch.randint(len(idxs), (1,), generator=generator, device=dev).item())
    slot = idxs[pick_idx]
    letters = [t for t, _ in spectrum]
    logq = torch.log(torch.tensor([q for _, q in spectrum], device=dev) + 1e-12)
    gumbel = -torch.log(-torch.log(torch.rand(len(letters), generator=generator, device=dev) + 1e-12) + 1e-12)
    pick = int(torch.argmax(logq + gumbel).item())
    new = list(seq); new[slot] = letters[pick]
    return tuple(new)


def phase2_reproduce(world, snap_sid, snap_count, snap_faction, phe, table,
                     birth_tick, T, generator):
    """Slot-level vectorized reproduction. Loops over the <=4 directions, NOT over
    strains. Offspring move to the neighbor (B4 fix: roll the data into place; read
    target coords from the static meshgrid -- never roll a coordinate grid). Offspring
    carry the parent slot's faction. Mutants are batch-minted once per tick."""
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
    produced_mut = torch.zeros((H, W, K), dtype=torch.bool, device=dev)
    for (dy, dx) in ALL_DIRECTIONS:
        bit = ALL_DIRECTIONS.index((dy, dx))
        dir_mask = ((dir_bits >> bit) & 1).bool()        # slots that move this direction
        active = fires & dir_mask
        a = (snap_count * active).to(torch.int32)        # [H,W,K] firing counts
        scattered = binom(a, f, generator)               # offspring per source slot
        mut = binom(scattered, p_x, generator)           # mutant split
        non = scattered - mut
        produced_mut |= (mut > 0)
        # B4 fix: roll the DATA (counts, sid, faction) to the neighbor; coords stay static
        r_non = torch.roll(non, shifts=(dy, dx), dims=(0, 1))
        r_mut = torch.roll(mut, shifts=(dy, dx), dims=(0, 1))
        r_sid = torch.roll(snap_sid, shifts=(dy, dx), dims=(0, 1))
        r_fac = torch.roll(faction_long, shifts=(dy, dx), dims=(0, 1))
        rolled.append((r_non, r_mut, r_sid, r_fac))

    # --- batch-mint mutant children once per tick (deterministic: sorted parent order) ---
    n_before = len(table) + 1
    child_map = torch.arange(n_before, dtype=torch.int64, device=dev)  # parent->child sid
    mut_parents = torch.unique(sid_long[produced_mut])
    mutable = BB0_TEMPLATE["mutable"]
    for p in sorted(int(x) for x in mut_parents.tolist()):
        if p == 0:
            continue
        seq = table.sequence_of(p)
        spectrum = table.phenotype_of(p).spectrum
        child = table.get_or_mint(_mutate_sequence(seq, mutable, spectrum, generator))
        child_map[p] = child

    # --- pass 2: emit arrival records ---
    for (r_non, r_mut, r_sid, r_fac) in rolled:
        # non-mutant offspring keep the parent sid
        buf.add(ty.flatten(), tx.flatten(), r_sid.to(torch.int32).flatten(),
                r_non.flatten(), r_fac.flatten())
        # mutant offspring carry child sid (faction unchanged)
        child_sid = child_map[r_sid.long()].to(torch.int32)
        buf.add(ty.flatten(), tx.flatten(), child_sid.flatten(),
                r_mut.flatten(), r_fac.flatten())

    # --- migration out (unchanged: p_leave departers vanish, design line 115/309) ---
    leave = binom(snap_count, p_leave, generator)
    leave = torch.minimum(leave, snap_count)
    live = (world.count - leave).clamp(min=0)
    return buf, live
