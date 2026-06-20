# src/des/kernels/antagonism.py
from __future__ import annotations
import torch
from des.kernels.common import z_eff, fires_this_tick


def phase1_antagonism(
    strain_id: torch.Tensor,
    count: torch.Tensor,
    phe: dict[str, torch.Tensor],
    birth_tick: torch.Tensor,
    T: int,
    z_max: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """PHASE1 antagonism kernel.

    For every cell, every attacker slot i attacks every prey slot j where:
      - i != j (different slot index — not structural, but G10 enforces strain identity below)
      - both have count > 0
      - attacker fires this tick (fires_this_tick)
      - (prey_mask[i] & feature_mask[j]) != 0  (family-based targeting only)
      - strain_id[i] != strain_id[j]           (G10: same-strain immunity)

    kills from i on j = round(count[i] * z_eff(z_raw[i], z_max)), proportionally
    capped so total kills on j never exceed count[j].
    Attacker self-loss = kills / z_eff (with z_eff > 0 guard).

    Both directions computed from the same snapshot; applied simultaneously.
    """
    # --- gather per-slot phenotype arrays via index lookup [H, W, K] ---
    sid_long = strain_id.long()                      # [H, W, K]
    z_raw = phe["z_raw"][sid_long]                   # [H, W, K]
    prey_m = phe["prey_mask"][sid_long]              # [H, W, K]  int64 bitmask
    feat_m = phe["feature_mask"][sid_long]           # [H, W, K]  int64 bitmask
    period = phe["period"][sid_long]                 # [H, W, K]

    alive = count > 0                                # [H, W, K]
    fires = fires_this_tick(birth_tick, period, T) & alive  # [H, W, K]
    z = z_eff(z_raw, z_max)                          # [H, W, K]; 0 if z_raw <= 0

    # --- build valid hit matrix [H, W, K_i, K_j] ---
    # prey_mask of attacker i vs feature_mask of prey j
    pi = prey_m.unsqueeze(-1)                        # [H, W, K, 1]
    fj = feat_m.unsqueeze(-2)                        # [H, W, 1, K]
    hit = (pi & fj) != 0                             # [H, W, K, K] bool

    # G10: same-strain immunity — compare strain identity, not strength
    sid_i = sid_long.unsqueeze(-1)                   # [H, W, K, 1]
    sid_j = sid_long.unsqueeze(-2)                   # [H, W, 1, K]
    diff_strain = sid_i != sid_j                     # [H, W, K, K] bool

    fires_i = fires.unsqueeze(-1)                    # [H, W, K, 1]  attacker fires
    alive_j = alive.unsqueeze(-2)                    # [H, W, 1, K]  prey alive

    valid = hit & diff_strain & fires_i & alive_j    # [H, W, K, K] bool

    # --- raw kill from attacker i onto prey j ---
    a_i = count.unsqueeze(-1).float()               # [H, W, K, 1]
    z_i = z.unsqueeze(-1)                            # [H, W, K, 1]
    raw_kill = torch.where(valid, (a_i * z_i).round(),
                           torch.zeros_like(a_i))    # [H, W, K_i, K_j]

    # --- proportional cap: total kills on j must not exceed count[j] ---
    b_j = count.unsqueeze(-2).float()               # [H, W, 1, K_j]
    raw_total_on_j = raw_kill.sum(dim=-2, keepdim=True)  # [H, W, 1, K_j]
    over = raw_total_on_j > b_j
    ratio = torch.where(over,
                        b_j / raw_total_on_j.clamp(min=1e-9),
                        torch.ones_like(b_j))        # [H, W, 1, K_j]
    actual_kill = raw_kill * ratio                   # [H, W, K_i, K_j]

    # --- aggregate losses ---
    # prey j loses sum of kills from all attackers
    prey_loss = actual_kill.sum(dim=-2)              # [H, W, K_j]

    # attacker i self-loss = sum of (kills_i_j / z_eff_i) over j
    # guard: z_eff==0 means no attack was valid from that slot (z_raw<=0 → valid=False)
    # use clamp to avoid divide-by-zero in the no-z-raw-but-somehow-valid edge case
    z_safe = z.unsqueeze(-1).clamp(min=1e-9)        # [H, W, K_i, 1]
    self_loss = (actual_kill / z_safe).sum(dim=-1)   # [H, W, K_i]

    # --- apply both losses from snapshot simultaneously ---
    new_count = count.float() - prey_loss - self_loss
    return new_count.round().clamp(min=0).to(torch.int32)
