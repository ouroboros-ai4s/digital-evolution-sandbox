# src/des/kernels/antagonism.py
from __future__ import annotations
import torch
from des.kernels.common import z_eff, fires_this_tick


def phase1_antagonism(
    strain_id: torch.Tensor,
    count: torch.Tensor,
    faction: torch.Tensor,
    phe: dict[str, torch.Tensor],
    birth_tick: torch.Tensor,
    T: int,
    z_max: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """PHASE1 antagonism kernel.

    For every cell, every attacker slot i attacks every prey slot j where:
      - i != j (different slot index — not structural, but faction gate enforces below)
      - both have count > 0
      - attacker fires this tick (fires_this_tick)
      - (prey_mask[i] & feature_mask[j]) != 0  (family-based targeting only)
      - faction[i] != faction[j]               (same-faction immunity)

    kills from i on j = round(count[i] * z_eff(z_raw[i], z_max)), proportionally
    capped so total kills on j never exceed count[j].
    Attacker self-loss = kills / z_eff (with z_eff > 0 guard).

    Both directions computed from the same snapshot; applied simultaneously.

    --- v1 SCALABILITY NOTE (read before scaling past the 128x128 smoke batch) ---
    This kernel builds the DENSE [H,W,K,K] per-cell pairing tensor. Spec D5/section 3.3
    mandate SPARSE per-cell pairing (gather only count>0 slots, ~5-10 per cell) and
    explicitly forbid the dense [.,.,K,K] tensor. Dense was a deliberate v1 approximation
    (plan Task 6): it computes the IDENTICAL correct result (dense just does wasted work on
    empty slots), so it is bias-free and red-line-clean -- purely a memory/throughput issue.
    Dense peak ~ H*W*K*K*4 bytes * (a few live tensors): 128x128/K=64 ~ 1 GB (fine), but
    512x512/K=256 ~ 275 GB (impossible). MUST be rewritten to sparse per-cell pairing before
    any 512x512 target-dataset run. Tracked as final-review finding C1 (deferred to target scale).
    """
    # --- gather per-slot phenotype arrays via index lookup [H, W, K] ---
    sid_long = strain_id.long()                      # [H, W, K]
    z_raw = phe["z_raw"][sid_long]                   # [H, W, K]
    prey_m = phe["prey_mask"][sid_long]              # [H, W, K]  int64 bitmask
    feat_m = phe["feature_mask"][sid_long]           # [H, W, K]  int64 bitmask
    period = phe["anta_period"][sid_long]            # [H, W, K]  Z-primitive firing clock

    alive = count > 0                                # [H, W, K]
    fires = fires_this_tick(birth_tick, period, T) & alive  # [H, W, K]
    z = z_eff(z_raw, z_max)                          # [H, W, K]; 0 if z_raw <= 0

    # --- build valid hit matrix [H, W, K_i, K_j] ---
    # prey_mask of attacker i vs feature_mask of prey j
    pi = prey_m.unsqueeze(-1)                        # [H, W, K, 1]
    fj = feat_m.unsqueeze(-2)                        # [H, W, 1, K]
    hit = (pi & fj) != 0                             # [H, W, K, K] bool

    # faction gate: fight iff attacker and prey are on DIFFERENT factions.
    # faction is a SLOT-level state, NOT gathered through sid (dual-orthogonal identity:
    # phe[sid] never sees faction). Same-faction (any sid) is immune — exactly skip the kill.
    fac_slot = faction.long()                        # [H, W, K]
    fac_i = fac_slot.unsqueeze(-1)                    # [H, W, K, 1]
    fac_j = fac_slot.unsqueeze(-2)                    # [H, W, 1, K]
    diff_faction = fac_i != fac_j                     # [H, W, K, K] bool

    fires_i = fires.unsqueeze(-1)                    # [H, W, K, 1]  attacker fires
    alive_j = alive.unsqueeze(-2)                    # [H, W, 1, K]  prey alive

    valid = hit & diff_faction & fires_i & alive_j    # [H, W, K, K] bool

    # --- raw kill from attacker i onto prey j ---
    a_i = count.unsqueeze(-1).float()               # [H, W, K, 1]
    z_i = z.unsqueeze(-1)                            # [H, W, K, 1]
    raw_kill = torch.where(valid, (a_i * z_i).round(),
                           torch.zeros_like(a_i))    # [H, W, K_i, K_j]

    # --- S1: vis-mode hit weighting ---
    # Mode-0 attackers SKIP the multiply entirely (bit-identical default path).
    # Mode-1: scale raw_kill by vis_sum_prey / L  (scatter-nip: high-vis dies faster)
    # Mode-2: scale raw_kill by max(0, n_count_prey - vis_sum_prey) / L
    #         (ghost-spike: low-vis dies faster)
    # Branch only activates when at least one attacker slot has vis_mode > 0.
    _vis_mode_i = phe["vis_mode"][sid_long].to(torch.int32)       # [H, W, K]
    _mode_mask  = _vis_mode_i > 0                                  # [H, W, K]
    if _mode_mask.any():
        L = 16.0
        # Prey phenotype: gather over K (j axis), then unsqueeze to [H, W, 1, K_j]
        _vis_sum_j  = phe["vis_sum"][sid_long].to(torch.float32).unsqueeze(-2)   # [H, W, 1, K_j]
        _n_count_j  = phe["n_count"][sid_long].to(torch.float32).unsqueeze(-2)   # [H, W, 1, K_j]
        _p_hit_m1   = _vis_sum_j / L                               # [H, W, 1, K_j]
        _p_hit_m2   = torch.clamp(_n_count_j - _vis_sum_j, min=0.0) / L
        # Attacker mode: [H, W, K_i, 1] for broadcast
        _mode_col   = _vis_mode_i.unsqueeze(-1)                    # [H, W, K_i, 1]
        _p_hit      = torch.where(_mode_col == 1, _p_hit_m1,
                          torch.where(_mode_col == 2, _p_hit_m2,
                                      torch.ones_like(_p_hit_m1)))  # [H, W, K_i, K_j]
        _active     = (_mode_col > 0).expand_as(raw_kill)           # [H, W, K_i, K_j]
        _scaled     = (raw_kill.to(torch.float32) * _p_hit).floor().to(raw_kill.dtype)
        raw_kill    = torch.where(_active, _scaled, raw_kill)

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
