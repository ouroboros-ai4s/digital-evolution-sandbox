from __future__ import annotations
import torch


def z_eff(z_raw: torch.Tensor, z_max: float) -> torch.Tensor:
    out = torch.zeros_like(z_raw)
    pos = z_raw > 0
    out[pos] = z_max * z_raw[pos] / (z_max + z_raw[pos])
    return out


def fires_this_tick(birth_tick: torch.Tensor, period: torch.Tensor, T: int) -> torch.Tensor:
    safe = period.clamp(min=1)
    fired = ((T - birth_tick) % safe) == 0
    return fired & (period > 0)


def binom(count: torch.Tensor, prob: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    p = prob.clamp(0.0, 1.0)
    n = count.float().clamp(min=0.0)
    drawn = torch.binomial(n, p, generator=generator)
    return drawn.round().to(torch.int32).clamp(min=torch.zeros_like(count), max=count)
