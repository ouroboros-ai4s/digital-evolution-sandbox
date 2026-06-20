# src/des/world.py
from __future__ import annotations
import torch
from des.phenotype_cache import StrainTable
from des.registry import BB0_TEMPLATE


class World:
    def __init__(self, H: int, W: int, K: int, device: torch.device) -> None:
        self.H, self.W, self.K, self.device = H, W, K, device
        self.strain_id = torch.zeros((H, W, K), dtype=torch.int32, device=device)
        self.count = torch.zeros((H, W, K), dtype=torch.int32, device=device)

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.strain_id.clone(), self.count.clone()

    def occupancy(self) -> torch.Tensor:
        return self.count.sum(dim=-1)

    def distinct_per_cell(self) -> torch.Tensor:
        return (self.count > 0).sum(dim=-1)


def init_bb0(H: int, W: int, K: int, device: torch.device,
             table: StrainTable, fill_per_cell: int) -> World:
    assert fill_per_cell <= K, "fill must fit in K slots"
    w = World(H, W, K, device)
    bb0 = table.get_or_mint(BB0_TEMPLATE["layout"])
    w.strain_id[:, :, 0] = bb0
    w.count[:, :, 0] = fill_per_cell
    return w
