# src/des/world.py
from __future__ import annotations
import torch
from des.phenotype_cache import StrainTable
from des.registry import BB0_TEMPLATE, validate_bb0_layout


class World:
    def __init__(self, H: int, W: int, K: int, device: torch.device) -> None:
        self.H, self.W, self.K, self.device = H, W, K, device
        self.strain_id = torch.zeros((H, W, K), dtype=torch.int32, device=device)
        self.count = torch.zeros((H, W, K), dtype=torch.int32, device=device)
        self.faction = torch.zeros((H, W, K), dtype=torch.int8, device=device)

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.strain_id.clone(), self.count.clone(), self.faction.clone()

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


def init_factions(H: int, W: int, K: int, device: torch.device,
                  table: StrainTable, fill_per_cell: int, n_fac: int = 4,
                  layout: tuple[str, ...] | None = None,
                  layouts: "tuple[tuple[str, ...], ...] | None" = None) -> World:
    """Seed BB0 at the four quadrant centers, one faction each, everything else empty.
    The four centers are the D4-symmetric orbit of one point (equal to grid center,
    equal nearest-wall distance, pairwise-symmetric) → no faction gets a geometric edge.

    Genotype seeding (red-line 4, downgraded "全同"→"同模板结构"):
      - layouts: a sequence of exactly 4 layouts, one per faction (asymmetric start).
      - layout: a single layout shared by all four factions (backward-compatible).
      - both None → canonical BB0_TEMPLATE["layout"] shared by all four (default unchanged).
    Passing both layout and layouts is a ValueError. Every layout is independently
    gatekeeper-validated (same fixed template structure; only slot choices may differ)."""
    assert fill_per_cell <= K, "fill must fit in K slots"
    assert n_fac == 4, "v1 seeds exactly 4 factions at the 4 quadrant centers"
    if layouts is not None and layout is not None:
        raise ValueError("pass either layout (single, shared) or layouts (4), not both")
    if layouts is None:
        single = BB0_TEMPLATE["layout"] if layout is None else layout
        layouts = (single, single, single, single)
    if len(layouts) != 4:
        raise ValueError(f"layouts must have exactly 4 entries, got {len(layouts)}")
    for lay in layouts:
        validate_bb0_layout(lay)
    w = World(H, W, K, device)
    centers = [(H // 4, W // 4), (H // 4, 3 * W // 4),
               (3 * H // 4, W // 4), (3 * H // 4, 3 * W // 4)]
    for fac, (cy, cx) in enumerate(centers):
        bb0_f = table.get_or_mint(layouts[fac])
        w.strain_id[cy, cx, 0] = bb0_f
        w.count[cy, cx, 0] = fill_per_cell
        w.faction[cy, cx, 0] = fac
    return w
