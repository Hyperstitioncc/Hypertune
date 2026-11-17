from __future__ import annotations

import torch
import torch.nn as nn


class SoftBlock(nn.Module):
    def __init__(self, soft: torch.Tensor, frozen: torch.Tensor, row_tail: bool, pos0: bool):
        super().__init__()
        self.soft = nn.Parameter(soft.float())
        self.register_buffer("frozen", frozen.float())
        self.row_tail = row_tail
        self.pos0 = pos0

    def rows(self) -> torch.Tensor:
        return torch.cat([self.frozen, self.soft], 0)


def random_rows(k: int, d: int, estd: float, scale: float, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(k, d, device=device, generator=g) * estd * scale
