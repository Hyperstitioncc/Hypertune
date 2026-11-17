from __future__ import annotations

import torch
import torch.nn as nn


class Rung(nn.Module):
    def __init__(self, soft: torch.Tensor, rung: int, train_vec: bool, L: int, d: int, kv_dim: int):
        super().__init__()
        self.rung, self.K = rung, soft.shape[0]
        dev, K = soft.device, soft.shape[0]
        s = soft.detach().float()
        if rung == 0 or train_vec:
            self.soft = nn.Parameter(s)
        else:
            self.register_buffer("soft", s)
        if rung >= 1:
            self.m = nn.Parameter(torch.ones(L, K, device=dev))
        if rung >= 2:
            self.bias = nn.Parameter(torch.zeros(L, K, device=dev))
        if rung >= 3:
            self.gK = nn.Parameter(torch.ones(L, d, device=dev))
            self.gV = nn.Parameter(torch.ones(L, d, device=dev))
        if rung >= 4:
            self.dK = nn.Parameter(torch.zeros(L, K, kv_dim, device=dev))
            self.dV = nn.Parameter(torch.zeros(L, K, kv_dim, device=dev))

    def rows(self) -> torch.Tensor:
        return self.soft
