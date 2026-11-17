from __future__ import annotations

import torch
import torch.nn as nn


class Thinker(nn.Module):
    def __init__(self, lm: nn.Module, hdim: int, dec_d: int, k: int, pad: int, estd: float, seed: int,
                 device: torch.device):
        super().__init__()
        self.lm, self.k, self.pad, self.estd = lm, k, pad, estd
        self.device = device
        self.head = nn.Linear(hdim, dec_d, bias=False).to(device)
        nn.init.orthogonal_(self.head.weight, generator=torch.Generator(device).manual_seed(seed))

    def _hiddens(self, p_ids) -> torch.Tensor:
        ids = torch.tensor([list(p_ids) + [self.pad] * self.k], device=self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            return self.lm(input_ids=ids).logits[0, -self.k:]

    def forward(self, p_ids) -> torch.Tensor:
        return self.head(self._hiddens(p_ids).float())

    def calibrate(self, p_ids) -> float:
        with torch.no_grad():
            scale = self.estd / self(p_ids).std().clamp_min(1e-8).item()
            self.head.weight.mul_(scale)
        return scale
