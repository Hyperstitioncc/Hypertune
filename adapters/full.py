from __future__ import annotations

import torch.nn as nn


class Full(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(True)
