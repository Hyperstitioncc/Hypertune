from __future__ import annotations

import contextlib
import enum
from dataclasses import dataclass

import torch


class AdapterKind(enum.Enum):
    identity = "identity"
    lora = "lora"
    kv_prefix = "kv_prefix"
    full = "full"


@dataclass
class Adapter:
    kind: AdapterKind
    module: torch.nn.Module

    def params(self) -> list[torch.Tensor]:
        return [p for p in self.module.parameters() if p.requires_grad]

    def state_dict(self) -> dict:
        return {"kind": self.kind.value, "state": self.module.state_dict()}

    def load(self, sd: dict) -> None:
        if AdapterKind(sd["kind"]) is not self.kind:
            raise ValueError(f"checkpoint holds a {sd['kind']} adapter, this run builds {self.kind.value}")
        self.module.load_state_dict(sd["state"])


@contextlib.contextmanager
def partners(adapter: Adapter, step: int | None):
    mod = adapter.module
    if not getattr(mod, "partnered", False):
        yield
        return
    was, mod.present = mod.present, step
    try:
        yield
    finally:
        mod.present = was
