from __future__ import annotations

from typing import Protocol

from envs.base import Problem
from shared.adapter import Adapter
from shared.types import PartialMean


class Algorithm(Protocol):
    def step(self, adapter: Adapter, batch: list[Problem], step: int) -> dict[str, float]: ...

    def select(self, adapter: Adapter, step: int) -> PartialMean: ...
