from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Problem:
    id: str
    prompt: str
    truth: Any


class Env(Protocol):
    name: str
    stops: tuple[str, ...]
    splits: tuple[str, ...]

    def problems(self, split: str) -> list[Problem]: ...

    def grade(self, pairs: list[tuple[Problem, str]]) -> list[float]: ...
