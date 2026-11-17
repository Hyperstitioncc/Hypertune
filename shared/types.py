from __future__ import annotations

import enum
from dataclasses import dataclass


class Backend(enum.Enum):
    vllm = "vllm"
    hf_eager = "hf_eager"


@dataclass(frozen=True)
class Sampling:
    n: int
    temperature: float
    top_p: float
    max_tokens: int
    seed: int


@dataclass(frozen=True)
class DecodeCfg:
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 640

    def at(self, n: int, seed: int) -> Sampling:
        return Sampling(n=n, temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens, seed=seed)


@dataclass(frozen=True)
class PartialMean:
    total: float
    n: float

    @property
    def mean(self) -> float:
        return self.total / self.n


@dataclass(frozen=True)
class TokenRow:
    prompt_ids: tuple[int, ...]
    ids: tuple[int, ...]


@dataclass(frozen=True)
class Completion(TokenRow):
    text: str
    engine_lp: tuple[float, ...]
