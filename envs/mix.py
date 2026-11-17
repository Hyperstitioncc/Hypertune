from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from envs.base import Env, Problem
from envs.splits import check as check_split


@dataclass(frozen=True)
class MixCfg:
    mix_envs: tuple[str, ...]
    mix_splits: tuple[str, ...]


@dataclass(frozen=True)
class MixEnv:
    parts: tuple[Env, ...]
    part_splits: tuple[str, ...]
    splits = ("all",)

    @property
    def name(self) -> str:
        return "+".join(e.name for e in self.parts)

    @property
    def stops(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s for e in self.parts for s in e.stops))

    def _owner(self, p: Problem) -> int:
        k, _, _ = p.id.partition("/")
        return [e.name for e in self.parts].index(k)

    def problems(self, split: str) -> list[Problem]:
        check_split(self.splits, split)
        return [Problem(f"{e.name}/{p.id}", p.prompt, p.truth)
                for e, s in zip(self.parts, self.part_splits) for p in e.problems(s)]

    def grade(self, pairs: list[tuple[Problem, str]]) -> list[float]:
        out = [0.0] * len(pairs)
        for k, e in enumerate(self.parts):
            idx = [j for j, (p, _) in enumerate(pairs) if self._owner(p) == k]
            got = e.grade([(Problem(pairs[j][0].id.partition("/")[2], pairs[j][0].prompt, pairs[j][0].truth),
                            pairs[j][1]) for j in idx])
            for j, v in zip(idx, got):
                out[j] = v
        return out


def make(cfg: MixCfg, cfgs: Sequence) -> MixEnv:
    from envs import ENVS
    if len(cfg.mix_splits) != len(cfg.mix_envs):
        raise ValueError(f"--mix-splits needs one split per --mix-envs part, got {cfg.mix_splits} for {cfg.mix_envs}")
    return MixEnv(tuple(ENVS[n][1](c) for n, c in zip(cfg.mix_envs, cfgs)), cfg.mix_splits)
