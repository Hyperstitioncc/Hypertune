from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from envs.base import Problem
from envs.splits import check as check_split
from shared import config
from shared.fs import atomic_json

Rows = list[dict[str, Any]]
FILES = ("train", "dev", "test")
CORPUS = "corpus.json"


@dataclass(frozen=True)
class DomainEnv:
    name: str
    data: Path
    check: Callable[[str, Any], bool]
    splits: tuple[str, ...] = (*FILES, "all")
    stops = ()

    def rows(self, split: str) -> Rows:
        check_split(self.splits, split)
        out = []
        for name in FILES if split == "all" else (split,):
            with open(self.data / f"{name}.jsonl") as f:
                out += [json.loads(line) for line in f]
        return out

    def problems(self, split: str) -> list[Problem]:
        return [Problem(r["id"], r["prompt"], r["gold"]) for r in self.rows(split)]

    def write_corpus(self) -> None:
        atomic_json({r["id"]: [r["target"]] for r in self.rows("all")}, self.data / CORPUS)

    def grade(self, pairs: list[tuple[Problem, str]]) -> list[float]:
        return [float(self.check(t, p.truth)) for p, t in pairs]


@dataclass(frozen=True)
class BuildCfg:
    seed: int = 0
    n_train: int = 1000
    n_dev: int = 200
    n_test: int = 500


@dataclass(frozen=True)
class RawCfg:
    raw: Path


def main(cfg_cls: type, make: Callable[[Any], DomainEnv], build: Callable[..., tuple[dict[str, Rows], str]],
         *extra: type) -> None:
    cmd, *argv = sys.argv[1:]
    if cmd != "build":
        raise SystemExit(f"unknown command {cmd!r}")
    cfg, b, *more = config.parse(cfg_cls, BuildCfg, *extra, argv=argv)
    env = make(cfg)
    splits, src = build(cfg, b, random.Random(b.seed), *more)
    stored = {s for s in env.splits if s != "all"}
    if set(splits) != stored:
        raise ValueError(f"build wrote splits {sorted(splits)}, the env stores {sorted(stored)}")
    ids = [x["id"] for rows in splits.values() for x in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("splits overlap")
    keyed = [x for name in ("train", "dev", "test") for x in splits[name]]
    bad = [x["id"] for x in keyed if not env.check(x["target"], x["gold"])]
    if bad:
        raise ValueError(f"{len(bad)} gold targets fail their own grader, e.g. {bad[:3]}")
    wrong = sum(env.check(a["target"], b_["gold"]) for a, b_ in zip(keyed, keyed[1:]) if a["target"] != b_["target"])
    env.data.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(env.data / f"{name}.jsonl", "w") as f:
            for x in rows:
                f.write(json.dumps(x) + "\n")
    env.write_corpus()
    with open(env.data / "meta.json", "w") as f:
        json.dump({"domain": env.name, "source": src, "seed": b.seed, "n": {k: len(v) for k, v in splits.items()},
                   "neighbour_answer_accepted": wrong}, f, indent=1)
    print(f"{env.name}: " + " ".join(f"{k} {len(v)}" for k, v in splits.items()) + f" -> {env.data}\n  source: {src}\n  "
          f"every gold target passes its grader; another problem's (different) answer was accepted {wrong} of "
          f"{len(keyed) - 1} times")
