from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from envs.domains import BuildCfg, DomainEnv, Rows, main
from envs.domains.bf import FAMILIES, gen, grade


@dataclass(frozen=True)
class Bf2Cfg:
    bf2_data: Path


def make(cfg: Bf2Cfg) -> DomainEnv:
    return DomainEnv("bf2", cfg.bf2_data, grade)


def build(cfg: Bf2Cfg, b: BuildCfg, r: random.Random) -> tuple[dict[str, Rows], str]:
    items = gen(b.n_train + b.n_dev + b.n_test, seed=r.randrange(10**6), bench=True)
    rows = [{"id": f"bf2-{i}", "prompt": it["prompt"], "target": " " + it["target"], "gold": it["tests"],
             "family": it["family"]} for i, it in enumerate(items)]
    r.shuffle(rows)
    return ({"train": rows[:b.n_train], "dev": rows[b.n_train:b.n_train + b.n_dev], "test": rows[b.n_train + b.n_dev:]},
            f"bf_tasks.gen over all {len(FAMILIES)} families (verified gold programs)")


if __name__ == "__main__":
    main(Bf2Cfg, make, build)
