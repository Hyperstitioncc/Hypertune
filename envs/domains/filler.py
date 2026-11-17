from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

from envs.domains import BuildCfg, DomainEnv, Rows, main
from envs.fillers import QA_TASKS, FillerTask, qa_items

@dataclass(frozen=True)
class FillerCfg:
    filler_data: Path
    filler_task: FillerTask


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


def grade(text: str, gold: list[str]) -> bool:
    line = text.strip().split("\n")[0]
    return _norm(line) in {_norm(g) for g in gold}


def make(cfg: FillerCfg) -> DomainEnv:
    return DomainEnv(f"filler-{cfg.filler_task.value}", cfg.filler_data, grade)


def build(cfg: FillerCfg, b: BuildCfg, r: random.Random) -> tuple[dict[str, Rows], str]:
    task = cfg.filler_task
    n = b.n_train + b.n_dev + b.n_test
    if task in QA_TASKS:
        items = qa_items(task)
    else:
        import reasoning_gym as rg
        items, seen = [], set()
        for e in rg.create_dataset(task.value, size=3 * n + 50, seed=10_000):
            q, a = e["question"].strip(), str(e["answer"]).strip()
            if q not in seen:
                seen.add(q)
                items.append((q, [a]))
    items = [(q, a) for q, a in items if len(q) <= 1500 and a and a[0] and "\n" not in a[0] and len(a[0]) <= 300]
    r.shuffle(items)
    rows = [{"id": f"filler-{task.value}-{i}", "prompt": f"Problem: {q}\nAnswer:", "target": " " + a[0], "gold": a}
            for i, (q, a) in enumerate(items[:n])]
    return ({"train": rows[:b.n_train], "dev": rows[b.n_train:b.n_train + b.n_dev], "test": rows[b.n_train + b.n_dev:]},
            f"filler task {task.value}")


if __name__ == "__main__":
    main(FillerCfg, make, build)
