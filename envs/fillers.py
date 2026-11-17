from __future__ import annotations

import enum
import json
import random
from dataclasses import dataclass

from envs.base import Problem
from envs.knowledge import grade_em
from envs.math import PREFIX, STOPS
from envs.rg import grade_rg
from envs.splits import check as check_split


class FillerTask(enum.Enum):
    family_relationships = "family_relationships"
    letter_counting = "letter_counting"
    number_sorting = "number_sorting"
    sentence_reordering = "sentence_reordering"
    spell_backward = "spell_backward"
    syllogism = "syllogism"
    time_intervals = "time_intervals"
    word_sorting = "word_sorting"
    nq_open = "nq_open"
    sciq = "sciq"


QA_TASKS = (FillerTask.nq_open, FillerTask.sciq)
HELD = 200


def qa_items(task: FillerTask) -> list[tuple[str, list[str]]]:
    from datasets import load_dataset
    if task is FillerTask.nq_open:
        out = []
        for e in load_dataset("google-research-datasets/nq_open", split="train"):
            q = e["question"].strip()
            out.append((q[0].upper() + q[1:] + ("" if q.endswith("?") else "?"), list(e["answer"])))
        return out
    if task is FillerTask.sciq:
        return [(e["question"].strip(), [e["correct_answer"].strip()])
                for e in load_dataset("allenai/sciq", split="train")]
    raise ValueError(f"{task.value} is not a QA task")


@dataclass(frozen=True)
class FillersCfg:
    fillers_tasks: tuple[FillerTask, ...] = tuple(FillerTask)
    fillers_n: int = 80


def _items(task: FillerTask) -> list[tuple[str, dict]]:
    items = []
    if task in QA_TASKS:
        items = [(q, {"em": a}) for q, a in qa_items(task) if len(q) <= 1500]
    else:
        import reasoning_gym as rg
        seen = set()
        for e in rg.create_dataset(task.value, size=int(4200 * 1.3) + 50, seed=10_000):
            q, a = e["question"].strip(), str(e["answer"]).strip()
            if q in seen or len(q) > 1500 or not a or len(a) > 300:
                continue
            seen.add(q)
            meta = json.loads(json.dumps(e.get("metadata", {}), default=str))
            items.append((q, {"rg": task.value, "answer": a, "metadata": meta}))
    random.Random(0).shuffle(items)
    return items


@dataclass(frozen=True)
class FillersEnv:
    cfg: FillersCfg
    name = "fillers"
    stops = STOPS
    splits = ("all",)

    def problems(self, split: str) -> list[Problem]:
        check_split(self.splits, split)
        pool = []
        for task in self.cfg.fillers_tasks:
            items = _items(task)
            for k in range(HELD, min(len(items), HELD + self.cfg.fillers_n)):
                q, a = items[k]
                pool.append(Problem(f"{task.value}-{k}", PREFIX + f"Problem: {q}\nSolution:", a))
        return pool

    def grade(self, pairs: list[tuple[Problem, str]]) -> list[float]:
        return [float(grade_em(t, p.truth) if "em" in p.truth else grade_rg(t, p.truth)) for p, t in pairs]


def make(cfg: FillersCfg) -> FillersEnv:
    return FillersEnv(cfg)
