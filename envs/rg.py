from __future__ import annotations

import enum
import json
import random
import re
import zlib
from dataclasses import dataclass
from typing import Any, Callable

from envs.base import Problem
from envs.math import STOPS, Shots, math_equal, render
from envs.mathgrade import answer_key, boxed_run
from envs.splits import check as check_split

TIERS = {
    "dice":                 [dict(num_dice=3, max_dice_size=8), dict(num_dice=4, max_dice_size=12),
                             dict(num_dice=5, max_dice_size=20)],
    "lcm":                  [dict(min_numbers=3, max_numbers=3, max_value=100),
                             dict(min_numbers=3, max_numbers=4, max_value=300),
                             dict(min_numbers=4, max_numbers=5, max_value=1000)],
    "simple_equations":     [dict(min_terms=3, max_terms=3), dict(min_terms=4, max_terms=4),
                             dict(min_terms=5, max_terms=5)],
    "number_sequence":      [dict(max_complexity=2), dict(max_complexity=3),
                             dict(max_complexity=3, min_terms=4, max_terms=5)],
    "leg_counting":         [dict(min_animals=4, max_animals=6), dict(min_animals=7, max_animals=10),
                             dict(min_animals=10, max_animals=15, max_instances=30)],
    "count_primes":         [dict(min_n=1, max_n=150), dict(min_n=1, max_n=400),
                             dict(min_n=1, max_n=1000)],
    "calendar_arithmetic":  [dict(offset_upper_bound=30), dict(offset_upper_bound=100),
                             dict(offset_upper_bound=300)],
    "base_conversion":      [dict(max_value=1000), dict(max_value=10000), dict(max_value=100000)],
    "products":             [dict(min_digits=3, max_digits=3), dict(min_digits=4, max_digits=4),
                             dict(min_terms=3, max_terms=3, min_digits=3, max_digits=3)],
    "gsm_symbolic":         [dict(), dict(), dict()],
    "family_relationships": [dict(min_family_size=4, max_family_size=6),
                             dict(min_family_size=6, max_family_size=9),
                             dict(min_family_size=9, max_family_size=12)],
    "zebra_puzzles":        [dict(num_people=3, num_characteristics=3),
                             dict(num_people=4, num_characteristics=4),
                             dict(num_people=5, num_characteristics=5)],
    "knights_knaves":       [dict(n_people=3), dict(n_people=4), dict(n_people=5)],
    "countdown":            [dict(min_numbers=3, max_numbers=3), dict(min_numbers=4, max_numbers=4),
                             dict(min_numbers=5, max_numbers=5)],
}
RgTask = enum.Enum("RgTask", {t: t for t in TIERS})
MANY_ANSWERS = frozenset({"countdown"})


class RgLevel(enum.Enum):
    level1 = "Level 1"
    level2 = "Level 2"
    level3 = "Level 3"


MAX_Q_CHARS = 3000
SCORER_ERRORS = (ValueError, TypeError, KeyError, IndexError, AttributeError, ArithmeticError, SyntaxError)


def _seed(task: str, tier: int, base: int) -> int:
    return (zlib.crc32(f"{task}/{tier}".encode()) + 1000003 * base) % (2 ** 31)


_TEXT = re.compile(r"\\(?:text|textbf|mathrm|mbox)\{([^{}]*)\}")
_FNS: dict[str, Callable] = {}
JOINED = frozenset({"knights_knaves"})  # an answer split over adjacent boxes is one answer; the scorer rejects conflicts


def clean(pred: str) -> str:
    s = pred.strip().strip("$").strip()
    for _ in range(2):
        s = _TEXT.sub(r"\1", s)
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("\\,", "").replace("\\ ", " ")
    s = re.sub(r"\\[;:]|\\q?quad|~", " ", s)
    return s.strip().rstrip(".").strip()


def kk_clean(pred: str) -> str:
    """knights_knaves: "Zoey: knight; Ethan - knave" -> "Zoey knight  Ethan knave" (the scorer splits on spaces)."""
    return re.sub(r"[:;=\-]|\s+", " ", clean(pred.replace("\\\\", " "))).strip()


def grade_rg(text: str, truth: dict[str, Any]) -> bool:
    task = truth["rg"]
    run = boxed_run(text)
    if run is None:
        return False
    if task in JOINED:
        pred = ", ".join(run)
    elif len({answer_key(c) for c in run}) > 1:
        return False
    else:
        pred = run[-1]
    if task not in _FNS:
        import reasoning_gym as rg
        _FNS[task] = rg.get_score_answer_fn(task)
    entry = {"question": "", "answer": truth["answer"], "metadata": truth["metadata"]}
    cands = (pred, clean(pred), kk_clean(pred)) if task == "knights_knaves" else (pred, clean(pred))
    for cand in dict.fromkeys(cands):
        try:
            if _FNS[task](cand, entry) >= 1.0:
                return True
        except SCORER_ERRORS:
            pass
    if task in MANY_ANSWERS:
        return False
    return clean(pred).lower() == truth["answer"].strip().lower() or math_equal(pred, truth["answer"])


@dataclass(frozen=True)
class RgCfg:
    rg_levels: tuple[RgLevel, ...]
    rg_tasks: tuple[RgTask, ...] = tuple(RgTask)
    rg_per: int = 60
    rg_seed: int = 0
    rg_prompt: Shots = Shots.fewshot
    rg_cap: int = 320


@dataclass(frozen=True)
class RgEnv:
    cfg: RgCfg
    name = "rg"
    stops = STOPS
    splits = ("all",)

    def problems(self, split: str) -> list[Problem]:
        check_split(self.splits, split)
        import reasoning_gym as rg
        c = self.cfg
        want = {t.value for t in c.rg_tasks}
        pool, seen = [], set()
        for task, tiers in TIERS.items():
            if task not in want:
                continue
            for t, (level, gen_cfg) in enumerate(zip(RgLevel, tiers)):
                if level not in c.rg_levels:
                    continue
                ds = rg.create_dataset(task, size=c.rg_per, seed=_seed(task, t, c.rg_seed), **gen_cfg)
                for j, e in enumerate(ds):
                    q = e["question"].strip()
                    if q in seen or len(q) > MAX_Q_CHARS:
                        continue
                    seen.add(q)
                    meta = json.loads(json.dumps(e.get("metadata", {}), default=str))
                    pool.append(Problem(f"{task}-{t + 1}-{j}", render(q, c.rg_prompt),
                                        {"rg": task, "answer": str(e["answer"]), "metadata": meta}))
        random.Random(c.rg_seed).shuffle(pool)
        return pool[:c.rg_cap]

    def grade(self, pairs: list[tuple[Problem, str]]) -> list[float]:
        return [float(grade_rg(t, p.truth)) for p, t in pairs]


def make(cfg: RgCfg) -> RgEnv:
    if not cfg.rg_levels or not cfg.rg_tasks:
        raise ValueError("--rg-levels and --rg-tasks each need at least one value")
    return RgEnv(cfg)
