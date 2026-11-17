from __future__ import annotations

import enum
import functools
import json
import random
import re
import select
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from envs.base import Problem
from envs.mathgrade import cut_at_first_boxed, final_boxed, grade_str, last_boxed, strip_lhs
from envs.splits import check as check_split


class Shots(enum.Enum):
    fewshot = "fewshot"
    zeroshot = "zeroshot"


class MathLevel(enum.Enum):
    level1 = "Level 1"
    level2 = "Level 2"
    level3 = "Level 3"
    level4 = "Level 4"
    level5 = "Level 5"


class MathType(enum.Enum):
    algebra = "Algebra"
    counting = "Counting & Probability"
    geometry = "Geometry"
    intermediate_algebra = "Intermediate Algebra"
    number_theory = "Number Theory"
    prealgebra = "Prealgebra"
    precalculus = "Precalculus"


STOPS = ("\nProblem:", "\n\nProblem:")
HUB = "TianHongZXY/MATH"

DIAGRAM_RE = re.compile(r"\[asy\]|asymptote|figure|diagram|as shown|shown below|"
                        r"graph below|picture|image|\.png|\.jpg", re.IGNORECASE)
MC_RE = re.compile(r"\\textbf\{\(?[A-E]\)?\}|\\mathrm\{\(?[A-E]\)?\}|\(\s*[A-E]\s*\)\s*\\?")

MATH_SHOTS = [
    ("What is $1+2+3+\\cdots+10$?",
     "The sum of the first $n$ positive integers is $\\frac{n(n+1)}{2}$. For "
     "$n=10$ this is $\\frac{10\\cdot 11}{2}=55$. The final answer is $\\boxed{55}$."),
    ("If $x^2 = 49$ and $x > 0$, what is $x$?",
     "Taking the positive square root of both sides gives $x=7$. The final "
     "answer is $\\boxed{7}$."),
    ("Simplify $\\frac{6}{8}$.",
     "Dividing numerator and denominator by 2 gives $\\frac{3}{4}$. The final "
     "answer is $\\boxed{\\frac{3}{4}}$."),
    ("What is the greatest common divisor of 12 and 18?",
     "$12 = 2^2\\cdot 3$ and $18 = 2\\cdot 3^2$, so the GCD is $2\\cdot 3=6$. "
     "The final answer is $\\boxed{6}$."),
]
PREFIX = "".join(f"Problem: {p}\nSolution: {s}\n\n" for p, s in MATH_SHOTS)
ZS_INSTR = "Please reason step by step, and put your final answer within \\boxed{}."


def render(problem: str, mode: Shots) -> str:
    if mode is Shots.zeroshot:
        return f"Problem: {problem}\n{ZS_INSTR}\nSolution:"
    return PREFIX + f"Problem: {problem}\nSolution:"


SYMPY_TIMEOUT = 2.0
SYMPY_MAX_PRED = 200
_SYMPY_WORKER_SRC = """
import json, sys
sys.path[:0] = json.loads(sys.argv[1])
from envs.mathgrade import grade_sympy
for line in sys.stdin:
    p, t = json.loads(line)
    try:
        r = grade_sympy(p, t)
    except Exception:
        r = None
    sys.stdout.write(json.dumps(r) + "\\n")
    sys.stdout.flush()
"""


class SympyTimeout(Exception):
    pass


class SympyWorkerDied(Exception):
    pass


class _SympyWorker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.p: subprocess.Popen | None = None
        self.deaths = 0

    def _stop(self) -> None:
        p, self.p = self.p, None
        p.kill()
        p.wait()
        for f in (p.stdin, p.stdout):
            try:
                f.close()
            except OSError:
                pass

    def _ask(self, pred: str, truth: str, seconds: float) -> bool | None:
        try:
            self.p.stdin.write(json.dumps([pred, truth]) + "\n")
            self.p.stdin.flush()
        except OSError:
            self._stop()
            raise SympyWorkerDied
        if not select.select([self.p.stdout], [], [], seconds)[0]:
            self._stop()
            raise SympyTimeout
        line = self.p.stdout.readline()
        if not line:
            self._stop()
            raise SympyWorkerDied
        return json.loads(line)

    def _start(self) -> None:
        path = [str(Path(__file__).resolve().parent.parent), *sys.path]
        self.p = subprocess.Popen([sys.executable, "-u", "-c", _SYMPY_WORKER_SRC, json.dumps(path)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True)
        try:
            self._ask("1", "1", 60.0)
        except (SympyTimeout, SympyWorkerDied):
            raise RuntimeError("sympy worker failed to start")

    def __call__(self, pred: str, truth: str) -> bool | None:
        with self.lock:
            for _ in range(2):
                if self.p is None:
                    self._start()
                try:
                    return self._ask(pred, truth, SYMPY_TIMEOUT)
                except SympyWorkerDied:
                    self.deaths += 1
                    if self.deaths == 1:
                        print("sympy worker died (OOM?): restarting it; an item that kills it twice grades wrong",
                              file=sys.stderr, flush=True)
            raise SympyWorkerDied


_SYMPY = _SympyWorker()


@functools.lru_cache(maxsize=200_000)
def _sympy_verdict(pred: str, truth: str) -> bool | None:
    return _SYMPY(pred, truth)


def _equal(pred: str, truth: str) -> bool:
    if grade_str(pred, truth):
        return True
    if len(pred) > SYMPY_MAX_PRED:
        return False
    try:
        return _sympy_verdict(pred, truth) is True
    except (SympyTimeout, SympyWorkerDied):
        return False


def math_equal(pred: str, truth: str) -> bool:
    if _equal(pred, truth):
        return True
    if ("=" in pred) != ("=" in truth):  # "x = 5" against 5 (or the reverse): compare the right-hand side
        rhs = strip_lhs(pred if "=" in pred else truth)
        if rhs is not None:
            return _equal(rhs, truth) if "=" in pred else _equal(pred, rhs)
    return False


def grade_boxed(text: str, truth: str) -> bool:
    pred = final_boxed(text)
    return pred is not None and math_equal(pred, truth)


class AnswerPick(enum.Enum):
    last_boxed = "last_boxed"
    first_boxed = "first_boxed"


@dataclass(frozen=True)
class MathCfg:
    math_levels: tuple[MathLevel, ...]
    math_types: tuple[MathType, ...] = (MathType.number_theory,)
    math_prompt: Shots = Shots.fewshot
    math_cap: int = 320
    math_answer: AnswerPick = AnswerPick.last_boxed


@dataclass(frozen=True)
class MathEnv:
    cfg: MathCfg
    name = "math"
    stops = STOPS
    splits = ("train", "test")

    def problems(self, split: str) -> list[Problem]:
        check_split(self.splits, split)
        from datasets import load_dataset
        c = self.cfg
        levels = {lv.value for lv in c.math_levels}
        types = {t.value for t in c.math_types}
        pool, seen = [], set()
        for i, ex in enumerate(load_dataset(HUB)[split]):
            prob = ex["problem"]
            if ex["type"] not in types or ex["level"] not in levels:
                continue
            if prob in seen or len(prob) > 1200 or DIAGRAM_RE.search(prob):
                continue
            gt = last_boxed(ex["solution"])
            if gt is None or MC_RE.search(prob):
                continue
            seen.add(prob)
            pool.append(Problem(f"{split}-{i}", render(prob, c.math_prompt), gt))
        random.Random(0).shuffle(pool)
        return pool[:c.math_cap]

    def grade(self, pairs: list[tuple[Problem, str]]) -> list[float]:
        return [float(self._grade(t, p.truth)) for p, t in pairs]

    def _grade(self, text: str, truth: str) -> bool:
        if self.cfg.math_answer is AnswerPick.first_boxed:
            cut = cut_at_first_boxed(text)
            return cut is not None and grade_boxed(cut, truth)
        return grade_boxed(text, truth)


def make(cfg: MathCfg) -> MathEnv:
    if not cfg.math_levels or not cfg.math_types:
        raise ValueError("--math-levels and --math-types each need at least one value")
    return MathEnv(cfg)
