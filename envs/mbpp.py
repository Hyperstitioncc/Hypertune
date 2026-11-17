from __future__ import annotations

import enum
import functools
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from envs.base import Problem
from envs.math import STOPS
from envs.splits import check as check_split

HUB = "google-research-datasets/mbpp"
CODE_INSTR = "Write the solution as a Python function in a single ```python code block."
_CODE_RE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)(?:```|$)", re.DOTALL)
SENTINEL = "__mbpp_all_asserts_ran_7f3c__"
TIMEOUT = 5.0


class MbppPrompt(enum.Enum):
    qwen3moe = "qwen3moe"
    compose = "compose"


def extract_code(text: str) -> str | None:
    m = _CODE_RE.search(text)
    return m.group(1) if m else None


LIMITS = (("RLIMIT_AS", 4 << 30), ("RLIMIT_CPU", 8), ("RLIMIT_FSIZE", 16 << 20))
_LIMITED_EXEC = f"""
import os, resource, sys
for lim, v in {LIMITS!r}:
    try:
        resource.setrlimit(getattr(resource, lim), (v, v))
    except (ValueError, OSError):
        pass
os.execv(sys.executable, [sys.executable, "-I", sys.argv[1]])
"""


@functools.lru_cache(maxsize=200_000)
def passes(code: str, tests: tuple[str, ...], setup: str) -> bool:
    with tempfile.TemporaryDirectory() as td, open(os.path.join(td, "_out"), "w+b") as out_f:
        src = os.path.join(td, "_prog.py")
        with open(src, "w", encoding="utf-8", errors="surrogatepass") as f:
            f.write(f"{setup}\n{code}\n\n" + "\n".join(tests) + f"\nprint({SENTINEL!r}, flush=True)\n")
        p = subprocess.Popen([sys.executable, "-I", "-c", _LIMITED_EXEC, src], cwd=td,
                             stdin=subprocess.DEVNULL, stdout=out_f, stderr=subprocess.DEVNULL,
                             start_new_session=True, env={"PATH": "/usr/bin:/bin"})
        try:
            p.wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.kill()
            p.wait()
            return False
        n = out_f.seek(0, 2)
        out_f.seek(max(0, n - 4096))
        return SENTINEL.encode() in out_f.read()


def _visible_call(test_list: list[str]) -> str:
    t = test_list[0]
    for op in ("==", "!=", " is ", "<=", ">=", "<", ">"):
        i = t.find(op)
        if i > 0:
            t = t[:i]
            break
    return t.replace("assert ", "").strip()


def _compose_task(ex: dict[str, Any]) -> str:
    return (f"You are an expert Python programmer, and here is your task: {ex['text']} "
            f"Your code will be tested with calls like: {_visible_call(ex['test_list'])}\n[BEGIN]\n")


@dataclass(frozen=True)
class MbppCfg:
    mbpp_cap: int = 320
    mbpp_prefill: bool = False
    mbpp_hide_tests: bool = True  # graded tests never go in the prompt unless asked for explicitly
    mbpp_prompt: MbppPrompt = MbppPrompt.qwen3moe


@dataclass(frozen=True)
class MbppEnv:
    cfg: MbppCfg
    name = "mbpp"
    splits = ("all",)

    @property
    def stops(self) -> tuple[str, ...]:
        return ("[DONE]",) if self.cfg.mbpp_prompt is MbppPrompt.compose else STOPS

    def problems(self, split: str) -> list[Problem]:
        check_split(self.splits, split)
        from datasets import load_dataset
        c = self.cfg
        ds = load_dataset(HUB, "full")
        truth = lambda ex: {"code_tests": list(ex["test_list"]), "setup": ex.get("test_setup_code") or ""}
        if c.mbpp_prompt is MbppPrompt.compose:
            prefix = "".join(_compose_task(s) + f"{s['code']}\n[DONE]\n" for s in list(ds["prompt"])[:3])
            pool = list(ds["train"]) + list(ds["test"]) + list(ds["validation"])
            return [Problem(str(ex["task_id"]), prefix + _compose_task(ex), truth(ex))
                    for ex in random.Random(0).sample(pool, min(c.mbpp_cap, len(pool)))]
        pool = []
        for sp in ("train", "validation", "test", "prompt"):
            for ex in ds[sp]:
                if c.mbpp_hide_tests:
                    defs = re.findall(r"^def\s+(\w+)\s*\((.*?)\)\s*(?:->[^:]*)?:", ex["code"], re.M)
                    called = " ".join(ex["test_list"])
                    tgt = next(((n, a) for n, a in defs if re.search(rf"\b{n}\s*\(", called)), None)
                    if tgt is None:
                        continue
                    prob = ex["text"].strip() + f"\nThe function must be: def {tgt[0]}({tgt[1].strip()})"
                else:
                    prob = ex["text"].strip() + "\nYour code should pass these tests:\n" + "\n".join(ex["test_list"])
                pool.append(Problem(str(ex["task_id"]), f"Problem: {prob}\n{CODE_INSTR}\nSolution:"
                                    + ("\n```python\n" if c.mbpp_prefill else ""), truth(ex)))
        random.Random(0).shuffle(pool)
        return pool[:c.mbpp_cap]

    def _code(self, text: str) -> str | None:
        if self.cfg.mbpp_prompt is MbppPrompt.compose:
            return text.split("[DONE]")[0]
        if self.cfg.mbpp_prefill and not text.lstrip().startswith("```"):
            code = text.split("```")[0]
        else:
            code = extract_code(text)
        return code if code and code.strip() else None

    def _grade(self, text: str, truth: dict[str, Any]) -> bool:
        code = self._code(text)
        return code is not None and passes(code, tuple(truth["code_tests"]), truth["setup"])

    def grade(self, pairs: list[tuple[Problem, str]]) -> list[float]:
        # the cores this process may use, not the machine's: in an 8-core job on a 256-core node, 48 programs at
        # once start too slowly to finish inside TIMEOUT when Python sits on a network disk, and correct ones fail
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 8
        w = min(48, max(4, cores - 4))
        with ThreadPoolExecutor(w) as ex:
            return [float(v) for v in ex.map(lambda pt: self._grade(pt[1], pt[0].truth), pairs)]


def make(cfg: MbppCfg) -> MbppEnv:
    if cfg.mbpp_prompt is MbppPrompt.compose and (cfg.mbpp_prefill or cfg.mbpp_hide_tests):
        raise ValueError("--mbpp-prompt compose has its own format, which shows a graded test's call: it needs "
                         "--mbpp-hide-tests false and no --mbpp-prefill")
    return MbppEnv(cfg)
