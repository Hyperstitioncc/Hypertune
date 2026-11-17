from __future__ import annotations

import difflib
import glob
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from envs.base import Problem
from envs.math import DIAGRAM_RE, STOPS, Shots, grade_boxed, render
from envs.splits import check as check_split
from shared.config import parse


@dataclass(frozen=True)
class ContestCfg:
    contest_data: Path
    contest_prompt: Shots = Shots.fewshot
    contest_cap: int = 320


@dataclass(frozen=True)
class ContestEnv:
    cfg: ContestCfg
    name = "contest"
    stops = STOPS
    splits = ("all",)

    def problems(self, split: str) -> list[Problem]:
        check_split(self.splits, split)
        c = self.cfg
        pool = []
        with open(c.contest_data) as f:
            for i, line in enumerate(f):
                if line.strip():
                    r = json.loads(line)
                    pool.append(Problem(f"contest-{i}", render(r["problem"], c.contest_prompt), r["answer"]))
        random.Random(0).shuffle(pool)
        return pool[:c.contest_cap]

    def grade(self, pairs: list[tuple[Problem, str]]) -> list[float]:
        return [float(grade_boxed(t, p.truth)) for p, t in pairs]


def make(cfg: ContestCfg) -> ContestEnv:
    return ContestEnv(cfg)


CUTOFF = "2025-04-29"
HF = [("MathArena/aime_2026", "AIME 2026", "2026-02-04", "problem", "answer"),
      ("MathArena/hmmt_feb_2026", "HMMT Feb 2026", "2026-02-14", "problem", "answer"),
      ("MathArena/hmmt_nov_2025", "HMMT Nov 2025", "2025-11-08", "problem", "answer"),
      ("sonthenguyen/amc12-2025-non-figure", "AMC 12 2025", "2025-11-05", "question", "answer")]


def key(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())[:120]


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower())


def near_dup(t: str, kept: list[str]) -> bool:
    a = norm(t)
    for b in kept:
        if abs(len(a) - len(b)) < 0.3 * max(len(a), len(b)):
            m = difflib.SequenceMatcher(None, a, b)
            if m.quick_ratio() > 0.85 and m.ratio() > 0.85:
                return True
    return False


@dataclass(frozen=True)
class ContestBuildCfg:
    contest_src: Path


def build(cfg: ContestCfg, b: ContestBuildCfg) -> None:
    from datasets import load_dataset
    rows, seen, drop, normed = [], set(), {}, []

    def add(r):
        why = (None if r["date"] > CUTOFF else "date") or \
              (None if r["answer"].strip() and r["answer"].strip() not in ("None",) else "no answer") or \
              (None if len(r["problem"]) <= 1200 else "long") or \
              (None if not DIAGRAM_RE.search(r["problem"]) and not r.get("has_figure") else "figure") or \
              (None if key(r["problem"]) not in seen and not near_dup(r["problem"], normed)
               else "duplicate")
        if why:
            drop[why] = drop.get(why, 0) + 1
            return
        seen.add(key(r["problem"]))
        normed.append(norm(r["problem"]))
        rows.append(r)
    for name, label, date, pk, ak in HF:
        for ex in load_dataset(name, split="train"):
            num = ex.get("problem_idx") or ex.get("problem_number") or ""
            add({"source": f"{label} #{num}", "family": label.split(" ")[0], "date": date,
                 "problem": ex[pk].strip(), "answer": str(ex[ak]).strip(),
                 "has_figure": str(ex.get("has_figure", "False")) == "True", "url": f"hf:{name}"})
    for f in sorted(glob.glob(f"{b.contest_src}/*/problems.jsonl")):
        with open(f) as fi:
            lines = [line for line in fi if line.strip()]
        for line in lines:
            r = json.loads(line)
            r["family"] = r["source"].split(" ")[0]
            r["problem"], r["answer"] = r["problem"].strip(), str(r["answer"]).strip()
            add(r)
    with open(cfg.contest_data, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    fam = {}
    for r in rows:
        fam[r["family"]] = fam.get(r["family"], 0) + 1
    print(f"kept {len(rows)}  by family {fam}  dropped {drop}")


if __name__ == "__main__":
    cmd, *argv = sys.argv[1:]
    if cmd != "build":
        raise SystemExit(f"unknown command {cmd!r}")
    build(*parse(ContestCfg, ContestBuildCfg, argv=argv))
