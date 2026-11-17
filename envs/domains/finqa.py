from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from envs.domains import BuildCfg, DomainEnv, RawCfg, Rows, main

_FQ_OPS = ("add", "subtract", "multiply", "divide", "exp", "greater", "table_max", "table_min", "table_sum",
           "table_average")
_FQ_HEAD = re.compile(r"\s*,?\s*(" + "|".join(_FQ_OPS) + r")\(")


@dataclass(frozen=True)
class FinqaCfg:
    finqa_data: Path


def _fq_num(t):
    t = t.replace(",", "").strip()
    try:
        return float(t)
    except ValueError:
        pass
    try:
        if "%" in t:
            return float(t.replace("%", "")) / 100.0
        if "const_" in t:
            t = t.replace("const_", "")
            return -1.0 if t == "m1" else float(t)
    except ValueError:
        pass
    return None


def finqa_exec(program, table):
    res, pos, out = [], 0, None
    while pos < len(program):
        m = _FQ_HEAD.match(program, pos)
        if not m:
            return None if program[pos:].strip() else out
        op = m.group(1)
        end = program.find(", none)", m.end()) if op.startswith("table_") else program.find(")", m.end())
        if end < 0:
            return None
        body = program[m.end():end]
        pos = end + (len(", none)") if op.startswith("table_") else 1)
        try:
            if op.startswith("table_"):
                row = next((r for r in table if r[0] == body.strip()), None)
                if row is None:
                    return None
                nums = [_fq_num(c.replace("$", "").strip().split("(")[0]) for c in row[1:]]
                if not nums or any(n is None for n in nums):
                    return None
                out = {"table_max": max, "table_min": min, "table_sum": sum,
                       "table_average": lambda v: sum(v) / len(v)}[op](nums)
            else:
                args = [a.strip() for a in body.split(",")]
                if len(args) != 2:
                    return None
                a, b = (res[int(x[1:])] if x.startswith("#") else _fq_num(x) for x in args)
                if a is None or b is None or isinstance(a, str) or isinstance(b, str):
                    return None
                out = (a + b if op == "add" else a - b if op == "subtract" else a * b if op == "multiply" else
                       a / b if op == "divide" else a ** b if op == "exp" else ("yes" if a > b else "no"))
        except (ZeroDivisionError, OverflowError, IndexError, ValueError, TypeError):
            return None
        if isinstance(out, complex):  # a negative base to a fractional power: no real answer
            return None
        res.append(out)
    return out if isinstance(out, str) or out is None else round(out, 5)


def _fq_same(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return abs(a - b) <= 1e-4 * max(1.0, abs(b))


def grade(text: str, gold: dict[str, Any]) -> bool:
    line = text.strip().split("\n")[0]
    got = finqa_exec(line, gold["table"])
    return got is not None and _fq_same(got, gold["exe"])


def make(cfg: FinqaCfg) -> DomainEnv:
    return DomainEnv("finqa", cfg.finqa_data, grade)


def build(cfg: FinqaCfg, b: BuildCfg, r: random.Random, raw: RawCfg) -> tuple[dict[str, Rows], str]:
    def items(split):
        out, bad = [], 0
        with open(raw.raw / "finqa" / f"{split}.json") as f:
            rows = json.load(f)
        for x in rows:
            q, prog, exe = x["qa"]["question"], x["qa"]["program"], x["qa"]["exe_ans"]
            if isinstance(exe, str) and exe not in ("yes", "no"):
                try:
                    exe = float(exe)
                except ValueError:
                    bad += 1
                    continue
            got = finqa_exec(prog, x["table"])
            if got is None or not _fq_same(got, exe):
                bad += 1
                continue
            ctx = (" ".join(x["pre_text"]) + "\n\n" + "\n".join(" | ".join(row) for row in x["table"]) + "\n\n"
                   + " ".join(x["post_text"]))
            out.append({"id": f"finqa-{x['id']}", "prompt": (
                f"{ctx}\n\nQuestion: {q}\nWrite the calculation as a program. Operations: add, subtract, multiply, "
                "divide, exp, greater, table_sum, table_average, table_max, table_min; #n is the result of step n.\n"
                "Program:"), "target": " " + prog, "gold": {"exe": exe, "table": x["table"]}})
        r.shuffle(out)
        return out, bad
    (tr, b1), (dv, b2), (te, b3) = items("train"), items("dev"), items("test")
    return ({"train": tr[:b.n_train], "dev": dv[:b.n_dev], "test": te[:b.n_test]},
            f"czyssrs/FinQA train/dev/test; gold programs our executor could not reproduce were dropped: {b1}/{b2}/{b3}")


if __name__ == "__main__":
    main(FinqaCfg, make, build, RawCfg)
