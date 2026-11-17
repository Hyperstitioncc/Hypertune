from __future__ import annotations

import ast
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from envs.domains import BuildCfg, DomainEnv, Rows, main

_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class FuncallCfg:
    funcall_data: Path


def calls(text: str) -> list[tuple[str, str]] | None:
    out = []
    for blob in _CALL.findall(text):
        try:
            c = json.loads(blob)
        except Exception:
            try:
                c = ast.literal_eval(blob)
            except Exception:
                return None
        if not isinstance(c, dict) or "name" not in c:
            return None
        out.append((str(c["name"]), json.dumps(c.get("arguments", {}), sort_keys=True)))
    return sorted(out)


def grade(text: str, gold: list) -> bool:
    got = calls(text)
    return got is not None and [list(c) for c in got] == [list(c) for c in gold]


def make(cfg: FuncallCfg) -> DomainEnv:
    return DomainEnv("funcall", cfg.funcall_data, grade)


def build(cfg: FuncallCfg, b: BuildCfg, r: random.Random) -> tuple[dict[str, Rows], str]:
    from datasets import load_dataset
    ds = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling_singleturn")["train"]
    pool = []
    for x in ds:
        sysm, user, gpt = (t["value"] for t in x["conversations"])
        gold = calls(gpt)
        if not gold or len(sysm) + len(user) > 9000 or len(gpt) > 1500:
            continue
        pool.append({"id": f"funcall-{x['id']}", "prompt": f"{sysm.strip()}\n\nUser: {user.strip()}\n\nAssistant:",
                     "target": " " + gpt.strip(), "gold": gold})
    r.shuffle(pool)
    n_tr, n_dv = b.n_train, b.n_dev
    n_test = min(b.n_test, len(pool) - n_tr - n_dv)
    if n_test < 100:
        raise ValueError(f"only {len(pool)} usable rows")
    return ({"train": pool[:n_tr], "dev": pool[n_tr:n_tr + n_dv], "test": pool[n_tr + n_dv:n_tr + n_dv + n_test]},
            "NousResearch/hermes-function-calling-v1 func_calling_singleturn (one pool, disjoint splits)")


if __name__ == "__main__":
    main(FuncallCfg, make, build)
