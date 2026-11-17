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
        try:  # a Python literal can hold what JSON cannot (a set, mixed key types): unparsable, not a crash
            out.append((str(c["name"]), json.dumps(c.get("arguments", {}), sort_keys=True)))
        except (TypeError, ValueError):
            return None
    return sorted(out)


def canon(v):
    """An argument value up to equivalent spellings: a number by its value (5 == 5.0; bools stay bools), a string that
    holds a JSON object or array by its parsed content (the dataset passes documents as JSON strings, and their
    whitespace and key order mean nothing), and a key set to null the same as a key left out (an optional argument
    not given). Strings are otherwise exact, list order counts, and a string never equals a non-string."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return ("num", float(v))
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in ("{", "[") and s[-1:] in ("}", "]"):
            try:
                return ("json", canon(json.loads(s)))
            except ValueError:
                pass
        return v
    if isinstance(v, dict):
        return ("obj", tuple(sorted((str(k), canon(x)) for k, x in v.items() if x is not None)))
    if isinstance(v, list):
        return ("arr", tuple(canon(x) for x in v))
    return ("other", repr(v))


def _key(name: str, args: str) -> tuple:
    return name, canon(json.loads(args))


def grade(text: str, gold: list) -> bool:
    got = calls(text)
    if got is None:
        return False
    if [list(c) for c in got] == [list(c) for c in gold]:
        return True
    try:
        return sorted(map(repr, (_key(*c) for c in got))) == sorted(map(repr, (_key(*c) for c in gold)))
    except (ValueError, TypeError, RecursionError):
        return False


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
