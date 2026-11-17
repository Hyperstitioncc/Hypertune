from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from envs.domains import BuildCfg, DomainEnv, RawCfg, Rows, main
from envs.domains.chem import smol_file


@dataclass(frozen=True)
class FormulaCfg:
    formula_data: Path


_FORMULA = re.compile(r"((?:[A-Z][a-z]?\d*)+)([+-]\d*)?")


def composition(f: str) -> tuple[dict[str, int], int] | None:
    """Element counts and net charge of a formula such as C20H36N3O3+3 (None if it is not one). Any element order
    (Hill or not) and repeated elements (C2H5OH) count the same atoms; zero counts are kept, so C1H0 is not C."""
    m = _FORMULA.fullmatch(f)
    if not m:
        return None
    counts: dict[str, int] = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", m.group(1)):
        counts[el] = counts.get(el, 0) + int(n or 1)
    q = m.group(2) or ""
    return counts, ((1 if q[0] == "+" else -1) * int(q[1:] or 1) if q else 0)


def grade(text: str, gold: str) -> bool:
    t = text.strip().split()
    if not t:
        return False
    got = t[0].rstrip(".")
    if got == gold:
        return True
    c = composition(got)
    return c is not None and c == composition(gold)


def make(cfg: FormulaCfg) -> DomainEnv:
    return DomainEnv("formula", cfg.formula_data, grade)


def build(cfg: FormulaCfg, b: BuildCfg, r: random.Random, raw: RawCfg) -> tuple[dict[str, Rows], str]:
    def rows(split, k):
        keep, n = [], 0
        with open(smol_file(raw, split, "name_conversion-s2f")) as f:
            for line in f:
                n += 1
                if len(keep) < k:
                    keep.append(line)
                elif r.randrange(n) < k:
                    keep[r.randrange(k)] = line
        out = []
        for i, x in enumerate(map(json.loads, keep)):
            if len(x["input"]) <= 300:
                out.append({"id": f"formula-{split}-{i}", "prompt": ("Give the molecular formula of the molecule.\n"
                            f"SMILES: {x['input']}\nFormula:"), "target": " " + x["output"].strip(), "gold": x["output"].strip()})
        r.shuffle(out)
        return out
    pool, test = rows("train", 2 * (b.n_train + b.n_dev)), rows("test", 2 * b.n_test)
    return ({"train": pool[:b.n_train], "dev": pool[b.n_train:b.n_train + b.n_dev], "test": test[:b.n_test]},
            "osunlp/SMolInstruct name_conversion-s2f (train / test)")


if __name__ == "__main__":
    main(FormulaCfg, make, build, RawCfg)
