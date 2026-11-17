from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from envs.domains import BuildCfg, DomainEnv, RawCfg, Rows, main
from envs.domains.chem import smol_file


@dataclass(frozen=True)
class FormulaCfg:
    formula_data: Path


def grade(text: str, gold: str) -> bool:
    t = text.strip().split()
    return bool(t) and t[0].rstrip(".") == gold


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
