from __future__ import annotations

import json
import zlib
import random
from dataclasses import dataclass
from pathlib import Path

from envs.domains import BuildCfg, DomainEnv, RawCfg, Rows, main

_CHEM = {"name_conversion-i2s": "Convert the IUPAC name of the molecule to its SMILES string.\nIUPAC name: {x}\nSMILES:",
         "forward_synthesis": ("Predict the product of the reaction from its reactants and reagents, written as SMILES.\n"
                               "Reactants and reagents: {x}\nProduct:"),
         "retrosynthesis": ("Propose the reactants that produce the given product, written as SMILES and separated by "
                            "a period.\nProduct: {x}\nReactants:")}


@dataclass(frozen=True)
class ChemCfg:
    chem_data: Path


def canon(smiles: str) -> str | None:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    try:
        m = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(m) if m is not None else None
    except Exception:
        return None


def grade(text: str, gold: str) -> bool:
    t = text.strip().split()
    return bool(t) and canon(t[0]) == gold


def make(cfg: ChemCfg) -> DomainEnv:
    return DomainEnv("chem", cfg.chem_data, grade)


def smol_file(raw: RawCfg, split: str, task: str) -> Path:
    return raw.raw / "smol" / "data" / "raw" / split / f"{task}.jsonl"


def build(cfg: ChemCfg, b: BuildCfg, r: random.Random, raw: RawCfg) -> tuple[dict[str, Rows], str]:
    def sample(split, task, k):
        keep, n = [], 0
        with open(smol_file(raw, split, task)) as f:
            for line in f:
                n += 1
                if len(keep) < k:
                    keep.append(line)
                else:
                    j = r.randrange(n)
                    if j < k:
                        keep[j] = line
        out = []
        for line in keep:
            x = json.loads(line)
            c = canon(x["output"])
            if c is None or len(x["input"]) > 600:
                continue
            out.append({"id": f"chem-{task}-{split}-{len(out)}-{zlib.crc32(x['input'].encode()) % 10**8}",
                        "prompt": _CHEM[task].format(x=x["input"]), "target": " " + x["output"], "gold": c})
        r.shuffle(out)
        return out
    tasks = list(_CHEM)
    per = lambda n: -(-n // len(tasks))
    tr, dv, te = [], [], []
    for t in tasks:
        pool = sample("train", t, 3 * (per(b.n_train) + per(b.n_dev)))
        tr += pool[:per(b.n_train)]
        dv += pool[per(b.n_train):per(b.n_train) + per(b.n_dev)]
        te += sample("test", t, 3 * per(b.n_test))[:per(b.n_test)]
    for rows in (tr, dv, te):
        r.shuffle(rows)
    return ({"train": tr[:b.n_train], "dev": dv[:b.n_dev], "test": te[:b.n_test]},
            "osunlp/SMolInstruct raw train / test: " + ", ".join(tasks))


if __name__ == "__main__":
    main(ChemCfg, make, build, RawCfg)
