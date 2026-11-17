from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

from envs.domains import BuildCfg, DomainEnv, Rows, main
from envs.knowledge import MC_INSTR, letter_options


@dataclass(frozen=True)
class MedqaCfg:
    medqa_data: Path


def grade(text: str, gold: str) -> bool:
    m = re.match(r"\s*\(?([A-D])\)?(?![A-Za-z])", text)
    return bool(m) and m.group(1) == gold


def make(cfg: MedqaCfg) -> DomainEnv:
    return DomainEnv("medqa", cfg.medqa_data, grade)


def build(cfg: MedqaCfg, b: BuildCfg, r: random.Random) -> tuple[dict[str, Rows], str]:
    from datasets import load_dataset
    ds = load_dataset("GBaker/MedQA-USMLE-4-options")

    def item(x, i, split):
        opts = letter_options([v for _, v in sorted(x["options"].items())])
        return {"id": f"medqa-{split}-{i}", "prompt": f"Problem: {x['question']}\n{opts}\n{MC_INSTR}\nAnswer:",
                "target": f" {x['answer_idx']}", "gold": x["answer_idx"]}
    pool = [item(x, i, "train") for i, x in enumerate(ds["train"])]
    test = [item(x, i, "test") for i, x in enumerate(ds["test"])]
    r.shuffle(pool); r.shuffle(test)
    return ({"train": pool[:b.n_train], "dev": pool[b.n_train:b.n_train + b.n_dev], "test": test[:b.n_test]},
            "GBaker/MedQA-USMLE-4-options (train / test)")


if __name__ == "__main__":
    main(MedqaCfg, make, build)
