from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from envs.domains import BuildCfg, DomainEnv, Rows, main


@dataclass(frozen=True)
class ParseCfg:
    parse_data: Path


def grade(text: str, gold: str) -> bool:
    return " ".join(text.strip().split("\n")[0].split()) == gold


def make(cfg: ParseCfg) -> DomainEnv:
    return DomainEnv("parse", cfg.parse_data, grade)


def build(cfg: ParseCfg, b: BuildCfg, r: random.Random) -> tuple[dict[str, Rows], str]:
    from datasets import load_dataset
    ds = load_dataset("WillHeld/top_v2")

    def items(split):
        rows = [{"id": f"parse-{split}-{i}", "prompt": f"Utterance: {x['utterance']}\nParse:",
                 "target": " " + " ".join(x["semantic_parse"].split()), "gold": " ".join(x["semantic_parse"].split())}
                for i, x in enumerate(ds[split]) if len(x["semantic_parse"]) < 400]
        r.shuffle(rows)
        return rows
    return ({"train": items("train")[:b.n_train], "dev": items("eval")[:b.n_dev], "test": items("test")[:b.n_test]},
            "WillHeld/top_v2 (train / eval / test)")


if __name__ == "__main__":
    main(ParseCfg, make, build)
