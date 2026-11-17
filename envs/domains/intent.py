from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from envs.domains import BuildCfg, DomainEnv, Rows, main


@dataclass(frozen=True)
class IntentCfg:
    intent_data: Path


def grade(text: str, gold: str) -> bool:
    # trailing sentence punctuation is not part of a label: banking77 names one label "reverted_card_payment?", and no
    # two labels differ only in a trailing "?" or "."
    t = text.strip().split("\n")[0].strip().lower()
    return t.rstrip("?.") == gold.lower().rstrip("?.")


def make(cfg: IntentCfg) -> DomainEnv:
    return DomainEnv("intent", cfg.intent_data, grade)


def build(cfg: IntentCfg, b: BuildCfg, r: random.Random) -> tuple[dict[str, Rows], str]:
    from datasets import load_dataset
    ds = load_dataset("mteb/banking77")
    item = lambda x, i, sp: {"id": f"intent-{sp}-{i}", "prompt": f"Customer message: {x['text']}\nIntent:",
                             "target": " " + x["label_text"], "gold": x["label_text"]}
    pool = [item(x, i, "train") for i, x in enumerate(ds["train"])]
    test = [item(x, i, "test") for i, x in enumerate(ds["test"])]
    r.shuffle(pool); r.shuffle(test)
    return ({"train": pool[:b.n_train], "dev": pool[b.n_train:b.n_train + b.n_dev], "test": test[:b.n_test]},
            "mteb/banking77 (train / test)")


if __name__ == "__main__":
    main(IntentCfg, make, build)
