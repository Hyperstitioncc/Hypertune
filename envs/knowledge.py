from __future__ import annotations

import json
import random
import re
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from envs.base import Problem
from envs.math import STOPS, Shots, render
from envs.mathgrade import final_boxed
from envs.splits import check as check_split
from shared.config import parse


TEXT_CMD = re.compile(r"\\(?:text|textbf|textrm|textit|mathrm|mathbf|mbox|emph)\s*\{([^{}]*)\}")


def em_norm(s: str, hyphen: str = "") -> str:
    s = s.lower().strip().strip("$")
    s = TEXT_CMD.sub(r"\1", s)
    s = s.replace("-", hyphen)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_match(b: str, aliases: list[str]) -> bool:
    letters = {x.strip().upper() for x in aliases if re.fullmatch(r"[A-Za-z]", x.strip())}
    if letters:
        bb = TEXT_CMD.sub(r"\1", b).strip().strip("$").strip()
        m = re.fullmatch(r"\(?([A-Za-z])\)?[.:]?", bb)
        if m:
            return m.group(1).upper() in letters
        # "B. <option text>" whose text is cut short or extended ("C. Disseminated gonococcal infection (DGI)"): the
        # letter decides as long as one text is a word-prefix of the other; a paraphrase or another option's text fails
        m = re.fullmatch(r"\(?([A-Za-z])\)?[.:)]\s+(.+)", bb, re.S)
        if m and m.group(1).upper() in letters:
            t = em_norm(m.group(2))
            for x in aliases:
                o = em_norm(x)
                if x.strip().upper() in letters or not t or not o:
                    continue
                if (t + " ").startswith(o + " ") or (o + " ").startswith(t + " ") and len(t.split()) >= 2:
                    return True
        # "Plasmapheresis (C)", "Enterotoxigenic Escherichia coli (ETEC)": the option text plus its letter or a short tag
        m = re.fullmatch(r"(.+?)\s*\(([^()\s]{1,8})\)", bb, re.S)
        if m and (len(m.group(2)) > 1 or m.group(2).upper() in letters) and em_match(m.group(1), aliases):
            return True
    texts = [x for x in aliases if x.strip().upper() not in letters]
    # punctuation is deleted ("ben-hur" -> "benhur"); a hyphen read as a space is also accepted ("Thyroid-stimulating")
    return any(bool(em_norm(b, h)) and em_norm(b, h) in {em_norm(x, h) for x in texts} for h in ("", " "))


def grade_em(text: str, truth: dict[str, Any]) -> bool:
    b = final_boxed(text)
    return b is not None and em_match(b, truth["em"])


@dataclass(frozen=True)
class KnowledgeCfg:
    knowledge_data: Path
    knowledge_prompt: Shots = Shots.fewshot
    knowledge_cap: int = 320


@dataclass(frozen=True)
class KnowledgeEnv:
    cfg: KnowledgeCfg
    name = "knowledge"
    stops = STOPS
    splits = ("all",)

    def problems(self, split: str) -> list[Problem]:
        check_split(self.splits, split)
        c = self.cfg
        pool = []
        with open(c.knowledge_data) as f:
            for i, line in enumerate(f):
                if line.strip():
                    r = json.loads(line)
                    pool.append(Problem(f"{r['family']}-{i}", render(r["problem"], c.knowledge_prompt), r["answer"]))
        random.Random(0).shuffle(pool)
        return pool[:c.knowledge_cap]

    def grade(self, pairs: list[tuple[Problem, str]]) -> list[float]:
        return [float(grade_em(t, p.truth)) for p, t in pairs]


def make(cfg: KnowledgeCfg) -> KnowledgeEnv:
    return KnowledgeEnv(cfg)


L = "ABCDEFGHIJ"
MC_INSTR = "Answer with the letter of the correct option."
MAX_CHARS = 4000


def letter_options(opts: list[str]) -> str:
    return "\n".join(f"{L[i]}. {o}" for i, o in enumerate(opts))


def mc(q, opts, ans_idx, family, source):
    q = q.strip()
    body = q + "\n\n" + letter_options([o.strip() for o in opts]) + "\n\n" + MC_INSTR
    o = opts[ans_idx].strip()
    return {"problem": body, "answer": {"em": [L[ans_idx], f"{L[ans_idx]}. {o}", o]},
            "family": family, "source": source}


def law():
    from datasets import load_dataset
    out = []
    for split in ("test", "validation"):
        for ex in load_dataset("TIGER-Lab/MMLU-Pro", split=split):
            if ex["category"] == "law":
                out.append(mc(ex["question"], ex["options"], int(ex["answer_index"]), "law", "mmlu_pro"))
    return out


def casehold():
    from datasets import load_dataset
    out = []
    for split in ("train", "validation", "test"):
        for ex in load_dataset("coastalcph/lex_glue", "case_hold", split=split):
            out.append(mc(ex["context"], ex["endings"], int(ex["label"]), "casehold", "lex_glue"))
    return out


def medqa():
    from datasets import load_dataset
    out = []
    for split in ("train", "test"):
        for ex in load_dataset("GBaker/MedQA-USMLE-4-options", split=split):
            keys = sorted(ex["options"])
            out.append(mc(ex["question"], [ex["options"][k] for k in keys], keys.index(ex["answer_idx"]),
                          "medqa", "medqa_usmle4"))
    return out


def trivia():
    from datasets import load_dataset
    out = []
    for ex in load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation"):
        a = ex["answer"]
        al = list(dict.fromkeys([a["value"]] + list(a["aliases"]) + list(a["normalized_aliases"])))
        out.append({"problem": ex["question"].strip(), "answer": {"em": al}, "family": "trivia",
                    "source": "triviaqa_nocontext"})
    return out


@dataclass(frozen=True)
class KnowledgeBuildCfg:
    knowledge_out: Path
    knowledge_n_max: int = 1500


def build(b: KnowledgeBuildCfg) -> None:
    od, n_max = b.knowledge_out, b.knowledge_n_max
    od.mkdir(parents=True, exist_ok=True)
    for name, fn in (("law", law), ("casehold", casehold), ("medqa", medqa), ("trivia", trivia)):
        rows = [r for r in fn() if len(r["problem"]) <= MAX_CHARS]
        seen, uniq = set(), []
        for r in rows:
            if r["problem"] not in seen:
                seen.add(r["problem"]); uniq.append(r)
        random.Random(0).shuffle(uniq)
        uniq = uniq[:n_max]
        with open(od / f"{name}.jsonl", "w") as f:
            for r in uniq:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(uniq)} problems (of {len(rows)}) -> {od}/{name}.jsonl\n   e.g. {uniq[0]['problem'][:300]!r}"
              f"\n   answer {uniq[0]['answer']['em'][:3]}", flush=True)


if __name__ == "__main__":
    cmd, *argv = sys.argv[1:]
    if cmd != "build":
        raise SystemExit(f"unknown command {cmd!r}")
    build(*parse(KnowledgeBuildCfg, argv=argv))
