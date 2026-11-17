"""
Knowledge-domain banks (DATASET=knowledge): download once (the job scripts run HF offline) and write one
JSONL per domain, {"problem", "answer": {"em": [aliases]}, "family", "source"}, read by
common.load_knowledge via KNOW_FILE. Grading = common.fast_grade's exact-match branch on the last \\boxed{}.

  law      MMLU-Pro, category "law" (TIGER-Lab/MMLU-Pro test+validation; up to 10 options)
  casehold CaseHOLD (coastalcph/lex_glue "case_hold": a case citation's context, pick the holding; 5 options)
  medqa    MedQA-USMLE, 4 options (GBaker/MedQA-USMLE-4-options, train+test)
  trivia   TriviaQA, no context (mandarjoshi/trivia_qa "rc.nocontext" validation), short answer, alias EM

MC problems list the options as "A. ..." and ask for the letter; the aliases accept the letter alone,
"letter. option text" and the option text (fast_grade strips punctuation and articles before comparing).
    python build_knowledge_set.py OUT_DIR [N_MAX per domain, default 1500]
"""
import json, os, random, sys
from datasets import load_dataset

L = "ABCDEFGHIJ"
MC_TAIL = "\n\nAnswer with the letter of the correct option."
MAX_CHARS = 4000


def mc(q, opts, ans_idx, family, source):
    q = q.strip()
    body = q + "\n\n" + "\n".join(f"{L[i]}. {o.strip()}" for i, o in enumerate(opts)) + MC_TAIL
    o = opts[ans_idx].strip()
    return {"problem": body, "answer": {"em": [L[ans_idx], f"{L[ans_idx]}. {o}", o]},
            "family": family, "source": source}


def law():
    out = []
    for split in ("test", "validation"):
        for ex in load_dataset("TIGER-Lab/MMLU-Pro", split=split):
            if ex["category"] == "law":
                out.append(mc(ex["question"], ex["options"], int(ex["answer_index"]), "law", "mmlu_pro"))
    return out


def casehold():
    out = []
    for split in ("train", "validation", "test"):
        for ex in load_dataset("coastalcph/lex_glue", "case_hold", split=split):
            out.append(mc(ex["context"], ex["endings"], int(ex["label"]), "casehold", "lex_glue"))
    return out


def medqa():
    out = []
    for split in ("train", "test"):
        for ex in load_dataset("GBaker/MedQA-USMLE-4-options", split=split):
            keys = sorted(ex["options"])
            out.append(mc(ex["question"], [ex["options"][k] for k in keys], keys.index(ex["answer_idx"]),
                          "medqa", "medqa_usmle4"))
    return out


def trivia():
    out = []
    for ex in load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation"):
        a = ex["answer"]
        al = list(dict.fromkeys([a["value"]] + list(a["aliases"]) + list(a["normalized_aliases"])))
        out.append({"problem": ex["question"].strip(), "answer": {"em": al}, "family": "trivia",
                    "source": "triviaqa_nocontext"})
    return out


if __name__ == "__main__":
    od = sys.argv[1]
    n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    os.makedirs(od, exist_ok=True)
    for name, fn in (("law", law), ("casehold", casehold), ("medqa", medqa), ("trivia", trivia)):
        try:
            rows = [r for r in fn() if len(r["problem"]) <= MAX_CHARS]
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        seen, uniq = set(), []
        for r in rows:
            if r["problem"] not in seen:
                seen.add(r["problem"]); uniq.append(r)
        random.Random(0).shuffle(uniq)
        uniq = uniq[:n_max]
        with open(f"{od}/{name}.jsonl", "w") as f:
            for r in uniq:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(uniq)} problems (of {len(rows)}) -> {od}/{name}.jsonl\n   e.g. {uniq[0]['problem'][:300]!r}"
              f"\n   answer {uniq[0]['answer']['em'][:3]}", flush=True)
