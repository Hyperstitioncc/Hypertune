"""Merge post-release contest problems into one JSONL for DATASET=contest.

Uncontaminated by date: Qwen3-30B-A3B-Base's weights were published 2025-04-29, so
no problem first used at a contest held after that date can be in its training
data. Sources:
  - Hugging Face: MathArena/aime_2026, hmmt_feb_2026, hmmt_nov_2025;
    sonthenguyen/amc12-2025-non-figure (AMC 12A/B 2025, figure-free, MC -> value)
  - transcriptions of official contest PDFs: <SRC_DIR>/*/problems.jsonl
Dedup on normalised text (first 120 alnum chars); HF copies win (checked by others).
    python tools/build_contest_set.py SRC_DIR OUT.jsonl
"""
import glob, json, re, sys
CUTOFF = "2025-04-29"
DIAGRAM_RE = re.compile(r"\[asy\]|asymptote|figure|diagram|as shown|shown below|"
                        r"graph below|picture|image|\.png|\.jpg", re.IGNORECASE)
HF = [("MathArena/aime_2026", "AIME 2026", "2026-02-04", "problem", "answer"),
      ("MathArena/hmmt_feb_2026", "HMMT Feb 2026", "2026-02-14", "problem", "answer"),
      ("MathArena/hmmt_nov_2025", "HMMT Nov 2025", "2025-11-08", "problem", "answer"),
      ("sonthenguyen/amc12-2025-non-figure", "AMC 12 2025", "2025-11-05", "question", "answer")]


def key(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())[:120]


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower())


def near_dup(t, kept):
    """Same problem in another round / another transcription (difflib ratio > 0.85)."""
    import difflib
    a = norm(t)
    for b in kept:
        if abs(len(a) - len(b)) < 0.3 * max(len(a), len(b)):
            m = difflib.SequenceMatcher(None, a, b)
            if m.quick_ratio() > 0.85 and m.ratio() > 0.85:
                return True
    return False


def main(src, out):
    from datasets import load_dataset
    rows, seen, drop, normed = [], set(), {}, []
    def add(r):
        why = (None if r["date"] > CUTOFF else "date") or \
              (None if r["answer"].strip() and r["answer"].strip() not in ("None",) else "no answer") or \
              (None if len(r["problem"]) <= 1200 else "long") or \
              (None if not DIAGRAM_RE.search(r["problem"]) and not r.get("has_figure") else "figure") or \
              (None if key(r["problem"]) not in seen and not near_dup(r["problem"], normed)
               else "duplicate")
        if why:
            drop[why] = drop.get(why, 0) + 1
            return
        seen.add(key(r["problem"]))
        normed.append(norm(r["problem"]))
        rows.append(r)
    for name, label, date, pk, ak in HF:
        for ex in load_dataset(name, split="train"):
            num = ex.get("problem_idx") or ex.get("problem_number") or ""
            add({"source": f"{label} #{num}", "family": label.split(" ")[0], "date": date,
                 "problem": ex[pk].strip(), "answer": str(ex[ak]).strip(),
                 "has_figure": str(ex.get("has_figure", "False")) == "True", "url": f"hf:{name}"})
    for f in sorted(glob.glob(f"{src}/*/problems.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                r["family"] = r["source"].split(" ")[0]
                r["problem"], r["answer"] = r["problem"].strip(), str(r["answer"]).strip()
                add(r)
    with open(out, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    fam = {}
    for r in rows:
        fam[r["family"]] = fam.get(r["family"], 0) + 1
    print(f"kept {len(rows)}  by family {fam}  dropped {drop}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
