"""Append more transcribed contest problems to the contest set, after its existing rows.
    python tools/extend_contest_set.py SET.jsonl SRC_DIR [SRC_DIR ...]
SRC_DIR/*/problems.jsonl rows (fields as in SET.jsonl) go through build_contest_set.py's filters: held after the
model's release, a non-empty answer, at most 1200 characters, no figure, and no duplicate (same key or > 0.85 alike)
of any row already in SET.jsonl or appended before it. Problem ids are line numbers (envs/contest.py), so the rows
already in SET.jsonl stay exactly as they are and every bank drawn from them keeps its meaning."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_contest_set import CUTOFF, DIAGRAM_RE, key, near_dup, norm  # noqa: E402


def main(path: str, *srcs: str) -> None:
    old = [json.loads(line) for line in open(path) if line.strip()]
    seen, normed = {key(r["problem"]) for r in old}, [norm(r["problem"]) for r in old]
    rows, drop = [], {}
    for src in srcs:
        for f in sorted(Path(src).glob("*/problems.jsonl")):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                r["problem"], r["answer"] = r["problem"].strip(), str(r["answer"]).strip()
                r.setdefault("family", r["source"].split(" ")[0])
                r.setdefault("has_figure", False)
                why = ("date" if r["date"] <= CUTOFF else
                       "no answer" if not r["answer"] or r["answer"] == "None" else
                       "long" if len(r["problem"]) > 1200 else
                       "figure" if DIAGRAM_RE.search(r["problem"]) or r["has_figure"] else
                       "duplicate" if key(r["problem"]) in seen or near_dup(r["problem"], normed) else None)
                if why:
                    drop[why] = drop.get(why, 0) + 1
                    continue
                seen.add(key(r["problem"]))
                normed.append(norm(r["problem"]))
                rows.append({k: r[k] for k in ("source", "family", "date", "problem", "answer", "has_figure", "url")})
    with open(path, "a") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    print(f"{len(old)} rows kept as they were, {len(rows)} appended; dropped {drop}")


if __name__ == "__main__":
    main(*sys.argv[1:])
