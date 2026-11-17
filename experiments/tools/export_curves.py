"""Export everything behind tools/plot_curves.py's figures as plain files, so the curves can be redrawn anywhere.
    python tools/export_curves.py DEST [PNG_DIR]
DEST gets: points.csv / points.json (every plotted point, both splits), grades/ (per-problem grades of every scored
snapshot), banks/ (the plain-model base rates and the held / probe / clean problem ids), sweeps/ (every LR sweep's
scores and winner), runs.json (which training run each curve follows), meta.json (tasks, arms, labels, colours, axis
text) and the PNGs."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from curves import (DATA, EVAL_DIR, GR_BATCH, GR_G, OUT, PROGRESS, ROOT, SFT_BATCH, TASKS, WARM_BATCH, chain, eval_bank,  # noqa: E402
                         progress_points, winner)
from plot_curves import ARMS, LABEL, bank, name, score  # noqa: E402

FIELDS = ["task", "task_label", "kind", "arm", "arm_label", "where", "split", "examples_seen", "progress_point",
          "snapshot", "rate", "se_vs_plain", "plain_rate", "n_problems", "rollouts_per_problem", "grades_file"]


def rows_for(t, arm: str, where: str) -> list[dict]:
    c = chain(t, arm)
    if len(c) < 2:
        return []
    if where == "alone":
        pts = [(x, None, f, OUT / EVAL_DIR / "alone" / arm / t.name / f"{name(f)}.json") for f, x in c]
    else:
        pts = [(c[i][1], k, c[i][0], OUT / EVAL_DIR / "stack" / arm / t.name / f"stacks__{arm}_p{k}.json")
               for k, i in enumerate(progress_points(c))]
    b, out = bank(t), []
    for x, k, snap, p in pts:
        if not p.exists():
            continue
        grades = json.load(open(p))["grades"]
        rel = p.relative_to(OUT / EVAL_DIR).as_posix()
        for split in ("clean", "held"):
            rate, se, base = score(grades, b, split)
            ids = [i for i in b["held"] if i in grades and (split == "held" or i not in set(b["probe"]))]
            out.append({"task": t.name, "task_label": LABEL.get(t.name, t.name), "kind": "rl" if t.rl else "sft",
                        "arm": arm, "arm_label": ARMS[arm][0], "where": where, "split": split, "examples_seen": x,
                        "progress_point": k, "snapshot": snap, "rate": rate, "se_vs_plain": se, "plain_rate": base,
                        "n_problems": len(ids), "rollouts_per_problem": len(next(iter(grades.values()))),
                        "grades_file": f"grades/{rel}"})
    return out


def main(dest: str, png_dir: str = str(ROOT / "plots")) -> None:
    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    rows = [r for t in TASKS for arm in ARMS for where in ("alone", "stack")
            if not (arm == "lora8" and where == "stack") for r in rows_for(t, arm, where)]
    with open(d / "points.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(d / "points.json", "w"), indent=1)

    for r in {r["grades_file"] for r in rows}:  # the raw per-problem grades behind every point
        src = OUT / EVAL_DIR / r.removeprefix("grades/")
        (d / r).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d / r)
    (d / "banks").mkdir(exist_ok=True)
    for t in TASKS:
        shutil.copy2(DATA / eval_bank(t) / ("bank.json" if t.rl else "bank/bank.json"), d / "banks" / f"{t.name}.json")
    (d / "sweeps").mkdir(exist_ok=True)
    for f in (ROOT / "sweeps").glob("*.scores.json"):
        shutil.copy2(f, d / "sweeps" / f.name)

    runs = {}
    for t in TASKS:
        for arm in ARMS:
            stages = ("warm", "grpo") if t.rl else ("sft",)
            got = {s: winner(t, arm, s) for s in stages}
            if any(got.values()):
                runs[f"{t.name}/{arm}"] = got
    json.dump(runs, open(d / "runs.json", "w"), indent=1)
    json.dump({
        "tasks": [{"task": t.name, "label": LABEL.get(t.name, t.name), "kind": "rl" if t.rl else "sft"} for t in TASKS],
        "arms": {a: {"label": lab, "colour": col, "marker": mk} for a, (lab, col, mk) in ARMS.items()},
        "x_axis": {"rl": f"examples seen: warm-start steps x {WARM_BATCH} up to the warm checkpoint GRPO starts from, "
                         f"then + GRPO steps x {GR_BATCH} problems x {GR_G} rollouts",
                   "sft": f"examples seen: SFT steps x {SFT_BATCH}"},
        "stack": f"at progress point k (0..{PROGRESS - 1}) every task's adapter is its snapshot nearest k/{PROGRESS - 1} "
                 "of the examples that run sees; all 13 cartridges (16 K/V slots each, 208 in all) or all 13 rank-1 "
                 "LoRAs (summed exactly as one rank-13 LoRA) are loaded together",
        "splits": {"clean": "held-out problems never used to pick an LR or a checkpoint (the default figures)",
                   "held": "every held-out problem, probe ones included (the _held figures)"},
    }, open(d / "meta.json", "w"), indent=1)
    for p in Path(png_dir).glob("*.png"):
        shutil.copy2(p, d / p.name)
    print(f"{len(rows)} points -> {d}")


if __name__ == "__main__":
    main(*sys.argv[1:])
