"""Score against examples seen, one panel per task, cartridge and LoRA, from tools/curves.py's evals:
curves_alone.png (each adapter alone) and curves_stack.png (inside the stack of all thirteen at the same fraction
of training). Scored on the bank's clean problems: held out, and never used to pick an LR or a checkpoint (the
_held variants score every held problem, probe ones included). Shaded band = 1 SE of the paired difference to the
plain model (dashed line).
    python tools/plot_curves.py OUT_PNG_DIR"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from curves import DATA, OUT, TASKS, chain, progress_points  # noqa: E402

ARMS = {"cart": ("cartridge", "#2a78d6", "o"), "lora": ("LoRA rank 1", "#eb6834", "s"),
        "lora8": ("LoRA rank 8, trained alone", "#1baf7a", "^")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABEL = {"contest_zs": "math", "mbpp_nt_zs": "code", "logic_zs": "logic", "know_law": "law", "know_casehold": "casehold",
         "know_medqa": "medqa", "know_trivia": "trivia", "bf": "Brainfuck", "finqa": "FinQA", "funcall": "function calling",
         "intent": "intent", "parse": "parsing", "formula": "formula"}


def name(f: str) -> str:
    return Path(f).with_suffix("").as_posix().replace("/", "__")


def bank(t) -> dict:
    return json.load(open(DATA / t.name / ("bank.json" if t.rl else "bank/bank.json")))


def score(grades: dict[str, list[float]], b: dict, split: str) -> tuple[float, float, float]:
    """(rate, SE of the paired difference to the bank's base rate, base rate) over the split's problems."""
    probe = set(b["probe"])
    ids = [i for i in b["held"] if i in grades and (split == "held" or i not in probe)]
    if not ids:
        raise ValueError(f"no {split} problem among the graded ones")
    a = [sum(grades[i]) / len(grades[i]) for i in ids]
    d = [x - b["rates"][i] for x, i in zip(a, ids)]
    md = sum(d) / len(d)
    se = (sum((x - md) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5 if len(d) > 1 else 0.0
    return sum(a) / len(a), se, sum(b["rates"][i] for i in ids) / len(ids)


def points(t, arm: str, where: str, split: str) -> list[tuple[int, float, float, float]]:
    c = chain(t, arm)
    if where == "alone":
        pts = [(x, OUT / "curves/alone" / arm / t.name / f"{name(f)}.json") for f, x in c]
    else:
        idx = progress_points(c) if len(c) >= 2 else []
        pts = [(c[i][1], OUT / "curves/stack" / arm / t.name / f"stacks__{arm}_p{k}.json") for k, i in enumerate(idx)]
    b = bank(t) if pts else None
    return [(x, *score(json.load(open(p))["grades"], b, split)) for x, p in pts if p.exists()]


def figure(where: str, path: Path, split: str) -> None:
    cols = 5
    rows = -(-len(TASKS) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    for ax, t in zip(axes.flat, TASKS):
        base = None
        for arm, (lab, col, mk) in ARMS.items():
            if arm == "lora8" and where == "stack":
                continue
            pts = points(t, arm, where, split)
            if not pts:
                continue
            xs, ys, se, b = zip(*pts)
            base = b[0]
            ax.fill_between(xs, [y - s for y, s in zip(ys, se)], [y + s for y, s in zip(ys, se)], color=col, alpha=0.15,
                            linewidth=0)
            ax.plot(xs, ys, color=col, marker=mk, markersize=4, linewidth=2, label=lab)
        if base is not None:
            ax.axhline(base, color=MUTED, linestyle="--", linewidth=1, label="plain model")
        ax.set_title(LABEL.get(t.name, t.name), fontsize=10, color=INK, loc="left")
        ax.grid(color=GRID, linewidth=0.6)
        ax.tick_params(labelsize=8, colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("examples seen" + (" (warm start, then GRPO)" if t.rl else " (SFT)"), fontsize=8, color=MUTED)
        ax.set_ylabel("score", fontsize=8, color=MUTED)
    for ax in list(axes.flat)[len(TASKS):]:
        ax.axis("off")
    seen: dict[str, object] = {}
    for a in axes.flat:  # every arm that appears in any panel, once
        for hd, lb in zip(*a.get_legend_handles_labels()):
            seen.setdefault(lb, hd)
    fig.legend(list(seen.values()), list(seen), loc="lower right", frameon=False, fontsize=10)
    fig.suptitle({"alone": "Each adapter alone", "stack": f"Inside the stack of all {len(TASKS)} (same fraction of training)"}[where],
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    print(path)


def main(out: str) -> None:
    Path(out).mkdir(parents=True, exist_ok=True)
    for where in ("alone", "stack"):
        for split, tag in (("clean", ""), ("held", "_held")):
            figure(where, Path(out) / f"curves_{where}{tag}.png", split)


if __name__ == "__main__":
    main(sys.argv[1])
