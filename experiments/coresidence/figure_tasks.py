"""Co-residence learning curves from the learning-curve export (points.csv, written by tools/export_curves.py):
one panel per task, score against examples seen, each adapter alone (dashed, hollow) and inside the stack of all 13
(solid, filled), cartridges and rank-1 LoRAs on the same axes. The last panel is the mean over the 13 tasks with
every curve cut at its own best checkpoint and stretched to one width (read its note on the figure).

    python figure_tasks.py points.csv coresidence_tasks.png [--font-dir DIR] [--dpi 200]

"clean" split only: held-out problems never used to pick a learning rate or a checkpoint."""
from __future__ import annotations

import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

INK, MUTE, RULE, BG = "#000000", "#747372", "#e5e4e2", "#fdfdfd"
# arm -> (name, line, marker core, marker)
ARMS = {"cart": ("Cartridge", "#7fb034", "#23552a", "s"), "lora": ("LoRA rank 1", "#e65d3d", "#a72f18", "o")}
NOTE = {}


def draw(ax, xs, ys, arm, where, se=None):
    _, line, core, mark = ARMS[arm]
    if where == "stack":
        if se:
            ax.fill_between(xs, [y - s for y, s in zip(ys, se)], [y + s for y, s in zip(ys, se)], color=line, alpha=0.16, lw=0, zorder=2)
        ax.plot(xs, ys, color=line, lw=2.4, zorder=4, solid_capstyle="round")
        ax.plot(xs, ys, ls="none", marker=mark, ms=5.5, mfc=core, mec=BG, mew=0.8, zorder=5)
    else:
        ax.plot(xs, ys, color=line, lw=1.3, ls=(0, (4, 2.5)), zorder=3)
        ax.plot(xs, ys, ls="none", marker=mark, ms=4.5, mfc=BG, mec=core, mew=1.0, zorder=3)


def to_peak(ys):
    """a curve cut at its highest point and stretched over 0..1, read at 9 even steps (flat when the start is the highest)"""
    p = max(range(len(ys)), key=lambda k: ys[k])
    out = []
    for j in range(9):
        x = j / 8 * p
        lo = min(int(x), p)
        hi = min(lo + 1, p)
        out.append(ys[lo] + (ys[hi] - ys[lo]) * (x - lo))
    return out


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10.5, pad=8, color=INK)
    ax.grid(axis="y", color=RULE, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
    ax.tick_params(length=3, width=0.8, labelsize=8.5, colors=INK)
    ax.set_ylim(bottom=-0.035)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("points")
    ap.add_argument("out")
    ap.add_argument("--font-dir", default="")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    family = "DejaVu Sans"
    for f in sorted(glob.glob(os.path.join(a.font_dir, "*.otf"))) if a.font_dir else []:
        font_manager.fontManager.addfont(f)
        if "Regular" in f:
            family = font_manager.FontProperties(fname=f).get_name()
    plt.rcParams.update({"font.family": [family, "DejaVu Sans"], "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})

    rows = [r for r in csv.DictReader(open(a.points)) if r["split"] == "clean" and r["arm"] in ARMS]
    tasks = []
    for r in rows:
        if r["task_label"] not in tasks:
            tasks.append(r["task_label"])
    fig, axes = plt.subplots(3, 5, figsize=(16.5, 9.9))
    fig.subplots_adjust(left=0.04, right=0.985, top=0.835, bottom=0.115, wspace=0.26, hspace=0.5)
    flat = axes.flatten()
    series = {(arm, where): [] for arm in ARMS for where in ("alone", "stack")}      # per task: the 9 scores at the stacks' snapshots
    for ax, task in zip(flat, tasks):
        mine = [r for r in rows if r["task_label"] == task]
        top = 0.0
        for arm in ARMS:
            alone = sorted((r for r in mine if r["arm"] == arm and r["where"] == "alone"), key=lambda r: float(r["examples_seen"]))
            stack = sorted((r for r in mine if r["arm"] == arm and r["where"] == "stack"), key=lambda r: int(r["progress_point"]))
            by_snap = {r["snapshot"]: float(r["rate"]) for r in alone}
            draw(ax, [float(r["examples_seen"]) for r in alone], [float(r["rate"]) for r in alone], arm, "alone")
            draw(ax, [float(r["examples_seen"]) for r in stack], [float(r["rate"]) for r in stack], arm, "stack",
                 [float(r["se_vs_plain"]) for r in stack])
            top = max([top] + [float(r["rate"]) for r in alone + stack])
            series[arm, "stack"].append([float(r["rate"]) for r in stack])
            series[arm, "alone"].append([by_snap[r["snapshot"]] for r in stack])
        plain = float(mine[0]["plain_rate"])
        ax.axhline(plain, color=INK, lw=1.0, ls=(0, (1, 2.5)), zorder=2)
        style(ax, NOTE.get(task, task.upper()))
        ax.set_ylim(-0.06 * max(top, plain), max(top, plain) * 1.12)      # room under zero: a line at 0 must not sit on the axis
        ax.text(0.99, 1.035, f"{mine[0]['n_problems']} problems · {'RL' if mine[0]['kind'] == 'rl' else 'SFT'}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color=MUTE)
        ax.xaxis.set_major_formatter(lambda v, _: "0" if v == 0 else f"{v / 1000:g}k")
    for ax in flat[10:13]:
        ax.set_xlabel("Training examples seen", fontsize=9, labelpad=5)

    ax = flat[13]             # the mean of the 13 tasks, every curve cut at its own best checkpoint and stretched to one width
    xs = [k / 8 for k in range(9)]
    ends = []
    for (arm, where), curves in series.items():
        ys = [sum(c[j] for c in map(to_peak, curves)) / len(curves) for j in range(9)]
        draw(ax, xs, ys, arm, where)
        ends.append(ys[-1])
    at = None
    for v in sorted(ends):                    # end labels, kept from sitting on each other
        at = v if at is None else max(v, at + 0.045)
        ax.text(1.03, at, f"{v:.2f}", ha="left", va="center", fontsize=8.5, color=INK)
    plain = [float(next(r for r in rows if r["task_label"] == t)["plain_rate"]) for t in tasks]
    ax.axhline(sum(plain) / len(plain), color=INK, lw=1.0, ls=(0, (1, 2.5)), zorder=2)
    style(ax, "MEAN OF THE 13 · EACH CUT AT ITS BEST")
    ax.set_xlim(-0.04, 1.16)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "¼", "½", "¾", "best"])
    ax.set_xlabel("Fraction of training up to the best checkpoint", fontsize=9, labelpad=5)
    ax.set_facecolor("#f4f4f2")
    flat[14].axis("off")

    fig.text(0.012, 0.955, "13 finetunes loaded into one model at once", fontsize=18, color=INK, fontweight="bold")
    fig.text(0.012, 0.918, "Test score as training goes on: each task's adapter by itself, and the same adapter with the adapters of "
             "all 12 other tasks loaded beside it.  Qwen3-30B-A3B-Base.", fontsize=10.5, color=INK)
    handles = []
    for arm, (name, line, core, mark) in ARMS.items():
        handles.append(Line2D([], [], color=line, lw=2.4, marker=mark, ms=5.5, mfc=core, mec=BG, mew=0.8, label=f"{name} · in the stack of 13"))
        handles.append(Line2D([], [], color=line, lw=1.3, ls=(0, (4, 2.5)), marker=mark, ms=4.5, mfc=BG, mec=core, mew=1.0, label=f"{name} · alone"))
    handles.append(Line2D([], [], color=INK, lw=1.0, ls=(0, (1, 2.5)), label="Plain model"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.006, 0.905), ncol=5, frameon=False, fontsize=9.5, handlelength=3.2,
               columnspacing=2.2)
    note = ("Every adapter was trained on its own task beside\nsix filler partners, never beside another task's\nadapter.\n\n"
            "Stack: all 13 cartridges side by side (208 slots);\nthe 13 rank-1 LoRAs (attention and expert layers)\nsummed. At each point the stack holds every task's\n"
            "snapshot at that fraction of its training.\n\nMean panel: every task's curve is cut at its own\nbest checkpoint and stretched to one width. The\nbest checkpoints were picked on these test\nproblems, and for the stack lines they come from\ndifferent stacks: no single stack of 13 scored\nthat mean.\n\nBands: ±1 standard error against the plain model.\nTask panels: test problems never used to pick\na learning rate or a checkpoint.")
    flat[14].text(0.0, 1.0, note, transform=flat[14].transAxes, ha="left", va="top", fontsize=8.5, color=MUTE, linespacing=1.45)
    fig.savefig(a.out, dpi=a.dpi)
    print("wrote", a.out)
    for (arm, where), curves in series.items():
        print(f"mean of per-task best, {arm} {where}: {sum(max(c) for c in curves) / len(curves):.3f} | at the last checkpoint: {sum(c[-1] for c in curves) / len(curves):.3f}"
              f" | best is the start for {sum(max(range(9), key=lambda k: c[k]) == 0 for c in curves)} tasks")


if __name__ == "__main__":
    main()
