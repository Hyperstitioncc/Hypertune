"""Training step against the cost of co-residence, from the learning-curve export (points.csv).

Left: the mean score of the 13 tasks, each adapter alone and inside the stack of all 13, against the fraction of
training. Middle and right: per task, score in the stack minus score alone at the same snapshot (0 = loading the
other 12 costs nothing), cartridges and rank-1 LoRAs.

    python figure_cost.py points.csv coresidence_cost.png [--font-dir DIR] [--dpi 200]

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
ARMS = {"cart": ("Cartridges", "#7fb034", "#23552a", "s"), "lora": ("Rank-1 LoRAs", "#e65d3d", "#a72f18", "o")}
XS = [k / 8 for k in range(9)]


def style(ax, title, xlabel="Fraction of training"):
    ax.set_title(title, loc="left", fontsize=11.5, pad=12, color=INK)
    ax.grid(axis="y", color=RULE, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
    ax.tick_params(length=4, width=0.8, labelsize=10, colors=INK)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "¼", "½", "¾", "all"])
    ax.set_xlabel(xlabel, fontsize=10, labelpad=8)


def labels(ax, ends, x, gap, **kw):
    """text at the right end of lines, pushed apart (evenly, both ways) so that none sits on another"""
    ends = sorted(ends)
    at = [y for y, _ in ends]
    for _ in range(400):
        for i in range(len(at) - 1):
            short = gap - (at[i + 1] - at[i])
            if short > 0:
                at[i] -= short / 2
                at[i + 1] += short / 2
    for y, (_, text) in zip(at, ends):
        ax.text(x, y, text, ha="left", va="center", **kw)


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
    # curve[arm][task] = {"alone": [9 scores], "stack": [9 scores]} at the snapshots the stacks were built from
    curve = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for t in tasks:
            mine = [r for r in rows if r["task_label"] == t and r["arm"] == arm]
            alone = {r["snapshot"]: float(r["rate"]) for r in mine if r["where"] == "alone"}
            stack = sorted((r for r in mine if r["where"] == "stack"), key=lambda r: int(r["progress_point"]))
            curve[arm][t] = {"stack": [float(r["rate"]) for r in stack], "alone": [alone[r["snapshot"]] for r in stack]}
    kind = {t: next(r for r in rows if r["task_label"] == t)["kind"] for t in tasks}
    plain = sum(float(next(r for r in rows if r["task_label"] == t)["plain_rate"]) for t in tasks) / len(tasks)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.6), gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]})
    fig.subplots_adjust(left=0.05, right=0.925, top=0.76, bottom=0.2, wspace=0.4)

    ax = axes[0]
    ends = []
    for arm, (name, line, core, mark) in ARMS.items():
        for where in ("alone", "stack"):
            ys = [sum(curve[arm][t][where][k] for t in tasks) / len(tasks) for k in range(9)]
            if where == "stack":
                ax.plot(XS, ys, color=line, lw=2.6, zorder=4)
                ax.plot(XS, ys, ls="none", marker=mark, ms=7, mfc=core, mec=BG, mew=1.0, zorder=5)
            else:
                ax.plot(XS, ys, color=line, lw=1.5, ls=(0, (4, 2.5)), zorder=3)
                ax.plot(XS, ys, ls="none", marker=mark, ms=6, mfc=BG, mec=core, mew=1.2, zorder=3)
            ends.append((ys[-1], f"{ys[-1]:.2f}"))
    ax.axhline(plain, color=INK, lw=1.0, ls=(0, (1, 2.5)), zorder=2)
    ends.append((plain, f"{plain:.2f}"))
    ax.text(0.5, plain - 0.012, "plain model", ha="center", va="top", fontsize=9, color=INK)
    labels(ax, ends, 1.04, 0.035, fontsize=9.5, color=INK)
    style(ax, "MEAN SCORE OF THE 13 TASKS")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(0, 0.68)
    ax.set_ylabel("Test score", fontsize=10, labelpad=8)

    for ax, (arm, (name, line, core, mark)) in zip(axes[1:], ARMS.items()):
        ends = []
        for t in tasks:
            d = [s - al for s, al in zip(curve[arm][t]["stack"], curve[arm][t]["alone"])]
            ax.plot(XS, d, color=line, lw=1.2, alpha=0.8, zorder=3, ls="-" if kind[t] == "rl" else (0, (4, 2)))
            ends.append((d[-1], t))
        m = [sum(curve[arm][t]["stack"][k] - curve[arm][t]["alone"][k] for t in tasks) / len(tasks) for k in range(9)]
        ax.plot(XS, m, color=core, lw=3.0, zorder=5)
        ax.plot(XS, m, ls="none", marker=mark, ms=7, mfc=core, mec=BG, mew=1.0, zorder=6)
        ax.axhline(0, color=INK, lw=1.0, zorder=2)
        labels(ax, ends, 1.03, 0.047, fontsize=8, color=MUTE)
        where_ = dict(x=0.03, y=0.05, ha="left", va="bottom") if arm == "cart" else dict(x=0.97, y=0.95, ha="right", va="top")
        ax.text(s=f"thick line: mean of 13 · {m[-1]:+.2f} at the end", transform=ax.transAxes, fontsize=9.5, color=INK, fontweight="bold", **where_)
        style(ax, f"{name.upper()} · COST OF LOADING THE OTHER 12")
        ax.set_xlim(-0.04, 1.03)
        ax.set_ylim(-0.92, 0.26)
    axes[1].set_ylabel("Score in the stack minus score alone", fontsize=10, labelpad=8)

    fig.text(0.012, 0.94, "What loading 13 finetunes together costs, as training goes on", fontsize=17, color=INK, fontweight="bold")
    fig.text(0.012, 0.895, "13 tasks, one adapter each, Qwen3-30B-A3B-Base. At every point the stack holds every task's snapshot at that fraction of its own training, "
             "and each adapter is scored at that same snapshot alone.", fontsize=10.5, color=INK)
    handles = [Line2D([], [], color=INK, lw=2.6, marker="s", ms=7, mfc=INK, mec=BG, label="in the stack of 13"),
               Line2D([], [], color=INK, lw=1.5, ls=(0, (4, 2.5)), marker="s", ms=6, mfc=BG, mec=INK, mew=1.2, label="alone"),
               Line2D([], [], color=MUTE, lw=1.2, label="one task, trained by RL"),
               Line2D([], [], color=MUTE, lw=1.2, ls=(0, (4, 2)), label="one task, trained by SFT"),
               Line2D([], [], color="#7fb034", lw=2.6, marker="s", ms=7, mfc="#23552a", mec=BG, label="cartridges"),
               Line2D([], [], color="#e65d3d", lw=2.6, marker="o", ms=7, mfc="#a72f18", mec=BG, label="rank-1 LoRAs")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.006, 0.875), ncol=6, frameon=False, fontsize=9.5, handlelength=3.0, columnspacing=2.0)
    fig.text(0.012, 0.025, "Stack: all 13 cartridges side by side (208 slots); the 13 rank-1 LoRAs (attention and expert layers) summed, with no router choosing one. Every adapter was trained beside six filler partners,\n"
             "never beside another task's adapter. Test problems never used to pick a learning rate or a checkpoint. Math has 9 problems.",
             fontsize=8.5, color=MUTE, va="bottom", linespacing=1.5)
    fig.savefig(a.out, dpi=a.dpi)
    print("wrote", a.out)
    print("fraction of training:      " + "  ".join(f"{x:5.3f}" for x in XS))
    for arm in ARMS:
        for where in ("alone", "stack"):
            print(f"mean {arm} {where:5s}:          " + "  ".join(f"{sum(curve[arm][t][where][k] for t in tasks) / len(tasks):5.3f}" for k in range(9)))
    for arm in ARMS:
        for k_ in ("rl", "sft"):
            ts = [t for t in tasks if kind[t] == k_]
            print(f"end cost {arm} {k_} ({len(ts)} tasks): {sum(curve[arm][t]['stack'][-1] - curve[arm][t]['alone'][-1] for t in ts) / len(ts):+.3f}")
    print("cartridge cost by task (stack - alone):")
    for t in tasks:
        print(f"  {t:17s}        " + "  ".join(f"{curve['cart'][t]['stack'][k] - curve['cart'][t]['alone'][k]:+5.2f}" for k in range(9)))


if __name__ == "__main__":
    main()
