"""The serving-speed figure from bench.py's results: tokens/s per sequence against the number of distinct
finetunes in one batch of 512, one panel per model.

    python plot.py results.jsonl figure.png [--rank 1] [--metric tok_s_seq|tok_s_seq_decode]
                   [--series cart,lora] [--note "footer text"]
                   [--font-dir DIR_WITH_OTF_FILES] [--font "Family Name"]

A configuration that ran out of GPU memory is drawn as an "OOM" mark at its N, at the bottom of the panel. When a
configuration was run more than once the last run counts."""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

NS = [1, 8, 32, 128, 512]
INK, MUTE, RULE = "#000000", "#747372", "#e5e4e2"
# series: (key, label, line colour, marker edge, marker core, dash)
SERIES = [("cart", "Cartridges · 16 token slots each", "#b9e04a", "#7fb034", "#23552a", "-"),
          ("lora", "LoRA · vLLM defaults", "#f3b0a0", "#e65d3d", "#a72f18", "-"),
          ("lora-tuned", "LoRA · vLLM, kernels tuned for this batch", "#dfb5eb", "#b05cc4", "#5f2554", "-"),
          ("lora-gather", "LoRA · gather kernels", "#84a7f9", "#3742dc", "#293278", "-"),
          ("lora-gather-attn", "LoRA on attention only · gather kernels", "#9fd3d5", "#388083", "#1d4a4c", (0, (5, 2)))]


def family(variant: str) -> str | None:
    """The line a LoRA row belongs to, from its "variant" (slurm_job.sh); None = not drawn (greedy checks, other setups)."""
    parts = [p.split("=")[0] for p in variant.split("+") if p]
    if "greedy" in parts:
        return None
    if "attn" in parts:         # of the attention-only setups only the gather one is drawn: LoRA's best case here
        return "lora-gather-attn" if "gather" in parts else None
    return "lora-gather" if "gather" in parts else "lora-tuned" if "tuned" in parts else "lora" if not parts else None
TITLE = {"Qwen3-30B-A3B-Base": "QWEN3-30B-A3B · MIXTURE OF EXPERTS", "Qwen3-32B": "QWEN3-32B · DENSE"}


def load(path: str, rank: int = 1, metric: str = "tok_s_seq", temperature: float = 1.0) -> dict:
    """model -> {"base": tok/s, "basec": tok/s, series key -> {N: tok/s or "OOM"}}, LoRA lines of one rank"""
    out: dict = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("protocol") != 2 or r.get("temperature", 0.0) != temperature or r["kind"] == "check" or r["status"] == "error":
            continue        # "error" is a run that broke for a reason other than memory: not a result
        m = out.setdefault(r["model"], {})
        val = r[metric] if r["status"] == "ok" else r["status"]
        if r["kind"] == "base":
            if r["status"] == "ok":         # run more than once: the mean
                m.setdefault("_base", {}).setdefault("basec" if r.get("cart_engine") else "base", []).append(val)
        elif r["kind"] == "cart":
            m.setdefault("cart", {})[r["n"]] = val
        elif r["kind"] == "lora":
            key = family(r.get("variant") or "")
            if r["rank"] == rank and key:
                old = m.setdefault(key, {}).get(r["n"])
                # several tuned settings were tried: the fastest at each N is drawn; elsewhere the last run counts
                if key == "lora-tuned" and isinstance(old, (int, float)) and (not isinstance(val, (int, float)) or val < old):
                    continue
                m[key][r["n"]] = val
    for m in out.values():
        for k, v in m.pop("_base", {}).items():
            m[k] = sum(v) / len(v)
    return out


def panel(ax, name: str, d: dict, ymax: float) -> None:
    x = {n: i for i, n in enumerate(NS)}
    ax.set_xlim(-0.35, len(NS) - 0.65)
    ax.set_ylim(0, ymax)
    base, basec = d.get("base"), d.get("basec")
    if isinstance(base, (int, float)):
        ax.axhline(base, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.text(len(NS) - 0.68, base + ymax * 0.012, f"model alone · {base:.1f}", ha="right", va="bottom", fontsize=9.5, color=INK)
    if isinstance(basec, (int, float)):     # the reference for the cartridge line: same engine settings, no cartridge
        ax.axhline(basec, color="#23552a", lw=0.9, ls=(0, (1, 2.5)), zorder=2)
        ax.text(-0.3, basec - ymax * 0.055, f"model alone, prompt given as embeddings · {basec:.1f}", ha="left", va="top", fontsize=8.5, color="#23552a")
    oom_row = 0
    for key, label, line, edge, core, dash in SERIES:
        pts = d.get(key, {})
        ok = [(x[n], v) for n, v in sorted(pts.items()) if n in x and isinstance(v, (int, float))]
        if ok:
            xs, ys = zip(*ok)
            ax.plot(xs, ys, color=line, lw=2.2, ls=dash, zorder=3, solid_capstyle="butt")
            ax.plot(xs, ys, ls="none", marker="s", ms=9, mfc=line, mec=line, zorder=4)
            ax.plot(xs, ys, ls="none", marker="s", ms=5, mfc=core, mec=edge, mew=1.0, zorder=5, label=label)
            ax.text(xs[-1] + 0.09, ys[-1], f"{ys[-1]:.1f}", ha="left", va="center", fontsize=9.5, color=core)
        bad = [(x[n], v) for n, v in sorted(pts.items()) if n in x and not isinstance(v, (int, float))]
        for xi, v in bad:
            y = ymax * (0.05 + 0.075 * oom_row)
            ax.plot([xi], [y], ls="none", marker="s", ms=9, mfc="none", mec=edge, mew=1.4, zorder=5)
            ax.text(xi, y + ymax * 0.03, "out of memory", ha="center", va="bottom", fontsize=8.5, color=core)
        if bad:
            oom_row += 1
    ax.set_xticks(range(len(NS)), [str(n) for n in NS])
    ax.set_title(TITLE.get(name, name.upper()), loc="left", fontsize=11.5, pad=14, color=INK)
    ax.set_xlabel("Distinct finetunes in the batch of 512", fontsize=10, labelpad=8)
    ax.grid(axis="y", color=RULE, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(length=4, width=0.8, labelsize=10, colors=INK)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("out")
    ap.add_argument("--font-dir", default="")
    ap.add_argument("--font", default="DejaVu Sans Mono")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--metric", choices=["tok_s_seq", "tok_s_seq_decode"], default="tok_s_seq")   # whole pass, or without the prompt work
    ap.add_argument("--series", default="")       # the lines to draw, e.g. "cart,lora"; default: all
    ap.add_argument("--note", default="")         # replaces the footer line ("\\n" starts a new line)
    a = ap.parse_args()
    for f in glob.glob(os.path.join(a.font_dir, "*.otf")) + glob.glob(os.path.join(a.font_dir, "*.ttf")) if a.font_dir else []:
        font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": [a.font, "DejaVu Sans Mono"], "figure.facecolor": "#fdfdfd", "axes.facecolor": "#fdfdfd",
                         "savefig.facecolor": "#fdfdfd"})
    data = load(a.results, a.rank, a.metric)
    if a.series:
        keep = set(a.series.split(",")) | {"base", "basec"}
        data = {name: {k: v for k, v in d.items() if k in keep} for name, d in data.items()}
    names = [n for n in TITLE if n in data] + [n for n in data if n not in TITLE]
    fig, axes = plt.subplots(1, len(names), figsize=(6.6 * len(names), 5.4), squeeze=False)
    for ax, name in zip(axes[0], names):        # each model on its own scale
        nums = [v for s in data[name].values() for v in (s.values() if isinstance(s, dict) else [s]) if isinstance(v, (int, float))]
        panel(ax, name, data[name], max(nums) * 1.2)
    axes[0][0].set_ylabel("Tokens/s per sequence" + (" · decode only" if a.metric.endswith("decode") else ""), fontsize=10, labelpad=8)
    h, l = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    order = [x[1] for x in SERIES]
    hl = sorted(zip(h, l), key=lambda t: order.index(t[1]))
    fig.legend([x[0] for x in hl], [x[1] for x in hl], loc="upper left", bbox_to_anchor=(0.055, 0.995), ncol=3, frameon=False,
               fontsize=9.5, handletextpad=0.4, columnspacing=1.8, labelspacing=0.5)
    if a.note:
        fig.text(0.065, 0.03, a.note.replace("\\n", "\n"), ha="left", va="bottom", fontsize=8.5, color=MUTE, linespacing=1.5)
    else:
        fig.text(0.965, 0.04, f"512 sequences decoding 256 tokens each · LoRA rank {a.rank}, on every linear layer unless noted · one B200 · vLLM 0.33",
                 ha="right", va="bottom", fontsize=9, color=MUTE)
    fig.subplots_adjust(left=0.065, right=0.965, top=0.80, bottom=0.17, wspace=0.14)
    fig.savefig(a.out, dpi=a.dpi)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
