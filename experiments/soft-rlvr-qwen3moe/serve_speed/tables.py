"""The README's tables from bench.py's results (protocol 2 rows) and tune_kernels.py's kernel timings.

    python tables.py results2.jsonl [kernel_times.json]

Prints, per model, tokens/s per sequence for every line and adapter count: the whole pass and decode only.
When a configuration was run more than once, the last run counts (the model alone: every run is listed)."""
from __future__ import annotations

import json
import sys

NS = [1, 8, 32, 128, 512]
MODELS = {"Qwen3-30B-A3B-Base": "MoE (Qwen3-30B-A3B)", "Qwen3-32B": "Dense (Qwen3-32B)"}
# (kind, rank, family of the variant) -> label, in the order printed
LINES = [(("cart", 0, ""), "Cartridges, 16 slots each"),
         (("lora", 1, ""), "LoRA rank 1, vLLM defaults"),
         (("lora", 1, "tuned"), "LoRA rank 1, vLLM kernels tuned"),
         (("lora", 1, "gather"), "LoRA rank 1, gather kernels"),
         (("lora", 1, "tuned+attn"), "LoRA rank 1, attention only, vLLM tuned"),
         (("lora", 1, "gather+attn"), "LoRA rank 1, attention only, gather"),
         (("lora", 16, ""), "LoRA rank 16, vLLM defaults"),
         (("lora", 16, "tuned"), "LoRA rank 16, vLLM kernels tuned"),
         (("lora", 16, "gather"), "LoRA rank 16, gather kernels")]


def family(variant: str) -> str | None:
    """"tuned=tuned_m16+ds" -> "tuned": the tuned settings tried are one line (the fastest at each N); greedy runs are checks"""
    parts = [p.split("=")[0] for p in variant.split("+") if p]
    if "greedy" in parts:
        return None
    head = "gather" if "gather" in parts else "tuned" if "tuned" in parts else "ds" if "ds" in parts else ""
    return head + ("+attn" if "attn" in parts else "")


def cell(r: dict | None, metric: str) -> str:
    if r is None:
        return "not run"
    if r["status"] != "ok":
        return "out of memory" if r["status"] == "OOM" else "failed"
    v = r[metric]
    return f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def main() -> None:
    rows = [json.loads(l) for l in open(sys.argv[1])]
    rows = [r for r in rows if r.get("protocol") == 2 and r.get("temperature") == 1.0 and r["batch"] == 512
            and (r["status"] != "error" or r["kind"] == "lora")]
    for metric, title in (("tok_s_seq", "whole pass: 256 / seconds of generate"), ("tok_s_seq_decode", "decode only: 255 / (pass - pass stopped at the first token)")):
        print(f"\n### Tokens/s per sequence, {title}\n")
        for model, name in MODELS.items():
            rs = [r for r in rows if r["model"] == model]
            base = [f"{r[metric]:.1f}" for r in rs if r["kind"] == "base" and not r.get("cart_engine") and r["status"] == "ok"]
            basec = [f"{r[metric]:.1f}" for r in rs if r["kind"] == "base" and r.get("cart_engine") and r["status"] == "ok"]
            print(f"**{name}.** Model alone: {', '.join(base) or 'not run'}; with the prompt given as embeddings: {', '.join(basec) or 'not run'}.\n")
            print("| Distinct finetunes | " + " | ".join(str(n) for n in NS) + " |")
            print("|---|" + "---|" * len(NS))
            for (kind, rank, variant), label in LINES:
                last = {}
                for r in rs:
                    if r["kind"] == kind and r["rank"] == rank and family(r.get("variant") or "") == variant:
                        old = last.get(r["n"])
                        if old is not None and r["status"] != "ok" and old["status"] == "ok":
                            continue        # a later broken run does not replace a good one
                        if old is not None and variant.startswith("tuned") and r["status"] == "ok" and old["status"] == "ok" and r[metric] < old[metric]:
                            continue        # tuned: the fastest of the settings tried
                        last[r["n"]] = r
                if last:
                    print(f"| {label} | " + " | ".join(cell(last.get(n), metric) for n in NS) + " |")
            print()
    if len(sys.argv) > 2:
        k = json.load(open(sys.argv[2]))
        depth = {"dense": 64, "moe": 48}
        print("\n### LoRA kernel time per decode step of 512 tokens, linear layers only (ms): vLLM default / tuned / gather\n")
        print("| Model, rank | " + " | ".join(str(n) for n in NS) + " |")
        print("|---|" + "---|" * len(NS))
        for model in ("moe", "dense"):
            for rank in (1, 16):
                cells = []
                for n in NS:
                    rs = [r for r in k if r["model"] == model and r["rank"] == rank and r["n"] == n]
                    t = lambda a, b: sum(r[a] + r[b] for r in rs) * depth[model]
                    cells.append(f"{t('shrink_default_ms', 'expand_default_ms'):.0f} / {t('shrink_best_ms', 'expand_best_ms'):.0f} / "
                                 f"{t('shrink_gather_ms', 'expand_gather_ms'):.0f}")
                print(f"| {model}, rank {rank} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
