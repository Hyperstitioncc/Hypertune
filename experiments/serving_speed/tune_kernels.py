"""Tune vLLM's LoRA kernels for this benchmark's decode step, kernel by kernel, and time the alternative.

vLLM applies a LoRA to a linear layer with two Triton kernels, shrink (x @ A) and expand (... @ B), whose block
sizes come from a per-GPU table (vllm/lora/ops/triton_ops/README_TUNING.md). No table ships for the B200. This
script times both kernels alone on the shapes of one decode step of the benchmark (512 tokens, sequence i on
adapter i mod N) for a grid of block sizes, and writes the fastest per shape and adapter count in the folder
layout vLLM reads from VLLM_TUNED_CONFIG_FOLDER. It also times vLLM's own gather form of the same two operations
(lora/ops/torch_ops bgmv_shrink / bgmv_expand: pick each token's A and B by index, one batched product), which is
what the engine uses off GPU and does not launch work per adapter.

    python tune_kernels.py OUT_DIR [--iters 30]
writes OUT_DIR/tuned/NVIDIA_B200_{SHRINK,EXPAND_TRUE,EXPAND_FALSE}.json and OUT_DIR/kernel_times.json, and prints
a table: per layer shape and adapter count, milliseconds for the default settings, the best found, and the gather."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import torch

import vllm.lora.ops.triton_ops.lora_expand_op as E
import vllm.lora.ops.triton_ops.lora_shrink_op as S
from vllm.lora.ops.torch_ops import bgmv_expand_slice, bgmv_shrink
from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_expand, lora_shrink
from vllm.lora.ops.triton_ops.utils import get_lora_op_configs

TOKENS = 512
NS = [1, 8, 32, 128, 512]
RANKS = [1, 16]
# (model, layer, input width, output width of every slice): the linear layers a PEFT all-linear adapter touches.
# The MoE expert layers go through a different kernel (fused_moe_lora) and are not tuned here.
LAYERS = [("dense", "qkv_proj", 5120, [8192, 1024, 1024]), ("dense", "o_proj", 8192, [5120]),
          ("dense", "gate_up_proj", 5120, [25600, 25600]), ("dense", "down_proj", 25600, [5120]),
          ("moe", "qkv_proj", 2048, [4096, 512, 512]), ("moe", "o_proj", 4096, [2048])]
SHRINK_GRID = [dict(block_m=m, block_n=16, block_k=k, split_k=s, num_warps=4, num_ctas=1, group_size_m=8, num_stages=2, max_nreg=None)
               for m, k, s in itertools.product([16, 32, 64, 128], [32, 64, 128, 256], [1, 4, 8, 16])]
EXPAND_GRID = [dict(block_m=m, block_n=n, block_k=k, num_warps=w, num_ctas=1, num_stages=2, max_nreg=None)
               for m, n, k, w in itertools.product([16, 32, 64, 128], [32, 64, 128, 256], [16, 32], [4, 8])]
# wider output blocks than vLLM's tuning notes list: with many adapters the fastest settings sat at the 256 edge
EXPAND_GRID += [dict(block_m=m, block_n=n, block_k=16, num_warps=w, num_ctas=1, num_stages=2, max_nreg=None)
                for m, n, w in itertools.product([16, 32], [512, 1024, 2048, 4096], [4, 8])]
DEV = "cuda"


def timed(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


def meta_for(n: int):
    meta = LoRAKernelMeta.make(max_loras=n, max_num_tokens=TOKENS, device=DEV)
    mapping = (torch.arange(TOKENS, device=DEV) % n).long()
    meta.prepare_tensors(mapping)
    return meta, mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "tuned"), exist_ok=True)
    print("GPU:", torch.cuda.get_device_name(), flush=True)
    real = {"shrink": S.get_lora_op_configs, "expand": E.get_lora_op_configs}
    rows, tables = [], {"shrink": {}, "expand": {}}
    for (model, layer, k_in, outs), rank, n in itertools.product(LAYERS, RANKS, NS):
        meta, mapping = meta_for(n)
        args = meta.meta_args(TOKENS, False)
        slices = len(outs)
        x = torch.randn(TOKENS, k_in, device=DEV, dtype=torch.bfloat16)
        A = [torch.randn(n, rank, k_in, device=DEV, dtype=torch.bfloat16) * 0.02 for _ in outs]
        B = [torch.randn(n, o, rank, device=DEV, dtype=torch.bfloat16) * 0.02 for o in outs]
        mid = torch.zeros(slices, TOKENS, rank, device=DEV, dtype=torch.float32)
        y = torch.zeros(TOKENS, sum(outs), device=DEV, dtype=torch.bfloat16)
        ops = {"shrink": lambda: lora_shrink(x, A, mid, *args, 1.0),
               "expand": lambda: lora_expand(mid, B, y, *args, offset_start=0, add_inputs=True)}
        key = {"shrink": (str(args[4].size(0)), str(slices), str(TOKENS), str(k_in), str(rank)),
               "expand": (str(args[4].size(0)), str(slices), str(TOKENS), str(rank), str(max(outs)))}
        res = {"model": model, "layer": layer, "rank": rank, "n": n}
        for op, grid, mod in (("shrink", SHRINK_GRID, S), ("expand", EXPAND_GRID, E)):
            mod.get_lora_op_configs = real[op]
            try:
                res[op + "_default_ms"] = timed(ops[op], a.iters)
            except Exception as e:
                res[op + "_default_ms"] = None
                print("default failed", op, layer, repr(e)[:120], flush=True)
            best, best_cfg = float("inf"), None
            for cfg in grid:
                mod.get_lora_op_configs = lambda *aa, _c=cfg, **kk: _c
                try:
                    t = timed(ops[op], a.iters)
                except Exception:          # block sizes the GPU cannot hold (shared memory) and the like
                    torch.cuda.synchronize()
                    continue
                if t < best:
                    best, best_cfg = t, cfg
            mod.get_lora_op_configs = real[op]
            res[op + "_best_ms"], res[op + "_best"] = best, best_cfg
            d = tables[op]
            for part in key[op][:-1]:
                d = d.setdefault(part, {})
            d[key[op][-1]] = best_cfg

        def gather_shrink():
            for i in range(slices):
                bgmv_shrink(x, A[i], mid[i], mapping, 1.0)

        def gather_expand():
            off = 0
            for i, o in enumerate(outs):
                bgmv_expand_slice(mid[i].to(y.dtype), B[i], y, mapping, off, o, add_inputs=True)
                off += o
        for name, fn in (("shrink_gather_ms", gather_shrink), ("expand_gather_ms", gather_expand)):
            try:
                res[name] = timed(fn, a.iters)
            except Exception as e:
                res[name] = None
                print("gather failed", name, layer, rank, n, repr(e)[:160], flush=True)
        rows.append(res)
        f = lambda v: "   -  " if v is None else f"{v:6.3f}"
        print(f"{model:5s} {layer:13s} r{rank:<2d} N={n:<3d} shrink default {f(res['shrink_default_ms'])} best {f(res['shrink_best_ms'])} "
              f"gather {f(res['shrink_gather_ms'])} | expand default {f(res['expand_default_ms'])} best {f(res['expand_best_ms'])} "
              f"gather {f(res['expand_gather_ms'])} ms", flush=True)
    gpu = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")
    for name, table in ((f"{gpu}_SHRINK.json", tables["shrink"]), (f"{gpu}_EXPAND_TRUE.json", tables["expand"]),
                        (f"{gpu}_EXPAND_FALSE.json", tables["expand"])):
        with open(os.path.join(a.out, "tuned", name), "w") as fh:
            json.dump(table, fh)
    with open(os.path.join(a.out, "kernel_times.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    # what one decode step pays for LoRA on the linear layers, summed over the layers of the model
    depth = {"dense": 64, "moe": 48}
    for model in ("dense", "moe"):
        for rank in RANKS:
            for n in NS:
                rs = [r for r in rows if r["model"] == model and r["rank"] == rank and r["n"] == n]
                tot = lambda a_, b_: sum((r[a_] or 0) + (r[b_] or 0) for r in rs) * depth[model]
                print(f"STEP {model} r{rank} N={n}: default {tot('shrink_default_ms', 'expand_default_ms'):8.1f} ms, "
                      f"best {tot('shrink_best_ms', 'expand_best_ms'):8.1f} ms, gather {tot('shrink_gather_ms', 'expand_gather_ms'):8.1f} ms "
                      f"per decode step of {TOKENS} tokens", flush=True)


if __name__ == "__main__":
    main()
