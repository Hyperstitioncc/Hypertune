"""Decode speed per sequence against the number of distinct finetunes in one batch.

The batch is always BATCH identical short prompts; the only thing that changes is how many different adapters
those sequences use (prompt i gets adapter i mod N). Every sequence decodes exactly MAX_TOKENS tokens
(ignore_eos), one warm-up pass, then timed passes of llm.generate; the figure is MAX_TOKENS / elapsed seconds
(tok_s_seq). A pass stopped after the first token gives the prompt work, and tok_s_seq_decode is the speed of
the remaining tokens alone. The prompts differ by one number each, and every engine runs without prefix cache
and without chunked prefill, so no line gets prompt work for free that another line pays for.
Tokens are sampled at vLLM's default temperature of 1, so the sequences differ from one another in every
configuration. With --temperature 0 the plain model writes 512 identical sequences, every token of a step goes to
the same few experts of the MoE, and the step is much faster than any real batch: not a baseline to compare to.

    python bench.py --model DIR --kind base                          the model alone, stock vLLM
    python bench.py --model DIR --kind base --cart-engine true       the model alone, engine set up as for cartridges
    python bench.py --model DIR --kind lora --n N --rank R --adapters DIR     stock vLLM LoRA (make_loras.py folders)
        [--lora-ops gather]   the LoRA products by gather (lora_gather.py; this folder on PYTHONPATH)
        [--targets attn]      the engine applies LoRA to q/k/v/o only
    python bench.py --model DIR --kind cart --n N                    N cartridges of 16 token slots (shared.kv_graph)
    python bench.py --model DIR --kind check --adapters DIR --rank R         do the expert layers get LoRA?
  common: --out results.jsonl --log THIS_RUN'S_LOG [--batch 512] [--max-tokens 256] [--repeats 3]

Adapters and cartridges are untrained noise: speed does not depend on the weights. One JSON line per run is
appended to --out; a run that dies of GPU memory is recorded as status "OOM" (found in --log) and not retried.
PYTHONPATH must hold the library root (the repository root) for --kind cart."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import socket
import subprocess
import sys
import time

PROMPT = "The lighthouse keeper found a bottle on the rocks. Inside it was a letter, and the letter said:"
SLOTS = 16


def str2bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes")


def folder_digest(d: str) -> str:
    """which tuned kernel table a run used: a short hash over the files of VLLM_TUNED_CONFIG_FOLDER"""
    import hashlib
    h = hashlib.md5()
    for f in sorted(glob.glob(os.path.join(d, "*.json"))) if d else []:
        h.update(os.path.basename(f).encode())
        h.update(open(f, "rb").read())
    return h.hexdigest()[:8] if d else ""


def gpu_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                              "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def embed_table(model_dir: str):
    from safetensors import safe_open
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                if k.endswith("embed_tokens.weight"):
                    return sf.get_tensor(k)
    raise FileNotFoundError(f"no embed_tokens.weight under {model_dir}")


def prompts(batch: int) -> list[str]:
    """One prompt per sequence, all different, so that no engine can reuse one sequence's prompt work for another."""
    return [f"Letter {i + 1} of {batch}. {PROMPT}" for i in range(batch)]


def prompt_embeds(model_dir: str, texts: list[str]):
    import torch
    from transformers import AutoTokenizer
    emb = embed_table(model_dir).float()
    tok = AutoTokenizer.from_pretrained(model_dir)
    return emb, [emb[torch.tensor(tok(t)["input_ids"])].to(torch.bfloat16) for t in texts]


def cart_requests(model_dir: str, n: int, batch: int, work: str):
    """N random cartridges in the library's engine format, and one prompt_embeds request per sequence: the prompt
    with the 16 placeholder rows of its cartridge before its last token (the library's --row-tail layout)."""
    import torch
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_dir)
    L, H, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    emb, pes = prompt_embeds(model_dir, prompts(batch))
    g = torch.Generator().manual_seed(n)
    R = n * SLOTS
    soft = (torch.randn(R, emb.shape[1], generator=g) * emb.std()).to(torch.bfloat16)
    k = torch.randn(L, R, H, hd, generator=g).to(torch.bfloat16)
    v = (torch.randn(L, R, H, hd, generator=g) * 0.05).to(torch.bfloat16)
    path = os.path.join(work, f"kv_{os.path.basename(model_dir.rstrip('/'))}_{n}_{os.getpid()}.pt")     # runs of other models share the folder
    torch.save({"base": soft, "dk": k, "dv": v, "rm": torch.zeros(L, R, H), "replace": True, "pos0": True}, path)
    reqs = []
    for i, pe in enumerate(pes):
        rows = soft[(i % n) * SLOTS:(i % n + 1) * SLOTS]
        reqs.append({"prompt_embeds": torch.cat([pe[:-1], rows, pe[-1:]], 0)})
    return path, reqs, max(pe.shape[0] for pe in pes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", choices=["base", "lora", "cart", "check"], required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--adapters", default="")
    ap.add_argument("--cart-engine", type=str2bool, default=False)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)   # vLLM's default: the 512 sequences drift apart, as real traffic does
    ap.add_argument("--lora-ops", choices=["vllm", "gather"], default="vllm")    # gather: lora_gather.py in place of vLLM's GPU kernels
    ap.add_argument("--targets", choices=["all", "attn"], default="all")          # attn: the engine gives LoRA to q/k/v/o only
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default="")
    ap.add_argument("--work", default=os.environ.get("TMPDIR", "/tmp"))
    a = ap.parse_args()
    name = os.path.basename(a.model.rstrip("/"))
    row = {"protocol": 2, "model": name, "kind": a.kind, "n": a.n, "rank": a.rank, "batch": a.batch, "max_tokens": a.max_tokens,
           "lora_ops": a.lora_ops if a.kind == "lora" else "", "targets": a.targets if a.kind == "lora" else "",
           "gather_skips_reactivation": a.kind == "lora" and a.lora_ops == "gather",     # lora_gather.py from its second version on
           "temperature": a.temperature, "cart_engine": a.cart_engine or a.kind == "cart", "host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID"),
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           # how vLLM's LoRA path was set up for this run (slurm_job.sh sets these from the configuration's variant)
           "variant": os.environ.get("SS_VARIANT", ""), "adapters": os.path.basename(a.adapters.rstrip("/")),
           "dual_stream": os.environ.get("VLLM_LORA_ENABLE_DUAL_STREAM", ""),
           "tuned_kernels": folder_digest(os.environ.get("VLLM_TUNED_CONFIG_FOLDER", ""))}

    def write(**more) -> None:
        row.update(more)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("RESULT " + json.dumps(row), flush=True)

    try:
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"      # a compiled model cached by another configuration must not be reused
        # the same engine for every line: no prefix cache and no chunked prefill, which a cartridge engine cannot use
        kw = dict(model=a.model, dtype="bfloat16", max_num_seqs=a.batch, max_model_len=1024,
                  gpu_memory_utilization=0.92, seed=0, enable_prefix_caching=False, enable_chunked_prefill=False)
        reqs: list = prompts(a.batch)
        loras = None
        fused = False
        if a.kind in ("lora", "check"):
            n = 2 if a.kind == "check" else a.n
            kw.update(enable_lora=True, max_lora_rank=a.rank, max_loras=n)
            with open(os.path.join(a.model, "config.json")) as f:
                fused = json.load(f)["model_type"].endswith("_moe")
            if fused:       # PEFT writes the expert adapters as one fused tensor pair per layer; vLLM has to be told
                kw["enable_mixed_moe_lora_format"] = True
            if a.lora_ops == "gather":
                kw["worker_extension_cls"] = "lora_gather.GatherWorker"
                if os.environ.get("SS_GATHER_CHECK") == "1":     # lora_gather prints its comparison with vLLM's kernels: no CUDA graphs
                    kw["enforce_eager"] = True
            if a.targets == "attn":
                kw["lora_target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lreq = lambda name, i, path: LoRARequest(name, i, path, is_3d_lora_weight=fused)
        if row["cart_engine"]:       # a cartridge's prompt goes in as embeddings; the plain model can be run the same way
            kw["enable_prompt_embeds"] = True
            if a.kind == "base":
                reqs = [{"prompt_embeds": pe} for pe in prompt_embeds(a.model, prompts(a.batch))[1]]
        if a.kind == "cart":
            os.environ["KV_GRAPH_ROWS"] = str(max(256, a.n * SLOTS))
            kw["worker_extension_cls"] = "shared.kv_graph_worker.KVGraphWorker"
            path, reqs, plen = cart_requests(a.model, a.n, a.batch, a.work)
            row["prompt_tokens"] = plen
        row.update(vllm=vllm.__version__, engine={k: v for k, v in kw.items() if k != "model"})
        t0 = time.perf_counter()
        llm = LLM(**kw)
        row["engine_start_s"] = round(time.perf_counter() - t0, 1)
        if a.kind == "cart":
            llm.collective_rpc("set_kv_delta", args=(path,))
            llm.reset_prefix_cache()
            os.remove(path)

        if a.kind == "check":       # attention-only and experts/MLP-only adapters against the plain model
            sp = SamplingParams(temperature=0.0, max_tokens=48, ignore_eos=True)
            def toks(lr):
                o = llm.generate([PROMPT] * 4, sp, lora_request=lr, use_tqdm=False)      # one prompt four times, on purpose
                return [list(x.outputs[0].token_ids) for x in o]
            base = toks(None)
            got = {}
            for j, part in enumerate(("attn", "mlp")):
                t = toks(lreq(part, j + 1, os.path.join(a.adapters, part, "0000")))
                got[part] = {"differs_from_plain_model": t[0] != base[0], "same_across_copies": all(x == t[0] for x in t),
                             "first_tokens": t[0][:12]}
            write(status="ok", plain_first_tokens=base[0][:12], adapters=got,
                  experts_adapted=got["mlp"]["differs_from_plain_model"])
            return

        if a.kind == "lora":
            reqs_l = [lreq(f"a{j}", j + 1, os.path.join(a.adapters, f"{j:04d}")) for j in range(a.n)]
            loras = [reqs_l[i % a.n] for i in range(a.batch)]
        sp = SamplingParams(temperature=a.temperature, max_tokens=a.max_tokens, ignore_eos=True)

        def run(params=sp, want=a.max_tokens):
            t = time.perf_counter()
            outs = llm.generate(reqs, params, lora_request=loras, use_tqdm=False)
            dt = time.perf_counter() - t
            lens = {len(o.outputs[0].token_ids) for o in outs}
            if lens != {want}:
                raise RuntimeError(f"sequences decoded {sorted(lens)} tokens, not {want}")
            return dt, outs

        warm, outs = run()
        row.update(warmup_s=round(warm, 2), gpu_mem_mib=gpu_mib(), load_avg=os.getloadavg()[0],
                   first_tokens=list(outs[0].outputs[0].token_ids)[:12],
                   tokens_head=[list(o.outputs[0].token_ids)[:8] for o in outs[:8]],
                   distinct_outputs=len({tuple(o.outputs[0].token_ids) for o in outs}))
        # a configuration whose pass takes minutes is timed once: the spread between passes is a few percent
        slow = warm > 90
        el = [run()[0] for _ in range(1 if slow else a.repeats)]
        best = sorted(el)[len(el) // 2]
        # the same batch stopped after its first token: the prompt work, which the decode-only figure leaves out
        one = SamplingParams(temperature=a.temperature, max_tokens=1, ignore_eos=True)
        pre = min(run(one, 1)[0] for _ in range(1 if slow else 2))
        write(status="ok", elapsed_s=[round(x, 3) for x in el], tok_s_seq=round(a.max_tokens / best, 2),
              tok_s_seq_all=[round(a.max_tokens / x, 2) for x in el], tok_s_total=round(a.batch * a.max_tokens / best, 1),
              first_token_s=round(pre, 3), tok_s_seq_decode=round((a.max_tokens - 1) / max(best - pre, 1e-6), 2))
    except BaseException as e:      # the engine runs in its own process: its death arrives here as a plain error
        sys.stdout.flush()
        time.sleep(3)
        text = repr(e)
        if a.log and os.path.isfile(a.log):
            with open(a.log, errors="replace") as f:
                text += f.read()
        oom = re.search(r"out of memory|OutOfMemoryError", text, re.I) is not None
        write(status="OOM" if oom else "error", error=repr(e)[:400], gpu_mem_mib=gpu_mib())
        if not isinstance(e, Exception):
            raise


if __name__ == "__main__":          # vLLM starts its engine in a spawned process, which re-imports this file
    main()
