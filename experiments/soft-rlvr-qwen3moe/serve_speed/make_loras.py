"""Random LoRA adapters for the serving-speed benchmark, written with PEFT. Speed does not depend on the weights, so
nothing is trained: every adapter is non-zero noise (init_lora_weights=False, then redrawn per adapter so the files
differ). All linear layers are adapted: q/k/v/o, and gate/up/down, which on the MoE are the expert projections
(this transformers stores them as two fused parameters per layer, gate_up_proj and down_proj, which PEFT adapts
with target_parameters).

    python make_loras.py --model DIR_OR_NAME --rank 1 --n 512 --out DIR [--start 0] [--targets all|attn|mlp]

Writes DIR/0000 ... DIR/<n-1>, each a PEFT adapter folder (save_pretrained). Existing folders are kept."""
import argparse
import os
import shutil
import time

import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]
STD = 0.02                                   # of every adapter value: small enough that the model stays finite


def shell(name: str):
    """The model with its real shapes and no weights read or drawn: PEFT only needs the shapes."""
    cfg = AutoConfig.from_pretrained(name)
    transformers.PreTrainedModel.init_weights = lambda self: None
    try:
        return cfg, AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    except TypeError:
        return cfg, AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--targets", choices=["all", "attn", "mlp"], default="all")
    a = ap.parse_args()
    t0 = time.time()
    cfg, model = shell(a.model)
    moe = cfg.model_type.endswith("_moe")
    kw: dict = {}
    if a.targets in ("all", "attn"):
        kw["target_modules"] = list(ATTN)
    if a.targets in ("all", "mlp"):
        if moe:
            kw["target_parameters"] = ["gate_up_proj", "down_proj"]
        else:
            kw["target_modules"] = kw.get("target_modules", []) + MLP
    pm = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.rank, init_lora_weights=False, **kw),
                        autocast_adapter_dtype=False)
    lora = [(n, p) for n, p in pm.named_parameters() if "lora_" in n]
    values = sum(p.numel() for _, p in lora)
    kinds = sorted({n.split(".lora_")[0].rsplit(".", 1)[-1] for n, _ in lora})
    print(f"{a.model}: {'MoE' if moe else 'dense'}, rank {a.rank}, {len(lora)} adapter tensors, {values / 1e6:.1f}M values, "
          f"on {kinds}, dtype {lora[0][1].dtype}, model shell built in {time.time() - t0:.0f} s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    for i in range(a.start, a.start + a.n):
        d = os.path.join(a.out, f"{i:04d}")
        if os.path.isfile(os.path.join(d, "adapter_model.safetensors")):
            continue
        g = torch.Generator().manual_seed(100_000 * a.rank + i)
        with torch.no_grad():
            for _, p in lora:
                p.copy_((torch.randn(p.shape, generator=g) * STD).to(p.dtype))
        shutil.rmtree(d + ".tmp", ignore_errors=True)
        pm.save_pretrained(d + ".tmp")
        os.replace(d + ".tmp", d)
        if i % 32 == 0 or i == a.start + a.n - 1:
            mb = os.path.getsize(os.path.join(d, "adapter_model.safetensors")) / 1e6
            print(f"adapter {i}: {mb:.1f} MB, {time.time() - t0:.0f} s", flush=True)
    print(f"DONE {a.n} adapters in {a.out}", flush=True)


if __name__ == "__main__":
    main()
