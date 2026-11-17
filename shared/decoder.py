from __future__ import annotations

import os
import re
from dataclasses import dataclass

import torch

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


@dataclass(frozen=True)
class DecoderCfg:
    model: str
    dtype: str = "bf16"
    attn: str = "sdpa"
    experts_impl: str = "grouped_mm"
    trust_remote_code: bool = False
    device: str = "cuda"
    gpus: tuple[int, ...] = ()
    gpu_mem: str = "24GiB"
    tf32: bool = False
    spell_reserved: bool = False


@dataclass
class Decoder:
    cfg: DecoderCfg
    tok: object
    model: torch.nn.Module
    emb: torch.nn.Module
    head: torch.nn.Module
    dtype: torch.dtype
    dev: torch.device
    hdev: torch.device
    d: int
    estd: float


class SpelledTokenizer:
    def __init__(self, tok):
        self._tok = tok
        names = sorted(tok.get_added_vocab(), key=len, reverse=True)
        self._rx = re.compile("(" + "|".join(map(re.escape, names)) + ")") if names else None

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __len__(self) -> int:
        return len(self._tok)

    def encode(self, text: str, add_special_tokens: bool = False, **kw) -> list[int]:
        if self._rx is None or not self._rx.search(text):
            return self._tok.encode(text, add_special_tokens=add_special_tokens, **kw)
        enc = lambda t: self._tok.encode(t, add_special_tokens=False, **kw)
        ids: list[int] = []
        for part in self._rx.split(text):
            if part:
                ids += enc(part[:-1]) + enc(part[-1:]) if self._rx.fullmatch(part) else enc(part)
        return ids


def _max_memory(cfg: DecoderCfg) -> dict:
    if cfg.gpu_mem == "auto":
        return {g: int(0.92 * torch.cuda.get_device_properties(g).total_memory) for g in cfg.gpus}
    return {g: cfg.gpu_mem for g in cfg.gpus}


def load(cfg: DecoderCfg) -> Decoder:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.backends.cudnn.allow_tf32 = cfg.tf32
    tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=cfg.trust_remote_code)
    if cfg.spell_reserved:
        tok = SpelledTokenizer(tok)
    kw: dict = dict(dtype=DTYPES[cfg.dtype], attn_implementation=cfg.attn, experts_implementation=cfg.experts_impl)
    if cfg.gpus:
        kw.update(device_map="balanced", max_memory=_max_memory(cfg))
    model = AutoModelForCausalLM.from_pretrained(cfg.model, **kw)
    if not cfg.gpus:
        model = model.to(cfg.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if cfg.gpus:
        devs = {str(p.device) for p in model.parameters()}
        allowed = {f"cuda:{g}" for g in cfg.gpus}
        if not devs <= allowed:
            raise ValueError(f"decoder weights landed on {devs}, expected {allowed}: raise --gpu-mem or add --gpus")
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    return Decoder(cfg=cfg, tok=tok, model=model, emb=emb, head=head, dtype=DTYPES[cfg.dtype],
                   dev=emb.weight.device, hdev=head.weight.device, d=emb.weight.shape[1],
                   estd=emb.weight.float().std().item())


def load_embedding(cfg: DecoderCfg) -> torch.Tensor:
    from huggingface_hub import file_exists, hf_hub_download
    from safetensors import safe_open
    from shared.fs import read_json
    key, index = "model.embed_tokens.weight", "model.safetensors.index.json"
    if os.path.isdir(cfg.model):
        get, has = (lambda f: os.path.join(cfg.model, f)), (lambda f: os.path.exists(os.path.join(cfg.model, f)))
    else:
        get, has = (lambda f: hf_hub_download(cfg.model, f)), (lambda f: file_exists(cfg.model, f))
    shard = read_json(get(index))["weight_map"][key] if has(index) else "model.safetensors"
    with safe_open(get(shard), framework="pt") as f:
        w = f.get_tensor(key)
    return w.to(torch.bfloat16)


def head_dim(model: torch.nn.Module) -> int:
    c = model.config
    return getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
