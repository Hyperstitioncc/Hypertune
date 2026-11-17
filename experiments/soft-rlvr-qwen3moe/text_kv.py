from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from adapters.soft_block import random_rows
from shared import decoder as dec
from envs import env_from
from shared.config import parse, parse_known
from shared.eval import Bank
from shared.fs import Roots, atomic_torch


@dataclass(frozen=True)
class TextKVCfg:
    env: str
    bank: Path
    kv_out: Path
    k: int = 16
    idx: int = 0
    init_seed: int = 0


def main(argv: list[str]) -> None:
    (c,), rest = parse_known(TextKVCfg, argv=argv)
    env_cls, env_make = env_from(c.env, rest)
    ecfg, dcfg, roots = parse(env_cls, dec.DecoderCfg, Roots, argv=rest)
    env = env_make(ecfg)
    bank = Bank.load(roots.data / c.bank / "bank.json")
    prob = {p.id: p for p in env.problems(bank.split)}[bank.train[c.idx]]
    d = dec.load(dcfg)
    cfg = d.model.config
    L, Hkv, hd, K = cfg.num_hidden_layers, cfg.num_key_value_heads, dec.head_dim(d.model), c.k
    ids = d.tok.encode(prob.prompt, add_special_tokens=False)
    T = len(ids)
    if T <= K:
        raise ValueError(f"--k {K} needs a prompt longer than {K} tokens; problem {prob.id} has {T}")
    cap = {}
    for li, layer in enumerate(d.model.model.layers):
        layer.self_attn.k_norm.register_forward_hook(lambda m, a, o, li=li: cap.__setitem__(("k", li), o))
        layer.self_attn.v_proj.register_forward_hook(lambda m, a, o, li=li: cap.__setitem__(("v", li), o))
    with torch.no_grad():
        d.model.model(input_ids=torch.tensor([ids], device=d.dev),
                      position_ids=torch.arange(T, device=d.dev)[None], use_cache=False)
    kf = torch.stack([cap["k", li][0, T - K:T].float().reshape(K, Hkv, hd) for li in range(L)]).cpu()
    vf = torch.stack([cap["v", li][0, T - K:T].float().reshape(K, Hkv, hd) for li in range(L)]).cpu()
    soft = random_rows(K, d.d, d.estd, 1.0, c.init_seed, torch.device("cpu"))
    atomic_torch({"soft": soft, "kfix": kf, "vfix": vf}, roots.data / c.kv_out)


if __name__ == "__main__":
    main(sys.argv[1:])
