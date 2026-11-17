from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

from envs.base import Problem
from shared.adapter import Adapter, AdapterKind
from shared.eval import Bank
from shared.types import Backend, Completion, DecodeCfg, Sampling

V = 11


class Bigram(torch.nn.Module):
    def __init__(self, seed: int, train: bool):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w = 0.3 * torch.randn(V, V, generator=g, dtype=torch.float64)
        if train:
            self.w = torch.nn.Parameter(w)
        else:
            self.register_buffer("w", torch.zeros(V, V, dtype=torch.float64))


def adapter(seed: int = 0) -> Adapter:
    return Adapter(AdapterKind.lora, Bigram(seed, True))


def identity() -> Adapter:
    return Adapter(AdapterKind.identity, Bigram(0, False))


def logprobs(decoder, ad: Adapter, rows: list[Completion], cand: list[torch.Tensor], cfg=None) -> list[torch.Tensor]:
    out = []
    for r, c in zip(rows, cand):
        seq = list(r.prompt_ids) + list(r.ids)
        prev = torch.tensor(seq[len(r.prompt_ids) - 1:-1], dtype=torch.long)
        lp = torch.log_softmax(ad.module.w[prev] + 0.1 * prev[:, None].double(), -1)
        out.append(lp.gather(1, c.long()))
    return out


import pytest  # noqa: E402

import shared.backprop  # noqa: E402

BP = shared.backprop.BackpropCfg()


@pytest.fixture(autouse=True)
def _fake_logprobs(monkeypatch):
    monkeypatch.setattr(shared.backprop, "logprobs", logprobs)


class Tok:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) % V for ch in text][:20]

    def __len__(self) -> int:
        return V


@dataclass
class Decoder:
    tok: Tok


class Env:
    name = "fake"
    stops = ("\nQ:",)

    def __init__(self, n: int = 40):
        self._p = [Problem(f"p{i}", f"Q{i}:", "Y") for i in range(n)]

    def problems(self, split: str) -> list[Problem]:
        return list(self._p)

    def grade(self, pairs):
        return [1.0 if t.startswith("Y") else 0.0 for _, t in pairs]


class Sampler:
    backend = Backend.vllm

    def __init__(self, ad: Adapter):
        self.ad = ad
        self.calls: list[Sampling] = []

    def sample(self, prompts, stops, s: Sampling):
        self.calls.append(s)
        bias = float(self.ad.module.w.detach().sum())
        res = []
        for k, p in enumerate(prompts):
            rng = random.Random((s.seed + k) * 7919 + int(1e6 * bias) % 997)
            cs = []
            for g in range(s.n):
                n = rng.randrange(1, 6)
                ids = tuple(rng.randrange(V) for _ in range(n))
                yes = rng.random() < 0.4 + 0.1 * math.tanh(bias)
                cs.append(Completion(prompt_ids=tuple(ord(ch) % V for ch in p), ids=ids,
                                     text=("Y" if yes else "N") + "".join(map(str, ids)),
                                     engine_lp=tuple(-0.5 - 0.1 * rng.random() for _ in ids)))
            res.append(cs)
        return res

    def close(self):
        pass


def bank(env: Env, backend: Backend = Backend.vllm) -> Bank:
    ids = [p.id for p in env.problems("train")]
    rates = {i: 0.05 + 0.4 * (k % 7) / 7 for k, i in enumerate(ids)}
    return Bank(backend=backend, split="train", decode=DecodeCfg(0.8, 0.95, 640), rates=rates,
                train=tuple(ids[:30]), held=tuple(ids[30:]), probe=tuple(ids[30:35]), pool_hi=0.5)
