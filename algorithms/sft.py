from __future__ import annotations

import enum
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from envs.base import Env, Problem
from shared.adapter import Adapter, partners
from shared.backprop import BackpropCfg, seq_lp, target
from shared.eval import Bank, Sampler, draw
from shared.fs import read_json
from shared import dist
from shared.types import DecodeCfg, PartialMean, TokenRow

from algorithms.grpo import IrrMeasure, IrrScale, add_penalty, irr_measure, load_irr


class LossNorm(enum.Enum):
    row_mean = "row_mean"
    token_sum = "token_sum"


class SFTSelect(enum.Enum):
    logprob = "logprob"
    grade = "grade"


@dataclass(frozen=True)
class SFTCfg:
    sft_corpus: str
    sft_norm: LossNorm = LossNorm.row_mean
    sft_eos: bool = False
    sft_select: SFTSelect = SFTSelect.logprob
    sft_select_seed: int = 0
    sft_max_len: int = 0  # drop (not cut) targets longer than this many tokens; 0 keeps every target
    irr_pen: float = 0.0
    irr_mode: IrrScale = IrrScale.ratio
    irr_kind: IrrMeasure = IrrMeasure.write
    irr_n: int = 8
    irr_resp: int = 128
    irr_items: str | None = None
    irr_seed: int = 0

    def __post_init__(self):
        if self.irr_pen > 0 and self.irr_n < 1:
            raise ValueError(f"--irr-n must be >= 1 with --irr-pen > 0, got {self.irr_n}")


@dataclass(frozen=True)
class GradedProbe:
    sampler: Sampler
    env: Env
    problems: list[Problem]
    decode: DecodeCfg
    seed: int

    def mean(self) -> PartialMean:
        mine = dist.shard(self.problems)
        if not mine:
            return PartialMean(0.0, 0)
        rates = draw(self.sampler, self.env, mine, self.decode.at(1, self.seed))
        return PartialMean(sum(sum(g) / len(g) for g in rates.values()), len(rates))


def clone_weights(lengths: list[int], norm: LossNorm, total: tuple[float, float] | None = None) -> list[float]:
    """total: the (rows, tokens) the loss is normalised by; default this batch's own."""
    rows, toks = total or (len(lengths), sum(lengths))
    if norm is LossNorm.row_mean:
        return [1.0 / (n * rows) for n in lengths]
    return [1.0 / toks] * len(lengths)


def clone_loss(s: torch.Tensor, lengths: list[int], norm: LossNorm) -> torch.Tensor:
    return -(torch.tensor(clone_weights(lengths, norm), dtype=s.dtype, device=s.device) * s).sum()


def clone_step(decoder, adapter: Adapter, rows: list[TokenRow], bp: BackpropCfg, norm: LossNorm) -> dict[str, float]:
    """One backward per row: the rows' loss terms are separate, so this is the batch gradient with one row's graph in
    memory at a time. A non-finite row drops the whole step on every rank, as a single batch backward would. Under data
    parallel the loss is normalised by the counts over all ranks, divided by world as all_reduce_grads averages."""
    lengths = [len(r.ids) for r in rows]
    n, toks = dist.sum_vec([len(rows), sum(lengths)])
    loss, bad = 0.0, False
    for r, w in zip(rows, clone_weights(lengths, norm, (n / dist.world(), toks / dist.world()))):
        term = -w * seq_lp(decoder, adapter, [r], bp)[0]
        if not torch.isfinite(term):
            bad = True
            break
        term.backward()
        loss += float(term.detach())
    bad, loss = dist.sum_vec([bad, loss])
    if bad:
        for p in adapter.params():
            p.grad = None
        return {"loss": 0.0, "rows": n, "n_units": 1.0, "n_skip": 1.0}
    return {"loss": loss / dist.world(), "rows": n, "n_units": 1.0, "n_skip": 0.0}


def held_lp(decoder, adapter: Adapter, rows: list[TokenRow], bp: BackpropCfg, norm: LossNorm) -> PartialMean:
    if not rows:
        return PartialMean(0.0, 0)
    with torch.no_grad():
        s = [float(x) for x in seq_lp(decoder, adapter, rows, bp)]
    if norm is LossNorm.row_mean:
        return PartialMean(sum(x / len(r.ids) for x, r in zip(s, rows)), len(rows))
    return PartialMean(sum(s), sum(len(r.ids) for r in rows))


@dataclass
class SFT:
    decoder: object
    bp: BackpropCfg
    norm: LossNorm
    targets: dict[str, list[TokenRow]]
    held_rows: list[TokenRow]
    probe: GradedProbe | None = None
    cfg: SFTCfg | None = None
    irr: list[TokenRow] = field(default_factory=list)

    def step(self, adapter: Adapter, batch: list[Problem], step: int) -> dict[str, float]:
        rows = []
        for p in batch:
            ts = self.targets[p.id]
            rows.append(ts[step % len(ts)])
        t0 = time.time()
        with partners(adapter, step):
            m = clone_step(self.decoder, adapter, rows, self.bp, self.norm)
        m["sec_fit"] = time.time() - t0
        if self.irr and not m["n_skip"]:
            t1 = time.time()
            dist.all_reduce_grads(adapter.params())  # ratio mode scales the penalty to the batch's gradient
            m.update(add_penalty(adapter.params(), lambda: self.irr_backward(adapter, step), self.cfg.irr_pen,
                                 self.cfg.irr_mode, "sft"))
            m["sec_pen"] = time.time() - t1
        return m

    def irr_backward(self, adapter: Adapter, step: int) -> float:
        c = self.cfg
        items = random.Random(f"irr/{c.irr_seed}/{dist.rank()}/{step}").sample(self.irr, min(c.irr_n, len(self.irr)))
        tot = 0.0
        for row in items:
            pen = irr_measure(self.decoder, self.bp, adapter, row, c.irr_kind)
            (pen / len(items)).backward()
            tot += float(pen.detach())
        return tot / len(items)

    def trainable(self, pid: str) -> bool:
        return pid in self.targets

    def select(self, adapter: Adapter, step: int) -> PartialMean:
        if self.probe is not None:
            return self.probe.mean()
        return held_lp(self.decoder, adapter, dist.shard(self.held_rows), self.bp, LossNorm.token_sum)


def build(decoder, bp: BackpropCfg, env: Env, bank: Bank, corpus: dict[str, list[str]], norm: LossNorm, max_len: int = 0,
          eos: bool = False) -> SFT:
    problems = {p.id: p for p in env.problems(bank.split)}
    end = (decoder.tok.eos_token_id,) if eos else ()
    cap = bank.decode.max_tokens
    every, targets = {}, {}
    for i, texts in corpus.items():
        if i not in problems:
            continue
        ts = [target(decoder, problems[i].prompt, t, cap + 1) for t in texts]  # one over the cap tells a cut target
        ts = [(TokenRow(t.prompt_ids, t.ids[:cap] + (end if len(t.ids) <= cap else ())), min(len(t.ids), cap))
              for t in ts if t.ids]  # a cut target does not end, so it gets no EOS
        if ts:
            every[i] = [t for t, _ in ts]
        ts = [t for t, n in ts if not (max_len and n > max_len)]
        if ts:
            targets[i] = ts
    held = [every[i][0] for i in bank.probe if i in every]  # scored without grad: the length cap is for training
    if not held:
        raise ValueError("no probe problem has a target in the corpus: nothing to select on")
    return SFT(decoder, bp, norm, targets, held)


def make(cfg: SFTCfg, env: Env, sampler, decoder, bp: BackpropCfg, bank: Bank, decode: DecodeCfg, bank_dir: Path, data: Path) -> SFT:
    sft = build(decoder, bp, env, bank, read_json(Path(data) / cfg.sft_corpus), cfg.sft_norm, cfg.sft_max_len, cfg.sft_eos)
    if cfg.sft_select is SFTSelect.grade:
        problems = {p.id: p for p in env.problems(bank.split)}
        sft.probe = GradedProbe(sampler, env, [problems[i] for i in bank.probe], bank.decode, cfg.sft_select_seed)
    if cfg.irr_pen > 0:
        if cfg.irr_items is None:
            raise ValueError("--irr-pen needs --irr-items: the off-task prompts the adapter must stay quiet on")
        sft.cfg, sft.irr = cfg, load_irr(decoder, Path(data) / cfg.irr_items, cfg.irr_resp)
    return sft
