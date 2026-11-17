from __future__ import annotations

import enum
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import adapters
from envs.base import Env, Problem
from shared import backprop
from shared.adapter import Adapter, AdapterKind, partners
from shared.attach import attach
from shared.backprop import BackpropCfg, seq_lp, target
from shared.eval import Bank, Sampler, chunk_seed, graded, log_ratio_per_token, probe_uplift
from shared.fs import read_json
from shared import dist
from shared.types import Backend, Completion, DecodeCfg, PartialMean, TokenRow


class IrrScale(enum.Enum):
    ratio = "ratio"
    coef = "coef"


class IrrMeasure(enum.Enum):
    attn = "attn"
    write = "write"


@dataclass(frozen=True)
class GRPOCfg:
    grpo_g: int = 8
    grpo_seed: int = 1500
    grpo_pos_only: bool = False
    grpo_neg_scale: float = 1.0
    grpo_pos_scale: float = 1.0
    grpo_adv_perm: bool = False
    grpo_mask_trunc: bool = False
    grpo_tis: bool = False
    grpo_tis_cap: float = 2.0
    grpo_kl_beta: float = 0.0
    grpo_kl_meter: bool = True
    grpo_kl_every: int = 5
    grpo_kl_rows: int = 16
    grpo_anchor: float = 0.0
    grpo_anchor_n: int = 8
    grpo_corpus: str = "correct.json"
    irr_pen: float = 0.0
    irr_mode: IrrScale = IrrScale.ratio
    irr_kind: IrrMeasure = IrrMeasure.write
    irr_n: int = 8
    irr_resp: int = 512
    irr_items: str | None = None
    grpo_probe_n: int = 16
    grpo_probe_seed: int = 0
    grpo_probe_fixed: bool = False

    def __post_init__(self):
        if (self.irr_pen > 0 or self.irr_items is not None) and self.irr_n < 1:
            raise ValueError(f"--irr-n must be >= 1 with an irr penalty, got {self.irr_n}")


PROMPT_GRAD_TOKENS = 2048


def chunks(sizes: list[int], budget: int) -> list[list[int]]:
    """Consecutive index runs whose sizes sum to at most budget (a single larger item gets a run of its own)."""
    out, cur, n = [], [], 0
    for i, k in enumerate(sizes):
        if cur and n + k > budget:
            out.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += k
    return out + ([cur] if cur else [])


def tis_rho(s: torch.Tensor, rows: list[Completion], cap: float) -> torch.Tensor:
    ref = torch.tensor([sum(r.engine_lp) for r in rows], dtype=s.dtype, device=s.device)
    L = torch.tensor([len(r.ids) for r in rows], dtype=s.dtype, device=s.device)
    return torch.exp((s.detach() - ref) / L).clamp(max=cap)


def k2(s: torch.Tensor, sb: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    return (0.5 * (s - sb) ** 2 / L).sum()


def pg_loss(s: torch.Tensor, weights: torch.Tensor, rho: torch.Tensor, kl_beta: float, sb: torch.Tensor,
            L: torch.Tensor) -> torch.Tensor:
    loss = -(weights * rho * s).sum()
    return loss + kl_beta * k2(s, sb, L) if kl_beta > 0 else loss


def advantages(rewards: list[float], trunc: list[bool], cfg: GRPOCfg, step: int, j: int) -> list[tuple[int, float]]:
    G = len(rewards)
    if not 0 < sum(rewards) < G:
        return []
    mu = sum(rewards) / G
    rw = list(rewards)
    if cfg.grpo_adv_perm:
        random.Random(f"perm/{cfg.grpo_seed}/{step}/{j}").shuffle(rw)
    out = []
    for k, (rew, tr) in enumerate(zip(rw, trunc)):
        a = rew - mu
        if cfg.grpo_pos_only and a <= 0:
            continue
        if cfg.grpo_mask_trunc and tr:
            continue
        if a < 0:
            a *= cfg.grpo_neg_scale
        elif a > 0:
            if cfg.grpo_pos_scale == 0:
                continue
            a *= cfg.grpo_pos_scale
        out.append((k, a))
    return out


def irr_scale(cfg: GRPOCfg, n_grpo: float, n_pen: float) -> float:
    return pen_scale(cfg.irr_pen, cfg.irr_mode, n_grpo, n_pen)


def pen_scale(pen: float, mode: IrrScale, n_main: float, n_pen: float) -> float:
    if mode is IrrScale.ratio:
        return pen * n_main / n_pen if n_pen > 0 else 0.0
    return pen


def irr_measure(decoder, bp: BackpropCfg, adapter: Adapter, row: TokenRow, kind: IrrMeasure) -> torch.Tensor:
    e = attach(adapter, Backend.hf_eager, decoder)
    probe = e.attn_write if kind is IrrMeasure.write else e.attn_mass
    if probe is None:
        raise ValueError(f"no {kind.value} probe for a {adapter.kind.value} adapter")
    with probe(len(row.prompt_ids)) as acc:
        backprop.logprobs(decoder, adapter, [row], [torch.tensor(row.ids)[:, None]], bp)
    assert acc, "irr penalty forward saw no response queries"
    return torch.stack(acc).mean()


def add_penalty(params: list[torch.Tensor], backward_pen, pen: float, mode: IrrScale, main: str) -> dict[str, float]:
    n1 = grads_norm(params)
    g1 = [None if p.grad is None else p.grad.detach().clone() for p in params]
    for p in params:
        p.grad = None
    val = backward_pen()
    n2 = grads_norm(params)
    sc = pen_scale(pen, mode, n1, n2)
    for p, g in zip(params, g1):
        if p.grad is not None:
            p.grad.mul_(sc)
        if g is not None:
            p.grad = g if p.grad is None else p.grad.add_(g)
    return {"irr": val, f"irr_g_{main}": n1, "irr_g_pen": n2, "irr_scale": sc}


def grads_norm(params: list[torch.Tensor]) -> float:
    return math.sqrt(sum(float(p.grad.double().pow(2).sum()) for p in params if p.grad is not None))


@dataclass
class GRPO:
    cfg: GRPOCfg
    env: Env
    sampler: Sampler
    decoder: object
    bp: BackpropCfg
    bank: Bank
    decode: DecodeCfg
    base: Adapter
    problems: dict[str, Problem]
    corpus: dict[str, list[str]]
    irr: list[Completion]

    def backward(self, adapter: Adapter, rows: list[TokenRow], weights: list[float], tis: bool,
                 kl_beta: float) -> tuple[float, bool, list[float]]:
        """pg_loss is a sum over rows, so one backward per chunk of rows (at most one scoring unit of tokens) gives the
        group's gradient with one chunk's graph in memory at a time. An adapter that changes the prompt (LoRA) keeps a
        graph through the prompt too, so its chunks are capped at PROMPT_GRAD_TOKENS. A non-finite chunk drops the
        whole group: the gradients from before it are put back."""
        params = adapter.params()
        before = [None if p.grad is None else p.grad.clone() for p in params]
        budget = self.bp.unit_tokens
        if adapter.kind is not AdapterKind.kv_prefix:  # only a cartridge never needs a graph through the prompt
            budget = min(budget, PROMPT_GRAD_TOKENS)
        loss, rho = 0.0, []
        for idx in chunks([len(r.ids) + 1 for r in rows], budget):
            sub, wsub = [rows[i] for i in idx], [weights[i] for i in idx]
            s = seq_lp(self.decoder, adapter, sub, self.bp)
            w = torch.tensor(wsub, dtype=s.dtype, device=s.device)
            L = torch.tensor([len(r.ids) for r in sub], dtype=s.dtype, device=s.device)
            r = tis_rho(s, sub, self.cfg.grpo_tis_cap) if tis else torch.ones_like(w)
            rho += r.tolist() if tis else []
            sb = s.detach()
            if kl_beta > 0:
                with torch.no_grad():
                    sb = seq_lp(self.decoder, self.base, sub, self.bp)
            term = pg_loss(s, w, r, kl_beta, sb, L)
            if not torch.isfinite(term):
                for p, g in zip(params, before):
                    p.grad = g
                return 0.0, True, rho
            term.backward()
            loss += float(term.detach())
        return loss, False, rho

    def logratio(self, adapter: Adapter, groups: list[list[Completion]]) -> float:
        rows: list[Completion] = []
        for comps in groups:
            rows += [c for c in comps[:4] if c.ids]
            if len(rows) >= self.cfg.grpo_kl_rows:
                break
        return log_ratio_per_token(self.decoder, adapter, self.base, rows[:self.cfg.grpo_kl_rows], self.bp)

    def anchor_rows(self, batch: list[Problem], step: int) -> list[TokenRow]:
        rng = random.Random(f"anchor/{self.cfg.grpo_seed}/{step}")
        n = self.cfg.grpo_anchor_n
        cand = [p.id for p in batch if self.corpus.get(p.id)]
        if len(cand) < n:
            pool = self.bank.pool
            cand += rng.sample(pool, min(len(pool), 4 * n))
            cand = [i for i in dict.fromkeys(cand) if self.corpus.get(i)]
        rows = []
        for i in rng.sample(cand, min(n, len(cand))):
            txt = rng.choice(self.corpus[i])
            r = target(self.decoder, self.problems[i].prompt, txt, self.bank.decode.max_tokens)
            if r.ids:
                rows.append(r)
        return rows

    def irr_one(self, adapter: Adapter, row: Completion) -> torch.Tensor:
        return irr_measure(self.decoder, self.bp, adapter, row, self.cfg.irr_kind)

    def irr_backward(self, adapter: Adapter, step: int) -> float:
        items = random.Random(f"irr/{self.cfg.grpo_seed}/{step}").sample(self.irr, min(self.cfg.irr_n, len(self.irr)))
        tot = 0.0
        for row in items:
            pen = self.irr_one(adapter, row)
            (pen / len(items)).backward()
            tot += float(pen.detach())
        return tot / len(items)

    def irr_apply(self, adapter: Adapter, step: int) -> dict[str, float]:
        with partners(adapter, None):
            return add_penalty(adapter.params(), lambda: self.irr_backward(adapter, step), self.cfg.irr_pen,
                               self.cfg.irr_mode, "grpo")

    def select(self, adapter: Adapter, step: int) -> PartialMean:
        c = self.cfg
        probe = [self.problems[i] for i in dist.shard(self.bank.probe)]
        if not probe:
            return PartialMean(0.0, 0)
        seed = chunk_seed(c.grpo_probe_seed, 0 if c.grpo_probe_fixed else step)
        return probe_uplift(self.sampler, self.env, probe, self.bank, c.grpo_probe_n, seed)

    def step(self, adapter: Adapter, batch: list[Problem], step: int) -> dict[str, float]:
        with partners(adapter, step):
            return self.rollout_step(adapter, batch, step)

    def rollout_step(self, adapter: Adapter, batch: list[Problem], step: int) -> dict[str, float]:
        c = self.cfg
        G = c.grpo_g
        norm = float(self.bank.decode.max_tokens)
        s = self.decode.at(G, chunk_seed(c.grpo_seed, step))
        t0 = time.time()
        groups = graded(self.sampler, self.env, batch, s) if batch else []
        t1 = time.time()
        n_rows, loss, n_units, n_skip, n_mixed, n_trunc = 0, 0.0, 0, 0, 0, 0
        nll_s, nll_n, rho_log, rew_tot = 0.0, 0, [], 0.0
        for j, grp in enumerate(groups):
            comps = [x for x, _ in grp]
            rewards = [r for _, r in grp]
            rew_tot += sum(rewards)
            for x in comps:
                if len(x.engine_lp):
                    nll_s += -sum(x.engine_lp) / len(x.engine_lp)
                    nll_n += 1
            trunc = [len(x.ids) >= s.max_tokens for x in comps]
            n_trunc += sum(trunc)
            adv = advantages(rewards, trunc, c, step, j * dist.world() + dist.rank())  # the problem's index in the batch
            n_mixed += bool(0 < sum(rewards) < G)
            rows = [comps[k] for k, _ in adv if comps[k].ids]
            advs = [a for k, a in adv if comps[k].ids]
            if rows:
                n_rows += len(rows)
                lv, skipped, rho = self.backward(adapter, rows, [a / norm for a in advs], c.grpo_tis, c.grpo_kl_beta)
                n_units += 1
                n_skip += skipped
                loss += lv
                rho_log += rho
        m: dict[str, float] = {"sec_gen": t1 - t0, "sec_score": time.time() - t1}
        if c.grpo_anchor > 0:
            a_rows = self.anchor_rows(batch, step)
            if a_rows:
                lv, skipped, _ = self.backward(adapter, a_rows, [c.grpo_anchor / norm] * len(a_rows), False, 0.0)
                n_rows += len(a_rows)
                n_units += 1
                n_skip += skipped
                m["anchor_lp"] = -lv / max(1, sum(len(r.ids) for r in a_rows)) * norm / c.grpo_anchor
        n_comp, rho_s, rho_c = len(groups) * G, sum(rho_log), sum(x >= c.grpo_tis_cap - 1e-6 for x in rho_log)
        (n_rows, loss, n_units, n_skip, n_mixed, n_trunc, nll_s, nll_n, rew_tot, n_comp, rho_s, rho_c, rho_n) = dist.sum_vec(
            [n_rows, loss, n_units, n_skip, n_mixed, n_trunc, nll_s, nll_n, rew_tot, n_comp, rho_s, rho_c, len(rho_log)])
        if n_rows:  # over every rank; all_reduce_grads then averages the ranks, so each holds its share times world
            for p in adapter.params():
                if p.grad is not None:
                    p.grad.div_(n_rows / dist.world())
            loss /= n_rows
            if self.irr:
                dist.all_reduce_grads(adapter.params())  # ratio mode scales the penalty to the batch's gradient
                t2 = time.time()
                m.update(self.irr_apply(adapter, step))
                m["sec_pen"] = time.time() - t2
        if c.grpo_kl_meter and (step + 1) % c.grpo_kl_every == 0 and groups:
            m["logratio"] = self.logratio(adapter, [[x for x, _ in grp] for grp in groups])
        m.update(reward=rew_tot / max(1.0, n_comp), mixed=n_mixed, rows=n_rows, loss=loss,
                 nll=nll_s / nll_n if nll_n else math.nan, trunc=n_trunc / max(1.0, n_comp),
                 n_units=n_units, n_skip=n_skip)
        if rho_n:
            m["rho_mean"] = rho_s / rho_n
            m["rho_capped_frac"] = rho_c / rho_n
        return m


def load_irr(decoder, path: Path, resp_tokens: int) -> list[Completion]:
    out = []
    for it in read_json(path):
        r = target(decoder, it["prompt"], it["response"], resp_tokens)
        if r.ids:
            out.append(r)
    return out


def make(cfg: GRPOCfg, env: Env, sampler: Sampler, decoder, bp: BackpropCfg, bank: Bank, decode: DecodeCfg, bank_dir: Path,
         data: Path) -> GRPO:
    base = adapters.build(adapters.AdapterCfg(adapter=AdapterKind.identity), decoder)
    corpus = read_json(Path(bank_dir) / cfg.grpo_corpus) if cfg.grpo_anchor > 0 else {}
    irr = load_irr(decoder, Path(data) / cfg.irr_items, cfg.irr_resp) if cfg.irr_items is not None else []
    problems = {p.id: p for p in env.problems(bank.split)}
    return GRPO(cfg, env, sampler, decoder, bp, bank, decode, base, problems, corpus, irr)
