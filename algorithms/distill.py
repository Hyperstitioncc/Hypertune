from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from envs.base import Env, Problem
from shared import backprop
from envs.mathgrade import cut_at_first_boxed
from shared.types import DecodeCfg
from shared.adapter import Adapter
from shared.backprop import BackpropCfg, encode
from shared.eval import Bank, Sampler, chunk_seed
from shared.fs import atomic_json, read_json
from shared import dist
from shared.types import PartialMean, TokenRow


@dataclass(frozen=True)
class DistillCfg:
    distill_targets: str
    distill_seed: int = 0
    distill_correct_only: bool = False
    distill_dev_rows: int = 96
    distill_nll: float = 0.0
    distill_topk: int = 128
    distill_g: int = 8
    distill_n_dev: int = 40
    distill_max_len: int = 1400


@dataclass(frozen=True)
class TeacherTopK:
    row: TokenRow
    t_idx: torch.Tensor
    t_p: torch.Tensor


def topk_kl(log_q_top: torch.Tensor, t_p: torch.Tensor) -> torch.Tensor:
    q_rest = (1.0 - log_q_top.exp().sum(-1)).clamp_min(1e-9)
    p = t_p.clamp_min(0.0)
    term = (p * (p.clamp_min(1e-12).log() - log_q_top)).sum(-1)
    r = (1.0 - p.sum(-1)).clamp_min(0.0)
    return term + r * (r.clamp_min(1e-12).log() - q_rest.log())


def train_dev_split(ids: list[str], n_dev: int, seed: int) -> tuple[list[str], list[str]]:
    idxs = sorted(ids)
    dev = set(random.Random(seed).sample(idxs, min(n_dev, len(idxs))))
    return [i for i in idxs if i not in dev], sorted(dev)


@dataclass
class Distill:
    decoder: object
    bp: BackpropCfg
    nll_weight: float
    seed: int
    train: list[TeacherTopK]
    dev: list[TeacherTopK]

    def terms(self, adapter: Adapter, t: TeacherTopK) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.tensor(t.row.ids, dtype=torch.long)[:, None]
        lp = backprop.logprobs(self.decoder, adapter, [t.row], [torch.cat([ids, t.t_idx.long()], 1)], self.bp)[0]
        return topk_kl(lp[:, 1:].float(), t.t_p.to(lp.device).float()).sum(), -lp[:, 0].float().sum()

    def step(self, adapter: Adapter, batch: list[Problem], step: int) -> dict[str, float]:
        rows = [random.Random(f"train/{self.seed}/{step}/{p.id}").choice(self.train) for p in batch]
        denom = float(sum(len(t.row.ids) for t in rows))
        kl_tot, nll_tot, n_skip = 0.0, 0.0, 0
        for t in rows:
            kl, nll = self.terms(adapter, t)
            loss = (kl + self.nll_weight * nll) / denom
            if torch.isfinite(loss):
                loss.backward()
            else:
                n_skip += 1
            kl_tot += float(kl)
            nll_tot += float(nll)
        return {"kl": kl_tot / denom, "nll": nll_tot / denom, "n_units": float(len(rows)), "n_skip": float(n_skip)}

    def dev_terms(self, adapter: Adapter, dev: list[TeacherTopK]) -> tuple[float, float, int]:
        skl, snll, ntok = 0.0, 0.0, 0
        with torch.no_grad():
            for t in dev:
                kl, nll = self.terms(adapter, t)
                skl += float(kl)
                snll += float(nll)
                ntok += len(t.row.ids)
        return skl, snll, ntok

    def select(self, adapter: Adapter, step: int) -> PartialMean:
        kl, _, ntok = self.dev_terms(adapter, dist.shard(self.dev))
        return PartialMean(-kl, ntok)


def load_targets(path: Path) -> list[dict]:
    rows = []
    for f in sorted(Path(path).glob("shard_*.pt")):
        rows.extend(torch.load(f, map_location="cpu", weights_only=False))
    return rows


def _target(r: dict) -> TeacherTopK:
    return TeacherTopK(TokenRow(prompt_ids=tuple(r["p_ids"]), ids=tuple(r["c_ids"])),
                  r["t_idx"], r["t_p"])


def make(cfg: DistillCfg, env: Env, sampler, decoder, bp: BackpropCfg, bank: Bank, decode: DecodeCfg, bank_dir: Path,
         data: Path) -> Distill:
    rows = load_targets(Path(data) / cfg.distill_targets)
    tr = [_target(r) for r in rows if r["split"] == "train" and (r["correct"] or not cfg.distill_correct_only)]
    dv = [_target(r) for r in rows if r["split"] == "dev"]
    random.Random(f"dev/{cfg.distill_seed}").shuffle(dv)
    return Distill(decoder, bp, cfg.distill_nll, cfg.distill_seed, tr, dv[:cfg.distill_dev_rows])


def gen(sampler: Sampler, env: Env, bank: Bank, decode: DecodeCfg, cfg: DistillCfg, out: Path) -> None:
    problems = {p.id: p for p in env.problems(bank.split)}
    train, dev = train_dev_split(list(bank.train), cfg.distill_n_dev, cfg.distill_seed)
    recs, n_drop = [], 0
    for k, (name, ids) in enumerate([("train", train), ("dev", dev), ("held", list(bank.held))]):
        ps = [problems[i] for i in ids]
        outs = sampler.sample([p.prompt for p in ps], env.stops,
                              decode.at(cfg.distill_g, chunk_seed(cfg.distill_seed, k)))
        kept = []
        for p, cs in zip(ps, outs):
            for c in cs:
                if not c.text.strip():
                    continue
                t = cut_at_first_boxed(c.text)
                if t is None:
                    n_drop += 1
                    continue
                kept.append((p, t))
        for (p, t), ok in zip(kept, env.grade(kept)):
            recs.append({"rid": len(recs), "id": p.id, "split": name, "text": t, "correct": ok > 0})
    atomic_json({"n_dropped_no_boxed": n_drop, "records": recs}, Path(out) / "teacher_rollouts.json")


def cache(teacher, identity: Adapter, bp: BackpropCfg, env: Env, bank: Bank, cfg: DistillCfg, out: Path) -> None:
    problems = {p.id: p for p in env.problems(bank.split)}
    recs = read_json(Path(out) / "teacher_rollouts.json")["records"]
    V = len(teacher.tok)
    shard, sid, pad_sum, top_sum, n_pos = [], 0, 0.0, 0.0, 0
    for r in recs:
        p_ids, c_ids = encode(teacher, problems[r["id"]].prompt), encode(teacher, r["text"])
        if not c_ids or len(p_ids) + len(c_ids) > cfg.distill_max_len:
            continue
        row = TokenRow(prompt_ids=p_ids, ids=c_ids)
        with torch.no_grad():
            lp = backprop.logprobs(teacher, identity, [row], [torch.arange(V)[None, :].expand(len(c_ids), V)], bp)[0]
            lse = torch.logsumexp(lp.float(), -1, keepdim=True)
            pad_sum += float((1 - lse.exp()).sum())
            tp, ti = torch.topk((lp.float() - lse).exp(), cfg.distill_topk, dim=-1)
        top_sum += float(tp.sum())
        n_pos += tp.shape[0]
        shard.append({"id": r["id"], "split": r["split"], "correct": r["correct"], "p_ids": list(p_ids),
                      "c_ids": list(c_ids), "t_idx": ti.to(torch.int32).cpu(), "t_p": tp.to(torch.float16).cpu()})
        if len(shard) >= 256:
            torch.save(shard, Path(out) / f"shard_{sid:04d}.pt")
            shard, sid = [], sid + 1
    if shard:
        torch.save(shard, Path(out) / f"shard_{sid:04d}.pt")
    atomic_json({"topk": cfg.distill_topk, "real_vocab": V, "n_positions": n_pos,
                 "pad_mass_mean": pad_sum / max(n_pos, 1), "topk_mass_mean": top_sum / max(n_pos, 1)},
                Path(out) / "meta.json")


def main(argv: list[str]) -> None:
    import adapters
    from shared import decoder as dec
    from shared import rollout
    from shared.adapter import AdapterKind
    from envs import env_from
    from shared.cli import RunChoice
    from shared.config import dump, parse, parse_known
    from shared.eval import BankCfg
    from shared.fs import Roots

    if not argv or argv[0] != "targets":
        raise SystemExit("usage: python -m algorithms.distill targets --env NAME --backend B ...")
    (sel,), rest = parse_known(RunChoice, argv=argv[1:])
    env_cls, env_make = env_from(sel.env, rest)
    ecfg, dcfg, rcfg, pcfg, bcfg, decode, cfg, roots = parse(env_cls, dec.DecoderCfg, rollout.RolloutCfg, BackpropCfg,
                                                             BankCfg, DecodeCfg, DistillCfg, Roots, argv=rest)
    out = roots.data / cfg.distill_targets
    out.mkdir(parents=True, exist_ok=True)
    dump([sel, ecfg, dcfg, rcfg, pcfg, bcfg, decode, cfg, roots], out / "config.json")
    env = env_make(ecfg)
    bank = Bank.load(roots.data / bcfg.bank / "bank.json")
    teacher = dec.load(dcfg)
    identity = adapters.build(adapters.AdapterCfg(adapter=AdapterKind.identity), teacher)
    if not (out / "teacher_rollouts.json").exists():
        sampler = rollout.sampler(identity, sel.backend, teacher, rcfg)
        gen(sampler, env, bank, decode, cfg, out)
        sampler.close()
    cache(teacher, identity, pcfg, env, bank, cfg, out)


if __name__ == "__main__":
    main(sys.argv[1:])
