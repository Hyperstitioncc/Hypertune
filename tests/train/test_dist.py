import dataclasses
import json
import os
import random
import socket

import pytest
import torch
import torch.multiprocessing as mp

import algorithms.grpo
import algorithms.sft
import shared.backprop
from algorithms.grpo import GRPO, GRPOCfg
from algorithms.sft import LossNorm, SFTCfg, build
from shared import dist
from shared.optim import Optim, OptimCfg
from shared.train import BatchOrder, TrainCfg, train
from shared.types import Completion
from tests.train.conftest import BP, V, Decoder, Env, Sampler, Tok, adapter, bank, identity, logprobs

T = TrainCfg(run="r", steps=6, batch=8, eval_every=3, pick_seed=60, early_stop=False, reward_floor=0.0,
             trip_logratio=False)
O = OptimCfg(lr=0.05, sched_steps=6, max_grad_norm=1.0)


def run(out):
    shared.backprop.logprobs = logprobs
    env = Env()
    bk = bank(env)
    corpus = {p.id: [f"Y{k}{p.id}" for k in range(3)] for p in env.problems("train")}
    ad = adapter(7)
    train(build(Decoder(Tok()), BP, env, bk, corpus, LossNorm.row_mean), ad, env, Optim(ad.params(), O), bk, T, out)
    return ad.module.w.detach().clone()


def worker(r, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(r), LOCAL_RANK=str(r), WORLD_SIZE="2",
                      CUDA_VISIBLE_DEVICES="")  # CPU ranks (gloo), also on a GPU host
    dist.init(dist.DistCfg(dist=True, dist_timeout_min=1))
    w = run(out / "run")
    torch.save(w, out / f"w{r}.pt")
    dist.teardown()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_two_ranks_match_one(tmp_path):
    ref = run(tmp_path / "one")
    mp.spawn(worker, args=(free_port(), tmp_path), nprocs=2, join=True)
    w0, w1 = torch.load(tmp_path / "w0.pt"), torch.load(tmp_path / "w1.pt")
    assert torch.equal(w0, w1)
    assert torch.allclose(w0, ref, atol=1e-9)
    sd = torch.load(tmp_path / "run" / "state.pt", weights_only=False)
    assert sd["step"] == 6 and torch.allclose(sd["adapter"]["state"]["w"], ref, atol=1e-9)
    one = (tmp_path / "one" / "metrics.jsonl").read_text().splitlines()
    two = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert len(one) == len(two) == 6


class PromptSampler(Sampler):
    """Draws seeded by the prompt, not its index in the request, so a rank's shard draws what the full batch would."""

    def sample(self, prompts, stops, s):
        out = []
        for p in prompts:
            rng = random.Random(f"{s.seed}/{p}")
            cs = []
            for _ in range(s.n):
                ids = tuple(rng.randrange(V) for _ in range(rng.randrange(1, 6)))
                cs.append(Completion(tuple(ord(ch) % V for ch in p), ids,
                                     ("Y" if rng.random() < 0.45 else "N") + "".join(map(str, ids)),
                                     tuple(-0.5 - 0.1 * rng.random() for _ in ids)))
            out.append(cs)
        return out


def pen(dec, bp, a, row, kind):
    return (a.module.w ** 2).sum() * len(row.ids)


IRR = [Completion((1, 2), (3, 4, 5), "", ()), Completion((2,), (6, 7), "", ())]
UT = TrainCfg(run="r", steps=7, batch=4, eval_every=3, batch_order=BatchOrder.epoch, pick_seed=60, early_stop=False,
              reward_floor=0.0, trip_logratio=False)
UO = OptimCfg(lr=0.05, sched_steps=7, max_grad_norm=1.0)
# pool 25 over batch 4: the last chunk is one problem, so rank 1 gets none; pool 27: the last chunk splits 2/1. The
# probe is one problem, so rank 1's probe shard is empty.
CASES = {"grpo_empty": ("grpo", 25), "grpo_uneven": ("grpo", 27), "sft_empty": ("sft_row", 25),
         "sft_uneven": ("sft_tok", 27)}


def uneven_run(case, out):
    shared.backprop.logprobs = logprobs
    algorithms.grpo.irr_measure = algorithms.sft.irr_measure = pen
    kind, n = CASES[case]
    env = Env()
    ids = [p.id for p in env.problems("train")]
    bk = dataclasses.replace(bank(env), train=tuple(ids[:n]), probe=(ids[30],))
    ad = adapter(7)
    if kind == "grpo":
        cfg = GRPOCfg(grpo_g=4, grpo_kl_meter=False, grpo_probe_n=4, grpo_adv_perm=True, irr_pen=0.3, irr_n=2)
        algo = GRPO(cfg, env, PromptSampler(ad), Decoder(Tok()), BP, bk, bk.decode, identity(),
                    {p.id: p for p in env.problems("train")}, {}, IRR)
    else:
        corpus = {p.id: [f"Y{k}{p.id}" + "z" * (k * int(p.id[1:]) % 5) for k in range(3)] for p in env.problems("train")}
        algo = build(Decoder(Tok()), BP, env, bk, corpus, LossNorm.row_mean if kind == "sft_row" else LossNorm.token_sum)
        algo.cfg, algo.irr = SFTCfg(sft_corpus="x", irr_pen=0.3, irr_n=1), IRR[:1]  # one item: every rank draws it
    train(algo, ad, env, Optim(ad.params(), UO), bk, UT, out)
    return ad.module.w.detach().clone()


def uneven_worker(r, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(r), LOCAL_RANK=str(r), WORLD_SIZE="2",
                      CUDA_VISIBLE_DEVICES="")  # CPU ranks (gloo), also on a GPU host
    dist.init(dist.DistCfg(dist=True, dist_timeout_min=1))
    for case in CASES:
        torch.save(uneven_run(case, out / case), out / f"{case}_w{r}.pt")
    got = [dist.mean_metrics({"a": 1.0 + r, **({"only0": 5.0} if r == 0 else {})}, weight=3 - 2 * r),
           dist.mean_metrics({"z": float(r)}, weight=0)]
    torch.save(got, out / f"mm{r}.pt")
    dist.teardown()


def metrics(path):
    return [{k: v for k, v in json.loads(ln).items() if k != "time" and not k.startswith("sec")}
            for ln in path.read_text().splitlines()]


def test_uneven_and_empty_shards_match_one_rank(tmp_path, monkeypatch):
    monkeypatch.setattr(algorithms.grpo, "irr_measure", pen)
    monkeypatch.setattr(algorithms.sft, "irr_measure", pen)
    ref = {case: uneven_run(case, tmp_path / "one" / case) for case in CASES}
    (tmp_path / "dp").mkdir()
    mp.spawn(uneven_worker, args=(free_port(), tmp_path / "dp"), nprocs=2, join=True)
    for case in CASES:
        w0, w1 = torch.load(tmp_path / "dp" / f"{case}_w0.pt"), torch.load(tmp_path / "dp" / f"{case}_w1.pt")
        assert torch.equal(w0, w1), case
        assert torch.allclose(w0, ref[case], rtol=0, atol=1e-9), case
        one, two = metrics(tmp_path / "one" / case / "metrics.jsonl"), metrics(tmp_path / "dp" / case / "metrics.jsonl")
        assert len(one) == len(two) == 7
        for a, b in zip(one, two):
            assert set(a) == set(b) and all(b[k] == pytest.approx(a[k], rel=1e-9, abs=1e-12) for k in a), (case, a, b)
    for r in range(2):
        mm = torch.load(tmp_path / "dp" / f"mm{r}.pt")
        assert mm[0] == {"a": pytest.approx((3 * 1.0 + 1 * 2.0) / 4), "only0": 5.0} and mm[1] == {"z": 0.5}
