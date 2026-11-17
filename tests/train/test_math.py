import dataclasses
import math
from pathlib import Path

import pytest
import torch

from algorithms.distill import topk_kl, train_dev_split
from algorithms.grpo import GRPOCfg, advantages, k2
from algorithms.sft import LossNorm, clone_weights
from shared.decoder import DecoderCfg
from shared.eval import Bank, chunk_seed, evaluate, paired, verdict
from shared.optim import Optim, OptimCfg, Schedule
from shared.types import DecodeCfg, Backend, Sampling
from tests.train.conftest import Env, Sampler, adapter, bank


def test_advantages_dr_grpo():
    adv = advantages([1.0, 0.0, 0.0, 1.0], [False] * 4, GRPOCfg(), 0, 0)
    assert adv == [(0, 0.5), (1, -0.5), (2, -0.5), (3, 0.5)]
    assert advantages([1.0] * 4, [False] * 4, GRPOCfg(), 0, 0) == []
    assert advantages([1.0, 0.0, 0.0, 0.0], [False] * 4, GRPOCfg(grpo_pos_only=True), 0, 0) == [(0, 0.75)]
    a = advantages([1.0, 0.0, 0.0, 0.0], [False, True, False, False],
                   GRPOCfg(grpo_mask_trunc=True, grpo_neg_scale=0.5), 0, 0)
    assert a == [(0, 0.75), (2, -0.125), (3, -0.125)]
    assert advantages([1.0, 0.0], [False] * 2, GRPOCfg(grpo_pos_scale=0.0), 0, 0) == [(1, -0.5)]
    p = advantages([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [False] * 6, GRPOCfg(grpo_adv_perm=True), 3, 2)
    assert sorted(a for _, a in p) == pytest.approx(sorted([2 / 3, -1 / 3, -1 / 3, -1 / 3, 2 / 3, -1 / 3]))


def test_k2():
    s, sb, L = torch.tensor([-3.0, -1.0]), torch.tensor([-2.0, -1.5]), torch.tensor([2.0, 1.0])
    assert abs(float(k2(s, sb, L)) - (0.5 * 1 / 2 + 0.5 * 0.25 / 1)) < 1e-7




def test_clone_weights():
    assert clone_weights([2, 4], LossNorm.row_mean) == [0.25, 0.125]
    assert clone_weights([2, 4], LossNorm.token_sum) == [1 / 6, 1 / 6]


def test_topk_kl_exact_when_k_is_vocab():
    p = torch.softmax(torch.tensor([[0.3, 1.0, -0.5, 2.0]]), -1)
    q = torch.log_softmax(torch.tensor([[0.1, 0.2, 0.3, 0.4]]), -1)
    full = (p * (p.log() - q)).sum(-1)
    idx = torch.tensor([[3, 1]])
    tp = p.gather(1, idx)
    coarse = topk_kl(q.gather(1, idx), tp)
    assert float(coarse) <= float(full) + 1e-7
    assert torch.allclose(topk_kl(q, p), full, atol=1e-6)


def test_train_dev_split():
    tr, dv = train_dev_split([f"p{i}" for i in range(50)], 10, 7)
    assert len(dv) == 10 and not set(tr) & set(dv) and len(tr) == 40


def test_paired_refuses_cross_backend():
    env = Env()
    bk = bank(env, Backend.vllm)
    rates = {i: bk.rates[i] + 0.1 for i in bk.held}
    with pytest.raises(ValueError):
        paired(rates, Backend.hf_eager, bk.decode, bk)
    with pytest.raises(ValueError):
        paired(rates, Backend.vllm, dataclasses.replace(bk.decode, max_tokens=2048), bk)
    up, se = paired(rates, Backend.vllm, bk.decode, bk)
    assert abs(up - 0.1) < 1e-12 and se < 1e-12
    ups = [0.1, 0.3, -0.2, 0.0, 0.4]
    r = {i: bk.rates[i] + u for i, u in zip(bk.held, ups)}
    m = sum(ups) / 5
    sd = math.sqrt(sum((u - m) ** 2 for u in ups) / 4)
    up, se = paired(r, Backend.vllm, bk.decode, bk)
    assert abs(up - m) < 1e-12 and abs(se - sd / math.sqrt(5)) < 1e-12


def test_bank_roundtrip(tmp_path):
    bk = bank(Env())
    bk.save(tmp_path / "bank.json")
    assert Bank.load(tmp_path / "bank.json") == bk
    assert set(bk.pool) <= set(bk.train) and all(0 < bk.rates[i] < 0.5 for i in bk.pool)
    assert bk.clean == [i for i in bk.held if i not in bk.probe]


def test_evaluate_chunks_seeds_resume(tmp_path):
    env = Env()
    s = Sampler(adapter(0))
    ps = env.problems("train")[:5]
    sm = Sampling(n=4, temperature=0.8, top_p=0.95, max_tokens=640, seed=6400)
    g = evaluate(s, env, ps, sm, 3, tmp_path)
    assert [c.seed for c in s.calls] == [chunk_seed(6400, c) for c in range(3)]
    assert all(len(v) == 12 for v in g.values())
    s2 = Sampler(adapter(0))
    assert evaluate(s2, env, ps, sm, 3, tmp_path) == g and s2.calls == []
    with pytest.raises(ValueError):
        evaluate(s2, env, ps, Sampling(n=8, temperature=0.8, top_p=0.95, max_tokens=640, seed=6400), 3, tmp_path)
    bk = bank(env)
    v = verdict({i: x for i, x in g.items()}, Backend.vllm, bk.decode, bk)
    assert v["n_per_problem"] == 12 and v["paired_with_bank"] and "mean_uplift" in v
    v2 = verdict({i: x for i, x in g.items()}, Backend.vllm, dataclasses.replace(bk.decode, max_tokens=2048), bk)
    assert not v2["paired_with_bank"] and "mean_uplift" not in v2 and v2["absolute_rate"] == v["absolute_rate"]


def test_build_bank(tmp_path):
    from shared.eval import BankCfg, build_bank
    env = Env(200)
    cfg = BankCfg(bank=tmp_path, cal_n=16, n_chunks=3, n_per_chunk=32, min_pool=5)
    ident = (DecoderCfg(model="m1"),)
    bk = build_bank(Sampler(adapter(0)), env, cfg, DecodeCfg(), tmp_path, ident)
    assert bk.backend is Backend.vllm and set(bk.probe) <= set(bk.held) and not set(bk.held) & set(bk.train)
    assert len(bk.clean) == len(bk.held) - len(bk.probe) >= min(cfg.held_min, len(bk.train))
    assert Bank.load(tmp_path / "bank.json") == bk
    calls = Sampler(adapter(0))
    assert build_bank(calls, env, cfg, DecodeCfg(), tmp_path, ident) == bk and calls.calls == []
    with pytest.raises(SystemExit):
        build_bank(calls, env, cfg, DecodeCfg(), tmp_path, (DecoderCfg(model="m2"),))
    with pytest.raises(SystemExit):
        build_bank(calls, env, cfg, DecodeCfg(max_tokens=2048), tmp_path, ident)


def test_build_bank_select_all_keeps_dataset_order(tmp_path):
    from shared.eval import BankCfg, BankSelect, build_bank
    env = Env(200)
    cfg = BankCfg(bank=tmp_path, select=BankSelect.all, cal_n=16, n_chunks=3, n_per_chunk=32, min_pool=5)
    s = Sampler(adapter(0))
    build_bank(s, env, cfg, DecodeCfg(), tmp_path, ())
    from shared.fs import read_json
    assert read_json(tmp_path / "selected.json")["selected"] == [p.id for p in env.problems("train")]
    assert all(c.n == 32 for c in s.calls)


def _cos(t, T):
    return 0.5 * (1 + math.cos(math.pi * min(1.0, t / T)))


def test_cosine_matches_torch():
    p = torch.nn.Parameter(torch.zeros(3))
    o = Optim([p], OptimCfg(lr=0.05, sched_steps=300))
    ref_opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(3))], lr=0.05)
    ref = torch.optim.lr_scheduler.CosineAnnealingLR(ref_opt, T_max=300)
    for t in range(300):
        assert abs(o.lr() - ref.get_last_lr()[0]) < 1e-12
        p.grad = torch.ones(3)
        o.step()
        ref_opt.step()
        ref.step()
    assert o.lr() < 1e-6


def test_warmup_cosine_matches_sft_train_lambda():
    p = torch.nn.Parameter(torch.zeros(2))
    o = Optim([p], OptimCfg(lr=1e-5, warmup=20, sched_steps=100, adam_beta2=0.95, weight_decay=0.0))
    for t in range(120):
        want = 1e-5 * min(1.0, (t + 1) / 20) * _cos(t, 100)
        assert abs(o.lr() - want) < 1e-18
        o.step()


def test_const_schedule_and_no_grad_step_advances():
    p = torch.nn.Parameter(torch.zeros(2))
    o = Optim([p], OptimCfg(lr=0.1, schedule=Schedule.const))
    r = o.step()
    assert r["skipped"] == 1.0 and o.t == 1 and o.lr() == 0.1


def test_clip_rel_median_and_abs():
    p = torch.nn.Parameter(torch.zeros(1))
    o = Optim([p], OptimCfg(lr=0.0, clip_rel=2.0, clip_warmup=3, max_grad_norm=100.0))
    for g in (1.0, 3.0, 2.0):
        p.grad = torch.tensor([g])
        assert o.step()["clipped"] == 0.0
    p.grad = torch.tensor([10.0])
    r = o.step()
    assert r["clipped"] == 1.0 and abs(float(p.grad) - 4.0) < 1e-6 and r["grad_norm"] == 10.0


def test_nonfinite_skip_vs_raise():
    p = torch.nn.Parameter(torch.zeros(1))
    o = Optim([p], OptimCfg(skip_nonfinite=True))
    p.grad = torch.tensor([float("nan")])
    assert o.step()["skipped"] == 1.0 and float(p.detach()) == 0.0 and o.t == 1
    o2 = Optim([p], OptimCfg())
    p.grad = torch.tensor([float("nan")])
    with pytest.raises(SystemExit):
        o2.step()


def test_optim_resume_refuses_other_schedule():
    p = torch.nn.Parameter(torch.zeros(1))
    sd = Optim([p], OptimCfg(sched_steps=600)).state_dict()
    with pytest.raises(ValueError):
        Optim([p], OptimCfg(sched_steps=300)).load(sd)
    o = Optim([p], OptimCfg(sched_steps=600))
    o.load(sd)


def test_optim_resume_equivalence():
    def run(split):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(4))
        o = Optim([p], OptimCfg(lr=0.05, sched_steps=30, clip_rel=1.5, clip_warmup=2))
        for t in range(30):
            if t == split:
                sd, pv = o.state_dict(), p.detach().clone()
                p = torch.nn.Parameter(pv)
                o = Optim([p], OptimCfg(lr=0.05, sched_steps=30, clip_rel=1.5, clip_warmup=2))
                o.load(sd)
            o.zero_grad()
            ((p - torch.arange(4.0)) ** 2 * (1 + t % 3)).sum().backward()
            o.step()
        return p.detach()
    assert torch.equal(run(-1), run(13))


def test_cosine_restarts_matches_torch():
    p = torch.nn.Parameter(torch.zeros(3))
    o = Optim([p], OptimCfg(lr=2e-6, sched_steps=100, schedule=Schedule.cosine_restarts))
    ref_opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(3))], lr=2e-6)
    ref = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(ref_opt, T_0=100)
    for t in range(350):
        assert abs(o.lr() - ref.get_last_lr()[0]) < 1e-15, t
        p.grad = torch.ones(3)
        o.step()
        ref_opt.step()
        ref.step()


def test_split_solved_keeps_probe_apart_from_clean_held():
    from shared.eval import BankCfg, split_solved
    ids = [f"p{i}" for i in range(269)]
    probe, held, train = split_solved(ids, BankCfg(bank=Path("x")))
    assert (len(probe), len(held), len(train)) == (24, 54, 191)  # mbpp-sized: 54 clean, not 53 - 24 = 29
    assert sorted(probe + held + train) == sorted(ids) and not set(probe) & set(held)
    probe, held, train = split_solved(ids[:167], BankCfg(bank=Path("x")))
    assert (len(probe), len(held)) == (24, 50) and len(train) == 93  # held_min lifts 33 to 50
    probe, held, train = split_solved(ids[:40], BankCfg(bank=Path("x")))
    assert len(probe) == 10 and len(held) == 15 and len(train) == 15  # small banks: train never below held


def test_build_bank_keeps_split_of_a_bank_begun_before_split_v2(tmp_path):
    from shared.eval import BankCfg, build_bank
    from shared.fs import atomic_json, read_json
    env = Env(200)
    cfg = BankCfg(bank=tmp_path, cal_n=16, n_chunks=3, n_per_chunk=32, min_pool=5)
    ident = (DecoderCfg(model="m1"),)
    bk = build_bank(Sampler(adapter(0)), env, cfg, DecodeCfg(), tmp_path, ident)
    key = read_json(tmp_path / "bank_key.json")
    for f in ("held_frac", "held_min"):
        key["BankCfg"].pop(f)
    atomic_json(key, tmp_path / "bank_key.json")  # as an older library wrote it
    assert build_bank(Sampler(adapter(0)), env, cfg, DecodeCfg(), tmp_path, ident) == bk


def test_merge_banks_balances_train_and_keeps_held():
    from shared.eval import merge_banks
    a, b = bank(Env(60)), bank(Env(120))
    m = merge_banks({"math": a, "mbpp": b}, True, 0, 24, 11)
    na = sum(i.startswith("math/") for i in m.train)
    nb = sum(i.startswith("mbpp/") for i in m.train)
    assert na == nb == min(len(a.pool), len(b.pool))
    assert len(m.held) == len(a.held) + len(b.held) and set(m.probe) <= set(m.held)
    assert set(m.probe) <= {f"math/{i}" for i in a.probe} | {f"mbpp/{i}" for i in b.probe}  # parts' clean stays clean
    assert m.rates["math/" + a.held[0]] == a.rates[a.held[0]] and m.split == "all"
    with pytest.raises(ValueError):
        merge_banks({"math": a, "x": dataclasses.replace(b, decode=dataclasses.replace(b.decode, max_tokens=2048))}, True, 0, 24, 11)


def test_eval_pairs_refuse_other_decode(tmp_path):
    from shared.eval import EvalRates, diffs, load_eval, verdict
    from shared.fs import atomic_json
    env = Env()
    bk = bank(env, Backend.vllm)
    grades = {i: [1.0, 0.0] for i in bk.held}
    for name, dc in (("a", bk.decode), ("b", dataclasses.replace(bk.decode, max_tokens=2048))):
        atomic_json({"verdict": verdict(grades, Backend.vllm, dc, bk), "grades": grades}, tmp_path / f"{name}.json")
    a, b = load_eval(tmp_path / "a.json"), load_eval(tmp_path / "b.json")
    assert a == EvalRates({i: 0.5 for i in bk.held}, Backend.vllm, bk.decode)
    assert set(diffs(a, a).values()) == {0.0}
    with pytest.raises(ValueError):
        diffs(a, b)


def test_extend_bank_adds_new_problems_to_held_only(tmp_path):
    import importlib.util
    from shared.eval import BankCfg, build_bank
    spec = importlib.util.spec_from_file_location(
        "extend_bank", Path(__file__).parents[2] / "experiments/tools/extend_bank.py")
    eb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eb)
    env = Env(200)
    old = {f"p{i}" for i in range(120)}
    cfg = BankCfg(bank=tmp_path / "base", cal_n=16, n_chunks=2, n_per_chunk=16, min_pool=5)
    ident = (DecoderCfg(model="m1"),)
    base = build_bank(Sampler(adapter(0)), eb.Only(env, old), cfg, DecodeCfg(), tmp_path / "base", ident)
    out = eb.extend(tmp_path / "base", tmp_path / "x", env, Sampler(adapter(0)), cfg, DecodeCfg(), ident, None)
    added = set(out.held) - set(base.held)
    assert added and added <= set(env_ids := [p.id for p in env.problems("all")]) - old
    assert out.train == base.train and out.probe == base.probe and out.held[:len(base.held)] == base.held
    assert set(out.clean) == set(base.clean) | added and not set(out.held) & set(out.train)
    sel = Bank.load(tmp_path / "x" / "ext" / "bank.json")  # the new problems went through the same window
    assert added == set(sel.train) | set(sel.held) and all(0 < out.rates[i] < 1 for i in added)
    assert len(env_ids) == 200 and Bank.load(tmp_path / "x" / "bank.json") == out
