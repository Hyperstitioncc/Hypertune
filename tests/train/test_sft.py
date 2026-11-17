import dataclasses
import json

import pytest

import torch

from algorithms.sft import GradedProbe, LossNorm, SFTCfg, build
from envs.domains import DomainEnv
from shared.eval import Bank, BankCfg, BankSelect, build_bank
from shared.types import DecodeCfg, TokenRow
from tests.train.conftest import BP, Decoder, Env, Sampler, Tok, adapter, bank

GREEDY = DecodeCfg(temperature=0.0, top_p=1.0, max_tokens=8)
EOS = 3


class EosTok(Tok):
    eos_token_id = EOS


def domain(root, n=(12, 5, 7)) -> DomainEnv:
    for name, k in zip(("train", "dev", "test"), n):
        with open(root / f"{name}.jsonl", "w") as f:
            for i in range(k):
                f.write(json.dumps({"id": f"{name}-{i}", "prompt": f"Q{name}{i}:", "target": " Y", "gold": "Y"}) + "\n")
    return DomainEnv("toy", root, lambda text, gold: text.startswith(gold))


def test_split_bank_takes_the_env_splits_as_given(tmp_path):
    env = domain(tmp_path)
    cfg = BankCfg(bank=tmp_path / "b", bank_split="all", select=BankSelect.splits, n_chunks=1, n_per_chunk=1)
    s = Sampler(adapter(0))
    bk = build_bank(s, env, cfg, GREEDY, tmp_path / "b", ())
    assert bk.split == "all" and bk.train == tuple(f"train-{i}" for i in range(12))
    assert bk.probe == tuple(f"dev-{i}" for i in range(5)) and bk.clean == [f"test-{i}" for i in range(7)]
    assert set(bk.rates) == set(bk.held) and [c.n for c in s.calls] == [1]
    assert Bank.load(tmp_path / "b" / "bank.json") == bk
    again = Sampler(adapter(0))
    assert build_bank(again, env, cfg, GREEDY, tmp_path / "b", ()) == bk and again.calls == []
    with pytest.raises(ValueError, match="train-pool solved"):
        bk.pool
    with pytest.raises(ValueError, match="bank-split all"):
        build_bank(s, env, dataclasses.replace(cfg, bank_split="train"), GREEDY, tmp_path / "c", ())


def test_corpus_holds_the_gold_target_of_every_split(tmp_path):
    env = domain(tmp_path)
    env.write_corpus()
    corpus = json.loads((tmp_path / "corpus.json").read_text())
    assert set(corpus) == {p.id for p in env.problems("all")} and corpus["dev-0"] == [" Y"]
    assert [p.id for p in env.problems("all")][:13] == [f"train-{i}" for i in range(12)] + ["dev-0"]


def test_eos_ends_every_target():
    env = Env()
    bk = bank(env)
    corpus = {p.id: ["Yab"] for p in env.problems("train")}
    plain = build(Decoder(Tok()), BP, env, bk, corpus, LossNorm.row_mean)
    ended = build(Decoder(EosTok()), BP, env, bk, corpus, LossNorm.row_mean, eos=True)
    assert all(ended.targets[i][0].ids == rows[0].ids + (EOS,) for i, rows in plain.targets.items())
    assert all(r.ids[-1] == EOS for r in ended.held_rows)


def test_graded_probe_is_the_mean_grade_of_one_draw_per_problem():
    env = Env()
    bk = bank(env)
    ad = adapter(0)
    s = Sampler(ad)
    by_id = {p.id: p for p in env.problems("train")}
    held = [by_id[i] for i in bk.probe]
    probe = GradedProbe(s, env, held, bk.decode, 5)
    got = probe.mean()
    rows = Sampler(ad).sample([p.prompt for p in held], env.stops, bk.decode.at(1, 5))
    want = env.grade([(p, c.text) for p, cs in zip(held, rows) for c in cs])
    assert got.n == len(held) and got.total == pytest.approx(sum(want)) and s.calls[-1].n == 1
    sft = build(Decoder(Tok()), BP, env, bk, {p.id: ["Yab"] for p in env.problems("train")}, LossNorm.row_mean)
    sft.probe = probe
    assert sft.select(ad, 0) == got


def test_penalty_gradient_is_scaled_to_the_task_gradient(monkeypatch):
    import algorithms.sft as sft_mod
    env = Env()
    bk = bank(env)
    ad = adapter(3)
    sft = build(Decoder(Tok()), BP, env, bk, {p.id: ["Yab"] for p in env.problems("train")}, LossNorm.row_mean)
    batch = env.problems("train")[:4]
    sft.step(ad, batch, 0)
    g_task = ad.module.w.grad.clone()
    ad.module.w.grad = None
    monkeypatch.setattr(sft_mod, "irr_measure", lambda dec, bp, a, row, kind: (a.module.w ** 2).sum())
    sft.cfg, sft.irr = SFTCfg(sft_corpus="x", irr_pen=0.5), [TokenRow((1, 2), (3,))] * 3
    m = sft.step(ad, batch, 0)
    g_pen = 2 * ad.module.w.detach()
    scale = 0.5 * g_task.norm() / g_pen.norm()
    assert torch.allclose(ad.module.w.grad, g_task + scale * g_pen, atol=1e-9)
    assert m["irr_scale"] == pytest.approx(float(scale)) and m["irr"] == pytest.approx(float((ad.module.w ** 2).sum()))


def test_eos_ends_only_targets_that_were_not_cut():
    env = Env()
    bk = dataclasses.replace(bank(env), decode=DecodeCfg(0.8, 0.95, 3))
    corpus = {p.id: ["Yab", "Yabcdef"] for p in env.problems("train")}
    whole, cut = build(Decoder(EosTok()), BP, env, bk, corpus, LossNorm.row_mean, eos=True).targets["p0"]
    enc = tuple(ord(c) % 11 for c in "Yab")
    assert whole.ids == enc + (EOS,) and cut.ids == enc
    plain = build(Decoder(Tok()), BP, env, bk, corpus, LossNorm.row_mean).targets["p0"]
    assert [r.ids for r in plain] == [enc, enc]


def test_an_empty_shard_steps_and_scores_nothing():
    from algorithms.sft import held_lp
    from shared.types import PartialMean
    env = Env()
    bk = bank(env)
    ad = adapter(3)
    sft = build(Decoder(Tok()), BP, env, bk, {p.id: ["Yab"] for p in env.problems("train")}, LossNorm.row_mean)
    m = sft.step(ad, [], 0)
    assert m["rows"] == 0 and m["loss"] == 0 and ad.module.w.grad is None
    assert held_lp(Decoder(Tok()), ad, [], BP, LossNorm.token_sum) == PartialMean(0.0, 0)
    assert GradedProbe(Sampler(ad), env, [], bk.decode, 0).mean() == PartialMean(0.0, 0)


def test_irr_pen_needs_items_to_draw():
    with pytest.raises(ValueError, match="irr-n"):
        SFTCfg(sft_corpus="x", irr_pen=0.1, irr_n=0)
    SFTCfg(sft_corpus="x", irr_n=0)
