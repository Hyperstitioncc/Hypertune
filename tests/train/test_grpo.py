import math
from types import SimpleNamespace

import torch

from algorithms.grpo import GRPO, GRPOCfg, IrrScale
from shared.eval import chunk_seed
from tests.train.conftest import BP, V, Env, Sampler, Tok, adapter, bank, identity, logprobs


def ref_backward_weighted(ad, base, rows, weights, kl_beta, ref_lp, is_cap):
    s = torch.stack([lp.sum() for lp in logprobs(None, ad, rows, [torch.tensor(r.ids)[:, None] for r in rows])])
    L = torch.tensor([float(len(r.ids)) for r in rows], dtype=s.dtype)
    w = torch.tensor(weights, dtype=s.dtype)
    if ref_lp is not None:
        w = w * torch.exp((s.detach() - torch.tensor(ref_lp, dtype=s.dtype)) / L).clamp(max=is_cap)
    loss = -(w * s).sum()
    if kl_beta > 0:
        with torch.no_grad():
            sb = torch.stack([lp.sum() for lp in logprobs(None, base, rows, [torch.tensor(r.ids)[:, None] for r in rows])])
        loss = loss + kl_beta * (0.5 * (s - sb) ** 2 / L).sum()
    loss.backward()
    return float(loss.detach())


def ref_step(ad, base, sampler, env, bk, batch, step, G, kl_beta, tis, cap, pos_only=False):
    s = bk.decode.at(G, chunk_seed(1500, step))
    outs = sampler.sample([p.prompt for p in batch], env.stops, s)
    n_rows, loss = 0, 0.0
    for j, (p, comps) in enumerate(zip(batch, outs)):
        rewards = [1.0 if c.text.startswith("Y") else 0.0 for c in comps]
        if not 0 < sum(rewards) < G:
            continue
        mu = sum(rewards) / G
        rows, advs, refs = [], [], []
        for c, rew in zip(comps, rewards):
            a = rew - mu
            if pos_only and a <= 0:
                continue
            rows.append(c)
            advs.append(a)
            refs.append(float(sum(c.engine_lp)))
        if rows:
            n_rows += len(rows)
            loss += ref_backward_weighted(ad, base, rows, [a / 640.0 for a in advs], kl_beta,
                                          refs if tis else None, cap)
    for p in ad.params():
        p.grad.div_(n_rows)
    return loss / n_rows, n_rows


def make(cfg, ad):
    env = Env()
    bk = bank(env)
    dec = SimpleNamespace(tok=Tok(), head=torch.nn.Linear(1, V, bias=False))
    return GRPO(cfg, env, Sampler(ad), dec, BP, bk, bk.decode, identity(), {p.id: p for p in env.problems("train")},
                {}, []), env, bk


def check(cfg, kl_beta, tis, cap, pos_only=False):
    a_new, a_ref = adapter(3), adapter(3)
    g, env, bk = make(cfg, a_new)
    batch = env.problems("train")[:6]
    m = g.step(a_new, batch, 4)
    want_loss, want_rows = ref_step(a_ref, identity(), Sampler(a_ref), env, bk, batch, 4, cfg.grpo_g, kl_beta, tis, cap,
                                  pos_only)
    assert m["rows"] == want_rows > 0
    assert abs(m["loss"] - want_loss) < 1e-12
    for p, q in zip(a_new.params(), a_ref.params()):
        assert torch.allclose(p.grad, q.grad, rtol=0, atol=1e-15)
    return m


def test_dr_grpo_plain():
    check(GRPOCfg(grpo_g=8, grpo_kl_meter=False), 0.0, False, 2.0)


def test_dr_grpo_tis_kl():
    m = check(GRPOCfg(grpo_g=8, grpo_tis=True, grpo_tis_cap=1.3, grpo_kl_beta=0.7, grpo_kl_meter=False), 0.7, True, 1.3)
    assert 0 < m["rho_mean"] <= 1.3


def test_pos_only():
    check(GRPOCfg(grpo_g=8, grpo_pos_only=True, grpo_kl_meter=False), 0.0, False, 2.0, pos_only=True)


def test_rho_value():
    from algorithms.grpo import tis_rho
    from shared.types import Completion
    rows = [Completion((1,), (2, 3), "", (-1.0, -2.0)), Completion((1,), (2,), "", (-5.0,))]
    s = torch.tensor([-2.0, -1.0], dtype=torch.float64)
    rho = tis_rho(s, rows, 2.0)
    assert torch.allclose(rho, torch.tensor([math.exp(0.5), 2.0], dtype=torch.float64))


def test_adv_perm_keeps_counts():
    cfg = GRPOCfg(grpo_g=8, grpo_adv_perm=True, grpo_kl_meter=False)
    ad = adapter(1)
    g, env, _ = make(cfg, ad)
    m = g.step(ad, env.problems("train")[:6], 2)
    assert m["rows"] > 0


def test_logratio_meter():
    ad = adapter(2)
    g, env, _ = make(GRPOCfg(grpo_kl_every=1), ad)
    m = g.step(ad, env.problems("train")[:4], 0)
    assert "logratio" in m and math.isfinite(m["logratio"])


def test_irr_ratio_scaling(monkeypatch):
    import algorithms.grpo as grpo_mod
    from shared.types import Completion
    monkeypatch.setattr(grpo_mod, "irr_measure", lambda dec, bp, a, row, kind: (a.module.w ** 2).sum() * len(row.ids))
    ad = adapter(5)
    items = [Completion((1, 2), (3, 4, 5), "", ()), Completion((2,), (6, 7), "", ())]
    cfg = GRPOCfg(irr_pen=0.25, irr_mode=IrrScale.ratio, irr_n=2)
    g, env, _ = make(cfg, ad)
    g.irr = items
    (ad.module.w ** 2).sum().backward()
    g1 = ad.module.w.grad.clone()
    m = g.irr_apply(ad, 0)
    pen_part = ad.module.w.grad - g1
    assert abs(pen_part.norm().item() - 0.25 * g1.norm().item()) < 1e-9
    assert m["irr"] > 0


def test_an_empty_shard_steps_and_scores_nothing():
    import dataclasses
    import pytest
    ad = adapter(2)
    g, env, bk = make(GRPOCfg(grpo_kl_every=1), ad)
    m = g.step(ad, [], 0)
    assert m["rows"] == 0 and m["reward"] == 0 and m["loss"] == 0 and "logratio" not in m and ad.module.w.grad is None
    g.bank = dataclasses.replace(bk, probe=())
    assert g.select(ad, 0).n == 0
    with pytest.raises(ValueError, match="irr-n"):
        GRPOCfg(irr_pen=0.1, irr_n=0)
