import json

import pytest
import torch

from algorithms.grpo import GRPO, GRPOCfg
from algorithms.sft import LossNorm, build
from shared.optim import Optim, OptimCfg
from shared.train import TrainCfg, pick, train
from tests.train.conftest import BP, Decoder, Env, Sampler, Tok, adapter, bank, identity


class Crash(Exception):
    pass


class Recorder:
    def __init__(self, algo, crash_at, seen):
        self.algo, self.crash_at, self.seen = algo, crash_at, seen

    def step(self, ad, batch, step):
        if step == self.crash_at:
            raise Crash()
        self.seen.append((step, [p.id for p in batch]))
        return self.algo.step(ad, batch, step)

    def select(self, ad, step):
        return self.algo.select(ad, step)


def grpo_proc(env, bk, ad):
    return GRPO(GRPOCfg(grpo_g=4, grpo_kl_every=5, grpo_probe_n=4), env, Sampler(ad), Decoder(Tok()), BP, bk,
                bk.decode, identity(), {p.id: p for p in env.problems("train")}, {}, [])


def sft_proc(env, bk, ad):
    corpus = {p.id: [f"Y{k}{p.id}" for k in range(3)] for p in env.problems("train")}
    return build(Decoder(Tok()), BP, env, bk, corpus, LossNorm.row_mean)


def run(tmp, make, tcfg, ocfg, crashes):
    env = Env()
    bk = bank(env)
    seen = []
    for c in crashes + [None]:
        ad = adapter(7)
        algo = Recorder(make(env, bk, ad), c, seen)
        try:
            train(algo, ad, env, Optim(ad.params(), ocfg), bk, tcfg, tmp)
            break
        except Crash:
            continue
    st = torch.load(tmp / "state.pt", weights_only=False)
    best = torch.load(tmp / "best.pt", weights_only=False)
    return st, best, untimed((tmp / "metrics.jsonl").read_text()), seen


def untimed(log: str) -> list[dict]:
    """The log without its wall-clock fields, which differ run to run."""
    return [{k: v for k, v in json.loads(ln).items() if k != "time" and not k.startswith("sec")}
            for ln in log.splitlines()]


GRPO_T = TrainCfg(run="r", steps=40, batch=6, eval_every=5, patience=3, trip_logratio=False)
GRPO_O = OptimCfg(lr=0.05, sched_steps=40, clip_rel=3.0)


@pytest.mark.parametrize("crashes", [[3, 11, 22], [5, 10], [0]])
def test_grpo_resume_identical(tmp_path, crashes):
    ref = run(tmp_path / "ref", grpo_proc, GRPO_T, GRPO_O, [])
    got = run(tmp_path / "x", grpo_proc, GRPO_T, GRPO_O, crashes)
    assert got[0]["step"] == ref[0]["step"] and got[0]["ctr"] == ref[0]["ctr"]
    assert got[0]["hist"] == ref[0]["hist"] and got[2] == ref[2]
    for k in ref[0]["adapter"]["state"]:
        assert torch.equal(got[0]["adapter"]["state"][k], ref[0]["adapter"]["state"][k])
        assert torch.equal(got[1]["adapter"]["state"][k], ref[1]["adapter"]["state"][k])
    assert got[1]["best_step"] == ref[1]["best_step"] and got[1]["final"]
    assert dict(got[3]) == dict(ref[3])


def test_next_batch_after_resume():
    env = Env()
    pool = bank(env).pool
    assert pick(pool, 17, GRPO_T) == pick(list(pool), 17, GRPO_T)
    from shared.train import BatchOrder
    cfg = TrainCfg(run="r", batch=4, batch_order=BatchOrder.epoch, pick_seed=1234)
    epoch = [i for s in range(3) for i in pick(pool, s, cfg)]
    n = -(-len(pool) // 4)
    full = [i for s in range(n) for i in pick(pool, s, cfg)]
    assert sorted(full) == sorted(pool) and epoch == full[:len(epoch)]


def test_finished_run_does_not_train_again(tmp_path):
    run(tmp_path, grpo_proc, GRPO_T, GRPO_O, [])
    env = Env()
    bk = bank(env)
    ad = adapter(7)
    train(Recorder(grpo_proc(env, bk, ad), 0, []), ad, env, Optim(ad.params(), GRPO_O), bk, GRPO_T, tmp_path)


SFT_T = TrainCfg(run="r", steps=300, batch=8, eval_every=50, pick_seed=60, early_stop=False, reward_floor=0.0,
                 trip_logratio=False)
SFT_O = OptimCfg(lr=0.05, sched_steps=300)


def test_offpolicy_budget_and_resume(tmp_path):
    ref = run(tmp_path / "ref", sft_proc, SFT_T, SFT_O, [])
    assert ref[0]["step"] == 300 and ref[0]["hist"][-1]["step"] == 300
    assert ref[0]["optim"]["t"] == 300
    last = ref[2][-1]
    assert last["lr"] < 2e-6
    for crashes in ([37], [150, 251]):
        got = run(tmp_path / f"c{crashes[0]}", sft_proc, SFT_T, SFT_O, crashes)
        for k in ref[1]["adapter"]["state"]:
            assert torch.equal(got[1]["adapter"]["state"][k], ref[1]["adapter"]["state"][k])
        assert got[0]["hist"] == ref[0]["hist"]


def test_offpolicy_refuses_other_schedule(tmp_path):
    t600 = TrainCfg(**{**SFT_T.__dict__, "steps": 600})
    env = Env()
    bk = bank(env)
    ad = adapter(7)
    with pytest.raises(Crash):
        train(Recorder(sft_proc(env, bk, ad), 100, []), ad, env, Optim(ad.params(), OptimCfg(lr=0.05, sched_steps=600)),
              bk, t600, tmp_path)
    with pytest.raises(ValueError):
        run(tmp_path, sft_proc, SFT_T, SFT_O, [])
    st, _, _, _ = run(tmp_path, sft_proc, t600, OptimCfg(lr=0.05, sched_steps=600), [])
    assert st["step"] == 600


def test_probe_drop_guard_rolls_back_and_halves_lr(tmp_path):
    import dataclasses
    probes = iter([0.3, 0.1, 0.25])

    class Drop:
        def __init__(self, inner):
            self.inner = inner

        def step(self, ad, batch, step):
            return self.inner.step(ad, batch, step)

        def select(self, ad, step):
            from shared.types import PartialMean
            return PartialMean(next(probes), 1)

    env, bk, ad = Env(), bank(Env()), adapter(7)
    cfg = dataclasses.replace(GRPO_T, steps=15, eval_every=5, early_stop=False, guard_drop=0.15, guard_min_best=0.05)
    opt = Optim(ad.params(), GRPO_O)
    train(Drop(grpo_proc(env, bk, ad)), ad, env, opt, bk, cfg, tmp_path)
    log = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    rb = [r for r in log if "guard_rollback_to" in r]
    assert len(rb) == 1 and rb[0]["step"] == 10 and rb[0]["guard_rollback_to"] == 5.0
    assert opt.mult == 0.5


def test_reward_floor_ignores_algorithms_without_reward(tmp_path):
    t = TrainCfg(run="r", steps=60, batch=8, eval_every=20, pick_seed=60, early_stop=False, trip_logratio=False)
    st = run(tmp_path, sft_proc, t, OptimCfg(lr=0.05, sched_steps=60), [])[0]
    assert st["step"] == 60


def test_init_from_loads_other_run_then_resume_wins(tmp_path):
    import dataclasses
    run(tmp_path / "warm", grpo_proc, GRPO_T, GRPO_O, [])
    warm = torch.load(tmp_path / "warm" / "best.pt", weights_only=False)["adapter"]
    cfg = dataclasses.replace(GRPO_T, run="x", init_from="warm", steps=1, probe0=True)
    env, bk, ad = Env(), bank(Env()), adapter(7)
    frozen = dataclasses.replace(GRPO_O, lr=0.0)
    train(grpo_proc(env, bk, ad), ad, env, Optim(ad.params(), frozen), bk, cfg, tmp_path / "x")
    got = torch.load(tmp_path / "x" / "best.pt", weights_only=False)["adapter"]["state"]
    for k in warm["state"]:
        assert torch.equal(got[k], warm["state"][k])
    cfg2 = dataclasses.replace(GRPO_T, run="y", init_from="warm", steps=6)
    ad2 = adapter(7)
    train(grpo_proc(env, bk, ad2), ad2, env, Optim(ad2.params(), GRPO_O), bk, cfg2, tmp_path / "y")
    ad3 = adapter(7)
    st = torch.load(tmp_path / "y" / "state.pt", weights_only=False)
    (tmp_path / "y" / "best.pt").unlink()
    torch.save(st, tmp_path / "y" / "state.pt")
    train(grpo_proc(env, bk, ad3), ad3, env, Optim(ad3.params(), GRPO_O), bk, cfg2, tmp_path / "y")
    for k in st["adapter"]["state"]:
        assert torch.equal(ad3.state_dict()["state"][k], st["adapter"]["state"][k])




def test_snapshots_every_n_steps_and_at_the_start_survive_resume(tmp_path):
    import dataclasses
    cfg = dataclasses.replace(GRPO_T, steps=6, snap_every=2)
    env, bk, ad = Env(), bank(Env()), adapter(3)
    with pytest.raises(Crash):
        train(Recorder(grpo_proc(env, bk, ad), 3, []), ad, env, Optim(ad.params(), GRPO_O), bk, cfg, tmp_path)
    assert sorted(p.name for p in (tmp_path / "snaps").iterdir()) == ["00000.pt", "00002.pt"]
    ad2 = adapter(3)
    train(grpo_proc(env, bk, ad2), ad2, env, Optim(ad2.params(), GRPO_O), bk, cfg, tmp_path)
    assert sorted(p.name for p in (tmp_path / "snaps").iterdir()) == ["00000.pt", "00002.pt", "00004.pt", "00006.pt"]
    last = torch.load(tmp_path / "snaps" / "00006.pt", weights_only=False)
    assert last["step"] == 6 and all(torch.equal(last["adapter"]["state"][k], v)
                                      for k, v in ad2.state_dict()["state"].items())


def test_stop_at_then_continue_matches_the_straight_run(tmp_path):
    import dataclasses
    env, bk = Env(), bank(Env())
    cfg = dataclasses.replace(GRPO_T, steps=6)
    ref = adapter(3)
    train(grpo_proc(env, bk, ref), ref, env, Optim(ref.params(), GRPO_O), bk, cfg, tmp_path / "ref")
    ad = adapter(3)
    train(grpo_proc(env, bk, ad), ad, env, Optim(ad.params(), GRPO_O), bk, dataclasses.replace(cfg, stop_at=2), tmp_path / "s")
    best = torch.load(tmp_path / "s" / "best.pt", weights_only=False)
    assert not best["final"] and best["hist"][-1]["step"] == 2
    ad2 = adapter(3)
    train(grpo_proc(env, bk, ad2), ad2, env, Optim(ad2.params(), GRPO_O), bk, cfg, tmp_path / "s")
    assert torch.load(tmp_path / "s" / "best.pt", weights_only=False)["final"]
    ev = [json.loads(ln) for ln in (tmp_path / "s" / "events.jsonl").read_text().splitlines()]
    assert [(e["event"], e["step"]) for e in ev] == [("start", 0), ("end", 2), ("start", 2), ("end", 6)]
    rows = [json.loads(ln) for ln in (tmp_path / "s" / "metrics.jsonl").read_text().splitlines()]
    assert all(r["sec"] >= 0 and r["time"] > ev[0]["time"] - 1 for r in rows)
    assert all(torch.allclose(ad2.state_dict()["state"][k], v) for k, v in ref.state_dict()["state"].items())


def test_the_last_step_is_snapshotted_even_off_the_snap_grid(tmp_path):
    import dataclasses
    env, bk, ad = Env(), bank(Env()), adapter(3)
    train(grpo_proc(env, bk, ad), ad, env, Optim(ad.params(), GRPO_O), bk, dataclasses.replace(GRPO_T, steps=5, snap_every=2),
          tmp_path)
    assert sorted(p.name for p in (tmp_path / "snaps").iterdir()) == ["00000.pt", "00002.pt", "00004.pt", "00005.pt"]


@pytest.mark.parametrize("stop", [7, 10])
def test_stop_at_then_continue_leaves_best_and_counters_as_the_straight_run(tmp_path, stop):
    """Early stop on (GRPO_T): a stop off the eval grid is probed for the sweep but must not count as an eval."""
    import dataclasses
    ref = run(tmp_path / "ref", grpo_proc, GRPO_T, GRPO_O, [])
    env, bk = Env(), bank(Env())
    ad = adapter(7)
    train(grpo_proc(env, bk, ad), ad, env, Optim(ad.params(), GRPO_O), bk, dataclasses.replace(GRPO_T, stop_at=stop),
          tmp_path / "s")
    mid = torch.load(tmp_path / "s" / "best.pt", weights_only=False)
    assert not mid["final"] and mid["hist"][-1]["step"] == stop and bool(mid["hist"][-1].get("stop_at")) == bool(stop % 5)
    ad = adapter(7)
    train(grpo_proc(env, bk, ad), ad, env, Optim(ad.params(), GRPO_O), bk, GRPO_T, tmp_path / "s")
    st = torch.load(tmp_path / "s" / "state.pt", weights_only=False)
    best = torch.load(tmp_path / "s" / "best.pt", weights_only=False)
    assert st["step"] == ref[0]["step"] and st["ctr"] == ref[0]["ctr"] and st["best"]["step"] == ref[0]["best"]["step"]
    assert [h for h in st["hist"] if not h.get("stop_at")] == ref[0]["hist"]
    assert any(h["step"] == stop for h in best["hist"])  # the sweep still reads its score after the continuation
    for k in ref[0]["adapter"]["state"]:
        assert torch.equal(st["adapter"]["state"][k], ref[0]["adapter"]["state"][k])
        assert torch.equal(best["adapter"]["state"][k], ref[1]["adapter"]["state"][k])
    drop = lambda log: [{k: v for k, v in r.items() if not (k == "probe" and r["step"] == stop and stop % 5)}  # noqa: E731
                        for r in log]
    assert drop(untimed((tmp_path / "s" / "metrics.jsonl").read_text())) == drop(ref[2])


class Spike:
    """GRPO with its nll replaced: flat, with a spike at the given steps, so --ent-guard trips there."""

    def __init__(self, inner, at):
        self.inner, self.at = inner, at

    def step(self, ad, batch, step):
        return {**self.inner.step(ad, batch, step), "nll": 5.0 if step in self.at else 1.0}

    def select(self, ad, step):
        return self.inner.select(ad, step)


@pytest.mark.parametrize("crashes", [[6], [9, 12]])
def test_ent_guard_resume_keeps_the_rollback_target(tmp_path, crashes):
    import dataclasses
    cfg = dataclasses.replace(GRPO_T, steps=16, ent_guard=True, early_stop=False, snap_every=1)
    make = lambda env, bk, ad: Spike(grpo_proc(env, bk, ad), {8})  # noqa: E731
    ref = run(tmp_path / "ref", make, cfg, GRPO_O, [])
    got = run(tmp_path / "x", make, cfg, GRPO_O, crashes)
    assert [r.get("rollback_to") for r in ref[2] if "rollback_to" in r] == [4.0]
    assert got[2] == ref[2] and got[0]["ctr"] == ref[0]["ctr"]
    for k in ref[0]["adapter"]["state"]:
        assert torch.equal(got[0]["adapter"]["state"][k], ref[0]["adapter"]["state"][k])
    assert sorted(p.name for p in (tmp_path / "x" / "snaps").iterdir()) == [f"{i:05d}.pt" for i in range(17)]


def test_resume_drops_a_half_written_metrics_line(tmp_path):
    import dataclasses
    cfg = dataclasses.replace(GRPO_T, steps=8)
    ref = run(tmp_path / "ref", grpo_proc, cfg, GRPO_O, [])
    env, bk, ad = Env(), bank(Env()), adapter(7)
    with pytest.raises(Crash):
        train(Recorder(grpo_proc(env, bk, ad), 5, []), ad, env, Optim(ad.params(), GRPO_O), bk, cfg, tmp_path / "x")
    with open(tmp_path / "x" / "metrics.jsonl", "a") as f:
        f.write('{"step": 6, "rew')
    ad = adapter(7)
    train(grpo_proc(env, bk, ad), ad, env, Optim(ad.params(), GRPO_O), bk, cfg, tmp_path / "x")
    assert untimed((tmp_path / "x" / "metrics.jsonl").read_text()) == ref[2]


def test_sched_steps_must_be_positive(tmp_path):
    import dataclasses
    env, bk, ad = Env(), bank(Env()), adapter(7)
    with pytest.raises(ValueError, match="sched-steps"):
        train(grpo_proc(env, bk, ad), ad, env, Optim(ad.params(), dataclasses.replace(GRPO_O, sched_steps=0)), bk,
              GRPO_T, tmp_path)
