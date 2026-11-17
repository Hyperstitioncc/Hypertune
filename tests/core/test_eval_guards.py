import dataclasses
import os
import sys
import threading

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from envs.base import Problem  # noqa: E402
from shared.config import parse  # noqa: E402
from shared.decoder import DecoderCfg  # noqa: E402
from shared.eval import (Bank, adapter_fingerprint, comparable, evaluate, mean_rates, merge_banks,  # noqa: E402
                         run_ident, verdict)
from shared.fs import atomic_json, atomic_write, read_json  # noqa: E402
from shared.rollout import Quant, RolloutCfg  # noqa: E402
from shared.types import Backend, Completion, DecodeCfg, Sampling  # noqa: E402


class Env:
    name = "fake"
    stops = ()

    def grade(self, pairs):
        return [float(t == "Y") for _, t in pairs]


class Sampler:
    backend = Backend.vllm

    def __init__(self, text="Y"):
        self.text, self.calls = text, 0

    def sample(self, prompts, stops, s):
        self.calls += 1
        return [[Completion([1], [2], self.text, [0.0]) for _ in range(s.n)] for _ in prompts]


PS = [Problem(f"p{i}", f"Q{i}", "Y") for i in range(3)]
SM = Sampling(n=2, temperature=0.8, top_p=0.95, max_tokens=64, seed=1)


@dataclasses.dataclass(frozen=True)
class FakeEnvCfg:
    fake_prompt: str = "fewshot"
    fake_data: str = "x"


def ident(model="m", quant=Quant.bf16, prompt="fewshot"):
    return run_ident("fake", FakeEnvCfg(prompt), DecoderCfg(model=model), RolloutCfg(quant=quant))


def bank(id_=None):
    return Bank(backend=Backend.vllm, split="train", decode=DecodeCfg(0.8, 0.95, 64), rates={p.id: 0.5 for p in PS},
                train=(), held=tuple(p.id for p in PS), probe=(), pool_hi=0.5, ident=id_)


def test_fingerprint_tracks_weights():
    a = {"kind": "lora", "state": {"w": torch.ones(4, 3, dtype=torch.bfloat16), "b": torch.zeros(3)}}
    b = {"kind": "lora", "state": {"b": torch.zeros(3), "w": torch.ones(4, 3, dtype=torch.bfloat16)}}
    assert adapter_fingerprint(a) == adapter_fingerprint(b)
    b["state"]["w"][3, 2] = 2.0
    assert adapter_fingerprint(a) != adapter_fingerprint(b)
    c = {"kind": "lora", "state": {"w": torch.ones(3, 4, dtype=torch.bfloat16), "b": torch.zeros(3)}}
    assert adapter_fingerprint(a) != adapter_fingerprint(c)
    assert adapter_fingerprint({"kind": "lora", "state": {"x": torch.tensor(1.0)}}) \
        != adapter_fingerprint({"kind": "lora", "state": {"x": torch.tensor(2.0)}})


def test_eval_resume_discards_chunks_of_another_adapter(tmp_path, capsys):
    g = evaluate(Sampler("Y"), Env(), PS, SM, 2, tmp_path, "fp-a")
    s = Sampler("N")
    assert evaluate(s, Env(), PS, SM, 2, tmp_path, "fp-a") == g and s.calls == 0
    g2 = evaluate(s, Env(), PS, SM, 2, tmp_path, "fp-b")
    assert s.calls == 2 and all(x == [0.0, 0.0, 0.0, 0.0] for x in g2.values())
    assert "discarding" in capsys.readouterr().out
    assert read_json(tmp_path / "eval_chunk_0.json")["key"]["adapter"] == "fp-b"
    with pytest.raises(ValueError):
        evaluate(s, Env(), PS, dataclasses.replace(SM, n=4), 2, tmp_path, "fp-b")


def test_comparable_checks_decoder_and_env(capsys):
    import shared.eval
    shared.eval._warned.clear()  # the "predates" note prints once a process; an earlier test may have used it
    d = DecodeCfg(0.8, 0.95, 64)
    bk = bank(ident())
    assert comparable(Backend.vllm, d, bk, ident())
    assert not comparable(Backend.vllm, d, bk, ident(model="other"))
    assert not comparable(Backend.vllm, d, bk, ident(quant=Quant.fp8))
    assert not comparable(Backend.vllm, d, bk, ident(prompt="zeroshot"))
    assert comparable(Backend.vllm, d, bank(None), ident(model="other"))
    assert "predates" in capsys.readouterr().out
    v = verdict({p.id: [1.0, 0.0] for p in PS}, Backend.vllm, d, bk, ident(quant=Quant.fp8))
    assert not v["paired_with_bank"] and "passed" not in v


def test_bank_ident_roundtrip_and_merge(tmp_path):
    bk = bank(ident())
    bk.save(tmp_path / "bank.json")
    got = Bank.load(tmp_path / "bank.json")
    assert got == bk
    assert comparable(Backend.vllm, bk.decode, got, ident())
    atomic_json({k: v for k, v in read_json(tmp_path / "bank.json").items() if k != "ident"}, tmp_path / "old.json")
    assert Bank.load(tmp_path / "old.json").ident is None
    m = merge_banks({"a": got, "b": bank(None)}, False, 0, 2, 0)
    assert m.ident == {"decoder": got.ident["decoder"], "env": None}
    with pytest.raises(ValueError):
        merge_banks({"a": got, "b": bank(ident(model="other"))}, False, 0, 2, 0)


def test_empty_grades_raise_clearly():
    bk = bank()
    with pytest.raises(ValueError, match="split is empty"):
        verdict({}, Backend.vllm, bk.decode, bk)
    with pytest.raises(ValueError, match="no grades"):
        mean_rates({"p0": []})
    with pytest.raises(ValueError, match="no grades"):
        verdict(evaluate(Sampler(), Env(), PS, SM, 0, "unused"), Backend.vllm, bk.decode, bk)


@dataclasses.dataclass(frozen=True)
class Knobs:
    long_name: int = 1
    other: int = 2


def test_abbreviated_flag_rejected():
    assert parse(Knobs, argv=["--long-name", "5"]) == (Knobs(5, 2),)
    with pytest.raises(SystemExit):
        parse(Knobs, argv=["--long", "5"])
    from shared.config import parse_known
    (k,), rest = parse_known(Knobs, argv=["--long", "5"])
    assert k == Knobs() and rest == ["--long", "5"]


def test_atomic_write_tmp_names_unique_across_threads(tmp_path):
    target, n = tmp_path / "x.json", 8
    seen, barrier = [], threading.Barrier(n)

    def write(tmp):
        seen.append(tmp)
        barrier.wait()
        with open(tmp, "w") as f:
            f.write(tmp)

    ts = [threading.Thread(target=atomic_write, args=(target, write)) for _ in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(set(seen)) == n and target.read_text() in seen
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_eval_refuses_existing_eval_json_with_other_flags(tmp_path):
    from shared import cli
    argv = ["eval", "--env", "math", "--backend", "vllm", "--math-levels", "level1", "--model", "m",
            "--bank", "b", "--data", str(tmp_path / "d"), "--out", str(tmp_path / "o"), "--eval-run", "r"]
    run = tmp_path / "o" / "r"
    atomic_json({"verdict": {}, "grades": {}}, run / "eval.json")
    cli.main(argv)  # an eval.json from before configs were stored is kept
    (sel,), rest = cli.parse_known(cli.RunChoice, argv=argv[1:])
    import adapters
    from envs import env_from
    from shared import backprop, decoder as dec, rollout
    env_cls, _ = env_from("math", rest)
    cfgs = cli.parse(env_cls, adapters.AdapterCfg, dec.DecoderCfg, rollout.RolloutCfg, backprop.BackpropCfg,
                     cli.BankCfg, DecodeCfg, cli.Roots, cli.EvalCfg, argv=rest)
    definition = cli.eval_definition([sel, *[c for c in cfgs if not isinstance(c, cli.Roots)]])
    atomic_json({"verdict": {}, "grades": {}, "config": definition}, run / "eval.json")
    cli.main(argv)
    cli.main(argv + ["--gpu-mem", "80GiB"])  # resources do not define an eval
    with pytest.raises(SystemExit, match="different eval config"):
        cli.main(argv + ["--eval-split", "clean"])
    with pytest.raises(SystemExit, match="different eval config"):
        cli.main(argv + ["--eval-n", "8"])
