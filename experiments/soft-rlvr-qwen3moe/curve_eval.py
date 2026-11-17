from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

import adapters
from envs import env_from
from shared import decoder as dec, rollout
from shared.config import dump, parse, parse_known
from shared.eval import Bank, BankCfg, adapter_fingerprint, evaluate, run_ident, verdict
from shared.fs import Roots, atomic_json
from shared.types import Backend, DecodeCfg


@dataclass(frozen=True)
class CurveCfg:
    env: str
    backend: Backend
    curve_files: tuple[str, ...]
    curve_out: str
    curve_n: int = 16
    curve_chunks: int = 1
    curve_seed: int = 6400
    curve_split: str = "held"


def put(adapter, obj: dict) -> None:
    mod = adapter.module
    if "adapter" in obj:
        adapter.load(obj["adapter"])
    elif "lora_A" in obj:
        mod.load_stack(obj)
    else:
        with torch.no_grad():
            mod.base.copy_(obj["soft"].to(mod.base.dtype))
            mod.kf0.copy_(obj["kfix"])
            mod.vf0.copy_(obj["vfix"])
            mod.dk.zero_()
            mod.dv.zero_()


def main(argv: list[str]) -> None:
    (c,), rest = parse_known(CurveCfg, argv=argv)
    if not c.curve_files:
        raise ValueError("--curve-files: nothing to score")
    env_cls, env_make = env_from(c.env, rest)
    ecfg, acfg, dcfg, rcfg, bcfg, decode, roots = parse(env_cls, adapters.AdapterCfg, dec.DecoderCfg, rollout.RolloutCfg,
                                                        BankCfg, DecodeCfg, Roots, argv=rest)
    out = roots.out / c.curve_out
    todo = [f for f in c.curve_files if not (out / f"{Path(f).with_suffix('').as_posix().replace('/', '__')}.json").exists()]
    if not todo:
        return
    dump([c, ecfg, acfg, dcfg, rcfg, bcfg, decode, roots], out / "config.json")
    env = env_make(ecfg)
    bank = Bank.load(roots.data / bcfg.bank / "bank.json")
    d = dec.load(dcfg)
    adapter = adapters.build(acfg, d)
    smp = rollout.sampler(adapter, c.backend, d, rcfg)
    problems = {p.id: p for p in env.problems(bank.split)}
    ids = list(bank.held) if c.curve_split == "held" else bank.clean
    ident = run_ident(env.name, ecfg, dcfg, rcfg)
    for f in todo:
        name = Path(f).with_suffix("").as_posix().replace("/", "__")
        src = roots.out / f if (roots.out / f).is_file() else roots.data / f
        put(adapter, torch.load(src, map_location="cpu", weights_only=False))
        grades = evaluate(smp, env, [problems[i] for i in ids], decode.at(c.curve_n, c.curve_seed), c.curve_chunks,
                          out / f"{name}_chunks", adapter=adapter_fingerprint(adapter.state_dict()))
        atomic_json({"file": f, "verdict": verdict(grades, smp.backend, decode, bank, ident), "grades": grades}, out / f"{name}.json")
        print(f"curve {c.curve_out} {f}: {mean_rate(grades):.4f}", flush=True)
    smp.close()


def mean_rate(grades: dict[str, list[float]]) -> float:
    return sum(sum(g) / len(g) for g in grades.values()) / len(grades)


if __name__ == "__main__":
    main(sys.argv[1:])
