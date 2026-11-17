"""Add new problems to the held-out set of a calibrated bank, drawn exactly as the bank's own problems were.
    python tools/extend_bank.py BASE NEW <the `shared.cli bank` flags BASE was drawn with, without --bank>
BASE, NEW: bank names under --data. The env's problems that BASE never calibrated (absent from its selected.json)
go through BASE's own pipeline (build_bank, same calibration window, chunks and solved filter, in NEW/ext/); every
one that passes joins BASE's held-out set. BASE's train and probe are kept, so runs trained on BASE are scored on
more problems of the same distribution, and nothing they trained or picked a checkpoint on moves into the score.
Run it with --adapter identity and the same backend, decoder, decode and bank flags as BASE (BASE/config.json)."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import adapters
from envs import env_from
from shared import backprop, decoder as dec, rollout
from shared.adapter import AdapterKind
from shared.cli import RunChoice
from shared.config import dump, parse, parse_known
from shared.eval import Bank, BankCfg, build_bank, run_ident
from shared.fs import Roots, atomic_json, read_json
from shared.types import DecodeCfg


class Only:
    """The env with only the problems in keep."""

    def __init__(self, env, keep: set[str]) -> None:
        self._env, self._keep = env, keep

    def __getattr__(self, name):
        return getattr(self._env, name)

    def problems(self, split: str):
        return [p for p in self._env.problems(split) if p.id in self._keep]


def main(argv: list[str]) -> None:
    base_name, new_name, *argv = argv
    (sel,), rest = parse_known(RunChoice, argv=argv)
    env_cls, env_make = env_from(sel.env, rest)
    ecfg, acfg, dcfg, rcfg, pcfg, bcfg, decode, roots = parse(
        env_cls, adapters.AdapterCfg, dec.DecoderCfg, rollout.RolloutCfg, backprop.BackpropCfg, BankCfg, DecodeCfg,
        Roots, argv=rest + ["--bank", f"{new_name}/ext"])
    if acfg.adapter is not AdapterKind.identity:
        raise ValueError("a bank is drawn with --adapter identity")
    base_dir, new_dir = roots.data / base_name, roots.data / new_name
    base = Bank.load(base_dir / "bank.json")
    if base.decode != decode or base.split != bcfg.bank_split:
        raise SystemExit(f"{base_name} was drawn with decode {base.decode}, split {base.split}: pass the same flags")
    was = read_json(base_dir / "config.json")["BankCfg"]
    for f in ("select", "cal_n", "cal_seed", "cal_lo", "cal_hi", "n_chunks", "n_per_chunk", "gen_seed", "pool_hi"):
        if str(was[f]) != str(dataclasses.asdict(bcfg)[f]).removeprefix("BankSelect."):
            raise SystemExit(f"--{f.replace('_', '-')} differs from {base_name}'s ({was[f]})")

    env = env_make(ecfg)
    decoder = dec.load(dcfg)
    sampler = rollout.sampler(adapters.build(acfg, decoder), sel.backend, decoder, rcfg)
    new_dir.mkdir(parents=True, exist_ok=True)
    dump([sel, ecfg, acfg, dcfg, rcfg, pcfg, bcfg, decode, roots], new_dir / "config.json")
    out = extend(base_dir, new_dir, env, sampler, bcfg, decode, (ecfg, dcfg, rcfg), run_ident(env.name, ecfg, dcfg, rcfg))
    sampler.close()
    print(f"{new_name}: held {len(base.held)} -> {len(out.held)}, clean {len(out.clean)}, "
          f"train {len(out.train)} and probe {len(out.probe)} as in {base_name}", flush=True)


def extend(base_dir: Path, new_dir: Path, env, sampler, bcfg: BankCfg, decode: DecodeCfg, identity, ident) -> Bank:
    base = Bank.load(base_dir / "bank.json")
    seen = set(read_json(base_dir / "selected.json")["rate1"])
    new = {p.id for p in env.problems(bcfg.bank_split)} - seen
    if not new:
        raise SystemExit(f"every problem of the env was already calibrated for {base_dir.name}")
    print(f"{len(new)} problems {base_dir.name} never calibrated", flush=True)
    # every new problem is calibrated (no cal_max cut), every one in the window kept (no cal_keep cut), and the
    # sub-bank's own train/held split is ignored: all of its solved problems are held out here
    cfg = dataclasses.replace(bcfg, bank=new_dir / "ext", cal_max=len(new), cal_keep=len(new), min_pool=0)
    sub = build_bank(sampler, Only(env, new), cfg, decode, new_dir / "ext", identity, ident)
    added = tuple(sorted(set(sub.train) | set(sub.held)))
    out = Bank(backend=base.backend, split=base.split, decode=base.decode, rates={**base.rates, **sub.rates},
               train=base.train, held=base.held + added, probe=base.probe, pool_hi=base.pool_hi, ident=base.ident)
    in_window = len(read_json(new_dir / "ext" / "selected.json")["selected"])
    atomic_json({"base": base_dir.name, "calibrated": len(new), "in_window": in_window, "added": list(added)},
                new_dir / "extension.json")
    out.save(new_dir / "bank.json")
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
