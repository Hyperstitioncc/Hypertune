from __future__ import annotations

import enum
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from shared import dist
from shared.adapter import AdapterKind
from shared.config import dump, parse, parse_known, plain
from shared.eval import (Bank, BankCfg, adapter_fingerprint, build_bank, evaluate, log_ratio_per_token, merge_banks,
                         run_ident, verdict)
from shared.fs import Roots, atomic_json, read_json
from shared.optim import Optim, OptimCfg
from shared.train import TrainCfg, train
from shared.types import Backend, DecodeCfg


class AlgoName(enum.Enum):
    grpo = "grpo"
    offpolicy = "offpolicy"
    sft = "sft"
    distill = "distill"


@dataclass(frozen=True)
class RunChoice:
    env: str
    backend: Backend
    algo: AlgoName = AlgoName.grpo


class EvalSplit(enum.Enum):
    held = "held"
    clean = "clean"


@dataclass(frozen=True)
class EvalCfg:
    eval_run: str
    eval_adapter: str | None = None
    eval_n: int = 4
    eval_chunks: int = 4
    eval_seed: int = 6400
    eval_split: EvalSplit = EvalSplit.held
    eval_kl: bool = False
    eval_kl_n: int = 4
    eval_kl_seed: int = 4000


@dataclass(frozen=True)
class MergeCfg:
    merge_banks: tuple[str, ...]
    merge_names: tuple[str, ...]
    merge_out: str
    merge_balance: bool = True
    merge_seed: int = 0
    merge_probe_k: int = 24
    merge_probe_seed: int = 11


@dataclass(frozen=True)
class StackCfg:
    stack_parts: tuple[str, ...]
    stack_out: str
    stack_seed: int = 0


NOT_DEFINING = {"DecoderCfg": ("device", "gpus", "gpu_mem"), "RolloutCfg": ("gpu_frac", "max_num_seqs", "engine_groups")}


def eval_definition(cfgs: list[object]) -> dict:
    return {type(c).__name__: {k: v for k, v in plain(c).items() if k not in NOT_DEFINING.get(type(c).__name__, ())}
            for c in cfgs if not isinstance(c, Roots)}


def stack_part(roots: Roots, p: str) -> Path:
    for f in (roots.run(p) / "best.pt", roots.out / p, roots.data / p):
        if f.is_file():
            return f
    raise FileNotFoundError(f"{p}: no run best.pt under --out, and no file under --out or --data")


def stack(argv: list[str]) -> None:
    from adapters.kv_prefix import stack as stack_kv
    from adapters.lora import stack as stack_lora
    from shared.fs import atomic_torch
    scfg, roots = parse(StackCfg, Roots, argv=argv)
    parts = [torch.load(stack_part(roots, p), map_location="cpu", weights_only=True) for p in scfg.stack_parts]
    lora = parts[0].get("adapter", {}).get("kind") == "lora"
    joined = stack_lora(parts) if lora else stack_kv(parts, scfg.stack_seed)
    atomic_torch({**joined, "parts": list(scfg.stack_parts)}, roots.data / scfg.stack_out)


def merge(argv: list[str]) -> None:
    mcfg, roots = parse(MergeCfg, Roots, argv=argv)
    if len(mcfg.merge_banks) != len(mcfg.merge_names):
        raise ValueError("--merge-banks and --merge-names need one name per bank")
    out = roots.data / mcfg.merge_out
    out.mkdir(parents=True, exist_ok=True)
    dump([mcfg, roots], out / "config.json")
    parts = {n: Bank.load(roots.data / b / "bank.json") for n, b in zip(mcfg.merge_names, mcfg.merge_banks)}
    merge_banks(parts, mcfg.merge_balance, mcfg.merge_seed, mcfg.merge_probe_k, mcfg.merge_probe_seed).save(
        out / "bank.json")
    atomic_json({f"{n}/{i}": t for n, b in zip(mcfg.merge_names, mcfg.merge_banks)
                 for i, t in read_json(roots.data / b / "correct.json").items()}, out / "correct.json")


def algo_from(name: AlgoName):
    if name is AlgoName.grpo:
        from algorithms import grpo
        return grpo.GRPOCfg, grpo.make
    if name is AlgoName.offpolicy:
        from algorithms import offpolicy
        return offpolicy.OffPolicyCfg, offpolicy.make
    if name is AlgoName.sft:
        from algorithms import sft
        return sft.SFTCfg, sft.make
    from algorithms import distill
    return distill.DistillCfg, distill.make


def main(argv: list[str]) -> None:
    import adapters
    from envs import env_from
    from shared import backprop, decoder as dec, rollout

    if argv and argv[0] == "merge":
        return merge(argv[1:])
    if argv and argv[0] == "stack":
        return stack(argv[1:])
    if not argv or argv[0] not in ("bank", "train", "eval"):
        raise SystemExit("usage: python -m shared.cli {bank,train,eval} --env NAME [--help] ... | merge ... | stack ...")
    cmd, argv = argv[0], argv[1:]
    (sel,), rest = parse_known(RunChoice, argv=argv)
    env_cls, env_make = env_from(sel.env, rest)
    common = (env_cls, adapters.AdapterCfg, dec.DecoderCfg, rollout.RolloutCfg, backprop.BackpropCfg, BankCfg, DecodeCfg,
              Roots)

    if cmd == "bank":
        ecfg, acfg, dcfg, rcfg, pcfg, bcfg, decode, roots = parse(*common, argv=rest)
        if acfg.adapter is not AdapterKind.identity:
            raise ValueError(f"a bank is drawn with --adapter identity, not {acfg.adapter.name}")
        out = roots.data / bcfg.bank
        out.mkdir(parents=True, exist_ok=True)
        dump([sel, ecfg, acfg, dcfg, rcfg, pcfg, bcfg, decode, roots], out / "config.json")
        decoder = dec.load(dcfg)
        adapter = adapters.build(acfg, decoder)
        sampler = rollout.sampler(adapter, sel.backend, decoder, rcfg)
        env = env_make(ecfg)
        build_bank(sampler, env, bcfg, decode, out, (ecfg, dcfg, rcfg), run_ident(env.name, ecfg, dcfg, rcfg))
        sampler.close()
        return

    if cmd == "eval":
        ecfg, acfg, dcfg, rcfg, pcfg, bcfg, decode, roots, vcfg = parse(*common, EvalCfg, argv=rest)
        run_dir = roots.run(vcfg.eval_run)
        definition = eval_definition([sel, ecfg, acfg, dcfg, rcfg, pcfg, bcfg, decode, vcfg])
        if (run_dir / "eval.json").exists():
            was = read_json(run_dir / "eval.json").get("config")
            if was is None:
                print(f"{run_dir / 'eval.json'} exists (written before evals stored their config): kept as is")
            elif was != definition:
                raise SystemExit(f"{run_dir / 'eval.json'} was written under a different eval config.\n"
                                 f"  on disk: {was}\n  now:     {definition}\n"
                                 "Use a new --eval-run (or --out), or delete that eval.json.")
            return
        ckpt = roots.out / vcfg.eval_adapter if vcfg.eval_adapter is not None else run_dir / "best.pt"
        trained = ckpt.exists()
        if vcfg.eval_adapter is not None and not trained:
            raise FileNotFoundError(f"--eval-adapter {ckpt} missing")
        if not trained and acfg.adapter is not AdapterKind.identity and acfg.init_path is None:
            raise FileNotFoundError(f"{run_dir / 'best.pt'} missing and no --init-path: nothing trained to evaluate")
        dump([sel, ecfg, acfg, dcfg, rcfg, pcfg, bcfg, decode, roots, vcfg], run_dir / "eval_config.json")
        env = env_make(ecfg)
        bank = Bank.load(roots.data / bcfg.bank / "bank.json")
        decoder = dec.load(dcfg)
        adapter = adapters.build(acfg, decoder)
        if trained:
            adapter.load(torch.load(ckpt, map_location="cpu", weights_only=False)["adapter"])
        fingerprint = adapter_fingerprint(adapter.state_dict())
        sampler = rollout.sampler(adapter, sel.backend, decoder, rcfg)
        problems = {p.id: p for p in env.problems(bank.split)}
        ids = list(bank.held) if vcfg.eval_split is EvalSplit.held else bank.clean
        grades = evaluate(sampler, env, [problems[i] for i in ids], decode.at(vcfg.eval_n, vcfg.eval_seed),
                          vcfg.eval_chunks, run_dir / "eval", fingerprint)
        out = {"verdict": verdict(grades, sampler.backend, decode, bank, run_ident(env.name, ecfg, dcfg, rcfg)),
               "grades": grades, "config": definition, "adapter_fingerprint": fingerprint}
        if vcfg.eval_kl:
            held = [problems[i] for i in ids]
            rows = [c for cs in sampler.sample([p.prompt for p in held], env.stops,
                                               decode.at(vcfg.eval_kl_n, vcfg.eval_kl_seed)) for c in cs]
            base = adapters.build(adapters.AdapterCfg(adapter=AdapterKind.identity), decoder)
            out["verdict"]["log_ratio_per_token"] = log_ratio_per_token(decoder, adapter, base, rows, pcfg)
        sampler.close()
        atomic_json(out, run_dir / "eval.json")
        return

    algo_cls, algo_make = algo_from(sel.algo)
    cfgs = parse(*common, OptimCfg, TrainCfg, dist.DistCfg, algo_cls, argv=rest)
    ecfg, acfg, dcfg, rcfg, pcfg, bcfg, decode, roots, ocfg, tcfg, xcfg, gcfg = cfgs
    dist.init(xcfg)
    run_dir = roots.run(tcfg.run)
    if dist.rank() == 0:
        dump([sel, *cfgs], run_dir / "config.json")
    env = env_make(ecfg)
    bank_dir = roots.data / bcfg.bank
    bank = Bank.load(bank_dir / "bank.json")
    decoder = dec.load(dcfg)
    adapter = adapters.build(acfg, decoder)
    if acfg.adapter is AdapterKind.thinker:
        first = {p.id: p for p in env.problems(bank.split)}[bank.pool[0]]
        adapter.module.calibrate(decoder.tok.encode(first.prompt, add_special_tokens=False))
    sampler = rollout.sampler(adapter, sel.backend, decoder, rcfg)
    algo = algo_make(gcfg, env, sampler, decoder, pcfg, bank, decode, bank_dir, roots.data)
    train(algo, adapter, env, Optim(adapter.params(), ocfg), bank, tcfg, run_dir)
    sampler.close()
    dist.teardown()


if __name__ == "__main__":
    import os
    import traceback
    try:
        main(sys.argv[1:])
    except SystemExit as e:  # a live vLLM engine process would otherwise keep a stopped run (and its GPUs) alive
        if e.code not in (None, 0):
            print(e.code, file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(e.code if isinstance(e.code, int) else int(e.code is not None))
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
