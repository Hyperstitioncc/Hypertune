from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Sequence

from envs.base import Env, Problem
from shared.config import plain
from shared.fs import atomic_json, read_json
from shared.types import Backend, Completion, DecodeCfg, PartialMean, Sampling

if TYPE_CHECKING:
    from shared.adapter import Adapter
    from shared.backprop import BackpropCfg
    from shared.decoder import Decoder


class Sampler(Protocol):
    backend: Backend

    def sample(self, prompts: list[str], stops: tuple[str, ...], s: Sampling) -> list[list[Completion]]: ...


def chunk_seed(seed0: int, chunk: int) -> int:
    return seed0 * 1_000_003 + chunk * 10_007


def graded(sampler: Sampler, env: Env, problems: list[Problem],
           s: Sampling) -> list[list[tuple[Completion, float]]]:
    outs = sampler.sample([p.prompt for p in problems], env.stops, s)
    g = env.grade([(p, c.text) for p, cs in zip(problems, outs) for c in cs])
    res, k = [], 0
    for cs in outs:
        res.append(list(zip(cs, g[k:k + len(cs)])))
        k += len(cs)
    return res


def draw(sampler: Sampler, env: Env, problems: list[Problem], s: Sampling) -> dict[str, list[float]]:
    return {p.id: [r for _, r in row] for p, row in zip(problems, graded(sampler, env, problems, s))}


def _feed(h, v: object) -> None:
    import torch
    if isinstance(v, dict):
        h.update(b"{%d" % len(v))
        for k in sorted(v, key=str):
            h.update(repr(k).encode())
            _feed(h, v[k])
    elif isinstance(v, (list, tuple)):
        h.update(b"[%d" % len(v))
        for x in v:
            _feed(h, x)
    elif torch.is_tensor(v):
        t = v.detach().contiguous().reshape(-1)
        h.update(f"T{t.dtype}{tuple(v.shape)}".encode())
        for part in t.split(1 << 26):
            h.update(part.cpu().view(torch.uint8).numpy())
    else:
        h.update(repr(v).encode())


def adapter_fingerprint(state: dict) -> str:
    """blake2b over every tensor's full bytes (plus names, dtypes, shapes), streamed in 64M-element pieces:
    ~1 s per GB on CPU, so exact rather than a subsample, which could miss a sparse update."""
    h = hashlib.blake2b(digest_size=16)
    _feed(h, state)
    return h.hexdigest()


def evaluate(sampler: Sampler, env: Env, problems: list[Problem], s: Sampling, chunks: int,
             out_dir: Path, adapter: str | None = None) -> dict[str, list[float]]:
    agg: dict[str, list[float]] = {p.id: [] for p in problems}
    for c in range(chunks):
        path = Path(out_dir) / f"eval_chunk_{c}.json"
        sc = dataclasses.replace(s, seed=chunk_seed(s.seed, c))
        key = {**plain(sc), "backend": sampler.backend.name, "ids": [p.id for p in problems]}
        if adapter is not None:
            key["adapter"] = adapter
        rec = read_json(path) if path.exists() else None
        if rec is not None:
            was = {k: v for k, v in rec["key"].items() if k != "adapter"}
            if was != {k: v for k, v in key.items() if k != "adapter"}:
                raise ValueError(f"{path} was drawn under {rec['key']}, now {key}")
            if rec["key"].get("adapter") != adapter:
                print(f"{path}: drawn with adapter {rec['key'].get('adapter')}, now {adapter}: discarding it",
                      flush=True)
                rec = None
        if rec is None:
            rec = {"key": key, "grades": draw(sampler, env, problems, sc)}
            atomic_json(rec, path)
        for i, g in rec["grades"].items():
            agg[i] += g
    return agg


@dataclass(frozen=True)
class Bank:
    backend: Backend
    split: str
    decode: DecodeCfg
    rates: dict[str, float]
    train: tuple[str, ...]
    held: tuple[str, ...]
    probe: tuple[str, ...]
    pool_hi: float
    ident: dict | None = None

    @property
    def pool(self) -> list[str]:
        if any(i not in self.rates for i in self.train):
            raise ValueError("this bank holds no base rates for its train problems (--select splits): "
                             "train with --train-pool solved")
        return [i for i in self.train if 0.0 < self.rates[i] < self.pool_hi]

    @property
    def clean(self) -> list[str]:
        pr = set(self.probe)
        return [i for i in self.held if i not in pr]

    def save(self, path: Path) -> None:
        atomic_json(plain(self), path)

    @staticmethod
    def load(path: Path) -> Bank:
        d = read_json(path)
        return Bank(backend=Backend[d["backend"]], split=d["split"], decode=DecodeCfg(**d["decode"]), rates=d["rates"],
                    train=tuple(d["train"]), held=tuple(d["held"]), probe=tuple(d["probe"]), pool_hi=d["pool_hi"],
                    ident=d.get("ident"))


DECODER_IDENT = {"DecoderCfg": ("model", "dtype", "spell_reserved"), "RolloutCfg": ("quant", "spec")}


def run_ident(env_name: str, ecfg: object, dcfg: object, rcfg: object) -> dict:
    """What, besides backend and DecodeCfg, changes the draws: the decoder (weights, dtype, quantisation,
    speculative decoding, tokenizer spelling) and the env's prompt config (host paths left out)."""
    dec = {f: plain(getattr(c, f)) for c in (dcfg, rcfg) for f in DECODER_IDENT[type(c).__name__]}
    env = {f.name: plain(getattr(ecfg, f.name)) for f in dataclasses.fields(ecfg)
           if not isinstance(getattr(ecfg, f.name), Path)}
    return {"decoder": dec, "env": {"name": env_name, "cfg": env}}


_warned: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        print(f"warning: {msg}", flush=True)


def comparable(backend: Backend, decode: DecodeCfg, bank: Bank, ident: dict | None = None) -> bool:
    if backend is not bank.backend or decode != bank.decode:
        return False
    if bank.ident is None:
        _warn_once("bank.json predates decoder/env identity: pairing checks only backend and decode")
        return True
    if ident is None:
        _warn_once("no decoder/env identity given: pairing checks only backend and decode")
        return True
    bad = [k for k in ("decoder", "env") if bank.ident.get(k) is not None and bank.ident[k] != ident.get(k)]
    if bad:
        _warn_once(f"not paired with the bank: {bad} differ: bank {bank.ident}, now {ident}")
    return not bad


@dataclass(frozen=True)
class EvalRates:
    rates: dict[str, float]
    backend: Backend
    decode: DecodeCfg


def load_eval(path: Path) -> EvalRates:
    e = read_json(path)
    v = e["verdict"]
    return EvalRates(mean_rates(e["grades"]), Backend[v["backend"]], DecodeCfg(**v["decode"]))


def diffs(a: EvalRates, b: EvalRates) -> dict[str, float]:
    if (a.backend, a.decode) != (b.backend, b.decode):
        raise ValueError(f"refusing to pair a {a.backend.name} {a.decode} measurement with a "
                         f"{b.backend.name} {b.decode} one")
    missing = a.rates.keys() - b.rates.keys()
    if missing:
        raise ValueError(f"{len(missing)} problems have no paired rate, e.g. {sorted(missing)[:3]}")
    return {i: r - b.rates[i] for i, r in a.rates.items()}


def mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return mean, sd / math.sqrt(n)


def paired(rates: dict[str, float], backend: Backend, decode: DecodeCfg, bank: Bank) -> tuple[float, float]:
    return mean_se(list(diffs(EvalRates(rates, backend, decode), EvalRates(bank.rates, bank.backend,
                                                                          bank.decode)).values()))


def mean_rates(grades: dict[str, list[float]]) -> dict[str, float]:
    if not grades:
        raise ValueError("no problems were graded: the eval split is empty")
    empty = [i for i, g in grades.items() if not g]
    if empty:
        raise ValueError(f"{len(empty)} problems hold no grades (0 chunks or n=0?), e.g. {empty[:3]}")
    return {i: sum(g) / len(g) for i, g in grades.items()}


def verdict(grades: dict[str, list[float]], backend: Backend, decode: DecodeCfg, bank: Bank,
            ident: dict | None = None) -> dict:
    rates = mean_rates(grades)
    out = {"n_problems": len(rates), "n_per_problem": len(next(iter(grades.values()))),
           "absolute_rate": sum(rates.values()) / len(rates), "backend": backend.name,
           "decode": dataclasses.asdict(decode), "paired_with_bank": comparable(backend, decode, bank, ident)}
    if not out["paired_with_bank"]:
        return out
    up, se = paired(rates, backend, decode, bank)
    out.update(mean_uplift=up, se=se, passed=up > 0 and up > 2 * se,
               base_rate=sum(bank.rates[i] for i in rates) / len(rates))
    clean = {i: rates[i] for i in bank.clean if i in rates}
    if clean:
        cu, cse = paired(clean, backend, decode, bank)
        out["clean_subset"] = {"n_problems": len(clean), "mean_uplift": cu, "se": cse,
                               "passed": cu > 0 and cu > 2 * cse}
    return out


def probe_uplift(sampler: Sampler, env: Env, problems: list[Problem], bank: Bank, n: int, seed: int) -> PartialMean:
    got = EvalRates(mean_rates(draw(sampler, env, problems, bank.decode.at(n, seed))), sampler.backend, bank.decode)
    d = diffs(got, EvalRates(bank.rates, bank.backend, bank.decode))
    return PartialMean(sum(d.values()), len(d))


class BankSelect(enum.Enum):
    calibrated = "calibrated"
    all = "all"
    splits = "splits"


@dataclass(frozen=True)
class BankCfg:
    bank: Path
    bank_split: str = "train"
    select: BankSelect = BankSelect.calibrated
    cal_n: int = 64
    cal_max: int = 600
    cal_seed: int = 99
    cal_lo: float = 0.0
    cal_hi: float = 0.75
    cal_keep: int = 320
    n_chunks: int = 2
    n_per_chunk: int = 128
    gen_seed: int = 40
    keep_per_class: int = 4
    seed_wall: float = 0.3
    min_pool: int = 20
    pool_hi: float = 0.5
    probe_k: int = 24
    probe_seed: int = 11


def build_bank(sampler: Sampler, env: Env, cfg: BankCfg, decode: DecodeCfg, out_dir: Path,
               identity: Sequence[object], ident: dict | None = None) -> Bank:
    out_dir = Path(out_dir)
    key = {type(c).__name__: plain(c) for c in (cfg, decode, *identity)}
    key.update(env=env.name, backend=sampler.backend.name)
    kp = out_dir / "bank_key.json"
    if kp.exists() and read_json(kp) != key:
        raise SystemExit(f"bank definition changed since its files were drawn.\n  on disk: {read_json(kp)}\n"
                         f"  now:     {key}\nUse a new --bank.")
    atomic_json(key, kp)

    if cfg.select is BankSelect.splits:
        return split_bank(sampler, env, cfg, decode, out_dir, ident)
    by_id = {p.id: p for p in env.problems(cfg.bank_split)}
    sel_path = out_dir / "selected.json"
    if not sel_path.exists() and cfg.select is BankSelect.all:
        atomic_json({"rate1": {}, "selected": list(by_id)}, sel_path)
    if not sel_path.exists():
        cal = list(by_id.values())[:cfg.cal_max]
        rows = graded(sampler, env, cal, decode.at(cfg.cal_n, chunk_seed(cfg.cal_seed, 0)))
        rate1 = {p.id: sum(r for _, r in row) / len(row) for p, row in zip(cal, rows)}
        sel = [i for i in rate1 if cfg.cal_lo < rate1[i] < cfg.cal_hi]
        random.Random(0).shuffle(sel)
        atomic_json({"rate1": rate1, "selected": sel[:cfg.cal_keep]}, sel_path)
    selected = [by_id[i] for i in read_json(sel_path)["selected"]]
    if not selected:
        raise SystemExit(f"bank: no problem calibrated in ({cfg.cal_lo}, {cfg.cal_hi})")

    chunks = []
    for c in range(cfg.n_chunks):
        path = out_dir / f"chunk_{c}.json"
        if not path.exists():
            recs = {}
            rows = graded(sampler, env, selected, decode.at(cfg.n_per_chunk, chunk_seed(cfg.gen_seed, c)))
            for p, row in zip(selected, rows):
                kept: dict[str, list[str]] = {"correct": [], "incorrect": []}
                for comp, r in row:
                    cls = "correct" if r > 0 else "incorrect"
                    if len(kept[cls]) < cfg.keep_per_class:
                        kept[cls].append(comp.text)
                recs[p.id] = {"n": len(row), "n_correct": sum(r > 0 for _, r in row), **kept}
            atomic_json({"chunk": c, "records": recs}, path)
        chunks.append(read_json(path)["records"])

    ratios = []
    for p in selected:
        counts = [ch[p.id]["n_correct"] for ch in chunks]
        m = sum(counts) / len(counts)
        q = m / cfg.n_per_chunk
        if 0.02 < q < 0.98 and len(counts) > 1:
            obs = sum((x - m) ** 2 for x in counts) / (len(counts) - 1)
            ratios.append(obs / (cfg.n_per_chunk * q * (1 - q)))
    if cfg.n_chunks > 1:
        if not ratios:
            raise SystemExit("seed wall could not evaluate: no problem in (0.02, 0.98)")
        wall = sum(ratios) / len(ratios)
        if wall <= cfg.seed_wall:
            raise SystemExit(f"seed wall {wall:.3f}: chunks correlated, sampler not delivering independent draws")

    rates, correct, solved = {}, {}, []
    for p in selected:
        n = sum(ch[p.id]["n"] for ch in chunks)
        n_ok = sum(ch[p.id]["n_correct"] for ch in chunks)
        corr = [t for ch in chunks for t in ch[p.id]["correct"]][:2 * cfg.keep_per_class]
        inc = [t for ch in chunks for t in ch[p.id]["incorrect"]][:2 * cfg.keep_per_class]
        rates[p.id] = n_ok / n
        if corr:
            correct[p.id] = corr
        if n_ok > 0 and inc:
            solved.append(p.id)
    shuf = solved[:]
    random.Random(1).shuffle(shuf)
    n_held = max(1, len(shuf) // 5)
    held, train = shuf[:n_held], shuf[n_held:]
    bank = Bank(backend=sampler.backend, split=cfg.bank_split, decode=decode, rates=rates, train=tuple(train),
                held=tuple(held), probe=tuple(random.Random(cfg.probe_seed).sample(held, min(cfg.probe_k, len(held)))),
                pool_hi=cfg.pool_hi, ident=ident)
    if len(bank.pool) < cfg.min_pool:
        raise SystemExit(f"train pool below {cfg.pool_hi} is {len(bank.pool)}: widen the family")
    atomic_json(correct, out_dir / "correct.json")
    bank.save(out_dir / "bank.json")
    return bank


def split_bank(sampler: Sampler, env: Env, cfg: BankCfg, decode: DecodeCfg, out_dir: Path,
               ident: dict | None = None) -> Bank:
    if cfg.bank_split != "all":
        raise ValueError("--select splits takes the env's train / dev / test as given: pass --bank-split all")
    train, dev, test = (env.problems(s) for s in ("train", "dev", "test"))
    held = dev + test
    grades = evaluate(sampler, env, held, decode.at(cfg.n_per_chunk, cfg.gen_seed), cfg.n_chunks, out_dir)
    bank = Bank(backend=sampler.backend, split="all", decode=decode, rates=mean_rates(grades),
                train=tuple(p.id for p in train), held=tuple(p.id for p in held), probe=tuple(p.id for p in dev),
                pool_hi=cfg.pool_hi, ident=ident)
    bank.save(out_dir / "bank.json")
    return bank


def log_ratio_per_token(decoder: Decoder, adapter: Adapter, base: Adapter, rows: list[Completion],
                        bp: BackpropCfg) -> float:
    import torch
    from shared.backprop import logprobs
    rows = [r for r in rows if r.ids]
    if not rows:
        raise ValueError("log ratio: every KL rollout is empty")
    cand = [torch.tensor(r.ids).unsqueeze(-1) for r in rows]
    with torch.no_grad():
        a = logprobs(decoder, adapter, rows, cand, bp)
        b = logprobs(decoder, base, rows, cand, bp)
    return float(sum(x.sum() - y.sum() for x, y in zip(a, b))) / sum(len(r.ids) for r in rows)


def merge_banks(parts: dict[str, Bank], balance: bool, seed: int, probe_k: int, probe_seed: int) -> Bank:
    first = next(iter(parts.values()))
    for name, b in parts.items():
        if (b.backend, b.decode, b.pool_hi) != (first.backend, first.decode, first.pool_hi):
            raise ValueError(f"bank {name} was drawn under a different backend, decode or pool_hi")
    decs = [b.ident.get("decoder") if b.ident else None for b in parts.values()]
    known = [d for d in decs if d is not None]
    if any(d != known[0] for d in known):
        raise ValueError(f"the banks were drawn under different decoders: {decs}")
    pools = {n: list(b.pool) for n, b in parts.items()}
    if balance:
        k = min(len(p) for p in pools.values())
        pools = {n: random.Random(seed).sample(p, k) for n, p in pools.items()}
    keep = {n: set(p) for n, p in pools.items()}
    train = tuple(f"{n}/{i}" for n, b in parts.items() for i in b.train if i in keep[n])
    held = tuple(f"{n}/{i}" for n, b in parts.items() for i in b.held)
    return Bank(backend=first.backend, split="all", decode=first.decode,
                rates={f"{n}/{i}": r for n, b in parts.items() for i, r in b.rates.items()},
                train=train, held=held, probe=tuple(random.Random(probe_seed).sample(held, min(probe_k, len(held)))),
                pool_hi=first.pool_hi, ident={"decoder": known[0], "env": None} if known else None)
