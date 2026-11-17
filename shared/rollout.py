from __future__ import annotations

import enum
import gc
import json
import multiprocessing as mp
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from shared.adapter import Adapter
from shared.attach import Attached, after_rows, attach, positions, splice
from shared.decoder import Decoder, DecoderCfg
from shared.types import Backend, Completion, Sampling

STOP_CHECK_EVERY = 16


class Quant(enum.Enum):
    bf16 = "bf16"
    fp8 = "fp8"


@dataclass(frozen=True)
class RolloutCfg:
    engine_seed: int = 0
    gpu_frac: float = 0.45
    tp: int = 1
    ep: bool = False
    max_model_len: int = 2048
    quant: Quant = Quant.bf16
    spec: str | None = None
    max_num_seqs: int | None = None
    engine_groups: tuple[str, ...] = ()
    score_stop: bool = False
    hf_batch: int = 32
    hf_multi: bool = False
    hf_chunk: int = 0


@dataclass(frozen=True)
class Sampler:
    backend: Backend
    sample: Callable[[list[str], tuple[str, ...], Sampling], list[list[Completion]]]
    close: Callable[[], None]


def truncate_at_stop(text: str, stops: Sequence[str]) -> str:
    for s in stops:
        i = text.find(s)
        if i >= 0:
            text = text[:i]
    return text


def rollout_ids(tok, text: str, ids: Sequence[int], max_tokens: int, score_stop: bool) -> tuple[list[int], str]:
    ids = list(ids)[:max_tokens]
    if not score_stop and tok.decode(ids).startswith(text):
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(tok.decode(ids[:mid])) >= len(text):
                hi = mid
            else:
                lo = mid + 1
        ids = ids[:lo]
    if tok.decode(ids).startswith(text):
        return ids, text
    return ids, tok.decode(ids)


def completion(tok, prompt_ids: Sequence[int], text: str, eng_ids: Sequence[int], eng_lp: Sequence[float],
               s: Sampling, cfg: RolloutCfg) -> Completion:
    r, text = rollout_ids(tok, text, eng_ids, s.max_tokens, cfg.score_stop)
    assert len(eng_lp) >= len(r), (len(eng_lp), len(r))
    return Completion(tuple(prompt_ids), tuple(r), text, tuple(float(x) for x in eng_lp[:len(r)]))


def engine_kwargs(cfg: RolloutCfg, dcfg: DecoderCfg, entry: Attached, seed: int, work: str) -> dict:
    kw: dict[str, Any] = dict(model=dcfg.model, dtype="bfloat16", tensor_parallel_size=cfg.tp,
                              gpu_memory_utilization=cfg.gpu_frac, max_model_len=cfg.max_model_len, seed=seed,
                              enable_prompt_embeds=True, enable_expert_parallel=cfg.ep, logprobs_mode="raw_logprobs")
    if dcfg.trust_remote_code:
        kw["trust_remote_code"] = True
    if cfg.spec is not None:
        kw["speculative_config"] = json.loads(cfg.spec)
    if cfg.quant is Quant.fp8:
        kw["quantization"] = "fp8"
    if cfg.max_num_seqs is not None:
        kw["max_num_seqs"] = cfg.max_num_seqs
    if entry.own_compile_cache:
        kw["compilation_config"] = {"cache_dir": os.path.join(work, "compile")}
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    kw.update(entry.engine_kw)
    return kw


def _close_llm(llm) -> None:
    llm.llm_engine.engine_core.shutdown()
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate(llm, reqs: list[dict], sps: list[dict], gen_kw: dict) -> list[list[tuple[list[int], list[float], str]]]:
    from vllm import SamplingParams
    outs = llm.generate(reqs, [SamplingParams(**k) for k in sps], use_tqdm=False, **gen_kw)
    res = []
    for o in outs:
        row = []
        for c in o.outputs:
            eid = list(c.token_ids)
            row.append((eid, [d[t].logprob for d, t in zip(c.logprobs, eid)], c.text))
        res.append(row)
    return res


class _InProcessEngine:
    def __init__(self, kw: dict):
        from vllm import LLM
        self.llm = LLM(**kw)

    def generate(self, reqs, sps, gen_kw):
        return _generate(self.llm, reqs, sps, gen_kw)

    def collective_rpc(self, method, args=()):
        return self.llm.collective_rpc(method, args=args)

    def reset_prefix_cache(self):
        return self.llm.reset_prefix_cache()

    def close(self):
        _close_llm(self.llm)


def _engine_main(conn, gpus: str, kw: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    try:
        eng = _InProcessEngine(kw)
    except BaseException as e:  # noqa: BLE001
        conn.send(("err", repr(e)))
        return
    conn.send(("ok", None))
    while True:
        op, args = conn.recv()
        try:
            conn.send(("ok", getattr(eng, op)(*args)))
        except BaseException as e:  # noqa: BLE001
            conn.send(("err", repr(e)))
        if op == "close":
            return


class _SpawnedEngine:
    def __init__(self, gpus: str, kw: dict):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_engine_main, args=(child, gpus, kw))
        self.proc.start()
        child.close()
        self.gpus = gpus

    def _result(self):
        tag, val = self.conn.recv()
        if tag == "err":
            raise RuntimeError(f"vLLM engine on GPUs {self.gpus}: {val}")
        return val

    def _call(self, op, *args):
        self.conn.send((op, args))
        return self._result()

    def wait(self) -> None:
        self._result()

    def generate(self, reqs, sps, gen_kw):
        return self._call("generate", reqs, sps, gen_kw)

    def collective_rpc(self, method, args=()):
        return self._call("collective_rpc", method, args)

    def reset_prefix_cache(self):
        return self._call("reset_prefix_cache")

    def close(self):
        self._call("close")
        self.proc.join()


_M64 = (1 << 64) - 1


def _splitmix(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def problem_seed(seed: int, i: int) -> int:
    """62-bit seed of problem i: engines give sample k seed + k, so [seed, seed + n) must not meet another problem's."""
    return _splitmix(_splitmix(seed & _M64) ^ (i & _M64)) >> 2


def _split(n: int, m: int) -> list[range]:
    b = [n * k // m for k in range(m + 1)]
    return [range(b[k], b[k + 1]) for k in range(m)]


def _vllm_sampler(adapter: Adapter, decoder: Decoder, cfg: RolloutCfg) -> Sampler:
    entry = attach(adapter, Backend.vllm, decoder)
    work = tempfile.mkdtemp(prefix="sampler_")
    if cfg.engine_groups:
        llms = [_SpawnedEngine(g, engine_kwargs(cfg, decoder.cfg, entry, cfg.engine_seed + 1000 * k, work))
                for k, g in enumerate(cfg.engine_groups)]
        for llm in llms:
            llm.wait()
    else:
        llms = [_InProcessEngine(engine_kwargs(cfg, decoder.cfg, entry, cfg.engine_seed, work))]
    tok = decoder.tok

    def sample(prompts: list[str], stops: tuple[str, ...], s: Sampling) -> list[list[Completion]]:
        ids = [tok.encode(p, add_special_tokens=False) for p in prompts]
        gen_kw = entry.push(llms, work)
        reqs = [entry.request(x) for x in ids]
        parts = _split(len(prompts), len(llms))

        def run(k: int) -> list:
            r = parts[k]
            if not len(r):
                return []
            sps = [dict(n=s.n, temperature=s.temperature, top_p=s.top_p, max_tokens=s.max_tokens, stop=list(stops),
                        logprobs=0, seed=problem_seed(s.seed, i)) for i in r]
            return llms[k].generate([reqs[i] for i in r], sps, gen_kw[k])
        if len(llms) == 1:
            outs = run(0)
        else:
            with ThreadPoolExecutor(len(llms)) as ex:
                outs = [o for res in ex.map(run, range(len(llms))) for o in res]
        return [[completion(tok, x, truncate_at_stop(text, stops), eid, lp, s, cfg) for eid, lp, text in o]
                for x, o in zip(ids, outs)]

    def close() -> None:
        for llm in llms:
            llm.close()
        llms.clear()
        shutil.rmtree(work, ignore_errors=True)
    return Sampler(Backend.vllm, sample, close)


def _draw(logits: torch.Tensor, temperature: float, top_p: float, gen: torch.Generator,
          k: int = 1024) -> torch.Tensor:
    if temperature == 0.0:
        return logits.argmax(-1, keepdim=True)
    probs = torch.softmax(logits.float() / temperature, -1)
    if top_p >= 1.0:
        return torch.multinomial(probs, 1, generator=gen)
    sp, si = torch.topk(probs, min(k, probs.shape[-1]), dim=-1)
    if (sp.sum(-1) < top_p).any():
        sp, si = torch.sort(probs, descending=True, dim=-1)
    cum = torch.cumsum(sp, -1)
    sp = sp.masked_fill(cum - sp > top_p, 0.0)
    idx = torch.multinomial(sp / sp.sum(-1, keepdim=True).clamp_min(1e-12), 1, generator=gen)
    return si.gather(-1, idx)


def _hf_sampler(adapter: Adapter, decoder: Decoder, cfg: RolloutCfg) -> Sampler:
    if decoder.cfg.attn != "eager":
        raise ValueError("hf_eager sampling needs attn=eager: sdpa cached decode disagrees with its own full "
                         "forward (meanKL 0.28-0.44, argmax 0.92)")
    entry = attach(adapter, Backend.hf_eager, decoder)
    model, tok, dev = entry.net.model, decoder.tok, entry.net.dev
    emb = entry.net.emb

    def prefix(ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        pe = emb(torch.tensor(ids, device=dev)).float()
        soft = entry.rows(ids).detach().float().to(dev)
        k, P = soft.shape[0], len(ids)
        ctx = splice(entry.row_tail, pe, soft)
        return ctx.to(decoder.dtype).unsqueeze(0), positions(entry, P, k, P + k, dev).unsqueeze(0)

    def expand(past, b):
        from transformers.cache_utils import DynamicCache
        out = DynamicCache()
        for li, lay in enumerate(past.layers):
            out.update(lay.keys.expand(b, -1, -1, -1).contiguous(), lay.values.expand(b, -1, -1, -1).contiguous(), li)
        return out

    def cont(past, logits0, b, s: Sampling, stops, gen: torch.Generator, mask=None, pos0=None):
        eos = tok.eos_token_id
        lsm = lambda lg, c: torch.log_softmax(lg.float(), -1).gather(-1, c)[:, 0]
        cur = _draw(logits0, s.temperature, s.top_p, gen)
        toks, lps = [cur[:, 0]], [lsm(logits0, cur)]
        end_at = torch.full((b,), -1, dtype=torch.long, device=dev)
        end_at = torch.where(cur[:, 0] == eos, torch.zeros_like(end_at), end_at)
        stopped, cut = [False] * b, [None] * b
        for step in range(1, s.max_tokens):
            kw = {}
            if mask is not None:
                kw["attention_mask"] = torch.cat([mask, torch.ones(b, step, dtype=mask.dtype, device=dev)], 1)
            if pos0 is not None:
                kw["position_ids"] = (pos0 + step - 1).unsqueeze(1)
            o = model(input_ids=cur, past_key_values=past, use_cache=True, **kw)
            past = o.past_key_values
            lg = o.logits[:, -1, :]
            cur = _draw(lg, s.temperature, s.top_p, gen)
            toks.append(cur[:, 0])
            lps.append(lsm(lg, cur))
            end_at = torch.where((cur[:, 0] == eos) & (end_at < 0), torch.full_like(end_at, step), end_at)
            if step % STOP_CHECK_EVERY == 0 or step == s.max_tokens - 1:
                grid = torch.stack(toks, 1).tolist()
                ends = end_at.tolist()
                done = True
                for j in range(b):
                    if stopped[j] or ends[j] >= 0:
                        continue
                    txt = tok.decode(grid[j], skip_special_tokens=True)
                    if any(x in txt for x in stops):
                        stopped[j], cut[j] = True, txt
                    else:
                        done = False
                if done:
                    break
        grid = torch.stack(toks, 1).tolist()
        lgrid = torch.stack(lps, 1).tolist()
        ends = end_at.tolist()
        out = []
        for j in range(b):
            e = ends[j] if ends[j] >= 0 else len(grid[j])
            text = cut[j] if cut[j] is not None else tok.decode(grid[j][:e], skip_special_tokens=True)
            out.append((truncate_at_stop(text, stops), grid[j][:e], lgrid[j][:e]))
        return out

    @torch.no_grad()
    def one(ids: list[int], s: Sampling, stops, i: int) -> list[tuple]:
        ctx, pos = prefix(ids)
        res = []
        with entry.armed(len(ids)):
            o = model(inputs_embeds=ctx, position_ids=pos, use_cache=True)
            past, logits0 = o.past_key_values, o.logits[:, -1, :].float()
            for c in range(0, s.n, cfg.hf_batch):
                b = min(cfg.hf_batch, s.n - c)
                gen = torch.Generator(device=dev).manual_seed(problem_seed(s.seed, i) + c)
                first = torch.full((b,), len(ids), dtype=torch.long, device=dev) if entry.pos0 else None
                res += cont(expand(past, b), logits0.expand(b, -1).contiguous(), b, s, stops, gen, pos0=first)
        return res

    @torch.no_grad()
    def multi(all_ids: list[list[int]], s: Sampling, stops, first: int = 0) -> list[list[tuple]]:
        from transformers.cache_utils import DynamicCache
        n = s.n
        Pmax = max(len(x) for x in all_ids)
        logits0, Ks, caches = [], [], []
        for x in all_ids:
            ctx, pos = prefix(x)
            Ks.append(ctx.shape[1] - len(x))
            with entry.armed(len(x)):
                o = model(inputs_embeds=ctx, position_ids=pos, use_cache=True)
            caches.append(o.past_key_values)
            logits0.append(o.logits[:, -1, :].float().expand(n, -1))
        T = max(len(x) + k for x, k in zip(all_ids, Ks))  # partners drawn per prompt: slot counts K differ
        keys: list[list[torch.Tensor]] = [[] for _ in caches[0].layers]
        vals: list[list[torch.Tensor]] = [[] for _ in caches[0].layers]
        for c in caches:
            for li, lay in enumerate(c.layers):
                kk, vv = lay.keys, lay.values
                if pad := T - kk.shape[2]:
                    z = torch.zeros(1, kk.shape[1], pad, kk.shape[3], dtype=kk.dtype, device=kk.device)
                    kk, vv = torch.cat([z, kk], 2), torch.cat([z, vv], 2)
                keys[li].append(kk.expand(n, -1, -1, -1))
                vals[li].append(vv.expand(n, -1, -1, -1))
        del caches
        past = DynamicCache()
        for li in range(len(keys)):
            past.update(torch.cat(keys[li], 0).contiguous(), torch.cat(vals[li], 0).contiguous(), li)
        B = len(all_ids) * n
        mask = torch.ones(B, T, dtype=torch.long, device=dev)
        pos0 = torch.empty(B, dtype=torch.long, device=dev)
        for j, (x, k) in enumerate(zip(all_ids, Ks)):
            sl = slice(j * n, (j + 1) * n)
            mask[sl, : T - len(x) - k] = 0
            pos0[sl] = after_rows(entry, len(x), k)
        gen = torch.Generator(device=dev).manual_seed(problem_seed(s.seed, first))
        with entry.armed(Pmax):
            flat = cont(past, torch.cat(logits0, 0), B, s, stops, gen, mask=mask, pos0=pos0)
        return [flat[j * n:(j + 1) * n] for j in range(len(all_ids))]

    def sample(prompts: list[str], stops: tuple[str, ...], s: Sampling) -> list[list[Completion]]:
        ids = [tok.encode(p, add_special_tokens=False) for p in prompts]
        if cfg.hf_multi:
            n = cfg.hf_chunk or len(ids)
            raw = [r for c in range(0, len(ids), n) for r in multi(ids[c:c + n], s, stops, c)]
        else:
            raw = [one(x, s, stops, i) for i, x in enumerate(ids)]
        return [[completion(tok, x, t, gi, gl, s, cfg) for t, gi, gl in r] for x, r in zip(ids, raw)]

    def close() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return Sampler(Backend.hf_eager, sample, close)


_SAMPLERS = {Backend.vllm: _vllm_sampler, Backend.hf_eager: _hf_sampler}


def sampler(adapter: Adapter, backend: Backend, decoder: Decoder, cfg: RolloutCfg) -> Sampler:
    return _SAMPLERS[backend](adapter, decoder, cfg)
