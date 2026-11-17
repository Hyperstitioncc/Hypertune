"""Does a real vLLM engine read a position-0 cartridge the way the HF scorer does? Samples a few prompts of a
domain through vLLM with the cartridge loaded, then scores the sampled tokens with shared.backprop.logprobs and
compares the two sets of log-probabilities token by token (the equality GRPO's importance weights rely on).
Also prints the greedy accuracy through vLLM on the first N test problems.
    python tools/check_vllm_pos0.py --env intent --intent-data DIR --model Qwen/Qwen3-30B-A3B-Base \
        --kv-file FILE.pt --graph true|false [--n 50] [--max-tokens 16]            (PYTHONPATH = the library root)"""
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

import adapters
from envs import env_from
from shared import decoder as dec_mod, rollout
from shared.adapter import AdapterKind
from shared.backprop import BackpropCfg, logprobs
from shared.config import parse, parse_known
from shared.types import Backend, Sampling


@dataclass(frozen=True)
class CheckCfg:
    env: str
    kv_file: str            # a kfix/vfix file, or "none": the plain model, as the yardstick for engine-vs-scorer noise
    graph: bool = False
    n: int = 50
    max_tokens: int = 16


def main() -> None:
    (c,), rest = parse_known(CheckCfg, argv=sys.argv[1:])
    env_cls, env_make = env_from(c.env, rest)
    ecfg, dcfg = parse(env_cls, dec_mod.DecoderCfg, argv=rest)
    env = env_make(ecfg)
    dec = dec_mod.load(dcfg)
    if c.kv_file == "none":
        a = adapters.build(adapters.AdapterCfg(adapter=AdapterKind.identity), dec)
    else:
        k = torch.load(c.kv_file, map_location="cpu")["kfix"].shape[1]
        a = adapters.build(adapters.AdapterCfg(adapter=AdapterKind.kv_prefix, k=k, row_tail=True, pos0=True, init_path=c.kv_file,
                                               kv_fix_path=c.kv_file, kv_graph=c.graph), dec)
    tag = "plain model" if c.kv_file == "none" else f"cartridge, graph={c.graph}"
    smp = rollout.sampler(a, Backend.vllm, dec, rollout.RolloutCfg(gpu_frac=0.42, max_model_len=4096))
    probs = env.problems("test")[:c.n]
    out = smp.sample([p.prompt for p in probs[:8]], env.stops, Sampling(n=4, temperature=0.8, top_p=0.95, max_tokens=c.max_tokens, seed=3))
    diffs = []
    for row in out:
        for comp in row:
            if not comp.ids:
                continue
            with torch.no_grad():
                lp = logprobs(dec, a, [comp], [torch.tensor(comp.ids)[:, None]], BackpropCfg())[0][:, 0].float().cpu()
            diffs.append((lp - torch.tensor(comp.engine_lp)).abs())
    d = torch.cat(diffs)
    print(f"RESULT {tag}: engine vs scorer log-prob on {d.numel()} sampled tokens (32 rollouts): mean {d.mean():.4f} "
          f"median {d.median():.4f} 95th pct {d.quantile(0.95):.4f} largest {d.max():.4f} nats")
    greedy = smp.sample([p.prompt for p in probs], env.stops, Sampling(n=1, temperature=0.0, top_p=1.0, max_tokens=c.max_tokens, seed=0))
    grades = env.grade([(p, r[0].text) for p, r in zip(probs, greedy)])
    print(f"RESULT {tag}: greedy through vLLM on {len(probs)} test problems: {sum(grades) / len(grades):.3f}; first answers "
          f"{[r[0].text.strip()[:30] for r in greedy[:3]]}")
    smp.close()


if __name__ == "__main__":          # vLLM starts its engine in a spawned process, which re-imports this file
    main()
