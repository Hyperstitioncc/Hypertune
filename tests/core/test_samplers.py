import dataclasses
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fakes"))

from adapters import AdapterCfg, build  # noqa: E402
from shared.adapter import AdapterKind  # noqa: E402
from shared.backprop import BackpropCfg, logprobs  # noqa: E402
from shared.decoder import DecoderCfg, load, load_embedding  # noqa: E402
from shared.rollout import RolloutCfg, sampler  # noqa: E402
from shared.types import Backend, Sampling  # noqa: E402

PROMPTS = ["Problem: What is 1+1?\nSolution:", "Problem: Name a prime.\nSolution:"]
STOPS = ("\nProblem:", "\n\nProblem:")


def check_rows(dec, adapter, out, s, tol):
    tok = dec.tok
    aligned = 0
    for p, row in zip(PROMPTS, out):
        assert len(row) == s.n
        for c in row:
            assert list(c.prompt_ids) == tok.encode(p, add_special_tokens=False)
            assert len(c.ids) <= s.max_tokens
            assert not any(x in c.text for x in STOPS)
            assert tok.decode(list(c.ids)).startswith(c.text) or list(c.ids) == tok.encode(c.text, add_special_tokens=False)
            if c.engine_lp and c.ids:
                aligned += 1
                assert len(c.engine_lp) == len(c.ids)
                with torch.no_grad():
                    lp = logprobs(dec, adapter, [c], [torch.tensor(c.ids)[:, None]], BackpropCfg())[0][:, 0]
                assert torch.allclose(lp, torch.tensor(c.engine_lp), atol=tol), (lp, c.engine_lp)
    assert aligned > 0


def test_load_embedding_matches_decoder(tiny_dir):
    dec = load(DecoderCfg(model=tiny_dir, dtype="fp32", attn="eager", experts_impl="eager", device="cpu"))
    assert torch.equal(load_embedding(dec.cfg).float(), dec.emb.weight)


@pytest.mark.parametrize("kind", ["identity", "kv_prefix", "kv_tail"])
@pytest.mark.parametrize("multi", [False, True])
def test_hf_sampler(tiny_dir, kind, multi):
    dec = load(DecoderCfg(model=tiny_dir, dtype="fp32", attn="eager", experts_impl="eager", device="cpu"))
    k = AdapterKind.identity if kind == "identity" else AdapterKind.kv_prefix
    a = build(AdapterCfg(adapter=k, k=4, row_tail=kind == "kv_tail", init_seed=3), dec)
    smp = sampler(a, Backend.hf_eager, dec, RolloutCfg(hf_batch=2, hf_multi=multi))
    s = Sampling(n=3, temperature=0.8, top_p=0.95, max_tokens=10, seed=5)
    out = smp.sample(PROMPTS, STOPS, s)
    again = smp.sample(PROMPTS, STOPS, s)
    assert [[c.ids for c in r] for r in out] == [[c.ids for c in r] for r in again], "seeded draws differ"
    check_rows(dec, a, out, s, 1e-3)
    smp.close()


def test_hf_multi_in_chunks_gives_the_same_greedy_answers(tiny_dir):
    dec = load(DecoderCfg(model=tiny_dir, dtype="fp32", attn="eager", experts_impl="eager", device="cpu"))
    a = build(AdapterCfg(adapter=AdapterKind.kv_prefix, k=4, init_seed=3, row_tail=True), dec)
    prompts = PROMPTS + ["Problem: Add 2 and 3.\nSolution:", "Problem: Is 9 odd?\nSolution:", "Problem: 7?\nSolution:"]
    s = Sampling(n=1, temperature=0.0, top_p=1.0, max_tokens=8, seed=5)
    whole = sampler(a, Backend.hf_eager, dec, RolloutCfg(hf_multi=True)).sample(prompts, STOPS, s)
    parts = sampler(a, Backend.hf_eager, dec, RolloutCfg(hf_multi=True, hf_chunk=2)).sample(prompts, STOPS, s)
    assert [[c.ids for c in r] for r in whole] == [[c.ids for c in r] for r in parts]
    assert len(parts) == len(prompts)


@pytest.mark.parametrize("multi", [False, True])
def test_hf_sampler_kv_tail_pos0(tiny_dir, multi):
    dec = load(DecoderCfg(model=tiny_dir, dtype="fp32", attn="eager", experts_impl="eager", device="cpu"))
    a = build(AdapterCfg(adapter=AdapterKind.kv_prefix, k=4, init_seed=3, row_tail=True, pos0=True), dec)
    with torch.no_grad():
        a.module.dk.normal_(0, 0.3)
        a.module.dv.normal_(0, 0.3)
    smp = sampler(a, Backend.hf_eager, dec, RolloutCfg(hf_batch=2, hf_multi=multi))
    s = Sampling(n=3, temperature=0.8, top_p=0.95, max_tokens=10, seed=5)
    check_rows(dec, a, smp.sample(PROMPTS, STOPS, s), s, 1e-3)
    smp.close()


@pytest.mark.parametrize("kind", ["identity", "lora", "lora_merge", "kv_prefix", "kv_tail", "full"])
def test_vllm_sampler(tiny_dir, kind):
    dec = load(DecoderCfg(model=tiny_dir, dtype="fp32", attn="sdpa", experts_impl="eager", device="cpu"))
    cfgs = {"identity": AdapterCfg(),
            "lora": AdapterCfg(adapter=AdapterKind.lora, lora_targets=("q_proj", "v_proj")),
            "lora_merge": AdapterCfg(adapter=AdapterKind.lora, lora_targets=("q_proj", "v_proj"), lora_merge=True),
            "kv_prefix": AdapterCfg(adapter=AdapterKind.kv_prefix, k=4, init_seed=3),
            "kv_tail": AdapterCfg(adapter=AdapterKind.kv_prefix, k=4, init_seed=3, row_tail=True),
            "full": AdapterCfg(adapter=AdapterKind.full)}
    a = build(cfgs[kind], dec)
    if kind.startswith("lora"):
        with torch.no_grad():
            for b in a.module.B:
                b.normal_(0, 0.05)
    smp = sampler(a, Backend.vllm, dec, RolloutCfg())
    s = Sampling(n=2, temperature=0.8, top_p=0.95, max_tokens=8, seed=11)
    out = smp.sample(PROMPTS, STOPS, s)
    if kind == "full":
        with torch.no_grad():
            dec.model.model.layers[0].self_attn.q_proj.weight.add_(0.1)
        out = smp.sample(PROMPTS, STOPS, s)
    if not kind.startswith("kv_"):
        check_rows(dec, a, out, s, 2e-2)
    smp.close()


def test_vllm_sampler_lora_with_partners(tiny_dir, tmp_path):
    from adapters.lora import stack
    from shared.adapter import partners
    dec = load(DecoderCfg(model=tiny_dir, dtype="fp32", attn="sdpa", experts_impl="eager", device="cpu"))
    lc = AdapterCfg(adapter=AdapterKind.lora, lora_targets=("q_proj", "v_proj"))
    ps = [build(dataclasses.replace(lc, init_seed=s), dec) for s in (1, 2)]
    for x in ps:
        with torch.no_grad():
            for b in x.module.B:
                b.normal_(0, 0.05)
    torch.save(stack([{"adapter": x.state_dict()} for x in ps]), tmp_path / "p.pt")
    a = build(dataclasses.replace(lc, lora_partners=str(tmp_path / "p.pt"), lora_partner_p=1.0), dec)
    with torch.no_grad():
        for b in a.module.B:
            b.normal_(0, 0.05)
    smp = sampler(a, Backend.vllm, dec, RolloutCfg())
    s = Sampling(n=2, temperature=0.8, top_p=0.95, max_tokens=8, seed=11)
    with partners(a, 4):
        out = smp.sample(PROMPTS, STOPS, s)
        check_rows(dec, a, out, s, 2e-2)
    alone = smp.sample(PROMPTS, STOPS, s)
    check_rows(dec, a, alone, s, 2e-2)
    smp.close()
