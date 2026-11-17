import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fakes"))

from adapters import AdapterCfg, build  # noqa: E402
from conftest import as_decoder, tiny_model  # noqa: E402
from shared.adapter import AdapterKind  # noqa: E402
from shared.backprop import BackpropCfg, logprobs  # noqa: E402
from shared.rollout import RolloutCfg, _splitmix, problem_seed, sampler  # noqa: E402
from shared.types import Backend, Sampling  # noqa: E402


class CharTok:
    """Printable ASCII -> ids 1..95 (tiny vocab 97), 0 is eos."""
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) - 31 for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i + 31) for i in ids if 0 < i < 96)


def eager_dec():
    return as_decoder(tiny_model("eager"), attn="eager", tok=CharTok())


def ids_of(out):
    return [[c.ids for c in r] for r in out]


# ---------------------------------------------------------------- (a) seed mixing

def test_problem_seed_ranges_are_disjoint_and_stable():
    assert _splitmix(0) == 0xE220A8397B1DCDAF, "not splitmix64"
    assert [problem_seed(0, 0), problem_seed(0, 1), problem_seed(5, 0), problem_seed(-1, 3)] == \
        [3008887562355236763, 156851287368183107, 4518720736667979917, 52491835732862560], "seeds changed across versions"
    n = 128
    seeds = sorted(problem_seed(s, i) for s in list(range(-3, 300)) + [2 ** 40, 12345678901] for i in range(200))
    assert all(0 <= x < 2 ** 62 for x in seeds)
    assert all(b - a >= n for a, b in zip(seeds, seeds[1:])), "two problems share a seed range [seed, seed + n)"
    assert [problem_seed(7, i) for i in range(50)] == [problem_seed(7, i) for i in range(50)]


def test_vllm_requests_get_mixed_seeds(tiny_dir, monkeypatch):
    import vllm
    from shared.decoder import DecoderCfg, load
    seen = []

    class Spy(vllm.SamplingParams):
        def __init__(self, **kw):
            seen.append(kw["seed"])
            super().__init__(**kw)
    monkeypatch.setattr(vllm, "SamplingParams", Spy)
    dec = load(DecoderCfg(model=tiny_dir, dtype="fp32", attn="sdpa", experts_impl="eager", device="cpu"))
    smp = sampler(build(AdapterCfg(), dec), Backend.vllm, dec, RolloutCfg())
    s = Sampling(n=4, temperature=0.8, top_p=0.95, max_tokens=4, seed=11)
    smp.sample(["Problem: 1+1?\nSolution:"] * 3, (), s)
    smp.close()
    assert seen == [problem_seed(11, i) for i in range(3)]
    assert all(abs(a - b) >= s.n for a in seen for b in seen if a != b)


# ---------------------------------------------------------------- (b) HF non-batched: one stream per problem

@pytest.mark.parametrize("kind", ["identity", "soft_block"])
def test_hf_one_gives_each_problem_its_own_stream(kind):
    dec = eager_dec()
    a = build(AdapterCfg(adapter=AdapterKind(kind), k=4, init_seed=3), dec)
    smp = sampler(a, Backend.hf_eager, dec, RolloutCfg(hf_batch=2))
    s = Sampling(n=4, temperature=1.0, top_p=1.0, max_tokens=8, seed=5)
    out = smp.sample(["What is 1+1? A:"] * 2, (), s)
    assert ids_of(out) == ids_of(smp.sample(["What is 1+1? A:"] * 2, (), s)), "seeded draws differ"
    assert ids_of(out)[0] != ids_of(out)[1], "two problems in one call drew from the same stream"
    smp.close()


# ---------------------------------------------------------------- (c) HF batched: per-prompt slot counts

PROMPTS = ["Q: 1+1? A:", "Q: name a prime. A:", "Q: add 2 and 3. A:", "Q: 9 odd? A:", "Q: 7? A:", "Q: x A:",
           "Q: what is 12 times 12? A:", "Q: hi A:"]


def partnered(tmp_path, pos0):
    from adapters.kv_prefix import stack
    dec = eager_dec()
    c = dec.model.config
    L, Hkv, hd, k = c.num_hidden_layers, c.num_key_value_heads, 16, 4
    g = torch.Generator().manual_seed(11)
    cart = lambda: {"soft": torch.randn(k, dec.d, generator=g) * 0.5, "kfix": torch.randn(L, k, Hkv, hd, generator=g),
                    "vfix": torch.randn(L, k, Hkv, hd, generator=g)}
    torch.save(cart(), tmp_path / "fix.pt")
    torch.save(stack([cart(), cart()], 5), tmp_path / "partners.pt")
    a = build(AdapterCfg(adapter=AdapterKind.kv_prefix, k=k, row_tail=True, pos0=pos0, init_path=str(tmp_path / "fix.pt"),
                         kv_fix_path=str(tmp_path / "fix.pt"), kv_partners=str(tmp_path / "partners.pt"),
                         kv_partner_p=0.5), dec)
    with torch.no_grad():
        a.module.dk.normal_(0, 0.3, generator=g)
        a.module.dv.normal_(0, 0.3, generator=g)
    a.module.present = 3
    Ks = [a.module.rows(dec.tok.encode(p)).shape[0] for p in PROMPTS]
    assert len(set(Ks)) > 1, f"test prompts all drew {Ks[0]} slots"
    return dec, a, Ks


@pytest.mark.parametrize("pos0", [False, True])
def test_hf_multi_with_per_prompt_slot_counts_matches_each_prompt_alone(tmp_path, pos0):
    """Greedy only: the batched path draws every row from one generator, so a sampled row depends on its batch
    position and neighbours; greedy rows must match the unpadded single-prompt paths exactly."""
    dec, a, _ = partnered(tmp_path, pos0)
    s = Sampling(n=2, temperature=0.0, top_p=1.0, max_tokens=8, seed=5)
    batched = sampler(a, Backend.hf_eager, dec, RolloutCfg(hf_multi=True)).sample(PROMPTS, (), s)
    single = sampler(a, Backend.hf_eager, dec, RolloutCfg()).sample(PROMPTS, (), s)
    multi_alone = [sampler(a, Backend.hf_eager, dec, RolloutCfg(hf_multi=True)).sample([p], (), s)[0] for p in PROMPTS]
    assert ids_of(batched) == ids_of(single) == ids_of(multi_alone)
    for rb, rs in zip(batched, single):
        for cb, cs in zip(rb, rs):
            assert torch.allclose(torch.tensor(cb.engine_lp), torch.tensor(cs.engine_lp), atol=1e-4)


@pytest.mark.parametrize("pos0", [False, True])
def test_hf_multi_sampled_logprobs_match_scoring_with_per_prompt_slot_counts(tmp_path, pos0):
    dec, a, _ = partnered(tmp_path, pos0)
    s = Sampling(n=3, temperature=0.8, top_p=0.95, max_tokens=6, seed=5)
    out = sampler(a, Backend.hf_eager, dec, RolloutCfg(hf_multi=True, hf_chunk=5)).sample(PROMPTS, (), s)
    assert len(out) == len(PROMPTS) and all(len(r) == s.n for r in out)
    scored = 0
    for r in out:
        for c in r:
            if not c.ids:
                continue
            with torch.no_grad():
                lp = logprobs(dec, a, [c], [torch.tensor(c.ids)[:, None]], BackpropCfg())[0][:, 0]
            assert torch.allclose(lp, torch.tensor(c.engine_lp), atol=1e-3), (lp, c.engine_lp)
            scored += 1
    assert scored > 0
