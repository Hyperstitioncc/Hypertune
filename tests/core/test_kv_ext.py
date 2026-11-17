import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fakes"))

from adapters import AdapterCfg, build  # noqa: E402
from adapters.identity import Identity  # noqa: E402
from adapters.kv_prefix import KVCart, KVPrefix, KVRead, load_units  # noqa: E402
from algorithms.grpo import GRPO, GRPOCfg, IrrMeasure  # noqa: E402
from conftest import V, as_decoder, tiny_model  # noqa: E402
from shared.adapter import Adapter, AdapterKind  # noqa: E402
from shared.attach import KV_GRAPH_WORKER, attach, dsm_gate, kv_payload  # noqa: E402
from shared.backprop import BackpropCfg, ScoreMode, logprobs  # noqa: E402
from shared.types import Backend, Completion  # noqa: E402

K = 4


def kv_module(dec, replace=False, pos0=False, remove_rest=False, graph=False, seed=3):
    c = dec.model.config
    L, Hkv, hd = c.num_hidden_layers, c.num_key_value_heads, 16
    g = torch.Generator().manual_seed(seed)
    base = (torch.randn(K, dec.d, generator=g) * 0.5).to(torch.bfloat16)
    units = load_units("all", L, Hkv, K)
    if remove_rest:
        units[:, ::2] = False
    kf0 = torch.randn(L, K, Hkv, hd, generator=g).bfloat16().float() if replace else torch.zeros(L, K, Hkv, hd)
    vf0 = torch.randn(L, K, Hkv, hd, generator=g).bfloat16().float() if replace else torch.zeros(L, K, Hkv, hd)
    m = KVPrefix(base, L, Hkv, hd, units, remove_rest, replace, kf0, vf0, pos0, graph)
    with torch.no_grad():
        m.dk.copy_((torch.randn(m.dk.shape, generator=g) * 0.3).to(torch.bfloat16).float())
        m.dv.copy_((torch.randn(m.dv.shape, generator=g) * 0.3).to(torch.bfloat16).float())
        if replace:
            m.dk.zero_()
            m.dv.zero_()
    return m


def cart_module(dec, gb, seed=4):
    c = dec.model.config
    g = torch.Generator().manual_seed(seed)
    kc = torch.randn(c.num_hidden_layers, K, c.num_key_value_heads, 16, generator=g)
    vc = torch.randn(c.num_hidden_layers, K, c.num_key_value_heads, 16, generator=g)
    m = KVCart(kc, vc, c.num_attention_heads, gb, dec.model.model.layers[0].self_attn.scaling)
    with torch.no_grad():
        m.kg.normal_(0, 0.5, generator=g)
    return m


def rows(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [Completion(tuple(torch.randint(0, V, (9,), generator=g).tolist()),
                       tuple(torch.randint(0, V, (6,), generator=g).tolist()), "", ())]


def cand(rs):
    return [torch.tensor(r.ids)[:, None] for r in rs]


# ---------------------------------------------------------------- vLLM-shaped Qwen3-MoE (the surface the patches touch)

class VAttn(nn.Module):
    def __init__(self, hf, rope, H, Hkv):
        super().__init__()
        self.hf, self.rope = hf, rope
        self.head_dim, self.num_heads, self.num_kv_heads = hf.head_dim, H, Hkv
        self.q_size, self.kv_size = H * hf.head_dim, Hkv * hf.head_dim
        self.q_norm, self.k_norm = hf.q_norm, hf.k_norm

    def qkv_proj(self, x):
        return torch.cat([self.hf.q_proj(x), self.hf.k_proj(x), self.hf.v_proj(x)], -1), None

    def o_proj(self, x):
        return self.hf.o_proj(x), None

    def rotary_emb(self, positions, q, k):
        from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb
        T, hd = q.shape[0], self.head_dim
        cos, sin = self.rope(q[None], positions[None])
        qh, kh = apply_rotary_pos_emb(q.view(1, T, -1, hd).transpose(1, 2), k.view(1, T, -1, hd).transpose(1, 2), cos, sin)
        return qh.transpose(1, 2).reshape(T, -1), kh.transpose(1, 2).reshape(T, -1)

    def attn(self, q, k, v):
        T, hd = q.shape[0], self.head_dim
        o = F.scaled_dot_product_attention(q.view(T, -1, hd).transpose(0, 1)[None], k.view(T, -1, hd).transpose(0, 1)[None],
                                           v.view(T, -1, hd).transpose(0, 1)[None], is_causal=True, enable_gqa=True)
        return o[0].transpose(0, 1).reshape(T, -1)

    def forward(self, positions, hidden_states):
        raise AssertionError("unpatched")


class VInner(nn.Module):
    def __init__(self, hf_model, attn):
        super().__init__()
        c = hf_model.config
        self.hfl = hf_model.model.layers
        self.norm = hf_model.model.norm
        self.layers = [SimpleNamespace(self_attn=attn(l.self_attn, hf_model.model.rotary_emb, c.num_attention_heads,
                                                       c.num_key_value_heads)) for l in self.hfl]

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        h = inputs_embeds
        for hl, vl in zip(self.hfl, self.layers):
            h = h + vl.self_attn(positions=positions, hidden_states=hl.input_layernorm(h))
            h = h + hl.mlp(hl.post_attention_layernorm(h)[None])[0]
        return self.norm(h)


class VModel(nn.Module):
    def __init__(self, hf_model, attn):
        super().__init__()
        self.model = VInner(hf_model, attn)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)


def hf_ref(dec, adapter, x, pos, P):
    e = attach(adapter, Backend.hf_eager, dec)
    with torch.no_grad(), e.armed(P):
        return dec.model.model(inputs_embeds=x[None], position_ids=pos[None]).last_hidden_state[0]


def context(dec, me, P=9):
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, V, (P + 5,), generator=g)
    with torch.no_grad():
        e = dec.emb(ids)
    x = torch.cat([e[:P], me.base.float(), e[P:]])
    pos = torch.arange(x.shape[0])
    if me.pos0:
        pos[P:P + K] = 0
        pos[P + K:] -= K
    return x, pos


def graph_model():
    from shared import kv_graph
    A, M = type("GAttn", (VAttn,), {}), type("GModel", (VModel,), {})
    kv_graph.patch(A, M, lambda: torch.device("cpu"))
    return lambda hf: M(hf, A)


@pytest.mark.parametrize("variant", ["delta", "pos0", "replace"])
def test_kv_graph_matches_hf(variant, tmp_path):
    from shared import kv_graph
    model = graph_model()
    dec = as_decoder(tiny_model())
    me = kv_module(dec, replace=variant == "replace", pos0=variant == "pos0", graph=True)
    a = Adapter(AdapterKind.kv_prefix, me)
    path = str(tmp_path / "kv.pt")
    torch.save(kv_payload(me), path)
    x, pos = context(dec, me)
    want = hf_ref(dec, a, x, pos, 9)
    seq = torch.arange(x.shape[0])
    w = kv_graph.KVGraphWorker()
    w.model_runner = SimpleNamespace(model=model(dec.model))
    w.set_kv_delta(path)
    with torch.no_grad():
        got = w.model_runner.model(None, seq, None, x)
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()
    me0 = kv_module(dec, graph=True, seed=9)
    with torch.no_grad():
        me0.dk.zero_()
        me0.dv.zero_()
    x0, _ = context(dec, me0)
    w = kv_graph.KVGraphWorker()
    w.model_runner = SimpleNamespace(model=model(dec.model))
    torch.save(kv_payload(me0), path)
    w.set_kv_delta(path)
    with torch.no_grad():
        h0 = w.model_runner.model(None, torch.arange(x0.shape[0]), None, x0)
        ref0 = dec.model.model(inputs_embeds=x0[None], position_ids=torch.arange(x0.shape[0])[None]).last_hidden_state[0]
    assert torch.allclose(h0, ref0, atol=1e-5), "zero deltas must be the plain model"


def test_row_positions_write_a_sequence_with_rows_k_later():
    from shared.kv_graph import row_positions
    # one flat engine batch: a prefill [3 prompt tokens, 2 rows, last prompt token, 1 answer token], a prefill with
    # no rows, then two decode tokens of other requests
    pos = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 41, 17])
    hit = torch.tensor([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
    tight = row_positions(pos, hit)
    assert tight.tolist() == [2, 3, 4, 2, 2, 5, 6, 0, 1, 2, 41, 17]
    # the same sequence with rows at 0 and later tokens numbered 2 earlier, all moved 2 later
    assert (tight[:7] - 2).tolist() == [0, 1, 2, 0, 0, 3, 4]
    assert row_positions(pos, torch.zeros(12, dtype=torch.bool)).tolist() == pos.tolist()


@pytest.mark.parametrize("graph", [True, False])
def test_kv_workers_match_hf_with_rows_at_position_0(graph, tmp_path):
    from shared import attach as attach_mod, kv_graph
    from shared.attach import positions
    dec = as_decoder(tiny_model())
    me = kv_module(dec, replace=True, pos0=True, graph=graph)
    a = Adapter(AdapterKind.kv_prefix, me)
    path = str(tmp_path / "kv.pt")
    torch.save(kv_payload(me), path)
    P = 9
    x, _ = context(dec, me, P)
    e = attach(a, Backend.hf_eager, dec)
    want = hf_ref(dec, a, x, positions(e, P, K, x.shape[0], x.device), P)
    gapped = hf_ref(dec, a, x, torch.arange(x.shape[0]).index_put((torch.arange(P, P + K),), torch.tensor(0)), P)
    assert not torch.allclose(want, gapped, atol=1e-3), "the two layouts should differ"
    if graph:
        w = kv_graph.KVGraphWorker()
        w.model_runner = SimpleNamespace(model=graph_model()(dec.model))
    else:
        w = attach_mod.KVInjectWorker()
        w.model_runner = SimpleNamespace(model=VModel(dec.model, VAttn))
    w.set_kv_delta(path)
    if not graph:                       # the worker holds bf16 tables for a bf16 engine; this stand-in model is fp32
        w._kvi.update(dk=w._kvi["dk"].float(), dv=w._kvi["dv"].float())
    with torch.no_grad():
        got = w.model_runner.model(None, torch.arange(x.shape[0]), None, x)
    assert torch.allclose(got, want, atol=2e-2 if not graph else 2e-4), (got - want).abs().max()


def test_kv_graph_attach(tiny_dir):
    from shared.decoder import DecoderCfg, load
    dec = load(DecoderCfg(model=tiny_dir, dtype="fp32", attn="sdpa", experts_impl="eager", device="cpu"))
    e = attach(Adapter(AdapterKind.kv_prefix, kv_module(dec, graph=True)), Backend.vllm, dec)
    assert e.engine_kw == {"worker_extension_cls": KV_GRAPH_WORKER} and e.own_compile_cache
    e = attach(Adapter(AdapterKind.kv_prefix, kv_module(dec)), Backend.vllm, dec)
    assert e.engine_kw["enforce_eager"] and not e.own_compile_cache
    with pytest.raises(ValueError, match="eager-only"):
        attach(Adapter(AdapterKind.kv_prefix, kv_module(dec, remove_rest=True, graph=True)), Backend.vllm, dec)


@pytest.mark.parametrize("graph", [False, True])
def test_kv_vllm_sampler(tiny_dir, graph):
    from shared.decoder import DecoderCfg, load
    from shared.rollout import RolloutCfg, sampler
    from shared.types import Sampling
    dec = load(DecoderCfg(model=tiny_dir, dtype="fp32", attn="sdpa", experts_impl="eager", device="cpu"))
    a = build(AdapterCfg(adapter=AdapterKind.kv_prefix, k=4, init_seed=3, kv_graph=graph), dec)
    smp = sampler(a, Backend.vllm, dec, RolloutCfg())
    smp.sample(["Problem: 1+1?\nSolution:"], (), Sampling(n=1, temperature=0.8, top_p=0.95, max_tokens=3, seed=1))
    smp.close()


@pytest.mark.parametrize("graph", [False, True])
def test_kv_vllm_sampler_pos0_with_partners(tiny_dir, graph, tmp_path):
    from adapters.kv_prefix import stack
    from shared.decoder import DecoderCfg, head_dim, load
    from shared.rollout import RolloutCfg, sampler
    from shared.types import Sampling
    dec = load(DecoderCfg(model=tiny_dir, dtype="fp32", attn="sdpa", experts_impl="eager", device="cpu"))
    c = dec.model.config
    L, Hkv, hd, k = c.num_hidden_layers, c.num_key_value_heads, head_dim(dec.model), 4
    g = torch.Generator().manual_seed(2)
    cart = lambda: {"soft": torch.randn(k, dec.d, generator=g) * 0.5, "kfix": torch.randn(L, k, Hkv, hd, generator=g),
                    "vfix": torch.randn(L, k, Hkv, hd, generator=g)}
    torch.save(cart(), tmp_path / "fix.pt")
    torch.save(stack([cart(), cart()], 1), tmp_path / "partners.pt")
    a = build(AdapterCfg(adapter=AdapterKind.kv_prefix, k=k, row_tail=True, pos0=True, kv_graph=graph,
                         init_path=str(tmp_path / "fix.pt"), kv_fix_path=str(tmp_path / "fix.pt"),
                         kv_partners=str(tmp_path / "partners.pt"), kv_partner_p=1.0), dec)
    e = attach(a, Backend.vllm, dec)
    assert e.engine_kw["enable_prefix_caching"] is False and e.engine_kw["enable_chunked_prefill"] is False and e.row_tail
    ids = dec.tok.encode("Problem: 1+1?\nSolution:", add_special_tokens=False)
    assert e.request(ids)["prompt_embeds"].shape[0] == len(ids) + k
    payload = kv_payload(a.module)
    assert payload["dk"].shape == (L, 3 * k, Hkv, hd) and payload["base"].shape[0] == 3 * k and payload["rm"].shape[1] == 3 * k
    a.module.present = 5
    assert e.request(ids)["prompt_embeds"].shape[0] == len(ids) + 3 * k
    smp = sampler(a, Backend.vllm, dec, RolloutCfg())
    out = smp.sample(["Problem: 1+1?\nSolution:"], (), Sampling(n=2, temperature=0.8, top_p=0.95, max_tokens=3, seed=1))
    assert len(out[0]) == 2
    smp.close()


# ---------------------------------------------------------------- dsm cartridge

def test_dsm_closed_gate_is_base():
    dec = as_decoder(tiny_model())
    a = Adapter(AdapterKind.kv_prefix, cart_module(dec, -1e4))
    rs = rows()
    got = logprobs(dec, a, rs, cand(rs), BackpropCfg())[0]
    want = logprobs(dec, Adapter(AdapterKind.identity, Identity()), rs, cand(rs), BackpropCfg())[0]
    assert torch.equal(got, want)


def test_dsm_open_gate_reads_the_slots():
    dec = as_decoder(tiny_model())
    me = cart_module(dec, 1e4)
    with torch.no_grad():
        me.vc.copy_(me.vc[:, :1].expand_as(me.vc))
    e = attach(Adapter(AdapterKind.kv_prefix, me), Backend.hf_eager, dec)
    seen = []
    with torch.no_grad(), e.armed(7):
        h = dec.model.model.layers[1].self_attn.o_proj.register_forward_pre_hook(lambda m, a: seen.append(a[0]))
        dec.model.model(inputs_embeds=dec.emb(torch.arange(7))[None])
    h.remove()
    H, Hkv = dec.model.config.num_attention_heads, dec.model.config.num_key_value_heads
    want = me.vc[1, 0].detach().repeat_interleave(H // Hkv, 0)
    assert torch.allclose(seen[0].view(7, H, 16), want.expand(7, H, 16), atol=1e-6)


@pytest.mark.parametrize("mode", ["prefix", "row"])
def test_dsm_grads(mode):
    dec = as_decoder(tiny_model())
    me = cart_module(dec, -1.0)
    a = Adapter(AdapterKind.kv_prefix, me)
    rs = rows()
    e = attach(a, Backend.hf_eager, dec)
    assert e.k == 0 and e.prompt_grad
    lp = logprobs(dec, a, rs, cand(rs), BackpropCfg(score_mode=ScoreMode[mode]))[0]
    x = dec.emb(torch.tensor(list(rs[0].prompt_ids) + list(rs[0].ids)))
    with e.armed(9):
        ref = F.log_softmax(dec.model(inputs_embeds=x[None]).logits[0].float(), -1)[8:14].gather(1, cand(rs)[0])
    assert torch.allclose(lp, ref, atol=1e-5)
    lp.sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in (me.kc, me.vc, me.kg))


def test_dsm_build_and_vllm(tiny_dir, tmp_path):
    from shared.decoder import DecoderCfg, load
    from shared.rollout import RolloutCfg, sampler
    from shared.types import Sampling
    dec = load(DecoderCfg(model=tiny_dir, dtype="fp32", attn="sdpa", experts_impl="eager", device="cpu"))
    c = dec.model.config
    fx = tmp_path / "fix.pt"
    torch.save({"kfix": torch.randn(c.num_hidden_layers, K, c.num_key_value_heads, 16),
                "vfix": torch.randn(c.num_hidden_layers, K, c.num_key_value_heads, 16)}, fx)
    a = build(AdapterCfg(adapter=AdapterKind.kv_prefix, kv_read=KVRead.dsm, kv_fix_path=str(fx), kv_gate_b=-4.0), dec)
    assert isinstance(a.module, KVCart) and len(a.params()) == 3
    smp = sampler(a, Backend.vllm, dec, RolloutCfg())
    smp.sample(["Problem: 1+1?\nSolution:"], (), Sampling(n=1, temperature=0.8, top_p=0.95, max_tokens=3, seed=1))
    smp.close()
    with pytest.raises(ValueError, match="CUDA-graph"):
        build(AdapterCfg(adapter=AdapterKind.kv_prefix, kv_read=KVRead.dsm, kv_fix_path=str(fx), kv_graph=True), dec)


# ---------------------------------------------------------------- irrelevance penalty kinds

def grpo(dec, kind, mode="prefix"):
    return GRPO(GRPOCfg(irr_pen=1.0, irr_kind=kind), None, None, dec, BackpropCfg(score_mode=ScoreMode[mode]), None, None,
                None, {}, {}, [])


@pytest.mark.parametrize("mode", ["prefix", "row"])
def test_irr_attn_is_cartridge_mass(mode):
    dec = as_decoder(tiny_model())
    me = kv_module(dec)
    a = Adapter(AdapterKind.kv_prefix, me)
    r = rows()[0]
    P, R = len(r.prompt_ids), len(r.ids)
    pen = grpo(dec, IrrMeasure.attn, mode).irr_one(a, r)
    ref_dec = as_decoder(tiny_model("eager"), attn="eager")
    e = attach(a, Backend.hf_eager, ref_dec)
    x = torch.cat([ref_dec.emb(torch.tensor(r.prompt_ids)), me.base.float(), ref_dec.emb(torch.tensor(r.ids))])
    with torch.no_grad(), e.armed(P):
        att = ref_dec.model.model(inputs_embeds=x[None], position_ids=torch.arange(x.shape[0])[None],
                                  output_attentions=True).attentions
    want = torch.stack([w[0, :, P + K:, P:P + K].sum(-1).mean() for w in att]).mean()
    assert abs(float(pen) - float(want)) < 1e-5, (float(pen), float(want))
    pen.backward()
    assert me.dk.grad is not None and me.dk.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["prefix", "row"])
def test_irr_write_is_the_rows_share_of_what_attention_writes(mode):
    dec = as_decoder(tiny_model())
    me = kv_module(dec, replace=True)
    a = Adapter(AdapterKind.kv_prefix, me)
    r = rows()[0]
    P = len(r.prompt_ids)
    pen = grpo(dec, IrrMeasure.write, mode).irr_one(a, r)
    ref_dec = as_decoder(tiny_model("eager"), attn="eager")
    e = attach(a, Backend.hf_eager, ref_dec)
    x = torch.cat([ref_dec.emb(torch.tensor(r.prompt_ids)), me.base.float(), ref_dec.emb(torch.tensor(r.ids))])
    vals = []
    with torch.no_grad(), e.armed(P):
        hs = [layer.self_attn.v_proj.register_forward_hook(lambda m, i, o: vals.append(o))
              for layer in ref_dec.model.model.layers]
        att = ref_dec.model.model(inputs_embeds=x[None], position_ids=torch.arange(x.shape[0])[None],
                                  output_attentions=True).attentions
        for h in hs:
            h.remove()
    want = []
    for w, v in zip(att, vals):
        v = v[0].view(-1, me.Hkv, me.hd).transpose(0, 1).repeat_interleave(w.shape[1] // me.Hkv, 0)
        w = w[0, :, P + K:]
        part = torch.einsum("hts,hsd->htd", w[:, :, P:P + K], v[:, P:P + K])
        whole = torch.einsum("hts,hsd->htd", w, v)
        want.append(part.pow(2).sum(-1).mean() / whole.pow(2).sum(-1).mean())
    want = torch.stack(want).mean()
    assert abs(float(pen.detach()) - float(want)) < 1e-5, (float(pen.detach()), float(want))
    pen.backward()
    assert me.dv.grad is not None and me.dv.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["prefix", "row"])
def test_irr_dsm_is_mean_gate(mode):
    dec = as_decoder(tiny_model())
    me = cart_module(dec, -1.0)
    a = Adapter(AdapterKind.kv_prefix, me)
    r = rows()[0]
    P = len(r.prompt_ids)
    pen = grpo(dec, IrrMeasure.attn, mode).irr_one(a, r)
    qs = {}
    hs = [l.self_attn.q_norm.register_forward_hook(lambda m, i, o, li=li: qs.__setitem__(li, o))
          for li, l in enumerate(dec.model.model.layers)]
    x = dec.emb(torch.tensor(list(r.prompt_ids) + list(r.ids)))
    with torch.no_grad(), attach(a, Backend.hf_eager, dec).armed(P):
        dec.model.model(inputs_embeds=x[None])
    for h in hs:
        h.remove()
    want = torch.stack([dsm_gate(qs[li][0, P:], me.kg[li], me.gb).mean() for li in sorted(qs)]).mean()
    assert abs(float(pen) - float(want)) < 1e-6
    pen.backward()
    assert me.kg.grad is not None and me.kg.grad.abs().sum() > 0


def test_irr_attn_needs_rows_in_context():
    from adapters.lora import LoRA
    dec = as_decoder(tiny_model())
    r = rows()[0]
    a = Adapter(AdapterKind.lora, LoRA(dec.model, 1, 32.0, ("q_proj",), (), 0, False))
    with pytest.raises(ValueError, match="no rows"):
        grpo(dec, IrrMeasure.attn).irr_one(a, r)


def test_stack_joins_cartridges_and_one_alone_is_unchanged(tmp_path):
    from adapters import AdapterCfg, build
    from adapters.kv_prefix import stack
    from shared.backprop import BackpropCfg, logprobs
    from shared.types import TokenRow
    dec = as_decoder(tiny_model())
    c = dec.model.config
    L, Hkv, hd, k = c.num_hidden_layers, c.num_key_value_heads, 16, 4
    g = torch.Generator().manual_seed(7)
    fix = {"soft": torch.randn(k, dec.d, generator=g) * 0.5, "kfix": torch.randn(L, k, Hkv, hd, generator=g),
           "vfix": torch.randn(L, k, Hkv, hd, generator=g)}
    torch.save(fix, tmp_path / "fix.pt")
    lay = dict(adapter=AdapterKind.kv_prefix, row_tail=True, pos0=True)
    a = build(AdapterCfg(k=k, init_path=str(tmp_path / "fix.pt"), kv_fix_path=str(tmp_path / "fix.pt"), **lay), dec)
    with torch.no_grad():
        a.module.dk.normal_(0, 0.3, generator=g)
        a.module.dv.normal_(0, 0.3, generator=g)
    trained = {"adapter": a.state_dict()}
    other = {"soft": torch.randn(k, dec.d, generator=g), "kfix": torch.randn(L, k, Hkv, hd, generator=g),
             "vfix": torch.randn(L, k, Hkv, hd, generator=g)}

    one = stack([trained], 0)
    assert torch.allclose(one["kfix"], fix["kfix"] + a.module.dk.detach()) and one["soft"].shape == (k, dec.d)
    two = stack([trained, other], 0)
    assert two["kfix"].shape == (L, 2 * k, Hkv, hd) and torch.equal(two["vfix"][:, k:], other["vfix"])
    assert len({tuple(r[:16].to(torch.bfloat16).tolist()) for r in two["soft"]}) == 2 * k
    with pytest.raises(ValueError, match="pure-K/V"):
        stack([{"adapter": {"kind": "kv_prefix", "state": {**a.state_dict()["state"], "kf0": torch.zeros(L, k, Hkv, hd)}}}], 0)

    torch.save(one, tmp_path / "one.pt")
    b = build(AdapterCfg(k=k, init_path=str(tmp_path / "one.pt"), kv_fix_path=str(tmp_path / "one.pt"), **lay), dec)
    rows = [TokenRow((5, 9, 3, 11, 7), (4, 8, 15, 2)), TokenRow((6, 2, 13), (9, 1))]
    cand = [torch.tensor(r.ids)[:, None] for r in rows]
    with torch.no_grad():
        want, got = (logprobs(dec, x, rows, cand, BackpropCfg()) for x in (a, b))
    assert all(torch.allclose(w, o, atol=1e-4) for w, o in zip(want, got)), "a cartridge stacked alone changed its outputs"


def test_partners_present_equal_the_stacked_cartridges(tmp_path):
    from adapters.kv_prefix import stack
    from shared.types import TokenRow
    dec = as_decoder(tiny_model())
    c = dec.model.config
    L, Hkv, hd, k = c.num_hidden_layers, c.num_key_value_heads, 16, 4
    g = torch.Generator().manual_seed(11)
    cart = lambda: {"soft": torch.randn(k, dec.d, generator=g) * 0.5, "kfix": torch.randn(L, k, Hkv, hd, generator=g),
                    "vfix": torch.randn(L, k, Hkv, hd, generator=g)}
    fix, p1, p2 = cart(), cart(), cart()
    torch.save(fix, tmp_path / "fix.pt")
    torch.save(stack([p1, p2], 5), tmp_path / "partners.pt")
    lay = dict(adapter=AdapterKind.kv_prefix, row_tail=True, pos0=True)
    own = dict(k=k, init_path=str(tmp_path / "fix.pt"), kv_fix_path=str(tmp_path / "fix.pt"), **lay)
    alone = build(AdapterCfg(**own), dec)
    with torch.no_grad():
        alone.module.dk.normal_(0, 0.3, generator=g)
        alone.module.dv.normal_(0, 0.3, generator=g)
    a = build(AdapterCfg(kv_partners=str(tmp_path / "partners.pt"), kv_partner_p=1.0, **own), dec)
    a.load(alone.state_dict())
    assert set(a.state_dict()["state"]) == set(alone.state_dict()["state"]), "partners leaked into the checkpoint"
    torch.save(stack([{"adapter": alone.state_dict()}, p1, p2], 5), tmp_path / "all.pt")
    both = build(AdapterCfg(k=3 * k, init_path=str(tmp_path / "all.pt"), kv_fix_path=str(tmp_path / "all.pt"), **lay), dec)

    rows = [TokenRow((5, 9, 3, 11, 7), (4, 8, 15, 2)), TokenRow((6, 2, 13), (9, 1))]
    cand = [torch.tensor(r.ids)[:, None] for r in rows]
    lp = lambda x: logprobs(dec, x, rows, cand, BackpropCfg())
    with torch.no_grad():
        ref_alone, ref_both, absent = lp(alone), lp(both), lp(a)
        a.module.present = 3
        present = lp(a)
    assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(absent, ref_alone)), "absent partners changed the outputs"
    assert all(torch.allclose(x, y, atol=1e-4) for x, y in zip(present, ref_both)), "present partners != the stacked cartridges"
    assert not all(torch.allclose(x, y, atol=1e-4) for x, y in zip(present, ref_alone))

    torch.stack([x.sum() for x in lp(a)]).sum().backward()
    assert a.module.dk.grad.abs().sum() > 0 and not a.module.pk.requires_grad
    a.module.partner_p = 0.5
    sizes = {a.module.rows(tuple(range(n, n + 5))).shape[0] for n in range(40)}
    assert sizes == {k, 2 * k, 3 * k} and torch.equal(a.module.rows((1, 2))[:k], a.module.base)
    a.module.present = None
    assert a.module.rows((1, 2)).shape[0] == k


def test_partners_switch_restores_what_it_found():
    from shared.adapter import partners

    class Mod:
        partnered, present = True, None
    ad = SimpleNamespace(module=Mod())
    with partners(ad, 7):
        assert ad.module.present == 7
        with partners(ad, None):
            assert ad.module.present is None
        assert ad.module.present == 7
    assert ad.module.present is None
    with partners(SimpleNamespace(module=object()), 7):
        pass


def test_find_rows_matches_match_rows():
    """the hashed lookup of the graph path against comparing every token with every row"""
    from adapters.kv_prefix import NDIM
    from shared.kv_graph import KEY_W, find_rows, match_rows, sort_keys
    g = torch.Generator().manual_seed(0)
    base = torch.randn(300, 32, generator=g).to(torch.bfloat16)
    e = torch.randn(500, 32, generator=g).to(torch.bfloat16)
    pick, is_row = torch.randint(0, 300, (500,), generator=g), torch.rand(500, generator=g) < 0.5
    e[is_row] = base[pick[is_row]]
    table = torch.full((512, NDIM), float("nan"), dtype=torch.bfloat16)
    table[:300] = base[:, :NDIM]
    keys, order = sort_keys(table[:300], KEY_W, 512)
    for x in (e, e.float()):
        cand, row = find_rows(x, table, keys, order, KEY_W)
        hit = match_rows(x, table)
        assert torch.equal(row, is_row) and torch.equal(row, hit.any(-1))
        assert torch.equal(cand[row], pick[row]) and torch.equal(cand[row], hit[row].to(torch.uint8).argmax(-1))
    assert len(set(keys[:300].tolist())) == 300
