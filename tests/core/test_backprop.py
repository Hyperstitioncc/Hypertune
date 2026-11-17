import contextlib
import dataclasses
import json

import pytest
import torch
import torch.nn.functional as F

from adapters.identity import Identity
from adapters.kv_prefix import KVPrefix, load_units
from adapters.lora import LoRA
from conftest import V, as_decoder, tiny_cfg, tiny_model
from shared.adapter import Adapter, AdapterKind
from shared.attach import attach, save_peft
from shared.backprop import BackpropCfg, ScoreMode, logprobs
from shared.types import Backend, Completion

K = 16


def make_rows(seed=0):
    g = torch.Generator().manual_seed(seed)
    pA = torch.randint(0, V, (9,), generator=g).tolist()
    pB = torch.randint(0, V, (13,), generator=g).tolist()
    return [Completion(tuple(p), tuple(torch.randint(0, V, (L,), generator=g).tolist()), "", ())
            for p, L in [(pA, 5), (pB, 1), (pA, 9), (pA, 2), (pB, 7)]]


W = [0.3, -0.7, 1.1, 0.2, -0.4]


def ids_cand(rows):
    return [torch.tensor(r.ids).unsqueeze(1) for r in rows]


def naive(dec, adapter, row, cand):
    e = attach(adapter, Backend.hf_eager, dec)
    P = len(row.prompt_ids)
    soft = e.rows(row.prompt_ids).to(dec.dtype)
    k = soft.shape[0]
    emb = lambda ids: dec.emb(torch.tensor(list(ids)))
    pe = emb(row.prompt_ids)
    parts = [pe[:-1], soft, pe[-1:]] if (e.row_tail and k) else [pe, soft]
    x = torch.cat(parts + [emb(row.ids)])
    pos = torch.arange(x.shape[0])
    if e.pos0 and k:
        i = P - 1 if e.row_tail else P
        pos[i:i + k] = 0
        pos[i + k:] -= k
    with e.armed(P):
        lp = F.log_softmax(dec.model(inputs_embeds=x[None], position_ids=pos[None]).logits[0].float(), -1)
    return lp[P + k - 1:P + k - 1 + len(row.ids)].gather(1, cand)


@contextlib.contextmanager
def fa2_emulation():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def varlen(module, q, k, v, mask, dropout=0.0, scaling=None, **kw):
        pid = kw["position_ids"][0]
        seg = torch.cumsum((pid == 0).long(), 0)
        T = pid.shape[0]
        ar = torch.arange(T, device=pid.device)
        allow = (seg[:, None] == seg[None, :]) & (ar[None, :] <= ar[:, None])
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=allow[None, None], scale=scaling, enable_gqa=True)
        return o.transpose(1, 2).contiguous(), None
    had = "flash_attention_2" in ALL_ATTENTION_FUNCTIONS._local_mapping
    old = ALL_ATTENTION_FUNCTIONS._local_mapping.get("flash_attention_2")
    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = varlen
    try:
        yield
    finally:
        if had:
            ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = old
        else:
            del ALL_ATTENTION_FUNCTIONS["flash_attention_2"]


def modes(dec):
    out = []
    for mode in ("row", "prefix", "pack"):
        for budget in (8192, 30):
            out.append((mode, budget))
    return out


def run_mode(dec, mode, budget, fn):
    attn = dec.cfg.attn
    with contextlib.ExitStack() as stack:
        if mode == "pack":
            stack.enter_context(fa2_emulation())
            dec.model.config._attn_implementation = "flash_attention_2"
            dec.cfg = dataclasses.replace(dec.cfg, attn="flash_attention_2")
        try:
            return fn(BackpropCfg(score_mode=ScoreMode[mode], unit_tokens=budget))
        finally:
            dec.model.config._attn_implementation = attn
            dec.cfg = dataclasses.replace(dec.cfg, attn=attn)


def check(dec, adapter, rows, cand, weights, params, mode_list, atol=2e-3, gatol=1e-4, pos0=False, grtol=1e-3):
    want = [naive(dec, adapter, r, c).detach() for r, c in zip(rows, cand)]
    for q in params:
        q.grad = None
    (-sum(w * naive(dec, adapter, r, c).sum() for w, r, c in zip(weights, rows, cand))).backward()
    gref = [q.grad.clone() for q in params]
    assert all(g.abs().sum() > 0 for g in gref)
    for mode, budget in mode_list:
        if mode == "pack" and pos0:
            continue

        def go(cfg):
            for q in params:
                q.grad = None
            got = logprobs(dec, adapter, rows, cand, cfg)
            for g, w_ in zip(got, want):
                assert g.shape == w_.shape and torch.allclose(g, w_, atol=atol), (mode, budget, g, w_)
            (-sum(w * g.sum() for w, g in zip(weights, got))).backward()
            for q, gr in zip(params, gref):
                assert torch.allclose(q.grad, gr, atol=gatol, rtol=grtol), (mode, budget, (q.grad - gr).abs().max())
        run_mode(dec, mode, budget, go)


def kv_adapter(dec, row_tail=False, pos0=False):
    c = dec.model.config
    L, Hkv, hd = c.num_hidden_layers, c.num_key_value_heads, 16
    g = torch.Generator().manual_seed(1)
    base = (torch.randn(K, dec.d, generator=g) * 0.5).to(torch.bfloat16)
    z = torch.zeros(L, K, Hkv, hd)
    mod = KVPrefix(base, L, Hkv, hd, load_units("all", L, Hkv, K), False, False, z, z, pos0, False, row_tail)
    with torch.no_grad():
        mod.dk.normal_(0, 0.3, generator=g)
        mod.dv.normal_(0, 0.3, generator=g)
    return Adapter(AdapterKind.kv_prefix, mod)


def test_identity_and_topk():
    dec = as_decoder(tiny_model())
    rows = make_rows()
    g = torch.Generator().manual_seed(5)
    cand = [torch.cat([torch.tensor(r.ids)[:, None], torch.randint(0, V, (len(r.ids), 3), generator=g)], 1)
            for r in rows]
    a = kv_adapter(dec)
    check(dec, a, rows, cand, W, a.params(), modes(dec))
    ident = Adapter(AdapterKind.identity, Identity())
    with torch.no_grad():
        for mode, budget in modes(dec):
            def go(cfg):
                got = logprobs(dec, ident, rows, cand, cfg)
                for r, c, x in zip(rows, cand, got):
                    assert torch.allclose(x, naive(dec, ident, r, c), atol=2e-3)
            run_mode(dec, mode, budget, go)


def lora_adapter(dec, experts, seed=0):
    tg = ("q_proj", "k_proj", "v_proj", "o_proj") + (("gate_proj", "up_proj", "down_proj") if experts else ())
    mod = LoRA(dec.model, 1, 32.0, tg, (), seed, False)
    with torch.no_grad():
        for b in mod.B:
            b.normal_(0, 0.05 if experts else 0.02)
    return Adapter(AdapterKind.lora, mod)


@pytest.mark.parametrize("experts", [False, True])
def test_lora(experts, tmp_path):
    dec = as_decoder(tiny_model())
    a = lora_adapter(dec, experts)
    rows = make_rows()
    check(dec, a, rows, ids_cand(rows), W, a.params(), modes(dec))
    ident = Adapter(AdapterKind.identity, Identity())
    cfg = BackpropCfg()
    with torch.no_grad():
        base = logprobs(dec, ident, rows, ids_cand(rows), cfg)
        adapted = logprobs(dec, a, rows, ids_cand(rows), cfg)
        m0 = tiny_model()
        plain = [F.log_softmax(m0(input_ids=torch.tensor([list(r.prompt_ids + r.ids)])).logits[0].float(), -1)
                 [len(r.prompt_ids) - 1:-1].gather(1, torch.tensor(r.ids)[:, None]) for r in rows]
    assert all(torch.allclose(b, p, atol=2e-3) for b, p in zip(base, plain)), "identity != base model"
    assert max((x - y).abs().max().item() for x, y in zip(adapted, base)) > 1e-2, "LoRA had no effect"
    d = tmp_path
    save_peft(a.module, f"{d}/ad", "tiny")
    from safetensors.torch import load_file
    Wt = load_file(f"{d}/ad/adapter_model.safetensors")
    with open(f"{d}/ad/adapter_config.json") as f:
        c = json.load(f)
    s = c["lora_alpha"] / c["r"]
    m2 = tiny_model()
    mods = dict(m2.named_modules())
    I_ = tiny_cfg().moe_intermediate_size
    with torch.no_grad():
        for key, A in Wt.items():
            if not key.endswith("lora_A.weight"):
                continue
            name = key[len("base_model.model."):-len(".lora_A.weight")]
            dW = s * (Wt[key.replace("lora_A", "lora_B")].float() @ A.float())
            if ".experts." in name:
                base_, e, proj = name.rsplit(".", 2)
                xm, e = mods[base_], int(e)
                if proj == "gate_proj":
                    xm.gate_up_proj[e, :I_] += dW
                elif proj == "up_proj":
                    xm.gate_up_proj[e, I_:] += dW
                else:
                    xm.down_proj[e] += dW
            else:
                mods[name].weight += dW
        merged = logprobs(as_decoder(m2), ident, rows, ids_cand(rows), cfg)
    tol = 2e-2 if experts else 2e-3
    assert all(torch.allclose(x, y, atol=tol) for x, y in zip(merged, adapted)), "PEFT merge != hooked LoRA"


@pytest.mark.parametrize("variant", ["delta", "pos0", "remove_rest", "replace", "replace_pos0_tail"])
def test_kv_prefix(variant):
    dec = as_decoder(tiny_model())
    c = dec.model.config
    L, Hkv, hd = c.num_hidden_layers, c.num_key_value_heads, 16
    g = torch.Generator().manual_seed(3)
    base = (torch.randn(K, dec.d, generator=g) * 0.5).to(torch.bfloat16)
    units = load_units("all", L, Hkv, K)
    if variant == "remove_rest":
        units[:, ::2] = False
    replace = variant.startswith("replace")
    pos0, tail = "pos0" in variant, "tail" in variant
    kf0 = torch.randn(L, K, Hkv, hd, generator=g) if replace else torch.zeros(L, K, Hkv, hd)
    vf0 = torch.randn(L, K, Hkv, hd, generator=g) if replace else torch.zeros(L, K, Hkv, hd)
    mod = KVPrefix(base, L, Hkv, hd, units, variant == "remove_rest", replace, kf0, vf0, pos0, False, tail)
    with torch.no_grad():
        mod.dk.normal_(0, 0.3)
        mod.dv.normal_(0, 0.3)
    a = Adapter(AdapterKind.kv_prefix, mod)
    rows = make_rows()
    assert attach(a, Backend.hf_eager, dec).row_tail is tail
    ms = [(m, b) for m, b in modes(dec) if not (m == "pack" and (pos0 or variant == "remove_rest"))]
    check(dec, a, rows, ids_cand(rows), W, a.params(), ms, pos0=pos0)


def test_full():
    dec = as_decoder(tiny_model())
    from adapters.full import Full
    a = Adapter(AdapterKind.full, Full(dec.model))
    rows = make_rows()
    ps = [p for n, p in dec.model.named_parameters() if "q_proj" in n or "embed" in n]
    check(dec, a, rows, ids_cand(rows), W, ps, modes(dec), gatol=2e-4)


def test_eager_decoder_all_modes():
    dec = as_decoder(tiny_model("eager"), attn="eager")
    a = kv_adapter(dec)
    rows = make_rows()
    check(dec, a, rows, ids_cand(rows), W, a.params(), [m for m in modes(dec) if m[0] != "pack"])


def test_full_trains_a_copy_so_identity_stays_base():
    import adapters
    dec = as_decoder(tiny_model())
    full = adapters.build(adapters.AdapterCfg(adapter=AdapterKind.full), dec)
    base = adapters.build(adapters.AdapterCfg(adapter=AdapterKind.identity), dec)
    rows = make_rows()
    cand = ids_cand(rows)
    bp = BackpropCfg(score_mode=ScoreMode.row)
    before = [t.detach().clone() for t in logprobs(dec, base, rows, cand, bp)]
    assert all(not p.requires_grad for p in dec.model.parameters())
    with torch.no_grad():
        for p in full.params():
            p.add_(0.05 * torch.randn_like(p))
    after_base = logprobs(dec, base, rows, cand, bp)
    after_full = logprobs(dec, full, rows, cand, bp)
    assert all(torch.equal(a, b) for a, b in zip(before, after_base))
    assert any(not torch.allclose(a, b) for a, b in zip(before, after_full))


@pytest.mark.parametrize("experts", [False, True])
def test_lora_partners_present_equal_the_summed_loras(experts, tmp_path):
    from adapters.lora import stack
    from shared.adapter import partners
    dec = as_decoder(tiny_model())
    own, p1, p2 = (lora_adapter(dec, experts, seed=s) for s in (0, 1, 2))
    torch.save(stack([{"adapter": p.state_dict()} for p in (p1, p2)]), tmp_path / "partners.pt")
    a = lora_adapter(dec, experts)
    a.module.set_partners(*(torch.load(tmp_path / "partners.pt")[k] for k in ("lora_A", "lora_B", "sizes")), 1.0)
    a.load(own.state_dict())
    assert set(a.state_dict()["state"]) == set(own.state_dict()["state"]), "partners leaked into the checkpoint"
    tg = a.module.targets
    summed = Adapter(AdapterKind.lora, LoRA(dec.model, 3, 96.0, tg, (), 0, False))
    summed.module.load_stack(stack([{"adapter": x.state_dict()} for x in (own, p1, p2)]))
    rows = make_rows()
    cfg = BackpropCfg()
    with torch.no_grad():
        alone, both, absent = logprobs(dec, own, rows, ids_cand(rows), cfg), logprobs(dec, summed, rows, ids_cand(rows), cfg), \
            logprobs(dec, a, rows, ids_cand(rows), cfg)
        with partners(a, 3):
            present = logprobs(dec, a, rows, ids_cand(rows), cfg)
    assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(absent, alone)), "absent partners changed the outputs"
    assert all(torch.allclose(x, y, atol=1e-4) for x, y in zip(present, both)), "present partners != the summed LoRAs"
    assert not all(torch.allclose(x, y, atol=1e-3) for x, y in zip(present, alone))
    with partners(a, 3):
        torch.stack([x.sum() for x in logprobs(dec, a, rows, ids_cand(rows), cfg)]).sum().backward()
        save_peft(a.module, f"{tmp_path}/ad", "tiny")
    assert all(p.grad is not None for p in a.params()) and not any(t.requires_grad for t in a.module.pA)
    c = json.load(open(f"{tmp_path}/ad/adapter_config.json"))
    assert c["r"] == 3 and c["lora_alpha"] / c["r"] == a.module.scale
    a.module.partner_p = 0.5
    assert {len(a.module.drawn()) for a.module.present in range(40)} == {0, 1, 2}


@pytest.mark.parametrize("experts", [False, True])
def test_lora_write_is_the_share_of_what_the_adapted_modules_write(experts):
    from algorithms.grpo import IrrMeasure, irr_measure
    dec = as_decoder(tiny_model())
    a = lora_adapter(dec, experts)
    r = make_rows()[0]
    pen = irr_measure(dec, BackpropCfg(), a, r, IrrMeasure.write)
    assert 0 < float(pen) < 1
    pen.backward()
    assert all(b.grad is not None and b.grad.abs().sum() > 0 for b in a.module.B)
    with torch.no_grad():
        for b in a.module.B:
            b.zero_()
    assert float(irr_measure(dec, BackpropCfg(), a, r, IrrMeasure.write)) == 0.0
    assert a.module.write_acc is None


@pytest.mark.parametrize("norm", ["row_mean", "token_sum"])
def test_clone_step_row_by_row_is_the_batch_gradient(norm):
    from algorithms.sft import LossNorm, clone_loss, clone_step
    from shared.backprop import seq_lp
    dec = as_decoder(tiny_model())
    rows = [r for r in make_rows() if r.ids]
    norm, cfg = LossNorm(norm), BackpropCfg()
    a = lora_adapter(dec, True)
    for p in a.params():
        p.grad = None
    clone_loss(seq_lp(dec, a, rows, cfg), [len(r.ids) for r in rows], norm).backward()
    want = [p.grad.clone() for p in a.params()]
    for p in a.params():
        p.grad = None
    m = clone_step(dec, a, rows, cfg, norm)
    assert m["n_skip"] == 0.0
    assert all(torch.allclose(p.grad, w, atol=1e-6, rtol=1e-5) for p, w in zip(a.params(), want))


def test_lora_prompt_pass_on_sdpa_matches_eager():
    """The grad-carrying prompt pass runs on SDPA (memory); scores and gradients must match all-eager."""
    from shared import backprop
    dec = as_decoder(tiny_model("eager"), attn="eager")
    rows = [r for r in make_rows() if r.ids]
    cfg = BackpropCfg()

    def run(sdpa):
        torch.manual_seed(0)
        a = lora_adapter(dec, True)
        old = backprop._sdpa_for
        if not sdpa:
            backprop._sdpa_for = lambda model, on: contextlib.nullcontext()
        try:
            lps = logprobs(dec, a, rows, ids_cand(rows), cfg)
            sum(lp.sum() for lp in lps).backward()
        finally:
            backprop._sdpa_for = old
        assert dec.model.config._attn_implementation == "eager"
        return [lp.detach() for lp in lps], [p.grad.clone() for p in a.params()]

    (l1, g1), (l2, g2) = run(True), run(False)
    assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(l1, l2))
    assert all(torch.allclose(x, y, atol=1e-5, rtol=1e-4) for x, y in zip(g1, g2))
