import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from adapters import AdapterCfg, build
from adapters.kv_prefix import KVPrefix, check_rows, load_units, random_rows, stack, stack_seed
from adapters.lora import LoRA, stack as stack_lora
from conftest import as_decoder, tiny_model
from shared.adapter import Adapter, AdapterKind

L, HKV, HD, K, D = 2, 2, 16, 4, 64
CPU = torch.device("cpu")


def text_cart(seed: int) -> dict:
    """What text_kv.py writes: own rows from random_rows(init_seed=0), K/V from the text."""
    g = torch.Generator().manual_seed(100 + seed)
    return {"soft": random_rows(K, D, 0.02, 1.0, 0, CPU), "kfix": torch.randn(L, K, HKV, HD, generator=g),
            "vfix": torch.randn(L, K, HKV, HD, generator=g)}


def kv_module() -> KVPrefix:
    g = torch.Generator().manual_seed(5)
    kf0, vf0 = (torch.randn(L, K, HKV, HD, generator=g) for _ in range(2))
    return KVPrefix(random_rows(K, D, 0.02, 1.0, 0, CPU).to(torch.bfloat16), L, HKV, HD, load_units("all", L, HKV, K),
                    False, True, kf0, vf0, True, False, True)


def test_stack_rows_are_distinct_and_off_the_init_stream():
    own = random_rows(K, D, 0.02, 1.0, 0, CPU)
    st = stack([text_cart(0), text_cart(1)], 0)
    assert st["soft"].shape == (2 * K, D) and st["sizes"] == [K, K]
    check_rows(torch.cat([own, st["soft"]]), "own + stack")
    old = torch.randn(2 * K, D, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(F.cosine_similarity(old[:K], own, -1), torch.ones(K)), "the old draw replayed the init stream"
    assert F.cosine_similarity(st["soft"][:K], own, -1).abs().max() < 0.9
    assert stack_seed(0) >= 2 ** 32 and stack_seed(0) != stack_seed(1)
    assert torch.equal(stack([text_cart(0), text_cart(1)], 0)["soft"], st["soft"]), "stacks are deterministic"
    assert not torch.equal(stack([text_cart(0), text_cart(1)], 1)["soft"], st["soft"])
    kv_module().set_partners(st["soft"], st["kfix"], st["vfix"], st["sizes"], 1.0)


def test_rows_equal_at_bf16_are_refused():
    m = kv_module()
    st = stack([text_cart(0), text_cart(1)], 0)
    soft = st["soft"].clone()
    soft[5] = m.base[1].float() * (1 + 1e-5)          # differs in fp32, equal once rounded to bf16
    assert not torch.equal(soft[5], m.base[1].float())
    with pytest.raises(ValueError, match="rows 1 and 9"):
        m.set_partners(soft, st["kfix"], st["vfix"], st["sizes"], 1.0)
    assert not m.partnered
    with pytest.raises(ValueError, match="rows 0 and 2"):
        check_rows(torch.cat([soft[:2], soft[:1]]), "stack")


def lora(dec, rank=1, alpha=32.0, layers=(), seed=0) -> Adapter:
    m = LoRA(dec.model, rank, alpha, ("q_proj", "v_proj"), layers, seed, False)
    with torch.no_grad():
        for b in m.B:
            b.normal_(0, 0.02)
    return Adapter(AdapterKind.lora, m)


def part(a: Adapter) -> dict:
    return {"adapter": a.state_dict()}


def test_lora_stack_refuses_mismatched_scales(tmp_path):
    dec = as_decoder(tiny_model())
    with pytest.raises(ValueError, match="scales"):
        stack_lora([part(lora(dec)), part(lora(dec, alpha=64.0, seed=1))])
    st = stack_lora([part(lora(dec, 1, 32.0)), part(lora(dec, 2, 64.0, seed=1))])
    assert st["scale"] == 32.0 and st["sizes"] == [1, 2]
    with pytest.raises(ValueError, match="scale"):
        lora(dec, 3, 32.0).module.load_stack(st)
    lora(dec, 3, 96.0).module.load_stack(st)
    with pytest.raises(ValueError, match="scale"):
        lora(dec, 1, 64.0).module.set_partners(st["lora_A"], st["lora_B"], st["sizes"], 1.0, st["scale"], st["names"])
    torch.save(stack_lora([part(lora(dec, 1, 64.0, seed=s)) for s in (1, 2)]), tmp_path / "p.pt")
    cfg = AdapterCfg(adapter=AdapterKind.lora, lora_targets=("q_proj", "v_proj"), lora_partners=str(tmp_path / "p.pt"))
    with pytest.raises(ValueError, match="scale"):
        build(cfg, dec)
    assert build(dataclasses.replace(cfg, lora_alpha=64.0), dec).module.partnered


def test_lora_stack_refuses_other_modules():
    dec = as_decoder(tiny_model())
    a0, a1 = lora(dec, layers=(0,)), lora(dec, layers=(1,), seed=1)
    assert len(a0.module.A) == len(a1.module.A)
    with pytest.raises(ValueError, match="modules"):
        stack_lora([part(a0), part(a1)])
    st = stack_lora([part(a0), part(lora(dec, layers=(0,), seed=2))])
    assert st["names"] == a0.module.names
    with pytest.raises(ValueError, match="modules"):
        a1.module.set_partners(st["lora_A"], st["lora_B"], st["sizes"], 1.0, st["scale"], st["names"])
    with pytest.raises(ValueError, match="modules"):
        lora(dec, 2, 64.0, layers=(1,)).module.load_stack(st)
    lora(dec, 2, 64.0, layers=(0,)).module.load_stack(st)
    with pytest.raises(ValueError, match="modules"):
        a1.load(a0.state_dict())


def test_load_stack_round_trip_and_old_files(tmp_path):
    dec = as_decoder(tiny_model())
    ps = [lora(dec, seed=s) for s in (0, 1)]
    torch.save(part(ps[0]), tmp_path / "ck.pt")
    ck = torch.load(tmp_path / "ck.pt", map_location="cpu", weights_only=True)
    torch.save(stack_lora([ck, part(ps[1])]), tmp_path / "s.pt")
    st = torch.load(tmp_path / "s.pt", map_location="cpu", weights_only=True)
    assert st["scale"] == 32.0 and st["names"] == ps[0].module.names
    m = lora(dec, 2, 64.0).module
    m.load_stack(st)
    assert all(torch.equal(a, torch.cat([p.module.A[i] for p in ps], -2)) for i, a in enumerate(m.A))
    assert all(torch.equal(b, torch.cat([p.module.B[i] for p in ps], -1)) for i, b in enumerate(m.B))
    old = {k: st[k] for k in ("lora_A", "lora_B", "sizes")}
    lora(dec, 2, 32.0).module.load_stack(old)
    lora(dec).module.set_partners(old["lora_A"], old["lora_B"], old["sizes"], 1.0)
    sd = ps[0].state_dict()
    sd["state"] = {k: v for k, v in sd["state"].items() if k != "_extra_state"}
    a = lora(dec, seed=3)
    a.load(sd)
    assert all(torch.equal(x, y) for x, y in zip(a.module.B, ps[0].module.B))
    assert stack_lora([{"adapter": sd}, part(ps[1])])["scale"] == 32.0
    bare = stack_lora([{"adapter": sd}, {"adapter": sd}])
    assert bare["scale"] is None and bare["names"] is None
    lora(dec, 2, 32.0).module.load_stack(bare)


def test_kv_graph_leaves_engine_positions_alone(tmp_path):
    from shared import kv_graph
    from shared.attach import kv_payload
    from test_kv_ext import context, graph_model, kv_module as ext_module
    dec = as_decoder(tiny_model())
    me = ext_module(dec, replace=True, pos0=True, graph=True)
    torch.save(kv_payload(me), tmp_path / "kv.pt")
    x, _ = context(dec, me)
    w = kv_graph.KVGraphWorker()
    w.model_runner = SimpleNamespace(model=graph_model()(dec.model))
    w.set_kv_delta(str(tmp_path / "kv.pt"))
    seq = torch.arange(x.shape[0])
    with torch.no_grad():
        first = w.model_runner.model(None, seq, None, x)
        again = w.model_runner.model(None, seq, None, x)
    assert torch.equal(seq, torch.arange(x.shape[0])), "the engine's positions were rewritten"
    assert torch.equal(first, again)
