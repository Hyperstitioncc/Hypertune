from __future__ import annotations

import contextlib
import dataclasses
import itertools
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ContextManager, Sequence

import torch
import torch.nn.functional as F

from adapters.kv_prefix import KVRead
from shared.adapter import Adapter, AdapterKind
from shared.decoder import Decoder
from shared.kv_graph import attn_forward, match_rows, row_positions, tp_heads
from shared.types import Backend

KV_GRAPH_WORKER = "shared.kv_graph_worker.KVGraphWorker"
KV_INJECT_WORKER = "shared.attach.KVInjectWorker"


@dataclass(frozen=True)
class Attached:
    k: int
    rows: Callable[[Sequence[int]], torch.Tensor]
    row_tail: bool
    pos0: bool
    prompt_grad: bool
    armed: Callable[[int], ContextManager]
    attn_mass: Callable[[int], ContextManager[list]]
    engine_kw: dict
    own_compile_cache: bool
    push: Callable[[Sequence[Any], str], list[dict]]
    request: Callable[[Sequence[int]], dict]
    net: Decoder
    attn_write: Callable[[int], ContextManager[list]] | None = None


def row_start(row_tail: bool, P: int) -> int:
    return P - 1 if row_tail else P


def splice(row_tail: bool, pe: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    i = row_start(row_tail, pe.shape[0])
    return torch.cat([pe[:i], soft, pe[i:]], 0)


def positions(e: Attached, P: int, k: int, n: int, device, start: int = 0) -> torch.Tensor:
    pos = torch.arange(start, start + n, device=device)
    if e.pos0:
        i = row_start(e.row_tail, P)
        pos[i:i + k] = 0
        pos[i + k:] -= k
    return pos


def after_rows(e: Attached, P: int, k: int) -> int:
    return P if e.pos0 else P + k


def _null(P: int) -> ContextManager:
    return contextlib.nullcontext()


def _no_mass(why: str) -> Callable[[int], ContextManager[list]]:
    def mass(P: int) -> ContextManager[list]:
        raise ValueError(f"no attention-mass probe: {why}")
    return mass


def _no_push(llms: Sequence[Any], work: str) -> list[dict]:
    return [{} for _ in llms]


def _token_request(ids: Sequence[int]) -> dict:
    return {"prompt_token_ids": list(ids)}


def _empty_rows(dec: Decoder) -> Callable[[Sequence[int]], torch.Tensor]:
    z = torch.zeros(0, dec.d, device=dec.dev)
    return lambda ids: z


def _rpc_all(llms: Sequence[Any], method: str, path: str) -> list[dict]:
    for llm in llms:
        llm.collective_rpc(method, args=(os.path.abspath(path),))
        llm.reset_prefix_cache()
    return [{} for _ in llms]


# ---------------------------------------------------------------- attention swap, per model instance

_WRAPPED = "adapter_"


def _wrapped_attention(module, *a, **kw):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    base = module.config._attn_implementation.removeprefix(_WRAPPED)
    fn = ALL_ATTENTION_FUNCTIONS.get_interface(base, sys.modules[type(module).__module__].eager_attention_forward)
    for w in module.attn_wraps:
        fn = partial(w, fn)
    return fn(module, *a, **kw)


@contextlib.contextmanager
def _attn_wrapped(model, wrap):
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface
    from transformers.modeling_utils import AttentionInterface
    cfg = model.config
    base = cfg._attn_implementation.removeprefix(_WRAPPED)
    name = _WRAPPED + base
    AttentionInterface.register(name, _wrapped_attention)
    AttentionMaskInterface.register(name, ALL_MASK_ATTENTION_FUNCTIONS[base])
    ats = [layer.self_attn for layer in model.model.layers]
    for at in ats:
        at.__dict__.setdefault("attn_wraps", []).append(wrap)
    cfg._attn_implementation = name
    try:
        yield
    finally:
        for at in ats:
            at.attn_wraps.remove(wrap)
        if not ats[0].attn_wraps:
            cfg._attn_implementation = base


def _attn_mass(module, query: torch.Tensor, key: torch.Tensor, c0: int, c1: int, r0: int) -> torch.Tensor | None:
    off = key.shape[2] - query.shape[2]
    q = query[0, :, max(r0 - off, 0):].float()
    Tr, T = q.shape[1], key.shape[2]
    if Tr == 0:
        return None
    k = key[0].float().repeat_interleave(query.shape[1] // key.shape[1], 0)
    lg = torch.einsum("htd,hsd->hts", q, k) * module.scaling
    qpos = torch.arange(Tr, device=lg.device) + max(r0, off)
    lg = lg.masked_fill(~(torch.arange(T, device=lg.device)[None, :] <= qpos[:, None])[None], float("-inf"))
    return torch.exp(torch.logsumexp(lg[:, :, c0:c1], -1) - torch.logsumexp(lg, -1)).mean()


def _attn_write(module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, c0: int, c1: int,
                r0: int) -> torch.Tensor | None:
    off = key.shape[2] - query.shape[2]
    q = query[0, :, max(r0 - off, 0):].float()
    Tr, T = q.shape[1], key.shape[2]
    if Tr == 0:
        return None
    g = query.shape[1] // key.shape[1]
    k, v = key[0].float().repeat_interleave(g, 0), value[0].float().repeat_interleave(g, 0)
    lg = torch.einsum("htd,hsd->hts", q, k) * module.scaling
    qpos = torch.arange(Tr, device=lg.device) + max(r0, off)
    att = torch.softmax(lg.masked_fill(~(torch.arange(T, device=lg.device)[None, :] <= qpos[:, None])[None], float("-inf")), -1)
    rows = torch.einsum("hts,hsd->htd", att[:, :, c0:c1], v[:, c0:c1])
    whole = torch.einsum("hts,hsd->htd", att, v)
    return rows.pow(2).sum(-1).mean() / (whole.pow(2).sum(-1).mean().detach() + 1e-8)


def _row_mass(net: Decoder, k: int, row_tail: bool, write: bool = False) -> Callable[[int], ContextManager[list]]:
    if not k:
        return _no_mass("the adapter puts no rows in context")

    @contextlib.contextmanager
    def mass(P: int):
        acc: list[torch.Tensor] = []
        c0 = row_start(row_tail, P)

        def wrap(inner, module, query, key, value, attention_mask, *a, **kw):
            out = inner(module, query, key, value, attention_mask, *a, **kw)
            m = (_attn_write(module, query, key, value, c0, c0 + k, c0 + k) if write
                 else _attn_mass(module, query, key, c0, c0 + k, P + k))
            if m is not None:
                acc.append(m.to(net.dev))
            return out
        with _attn_wrapped(net.model, wrap):
            yield acc
    return mass


def _hf(net: Decoder, rows, k, armed=_null, row_tail=False, pos0=False, prompt_grad=False,
        attn_mass=None) -> Attached:
    return Attached(k, rows, row_tail, pos0, prompt_grad, armed,
                 attn_mass or _row_mass(net, k, row_tail), {}, False, _no_push, _token_request, net,
                 _row_mass(net, k, row_tail, write=True))


def _embeds_request(dec: Decoder, rows, row_tail: bool) -> Callable[[Sequence[int]], dict]:
    table = dec.emb.weight.detach().to(torch.bfloat16)

    def request(ids: Sequence[int]) -> dict:
        pe = table[torch.tensor(list(ids), device=table.device)].float().cpu()
        with torch.no_grad():
            soft = rows(ids).detach().float().cpu()
        return {"prompt_embeds": splice(row_tail, pe, soft).to(torch.bfloat16)}
    return request


def _vllm(dec: Decoder, request, engine_kw: dict, push, rows, k: int, row_tail: bool = False,
          own_compile_cache: bool = False) -> Attached:
    return Attached(k, rows, row_tail, False, False, _null, _no_mass("vLLM entries are not probed"),
                 dict(engine_kw), own_compile_cache, push, request, dec)


# ---------------------------------------------------------------- identity / full

def _identity(adapter: Adapter, backend: Backend, dec: Decoder) -> Attached:
    rows = _empty_rows(dec)
    if backend is Backend.vllm:
        return _vllm(dec, _embeds_request(dec, rows, False), {}, _no_push, rows, 0)
    return _hf(dec, rows, 0)


def _full(adapter: Adapter, backend: Backend, dec: Decoder) -> Attached:
    model = adapter.module.model
    if backend is Backend.vllm:
        def push(llms: Sequence[Any], work: str) -> list[dict]:
            path = os.path.join(work, "weights")
            shutil.rmtree(path, ignore_errors=True)
            model.save_pretrained(path)
            return _rpc_all(llms, "load_full_weights", path)
        return _vllm(dec, _token_request, {"worker_extension_cls": "shared.attach.FullReloadWorker"}, push,
                     _empty_rows(dec), 0)
    net = dataclasses.replace(dec, model=model, emb=model.get_input_embeddings(), head=model.get_output_embeddings())
    return _hf(net, _empty_rows(dec), 0, prompt_grad=True)


# ---------------------------------------------------------------- lora

def _write_share(delta: torch.Tensor, whole: torch.Tensor) -> torch.Tensor:
    return (delta.float().pow(2).sum(-1) / whole.float().pow(2).sum(-1).clamp_min(1e-12)).mean()


def _lora_hook(mod, i):
    def hook(m, inp, out):
        x = inp[0]
        A, B = mod.eff(i)
        d = (x @ A.t().to(x.device, x.dtype)) @ (B.t() * mod.scale).to(x.device, x.dtype)
        if mod.write_acc is not None:
            mod.write_acc.append(_write_share(d, out + d))
        return out + d
    return hook


def _lora_write(mod) -> Callable[[int], ContextManager[list]]:
    @contextlib.contextmanager
    def write(P: int):
        acc: list[torch.Tensor] = []
        was, mod.write_acc = mod.write_acc, acc
        try:
            yield acc
        finally:
            mod.write_acc = was
    return write


def _lora_experts_forward(mod, xm, i0):
    from transformers.integrations.moe import _grouped_mm
    ig, iu, idn = i0, i0 + 1, i0 + 2

    def fwd(hidden_states, top_k_index, top_k_weights):
        dev = hidden_states.device
        (Ag, Bg), (Au, Bu), (Ad, Bd) = ((a.to(dev), b.to(dev)) for a, b in (mod.eff(j) for j in (ig, iu, idn)))
        r = Ag.shape[1]
        P = 8 * max(1, -(-2 * r // 8))
        dt = hidden_states.dtype
        s = mod.scale
        num_tokens, k = top_k_index.shape
        sample_w = top_k_weights.reshape(-1)
        eid, perm = torch.sort(top_k_index.reshape(-1))
        xg = hidden_states[perm // k]
        hin = eid.float() if xg.device.type in ("cpu", "mps") else eid.int()
        offs = torch.cumsum(torch.histc(hin, bins=xm.num_experts, min=0, max=xm.num_experts - 1), 0,
                            dtype=torch.int32)
        Agu = F.pad(torch.cat([Ag, Au], 1).to(dt), (0, 0, 0, P - 2 * r))
        Bgu = torch.cat([F.pad(Bg.to(dt), (0, P - r)), F.pad(Bu.to(dt), (r, P - 2 * r))], 1)
        h0 = _grouped_mm(xg, xm.gate_up_proj.transpose(-2, -1), offs=offs)
        u = _grouped_mm(xg, Agu.transpose(-2, -1), offs=offs)
        h = h0 + s * _grouped_mm(u, Bgu.transpose(-2, -1), offs=offs)
        a = xm._apply_gate(h)
        Adp = F.pad(Ad.to(dt), (0, 0, 0, P - r))
        Bdp = F.pad(Bd.to(dt), (0, P - r))
        o = _grouped_mm(a, xm.down_proj.transpose(-2, -1), offs=offs)
        o = o + s * _grouped_mm(_grouped_mm(a, Adp.transpose(-2, -1), offs=offs), Bdp.transpose(-2, -1), offs=offs)
        if mod.write_acc is not None:
            o0 = _grouped_mm(xm._apply_gate(h0), xm.down_proj.transpose(-2, -1), offs=offs)
            mod.write_acc.append(_write_share(o - o0, o))
        o = o * sample_w[perm].unsqueeze(-1)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=perm.device)
        return o[inv].view(num_tokens, k, -1).sum(1).to(dt)
    return fwd


def _lora_armed(mod, model) -> Callable[[int], ContextManager]:
    named = dict(model.named_modules())

    @contextlib.contextmanager
    def armed(P: int):
        handles, patched = [], []
        for i in range(mod.n_linear):
            handles.append(named[mod.names[i]].register_forward_hook(_lora_hook(mod, i)))
        for i0 in range(mod.n_linear, len(mod.names), 3):
            xm = named[mod.names[i0].rsplit(".", 1)[0]]
            xm.forward = _lora_experts_forward(mod, xm, i0)
            patched.append(xm)
        try:
            yield
        finally:
            for h in handles:
                h.remove()
            for xm in patched:
                del xm.forward
    return armed


def save_peft(mod, path: str, model_name: str) -> str:
    from safetensors.torch import save_file
    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    tensors = {}
    ab = [mod.eff(i) for i in range(len(mod.names))]
    r = ab[0][0].shape[-2]
    for n, (a, b) in zip(mod.names, ab):
        a, b = a.detach().cpu(), b.detach().cpu()
        if a.dim() == 3:
            base, proj = n.rsplit(".", 1)
            a16, b16 = a.to(torch.bfloat16), b.to(torch.bfloat16)
            for e in range(a.shape[0]):
                tensors[f"base_model.model.{base}.{e}.{proj}.lora_A.weight"] = a16[e].contiguous()
                tensors[f"base_model.model.{base}.{e}.{proj}.lora_B.weight"] = b16[e].contiguous()
        else:
            tensors[f"base_model.model.{n}.lora_A.weight"] = a.float().contiguous()
            tensors[f"base_model.model.{n}.lora_B.weight"] = b.float().contiguous()
    save_file(tensors, f"{tmp}/adapter_model.safetensors")
    with open(f"{tmp}/adapter_config.json", "w") as f:
        json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": model_name,
                   "r": r, "lora_alpha": mod.scale * r, "target_modules": list(mod.targets), "lora_dropout": 0.0,
                   "bias": "none", "fan_in_fan_out": False, "use_rslora": False, "modules_to_save": None}, f)
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)
    return path


def _lora(adapter: Adapter, backend: Backend, dec: Decoder) -> Attached:
    mod = adapter.module
    rows = _empty_rows(dec)
    if backend is Backend.hf_eager:
        return dataclasses.replace(_hf(dec, rows, 0, armed=_lora_armed(mod, dec.model), prompt_grad=True),
                                   attn_write=_lora_write(mod))
    if mod.merge:
        def merge(llms: Sequence[Any], work: str) -> list[dict]:
            path = save_peft(mod, os.path.join(work, "adapter"), dec.cfg.model)
            return _rpc_all(llms, "merge_lora_adapter", path)
        return _vllm(dec, _token_request, {"worker_extension_cls": "shared.attach.LoRAMergeWorker"}, merge, rows, 0)
    ids = itertools.count(1)

    def push(llms: Sequence[Any], work: str) -> list[dict]:
        from vllm.lora.request import LoRARequest
        path = save_peft(mod, os.path.join(work, "adapter"), dec.cfg.model)
        i = next(ids)
        return [{"lora_request": LoRARequest(f"a{i}", i, os.path.abspath(path))} for _ in llms]
    need = mod.rank + sum(b - a for a, b in mod.spans)
    kw = dict(enable_lora=True, max_loras=1, max_lora_rank=next(r for r in (8, 16, 32, 64, 128, 256, 320, 512) if r >= need),
              max_cpu_loras=2,
              lora_target_modules=list(mod.targets))
    return _vllm(dec, _token_request, kw, push, rows, 0)


# ---------------------------------------------------------------- kv_prefix: softmax read

def _kv_armed(me, dec: Decoder) -> Callable[[int], ContextManager]:
    inner = dec.model.model
    state: dict = {"oh": None}

    def pre(mod, args, kwargs):
        e = kwargs.get("inputs_embeds")
        oh = None if e is None else match_rows(e, me.table().to(e.device))
        state["oh"] = oh.to(torch.float32) if oh is not None and bool(oh.any()) else None

    def k_hook(li, mod, args, out):
        oh = state["oh"]
        if oh is None:
            return None
        dk, _ = me.eff()
        oh_ = oh.to(out.device)
        if me.replace:
            kf = me.full(li)[0].to(out.device)
            sel = oh_.sum(-1)[..., None, None]
            return (out.float() * (1 - sel) + torch.einsum("btk,khd->bthd", oh_, kf)).to(out.dtype)
        return out + torch.einsum("btk,khd->bthd", oh_, dk[li].to(out.device)).to(out.dtype)

    def v_hook(li, mod, args, out):
        oh = state["oh"]
        if oh is None:
            return None
        _, dv = me.eff()
        B, T, _ = out.shape
        vv = out.view(B, T, me.Hkv, me.hd)
        oh_ = oh.to(out.device)
        if me.replace:
            vf = me.full(li)[1].to(out.device)
            sel = oh_.sum(-1)[..., None, None]
            return (vv.float() * (1 - sel) + torch.einsum("btk,khd->bthd", oh_, vf)).to(out.dtype).reshape(B, T, -1)
        vv = vv + torch.einsum("btk,khd->bthd", oh_, dv[li].to(out.device)).to(vv.dtype)
        return vv.reshape(B, T, -1)

    def remove_rest(inner_fn, module, query, key, value, attention_mask, *a, **kw):
        li = module.layer_idx
        oh = state["oh"]
        if oh is None or not bool(me.rm[li].any()):
            return inner_fn(module, query, key, value, attention_mask, *a, **kw)
        oh = oh.to(query.device)
        out1, w1 = inner_fn(module, query, key, value, attention_mask, *a, **kw)
        Tq, Tk = query.shape[2], key.shape[2]
        ohk = torch.cat([oh.new_zeros(oh.shape[0], Tk - Tq, oh.shape[2]), oh], 1)
        zf = 1 - torch.einsum("btk,kh->bht", ohk, me.rm[li].to(oh.device))
        out2, _ = inner_fn(module, query, key * zf[..., None].to(key.dtype), value * zf[..., None].to(value.dtype),
                           attention_mask, *a, **kw)
        soft_q = oh.sum(-1) > 0
        return torch.where(soft_q[:, :, None, None], out1, out2), w1

    @contextlib.contextmanager
    def armed(P: int):
        with contextlib.ExitStack() as stack:
            hs = [inner.register_forward_pre_hook(pre, with_kwargs=True)]
            for li, layer in enumerate(inner.layers):
                at = layer.self_attn
                hs.append(at.k_norm.register_forward_hook(partial(k_hook, li)))
                hs.append(at.v_proj.register_forward_hook(partial(v_hook, li)))
            for h in hs:
                stack.callback(h.remove)
            stack.callback(state.update, oh=None)
            if bool(me.rm.any()):
                stack.enter_context(_attn_wrapped(dec.model, remove_rest))
            yield
    return armed


def kv_payload(me) -> dict:
    dk, dv = me.eff()
    rm = me.rm
    if me.replace:
        dk, dv = (torch.stack(x) for x in zip(*(me.full(li) for li in range(me.L))))
        rm = torch.cat([rm, rm.new_zeros(me.L, dk.shape[1] - me.K, me.Hkv)], 1)
    return {"base": me.table().to(torch.bfloat16).cpu(), "dk": dk.detach().to(torch.bfloat16).cpu(),
            "dv": dv.detach().to(torch.bfloat16).cpu(), "rm": rm.cpu(), "replace": me.replace, "pos0": me.pos0}


def _payload_push(method: str, payload: Callable[[], dict]) -> Callable[[Sequence[Any], str], list[dict]]:
    def push(llms: Sequence[Any], work: str) -> list[dict]:
        path = os.path.join(work, "kv.pt")
        torch.save(payload(), path + ".tmp")
        os.replace(path + ".tmp", path)
        return _rpc_all(llms, method, path)
    return push


def _kv_softmax(adapter: Adapter, backend: Backend, dec: Decoder) -> Attached:
    me = adapter.module
    rows = lambda ids: me.rows(ids)
    if backend is Backend.hf_eager:
        if bool(me.rm.any()) and "flash" in dec.cfg.attn:
            raise ValueError("kv_remove_rest needs a 4D mask path (attn sdpa|eager)")
        return _hf(dec, rows, me.K, armed=_kv_armed(me, dec), row_tail=me.row_tail, pos0=me.pos0)
    if me.graph and bool(me.rm.any()):
        raise ValueError("kv_graph: removed units (kv_remove_rest) are eager-only")
    kw = ({"worker_extension_cls": KV_GRAPH_WORKER} if me.graph else
          {"worker_extension_cls": KV_INJECT_WORKER, "enforce_eager": True})
    if me.pos0:
        kw.update(enable_prefix_caching=False, enable_chunked_prefill=False)
    return _vllm(dec, _embeds_request(dec, rows, me.row_tail), kw,
                 _payload_push("set_kv_delta", lambda: kv_payload(me)), rows, me.K, me.row_tail,
                 own_compile_cache=me.graph)


# ---------------------------------------------------------------- kv_prefix: dsm read (the cartridge)

def dsm_gate(q: torch.Tensor, kg: torch.Tensor, gb: float) -> torch.Tensor:
    return torch.sigmoid((q.float() * kg.float()).sum(-1) + gb)


def dsm_mix(x: torch.Tensor, q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, kg: torch.Tensor, gb: float,
            scaling: float) -> torch.Tensor:
    H, hd = q.shape[-2], q.shape[-1]
    Hkv = kc.shape[1]
    qg = q.float().reshape(*q.shape[:-2], Hkv, H // Hkv, hd)
    s = torch.einsum("...ngd,knd->...ngk", qg, kc.float()) * scaling
    o = torch.einsum("...ngk,knd->...ngd", torch.softmax(s, -1), vc.float()).reshape(*q.shape[:-2], H, hd)
    g = dsm_gate(q, kg, gb)[..., None]
    return (1 - g) * x.float() + g * o


def _dsm_armed(me, dec: Decoder) -> Callable[[int], ContextManager]:
    qs: dict = {}

    def q_hook(li, mod, args, out):
        qs[li] = out
        return None

    def o_pre(li, mod, args):
        q = qs.pop(li)
        x = args[0]
        dv = x.device
        mixed = dsm_mix(x.view(*q.shape), q, me.kc[li].to(dv), me.vc[li].to(dv), me.kg[li].to(dv), me.gb, me.scaling)
        return (mixed.reshape(x.shape).to(x.dtype),) + tuple(args[1:])

    @contextlib.contextmanager
    def armed(P: int):
        hs = []
        for li, layer in enumerate(dec.model.model.layers):
            hs.append(layer.self_attn.q_norm.register_forward_hook(partial(q_hook, li)))
            hs.append(layer.self_attn.o_proj.register_forward_pre_hook(partial(o_pre, li)))
        try:
            yield
        finally:
            for h in hs:
                h.remove()
            qs.clear()
    return armed


def _gate_mass(me, dec: Decoder) -> Callable[[int], ContextManager[list]]:
    @contextlib.contextmanager
    def mass(P: int):
        acc: list[torch.Tensor] = []
        seen: dict[int, int] = {}

        def q_hook(li, mod, args, out):
            t0 = seen.get(li, 0)
            seen[li] = t0 + out.shape[1]
            q = out[0, max(P - t0, 0):]
            if q.shape[0]:
                acc.append(dsm_gate(q, me.kg[li].to(q.device), me.gb).mean().to(dec.dev))
        hs = [layer.self_attn.q_norm.register_forward_hook(partial(q_hook, li))
              for li, layer in enumerate(dec.model.model.layers)]
        try:
            yield acc
        finally:
            for h in hs:
                h.remove()
    return mass


def cart_payload(me) -> dict:
    return {"scaling": me.scaling, "kc": me.kc.detach().float().cpu(), "vc": me.vc.detach().float().cpu(),
            "kg": me.kg.detach().float().cpu(), "gb": me.gb}


def _kv_dsm(adapter: Adapter, backend: Backend, dec: Decoder) -> Attached:
    me = adapter.module
    rows = _empty_rows(dec)
    if backend is Backend.hf_eager:
        return _hf(dec, rows, 0, armed=_dsm_armed(me, dec), prompt_grad=True, attn_mass=_gate_mass(me, dec))
    return _vllm(dec, _embeds_request(dec, rows, False), {"worker_extension_cls": KV_INJECT_WORKER, "enforce_eager": True},
                 _payload_push("set_kv_cart", lambda: cart_payload(me)), rows, 0)


_KV_READ = {KVRead.softmax: _kv_softmax, KVRead.dsm: _kv_dsm}


def _kv_prefix(adapter: Adapter, backend: Backend, dec: Decoder) -> Attached:
    return _KV_READ[adapter.module.read](adapter, backend, dec)


_ATTACH = {
    AdapterKind.identity: _identity,
    AdapterKind.lora: _lora,
    AdapterKind.kv_prefix: _kv_prefix,
    AdapterKind.full: _full,
}


def attach(adapter: Adapter, backend: Backend, decoder: Decoder) -> Attached:
    return _ATTACH[adapter.kind](adapter, backend, decoder)


# ================================================================ vLLM worker extensions

class FullReloadWorker:
    def load_full_weights(self, path):
        from safetensors.torch import load_file
        files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))

        def gen():
            for f in files:
                yield from load_file(os.path.join(path, f)).items()
        self.model_runner.reload_weights(weights_iterator=gen(), is_checkpoint_format=True)


class KVInjectWorker:
    def _kvi_install(self, zero_removed: bool):
        inner = self.model_runner.model.model
        S = self._kvi = {"oh": None, "pz": None}
        orig_fwd = inner.forward

        def fwd(input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kw):
            S["oh"] = S["pz"] = None
            if inputs_embeds is not None:
                hit = match_rows(inputs_embeds, S["base"])
                if bool(hit.any()):
                    S["oh"] = hit.to(inputs_embeds.dtype)
                    if S["pos0"]:
                        S["pz"] = row_positions(positions, hit.any(-1))
            return orig_fwd(input_ids, positions, intermediate_tensors, inputs_embeds, **kw)
        inner.forward = fwd

        def pos(p):
            return p if S["pz"] is None else S["pz"]

        for li, layer in enumerate(inner.layers):
            at = layer.self_attn

            def kv(kh, vh, li=li):
                oh = S["oh"]
                if oh is None:
                    return kh, vh
                dk = torch.einsum("tk,khd->thd", oh, S["dk"][li])
                dv = torch.einsum("tk,khd->thd", oh, S["dv"][li])
                if S["replace"]:
                    keep = 1 - oh.sum(-1)[:, None, None]
                    return kh * keep + dk, vh * keep + dv
                return kh + dk, vh + dv

            def post(o, q, at=at, li=li):
                if S["oh"] is not None and S["rmany"][li]:
                    _zero_cached(at.attn, S["oh"], S["rm"][li])
                return o
            at.forward = partial(attn_forward, at, kv=kv, pos=pos, post=post if zero_removed else None)

    def set_kv_delta(self, path):
        st = torch.load(path, map_location="cpu")
        at0 = self.model_runner.model.model.layers[0].self_attn
        dev = next(self.model_runner.model.parameters()).device
        h_loc = at0.kv_size // at0.head_dim
        dk, dv, rm = (tp_heads(st[x], 2, h_loc).contiguous().to(dev, torch.bfloat16) for x in ("dk", "dv", "rm"))
        rmany = [bool(x) for x in (rm.flatten(1).sum(1) > 0).tolist()]
        if not hasattr(self, "_kvi"):
            self._kvi_install(any(rmany))
        self._kvi.update(base=st["base"].to(dev, torch.bfloat16), dk=dk, dv=dv, rm=rm, rmany=rmany,
                         replace=bool(st["replace"]), pos0=bool(st["pos0"]))

    def _kvc_install(self):
        S = self._kvc = {}
        for li, layer in enumerate(self.model_runner.model.model.layers):
            at = layer.self_attn

            def post(o, q, li=li):
                return dsm_mix(o.view(*q.shape), q, S["kc"][li], S["vc"][li], S["kg"][li], S["gb"],
                               S["scaling"]).reshape(o.shape).to(o.dtype)
            at.forward = partial(attn_forward, at, post=post)

    def set_kv_cart(self, path):
        st = torch.load(path, map_location="cpu")
        if not hasattr(self, "_kvc"):
            self._kvc_install()
        at0 = self.model_runner.model.model.layers[0].self_attn
        dev = next(self.model_runner.model.parameters()).device
        H_l, Hkv_l = at0.q_size // at0.head_dim, at0.kv_size // at0.head_dim
        self._kvc.update(kc=tp_heads(st["kc"], 2, Hkv_l).to(dev, torch.float32).contiguous(),
                         vc=tp_heads(st["vc"], 2, Hkv_l).to(dev, torch.float32).contiguous(),
                         kg=tp_heads(st["kg"], 1, H_l).to(dev, torch.float32).contiguous(),
                         gb=float(st["gb"]), scaling=float(st["scaling"]))


def _zero_cached(attn, oh, rm_l):
    from vllm.forward_context import get_forward_context
    ctx = get_forward_context()
    md = ctx.attn_metadata
    if isinstance(md, list):
        md = md[0]
    if isinstance(md, dict):
        md = md[attn.layer_name]
    slots = md.slot_mapping
    kc = attn.kv_cache[ctx.virtual_engine] if isinstance(attn.kv_cache, (list, tuple)) else attn.kv_cache
    t_idx, rows = oh.nonzero(as_tuple=True)
    if len(t_idx) == 0:
        return
    rmh = rm_l[rows] > 0
    s = slots[t_idx].long()
    if kc.shape[0] == 2:
        bs = kc.shape[2]
        b, o = s // bs, s % bs
        for kv in (0, 1):
            sub = kc[kv, b, o]
            sub[rmh] = 0
            kc[kv, b, o] = sub
    elif kc.shape[1] == 2:
        bs = kc.shape[2]
        b, o = s // bs, s % bs
        for kv in (0, 1):
            sub = kc[b, kv, o]
            sub[rmh] = 0
            kc[b, kv, o] = sub
    elif kc.dim() == 4 and kc.shape[1] == rmh.shape[1]:
        bs = kc.shape[2]
        b, o = s // bs, s % bs
        sub = kc[b, :, o]
        sub[rmh] = 0
        kc[b, :, o] = sub
    else:
        raise NotImplementedError(f"kv cache layout {tuple(kc.shape)}")


_ATTN_CKPT = ("q_proj", "k_proj", "v_proj", "o_proj", "kv_a_proj_with_mqa")
_PAT = re.compile(r"\.(self_attn\.(" + "|".join(_ATTN_CKPT) + r")|mlp\.experts\.\d+\.(gate|up|down)_proj)\.weight$")


def _attn_map(a):
    if hasattr(a, "qkv_proj"):
        return (("qkv_proj", ("q_proj", "k_proj", "v_proj")), ("o_proj", ("o_proj",)))
    if hasattr(a, "kv_a_proj_with_mqa") and getattr(a, "q_proj", None) is not None:
        return (("q_proj", ("q_proj",)), ("kv_a_proj_with_mqa", ("kv_a_proj_with_mqa",)), ("o_proj", ("o_proj",)))
    raise NotImplementedError(f"lora merge: unsupported attention {type(a).__name__}")


class LoRAMergeWorker:
    def _lm_init(self):
        from safetensors import safe_open
        root = self.model_runner.model_config.model
        if not os.path.isdir(root):
            from huggingface_hub import snapshot_download
            root = snapshot_download(root, local_files_only=True)
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            idx = json.load(f)["weight_map"]
        cfg = self.model_runner.model_config.hf_config
        L = cfg.num_hidden_layers
        names = sorted((n for n in idx if _PAT.search(n) and n.startswith("model.layers.")
                        and int(n.split(".")[2]) < L), key=lambda n: (int(n.split(".")[2]), n))
        self._lm_attn = _attn_map(self.model_runner.get_model().model.layers[0].self_attn)
        want = {m for _, ms in self._lm_attn for m in ms}
        names = [n for n in names if ".mlp.experts." in n or n.split(".")[4] in want]
        raw, by_file = {}, {}
        for n in names:
            by_file.setdefault(idx[n], []).append(n)
        for f, ns in by_file.items():
            with safe_open(os.path.join(root, f), framework="pt", device="cpu") as fh:
                for n in ns:
                    raw[n] = fh.get_tensor(n).to(torch.bfloat16)
        E_cfg = (getattr(cfg, "num_experts", None) or getattr(cfg, "moe_num_experts", None)
                 or getattr(cfg, "n_routed_experts", 0))
        self._lm_L, self._lm_E = L, E_cfg
        dev = self.device
        self._lm_layers, self._lm_rows = [], None
        for l in range(L):
            p = f"model.layers.{l}"
            g = lambda s: raw.pop(f"{p}.{s}.weight")
            moe = f"{p}.mlp.experts.0.gate_proj.weight" in raw
            attn, rows = {}, {}
            for eng, ms in self._lm_attn:
                parts = [g(f"self_attn.{m}") for m in ms]
                attn[eng] = torch.cat(parts, 0).to(dev) if len(parts) > 1 else parts[0].to(dev)
                off = [0]
                for t in parts:
                    off.append(off[-1] + t.shape[0])
                rows[eng] = [(m, off[i], off[i + 1]) for i, m in enumerate(ms)]
            assert self._lm_rows in (None, rows), f"layer {l}: attention shapes differ from layer 0"
            self._lm_rows = rows
            self._lm_layers.append({
                "attn": attn,
                "w13": torch.stack([torch.cat([g(f"mlp.experts.{e}.gate_proj"), g(f"mlp.experts.{e}.up_proj")], 0)
                                    for e in range(E_cfg)]).to(dev) if moe else None,
                "w2": torch.stack([g(f"mlp.experts.{e}.down_proj") for e in range(E_cfg)]).to(dev) if moe else None,
            })
        assert not raw, f"unused base tensors: {list(raw)[:4]}"
        if "qkv_proj" in self._lm_rows:
            hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
            q_out, kv_out = cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd
            assert [(r0, r1) for _, r0, r1 in self._lm_rows["qkv_proj"]] == \
                [(0, q_out), (q_out, q_out + kv_out), (q_out + kv_out, q_out + 2 * kv_out)], self._lm_rows
        ok, why = self._lm_selfcheck()
        if not ok:
            raise RuntimeError(f"lora merge: the engine's weights are not the online-FP8 base copy ({why})")

    def _lm_targets(self, l):
        m = self.model_runner.get_model().model.layers[l]
        a = m.self_attn
        xs = [mod for mod in m.mlp.modules() if "w13_weight" in dict(mod.named_parameters(recurse=False))]
        assert len(xs) <= 1, f"layer {l}: {len(xs)} modules own w13_weight"
        lins = {}
        for eng, _ in self._lm_attn:
            lin = getattr(a, eng)
            lins[eng] = getattr(lin, "base_layer", lin)
        return lins, (xs[0] if xs else None)

    @staticmethod
    def _q_tensor(w):
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
        lo, hi = w.aminmax()
        amax = torch.maximum(lo.abs(), hi.abs()).reshape(1).to(torch.float32)
        scale = amax / torch.tensor(get_fp8_min_max()[1], device=w.device, dtype=torch.float32)
        q, _ = ops.scaled_fp8_quant(w, scale=scale)
        return q, scale

    @staticmethod
    def _q_experts(w):
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
        from vllm.platforms import current_platform
        lo, hi = w.flatten(1).aminmax(dim=-1)
        amax = torch.maximum(lo.abs(), hi.abs()).to(torch.float32)
        scale = amax / torch.tensor(get_fp8_min_max()[1], device=w.device, dtype=torch.float32)
        q = torch.empty_like(w, dtype=current_platform.fp8_dtype())
        for e in range(w.shape[0]):
            q[e], _ = ops.scaled_fp8_quant(w[e], scale=scale[e])
        return q, scale

    def _lm_write(self, l, attn_w, w13, w2):
        lins, x = self._lm_targets(l)
        for eng, w in attn_w.items():
            lin = lins[eng]
            q, s = self._q_tensor(w)
            lin.weight.data.copy_(q.t())
            lin.weight_scale.data.copy_(s.reshape(lin.weight_scale.shape))
        for name, w in (("w13", w13), ("w2", w2)):
            if w is None:
                assert x is None, f"layer {l}: engine has experts, base copy has none"
                continue
            q, s = self._q_experts(w)
            getattr(x, f"{name}_weight").data.copy_(q)
            getattr(x, f"{name}_weight_scale").data.copy_(s.reshape(getattr(x, f"{name}_weight_scale").shape))

    def _lm_selfcheck(self):
        for l in range(self._lm_L):
            lins, x = self._lm_targets(l)
            lay = self._lm_layers[l]
            if (x is None) != (lay["w13"] is None):
                return False, f"L{l} experts present in engine={x is not None} base={lay['w13'] is not None}"
            for nm, w in lay["attn"].items():
                lin = lins[nm]
                if not hasattr(lin, "weight_scale"):
                    return False, f"L{l} {nm}: engine layer has no weight_scale (not online FP8?)"
                q, s = self._q_tensor(w)
                if tuple(lin.weight.shape) != tuple(q.t().shape):
                    return False, f"L{l} {nm} shape {tuple(lin.weight.shape)} vs {tuple(q.t().shape)}"
                if not torch.equal(lin.weight.data.t().contiguous().view(torch.uint8), q.contiguous().view(torch.uint8)):
                    return False, f"L{l} {nm} weight bytes differ"
                if not torch.equal(lin.weight_scale.data.reshape(-1).float(), s.reshape(-1)):
                    return False, f"L{l} {nm} scale differs"
            for nm in ("w13", "w2") if x is not None else ():
                q, s = self._q_experts(lay[nm])
                pw, ps = getattr(x, f"{nm}_weight").data, getattr(x, f"{nm}_weight_scale").data
                if tuple(pw.shape) != tuple(q.shape):
                    return False, f"L{l} {nm} shape {tuple(pw.shape)} vs {tuple(q.shape)}"
                if not torch.equal(pw.view(torch.uint8), q.view(torch.uint8)):
                    return False, f"L{l} {nm} weight bytes differ"
                if not torch.equal(ps.reshape(-1).float(), s.reshape(-1)):
                    return False, f"L{l} {nm} scale differs"
        return True, ""

    def merge_lora_adapter(self, path):
        from safetensors.torch import load_file
        if not hasattr(self, "_lm_layers"):
            self._lm_init()
        W = load_file(os.path.join(path, "adapter_model.safetensors"), device=str(self.device))
        with open(os.path.join(path, "adapter_config.json")) as f:
            cfg = json.load(f)
        s = float(cfg["lora_alpha"]) / float(cfg["r"])
        n_ad = sum(1 for k in W if k.endswith("lora_A.weight"))
        used = 0
        k_ = lambda n, ab: W.get(f"base_model.model.{n}.lora_{ab}.weight")
        E = self._lm_E
        with torch.no_grad():
            for l, lay in enumerate(self._lm_layers):
                p = f"model.layers.{l}"
                outs = {}
                for nm, base in lay["attn"].items():
                    w = base.float()
                    for m, r0, r1 in self._lm_rows[nm]:
                        A, B = k_(f"{p}.self_attn.{m}", "A"), k_(f"{p}.self_attn.{m}", "B")
                        if A is not None:
                            assert B.shape[0] == r1 - r0, (m, B.shape, r0, r1)
                            w[r0:r1] += s * (B.float() @ A.float())
                            used += 1
                    outs[nm] = w.to(torch.bfloat16)
                if lay["w13"] is None:
                    self._lm_write(l, outs, None, None)
                    continue
                I = lay["w2"].shape[-1]
                w13, w2 = lay["w13"].float(), lay["w2"].float()
                for proj, sl, tgt in (("gate_proj", slice(0, I), w13), ("up_proj", slice(I, 2 * I), w13),
                                      ("down_proj", slice(None), w2)):
                    As = [k_(f"{p}.mlp.experts.{e}.{proj}", "A") for e in range(E)]
                    Bs = [k_(f"{p}.mlp.experts.{e}.{proj}", "B") for e in range(E)]
                    if As[0] is None:
                        continue
                    tgt[:, sl] += s * torch.bmm(torch.stack(Bs).float(), torch.stack(As).float())
                    used += E
                self._lm_write(l, outs, w13.to(torch.bfloat16), w2.to(torch.bfloat16))
        assert used == n_ad, f"merged {used} of {n_ad} adapter modules (name mismatch?)"
