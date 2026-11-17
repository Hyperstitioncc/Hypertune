"""LoRA in vLLM by gather: a reference for what a LoRA kernel costs when its work does not grow with the number
of adapters in the batch.

vLLM's GPU LoRA kernels launch work per (token block, active adapter). This wrapper replaces them with the form
vLLM itself uses off GPU (lora/ops/torch_ops, Punica's BGMV): pick every token's A and B by its adapter index and
do one batched product. Same adapters, same stacked weight tensors, same engine; only the two products change.
It is not part of vLLM's GPU path: results with it are labelled "gather" and are a reference, not "stock vLLM".

It also removes one cost that is not in the kernels. On every step vLLM re-activates every adapter of the batch,
and for each one rebuilds the set of loaded adapters (LRUCacheWorkerLoRAManager._apply_adapters -> add_adapter ->
list_adapters): Python work that grows with the square of the number of adapters, about 20 to 45 ms a step at 512.
Here a step whose adapter set is the one of the step before skips it.

    LLM(..., enable_lora=True, worker_extension_cls="lora_gather.GatherWorker")      (this folder on PYTHONPATH)

Linear layers: y[t] += B[a_t] (A[a_t] x[t]).   MoE experts, for token t, its k-th routed expert e and adapter a:
    gate/up:  y[t, k, slice] += B13[a, e] (A13[a, e] x[t])
    down:     y[t, k]        += w[t, k] * B2[a, e] (A2[a, e] h[t, k])
which is what fused_moe_lora computes (lora/ops/triton_ops/fused_moe_lora_op.py).

SS_GATHER_CHECK=1 (with enforce_eager): every call also runs vLLM's own kernel on a copy of the output and prints
how far the two LoRA contributions are apart ("GATHERCHECK ..." lines, the first few calls of each kind)."""
from __future__ import annotations

import collections
import os

import torch

from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.platforms import current_platform


CHECK = os.environ.get("SS_GATHER_CHECK") == "1"
_seen: collections.Counter = collections.Counter()


def _report(kind: str, got: torch.Tensor, ref: torch.Tensor, base: torch.Tensor) -> None:
    """got / ref: the output with the LoRA contribution added by gather / by vLLM's kernel; base: before either"""
    if _seen[kind] >= 6:
        return
    d_ref, d_got = ref.float() - base.float(), got.float() - base.float()
    if not bool(d_ref.any()) and not bool(d_got.any()):
        return                      # the engine's start-up passes: no adapter loaded yet
    _seen[kind] += 1
    print(f"GATHERCHECK {kind}: mean |LoRA contribution| vLLM kernel {d_ref.abs().mean():.3e}, gather {d_got.abs().mean():.3e}; "
          f"difference relative to the contribution {(d_ref - d_got).norm() / (d_ref.norm() + 1e-12):.4f}; shape {tuple(base.shape)}", flush=True)


class GatherPunica(PunicaWrapperGPU):
    def _adapters(self, n: int, mapping: torch.Tensor | None = None):
        idx = self.token_mapping_meta.token_lora_mapping[:n] if mapping is None else mapping[:n]
        return idx.clamp(min=0).long(), idx >= 0

    def add_lora_linear(self, y, x, lora_a_stacked, lora_b_stacked, scale, output_slices, *, buffer=None, **kwargs):
        add_inputs = kwargs.pop("add_inputs", True)
        if CHECK:
            base, ref = y.clone(), y.clone()
            super().add_lora_linear(ref, x, lora_a_stacked, lora_b_stacked, scale, output_slices, add_inputs=add_inputs)
        a, on = self._adapters(x.size(0))
        on = on.unsqueeze(-1).to(y.dtype)
        off = 0
        for A, B, size in zip(lora_a_stacked, lora_b_stacked, output_slices):
            Ag = A.reshape(A.shape[0], A.shape[-2], A.shape[-1])[a]            # [T, r, in]
            Bg = B.reshape(B.shape[0], B.shape[-2], B.shape[-1])[a]            # [T, out, r]
            mid = torch.bmm(Ag.to(x.dtype), x.unsqueeze(-1))                    # [T, r, 1]
            d = torch.bmm(Bg.to(y.dtype), mid.to(y.dtype)).squeeze(-1) * on
            if scale != 1.0:
                d = d * scale
            if add_inputs:
                y[:, off:off + size] += d
            else:
                y[:, off:off + size] = d
            off += size
        if CHECK:
            _report(f"linear, {len(output_slices)} slice(s) of {output_slices[0]}", y, ref, base if add_inputs else torch.zeros_like(y))

    def _experts(self, A, B, a, topk_ids):
        """A [L, E, r, in], B [L, E, out, r] -> the rows of every (token, routed expert) pair: [T, K, r, in], [T, K, out, r]"""
        E = A.shape[1]
        flat = (a.unsqueeze(-1) * E + topk_ids.long()).reshape(-1)
        T, K = topk_ids.shape
        return (A.reshape(-1, A.shape[-2], A.shape[-1])[flat].view(T, K, A.shape[-2], A.shape[-1]),
                B.reshape(-1, B.shape[-2], B.shape[-1])[flat].view(T, K, B.shape[-2], B.shape[-1]))

    def add_lora_w13(self, y, x, lora_a_stacked, lora_b_stacked, topk_ids, topk_weights, expert_map, w1, w2, num_tokens,
                     top_k_num, max_loras, adapter_enabled, local_num_experts, top_k, num_slices, fully_sharded,
                     use_tuned_config, add_inputs=True, token_lora_mapping=None):
        if expert_map is not None or fully_sharded:
            raise NotImplementedError("gather LoRA: one GPU, no expert parallelism")
        ret = (None, None, None, token_lora_mapping)
        if CHECK:
            base, ref = y.clone(), y.clone()
            ret = super().add_lora_w13(ref, x, lora_a_stacked, lora_b_stacked, topk_ids, topk_weights, expert_map, w1, w2,
                                       num_tokens, top_k_num, max_loras, adapter_enabled, local_num_experts, top_k, num_slices,
                                       fully_sharded, use_tuned_config, add_inputs=add_inputs, token_lora_mapping=token_lora_mapping)
        a, on = self._adapters(num_tokens, token_lora_mapping)
        on = (on & (adapter_enabled[a] > 0)).view(-1, 1, 1).to(y.dtype)
        out = y.view(num_tokens, top_k_num, -1)
        width = out.shape[-1] // len(lora_a_stacked)
        xt = x.view(num_tokens, -1)
        for s, (A, B) in enumerate(zip(lora_a_stacked, lora_b_stacked)):
            Ag, Bg = self._experts(A, B, a, topk_ids)
            mid = torch.einsum("tkrh,th->tkr", Ag.to(xt.dtype), xt)
            d = torch.einsum("tkor,tkr->tko", Bg.to(y.dtype), mid.to(y.dtype)) * on
            if add_inputs:
                out[:, :, s * width:(s + 1) * width] += d
            else:
                out[:, :, s * width:(s + 1) * width] = d
        self._topk_ids = topk_ids          # the down projection of this layer is called next and is not given them
        if CHECK:
            _report("experts gate/up", y, ref, base if add_inputs else torch.zeros_like(y))
        return ret

    def add_lora_w2(self, y, x, lora_a_stacked, lora_b_stacked, topk_weights, sorted_token_ids_lora, expert_ids_lora,
                    num_tokens_post_padded_lora, token_lora_mapping, num_tokens, w1, w2, top_k_num, max_loras,
                    adapter_enabled, top_k, fully_sharded, tp_rank, use_tuned_config, add_inputs=True):
        if CHECK:
            base, ref = y.clone(), y.clone()
            super().add_lora_w2(ref, x, lora_a_stacked, lora_b_stacked, topk_weights, sorted_token_ids_lora, expert_ids_lora,
                                num_tokens_post_padded_lora, token_lora_mapping, num_tokens, w1, w2, top_k_num, max_loras,
                                adapter_enabled, top_k, fully_sharded, tp_rank, use_tuned_config, add_inputs=add_inputs)
        a, on = self._adapters(num_tokens, token_lora_mapping)
        on = (on & (adapter_enabled[a] > 0)).view(-1, 1, 1).to(y.dtype)
        topk_ids = self._topk_ids
        Ag, Bg = self._experts(lora_a_stacked[0], lora_b_stacked[0], a, topk_ids)
        h = x.reshape(num_tokens, top_k_num, -1)
        mid = torch.einsum("tkri,tki->tkr", Ag.to(h.dtype), h)
        d = torch.einsum("tkor,tkr->tko", Bg.to(y.dtype), mid.to(y.dtype))
        d = d * topk_weights.view(num_tokens, top_k_num, 1).to(y.dtype) * on
        out = y.view(num_tokens, top_k_num, -1)
        if add_inputs:
            out += d
        else:
            out.copy_(d)
        if CHECK:
            _report("experts down", y, ref, base if add_inputs else torch.zeros_like(y))


class GatherWorker:
    """worker_extension_cls: importing this module in the engine's worker process is what installs the wrapper."""


def _apply_adapters_once(self, lora_requests) -> None:
    ids = frozenset(r.lora_int_id for r in lora_requests if r)
    active = self._adapter_manager._active_adapters         # the adapters that hold a GPU slot right now
    same = ids == getattr(self, "_same_as_last_step", None) and all(i in active for i in ids)
    if not same or any(r.load_inplace for r in lora_requests if r):
        _apply_adapters(self, lora_requests)
        self._same_as_last_step = ids


type(current_platform).get_punica_wrapper = classmethod(lambda cls: "lora_gather.GatherPunica")
_apply_adapters = LRUCacheWorkerLoRAManager._apply_adapters
LRUCacheWorkerLoRAManager._apply_adapters = _apply_adapters_once
