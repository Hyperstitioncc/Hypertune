"""CPU stand-in for vLLM: the surface shared.rollout uses, with the engine's contracts enforced.

Decoding is real: tokens are sampled from the checkpoint (HF, fp32, full forward per step) through the
request's context, so `logprobs` are the raw model logprobs of the sampled ids. prompt_embeds rows are
matched bit-for-bit against the checkpoint's bf16 embedding table; unmatched rows are soft rows. LoRA
requests / merged adapters are applied as W + alpha/r * B A; full weights are reloaded from disk; KV
payloads are recorded (their numerics are not emulated). Stops follow vLLM: generation ends at the token
that completes a stop string, the text excludes it, token_ids include it.
"""
import json
import os

import torch

__version__ = "0.0-fake"


class SamplingParams:
    def __init__(self, n=1, temperature=1.0, top_p=1.0, max_tokens=16, stop=None, logprobs=None, seed=None,
                 output_kind=None, **kw):
        assert not kw, f"fake SamplingParams: unsupported args {kw}"
        self.n, self.temperature, self.top_p = n, temperature, top_p
        self.max_tokens, self.stop, self.logprobs, self.seed = max_tokens, list(stop or []), logprobs, seed


class _Logprob:
    def __init__(self, lp):
        self.logprob = lp


class _Completion:
    def __init__(self, text, token_ids, logprobs):
        self.text, self.token_ids, self.logprobs = text, token_ids, logprobs


class _RequestOutput:
    def __init__(self, outputs):
        self.outputs, self.finished = outputs, True


class _Core:
    def __init__(self):
        self.alive = True

    def shutdown(self):
        self.alive = False


class _Engine:
    def __init__(self):
        self.engine_core = _Core()


class LLM:
    def __init__(self, model, dtype="bfloat16", tensor_parallel_size=1, gpu_memory_utilization=0.9,
                 max_model_len=2048, seed=0, enable_prompt_embeds=False, enable_expert_parallel=False,
                 speculative_config=None, quantization=None, logprobs_mode=None, enable_lora=False, max_loras=1,
                 max_lora_rank=16, max_cpu_loras=None, lora_target_modules=None, worker_extension_cls="",
                 max_num_seqs=None, enforce_eager=False, trust_remote_code=False, compilation_config=None,
                 enable_prefix_caching=True, enable_chunked_prefill=True, **kw):
        assert not kw, f"fake LLM: unsupported args {sorted(kw)}"
        assert logprobs_mode in (None, "raw_logprobs"), logprobs_mode
        from safetensors import safe_open
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model, self.max_model_len, self.seed = model, max_model_len, seed
        self.enable_prompt_embeds, self.enable_lora, self.max_lora_rank = enable_prompt_embeds, enable_lora, max_lora_rank
        self.lora_targets, self.worker_ext, self.enforce_eager = lora_target_modules, worker_extension_cls, enforce_eager
        self.compilation_config = compilation_config
        self.prefix_caching, self.chunked_prefill = enable_prefix_caching, enable_chunked_prefill
        self.tok = AutoTokenizer.from_pretrained(model)
        with open(os.path.join(model, "model.safetensors.index.json")) as f:
            idx = json.load(f)
        with safe_open(os.path.join(model, idx["weight_map"]["model.embed_tokens.weight"]), framework="pt") as f:
            self.E = f.get_tensor("model.embed_tokens.weight").to(torch.bfloat16)
        bits = self.E.view(torch.int16).numpy()
        self.row2id = {bits[i].tobytes(): i for i in range(bits.shape[0])}
        self.hf = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).eval()
        self._w0 = {n: p.detach().clone() for n, p in self.hf.named_parameters()}
        self.merged, self.kv, self.n_full = None, None, 0
        self.llm_engine = _Engine()
        self.calls = []

    def _decode_request(self, p):
        if "prompt_token_ids" in p:
            ids = list(p["prompt_token_ids"])
            return self.hf.get_input_embeddings()(torch.tensor(ids)).float(), ids, 0
        assert self.enable_prompt_embeds, "prompt_embeds request but enable_prompt_embeds=False"
        x = p["prompt_embeds"]
        assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == self.E.shape[1], (x.dtype, tuple(x.shape))
        ids, n_soft = [], 0
        for r in x:
            t = self.row2id.get(r.view(torch.int16).numpy().tobytes())
            if t is None:
                n_soft += 1
            else:
                ids.append(t)
        assert len(x) <= self.max_model_len
        return x.float(), ids, n_soft

    def _set_weights(self, ad):
        named = dict(self.hf.named_parameters())
        mods = dict(self.hf.named_modules())
        with torch.no_grad():
            for n, w in self._w0.items():
                named[n].copy_(w)
            if ad is None:
                return
            cfg, W = ad
            s = cfg["lora_alpha"] / cfg["r"]
            for k, A in W.items():
                if not k.endswith(".lora_A.weight"):
                    continue
                n = k[len("base_model.model."):-len(".lora_A.weight")]
                dW = s * (W[k.replace("lora_A", "lora_B")].float() @ A.float())
                if ".experts." in n:
                    base, e, proj = n.rsplit(".", 2)
                    xm, e = mods[base], int(e)
                    I = xm.down_proj.shape[-1]
                    if proj == "down_proj":
                        xm.down_proj[e] += dW
                    else:
                        xm.gate_up_proj[e, (slice(0, I) if proj == "gate_proj" else slice(I, 2 * I))] += dW
                else:
                    assert n in mods, f"adapter key {k}: no module {n}"
                    mods[n].weight += dW

    def _load_lora(self, path):
        from safetensors.torch import load_file
        with open(os.path.join(path, "adapter_config.json")) as f:
            cfg = json.load(f)
        assert cfg["peft_type"] == "LORA" and cfg["r"] <= self.max_lora_rank, cfg
        if self.lora_targets is not None:
            assert set(cfg["target_modules"]) <= set(self.lora_targets), cfg
        return cfg, load_file(os.path.join(path, "adapter_model.safetensors"))

    def collective_rpc(self, method, args=(), kwargs=None):
        if method == "merge_lora_adapter":
            assert self.worker_ext.endswith("LoRAMergeWorker") and not self.enable_lora
            self.merged = self._load_lora(args[0])
            return [{"merged": True}]
        if method == "set_kv_cart":
            assert self.worker_ext.endswith("KVInjectWorker") and self.enforce_eager, self.worker_ext
            self.kv = torch.load(args[0], map_location="cpu")
            return [None]
        if method == "set_kv_delta":
            assert ((self.worker_ext.endswith("KVInjectWorker") and self.enforce_eager)
                    or (self.worker_ext.endswith("KVGraphWorker") and not self.enforce_eager
                        and self.compilation_config is not None)), self.worker_ext
            self.kv = torch.load(args[0], map_location="cpu")
            return ["kv set"]
        if method == "load_full_weights":
            assert self.worker_ext.endswith("FullReloadWorker")
            from transformers import AutoModelForCausalLM
            m = AutoModelForCausalLM.from_pretrained(args[0], dtype=torch.float32)
            self._w0 = {n: p.detach().clone() for n, p in m.named_parameters()}
            self.n_full += 1
            return [{"files": 1}]
        raise AssertionError(f"fake LLM: unknown rpc {method}")

    def reset_prefix_cache(self):
        return True

    @torch.no_grad()
    def _sample(self, ctx, sp, g):
        emb = self.hf.get_input_embeddings()
        x, out, lps = ctx, [], []
        text = ""
        for _ in range(sp.max_tokens):
            lg = self.hf(inputs_embeds=x[None]).logits[0, -1].float()
            lp = torch.log_softmax(lg, -1)
            if sp.temperature == 0.0:
                t = int(lg.argmax())
            else:
                pr = torch.softmax(lg / sp.temperature, -1)
                sv, si = torch.sort(pr, descending=True)
                sv = sv.masked_fill(torch.cumsum(sv, 0) - sv > sp.top_p, 0.0)
                t = int(si[torch.multinomial(sv / sv.sum(), 1, generator=g)])
            out.append(t)
            lps.append({t: _Logprob(float(lp[t]))})
            if t == self.tok.eos_token_id:
                break
            text = self.tok.decode(out, skip_special_tokens=True)
            if any(s in text for s in sp.stop):
                break
            x = torch.cat([x, emb(torch.tensor([t])).float()], 0)
        for s in sp.stop:
            i = text.find(s)
            if i >= 0:
                text = text[:i]
        return _Completion(text, out, lps if sp.logprobs is not None else None)

    def generate(self, prompts, sampling_params, use_tqdm=True, lora_request=None):
        assert self.llm_engine.engine_core.alive, "generate() after shutdown()"
        sps = (list(sampling_params) if isinstance(sampling_params, (list, tuple))
               else [sampling_params] * len(prompts))
        assert len(sps) == len(prompts), "one SamplingParams per prompt"
        if lora_request is not None:
            assert self.enable_lora, "lora_request on an engine without enable_lora"
            assert self.merged is None
            ad = self._load_lora(lora_request.lora_path)
        else:
            ad = self.merged
        self._set_weights(ad)
        self.calls.append({"lora": lora_request, "n": len(prompts)})
        outs = []
        for p, sp in zip(prompts, sps):
            ctx, ids, n_soft = self._decode_request(p)
            assert ad is None or n_soft == 0, "LoRA request with soft rows"
            g = torch.Generator().manual_seed(sp.seed if sp.seed is not None else self.seed)
            outs.append(_RequestOutput([self._sample(ctx, sp, g) for _ in range(sp.n)]))
        return outs
