from __future__ import annotations

import os
from typing import Callable

import torch

from adapters.kv_prefix import NDIM

# rows the engine holds: every cartridge that can be in a batch, 16 rows each (KV_GRAPH_ROWS=8192 serves 512)
ROWS_MAX = int(os.environ.get("KV_GRAPH_ROWS", "256"))


def match_rows(e: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    return (e[..., None, :NDIM] == rows[:, :NDIM].to(e.dtype)).all(-1)


def _key_weights(n: int) -> list[int]:
    """n unrelated odd 64-bit multipliers (splitmix64), as signed integers: weights that are multiples of one
    another would make the key a function of one small sum, and rows would collide"""
    out, x = [], 0x9E3779B97F4A7C15
    for _ in range(n):
        x = (x + 0x9E3779B97F4A7C15) % (1 << 64)
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) % (1 << 64)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) % (1 << 64)
        z = (z ^ (z >> 31)) | 1
        out.append(z - (1 << 64) if z >= (1 << 63) else z)
    return out


KEY_W = torch.tensor(_key_weights(NDIM), dtype=torch.int64)


def row_keys(e: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """One integer per row of e[:, :NDIM]: an exact hash of the bfloat16 bit patterns, so that a token is looked up
    in the sorted keys of the table (find_rows) and not compared with every row of it."""
    bits = e[:, :NDIM].to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int64)
    return (bits * w).sum(-1)


def find_rows(e: torch.Tensor, base: torch.Tensor, keys: torch.Tensor, order: torch.Tensor, w: torch.Tensor):
    """For each token of e: the table row it is, and whether it is one (same answer as match_rows, at a cost that
    does not grow with the table). keys = the table's row_keys sorted, order = the row of each sorted key."""
    at = torch.searchsorted(keys, row_keys(e, w)).clamp(max=keys.shape[0] - 1)
    cand = order[at]
    return cand, (e[:, :NDIM] == base[cand, :NDIM].to(e.dtype)).all(-1)


def sort_keys(base: torch.Tensor, w: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The sorted keys of a table of size rows whose first rows are base (the rest never match)."""
    keys = torch.full((size,), torch.iinfo(torch.int64).max, dtype=torch.int64, device=w.device)
    keys[:base.shape[0]] = row_keys(base.to(w.device), w)
    return torch.sort(keys)


def row_positions(positions: torch.Tensor, hit: torch.Tensor) -> torch.Tensor:
    """Rotary positions of one flat engine batch whose rows (hit) sit at position 0 and take no position numbers.
    The engine numbers the rows anyway, so each sequence that holds K rows is written K later as a whole: its tokens
    before the rows move to position + K and its rows to K, which leaves the tokens after the rows (in the prefill
    and at every decode step) on the engine's own numbers. Rotary attention reads differences only, so this equals
    rows at 0 with the later tokens numbered K earlier. A sequence's rows and the tokens before them must arrive in
    one forward (no chunked prefill, no prefix cache)."""
    n = positions.shape[0]
    h = hit.to(positions.dtype)
    brk = torch.ones(n, dtype=positions.dtype, device=positions.device)
    brk[1:] = (positions[1:] != positions[:-1] + 1).to(positions.dtype)
    seg = torch.cumsum(brk, 0) - 1
    rows = torch.zeros(n, dtype=positions.dtype, device=positions.device).scatter_add_(0, seg, h)[seg]
    seen = torch.cumsum(h, 0)
    start = torch.full((n,), n + 1, dtype=positions.dtype, device=positions.device).scatter_reduce_(
        0, seg, seen - h, "amin")[seg]
    return torch.where(hit, rows, torch.where(seen - start < rows, positions + rows, positions))


def tp_heads(t: torch.Tensor, dim: int, n_local: int) -> torch.Tensor:
    n = t.shape[dim]
    if n == n_local:
        return t
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    tp, r = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
    if n != n_local * tp:
        raise ValueError(f"{n} heads do not split over tp={tp} into {n_local} per rank")
    return t.narrow(dim, r * n_local, n_local)


def attn_forward(at, positions: torch.Tensor, hidden_states: torch.Tensor, kv=None, pos=None, post=None) -> torch.Tensor:
    qkv, _ = at.qkv_proj(hidden_states)
    q, k, v = qkv.split([at.q_size, at.kv_size, at.kv_size], dim=-1)
    hd = at.head_dim
    qh = at.q_norm(q.view(*q.shape[:-1], q.shape[-1] // hd, hd))
    kh = at.k_norm(k.view(*k.shape[:-1], k.shape[-1] // hd, hd))
    vh = v.view(*v.shape[:-1], v.shape[-1] // hd, hd)
    if kv is not None:
        kh, vh = kv(kh, vh)
    if pos is not None:
        positions = pos(positions)
    q_pre = qh.clone() if post is not None else qh
    qr, kr = at.rotary_emb(positions, qh.view(q.shape), kh.reshape(k.shape))
    o = at.attn(qr, kr, vh.reshape(v.shape))
    if post is not None:
        o = post(o, q_pre)
    out, _ = at.o_proj(o)
    return out


def patch(A: type, M: type, dev: Callable[[], torch.device]) -> None:
    a_init, m_fwd = A.__init__, M.forward
    st: dict = {}

    def bufs() -> dict:
        if not st:
            from vllm.config import get_current_vllm_config
            sc = get_current_vllm_config().scheduler_config
            T = max(sc.max_num_batched_tokens, sc.max_num_seqs) + 1024
            d = dev()
            st.update(idx=torch.zeros(T, dtype=torch.long, device=d),
                      has=torch.zeros(T, dtype=torch.bfloat16, device=d),
                      base=torch.full((ROWS_MAX, NDIM), float("nan"), dtype=torch.bfloat16, device=d),
                      rep=torch.zeros(1, dtype=torch.bfloat16, device=d),
                      pos0=torch.zeros(1, dtype=torch.bool, device=d),
                      pos=torch.zeros(T, dtype=torch.long, device=d),
                      keyw=KEY_W.to(d),
                      keys=torch.full((ROWS_MAX,), torch.iinfo(torch.int64).max, dtype=torch.int64, device=d),
                      order=torch.zeros(ROWS_MAX, dtype=torch.long, device=d))
        return st

    def init(self, *a, **kw):
        a_init(self, *a, **kw)
        b = bufs()
        z = torch.zeros(ROWS_MAX, self.num_kv_heads, self.head_dim, dtype=torch.bfloat16, device=b["has"].device)
        for name in ("idx", "has", "base", "rep", "pos0", "keyw", "keys", "order"):
            self.register_buffer(f"kvi_{name}", b[name], persistent=False)
        self.register_buffer("kvi_dk", z.clone(), persistent=False)
        self.register_buffer("kvi_dv", z.clone(), persistent=False)

    def fwd(self, positions, hidden_states):
        # each token is at most one row: its table index and a 0/1 flag, so the cost does not grow with the table
        n = hidden_states.shape[0]
        idx, has = self.kvi_idx[:n], self.kvi_has[:n, None, None]
        keep = 1 - self.kvi_rep * has

        def kv(kh, vh):
            return kh * keep + has * self.kvi_dk[idx], vh * keep + has * self.kvi_dv[idx]
        return attn_forward(self, positions, hidden_states, kv=kv)

    def mfwd(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kw):
        idx, has = st["idx"], st["has"]
        if inputs_embeds is not None:
            n = inputs_embeds.shape[0]
            # the engine passes embeddings on every step once prompt embeddings are enabled: this runs per decode step
            cand, row = find_rows(inputs_embeds, st["base"], st["keys"], st["order"], st["keyw"])
            idx[:n].copy_(cand)
            has[:n].copy_(row.to(has.dtype))
            has[n:].zero_()
            # the shifted numbers go to the model in a buffer of our own (fixed address, as CUDA graphs need), never
            # into the engine's positions, which drafters (--spec) and the runner read after this forward
            pos = st["pos"][: positions.shape[0]]
            pos.copy_(positions)
            pos[:n].copy_(row_positions(positions[:n], row & st["pos0"]))
            positions = pos
        else:
            has.zero_()
        return m_fwd(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kw)

    A.__init__, A.forward, M.forward = init, fwd, mfwd


def patch_vllm() -> None:
    import vllm.model_executor.models.qwen3 as qd
    import vllm.model_executor.models.qwen3_moe as qm
    dev = lambda: torch.device("cuda", torch.cuda.current_device())
    patch(qm.Qwen3MoeAttention, qm.Qwen3MoeForCausalLM, dev)
    patch(qd.Qwen3Attention, qd.Qwen3ForCausalLM, dev)          # the dense Qwen3 has the same attention block


class KVGraphWorker:
    def set_kv_delta(self, path):
        st = torch.load(path, map_location="cpu")
        layers = self.model_runner.model.model.layers
        at0 = layers[0].self_attn
        dk = tp_heads(st["dk"], 2, at0.num_kv_heads)
        dv = tp_heads(st["dv"], 2, at0.num_kv_heads)
        K = dk.shape[1]
        if K > ROWS_MAX:
            raise ValueError(f"kv_graph holds {ROWS_MAX} rows, payload has {K}")
        with torch.no_grad():
            for li, layer in enumerate(layers):
                layer.self_attn.kvi_dk.zero_()
                layer.self_attn.kvi_dv.zero_()
                layer.self_attn.kvi_dk[:K].copy_(dk[li].to(torch.bfloat16))
                layer.self_attn.kvi_dv[:K].copy_(dv[li].to(torch.bfloat16))
            at0.kvi_base.fill_(float("nan"))
            at0.kvi_base[:K].copy_(st["base"][:, :NDIM].to(torch.bfloat16))
            keys, order = sort_keys(at0.kvi_base[:K], at0.kvi_keyw, ROWS_MAX)
            at0.kvi_keys.copy_(keys)
            at0.kvi_order.copy_(order)
            at0.kvi_rep.fill_(1.0 if st["replace"] else 0.0)
            at0.kvi_pos0.fill_(bool(st["pos0"]))
