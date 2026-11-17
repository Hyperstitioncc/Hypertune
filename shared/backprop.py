from __future__ import annotations

import contextlib
import enum
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from shared.adapter import Adapter
from shared.attach import Attached, attach, positions, splice
from shared.decoder import Decoder
from shared.types import Backend, TokenRow


class ScoreMode(enum.Enum):
    prefix = "prefix"
    pack = "pack"
    row = "row"


@dataclass(frozen=True)
class BackpropCfg:
    score_mode: ScoreMode = ScoreMode.prefix
    unit_tokens: int = 8192


def encode(decoder, text: str) -> tuple[int, ...]:
    return tuple(decoder.tok.encode(text, add_special_tokens=False))


def target(decoder, prompt: str, text: str, max_tokens: int) -> TokenRow:
    return TokenRow(prompt_ids=encode(decoder, prompt), ids=encode(decoder, text)[:max_tokens])


def units(rows: list[TokenRow], k: int, cfg: BackpropCfg) -> list[list[int]]:
    if cfg.score_mode is ScoreMode.row:
        return [[j] for j in range(len(rows))]
    if cfg.score_mode is ScoreMode.pack:
        out, cur, n = [], [], 0
        for j, r in enumerate(rows):
            T = len(r.prompt_ids) + k + len(r.ids)
            if cur and n + T > cfg.unit_tokens:
                out.append(cur)
                cur, n = [], 0
            cur.append(j)
            n += T
        return out + ([cur] if cur else [])
    groups: dict[tuple, list[int]] = {}
    for j, r in enumerate(rows):
        groups.setdefault(tuple(r.prompt_ids), []).append(j)
    out = []
    for idxs in groups.values():
        cur, n = [], 0
        for j in idxs:
            S = 1 + k + len(rows[j].ids)
            if cur and n + S > cfg.unit_tokens:
                out.append(cur)
                cur, n = [], 0
            cur.append(j)
            n += S
        if cur:
            out.append(cur)
    return out


def _head(dec: Decoder, h: torch.Tensor, sel: list[torch.Tensor], cand: list[torch.Tensor]) -> list[torch.Tensor]:
    idx = torch.cat(sel).to(h.device)
    lp = F.log_softmax(dec.head(h[idx].to(dec.hdev)).float(), -1)
    g = lp.gather(1, torch.cat(cand).to(dec.hdev).long())
    return list(g.split([c.shape[0] for c in cand]))


def _ids(dec: Decoder, ids) -> torch.Tensor:
    return dec.emb(torch.tensor(list(ids), device=dec.dev))


def _forward_packed(dec: Decoder, e: Attached, rows: list[TokenRow], cand: list[torch.Tensor]) -> list[torch.Tensor]:
    segs, pos, sel = [], [], []
    off = 0
    for r in rows:
        P, Lr = len(r.prompt_ids), len(r.ids)
        soft = e.rows(r.prompt_ids).to(dec.dev, dec.dtype)
        k = soft.shape[0]
        T = P + k + Lr
        segs += [splice(e.row_tail, _ids(dec, r.prompt_ids), soft), _ids(dec, r.ids)]
        pos.append(positions(e, P, k, T, dec.dev))
        sel.append(torch.arange(off + P + k - 1, off + T - 1, device=dec.dev))
        off += T
    mask = torch.ones(1, off, dtype=torch.long, device=dec.dev) if len(rows) == 1 else None
    h = dec.model.model(inputs_embeds=torch.cat(segs).to(dec.dtype).unsqueeze(0),
                        position_ids=torch.cat(pos).unsqueeze(0), attention_mask=mask,
                        use_cache=False).last_hidden_state[0]
    return _head(dec, h, sel, cand)


@contextlib.contextmanager
def _sdpa_for(model, on: bool):
    """The prompt pass of a prefix unit is plain causal attention with no cache and no mask. With grad (an adapter that
    changes the prompt, e.g. LoRA) eager attention keeps L x L scores per head per layer for backward, ~150-250 GB at
    3-4k tokens; SDPA computes the same function in O(L) memory."""
    cfg = model.config
    if not on or cfg._attn_implementation != "eager":
        yield
        return
    cfg._attn_implementation = "sdpa"
    try:
        yield
    finally:
        cfg._attn_implementation = "eager"


def _forward_prefix(dec: Decoder, e: Attached, rows: list[TokenRow], cand: list[torch.Tensor]) -> list[torch.Tensor]:
    from transformers import DynamicCache
    p_ids = rows[0].prompt_ids
    assert all(r.prompt_ids == p_ids for r in rows), "prefix unit mixes prompts"
    P = len(p_ids)
    C = P - 1
    cache = DynamicCache()
    if C:
        grad = torch.is_grad_enabled() and e.prompt_grad
        with torch.set_grad_enabled(grad), _sdpa_for(dec.model, grad):
            dec.model.model(inputs_embeds=_ids(dec, p_ids[:-1]).to(dec.dtype).unsqueeze(0),
                            position_ids=torch.arange(C, device=dec.dev).unsqueeze(0),
                            past_key_values=cache, use_cache=True)
    last = _ids(dec, p_ids[-1:])
    soft = e.rows(p_ids).to(dec.dev, dec.dtype)
    k = soft.shape[0]
    segs, pos, seg_of, sel = [], [], [], []
    off = 0
    for j, r in enumerate(rows):
        Lr = len(r.ids)
        S = 1 + k + Lr
        segs += [splice(e.row_tail, last, soft), _ids(dec, r.ids)]
        pos.append(positions(e, 1, k, S, dec.dev, start=C))
        seg_of.append(torch.full((S,), j, device=dec.dev))
        sel.append(torch.arange(off + k, off + k + Lr, device=dec.dev))
        off += S
    Q = off
    seg = torch.cat(seg_of)
    qi = torch.arange(Q, device=dec.dev)
    own = (seg[:, None] == seg[None, :]) & (qi[None, :] <= qi[:, None])
    mask = torch.cat([torch.ones(Q, C, dtype=torch.bool, device=dec.dev), own], 1)
    if dec.cfg.attn == "eager":
        mask = torch.zeros(mask.shape, dtype=dec.dtype, device=dec.dev).masked_fill(~mask, torch.finfo(dec.dtype).min)
    h = dec.model.model(inputs_embeds=torch.cat(segs).to(dec.dtype).unsqueeze(0),
                        position_ids=torch.cat(pos).unsqueeze(0), attention_mask=mask[None, None],
                        past_key_values=cache, use_cache=True).last_hidden_state[0]
    return _head(dec, h, sel, cand)


def logprobs(decoder: Decoder, adapter: Adapter, rows: list[TokenRow], cand: list[torch.Tensor],
             cfg: BackpropCfg) -> list[torch.Tensor]:
    e = attach(adapter, Backend.hf_eager, decoder)
    if e.positional and cfg.score_mode is not ScoreMode.row:
        raise ValueError(f"{adapter.kind.value} addresses rows by position: score_mode row only")
    if cfg.score_mode is ScoreMode.pack:
        if decoder.cfg.attn != "flash_attention_2":
            raise ValueError("score_mode pack needs attn=flash_attention_2")
        if e.pos0:
            raise ValueError("pos0 needs score_mode prefix|row")
    assert len(rows) == len(cand)
    for r, c in zip(rows, cand):
        assert c.dim() == 2 and c.shape[0] == len(r.ids), (tuple(c.shape), len(r.ids))
    fwd = _forward_prefix if cfg.score_mode is ScoreMode.prefix else _forward_packed
    out: list[torch.Tensor] = [torch.empty(0)] * len(rows)
    for u in units(rows, e.k, cfg):
        sub = [rows[j] for j in u]
        with e.armed(len(sub[0].prompt_ids)):
            got = fwd(e.net, e, sub, [cand[j] for j in u])
        for j, g in zip(u, got):
            out[j] = g
    return out


def seq_lp(decoder, adapter: Adapter, rows: list[TokenRow], bp: BackpropCfg) -> torch.Tensor:
    cand = [torch.tensor(r.ids, dtype=torch.long)[:, None] for r in rows]
    lps = logprobs(decoder, adapter, rows, cand, bp)
    return torch.stack([lp.sum() for lp in lps])
