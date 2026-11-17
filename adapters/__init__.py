from __future__ import annotations

from dataclasses import dataclass

import torch

from adapters.kv_prefix import NDIM, KVRead, random_rows
from shared.adapter import Adapter, AdapterKind
from shared.decoder import Decoder, head_dim


@dataclass(frozen=True)
class AdapterCfg:
    adapter: AdapterKind = AdapterKind.identity
    k: int = 16
    init_scale: float = 1.0
    init_seed: int = 0
    init_path: str | None = None
    row_tail: bool = False
    pos0: bool = False
    lora_rank: int = 1
    lora_alpha: float = 32.0
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    lora_layers: tuple[int, ...] = ()
    lora_merge: bool = False
    kv_units: str = "all"
    kv_remove_rest: bool = False
    kv_fix_path: str | None = None
    kv_graph: bool = False
    kv_read: KVRead = KVRead.softmax
    kv_gate_b: float = -4.0
    kv_partners: str | None = None
    kv_partner_p: float = 0.3
    lora_partners: str | None = None
    lora_partner_p: float = 0.3


def _soft(path: str, dev: torch.device) -> torch.Tensor:
    return torch.load(path, map_location="cpu")["soft"].float().to(dev)


def _init_rows(cfg: AdapterCfg, dec: Decoder) -> torch.Tensor:
    if cfg.init_path is not None:
        s = _soft(cfg.init_path, dec.dev)
        if s.shape != (cfg.k, dec.d):
            raise ValueError(f"{cfg.init_path}: shape {tuple(s.shape)} vs ({cfg.k}, {dec.d})")
        return s
    return random_rows(cfg.k, dec.d, dec.estd, cfg.init_scale, cfg.init_seed, dec.dev)


def _identity(cfg: AdapterCfg, dec: Decoder) -> torch.nn.Module:
    from adapters.identity import Identity
    return Identity()


def _lora(cfg: AdapterCfg, dec: Decoder) -> torch.nn.Module:
    from adapters.lora import LoRA
    mod = LoRA(dec.model, cfg.lora_rank, cfg.lora_alpha, cfg.lora_targets, cfg.lora_layers, cfg.init_seed, cfg.lora_merge)
    if cfg.init_path is not None:
        mod.load_stack(torch.load(cfg.init_path, map_location="cpu"))
    if cfg.lora_partners is not None:
        px = torch.load(cfg.lora_partners, map_location="cpu")
        mod.set_partners(px["lora_A"], px["lora_B"], px["sizes"], cfg.lora_partner_p, px.get("scale"), px.get("names"))
    return mod


def _kv_cart(cfg: AdapterCfg, dec: Decoder) -> torch.nn.Module:
    from adapters.kv_prefix import KVCart
    if cfg.kv_fix_path is None:
        raise ValueError("kv_read dsm starts from the slot K/V of --kv-fix-path (kfix/vfix)")
    if cfg.kv_units != "all" or cfg.kv_remove_rest:
        raise ValueError("kv_read dsm: unit sets not implemented")
    if cfg.kv_graph:
        raise ValueError("kv_read dsm has no CUDA-graph engine path")
    fx = torch.load(cfg.kv_fix_path, map_location="cpu")
    at = dec.model.model.layers[0].self_attn
    return KVCart(fx["kfix"].float().to(dec.dev), fx["vfix"].float().to(dec.dev), dec.model.config.num_attention_heads,
                  cfg.kv_gate_b, at.scaling)


def _kv_prefix(cfg: AdapterCfg, dec: Decoder) -> torch.nn.Module:
    from adapters.kv_prefix import KVPrefix, load_units
    if cfg.kv_read is KVRead.dsm:
        return _kv_cart(cfg, dec)
    c = dec.model.config
    L, Hkv, hd = c.num_hidden_layers, c.num_key_value_heads, head_dim(dec.model)
    base = _init_rows(cfg, dec)
    K = base.shape[0]
    units = load_units(cfg.kv_units, L, Hkv, K)
    if cfg.kv_fix_path is not None:
        fx = torch.load(cfg.kv_fix_path, map_location="cpu")
        if not torch.equal(fx["soft"].float()[:, :NDIM].to(base.device), base[:, :NDIM]):
            raise ValueError(f"{cfg.kv_fix_path} was not made from the base rows")
        if cfg.kv_remove_rest:
            raise ValueError("pure K/V: removed units not implemented")
        kf0, vf0 = fx["kfix"].float(), fx["vfix"].float()
    else:
        kf0 = vf0 = torch.zeros(L, K, Hkv, hd)
    mod = KVPrefix(base.to(torch.bfloat16), L, Hkv, hd, units, cfg.kv_remove_rest,
                   cfg.kv_fix_path is not None, kf0, vf0, cfg.pos0, cfg.kv_graph, cfg.row_tail)
    if cfg.kv_partners is not None:
        px = torch.load(cfg.kv_partners, map_location="cpu")
        mod.set_partners(px["soft"].float(), px["kfix"], px["vfix"], px.get("sizes", [px["kfix"].shape[1]]), cfg.kv_partner_p)
    return mod


def _full(cfg: AdapterCfg, dec: Decoder) -> torch.nn.Module:
    import copy
    from adapters.full import Full
    return Full(copy.deepcopy(dec.model))


_BUILD = {
    AdapterKind.identity: _identity,
    AdapterKind.lora: _lora,
    AdapterKind.kv_prefix: _kv_prefix,
    AdapterKind.full: _full,
}


def build(cfg: AdapterCfg, decoder: Decoder) -> Adapter:
    return Adapter(cfg.adapter, _BUILD[cfg.adapter](cfg, decoder))
