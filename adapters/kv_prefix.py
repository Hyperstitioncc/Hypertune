from __future__ import annotations

import enum
import json
import random
import zlib

import torch
import torch.nn as nn

NDIM = 16


class KVRead(enum.Enum):
    softmax = "softmax"
    dsm = "dsm"


def load_units(spec: str, L: int, Hkv: int, K: int) -> torch.Tensor:
    m = torch.zeros(L, K, Hkv, dtype=torch.bool)
    if spec == "all":
        m[:] = True
        return m
    with open(spec) as f:
        units = json.load(f)["units"]
    for u in units:
        l, h, i = u // (Hkv * K), (u // K) % Hkv, u % K
        m[l, i, h] = True
    return m


def full_kv(part: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if "adapter" in part:
        if part["adapter"]["kind"] != "kv_prefix":
            raise ValueError(f"cannot stack a {part['adapter']['kind']} checkpoint with K/V cartridges")
        st = part["adapter"]["state"]
        if not bool(st["kf0"].any()):
            raise ValueError("only pure-K/V cartridges (trained from --kv-fix-path) stack: this one adds deltas to computed K/V")
        m = st["tm"].float()[..., None]
        return st["base"].float(), st["kf0"].float() + st["dk"].float() * m, st["vf0"].float() + st["dv"].float() * m
    return part["soft"].float(), part["kfix"].float(), part["vfix"].float()


def check_rows(rows: torch.Tensor, what: str) -> None:
    r = rows[:, :NDIM].to(torch.bfloat16)
    same = (r[:, None] == r[None]).all(-1) & ~torch.eye(r.shape[0], dtype=torch.bool, device=r.device)
    if bool(same.any()):
        i, j = same.nonzero()[0].tolist()
        raise ValueError(f"{what}: identity rows {i} and {j} are equal in their first {NDIM} dims at bf16, so the engine "
                         "would give both slots' K/V to each (draw the stack again with another --stack-seed)")


def stack_seed(seed: int) -> int:
    return zlib.crc32(f"kv_prefix.stack/{seed}".encode()) | 1 << 32


def stack(parts: list[dict], seed: int) -> dict:
    rows, ks, vs = zip(*(full_kv(p) for p in parts))
    if len({(k.shape[0], *k.shape[2:]) for k in ks}) != 1:
        raise ValueError(f"cartridges for different decoders: {[tuple(k.shape) for k in ks]}")
    k, v = torch.cat(ks, 1), torch.cat(vs, 1)
    g = torch.Generator().manual_seed(stack_seed(seed))
    soft = torch.randn(k.shape[1], rows[0].shape[1], generator=g) * torch.cat(rows).std()
    check_rows(soft, "stack")
    return {"soft": soft, "kfix": k, "vfix": v, "sizes": [x.shape[1] for x in ks]}


class KVPrefix(nn.Module):
    read = KVRead.softmax

    def __init__(self, base: torch.Tensor, L: int, Hkv: int, hd: int, units: torch.Tensor,
                 remove_rest: bool, replace: bool, kf0: torch.Tensor, vf0: torch.Tensor, pos0: bool, graph: bool,
                 row_tail: bool = False):
        super().__init__()
        K = base.shape[0]
        dev = base.device
        self.L, self.Hkv, self.hd, self.K = L, Hkv, hd, K
        self.replace, self.pos0, self.graph, self.row_tail = replace, pos0, graph, row_tail
        if not kf0.shape == (L, K, Hkv, hd) == vf0.shape:
            raise ValueError(f"kv fix shapes {tuple(kf0.shape)} {tuple(vf0.shape)} vs {(L, K, Hkv, hd)}")
        self.register_buffer("base", base.detach().to(torch.bfloat16))
        self.register_buffer("tm", units.to(dev))
        self.register_buffer("rm", (~units if remove_rest else torch.zeros_like(units)).float().to(dev))
        self.register_buffer("kf0", kf0.float().to(dev))
        self.register_buffer("vf0", vf0.float().to(dev))
        self.dk = nn.Parameter(torch.zeros(L, K, Hkv, hd, device=dev))
        self.dv = nn.Parameter(torch.zeros(L, K, Hkv, hd, device=dev))
        self.register_buffer("pbase", None, persistent=False)
        self.register_buffer("pk", None, persistent=False)
        self.register_buffer("pv", None, persistent=False)
        self.spans: list[tuple[int, int]] = []
        self.partner_p = 0.0
        self.present: int | None = None

    def eff(self) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.tm.float()[..., None]
        return self.dk * m, self.dv * m

    def set_partners(self, soft: torch.Tensor, kfix: torch.Tensor, vfix: torch.Tensor, sizes: list[int], p: float) -> None:
        if not self.replace or bool(self.rm.any()):
            raise ValueError("partner cartridges go beside a pure-K/V cartridge (--kv-fix-path, no removed units)")
        if kfix.shape != (self.L, sum(sizes), self.Hkv, self.hd) or vfix.shape != kfix.shape or soft.shape[0] != sum(sizes):
            raise ValueError(f"partner file shapes {tuple(soft.shape)} {tuple(kfix.shape)} do not hold {sizes} slots of this decoder")
        check_rows(torch.cat([self.base.float(), soft.detach().float().to(self.base.device)]), "own + partner rows")
        dev = self.base.device
        self.pbase, self.pk, self.pv = soft.detach().to(dev, torch.bfloat16), kfix.float().to(dev), vfix.float().to(dev)
        ends = [sum(sizes[:i + 1]) for i in range(len(sizes))]
        self.spans, self.partner_p = [(e - n, e) for e, n in zip(ends, sizes)], p

    @property
    def partnered(self) -> bool:
        return self.pbase is not None

    def table(self) -> torch.Tensor:
        return self.base if self.pbase is None else torch.cat([self.base, self.pbase], 0)

    def full(self, li: int) -> tuple[torch.Tensor, torch.Tensor]:
        dk, dv = self.eff()
        k, v = self.kf0[li] + dk[li], self.vf0[li] + dv[li]
        if self.pbase is None:
            return k, v
        return torch.cat([k, self.pk[li]], 0), torch.cat([v, self.pv[li]], 0)

    def rows(self, ids=()) -> torch.Tensor:
        if self.present is None or self.pbase is None:
            return self.base
        rng = random.Random(f"partners/{self.present}/{zlib.crc32(repr(tuple(ids)).encode())}")
        keep = [torch.arange(a, b, device=self.base.device) for a, b in self.spans if rng.random() < self.partner_p]
        return torch.cat([self.base] + [self.pbase[i] for i in keep], 0)


class KVCart(nn.Module):
    read = KVRead.dsm

    def __init__(self, kc0: torch.Tensor, vc0: torch.Tensor, H: int, gate_b: float, scaling: float):
        super().__init__()
        L, K, Hkv, hd = kc0.shape
        if vc0.shape != kc0.shape:
            raise ValueError(f"kfix {tuple(kc0.shape)} vs vfix {tuple(vc0.shape)}")
        self.L, self.K, self.H, self.Hkv, self.hd = L, K, H, Hkv, hd
        self.gb, self.scaling = gate_b, scaling
        self.kc = nn.Parameter(kc0.float().clone())
        self.vc = nn.Parameter(vc0.float().clone())
        self.kg = nn.Parameter(torch.zeros(L, H, hd, device=kc0.device))
