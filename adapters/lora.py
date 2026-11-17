from __future__ import annotations

import math
import random

import torch
import torch.nn as nn

EXPERT_TARGETS = ("gate_proj", "up_proj", "down_proj", "experts")


def _layer_of(name: str) -> int:
    parts = name.split(".")
    i = parts.index("layers")
    return int(parts[i + 1])


class LoRA(nn.Module):
    def __init__(self, model: nn.Module, rank: int, alpha: float, targets: tuple[str, ...],
                 layers: tuple[int, ...], seed: int, merge: bool):
        super().__init__()
        self.rank, self.alpha, self.scale = rank, alpha, alpha / rank
        self.targets, self.merge = tuple(targets), merge
        g = torch.Generator().manual_seed(seed)
        keep = (lambda n: True) if not layers else (lambda n: _layer_of(n) in layers)
        lin_t = [t for t in targets if t not in EXPERT_TARGETS]
        mods = [(n, m) for n, m in model.named_modules()
                if isinstance(m, nn.Linear) and n.split(".")[-1] in lin_t and keep(n)]
        if not mods:
            raise ValueError(f"no {lin_t} Linear modules found")
        names, As, Bs = [], [], []
        for n, m in mods:
            A = torch.empty(rank, m.in_features)
            nn.init.kaiming_uniform_(A, a=math.sqrt(5), generator=g)
            names.append(n)
            As.append(nn.Parameter(A.to(m.weight.device)))
            Bs.append(nn.Parameter(torch.zeros(m.out_features, rank, device=m.weight.device)))
        self.n_linear = len(mods)
        if any(t in EXPERT_TARGETS for t in targets):
            xm = [(n, m) for n, m in model.named_modules()
                  if hasattr(m, "gate_up_proj") and hasattr(m, "down_proj")
                  and isinstance(getattr(m, "gate_up_proj"), nn.Parameter) and keep(n)]
            if not xm:
                raise ValueError("expert targets but no fused experts modules found")
            for n, m in xm:
                E, I2, d = m.gate_up_proj.shape
                I = I2 // 2
                if m.down_proj.shape != (E, d, I) or getattr(m, "is_transposed", False) or getattr(m, "has_bias", False):
                    raise ValueError(f"{n}: unsupported fused experts layout {tuple(m.down_proj.shape)}")
                for proj, fin, fout in (("gate_proj", d, I), ("up_proj", d, I), ("down_proj", I, d)):
                    A = (torch.rand(E, rank, fin, generator=g) * 2 - 1) / math.sqrt(fin)
                    names.append(f"{n}.{proj}")
                    As.append(nn.Parameter(A.to(m.gate_up_proj.device)))
                    Bs.append(nn.Parameter(torch.zeros(E, fout, rank, device=m.gate_up_proj.device)))
        self.names = names
        self.A = nn.ParameterList(As)
        self.B = nn.ParameterList(Bs)
        self.pA: list[torch.Tensor] = []
        self.pB: list[torch.Tensor] = []
        self.spans: list[tuple[int, int]] = []
        self.partner_p = 0.0
        self.present: int | None = None
        self.write_acc: list | None = None

    @property
    def partnered(self) -> bool:
        return bool(self.pA)

    def check(self, scale: float | None, names: list[str] | None, what: str) -> None:
        if scale is not None and not math.isclose(float(scale), self.scale, rel_tol=1e-6):
            raise ValueError(f"{what} was trained at LoRA scale alpha/r = {float(scale):g}, this LoRA's is {self.scale:g} "
                             f"(--lora-alpha {float(scale) * self.rank:g} at --lora-rank {self.rank})")
        if names is not None and list(names) != self.names:
            diff = sorted(set(names) ^ set(self.names))[:4]
            raise ValueError(f"{what} adapts other modules than this LoRA (same --lora-targets and --lora-layers): {diff}")

    def get_extra_state(self) -> dict:
        return {"scale": self.scale, "names": list(self.names)}

    def set_extra_state(self, st: dict) -> None:
        self.check(st.get("scale"), st.get("names"), "LoRA checkpoint")

    def _load_from_state_dict(self, sd, prefix, *a, **kw):
        sd.setdefault(prefix + "_extra_state", self.get_extra_state())  # older checkpoints hold no scale/names
        super()._load_from_state_dict(sd, prefix, *a, **kw)

    def set_partners(self, As: list[torch.Tensor], Bs: list[torch.Tensor], sizes: list[int], p: float,
                     scale: float | None = None, names: list[str] | None = None) -> None:
        if self.merge:
            raise ValueError("partner LoRAs go beside an unmerged LoRA (--lora-merge false)")
        self.check(scale, names, "partner LoRA file")
        if len(As) != len(self.A) or any(a.shape[:-2] + a.shape[-1:] != o.shape[:-2] + o.shape[-1:]
                                         or b.shape[:-1] != q.shape[:-1] or a.shape[-2] != sum(sizes)
                                         for a, b, o, q in zip(As, Bs, self.A, self.B)):
            raise ValueError("partner LoRA file does not match this LoRA's modules (same targets, layers and model)")
        self.pA = [a.detach().to(o.device, o.dtype) for a, o in zip(As, self.A)]
        self.pB = [b.detach().to(q.device, q.dtype) for b, q in zip(Bs, self.B)]
        ends = [sum(sizes[:i + 1]) for i in range(len(sizes))]
        self.spans, self.partner_p = [(e - n, e) for e, n in zip(ends, sizes)], p

    def drawn(self) -> list[tuple[int, int]]:
        if self.present is None:
            return []
        rng = random.Random(f"partners/{self.present}")
        return [s for s in self.spans if rng.random() < self.partner_p]

    def eff(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        keep = self.drawn()
        if not keep:
            return self.A[i], self.B[i]
        return (torch.cat([self.A[i]] + [self.pA[i][..., a:b, :] for a, b in keep], -2),
                torch.cat([self.B[i]] + [self.pB[i][..., a:b] for a, b in keep], -1))

    def load_stack(self, st: dict) -> None:
        self.check(st.get("scale"), st.get("names"), "stacked LoRA")
        with torch.no_grad():
            for o, q, a, b in zip(self.A, self.B, st["lora_A"], st["lora_B"], strict=True):
                if a.shape != o.shape or b.shape != q.shape:
                    raise ValueError(f"stacked LoRA {tuple(a.shape)} into --lora-rank {self.rank}: rank = sum of the parts' ranks")
                o.copy_(a)
                q.copy_(b)


def lora_state(part: dict) -> dict:
    if part.get("adapter", {}).get("kind") != "lora":
        raise ValueError("only LoRA checkpoints stack with LoRAs")
    st = part["adapter"]["state"]
    n = sum(k.startswith("A.") for k in st)
    ex = st.get("_extra_state", {})
    return {"lora_A": [st[f"A.{i}"].float() for i in range(n)], "lora_B": [st[f"B.{i}"].float() for i in range(n)],
            "scale": ex.get("scale"), "names": ex.get("names")}


def stack(parts: list[dict]) -> dict:
    sts = [lora_state(p) for p in parts]
    if len({len(s["lora_A"]) for s in sts}) != 1:
        raise ValueError("LoRAs with different targets or layers")
    scales = [float(s["scale"]) for s in sts if s["scale"] is not None]
    if any(not math.isclose(x, scales[0], rel_tol=1e-6) for x in scales):
        raise ValueError(f"LoRAs trained at different scales alpha/r {scales}: their sum would weigh them unequally")
    names = [list(s["names"]) for s in sts if s["names"] is not None]
    if any(n != names[0] for n in names):
        raise ValueError("LoRAs over different modules (--lora-targets / --lora-layers differ)")
    out = {"lora_A": [torch.cat(x, -2) for x in zip(*(s["lora_A"] for s in sts))],
           "lora_B": [torch.cat(x, -1) for x in zip(*(s["lora_B"] for s in sts))],
           "sizes": [s["lora_A"][0].shape[-2] for s in sts]}
    return {**out, "scale": scales[0] if scales else None, "names": names[0] if names else None}
