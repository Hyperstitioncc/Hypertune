from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import torch

from shared.config import median, plain


class Schedule(enum.Enum):
    const = "const"
    cosine = "cosine"
    cosine_restarts = "cosine_restarts"


@dataclass(frozen=True)
class OptimCfg:
    lr: float = 5e-3
    adam_eps: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.01
    schedule: Schedule = Schedule.cosine
    warmup: int = 0
    sched_steps: int = 150
    max_grad_norm: float = math.inf
    clip_rel: float = math.inf
    clip_warmup: int = 5
    skip_nonfinite: bool = False
    rollback_lr: float = 0.5


class Optim:
    def __init__(self, params: list[torch.Tensor], cfg: OptimCfg):
        self.params = list(params)
        self.cfg = cfg
        self.opt = torch.optim.AdamW(self.params, lr=cfg.lr, eps=cfg.adam_eps,
                                     betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)
        self.t = 0
        self.mult = 1.0
        self.gnorm_hist: list[float] = []

    def lr(self) -> float:
        c = self.cfg
        f = min(1.0, (self.t + 1) / max(c.warmup, 1))
        if c.schedule is Schedule.cosine:
            f *= 0.5 * (1 + math.cos(math.pi * min(1.0, self.t / c.sched_steps)))
        if c.schedule is Schedule.cosine_restarts:
            f *= 0.5 * (1 + math.cos(math.pi * (self.t % c.sched_steps) / c.sched_steps))
        return c.lr * self.mult * f

    def cut_lr(self) -> None:
        self.mult *= self.cfg.rollback_lr

    def restart(self) -> None:
        self.opt.state.clear()
        self.cut_lr()

    def rollback(self, sd: dict) -> None:
        mult = self.mult
        self.load(sd)
        self.mult = mult
        self.cut_lr()

    def zero_grad(self) -> None:
        self.opt.zero_grad(set_to_none=True)

    def step(self) -> dict[str, float]:
        lr = self.lr()
        grads = [p.grad for p in self.params if p.grad is not None]
        gnorm = math.sqrt(sum(float(g.double().pow(2).sum()) for g in grads))
        if gnorm == 0.0:
            self.t += 1
            return {"lr": lr, "grad_norm": 0.0, "clipped": 0.0, "skipped": 1.0}
        if not math.isfinite(gnorm) and self.cfg.skip_nonfinite:
            self.zero_grad()
            self.t += 1
            return {"lr": lr, "grad_norm": gnorm, "clipped": 0.0, "skipped": 1.0}
        bound = self.cfg.max_grad_norm
        if len(self.gnorm_hist) >= self.cfg.clip_warmup:
            bound = min(bound, self.cfg.clip_rel * median(self.gnorm_hist))
        self.gnorm_hist.append(gnorm)
        clipped = gnorm > bound
        if clipped:
            for g in grads:
                g.mul_(bound / gnorm)
        for grp in self.opt.param_groups:
            grp["lr"] = lr
        self.opt.step()
        self.t += 1
        if not all(bool(torch.isfinite(p).all()) for p in self.params):
            raise SystemExit(f"step {self.t}: params went non-finite; refusing to continue, lower lr")
        return {"lr": lr, "grad_norm": gnorm, "clipped": float(clipped), "skipped": 0.0}

    def state_dict(self) -> dict:
        return {"opt": self.opt.state_dict(), "t": self.t, "mult": self.mult,
                "gnorm_hist": list(self.gnorm_hist), "cfg": plain(self.cfg)}

    def load(self, sd: dict) -> None:
        mine = self.state_dict()["cfg"]
        if sd["cfg"] != mine:
            bad = {k: (sd["cfg"].get(k), v) for k, v in mine.items() if sd["cfg"].get(k) != v}
            raise ValueError(f"optimizer state was written under a different OptimCfg (saved, now): {bad}")
        self.opt.load_state_dict(sd["opt"])
        self.t, self.mult = sd["t"], sd["mult"]
        self.gnorm_hist = list(sd["gnorm_hist"])
