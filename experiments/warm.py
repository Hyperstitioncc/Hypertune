from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from shared.adapter import AdapterKind
from shared.config import parse
from shared.fs import Roots, atomic_torch


@dataclass(frozen=True)
class WarmCfg:
    warm_runs: tuple[str, ...]
    warm_out: Path


def main(argv: list[str]) -> None:
    c, roots = parse(WarmCfg, Roots, argv=argv)
    cks = [torch.load(roots.out / r / "best.pt", map_location="cpu", weights_only=False) for r in c.warm_runs]
    best = max(cks, key=lambda ck: ck["best_probe"])["adapter"]
    st = best["state"]
    if AdapterKind(best["kind"]) is not AdapterKind.kv_prefix:
        raise SystemExit(f"warm start: {best['kind']} checkpoint, expected a kv_prefix cartridge")
    m = st["tm"].float()[..., None]
    out = {"soft": st["base"].float(), "kfix": st["kf0"] + st["dk"] * m, "vfix": st["vf0"] + st["dv"] * m}
    atomic_torch(out, roots.data / c.warm_out)


if __name__ == "__main__":
    main(sys.argv[1:])
