from __future__ import annotations

import dataclasses
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist

from shared.decoder import Decoder, DecoderCfg, load


@dataclass(frozen=True)
class DistCfg:
    dist: bool = False
    dist_timeout_min: int = 120


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def init(cfg: DistCfg) -> torch.device:
    if not cfg.dist:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gloo for CPU tensors, nccl for CUDA ones, so CPU-side reductions (and CPU-only tests) work on GPU hosts too
    dist.init_process_group(backend="cpu:gloo,cuda:nccl" if torch.cuda.is_available() else "gloo",
                            timeout=timedelta(minutes=cfg.dist_timeout_min))
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local = dist.get_node_local_rank()
    torch.cuda.set_device(local)
    return torch.device("cuda", local)


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world() -> int:
    return dist.get_world_size() if is_dist() else 1


def teardown() -> None:
    if is_dist():
        dist.destroy_process_group()


def shard(seq: Sequence) -> list:
    return list(seq[rank()::world()])


def _dev() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")


def sum_scalar(x: float) -> float:
    return sum_vec([x])[0]


def sum_vec(xs: Sequence[float]) -> list[float]:
    if not is_dist():
        return [float(x) for x in xs]
    t = torch.tensor([float(x) for x in xs], dtype=torch.float64, device=_dev())
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def mean_metrics(m: dict[str, float], weight: float = 1.0) -> dict[str, float]:
    """Each key's mean over the ranks that report it, weighted by `weight` (e.g. the rank's number of problems, so a
    rate is the batch's rate); a key whose reporting ranks all weigh 0 gets their plain mean."""
    if not is_dist():
        return m
    keys: list = [None] * world()
    dist.all_gather_object(keys, sorted(m))
    names = sorted(set().union(*keys))
    w = float(weight)
    t = torch.tensor([[w * m[k], w, m[k], 1.0] if k in m else [0.0] * 4 for k in names], dtype=torch.float64,
                     device=_dev())
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return {k: float(vw / sw if sw > 0 else v / n) for k, (vw, sw, v, n) in zip(names, t.tolist())}


def all_reduce_grads(params: Sequence[torch.Tensor]) -> None:
    if not is_dist():
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world())


def replicas(cfg: DecoderCfg, gpus: Sequence[int]) -> list[Decoder]:
    with ThreadPoolExecutor(max(1, len(gpus))) as ex:
        return list(ex.map(lambda g: load(dataclasses.replace(cfg, gpus=(g,), gpu_mem="auto")), gpus))


def score_parallel(decoders: Sequence[Decoder], jobs: Sequence[Any], fn: Callable[[Decoder, Any], Any]) -> list:
    free: queue.Queue = queue.Queue()
    for d in decoders:
        free.put(d)

    def run(job):
        d = free.get()
        try:
            return fn(d, job)
        finally:
            free.put(d)
    with ThreadPoolExecutor(len(decoders)) as ex:
        return list(ex.map(run, jobs))
