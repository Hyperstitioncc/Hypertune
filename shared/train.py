from __future__ import annotations

import copy
import enum
import json
import math
import os
import random
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from algorithms.base import Algorithm
from envs.base import Env
from shared import dist
from shared.adapter import Adapter
from shared.config import median, plain
from shared.eval import Bank
from shared.fs import atomic_torch
from shared.optim import Optim


class BatchOrder(enum.Enum):
    sample = "sample"
    epoch = "epoch"


class TrainPool(enum.Enum):
    low = "low"
    solved = "solved"


class RunState(enum.Enum):
    running = "running"
    collapse = "collapse"
    rollbacks = "rollbacks"
    logratio = "logratio"
    reward_floor = "reward_floor"
    patience = "patience"


@dataclass(frozen=True)
class TrainCfg:
    run: str
    steps: int = 150
    batch: int = 16
    eval_every: int = 15
    ckpt_every: int = 1
    snap_every: int = 0
    stop_at: int = 0  # stop after this many steps, on the --steps schedule; rerun without it to continue (LR sweeps)
    pick_seed: int = 9000
    batch_order: BatchOrder = BatchOrder.sample
    train_pool: TrainPool = TrainPool.low
    probe0: bool = False
    init_from: str | None = None
    early_stop: bool = True
    patience: int = 4
    guard_drop: float = math.inf
    guard_min_best: float = 0.05
    reward_floor: float = 0.02
    collapse_steps: int = 4
    trip_logratio: bool = True
    dead_patience: int = 3
    ent_guard: bool = False
    ent_trip: float = 1.5
    ent_trip_abs: float = 0.3
    ent_snap: int = 5
    ent_warmup: int = 5
    ent_max_rb: int = 3
    ent_snap_ok: float = 1.3

    def __post_init__(self):
        for k in ("steps", "batch", "eval_every", "ckpt_every", "patience", "collapse_steps", "dead_patience",
                  "ent_snap"):
            if getattr(self, k) < 1:
                raise ValueError(f"--{k.replace('_', '-')} must be >= 1, got {getattr(self, k)}")


def pick(pool: list[str], step: int, cfg: TrainCfg) -> list[str]:
    if cfg.batch_order is BatchOrder.sample:
        return random.Random(cfg.pick_seed + step).sample(pool, min(cfg.batch, len(pool)))
    ep, k = divmod(step, math.ceil(len(pool) / cfg.batch))
    order = list(pool)
    random.Random(cfg.pick_seed).shuffle(order)
    for e in range(ep + 1):
        random.Random(cfg.pick_seed + e).shuffle(order)
    return order[k * cfg.batch:(k + 1) * cfg.batch]


def _cpu(sd):
    if isinstance(sd, torch.Tensor):
        return sd.detach().cpu().clone()
    if isinstance(sd, dict):
        return {k: _cpu(v) for k, v in sd.items()}
    return copy.deepcopy(sd)


def train(algo: Algorithm, adapter: Adapter, env: Env, optim: Optim, bank: Bank,
          cfg: TrainCfg, run_dir: Path) -> None:
    run_dir = Path(run_dir)
    lead = dist.rank() == 0
    if cfg.batch % dist.world():
        raise ValueError(f"--batch {cfg.batch} does not split over {dist.world()} ranks")
    if optim.cfg.sched_steps < 1:
        raise ValueError(f"--sched-steps must be >= 1, got {optim.cfg.sched_steps}")
    state_path, best_path, log_path = run_dir / "state.pt", run_dir / "best.pt", run_dir / "metrics.jsonl"
    if best_path.exists() and torch.load(best_path, map_location="cpu", weights_only=False)["final"]:
        return
    problems = {p.id: p for p in env.problems(bank.split)}
    pool = bank.pool if cfg.train_pool is TrainPool.low else list(bank.train)
    if hasattr(algo, "trainable"):  # e.g. SFT problems whose every target was dropped
        pool = [i for i in pool if algo.trainable(i)]

    if lead:
        run_dir.mkdir(parents=True, exist_ok=True)

    if cfg.init_from is not None and not state_path.exists():
        out = run_dir.parents[len(Path(cfg.run).parts) - 1]
        adapter.load(torch.load(out / cfg.init_from / "best.pt", map_location="cpu", weights_only=False)["adapter"])

    start, hist, good = 0, [], None
    best = {"probe": -math.inf, "step": -1, "adapter": _cpu(adapter.state_dict())}
    ctr = {"since_best": 0, "neg_kl_evals": 0, "floor_evals": 0, "recent_rew": [], "zero_run": 0,
           "logratio": [], "dead": 0, "nll_hist": [], "n_rollback": 0, "state": RunState.running}
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        if _ident(st["cfg"]) != _ident(plain(cfg)):
            raise ValueError(f"state.pt was written under {st['cfg']}, now {plain(cfg)}: refusing to mix runs")
        adapter.load(st["adapter"])
        optim.load(st["optim"])
        start, best, hist, ctr = st["step"], st["best"], st["hist"], st["ctr"]
        good = st.get("good")
    if lead:
        _event(run_dir, "start", start)
    if lead and log_path.exists():
        lines = log_path.read_text().splitlines()
        keep = [ln for ln, r in zip(lines, map(_row, lines)) if r is not None and r["step"] <= start]
        log_path.write_text("".join(ln + "\n" for ln in keep))

    def save_state(step_done: int) -> None:
        if lead:
            atomic_torch({"step": step_done, "adapter": _cpu(adapter.state_dict()), "optim": optim.state_dict(),
                          "best": best, "hist": hist, "ctr": ctr, "good": good, "cfg": plain(cfg)}, state_path)

    def write_best(final: bool) -> None:
        if lead:
            sd = best["adapter"] if best["step"] >= 0 else _cpu(adapter.state_dict())
            atomic_torch({"adapter": sd, "best_step": best["step"], "best_probe": best["probe"],
                          "hist": hist, "state": ctr["state"].name, "final": final}, best_path)

    def snap(step_done: int) -> None:
        if lead and cfg.snap_every and (step_done % cfg.snap_every == 0 or step_done == cfg.steps):
            path = run_dir / "snaps" / f"{step_done:05d}.pt"
            if not path.exists():
                atomic_torch({"adapter": _cpu(adapter.state_dict()), "step": step_done}, path)

    def select(step_done: int) -> float:
        part = algo.select(adapter, step_done)
        total, n = dist.sum_vec([part.total, part.n])
        return total / n

    def offer(up: float, step_done: int) -> bool:
        if up > best["probe"]:
            best.update(probe=up, step=step_done, adapter=_cpu(adapter.state_dict()))
            write_best(final=False)
            return True
        return False

    if cfg.probe0 and start == 0:
        offer(select(0), 0)
    if start == 0:
        snap(0)

    if cfg.ent_guard and good is None:
        good = (start, _cpu(adapter.state_dict()), _cpu(optim.state_dict()))
    end = min(cfg.steps, cfg.stop_at) if cfg.stop_at else cfg.steps
    last = start
    for step in range(start, end if ctr["state"] is RunState.running else start):
        t_step = time.time()
        batch = dist.shard([problems[i] for i in pick(pool, step, cfg)])
        optim.zero_grad()
        m = dict(algo.step(adapter, batch, step))
        dist.all_reduce_grads(adapter.params())
        m = dist.mean_metrics(m, len(batch))
        if "logratio" in m:
            ctr["logratio"].append(m["logratio"])
        if "n_units" in m:
            ctr["dead"] = ctr["dead"] + 1 if m["n_units"] and m["n_skip"] == m["n_units"] else 0
            if ctr["dead"] >= cfg.dead_patience:
                raise SystemExit(f"step {step}: every unit non-finite for {ctr['dead']} steps; the run is dead")
        if "reward" in m:
            rew = m["reward"]
            ctr["recent_rew"] = (ctr["recent_rew"] + [rew])[-cfg.eval_every:]
            ctr["zero_run"] = ctr["zero_run"] + 1 if rew < cfg.reward_floor else 0
            if ctr["zero_run"] >= cfg.collapse_steps and ctr["state"] is RunState.running:
                ctr["state"] = RunState.collapse

        tripped = False
        if cfg.ent_guard:
            if "nll" not in m:
                raise ValueError("--ent-guard needs an algorithm that reports nll")
            nll, nh = m["nll"], ctr["nll_hist"]
            if math.isfinite(nll) and len(nh) >= cfg.ent_warmup:
                med = median(nh)
                if nll > cfg.ent_trip * med and nll > med + cfg.ent_trip_abs:
                    tripped = True
                    ctr["n_rollback"] += 1
                    adapter.load(good[1])
                    optim.rollback(good[2])
                    m["rollback_to"] = float(good[0])
                    if ctr["n_rollback"] > cfg.ent_max_rb:
                        ctr["state"] = RunState.rollbacks
            if math.isfinite(nll) and not tripped:
                nh.append(nll)
        if tripped:
            optim.zero_grad()
        else:
            if cfg.ent_guard and (step + 1) % cfg.ent_snap == 0:
                nll, nh = m["nll"], ctr["nll_hist"]
                if math.isfinite(nll) and nll <= cfg.ent_snap_ok * (median(nh) if nh else nll):
                    good = (step, _cpu(adapter.state_dict()), _cpu(optim.state_dict()))
            m.update(optim.step())
        snap(step + 1)

        grid = (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps
        if grid or step + 1 == end:
            t_probe = time.time()
            up = select(step + 1)
            m["probe"] = up
            m["sec_probe"] = time.time() - t_probe
            if not grid:  # --stop-at off the eval grid: a score for the sweep; best and counters stay a straight run's
                hist.append({"step": step + 1, "probe": up, "stop_at": True})
            else:
                if offer(up, step + 1):
                    ctr["since_best"] = 0
                else:
                    ctr["since_best"] += 1
                    if best["step"] >= 0 and best["probe"] > cfg.guard_min_best and up < best["probe"] - cfg.guard_drop:
                        adapter.load(best["adapter"])
                        optim.restart()
                        m["guard_rollback_to"] = float(best["step"])
                hist.append({"step": step + 1, "probe": up})
                if ctr["logratio"]:
                    ctr["neg_kl_evals"] = ctr["neg_kl_evals"] + 1 if ctr["logratio"][-1] < 0 else 0
                    ctr["logratio"] = []
                rr = ctr["recent_rew"]
                ctr["floor_evals"] = (ctr["floor_evals"] + 1
                                      if rr and sum(rr) / len(rr) < cfg.reward_floor else 0)
                if cfg.trip_logratio and ctr["neg_kl_evals"] >= 2:
                    ctr["state"] = RunState.logratio
                elif ctr["floor_evals"] >= 2:
                    ctr["state"] = RunState.reward_floor
                elif cfg.early_stop and ctr["since_best"] >= cfg.patience:
                    ctr["state"] = RunState.patience
        m.update(sec=time.time() - t_step, time=time.time())
        if lead:
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step + 1, **m}) + "\n")
        stopped = ctr["state"] is not RunState.running
        last = step + 1
        if (step + 1) % cfg.eval_every == 0 or (step + 1) % cfg.ckpt_every == 0 or stopped or step + 1 == end:
            save_state(step + 1)
        if stopped:
            break
    write_best(final=end == cfg.steps or ctr["state"] is not RunState.running)
    if lead:
        _event(run_dir, "end", last)


def _event(run_dir: Path, kind: str, step: int) -> None:
    """run_dir/events.jsonl: one line per process start / end of training. proc_start is when the process began, so
    time - proc_start on a start event is the model and engine load."""
    try:
        proc_start = os.stat(f"/proc/{os.getpid()}").st_ctime
    except OSError:
        proc_start = None
    row = {"event": kind, "step": step, "time": time.time(), "proc_start": proc_start, "host": socket.gethostname(),
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "world": dist.world()}
    with open(run_dir / "events.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def _row(ln: str) -> dict | None:
    """A metrics.jsonl line, or None for one a kill left half-written."""
    try:
        r = json.loads(ln)
    except ValueError:
        return None
    return r if isinstance(r, dict) and "step" in r else None


def _ident(c: dict) -> dict:
    return {k: v for k, v in c.items() if k != "stop_at"}
