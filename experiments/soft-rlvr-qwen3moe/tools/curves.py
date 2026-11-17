"""Learning curves for every task, cartridge and LoRA, alone and co-resident, through the shared library (B200, Slurm).
The co-residence recipe of RESULTS_SFT.md throughout: six filler partners (trained here, same recipe) present with
probability 0.3, and the write penalty. Seven RL domains: warm start (cloned base successes) then GRPO. The six SFT
domains of RESULTS_SFT.md: SFT on the gold targets. Snapshots along the way are scored alone, and inside the stack of all thirteen taken at the same fraction
of training; every other domain is held out from every cartridge (no domain ever trains beside another).
    python tools/curves.py submit         submit the whole graph (each step skips itself when done; afterok chains)
    python tools/curves.py chain T ARM    one line per snapshot of task T: file under OUT, examples seen
    python tools/curves.py stacks ARM     join the stacks of all tasks at each progress point
    python tools/curves.py sweep T ARM STAGE   bracket the LR of a stage (warm, grpo, sft), finish the winner
Settings (environment):
    CURVES_ROOT        required: a directory every node sees; data, runs, logs and job scripts go under it
    CURVES_LIB         the library checkout the jobs run from (default $CURVES_ROOT/lib)
    CURVES_KNOWLEDGE   required for the knowledge tasks: a directory with law/casehold/medqa/trivia.jsonl
    CURVES_SFT_DATA    the SFT domains (DIR/<domain>), filler tasks (DIR/filler-<task>) and off-task pool (DIR/neg/<source>),
                       as runs_sft.sh builds them under DATA/sft (default $CURVES_ROOT/sft_data)
    CURVES_VENV        the Python environment the jobs use (default $HOME/venv)
    HF_HUB_CACHE, HF_DATASETS_CACHE   passed on to the jobs when set (they run offline)
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path



def setting(name: str, default: str | None = None, what: str = "") -> str:
    v = os.environ.get(name, default)
    if v is None:
        raise SystemExit(f"set ${name}: {what}")
    return v


ROOT = Path(setting("CURVES_ROOT", what="a directory every node sees, for data, runs, logs and job scripts"))
LIB = Path(setting("CURVES_LIB", str(ROOT / "lib")))
DATA, OUT, LOG = ROOT / "data", ROOT / "runs", ROOT / "logs"
SFT_DATA = Path(setting("CURVES_SFT_DATA", str(ROOT / "sft_data")))
VENV = setting("CURVES_VENV", "$HOME/venv")
HF_CACHES = "".join(f"{k}={os.environ[k]} " for k in ("HF_HUB_CACHE", "HF_DATASETS_CACHE") if os.environ.get(k))
X = "experiments/soft-rlvr-qwen3moe"
FILLERS = ("family_relationships", "letter_counting", "number_sorting", "sentence_reordering", "syllogism", "word_sorting")
NEG = ("agnews", "alpaca", "dolly", "gsm8k", "imdb", "mbpp", "squad", "wikitext")
PART = "short,long,overcap"
PROGRESS = 9
HOURS = 1  # every job fits the short partition (1 h, no GPU cap); runs resume if one is cut short
RL_BANK = "--cal-n 16 --n-chunks 2 --n-per-chunk 8"  # 16 rollouts per problem, calibration included
SFT_DP = 2
UNIT = "--unit-tokens 2048"  # HF eager attention is quadratic in the packed unit; 8192 OOMs on long FinQA / funcall rows
DRIVER_H = 8  # sweep drivers hold no GPU; they wait on their children
DRIVER_P = "overcap"  # no GPU cap applies to a CPU job there (long caps nodes per user); a requeued driver resumes from disk
EVAL_SPLIT = {"rl": 3, "sft": 3, "stack": 2}
SETUP = (f"export PATH={VENV}/bin:$PATH HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 {HF_CACHES}TOKENIZERS_PARALLELISM=false "
         f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH={LIB}; cd {LIB}; "
         "IFS=, read -ra G <<< \"$CUDA_VISIBLE_DEVICES\"; ")

MODEL = "--model Qwen/Qwen3-30B-A3B-Base"
IO = f"--data {DATA} --out {OUT}"
ZS = f"{MODEL} --quant fp8 --tp 2 --max-model-len 5504 --max-tokens 4096"
ZG = f"{MODEL} --quant fp8 --tp 2 --max-model-len 5504 --max-tokens 2048"
KVA = "--adapter kv_prefix --k 16 --row-tail true --pos0 true"
LORA = "--adapter lora --lora-rank 1 --lora-alpha 32 --lora-targets q_proj k_proj v_proj o_proj experts --init-seed 0"
LORA8 = LORA.replace("--lora-rank 1 --lora-alpha 32", "--lora-rank 8 --lora-alpha 256")  # same scale alpha/r = 32
BASELINE = ("contest_zs", "mbpp_nt_zs")  # rank-8 LoRA, GRPO alone (no partners, no penalty): a capacity baseline
PEN = "--irr-pen 1.0 --irr-kind write --irr-mode ratio --irr-n 8 --irr-resp 128"
WARM = ("--algo sft --steps 150 --sched-steps 150 --batch 8 --eval-every 50 --train-pool solved --reward-floor 0 "
        "--sft-max-len 2048 "  # a 4096-token target needs ~220 GB of eager attention activations
        "--early-stop false --adam-eps 1e-16 --snap-every 25")
WARM_BATCH, WARM_STEPS = 8, 150
GR = ("--algo grpo --steps 40 --sched-steps 40 --eval-every 10 --probe0 true --grpo-probe-seed 777 --grpo-probe-fixed true "
      "--adam-eps 1e-16 --grpo-neg-scale 0.5 --grpo-tis true --clip-rel 3 --trip-logratio false --early-stop false "
      "--snap-every 5")
GR_BATCH, GR_G = 16, 8
SFT_BATCH = 16
SFT_LR = {"cart": ("1e-2", 10), "lora": ("1e-4", 3)}
# LR sweeps: short runs on the full run's schedule (--stop-at), three points a half-decade apart, extended a point at a
# time until the best is strictly inside the grid on both sides; the winner then continues to the full length.
WARM_LR = {"cart": "3e-3", "lora": "1e-4", "lora8": "1e-4"}  # grid centres; SFT centres are SFT_LR, GRPO centres are the task's
SWEEP_STOP = {"warm": 40, "grpo": 10}  # SFT stops after one pass
SWEEP_MAX = 7


@dataclass(frozen=True)
class Task:
    name: str
    env: str
    rl: bool
    bank: str = ""
    kv_lr: str = "3e-3"
    lora_lr: str = "1.5e-4"
    cap: int = 0
    irr_skip: tuple[str, ...] = ()
    dec: str = ""

    @property
    def irr(self) -> str:
        return f"irr/{self.name}.json"


def know(d: str) -> str:
    return f"--env knowledge --knowledge-data {DATA}/knowledge/{d}.jsonl --knowledge-prompt zeroshot --knowledge-cap 800"


def sft_task(d: str, cap: int, dec: str = "") -> Task:
    return Task(d, f"--env {d} --{d}-data {SFT_DATA}/{d}", False, cap=cap, dec=dec)


TASKS = (
    Task("contest_zs", f"--env contest --contest-data {DATA}/contest/problems.jsonl --contest-prompt zeroshot --contest-cap 5000",
         True, "--cal-max 1000", "3e-3", "1.5e-4", irr_skip=("gsm8k",)),
    Task("mbpp_nt_zs", "--env mbpp --mbpp-prefill true --mbpp-hide-tests true --mbpp-cap 974", True,
         "--cal-max 974 --cal-keep 400", "3e-3", "3e-5", irr_skip=("mbpp",)),
    Task("logic_zs", "--env rg --rg-tasks knights_knaves zebra_puzzles --rg-levels level1 level2 level3 --rg-per 150 "
         "--rg-seed 7 --rg-prompt zeroshot --rg-cap 800", True, "--cal-max 800 --cal-keep 400", "1e-3", "5e-5",
         irr_skip=("filler-syllogism", "filler-family_relationships")),
    Task("know_law", know("law"), True, "--cal-max 800 --cal-keep 400", "3e-3", "1.5e-4"),
    Task("know_casehold", know("casehold"), True, "--cal-max 800 --cal-keep 400", "3e-3", "1e-4"),
    Task("know_medqa", know("medqa"), True, "--cal-max 800 --cal-keep 400", "1e-3", "5e-5"),
    Task("know_trivia", know("trivia"), True, "--cal-max 800 --cal-keep 400", "3e-3", "1.5e-4"),
    sft_task("bf", 700), sft_task("finqa", 96), sft_task("funcall", 512, "--spell-reserved true"), sft_task("intent", 16),
    sft_task("parse", 128), sft_task("formula", 24),
)
BY = {t.name: t for t in TASKS}


def filler(f: str) -> Task:
    return Task(f"filler-{f}", f"--env filler --filler-task {f} --filler-data {SFT_DATA}/filler-{f}", False, cap=64,
                irr_skip=(f"filler-{f}",))


def hf(t: Task) -> str:
    return (f"{MODEL} --attn eager {t.dec} --backend hf_eager --hf-multi true --hf-chunk 32 --temperature 0 --top-p 1 "
            f"--max-tokens {t.cap}")


def per_pass(t: Task) -> int:
    path = Path(t.env.split("-data ")[1].split()[0]) / "train.jsonl"
    return math.ceil(sum(1 for _ in open(path)) / SFT_BATCH)


def arm_flags(arm: str, init: str | None) -> str:
    if arm in ("lora", "lora8"):
        return LORA if arm == "lora" else LORA8
    return f"{KVA} --init-path {init} --kv-fix-path {init}"


def partners(arm: str) -> str:
    if arm == "cart":
        return f"--kv-partners {DATA}/partners/fill_cart.pt --kv-partner-p 0.3"
    return f"--lora-partners {DATA}/partners/fill_lora.pt --lora-partner-p 0.3"


def coreside(arm: str, t: Task) -> str:
    return "" if arm == "lora8" else f"{partners(arm)} {PEN} --irr-items {t.irr}"


def text_kv(t: Task) -> str:
    kv = f"{DATA}/{t.name}/textkv.pt"
    env = t.env if t.rl else f"{t.env} {t.dec}"
    return (f"for i in $(seq 0 400); do [ -f {kv} ] && break; python {X}/text_kv.py {env} {MODEL} --bank {t.name}/bank "
            f"--kv-out {t.name}/textkv.pt --idx $i {IO} || true; done; ") if not t.rl else \
           (f"[ -f {kv} ] || python {X}/text_kv.py {env} {MODEL} --bank {t.name} --kv-out {t.name}/textkv.pt {IO}; ")


def sft_bank(t: Task) -> str:
    return (f"python -m shared.cli bank {t.env} {hf(t)} --adapter identity --bank-split all --select splits "
            f"--n-chunks 1 --n-per-chunk 1 --bank {t.name}/bank {IO}; ")


def launcher(dp: int) -> str:
    return f"torchrun --standalone --nproc_per_node {dp} -m shared.cli train --dist true" if dp > 1 else \
        "python -m shared.cli train"


def sft_run(t: Task, arm: str, partnered: bool, lr: str | None = None, run: str | None = None, dp: int = 1) -> str:
    lr, passes = lr or SFT_LR[arm][0], SFT_LR[arm][1]
    per = per_pass(t)
    snap = per if arm == "cart" else max(1, per // 3)
    run = run or f"{t.name}/{arm}_sft"
    launch = launcher(dp)
    probe = per * passes if partnered else per  # domains: one probe at the end (the curve evals score the snapshots)
    return (f"{launch} {t.env} {hf(t)} {arm_flags(arm, f'{DATA}/{t.name}/textkv.pt')} "
            f"{partners(arm) + ' ' + UNIT if partnered else ''} {PEN} --irr-seed 0 --irr-items {t.irr} --algo sft "
            f"--sft-corpus {t.env.split('-data ')[1].split()[0]}/corpus.json --sft-eos true --sft-select grade --lr {lr} "
            f"--weight-decay 0 --steps {per * passes} --sched-steps {per * passes} --batch {SFT_BATCH} --batch-order epoch "
            f"--pick-seed 9000 --train-pool solved --eval-every {probe} --early-stop false --probe0 {str(not partnered).lower()} "
            f"--trip-logratio false --reward-floor 0 --snap-every {snap} --bank {t.name}/bank --run {run} {IO}; ")


def chain(t: Task, arm: str) -> list[tuple[str, int]]:
    def snaps(run: str) -> list[tuple[str, int]]:
        d = OUT / run / "snaps"
        return sorted((f"{run}/snaps/{p.name}", int(p.stem)) for p in d.glob("*.pt")) if d.is_dir() else []
    if not t.rl:
        return [(f, s * SFT_BATCH) for f, s in snaps(winner(t, arm, "sft") or "-")]
    w, g = winner(t, arm, "warm"), winner(t, arm, "grpo")
    if not w:
        return []
    import torch
    at = torch.load(OUT / w / "best.pt", map_location="cpu", weights_only=False)["best_step"]  # GRPO starts here
    return [(f, s * WARM_BATCH) for f, s in snaps(w) if s < at] + \
        [(f, at * WARM_BATCH + s * GR_BATCH * GR_G) for f, s in snaps(g or "-")]


def progress_points(c: list[tuple[str, int]]) -> list[int]:
    """The snapshot nearest each of PROGRESS evenly spaced fractions of the examples a run sees."""
    top = c[-1][1]
    return [min(range(len(c)), key=lambda i: abs(c[i][1] - k * top / (PROGRESS - 1))) for k in range(PROGRESS)]


def stacks(arm: str) -> None:
    os.makedirs(DATA / "stacks", exist_ok=True)
    chains = {t.name: chain(t, arm) for t in TASKS}
    short = [n for n, c in chains.items() if len(c) < 2]
    if short:
        raise SystemExit(f"no snapshots yet for {short}")
    for k in range(PROGRESS):
        out = f"stacks/{arm}_p{k}.pt"
        if (DATA / out).exists():
            continue
        parts = [c[progress_points(c)[k]][0] for c in chains.values()]
        subprocess.run([sys.executable, "-m", "shared.cli", "stack", "--stack-parts", *parts, "--stack-out", out,
                        "--data", str(DATA), "--out", str(OUT)], check=True)


def winner(t: Task, arm: str, stage: str) -> str | None:
    p = DATA / t.name / f"{stage}_{arm}.txt"
    return p.read_text().strip() if p.exists() else None


def lr_step(lr: str, d: int) -> str:
    """The next point a half-decade up (d=+1) or down (d=-1) on the 1, 3, 10, 30... ladder."""
    m, e = lr.split("e")
    i = 2 * int(e) + (m == "3") + d
    return f"{3 if i % 2 else 1}e{i // 2}"


def lr_val(lr: str) -> float:
    return float(lr)


def lr_snap(lr: str) -> str:
    """The ladder point nearest lr in log space."""
    x = math.log10(float(lr))
    pts = [f"{m}e{e}" for e in range(math.floor(x) - 1, math.floor(x) + 2) for m in (1, 3)]
    return min(pts, key=lambda p: abs(math.log10(float(p)) - x))


def stage_spec(t: Task, arm: str, stage: str) -> tuple[str, int, int, int]:
    """(grid centre, sweep stop step, GPUs for a sweep run, GPUs for the continuation)."""
    if stage == "sft":
        return SFT_LR[arm][0], per_pass(t), 1, SFT_DP
    if stage == "warm":
        return WARM_LR[arm], SWEEP_STOP["warm"], 1, SFT_DP
    return lr_snap(t.kv_lr if arm == "cart" else t.lora_lr), SWEEP_STOP["grpo"], 3, 3


def stage_cmd(t: Task, arm: str, stage: str, lr: str, dp: int) -> tuple[str, str]:
    run = f"{t.name}/{arm}_{stage}_lr{lr}"
    if stage == "sft":
        return run, sft_run(t, arm, True, lr, run, dp)
    if stage == "warm":
        return run, (f"{launcher(dp)} {t.env} {ZS} --backend hf_eager --attn eager {UNIT} {arm_flags(arm, f'{DATA}/{t.name}/textkv.pt')} "
                     f"{coreside(arm, t)} {WARM} --sft-corpus {t.name}/correct.json --lr {lr} "
                     f"--bank {t.name} --run {run} {IO}; ")
    init = (f"{KVA} --kv-graph true --init-path {DATA}/{t.name}/warm_cart.pt --kv-fix-path {DATA}/{t.name}/warm_cart.pt"
            if arm == "cart" else f"{arm_flags(arm, None)} --init-from {winner(t, arm, 'warm')}")
    return run, (f"python -m shared.cli train {t.env} {ZG} --backend vllm --engine-groups ${{G[1]}},${{G[2]}} {init} "
                 f"{coreside(arm, t)} {GR} --lr {lr} --bank {t.name} --run {run} {IO}; ")


def probe_at(run: str, step: int) -> float | None:
    import torch
    p = OUT / run / "best.pt"
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    got = next((h["probe"] for h in ck["hist"] if h["step"] == step), None)
    if got is None and ck["final"]:  # the run stopped (collapse, dead units) before the sweep's stop step
        return -math.inf
    return got


def bracket(scores: dict[str, float]) -> tuple[str | None, list[str]]:
    """(winner, []) once the best is strictly beaten on both sides, else (None, the points to add). Ties at the top
    count as one plateau; the winner is its middle point."""
    lrs = sorted(scores, key=lr_val)
    top = max(scores.values())
    at = [i for i, lr in enumerate(lrs) if scores[lr] == top]
    add = ([lr_step(lrs[0], -1)] if at[0] == 0 else []) + ([lr_step(lrs[-1], 1)] if at[-1] == len(lrs) - 1 else [])
    return (None, add) if add else (lrs[at[len(at) // 2]], [])


class Children:
    """Child jobs of a sweep driver; a child that ends in anything but COMPLETED is resubmitted (runs resume)."""
    RETRY = 3  # real failures; time limits and preemption resume and are retried up to RESUMES times
    RESUMES = 12
    FAIL = {"FAILED", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE"}

    def __init__(self, key: str) -> None:
        self.path = ROOT / "sweeps" / f"{key}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.jobs = json.load(open(self.path)) if self.path.exists() else {}
        self.ends: dict[str, dict[str, str]] = {}

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        json.dump(self.jobs, open(tmp, "w"), indent=1)
        tmp.replace(self.path)

    def submit(self, name: str, cmd: str, gpus: int) -> None:
        args = ["sbatch", "--parsable", "-p", PART, "-N", "1", "--requeue", "-J", f"cv-{name}", "-o",
                f"{LOG}/{name}_%j.log", "-t", f"{HOURS}:00:00", "-c", str(max(4, 12 * gpus)), "--mem",
                f"{max(32, 120 * gpus)}G", f"--gres=gpu:{gpus}"]
        script = ROOT / "jobs" / f"{name}.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(f"#!/bin/bash\nset -eo pipefail\n{SETUP}\n{cmd}\n")
        jid = subprocess.run(args + [str(script)], check=True, capture_output=True, text=True).stdout.strip()
        self.jobs.setdefault(name, []).append(jid)
        self.save()
        print(f"submit {name} {jid}", flush=True)

    def wait(self, want: dict[str, tuple[str, int]], done) -> None:
        """Run every child in want (name -> (cmd, gpus)) until done(name) holds."""
        import time
        while self.poll(want, done):
            time.sleep(30)

    def cancel(self, name: str) -> None:
        if name in self.jobs:
            subprocess.run(["scancel", self.jobs[name][-1]])

    def poll(self, want: dict[str, tuple[str, int]], done) -> list[str]:
        """One pass: (re)submit every child in want that is neither live nor done; the names still to finish."""
        if True:
            todo = [n for n in want if not done(n)]
            if not todo:
                return []
            live = {self.jobs[n][-1]: n for n in todo if n in self.jobs}
            st = {}
            if live:
                out = subprocess.run(["sacct", "-X", "-n", "-P", "-o", "JobID,State", "-j", ",".join(live)],
                                     check=True, capture_output=True, text=True).stdout
                st = {j: x.split()[0] for j, x in (ln.split("|") for ln in out.splitlines() if "|" in ln)}
            for n in todo:
                j = self.jobs.get(n, [None])[-1]
                s = st.get(j, "PENDING" if j else None)
                if s in ("PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING", "RESIZING", "SUSPENDED"):
                    continue
                if s == "COMPLETED" and done(n):
                    continue
                if s is not None:
                    self.ends.setdefault(n, {})[j] = s
                ends = self.ends.get(n, {}).values()
                if sum(e in self.FAIL for e in ends) > self.RETRY or len(self.jobs.get(n, [])) > self.RESUMES:
                    raise SystemExit(f"{n}: {len(self.jobs[n])} attempts, last {j} {s}; see {LOG}/{n}_{j}.log")
                self.submit(n, *want[n])
            return todo


def sweep(t: Task, arm: str, stage: str) -> None:
    """Bracket the LR of one stage of one task and arm with short runs, then finish the winner. The current best point
    is continued to full length while the bracket is still open (speculatively), so the winner's run is usually well
    under way when the bracket closes; continuations of points that lose are cancelled."""
    import time
    import torch
    if winner(t, arm, stage):
        return
    centre, stop, g_sweep, g_full = stage_spec(t, arm, stage)
    key = f"{t.name}-{arm}-{stage}"
    kids = Children(key)
    grid = [lr_step(centre, -1), centre, lr_step(centre, 1)]
    scores: dict[str, float] = {}
    spec: list[str] = []

    def point(lr: str) -> str:
        return f"{key}-lr{lr}"

    def full(lr: str) -> str:
        return f"{key}-full-lr{lr}"

    def run(lr: str) -> str:
        return stage_cmd(t, arm, stage, lr, 1)[0]

    def final(lr: str) -> bool:
        p = OUT / run(lr) / "best.pt"
        return p.exists() and torch.load(p, map_location="cpu", weights_only=False)["final"]

    def leader() -> str:
        lrs = sorted(scores, key=lr_val)
        top = [lr for lr in lrs if scores[lr] == max(scores.values())]
        return top[len(top) // 2]

    def fulls() -> dict[str, tuple[str, int]]:
        return {full(lr): (stage_cmd(t, arm, stage, lr, g_full if stage != "grpo" else 1)[1], g_full) for lr in spec}

    full_done = {full(lr): lr for lr in grid}
    while True:  # a restarted driver first replays the extensions it already ran, so its leader is the real one
        for lr in list(grid):
            if lr not in scores and (got := probe_at(run(lr), stop)) is not None:
                scores[lr] = got
        if not all(lr in scores for lr in grid) or bracket(scores)[0] or len(scores) >= SWEEP_MAX:
            break
        grid += [lr for lr in bracket(scores)[1] if lr not in grid]
    while True:
        grid_now = list(grid)
        full_done.update({full(lr): lr for lr in grid_now})
        want = {point(lr): (stage_cmd(t, arm, stage, lr, 1)[1].replace(" --run ", f" --stop-at {stop} --run "), g_sweep)
                for lr in grid_now if lr not in scores}
        for lr in grid_now:
            if lr not in scores and (got := probe_at(run(lr), stop)) is not None:
                scores[lr] = got
                want.pop(point(lr), None)
        for lr in [lr for lr in spec if scores[lr] < max(scores.values())]:  # scores are final: it cannot win now
            kids.cancel(full(lr))
            spec.remove(lr)
            print(f"{stage} {t.name} {arm}: cancelled the continuation of {lr} (no longer the best)", flush=True)
        if scores and max(scores.values()) > -math.inf and leader() not in spec:
            spec.append(leader())
            print(f"{stage} {t.name} {arm}: continuing the current best {leader()} while the bracket is open", flush=True)
        kids.poll({**want, **fulls()},
                  lambda n: final(full_done[n]) if n in full_done else probe_at(run(n.rsplit("-lr", 1)[1]), stop) is not None)
        if all(lr in scores for lr in grid):
            print(f"{stage} {t.name} {arm}: " + "  ".join(f"{lr}={scores[lr]:.4f}" for lr in sorted(scores, key=lr_val)),
                  flush=True)
            if max(scores.values()) == -math.inf:
                raise SystemExit(f"{stage} {t.name} {arm}: every sweep point stopped early: {scores}")
            best, add = bracket(scores)
            if best:
                break
            if len(scores) + len(add) > SWEEP_MAX:  # flat across the whole grid: the plateau's middle, flagged
                best = leader()
                print(f"{stage} {t.name} {arm}: UNBRACKETED (flat over {SWEEP_MAX} points), taking {best}", flush=True)
                break
            grid += [lr for lr in add if lr not in grid]
        time.sleep(30)
    atomic_write(ROOT / "sweeps" / f"{key}.scores.json",
                 json.dumps({"scores": {k: (v if v != -math.inf else None) for k, v in scores.items()}, "winner": best,
                             "stop": stop, "bracketed": bracket(scores)[0] is not None}, indent=1))
    for lr in [lr for lr in spec if lr != best]:
        kids.cancel(full(lr))
        spec.remove(lr)
        print(f"{stage} {t.name} {arm}: cancelled the continuation of {lr} (lost to {best})", flush=True)
    if best not in spec:
        spec.append(best)
    kids.wait({full(best): fulls()[full(best)]}, lambda n: final(best))
    if stage == "warm" and arm == "cart":
        subprocess.run([sys.executable, f"{X}/warm.py", "--warm-runs", run(best), "--warm-out", f"{t.name}/warm_cart.pt",
                        "--data", str(DATA), "--out", str(OUT)], check=True)
    atomic_write(DATA / t.name / f"{stage}_{arm}.txt", run(best))


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


class Graph:
    """Submits each step once. On a resubmit, a step whose job script and dependencies are unchanged keeps its live or
    finished job; a changed step replaces its old job (cancelled if still queued or running), and so do its dependents."""
    KEEP = {"PENDING", "RUNNING", "COMPLETED", "REQUEUED", "CONFIGURING", "COMPLETING"}

    def __init__(self) -> None:
        self.ids: dict[str, str] = {}
        self.old: dict[str, str] = {}
        for f in sorted(ROOT.glob("jobs_*.json"), key=lambda f: f.stat().st_mtime):
            self.old.update(json.load(open(f)))
        self.state: dict[str, str] = {}
        if self.old:
            out = subprocess.run(["sacct", "-X", "-n", "-P", "-o", "JobID,State", "-j", ",".join(self.old.values())],
                                 check=True, capture_output=True, text=True).stdout
            self.state = {j: st.split()[0] for j, st in (ln.split("|") for ln in out.splitlines() if "|" in ln)}
        self.kept: set[str] = set()

    def job(self, name: str, cmd: str, gpus: int, hours: float, deps: list[str] = (), cpus: int = 0, mem: str = "",
            part: str = PART) -> str:
        dep = [self.ids[d] for d in deps if self.ids.get(d) and self.state.get(self.ids[d]) != "COMPLETED"]
        args = ["sbatch", "--parsable", "-p", part, "-N", "1", "--requeue", "-J", f"cv-{name}", "-o", f"{LOG}/{name}_%j.log",
                "-t", f"{int(hours)}:{int(hours % 1 * 60):02d}:00", "-c", str(cpus or max(4, 12 * gpus)),
                "--mem", mem or f"{max(32, 120 * gpus)}G"]
        if gpus:
            args += [f"--gres=gpu:{gpus}"]
        if dep:
            args += [f"--dependency=afterok:{':'.join(dep)}"]
        script = ROOT / "jobs" / f"{name}.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        text = f"#!/bin/bash\nset -eo pipefail\n{SETUP}\n{cmd}\n"
        sig = json.dumps([text, args[:-1] if dep else args])
        sig_path = script.with_suffix(".sig")
        old = self.old.get(name)
        same = (sig_path.read_text() == sig if sig_path.exists() else script.exists() and script.read_text() == text) \
            and all(d in self.kept for d in deps)
        if old and same and self.state.get(old) in self.KEEP:
            self.ids[name] = old
            self.kept.add(name)
            if self.state[old] == "PENDING":
                subprocess.run(["scontrol", "update", f"JobId={old}", f"TimeLimit={args[args.index('-t') + 1]}"], check=True)
            print(f"{name} {old} kept ({self.state[old]})", flush=True)
            return old
        if old and self.state.get(old) in self.KEEP - {"COMPLETED"}:
            subprocess.run(["scancel", old], check=True)
        script.write_text(text)
        jid = subprocess.run(args + [str(script)], check=True, capture_output=True, text=True).stdout.strip()
        sig_path.write_text(sig)
        self.ids[name] = jid
        print(f"{name} {jid}", flush=True)
        return jid


def curve_cmd(t: Task, files: str, out: str, adapter: str) -> str:
    if t.rl:
        flags = f"{t.env} {ZS} --backend vllm --engine-groups ${{G[0]}},${{G[1]}} {adapter} --bank {t.name}"
        n = "--curve-n 8"  # the bottom of the 8-16 range: halves the scoring tail
    else:
        flags = f"{t.env} {hf(t)} {adapter} --bank {t.name}/bank"
        n = "--curve-n 1 --curve-split clean"
    return f"python {X}/curve_eval.py {flags} --curve-files {files} --curve-out {out} {n} {IO}; "


def submit() -> None:
    for d in (DATA, OUT, LOG, DATA / "irr", DATA / "partners", DATA / "knowledge", DATA / "contest"):
        os.makedirs(d, exist_ok=True)
    g = Graph()
    srcs = {**{s: SFT_DATA / "neg" / s for s in NEG}, **{f"filler-{f}": SFT_DATA / f"filler-{f}" for f in FILLERS}}
    items = []
    for t in [*TASKS, *(filler(f) for f in FILLERS)]:
        keep = " ".join(str(p) for s, p in srcs.items() if s not in t.irr_skip)
        items.append(f"[ -f {DATA}/{t.irr} ] || python {X}/tools/irr_items.py {DATA}/{t.irr} {keep}; ")
    know = setting("CURVES_KNOWLEDGE", what="a directory with law/casehold/medqa/trivia.jsonl (the knowledge tasks)")
    know_ = " ".join(f"ln -sfn {know}/{d}.jsonl {DATA}/knowledge/{d}.jsonl;" for d in ("law", "casehold", "medqa", "trivia"))
    g.job("prep", f"cp -n {LIB}/{X}/contest/problems.jsonl {DATA}/contest/; {know_} {''.join(items)}", 0, 0.5, cpus=2, mem="16G")

    fill = {"cart": [], "lora": []}
    for f in FILLERS:
        t = filler(f)
        g.job(f"fill-{f}-bank", sft_bank(t) + text_kv(t), 1, 1, ["prep"])
        for arm in ("cart", "lora"):
            n = f"fill-{f}-{arm}"
            g.job(n, sft_run(t, arm, False), 1, 6, [f"fill-{f}-bank"])
            fill[arm].append(n)
    for arm in ("cart", "lora"):
        parts = " ".join(f"filler-{f}/{arm}_sft" for f in FILLERS)
        g.job(f"partners-{arm}", f"python -m shared.cli stack --stack-parts {parts} --stack-out partners/fill_{arm}.pt {IO}",
              0, 0.5, fill[arm], cpus=2, mem="32G")

    trained = {"cart": [], "lora": []}
    for t in TASKS:
        if not t.rl:
            g.job(f"{t.name}-bank", sft_bank(t) + text_kv(t), 1, HOURS, ["prep"])
            for arm in ("cart", "lora"):
                n = f"{t.name}-{arm}-sft"
                g.job(n, f"python {X}/tools/curves.py sweep {t.name} {arm} sft", 0, DRIVER_H,
                      [f"{t.name}-bank", f"partners-{arm}"], cpus=1, mem="8G", part=DRIVER_P)
                trained[arm].append(n)
            continue
        groups = "--engine-groups ${G[0]},${G[1]} ${G[2]},${G[3]}"
        g.job(f"{t.name}-bank", f"python -m shared.cli bank {t.env} {ZS} --backend vllm {groups} --bank-split all "
              f"{RL_BANK} {t.bank} --bank {t.name} {IO}; " + text_kv(t), 4, HOURS, ["prep"], mem="450G")
        for arm in ("cart", "lora"):
            g.job(f"{t.name}-{arm}-warm", f"python {X}/tools/curves.py sweep {t.name} {arm} warm", 0, DRIVER_H,
                  [f"{t.name}-bank", f"partners-{arm}"], cpus=1, mem="8G", part=DRIVER_P)
            g.job(f"{t.name}-{arm}-grpo", f"python {X}/tools/curves.py sweep {t.name} {arm} grpo", 0, DRIVER_H,
                  [f"{t.name}-{arm}-warm"], cpus=1, mem="8G", part=DRIVER_P)
            trained[arm].append(f"{t.name}-{arm}-grpo")
        if t.name in BASELINE:
            g.job(f"{t.name}-lora8-warm", f"python {X}/tools/curves.py sweep {t.name} lora8 warm", 0, DRIVER_H,
                  [f"{t.name}-bank"], cpus=1, mem="8G", part=DRIVER_P)
            g.job(f"{t.name}-lora8-grpo", f"python {X}/tools/curves.py sweep {t.name} lora8 grpo", 0, DRIVER_H,
                  [f"{t.name}-lora8-warm"], cpus=1, mem="8G", part=DRIVER_P)

    for arm in ("cart", "lora", "lora8"):
        for t in TASKS:
            if arm == "lora8" and t.name not in BASELINE:
                continue
            name = f"{t.name}-{arm}-grpo" if t.rl else f"{t.name}-{arm}-sft"
            init = f"{DATA}/{t.name}/warm_cart.pt" if t.rl else f"{DATA}/{t.name}/textkv.pt"
            ad = arm_flags(arm, init) + (" --kv-graph true" if arm == "cart" and t.rl else "")
            k = EVAL_SPLIT["rl" if t.rl else "sft"]
            for i in range(k):
                files = f"$(python {X}/tools/curves.py chain {t.name} {arm} | cut -d' ' -f1 | awk 'NR%{k}=={i}')"
                g.job(f"{t.name}-{arm}-alone{i}", curve_cmd(t, files, f"curves/alone/{arm}/{t.name}", ad),
                      2 if t.rl else 1, HOURS, [name])
        if arm == "lora8":
            continue
        g.job(f"stacks-{arm}", f"python {X}/tools/curves.py stacks {arm}", 0, 1, trained[arm], cpus=4, mem="64G")
        n = len(TASKS)
        for t in TASKS:
            if arm == "cart":
                ad = (f"--adapter kv_prefix --k {16 * n} --row-tail true --pos0 true --init-path {DATA}/stacks/cart_p0.pt "
                      f"--kv-fix-path {DATA}/stacks/cart_p0.pt" + (" --kv-graph true" if t.rl else ""))
            else:
                ad = (f"--adapter lora --lora-rank {n} --lora-alpha {32 * n} --lora-targets q_proj k_proj v_proj o_proj "
                      f"experts --init-path {DATA}/stacks/lora_p0.pt")
            k = EVAL_SPLIT["stack"]
            for i in range(k):
                files = " ".join(f"stacks/{arm}_p{p}.pt" for p in range(PROGRESS) if p % k == i)
                g.job(f"{t.name}-{arm}-stack{i}", curve_cmd(t, files, f"curves/stack/{arm}/{t.name}", ad),
                      2 if t.rl else 1, HOURS, [f"stacks-{arm}"])
    gone = [j for n, j in g.old.items() if n not in g.ids and g.state.get(j) in Graph.KEEP - {"COMPLETED"}]
    if gone:
        subprocess.run(["scancel", *gone], check=True)
        print(f"cancelled {len(gone)} jobs of steps no longer in the graph", flush=True)
    olds = list(ROOT.glob("jobs_*.json"))
    json.dump(g.ids, open(ROOT / f"jobs_{os.getpid()}.json", "w"), indent=1)
    for f in olds:
        f.rename(f.with_name("old_" + f.name))


def main(argv: list[str]) -> None:
    cmd, *a = argv
    if cmd == "submit":
        submit()
    elif cmd == "chain":
        for f, x in chain(BY[a[0]], a[1]):
            print(f, x)
    elif cmd == "stacks":
        stacks(a[0])
    elif cmd == "sweep":
        sweep(BY[a[0]], a[1], a[2])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
