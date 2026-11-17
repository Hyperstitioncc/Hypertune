"""Campaign watchdog for tools/curves.py: one pass over every graph job and every sweep child, every minute.

Alerts (one line each, appended to ROOT/alerts.log and printed; each alert fires once):
  DONE / FAIL / TIMEOUT / OOM / PREEMPT ...  a job reached a terminal state (sweep children: non-COMPLETED only)
  ERROR    a log of a live job shows a Python traceback, CUDA OOM, NCCL error or a non-finite loss
  HUNG     a live job made no progress for too long (training: metrics.jsonl; evals: their output dir)
  SLOW     a live training job will not reach its last step within its time limit at its current pace
  QUEUE    a job eligible to run has waited long; HELD jobs whose release condition is met are released
  DISK     the disk holding $CURVES_ROOT is nearly full
Actions (with --act): a HUNG sweep child is cancelled (its driver resubmits it and the run resumes); a graph job lost
to preemption or a node failure is resubmitted through the graph (which replaces only such steps);
held alone-eval jobs are released once both stack builds have started.
    python tools/watchdog.py [--act] [--loop SECONDS] [--table] [--prime]   (--prime: mark what is already there seen)"""
from __future__ import annotations

import calendar
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["TZ"] = "America/Los_Angeles"  # alerts are stamped in Pacific time
time.tzset()

ROOT = Path(os.environ.get("CURVES_ROOT") or sys.exit("set $CURVES_ROOT (the campaign directory of tools/curves.py)"))
USER = os.environ.get("USER") or __import__("getpass").getuser()
VENV = os.path.expandvars(os.environ.get("CURVES_VENV", "$HOME/venv"))
LOG, OUT, SWEEPS = ROOT / "logs", ROOT / "runs", ROOT / "sweeps"
ALERTS, SEEN = ROOT / "alerts.log", ROOT / "watchdog_seen.json"
LOAD_MAX = 15 * 60  # model + engine load before the first logged step
STEP_MAX = 20 * 60  # between metrics lines (a step, or a step plus its probe)
EVAL_MAX = 25 * 60  # between output files of a curve eval
QUEUE_MAX = 30 * 60
BAD = re.compile(r"Traceback \(most recent call last\)|OutOfMemoryError|CUDA error|NCCL error|"
                 r"illegal memory access|Segmentation fault|non-finite|loss is nan", re.I)
RESUBMIT = {"PREEMPTED", "NODE_FAIL", "BOOT_FAIL"}  # the job never got to fail on its own
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL",
            "DEADLINE"}


def sh(*cmd: str) -> str:  # Slurm tools in UTC whatever this process's TZ: sacct times are parsed as UTC
    return subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "TZ": "UTC"}).stdout


def jobs() -> dict[str, tuple[str, str]]:
    """job id -> (name, kind) for graph jobs and sweep children; kind is graph or child."""
    out: dict[str, tuple[str, str]] = {}
    for f in ROOT.glob("jobs_*.json"):
        out.update({j: (n, "graph") for n, j in json.load(open(f)).items()})
    for f in SWEEPS.glob("*.json"):
        if f.name.endswith(".scores.json"):
            continue
        for n, ids in json.load(open(f)).items():
            out.update({j: (n, "child") for j in ids})
    return out


def sacct(ids: list[str]) -> dict[str, dict]:
    rows = {}
    for k in range(0, len(ids), 400):
        for ln in sh("sacct", "-X", "-n", "-P", "-o", "JobID,State,Start,Elapsed,TimelimitRaw,Reason,Eligible",
                     "-j", ",".join(ids[k:k + 400])).splitlines():
            j, st, start, el, lim, reason, eligible = ln.split("|")
            rows[j] = {"state": st.split()[0], "start": start, "elapsed": el, "limit": int(lim or 0) * 60,
                       "reason": reason, "eligible": eligible}
    return rows


def secs(hms: str) -> int:
    d, _, rest = hms.rpartition("-")
    parts = [int(x) for x in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(d or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def run_dir(name: str) -> Path | None:
    """The training run a sweep child writes: T-ARM-STAGE-lrX or T-ARM-STAGE-full (the winner's run)."""
    m = re.match(r"(.+)-(cart|lora8|lora)-(warm|grpo|sft)-(full-lr.+|lr.+|full)$", name)
    if not m:
        return None
    t, arm, stage, point = m.groups()
    point = point.removeprefix("full-")
    key = f"{t}-{arm}-{stage}"
    if point == "full":
        sc = SWEEPS / f"{key}.scores.json"
        if not sc.exists():
            return None
        point = "lr" + json.load(open(sc))["winner"]
    return OUT / t / f"{arm}_{stage}_{point}"


def eval_dir(name: str) -> Path | None:
    m = re.match(r"(.+)-(cart|lora8|lora)-(alone|stack)\d$", name)
    return OUT / "curves" / m.group(3) / m.group(2) / m.group(1) if m else None


def newest(d: Path) -> float:
    try:
        return max((p.stat().st_mtime for p in d.rglob("*") if p.is_file()), default=0.0)
    except FileNotFoundError:
        return 0.0


def tail(path: Path, n: int = 400_000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - n))
            return f.read().decode("utf-8", "replace").replace("\r", "\n")
    except OSError:
        return ""


def steps(d: Path) -> tuple[int, list[float], float]:
    """(last step, recent step seconds, mtime) of a run's metrics.jsonl."""
    p = d / "metrics.jsonl"
    if not p.exists():
        return 0, [], 0.0
    rows = [json.loads(x) for x in p.read_text().splitlines()[-6:] if x.strip()]
    return (rows[-1]["step"] if rows else 0), [r["sec"] for r in rows if "sec" in r], p.stat().st_mtime


def total_steps(d: Path) -> int | None:
    try:
        c = json.load(open(d / "config.json"))
    except (OSError, ValueError):
        return None
    tc = c.get("TrainCfg", {})
    stop = tc.get("stop_at") or 0
    return min(tc["steps"], stop) if stop and "steps" in tc else tc.get("steps")


def check(act: bool, prime: bool = False) -> tuple[list[str], list[dict]]:
    seen = json.load(open(SEEN)) if SEEN.exists() else {}
    known = jobs()
    acct = sacct(list(known))
    live_reason = dict(ln.split(" ", 1) for ln in sh("squeue", "-u", USER, "-h", "-t", "PD",
                                                       "-o", "%i %r").splitlines() if " " in ln)
    now = time.time()
    alerts, table = [], []

    def alert(key: str, line: str) -> None:
        if seen.get(key):
            return
        seen[key] = now
        alerts.append(time.strftime("%H:%M ", time.localtime(now)) + line)

    for j, (name, kind) in known.items():
        a = acct.get(j)
        if not a:
            continue
        st, logf = a["state"], LOG / f"{name}_{j}.log"
        if st in TERMINAL:
            if st != "COMPLETED" or kind == "graph":
                alert(f"term/{j}", f"{'DONE' if st == 'COMPLETED' else st} {name} ({j})"
                      + ("" if st == "COMPLETED" else f" log {logf}"))
            continue
        if st == "PENDING":
            try:  # from when it became eligible (deps met, hold released), not from submission
                wait = now - calendar.timegm(time.strptime(a["eligible"], "%Y-%m-%dT%H:%M:%S"))  # sacct is in UTC
            except ValueError:
                continue
            reason = live_reason.get(j, "")  # sacct's Reason is stale for pending jobs; squeue's is live
            if reason and reason not in ("Dependency", "JobHeldUser") and wait > QUEUE_MAX:
                alert(f"queue/{j}/{int(wait // QUEUE_MAX)}", f"QUEUE {name} ({j}) waiting {wait / 60:.0f} min: {reason}")
            continue
        if st != "RUNNING":
            continue
        el = secs(a["elapsed"])
        started = now - el
        text = tail(logf)
        bad = BAD.search(text)
        row = {"name": name, "job": j, "elapsed_min": round(el / 60, 1)}
        if kind == "child":
            d = run_dir(name)
            step, recent, mt = steps(d) if d else (0, [], 0.0)
            tot = total_steps(d) if d else None
            row.update(step=step, total=tot, s_per_step=round(sorted(recent)[len(recent) // 2], 1) if recent else None)
            fresh = mt if mt > started else 0.0
            hung = (now - started > LOAD_MAX) if not fresh else (now - fresh > STEP_MAX)
            if bad:
                alert(f"err/{j}", f"ERROR {name} ({j}): {bad.group(0)} in {logf}")
            if hung:  # a crashed run whose engine kept it alive lands here too, once its last step goes stale
                why = "no step since start" if not fresh else f"no step for {(now - fresh) / 60:.0f} min"
                alert(f"hung/{j}", f"HUNG {name} ({j}) {why}" + (" -> cancelled, driver resumes it" if act else ""))
                if act:
                    sh("scancel", j)
            elif recent and tot and a["limit"]:
                left = (tot - step) * sorted(recent)[len(recent) // 2]
                if el + left > a["limit"]:
                    alert(f"slow/{j}", f"SLOW {name} ({j}) step {step}/{tot} at {row['s_per_step']} s/step: "
                          f"needs ~{(el + left) / 60:.0f} min of a {a['limit'] / 60:.0f} min limit (resumes after)")
        else:
            d = eval_dir(name)
            if d is not None:
                mt = newest(d)
                fresh = mt if mt > started else 0.0
                row["last_output_min"] = round((now - fresh) / 60, 1) if fresh else None
                if (not fresh and now - started > EVAL_MAX) or (fresh and now - fresh > EVAL_MAX):
                    alert(f"hung/{j}", f"HUNG {name} ({j}) no eval output for {EVAL_MAX // 60}+ min")
            if bad and d is not None:
                alert(f"err/{j}", f"ERROR {name} ({j}): {bad.group(0)} in {logf}")
        table.append(row)

    lost = [j for j, (n, k) in known.items() if k == "graph" and acct.get(j, {}).get("state") in RESUBMIT
            and not seen.get(f"resub/{j}")]
    if lost and act:  # the graph resubmits exactly the steps whose job ended this way (runs and evals resume)
        lib = Path(os.environ.get("CURVES_LIB", ROOT / "lib"))
        r = subprocess.run([f"{VENV}/bin/python", f"{lib}/experiments/tools/curves.py",
                            "submit"], cwd=lib, capture_output=True, text=True, env={**os.environ, "TZ": "UTC"})
        new = [ln for ln in r.stdout.splitlines() if "kept" not in ln]
        for j in lost:
            seen[f"resub/{j}"] = now
        alert(f"resubmit/{now}", f"RESUBMIT {len(lost)} lost graph job(s) ({' '.join(lost)}): "
              f"{'; '.join(new) or r.stderr.strip()[-200:]}")

    held = ROOT / "held_alone.txt"
    if held.exists():
        g = {n: j for j, (n, k) in known.items() if k == "graph"}
        stk = [acct.get(g.get(f"stacks-{a}", ""), {}).get("state") for a in ("cart", "lora")]
        if all(s and s != "PENDING" for s in stk):
            ids = held.read_text().split()
            if act:
                sh("scontrol", "release", *ids)
                held.rename(held.with_suffix(".released"))
            alert("released", f"RELEASED {len(ids)} held alone-eval jobs (stack builds started)")

    use = shutil.disk_usage(ROOT)
    if use.used / use.total > 0.9:
        alert(f"disk/{int(now // 3600)}", f"DISK {ROOT} {100 * use.used / use.total:.0f}% full")

    tmp = SEEN.with_suffix(".tmp")
    json.dump(seen, open(tmp, "w"))
    tmp.replace(SEEN)
    if alerts and not prime:
        with open(ALERTS, "a") as f:
            f.write("".join(a + "\n" for a in alerts))
    return alerts, table


def main(argv: list[str]) -> None:
    act, table, prime = "--act" in argv, "--table" in argv, "--prime" in argv
    if prime:
        check(False, prime=True)
        return
    loop = int(argv[argv.index("--loop") + 1]) if "--loop" in argv else 0
    while True:
        try:
            alerts, rows = check(act)
            for a in alerts:
                print(a, flush=True)
            if table:
                for r in sorted(rows, key=lambda r: r["name"]):
                    print(json.dumps(r), flush=True)
        except Exception as e:  # the watchdog must outlive any one bad read
            print(time.strftime("%H:%M ") + f"WATCHDOG error: {e!r}", flush=True)
        if not loop:
            return
        time.sleep(loop)


if __name__ == "__main__":
    main(sys.argv[1:])
