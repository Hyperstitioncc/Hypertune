"""Grade the curve evals again from their saved completion texts, with the library's current graders.
    python tools/regrade_curves.py [EVAL_DIR] [--dry] [--tasks=T1,T2]   (EVAL_DIR under runs/, default $CURVES_EVAL_DIR)
Every scored file runs/EVAL_DIR/{alone,stack}/ARM/TASK/NAME.json whose chunk files (NAME_chunks/eval_chunk_*.json)
hold texts (curve_eval --curve-texts true) is graded again: its "grades" become the new grades, the grades given
when it was sampled move to "grades_sampled" (kept from the first regrade on), and "regraded" counts the flips.
No sampling: a grader fix reaches every plotted point in minutes. --dry only counts the flips."""
from __future__ import annotations

import dataclasses
import enum
import json
import sys
import typing
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from curves import EVAL_DIR, OUT  # noqa: E402
from envs import env_from  # noqa: E402
from shared.eval import Bank  # noqa: E402
from shared.fs import atomic_json  # noqa: E402


def build(cls: type, d: dict):
    """A config dataclass from its dump (shared.config.dump)."""
    hints = typing.get_type_hints(cls)
    kw = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v, tp = d[f.name], hints[f.name]
        base = typing.get_origin(tp) or tp
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if base is typing.Union and args:
            base = typing.get_origin(args[0]) or args[0]
        if v is None:
            kw[f.name] = None
        elif isinstance(base, type) and issubclass(base, enum.Enum):
            kw[f.name] = base[v] if v in base.__members__ else base(v)
        elif base is Path:
            kw[f.name] = Path(v)
        elif base is tuple:
            el = typing.get_args(tp)[0] if typing.get_args(tp) else None
            if isinstance(el, type) and issubclass(el, enum.Enum):
                kw[f.name] = tuple(el[x] if x in el.__members__ else el(x) for x in v)
            else:
                kw[f.name] = tuple(v)
        else:
            kw[f.name] = v
    return cls(**kw)


def regrade_dir(d: Path, dry: bool) -> Counter:
    cfg = json.load(open(d / "config.json"))
    name = cfg["CurveCfg"]["env"]
    env_cls, make = env_from(name, [])
    ecfg_d = cfg[env_cls.__name__] if env_cls.__name__ in cfg else next(v for k, v in cfg.items() if k.endswith("Cfg")
                                                                         and k not in ("CurveCfg",))
    env = make(build(env_cls, ecfg_d))
    data = Path(cfg["Roots"]["data"])
    bank = Bank.load(data / cfg["BankCfg"]["bank"] / "bank.json")
    problems = {p.id: p for p in env.problems(bank.split)}
    flips = Counter()
    for res in sorted(d.glob("*.json")):
        if res.name == "config.json":
            continue
        chunks = sorted((d / f"{res.stem}_chunks").glob("eval_chunk_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        recs = [json.load(open(c)) for c in chunks]
        if not recs or any("texts" not in r for r in recs):
            flips["no texts"] += 1
            continue
        pairs = [(problems[i], t) for r in recs for i, ts in r["texts"].items() for t in ts]
        new = env.grade(pairs)
        grades, k = {}, 0
        for r in recs:
            for i, ts in r["texts"].items():
                grades.setdefault(i, []).extend(new[k:k + len(ts)])
                k += len(ts)
        obj = json.load(open(res))
        was = obj.get("grades_sampled", obj["grades"])
        n_flip = sum(a != b for i in grades for a, b in zip(was[i], grades[i]))
        up = sum(a < b for i in grades for a, b in zip(was[i], grades[i]))
        flips["files"] += 1
        flips["flips"] += n_flip
        flips["to_correct"] += up
        if not dry:
            obj["grades_sampled"] = was
            obj["grades"] = grades
            obj["regraded"] = {"flips": n_flip, "to_correct": up}
            atomic_json(obj, res)
    return flips


def main(argv: list[str]) -> None:
    dry = "--dry" in argv
    only = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--tasks=")), None)
    argv = [a for a in argv if a != "--dry" and not a.startswith("--tasks=")]
    root = OUT / (argv[0] if argv else EVAL_DIR)
    total = Counter()
    for cfg in sorted(root.glob("*/*/*/config.json")):
        if only and cfg.parent.name not in only:
            continue
        f = regrade_dir(cfg.parent, dry)
        total.update(f)
        print(f"{cfg.parent.relative_to(root)}: {dict(f)}", flush=True)
    print(f"{'(dry) ' if dry else ''}total {dict(total)}")


if __name__ == "__main__":
    main(sys.argv[1:])
