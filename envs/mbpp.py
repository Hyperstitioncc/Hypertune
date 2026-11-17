from __future__ import annotations

import enum
import functools
import json
import os
import random
import re
import secrets
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from envs.base import Problem
from envs.math import STOPS
from envs.splits import check as check_split

HUB = "google-research-datasets/mbpp"
CODE_INSTR = "Write the solution as a Python function in a single ```python code block."
_CODE_RE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)(?:```|$)", re.DOTALL)
# A program may use TIMEOUT seconds of CPU (RLIMIT_CPU), not of wall clock: a loaded node must not change a grade.
# WALL only stops programs that block or sleep instead of computing.
TIMEOUT = 8.0
WALL = 30.0


class MbppPrompt(enum.Enum):
    qwen3moe = "qwen3moe"
    compose = "compose"


def extract_code(text: str) -> str | None:
    m = _CODE_RE.search(text)
    return m.group(1) if m else None


LIMITS = (("RLIMIT_AS", 4 << 30), ("RLIMIT_CPU", int(TIMEOUT)), ("RLIMIT_FSIZE", 16 << 20))
# A pass must hold under each hash seed: the iteration order of a set or dict of strings cannot decide a grade, and the
# same completion always gets the same grade.
HASH_SEEDS = ("0", "1")
# A program whose top level crashes (input() on an empty stdin, a demo call or a self-written assert that fails,
# sys.exit) is run again statement by statement, skipping the ones that raise: the graded tests then see the functions
# it defines. Skipping statements never adds behaviour, so it cannot turn wrong functions right.
SKIP_CRASHING_TOP_LEVEL = False
_TEST_MODULES = ("sys", "math", "cmath", "re", "collections", "itertools", "heapq", "string", "functools", "operator")
_ENV = {"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

# The program runs inside this driver (one interpreter; stdin carries the job, then reads as empty). The tests never
# touch the disk, and the pass token is a per-run nonce written to a private pipe, so a program cannot read the tests
# or print a pass. Before each operand of a test is evaluated, builtins and the modules the tests name are restored,
# and every operand must be plain data (builtin types, or classes that override no comparison/container/number hooks):
# an object whose __eq__ always answers True cannot pass. What remains open needs frame/gc introspection or ctypes.
_DRIVER = r"""
def _drive():
    import ast, builtins, collections, json, operator as op, os, resource, sys, types, __future__
    job = json.loads(sys.stdin.buffer.read())
    nul = os.open(os.devnull, os.O_RDONLY)
    os.dup2(nul, 0)
    os.close(nul)
    for lim, v in job["limits"]:
        try:
            resource.setrlimit(getattr(resource, lim), (v, v))
        except (ValueError, OSError):
            pass
    write, quit_, vfd, nonce = os.write, os._exit, job["vfd"], job["nonce"].encode()
    _type, _id, _iss, _vars, _exec, _eval, _getattr, _zip = type, id, issubclass, vars, exec, eval, getattr, zip
    _meta, _len, _any, _set = type, len, any, set
    settrace, setprofile, dkeys, dvals = sys.settrace, sys.setprofile, dict.keys, dict.values
    path = os.path.abspath("_prog.py")
    fut = 0
    for f in __future__.all_feature_names:
        fut |= getattr(__future__, f).compiler_flag
    OPS = {ast.Eq: op.eq, ast.NotEq: op.ne, ast.Lt: op.lt, ast.LtE: op.le, ast.Gt: op.gt, ast.GtE: op.ge,
           ast.Is: op.is_, ast.IsNot: op.is_not, ast.In: lambda a, b: op.contains(b, a),
           ast.NotIn: lambda a, b: not op.contains(b, a)}
    expr = lambda e: compile(ast.Expression(e), "<test>", "eval", dont_inherit=True)
    try:
        whole = compile(job["code"], path, "exec", dont_inherit=True)
        parts = [compile(ast.Module([s], []), path, "exec", flags=whole.co_flags & fut, dont_inherit=True)
                 for s in ast.parse(job["code"]).body]
        setup = compile(job["setup"], "<setup>", "exec", dont_inherit=True)
        plan, names = [], set()
        for t in job["tests"]:
            tree = ast.parse(t)
            names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            for s in tree.body:
                if isinstance(s, ast.Assert) and isinstance(s.test, ast.Compare):
                    plan.append((0, [expr(e) for e in [s.test.left, *s.test.comparators]],
                                 [OPS[type(o)] for o in s.test.ops]))
                elif isinstance(s, ast.Assert):
                    plan.append((1, [expr(s.test)], []))
                else:
                    plan.append((2, compile(ast.Module([s], []), "<test>", "exec", dont_inherit=True), []))
    except (SyntaxError, ValueError):
        quit_(1)
    mods = {m: __import__(m) for m in job["modules"] if m in names}
    snaps = [(d, dict(d)) for d in [_vars(builtins)] + [_vars(m) for m in mods.values()]]
    HEAP, LEAF = 1 << 9, {int, float, complex, bool, str, bytes, _type(None)}
    SEQ = tuple((b, b.__iter__) for b in (list, tuple, set, frozenset, collections.deque))
    CDESC = (types.WrapperDescriptorType, types.MethodDescriptorType, types.ClassMethodDescriptorType,
             types.GetSetDescriptorType, types.MemberDescriptorType)
    QUIET = frozenset("__module__ __qualname__ __doc__ __dict__ __weakref__ __slots__ __init__ __new__ __repr__ "
                      "__str__ __format__ __annotations__ __annotate__ __firstlineno__ __static_attributes__ "
                      "__match_args__ __getnewargs__ __getnewargs_ex__ __orig_bases__ __parameters__ __type_params__ "
                      "__dataclass_fields__ __dataclass_params__ __init_subclass__ __class_getitem__ __reduce__ "
                      "__reduce_ex__ __getstate__ __setstate__ __copy__ __deepcopy__ __del__".split())
    dunders = lambda c: {n: v for n, v in _vars(c).items() if n[:2] == "__" == n[-2:] and n not in QUIET}
    trusted = {c: dunders(c) for c in (collections.Counter,)}  # Python-level stdlib classes: frozen as they are now

    def cls_ok(k):
        if _type(k) is not _meta:
            return False
        for c in k.__mro__:
            if c in trusted:
                now = dunders(c)
                if _len(now) != _len(trusted[c]) or _any(now.get(n) is not v for n, v in trusted[c].items()):
                    return False
            elif c.__flags__ & HEAP:
                for n, v in dunders(c).items():
                    if not (_type(v) in CDESC and _getattr(v, "__objclass__", None) is c):
                        return False
        return True

    def plain(x):
        todo, seen = [x], _set()
        while todo:
            v = todo.pop()
            t = _type(v)
            if t in LEAF or _id(v) in seen:
                continue
            seen.add(_id(v))
            if not cls_ok(t):
                return False
            if _iss(t, dict):
                todo.extend(dkeys(v))
                todo.extend(dvals(v))
                continue
            for b, it in SEQ:
                if _iss(t, b):
                    todo.extend(it(v))
                    break
            else:
                dt = _getattr(v, "dtype", None) if _getattr(t, "__module__", "") == "numpy" else None
                if dt is not None and dt.hasobject:
                    return False
        return True

    def restore():
        settrace(None)
        setprofile(None)
        for d, snap in snaps:
            for n in [n for n in d if n not in snap]:
                del d[n]
            d.update(snap)
        g["__builtins__"] = builtins
        g.update(mods)

    main = types.ModuleType("__main__")
    g = main.__dict__
    g.update(__file__=path, __builtins__=builtins)
    g0 = dict(g)
    sys.modules["__main__"] = main
    try:
        _exec(whole, g)
    except BaseException:
        if not job["skip_crashing"]:
            quit_(1)
        g.clear()
        g.update(g0)
        for p in parts:
            try:
                _exec(p, g)
            except BaseException:
                pass
    try:
        _exec(setup, g)
        for kind, a, ops in plan:
            if kind == 2:
                restore()
                _exec(a, g)
                continue
            vals = []
            for e in a:
                restore()
                v = _eval(e, g)
                if not plain(v):
                    quit_(1)
                vals.append(v)
            if kind == 1 and not vals[0]:
                quit_(1)
            for f, x, y in _zip(ops, vals, vals[1:]):
                if not f(x, y):
                    quit_(1)
        write(vfd, nonce)
    except BaseException:
        quit_(1)
    quit_(0)


_drive()
"""


@functools.lru_cache(maxsize=200_000)
def passes(code: str, tests: tuple[str, ...], setup: str) -> bool:
    """The code, then the setup, then the tests; True when every test held under every hash seed."""
    return all(_run(code, tests, setup, seed) for seed in HASH_SEEDS)


def _run(code: str, tests: tuple[str, ...], setup: str, seed: str) -> bool:
    nonce = secrets.token_hex(16)
    r, w = os.pipe()
    try:
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "_prog.py"), "w", encoding="utf-8", errors="surrogatepass") as f:
                f.write(code)
            job = json.dumps({"code": code, "setup": setup, "tests": list(tests), "limits": LIMITS, "vfd": w,
                              "nonce": nonce, "modules": _TEST_MODULES, "skip_crashing": SKIP_CRASHING_TOP_LEVEL})
            # -s -P rather than -I: -I implies -E, which would ignore PYTHONHASHSEED; the environment is ours anyway
            p = subprocess.Popen([sys.executable, "-s", "-P", "-c", _DRIVER], cwd=td, stdin=subprocess.PIPE,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, pass_fds=(w,),
                                 start_new_session=True, env={**_ENV, "PYTHONHASHSEED": seed})
            os.close(w)
            w = -1
            try:
                p.communicate(job.encode(), timeout=WALL)
            except subprocess.TimeoutExpired:
                pass
            finally:
                try:  # the program's own children too, finished or not
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                p.kill()
                p.wait()
            os.set_blocking(r, False)
            try:
                got = os.read(r, 1 << 16)
            except BlockingIOError:
                got = b""
            return nonce.encode() in got
    finally:
        os.close(r)
        if w >= 0:
            os.close(w)


def _visible_call(test_list: list[str]) -> str:
    t = test_list[0]
    for op in ("==", "!=", " is ", "<=", ">=", "<", ">"):
        i = t.find(op)
        if i > 0:
            t = t[:i]
            break
    return t.replace("assert ", "").strip()


def _compose_task(ex: dict[str, Any]) -> str:
    return (f"You are an expert Python programmer, and here is your task: {ex['text']} "
            f"Your code will be tested with calls like: {_visible_call(ex['test_list'])}\n[BEGIN]\n")


@dataclass(frozen=True)
class MbppCfg:
    mbpp_cap: int = 320
    mbpp_prefill: bool = False
    mbpp_hide_tests: bool = True  # graded tests never go in the prompt unless asked for explicitly
    mbpp_prompt: MbppPrompt = MbppPrompt.qwen3moe


@dataclass(frozen=True)
class MbppEnv:
    cfg: MbppCfg
    name = "mbpp"
    splits = ("all",)

    @property
    def stops(self) -> tuple[str, ...]:
        return ("[DONE]",) if self.cfg.mbpp_prompt is MbppPrompt.compose else STOPS

    def problems(self, split: str) -> list[Problem]:
        check_split(self.splits, split)
        from datasets import load_dataset
        c = self.cfg
        ds = load_dataset(HUB, "full")
        truth = lambda ex: {"code_tests": list(ex["test_list"]), "setup": ex.get("test_setup_code") or ""}
        if c.mbpp_prompt is MbppPrompt.compose:
            prefix = "".join(_compose_task(s) + f"{s['code']}\n[DONE]\n" for s in list(ds["prompt"])[:3])
            pool = list(ds["train"]) + list(ds["test"]) + list(ds["validation"])
            return [Problem(str(ex["task_id"]), prefix + _compose_task(ex), truth(ex))
                    for ex in random.Random(0).sample(pool, min(c.mbpp_cap, len(pool)))]
        pool = []
        for sp in ("train", "validation", "test", "prompt"):
            for ex in ds[sp]:
                if c.mbpp_hide_tests:
                    defs = re.findall(r"^def\s+(\w+)\s*\((.*?)\)\s*(?:->[^:]*)?:", ex["code"], re.M)
                    called = " ".join(ex["test_list"])
                    tgt = next(((n, a) for n, a in defs if re.search(rf"\b{n}\s*\(", called)), None)
                    if tgt is None:
                        continue
                    prob = ex["text"].strip() + f"\nThe function must be: def {tgt[0]}({tgt[1].strip()})"
                else:
                    prob = ex["text"].strip() + "\nYour code should pass these tests:\n" + "\n".join(ex["test_list"])
                pool.append(Problem(str(ex["task_id"]), f"Problem: {prob}\n{CODE_INSTR}\nSolution:"
                                    + ("\n```python\n" if c.mbpp_prefill else ""), truth(ex)))
        random.Random(0).shuffle(pool)
        return pool[:c.mbpp_cap]

    def _code(self, text: str) -> str | None:
        if self.cfg.mbpp_prompt is MbppPrompt.compose:
            return text.split("[DONE]")[0]
        if self.cfg.mbpp_prefill and not text.lstrip().startswith("```"):
            code = text.split("```")[0]
        else:
            code = extract_code(text)
        return code if code and code.strip() else None

    def _grade(self, text: str, truth: dict[str, Any]) -> bool:
        code = self._code(text)
        return code is not None and passes(code, tuple(truth["code_tests"]), truth["setup"])

    def grade(self, pairs: list[tuple[Problem, str]]) -> list[float]:
        # the cores this process may use, not the machine's: in an 8-core job on a 256-core node, 48 programs at
        # once start too slowly to finish inside TIMEOUT when Python sits on a network disk, and correct ones fail
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 8
        w = min(48, max(4, cores - 4))
        with ThreadPoolExecutor(w) as ex:
            return [float(v) for v in ex.map(lambda pt: self._grade(pt[1], pt[0].truth), pairs)]


def make(cfg: MbppCfg) -> MbppEnv:
    if cfg.mbpp_prompt is MbppPrompt.compose and (cfg.mbpp_prefill or cfg.mbpp_hide_tests):
        raise ValueError("--mbpp-prompt compose has its own format, which shows a graded test's call: it needs "
                         "--mbpp-hide-tests false and no --mbpp-prefill")
    return MbppEnv(cfg)
