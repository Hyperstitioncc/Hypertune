"""Regression tests from the audit of the MBPP grader (envs/mbpp.py): false negatives it used to give, and ways a
program could make the graded tests pass without the right function."""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs import mbpp  # noqa: E402
from envs.base import Problem  # noqa: E402

ADD = "def f(a, b):\n    return a + b\n"
T = ("assert f(1, 2) == 3", "assert f(2, 2) == 4")


def fresh(code, tests=T, setup=""):
    return mbpp.passes.__wrapped__(code, tuple(tests), setup)


# false negatives

def test_setup_runs_after_the_code():
    # MBPP 367 and 927: test_setup_code builds a tree from the solution's own Node class; run first, it never could
    code = ("class Node:\n    def __init__(self, data):\n        self.data, self.left, self.right = data, None, None\n"
            "def max_height(node):\n    return 0 if node is None else 1 + max(max_height(node.left), "
            "max_height(node.right))\n")
    setup = "root = Node(1)\nroot.left = Node(2)\nroot.left.left = Node(3)"
    assert fresh(code, ("assert max_height(root) == 3",), setup)
    assert not fresh(code, ("assert max_height(root) == 2",), setup)
    assert fresh(ADD, T, "import math")


def test_tests_may_name_a_module_the_code_did_not_import():
    # MBPP tests call sys.getsizeof; with the tests hidden the model cannot know it must import sys
    code = "def tuple_size(t):\n    import sys as s\n    return s.getsizeof(t)\n"
    assert fresh(code, ("assert tuple_size((1, 2)) == sys.getsizeof((1, 2))",))


@pytest.mark.parametrize("tail", [
    "x = int(input('n: '))\nprint(f(x, 1))\n",  # stdin is empty: EOFError
    "if __name__ == '__main__':\n    print(f(int(input()), 2))\n",
    "assert f(1, 1) == 3  # a wrong self-written check\n",
    "print(f(1))  # a demo call that raises\n",
    "import sys\nsys.exit(0)\n",
    "raise SystemExit(1)\n",
])
def test_a_crashing_top_level_does_not_hide_a_right_function(tail, monkeypatch):
    monkeypatch.setattr(mbpp, "SKIP_CRASHING_TOP_LEVEL", True)  # off by default (whole-script grading)
    assert fresh(ADD + tail)
    assert not fresh(ADD.replace("a + b", "a - b") + tail)


def test_crash_skipping_keeps_later_definitions_and_constants(monkeypatch):
    monkeypatch.setattr(mbpp, "SKIP_CRASHING_TOP_LEVEL", True)  # off by default (whole-script grading)
    code = "K = 10\ndef g(x):\n    return x * K\nprint(g())\ndef f(a, b):\n    return g(a + b) // K\n"
    assert fresh(code)
    assert not fresh("def f(a, b):\n    return a + b\nprint(f())\ndef f(a, b):\n    return 0\n")


def test_mbpp_switch_off_skipping(monkeypatch):
    assert not mbpp.SKIP_CRASHING_TOP_LEVEL  # the default: a program whose top level crashes fails, as in MBPP
    monkeypatch.setattr(mbpp, "SKIP_CRASHING_TOP_LEVEL", False)
    assert not fresh(ADD + "x = int(input())\n")
    assert fresh(ADD)


def test_cpu_time_not_wall_time_decides():
    # 48 programs at once that each compute ~1 s: on a few cores the wall time of each is many times that
    burn = "def f(a, b):\n    s = 0\n    for i in range(12_000_000):\n        s += i\n    return a + b\n"
    jobs = [burn + f"# {i}\n" for i in range(48)]
    with ThreadPoolExecutor(48) as ex:
        assert all(ex.map(lambda c: fresh(c), jobs))
    t = time.time()
    assert not fresh("import time\ntime.sleep(1)\n" + ADD.replace("a + b", "a - b"))
    assert time.time() - t < mbpp.WALL
    assert fresh("import time\ntime.sleep(1)\n" + ADD)  # waiting is not computing


def test_a_string_set_order_cannot_decide_a_grade():
    # MBPP 484: this completion (from the bank) passed under hash seed 1 and failed under seed 0; under random seeds it
    # passed 3 times in 16. A pass must now hold under every seed of HASH_SEEDS, and the same text gets the same grade.
    code = ("def remove_matching_tuple(test_list1, test_list2):\n    set1 = set(test_list1)\n"
            "    set1.difference_update(set(test_list2))\n    return list(set1)\n")
    tests = ("assert remove_matching_tuple([('Hello', 'dude'), ('How', 'are'), ('you', '?')], [('Hello', 'dude'), "
             "('How', 'are')]) == [('you', '?')]",
             "assert remove_matching_tuple([('Part', 'of'), ('the', 'journey'), ('is ', 'end')], [('Journey', 'the'), "
             "('is', 'end')]) == [('Part', 'of'), ('the', 'journey'), ('is ', 'end')]",
             "assert remove_matching_tuple([('Its', 'been'), ('a', 'long'), ('day', 'without')], [('a', 'long'), "
             "('my', 'friend')]) == [('Its', 'been'), ('day', 'without')]")
    assert {fresh(code, tests) for _ in range(6)} == {False}
    right = "def remove_matching_tuple(a, b):\n    return [x for x in a if x not in b]\n"
    assert {fresh(right, tests) for _ in range(3)} == {True}
    assert fresh("", ("import sys", "assert sys.flags.hash_randomization == 0 or hash('a') == hash('a')"))


def test_numpy_imports_under_the_limits():
    pytest.importorskip("numpy")
    assert fresh("import numpy as np\ndef f(a, b):\n    return int(np.array([a, b]).sum())\n")


def test_children_are_killed(tmp_path):
    pid_file = tmp_path / "pid"
    code = (f"import os, time\nif os.fork() == 0:\n    open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
            "    time.sleep(120)\n    os._exit(0)\ntime.sleep(0.3)\n" + ADD)
    assert fresh(code)
    pid = int(pid_file.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"child {pid} outlived its grading")


# false positives: a wrong function must not pass

HACKS = {
    "always-equal": "class A:\n    def __eq__(self, o):\n        return True\ndef f(a, b):\n    return A()\n",
    "builtin as __eq__": "class A:\n    __eq__ = id\ndef f(a, b):\n    return A()\n",
    "inside a list": "class A:\n    __eq__ = lambda s, o: True\ndef f(a, b):\n    return [A()]\n",
    "inside a dict": "class A:\n    __eq__ = lambda s, o: True\n    __hash__ = lambda s: 3\ndef f(a, b):\n    return {A(): 1}\n",
    "int subclass": "class A(int):\n    def __eq__(self, o):\n        return True\ndef f(a, b):\n    return A(0)\n",
    "metaclass": "class M(type):\n    def __eq__(c, o):\n        return True\nclass A(metaclass=M):\n    pass\n"
                 "def f(a, b):\n    return A\n",
    "patched Counter": "from collections import Counter\ndef f(a, b):\n    Counter.__eq__ = lambda s, o: True\n"
                       "    return Counter()\n",
    "prints the old sentinel": "print('__mbpp_all_asserts_ran_7f3c__', flush=True)\nimport os\nos._exit(0)\n",
    "exits 0 early": "import os\nos._exit(0)\n" + ADD,
    "reads the tests": "import glob, os, re\nsrc = ''.join(open(p).read() for p in glob.glob('*') if os.path.isfile(p))\n"
                       "ans = {(int(a), int(b)): int(c) for a, b, c in re.findall(r'f\\((\\d+), (\\d+)\\) == (\\d+)', src)}\n"
                       "def f(a, b):\n    return ans.get((a, b), 0)\n",
    "installs a tracer": "import sys\ndef t(fr, ev, arg):\n    return t\nsys.settrace(t)\n"
                         "sys._getframe().f_trace = t\ndef f(a, b):\n    return 0\n",
}


@pytest.mark.parametrize("name", list(HACKS))
def test_hacks_fail(name):
    assert not fresh(HACKS[name], ("assert f(1, 2) == 3", "assert f(2, 2) == 4"))


def test_patched_builtins_and_modules_are_restored_for_the_tests():
    assert not fresh("import builtins\nbuiltins.int = lambda *a: 0\ndef f(a, b):\n    return 0\n",
                     ("assert f(2, 2) == int('4')",))
    assert not fresh("def f(a, b):\n    import builtins\n    builtins.int = lambda *a: 0\n    return 0\n",
                     ("assert f(2, 2) == int('4')",))


def test_vacuous_comparisons_against_patched_modules_fail():
    code = "import sys\ndef tuple_size(t):\n    sys.getsizeof = lambda x: 0\n    return 0\n"
    assert not fresh(code, ("assert tuple_size((1, 2)) == sys.getsizeof((1, 2))",))
    np = pytest.importorskip("numpy")
    assert np
    assert not fresh("import numpy as np\nclass A:\n    __eq__ = lambda s, o: True\n"
                     "def f(a, b):\n    return np.array([A(), A()], dtype=object)\n", ("assert f(1, 2) == [3, 3]",))


def test_plain_returns_still_pass():
    ok = {
        "Counter": ("from collections import Counter\ndef f(s):\n    return Counter(s)\n", "assert f('aab') == {'a': 2, 'b': 1}"),
        "namedtuple": ("from collections import namedtuple\nP = namedtuple('P', 'a b')\ndef f(s):\n    return P(1, 2)\n",
                       "assert f(0) == (1, 2)"),
        "defaultdict": ("from collections import defaultdict\ndef f(s):\n    d = defaultdict(list)\n    d[s].append(1)\n"
                        "    return d\n", "assert f('x') == {'x': [1]}"),
        "dict subclass": ("class D(dict):\n    pass\ndef f(s):\n    return D(a=1)\n", "assert f(0) == {'a': 1}"),
        "plain object": ("class Node:\n    def __init__(self, v):\n        self.v = v\n    def __repr__(self):\n"
                         "        return 'N'\ndef f(s):\n    return None\n", "assert f(Node(1)) == None"),
        "re.Match is None": ("import re\ndef f(s):\n    return re.match('a', s)\n", "assert f('b') == None"),
        "float": ("def f(s):\n    return 0.1 + 0.2\n", "assert f(0) == 0.30000000000000004"),
        "deep recursion": ("import sys\nsys.setrecursionlimit(10000)\ndef f(n):\n    return 0 if n == 0 else 1 + f(n - 1)\n",
                           "assert f(5000) == 5000"),
        "future import": ("from __future__ import annotations\ndef f(s: Missing) -> int:\n    return s\n", "assert f(3) == 3"),
        "shadowed print": ("print = None\ndef f(s):\n    return s\n", "assert f(3) == 3"),
    }
    for name, (code, test) in ok.items():
        assert fresh(code, (test,)), name


# extraction (unchanged behaviour, pinned)

def test_prefill_extraction():
    env = mbpp.make(mbpp.MbppCfg(mbpp_prefill=True))
    truth = {"code_tests": list(T), "setup": ""}
    texts = {
        ADD + "```\nThis adds.\n```python\nprint(f(1, 2))\n```": 1.0,  # only the first block counts
        ADD + "\nExplanation without a closing fence": 0.0,
        ADD: 1.0,  # cut by the stop string before any fence
        "```python\n" + ADD + "```": 1.0,
        "  " + ADD.replace("\n    ", "\n      "): 0.0,  # indented as a whole: IndentationError
        "": 0.0,
    }
    got = env.grade([(Problem("1", "", truth), t) for t in texts])
    assert got == list(texts.values())


# every MBPP reference solution passes (needs the dataset in the local HF cache)

def test_every_reference_solution_passes():
    try:
        from datasets import load_dataset
        ds = load_dataset(mbpp.HUB, "full")
    except Exception as e:  # offline without a cache
        pytest.skip(f"MBPP not available: {e}")
    refs = [ex for sp in ds for ex in ds[sp]]
    with ThreadPoolExecutor(max(4, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 4)) as ex:
        ok = list(ex.map(lambda e: fresh(e["code"], tuple(e["test_list"]), e["test_setup_code"] or ""), refs))
    assert [e["task_id"] for e, o in zip(refs, ok) if not o] == []
