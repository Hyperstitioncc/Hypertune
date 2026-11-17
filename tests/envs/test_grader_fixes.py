import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs import math as em  # noqa: E402
from envs import mbpp  # noqa: E402
from envs.domains import sql  # noqa: E402


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "d.sqlite"
    c = sqlite3.connect(p)
    c.execute("create table t (a int, b text)")
    c.executemany("insert into t values (?, ?)", [(1, "x"), (2, "x"), (3, "y")])
    c.commit()
    c.close()
    return str(p)


def test_sql_duplicates_and_order(db):
    gold = {"db": db, "sql": "select b from t"}
    assert sql.grade("select b from t where a > 0", gold)
    assert not sql.grade("select distinct b from t", gold)
    assert sql.grade("select b from t order by b desc", gold)
    ordered = {"db": db, "sql": "select a from t order by a desc"}
    assert sql.grade("select a from t order by -a", ordered)
    assert not sql.grade("select a from t order by a", ordered)
    assert not sql.grade("select bogus", gold) and not sql.grade("", gold)


def test_mbpp_pass_fail_timeout_without_preexec():
    assert mbpp.passes("def f(a, b):\n    return a + b\n", ("assert f(1, 2) == 3",), "")
    assert not mbpp.passes("def f(a, b):\n    return a - b\n", ("assert f(1, 2) == 3",), "")
    # exiting or raising before the tests cannot pass them (the crashing statement is skipped, the tests still run)
    assert not mbpp.passes("def f(a, b):\n    return a - b\nimport sys\nsys.exit(0)\n", ("assert f(1, 2) == 3",), "")
    assert not mbpp.passes("raise SystemError\n", ("assert f(1, 2) == 3",), "")
    assert mbpp.passes("", ("import os, sys", "assert sys.flags.no_user_site and sys.flags.safe_path",
                            "assert os.environ.get('PYTHONPATH') is None"), "")
    t = time.time()
    assert not mbpp.passes("while True:\n    pass\n", (), "")
    assert time.time() - t < mbpp.TIMEOUT + 3


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS is enforced on Linux")
def test_mbpp_limits_still_apply():
    assert mbpp.passes("", ("import resource as r", "assert r.getrlimit(r.RLIMIT_AS)[0] == 4 << 30",
                            "assert r.getrlimit(r.RLIMIT_CPU)[0] == 8", "assert r.getrlimit(r.RLIMIT_FSIZE)[0] == 16 << 20"),
                       "")
    assert not mbpp.passes("x = bytearray(6 << 30)\n", ("assert len(x) > 0",), "")


def test_mbpp_threaded_grading():
    good, bad = "def f(a, b):\n    return a + b\n", "def f(a, b):\n    return a * b\n"
    jobs = [(good if i % 2 else bad) + f"# {i}\n" for i in range(64)]
    with ThreadPoolExecutor(32) as ex:
        got = list(ex.map(lambda c: mbpp.passes(c, ("assert f(1, 2) == 3",), ""), jobs))
    assert got == [bool(i % 2) for i in range(64)]


DYING = em._SYMPY_WORKER_SRC.replace("for line in sys.stdin:\n    p, t = json.loads(line)\n",
                                     "for line in sys.stdin:\n    p, t = json.loads(line)\n"
                                     "    if p == 'die':\n        import os; os._exit(9)\n")


def test_sympy_worker_restarts_after_death(monkeypatch, capsys):
    assert DYING != em._SYMPY_WORKER_SRC
    monkeypatch.setattr(em, "_SYMPY_WORKER_SRC", DYING)
    w = em._SympyWorker()
    assert w("1/2", "0.5") is True
    w.p.kill()
    w.p.wait()
    assert w("1/2", "0.5") is True
    with pytest.raises(em.SympyWorkerDied):
        w("die", "1")
    assert w("2", "2") is True
    assert capsys.readouterr().err.count("sympy worker died") == 1
    w._stop()


def test_math_equal_grades_a_worker_killer_wrong(monkeypatch):
    monkeypatch.setattr(em, "_SYMPY_WORKER_SRC", DYING)
    monkeypatch.setattr(em, "_SYMPY", em._SympyWorker())
    em._sympy_verdict.cache_clear()
    assert em.math_equal("die", "1") is False
    assert em.math_equal("2/4", "\\frac{1}{2}") is True
    em._SYMPY._stop()
    em._sympy_verdict.cache_clear()
