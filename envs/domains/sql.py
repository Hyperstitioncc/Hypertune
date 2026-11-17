from __future__ import annotations

import functools
import glob
import json
import os
import random
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from envs.domains import BuildCfg, DomainEnv, RawCfg, Rows, main


@dataclass(frozen=True)
class SqlCfg:
    sql_data: Path


def _schema(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return "\n".join(row[0] for row in c.execute("select sql from sqlite_master where type='table' and sql not null"))
    finally:
        c.close()


_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


def _query(db: str, sql: str, seconds: float) -> tuple:
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        c.text_factory = lambda b: b.decode(errors="ignore")
        t0 = time.time()
        c.set_progress_handler(lambda: 1 if time.time() - t0 > seconds else 0, 100_000)
        return tuple(c.execute(sql).fetchall())
    finally:
        c.close()


@functools.lru_cache(maxsize=200_000)
def sql_rows(db: str, sql: str, seconds: float = 30.0) -> tuple | None:
    try:
        return _query(db, sql, seconds)
    except Exception:
        return None


def grade(text: str, gold: dict[str, Any]) -> bool:
    q = text.strip().split("```")[0].split("\n\n")[0].split(";")[0].strip()
    if not q:
        return False
    got, want = sql_rows(gold["db"], q), sql_rows(gold["db"], gold["sql"])
    if got is None or want is None:
        return False
    return got == want if _ORDER_BY.search(gold["sql"]) else Counter(got) == Counter(want)


def make(cfg: SqlCfg) -> DomainEnv:
    return DomainEnv("sql", cfg.sql_data, grade)


def _sql_item(x, db, i, tag, schemas):
    if db not in schemas:
        schemas[db] = _schema(db) if os.path.exists(db) else None
    if schemas[db] is None or len(schemas[db]) > 12_000:
        return None
    hint = f"Hint: {x['evidence']}\n" if x.get("evidence") else ""
    return {"id": f"sql-{tag}-{i}", "prompt": (f"Database schema:\n{schemas[db]}\n\nQuestion: {x['question']}\n{hint}"
                                               "Write one SQLite query that answers the question.\nSQL:"),
            "target": " " + " ".join(x["SQL"].split()), "gold": {"db": db, "sql": x["SQL"]}}


def build(cfg: SqlCfg, b: BuildCfg, r: random.Random, raw: RawCfg) -> tuple[dict[str, Rows], str]:
    RAW = str(raw.raw)
    schemas = {}
    tdb = {os.path.basename(p)[:-7]: p for p in glob.glob(f"{RAW}/bird/train/**/*.sqlite", recursive=True)}
    with open(f"{RAW}/bird/train_filtered.jsonl") as f:
        rows = [json.loads(line) for line in f]
    pool = [_sql_item(x, tdb[x["db_id"]], i, "train", schemas) for i, x in enumerate(rows) if x["db_id"] in tdb]
    n_all = len(rows)
    pool = [p for p in pool if p]
    ddir = f"{RAW}/bird/dev/dev_20240627"
    with open(f"{ddir}/dev.json") as f:
        dev = json.load(f)
    test = [_sql_item(x, f"{ddir}/dev_databases/{x['db_id']}/{x['db_id']}.sqlite", x["question_id"], "dev", schemas)
            for x in dev]
    test = [t for t in test if t and sql_rows(t["gold"]["db"], t["gold"]["sql"])]
    r.shuffle(pool); r.shuffle(test)
    keep, skipped = [], 0
    for p_ in pool:
        if len(keep) == b.n_train + b.n_dev:
            break
        if grade(p_["target"], p_["gold"]):
            keep.append(p_)
        else:
            skipped += 1
    return ({"train": keep[:b.n_train], "dev": keep[b.n_train:b.n_train + b.n_dev], "test": test[:b.n_test]},
            f"BIRD: train = birdsql/bird23-train-filtered ({len(pool)} of {n_all} have a schema under 12k chars; "
            f"{skipped} drawn examples skipped because their gold query does not run as written), test = official dev "
            "whose gold query runs and returns rows; the 69 training databases and the 11 dev databases are disjoint")


if __name__ == "__main__":
    main(SqlCfg, make, build, RawCfg)
