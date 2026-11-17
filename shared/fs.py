from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class Roots:
    data: Path
    out: Path

    def run(self, name: str) -> Path:
        return self.out / name


def atomic_write(path: str | os.PathLike, write: Callable[[str], None]) -> None:
    path = os.fspath(path)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        write(tmp)
        _fsync(tmp)
        os.replace(tmp, path)
        _fsync(d or ".")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _fsync(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_json(obj: Any, path: str | os.PathLike) -> None:
    def w(tmp: str) -> None:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1)
    atomic_write(path, w)


def atomic_torch(obj: Any, path: str | os.PathLike) -> None:
    import torch
    atomic_write(path, lambda tmp: torch.save(obj, tmp))


def read_json(path: str | os.PathLike) -> Any:
    with open(path) as f:
        return json.load(f)
