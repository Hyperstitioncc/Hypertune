from __future__ import annotations

import argparse
import dataclasses
import enum
import types
import typing
from pathlib import Path
from typing import Any, Sequence

from shared.fs import atomic_json


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _bool(s: str) -> bool:
    v = s.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {s!r}")


def _enum(tp: type[enum.Enum]):
    def conv(s: str) -> enum.Enum:
        if s not in tp.__members__:
            raise argparse.ArgumentTypeError(f"expected one of {list(tp.__members__)}, got {s!r}")
        return tp[s]
    return conv


def _add(p: argparse.ArgumentParser, f: dataclasses.Field, tp: Any) -> None:
    kw: dict[str, Any] = {"dest": f.name}
    if f.default is not dataclasses.MISSING:
        kw["default"] = f.default
    elif f.default_factory is not dataclasses.MISSING:
        kw["default"] = f.default_factory()
    else:
        kw["required"] = True

    origin = typing.get_origin(tp)
    if origin in (types.UnionType, typing.Union):
        arms = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(arms) != 1 or f.default is not None:
            raise TypeError(f"{f.name}: only X | None = None is supported")
        tp, origin = arms[0], None
    if origin is tuple:
        tp, *rest = typing.get_args(tp)
        if rest != [Ellipsis]:
            raise TypeError(f"{f.name}: only tuple[X, ...] is supported")
        kw["nargs"] = "*"
    elif origin is not None:
        raise TypeError(f"{f.name}: unsupported field type {tp!r}")
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        kw.update(type=_enum(tp), metavar="{" + ",".join(tp.__members__) + "}")
    elif tp is bool:
        kw.update(type=_bool, metavar="BOOL")
    elif tp in (int, float, str, Path):
        kw.update(type=tp)
    else:
        raise TypeError(f"{f.name}: unsupported field type {tp!r}")
    p.add_argument(_flag(f.name), **kw)


def parse(*classes: type, argv: Sequence[str] | None = None) -> tuple:
    return _parse(classes, argv, known=False)[0]


def parse_known(*classes: type, argv: Sequence[str] | None = None) -> tuple[tuple, list[str]]:
    return _parse(classes, argv, known=True)


def _parse(classes: Sequence[type], argv: Sequence[str] | None, known: bool) -> tuple[tuple, list[str]]:
    p = argparse.ArgumentParser(add_help=not known, allow_abbrev=False)
    owner: dict[str, type] = {}
    for cls in classes:
        if not (dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen):
            raise TypeError(f"{cls.__name__} must be a frozen dataclass")
        hints = typing.get_type_hints(cls)
        g = p.add_argument_group(cls.__name__)
        for f in dataclasses.fields(cls):
            if f.name in owner:
                raise ValueError(f"{_flag(f.name)} is declared by both {owner[f.name].__name__} and {cls.__name__}")
            owner[f.name] = cls
            _add(g, f, hints[f.name])
    if known:
        ns_, rest = p.parse_known_args(argv)
    else:
        ns_, rest = p.parse_args(argv), []
    ns = vars(ns_)
    out = []
    for cls in classes:
        hints = typing.get_type_hints(cls)
        kw = {}
        for f in dataclasses.fields(cls):
            v = ns[f.name]
            kw[f.name] = tuple(v) if typing.get_origin(hints[f.name]) is tuple else v
        out.append(cls(**kw))
    return tuple(out), list(rest)


def plain(v: Any) -> Any:
    if isinstance(v, enum.Enum):
        return v.name
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (tuple, list)):
        return [plain(x) for x in v]
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if dataclasses.is_dataclass(v):
        return plain(dataclasses.asdict(v))
    return v


def median(xs: Sequence[float]) -> float:
    return sorted(xs)[len(xs) // 2]


def dump(cfgs: Sequence[Any], path: str | Path) -> None:
    atomic_json({type(c).__name__: plain(c) for c in cfgs}, path)
