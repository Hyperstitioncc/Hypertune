from __future__ import annotations

import dataclasses
import typing
from typing import Callable, Sequence

from shared.config import parse_known

from envs import contest, fillers, knowledge, math, mbpp, mix, rg
from envs.domains import bf, bf2, chem, filler, finqa, formula, funcall, intent, medqa, parse, sql

ENVS: dict[str, tuple[type, Callable]] = {
    "math": (math.MathCfg, math.make),
    "rg": (rg.RgCfg, rg.make),
    "contest": (contest.ContestCfg, contest.make),
    "knowledge": (knowledge.KnowledgeCfg, knowledge.make),
    "fillers": (fillers.FillersCfg, fillers.make),
    "mbpp": (mbpp.MbppCfg, mbpp.make),
    "medqa": (medqa.MedqaCfg, medqa.make),
    "funcall": (funcall.FuncallCfg, funcall.make),
    "finqa": (finqa.FinqaCfg, finqa.make),
    "chem": (chem.ChemCfg, chem.make),
    "sql": (sql.SqlCfg, sql.make),
    "bf": (bf.BfCfg, bf.make),
    "bf2": (bf2.Bf2Cfg, bf2.make),
    "intent": (intent.IntentCfg, intent.make),
    "parse": (parse.ParseCfg, parse.make),
    "formula": (formula.FormulaCfg, formula.make),
    "filler": (filler.FillerCfg, filler.make),
}


def env_from(name: str, argv: Sequence[str]) -> tuple[type, Callable]:
    if name != "mix":
        return ENVS[name]
    (mc,), _ = parse_known(mix.MixCfg, argv=argv)
    parts = [ENVS[n][0] for n in mc.mix_envs]
    fields = []
    for c in (mix.MixCfg, *parts):
        hints = typing.get_type_hints(c)
        fields += [(f.name, hints[f.name], dataclasses.field(default=f.default, default_factory=f.default_factory,
                                                            kw_only=True)) for f in dataclasses.fields(c)]
    cls = dataclasses.make_dataclass("MixCfgAll", fields, frozen=True, kw_only=True)

    def make(cfg) -> mix.MixEnv:
        pick = lambda c: c(**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(c)})
        return mix.make(pick(mix.MixCfg), [pick(c) for c in parts])
    return cls, make
