from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from algorithms.sft import SFT, LossNorm, build
from envs.base import Env
from shared.types import DecodeCfg
from shared.backprop import BackpropCfg
from shared.eval import Bank
from shared.fs import read_json


@dataclass(frozen=True)
class OffPolicyCfg:
    offpolicy_corpus: str = "correct.json"


def make(cfg: OffPolicyCfg, env: Env, sampler, decoder, bp: BackpropCfg, bank: Bank, decode: DecodeCfg, bank_dir: Path,
         data: Path) -> SFT:
    return build(decoder, bp, env, bank, read_json(Path(bank_dir) / cfg.offpolicy_corpus), LossNorm.row_mean)
