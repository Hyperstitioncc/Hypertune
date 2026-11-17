import pytest

from adapters import AdapterCfg, build
from conftest import as_decoder, tiny_model
from shared.adapter import AdapterKind
from shared.rollout import RolloutCfg, sampler
from shared.types import Backend


def test_hf_sampler_needs_eager():
    dec = as_decoder(tiny_model())
    a = build(AdapterCfg(adapter=AdapterKind.kv_prefix, k=4, init_seed=3), dec)
    with pytest.raises(ValueError, match="eager"):
        sampler(a, Backend.hf_eager, dec, RolloutCfg())
