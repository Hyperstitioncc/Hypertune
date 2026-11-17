import pytest
import torch

from adapters.rung import Rung
from adapters.soft_block import SoftBlock
from conftest import as_decoder, tiny_model
from shared.adapter import Adapter, AdapterKind
from shared.attach import attach
from shared.rollout import RolloutCfg, sampler
from shared.types import Backend


def test_rung_on_vllm_raises():
    dec = as_decoder(tiny_model("eager"), attn="eager")
    a = Adapter(AdapterKind.rung, Rung(torch.randn(4, dec.d), 2, False, 2, dec.d, 32))
    with pytest.raises(ValueError, match="vLLM"):
        attach(a, Backend.vllm, dec)
    assert attach(a, Backend.hf_eager, dec).positional


def test_rung_needs_eager():
    dec = as_decoder(tiny_model())
    a = Adapter(AdapterKind.rung, Rung(torch.randn(4, dec.d), 2, False, 2, dec.d, 32))
    with pytest.raises(ValueError, match="eager"):
        attach(a, Backend.hf_eager, dec)


def test_soft_pos0_on_vllm_raises():
    dec = as_decoder(tiny_model())
    a = Adapter(AdapterKind.soft_block, SoftBlock(torch.randn(4, dec.d), torch.zeros(0, dec.d), False, True))
    with pytest.raises(ValueError, match="pos0"):
        attach(a, Backend.vllm, dec)
    assert attach(a, Backend.hf_eager, dec).pos0


def test_hf_sampler_needs_eager():
    dec = as_decoder(tiny_model())
    a = Adapter(AdapterKind.soft_block, SoftBlock(torch.randn(4, dec.d), torch.zeros(0, dec.d), False, False))
    with pytest.raises(ValueError, match="eager"):
        sampler(a, Backend.hf_eager, dec, RolloutCfg())
