import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from shared.decoder import Decoder, DecoderCfg  # noqa: E402

V = 97


def tiny_cfg(attn="sdpa", layers=2):
    from transformers import Qwen3MoeConfig
    return Qwen3MoeConfig(vocab_size=V, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                          num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                          num_experts=8, num_experts_per_tok=2, max_position_embeddings=1024,
                          attn_implementation=attn)


def tiny_model(attn="sdpa", seed=0):
    from transformers import Qwen3MoeForCausalLM
    torch.manual_seed(seed)
    m = Qwen3MoeForCausalLM(tiny_cfg(attn)).float().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    m.config._attn_implementation = attn
    return m


def as_decoder(m, attn="sdpa", tok=None):
    emb, head = m.get_input_embeddings(), m.get_output_embeddings()
    return Decoder(cfg=DecoderCfg(model="tiny", dtype="fp32", attn=attn, device="cpu"), tok=tok, model=m, emb=emb,
                   head=head, dtype=torch.float32, dev=emb.weight.device, hdev=head.weight.device,
                   d=emb.weight.shape[1], estd=emb.weight.float().std().item())


@pytest.fixture
def dec():
    return as_decoder(tiny_model())


REAL_TOK = "Qwen/Qwen3-30B-A3B-Base"


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    """Tiny random Qwen3-MoE with the real tokenizer and vocab, sharded safetensors + index."""
    import json
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM
    out = tmp_path_factory.mktemp("tiny")
    real = json.load(open(hf_hub_download(REAL_TOK, "config.json")))
    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(vocab_size=real["vocab_size"], hidden_size=64, intermediate_size=128,
                         moe_intermediate_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, num_experts=8, num_experts_per_tok=2, max_position_embeddings=4096,
                         rope_theta=real.get("rope_theta", 1e6), tie_word_embeddings=False,
                         bos_token_id=real.get("bos_token_id"), eos_token_id=real.get("eos_token_id"))
    Qwen3MoeForCausalLM(cfg).to(torch.bfloat16).save_pretrained(out, max_shard_size="8MB")
    AutoTokenizer.from_pretrained(REAL_TOK).save_pretrained(out)
    return str(out)
