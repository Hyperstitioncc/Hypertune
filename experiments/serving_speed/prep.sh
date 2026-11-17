#!/bin/bash
# CPU-only preparation for the serving-speed benchmark on a Slurm cluster: PEFT into the venv, the dense model into
# the HF cache, and a print of the adapter file layout PEFT writes for the fused MoE experts of this transformers.
# usage:  sbatch -p long -c 8 --mem=48G -o LOG prep.sh
set -euo pipefail
R=${CART_ROOT:?a shared folder holding venv/, bin/uv and hf/ (the Hugging Face cache)}; export PATH=$R/venv/bin:$PATH HF_HOME=$R/hf TOKENIZERS_PARALLELISM=false
echo "START prep on $(hostname) $(date -u +%FT%TZ)"
python -c "import peft" 2>/dev/null || $R/bin/uv pip install -p $R/venv/bin/python --no-deps peft
python -c "import peft, transformers, vllm; print('peft', peft.__version__, 'transformers', transformers.__version__, 'vllm', vllm.__version__)"
python - <<'PY'
import tempfile, torch
from safetensors import safe_open
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from peft import LoraConfig, get_peft_model
cfg = Qwen3MoeConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, moe_intermediate_size=16, num_experts=4, num_experts_per_tok=2)
m = Qwen3MoeForCausalLM(cfg)
print("expert parameters:", [(n, tuple(p.shape)) for n, p in m.named_parameters() if "layers.0.mlp" in n])
pm = get_peft_model(m, LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                  target_parameters=["gate_up_proj", "down_proj"], init_lora_weights=False))
d = tempfile.mkdtemp(); pm.save_pretrained(d)
with safe_open(d + "/adapter_model.safetensors", "pt") as f:
    for k in sorted(f.keys()):
        if "layers.0." in k:
            print("PEFT", k, tuple(f.get_tensor(k).shape), f.get_tensor(k).dtype, "zero" if not f.get_tensor(k).any() else "nonzero")
print(open(d + "/adapter_config.json").read()[:1500])
PY
python -c "
from huggingface_hub import snapshot_download
print('dense model at', snapshot_download('Qwen/Qwen3-32B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja']))"
ls -ld /scratch /data/scratch 2>&1 | cut -c1-120; mkdir -p /scratch/$USER/serving_speed && echo "scratch writable: /scratch/$USER/serving_speed"; df -h /scratch | tail -1
echo "END prep $(date -u +%FT%TZ)"
