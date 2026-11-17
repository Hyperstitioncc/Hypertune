"""Block sizes for vLLM's LoRA kernel inside the MoE expert layers (fused_moe_lora), on top of a tuned table for
the linear layers written by tune_kernels.py. That kernel reads its token-block size from the same tuned-config
folder (vllm/lora/ops/triton_ops/README_TUNING.md); the rest of its tiling it picks itself. Which block size is
fastest is decided by bench.py on the real batch.

    python tuned_configs.py LINEAR_TUNED_DIR OUT_ROOT     writes OUT_ROOT/tuned_m16, tuned_m32, tuned_m128
                                                          (the linear tables copied, the expert block size set)"""
import json
import os
import shutil
import sys

GPU = "NVIDIA_B200"                 # torch.cuda.get_device_name() with spaces and dashes as underscores
OPS = ["FUSED_MOE_LORA_W13_SHRINK", "FUSED_MOE_LORA_W13_EXPAND", "FUSED_MOE_LORA_W2_SHRINK", "FUSED_MOE_LORA_W2_EXPAND"]


def table(cfg: dict) -> dict:       # config[max_loras][num_slices][tokens][k][n][intermediate]; nearest key on every level but num_slices
    return {"1": {s: {"512": {"1": {"1": {"768": cfg}}}} for s in ("1", "2", "3")}}


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    for m in (16, 32, 128):
        d = os.path.join(out, f"tuned_m{m}")
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(src):
            shutil.copy(os.path.join(src, f), d)
        for op in OPS:
            cfg = {"block_m": m, "block_n": 64, "block_k": 32, "num_warps": 4, "num_stages": 3, "group_size_m": 8, "split_k": 1}
            with open(os.path.join(d, f"{GPU}_{op}.json"), "w") as fh:
                json.dump(table(cfg), fh)
        print(d, sorted(os.listdir(d)))


if __name__ == "__main__":
    main()
