#!/bin/bash
set -euo pipefail
DATA=$(realpath -m "${1:?usage: runs.sh DATA OUT}")
OUT=$(realpath -m "${2:?usage: runs.sh DATA OUT}")
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
X=experiments/soft-rlvr-qwen3moe

QWEN=(--model Qwen/Qwen3-30B-A3B-Base --backend vllm --quant fp8)
ATTN=(--lora-rank 1 --lora-alpha 32 --lora-targets q_proj k_proj v_proj o_proj)
EXPERT=(--lora-rank 1 --lora-alpha 32 --lora-targets q_proj k_proj v_proj o_proj experts)
OFF=(--algo offpolicy --adapter soft_block --init-seed 1234 --lr 0.05 --batch 8 --eval-every 50 --train-pool solved --reward-floor 0 --early-stop false)
SEALED=(--eval-n 64 --eval-chunks 4)

# ---- B.1  Qwen MATH bank (4-shot, 640 cap, n=256 base rates)
MATH=(--env math --math-types number_theory --math-levels level1 level2 level3 level4 level5 --math-cap 600)
python -m shared.cli bank "${MATH[@]}" "${QWEN[@]}" --max-tokens 640 --bank math640 --n-chunks 4 --n-per-chunk 64 --data "$DATA" --out "$OUT"
# B.1 arms table, offpolicy (600 steps, lr 0.05) and its sealed control
python -m shared.cli train "${MATH[@]}" "${QWEN[@]}" "${OFF[@]}" --steps 600 --sched-steps 600 --bank math640 --run math640/offpolicy --data "$DATA" --out "$OUT"
python -m shared.cli eval "${MATH[@]}" "${QWEN[@]}" --adapter soft_block --bank math640 --eval-run math640/offpolicy "${SEALED[@]}" --data "$DATA" --out "$OUT"
python $X/warm.py --warm-runs math640/offpolicy --warm-out math640/warm.pt --data "$DATA" --out "$OUT"
# B.1 controls: nosoft (zero-vector decode path)
python -m shared.cli eval "${MATH[@]}" "${QWEN[@]}" --adapter identity --bank math640 --eval-run math640/nosoft --eval-seed 6200 "${SEALED[@]}" --data "$DATA" --out "$OUT"
# B.1 150-step campaign: soft s0 / s1, lr 1e-2
python -m shared.cli train "${MATH[@]}" "${QWEN[@]}" --adapter soft_block --init-path "$DATA/math640/warm.pt" --lr 1e-2 --grpo-tis true --bank math640 --run math640/soft_s0 --data "$DATA" --out "$OUT"
python -m shared.cli train "${MATH[@]}" "${QWEN[@]}" --adapter soft_block --init-path "$DATA/math640/warm.pt" --lr 1e-2 --grpo-tis true --grpo-seed 1501 --bank math640 --run math640/soft_s1 --data "$DATA" --out "$OUT"
python -m shared.cli eval "${MATH[@]}" "${QWEN[@]}" --adapter soft_block --bank math640 --eval-run math640/soft_s0 --eval-seed 6600 "${SEALED[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${MATH[@]}" "${QWEN[@]}" --adapter soft_block --bank math640 --eval-run math640/soft_s1 --eval-seed 6601 "${SEALED[@]}" --data "$DATA" --out "$OUT"
# B.1 150-step campaign: attention LoRA r1 s0 / s1, lr 3e-4
python -m shared.cli train "${MATH[@]}" "${QWEN[@]}" --adapter lora "${ATTN[@]}" --init-seed 1234 --lr 3e-4 --grpo-tis true --bank math640 --run math640/lora_s0 --data "$DATA" --out "$OUT"
python -m shared.cli train "${MATH[@]}" "${QWEN[@]}" --adapter lora "${ATTN[@]}" --init-seed 1235 --lr 3e-4 --grpo-tis true --grpo-seed 1501 --bank math640 --run math640/lora_s1 --data "$DATA" --out "$OUT"
python -m shared.cli eval "${MATH[@]}" "${QWEN[@]}" --adapter lora "${ATTN[@]}" --bank math640 --eval-run math640/lora_s0 --eval-seed 6800 "${SEALED[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${MATH[@]}" "${QWEN[@]}" --adapter lora "${ATTN[@]}" --bank math640 --eval-run math640/lora_s1 --eval-seed 6801 "${SEALED[@]}" --data "$DATA" --out "$OUT"
# B.1 450-step runs: soft s0 / s1 lr 1e-2, expert LoRA s0 / s1 lr 3e-5 (clip 3x run median)
python -m shared.cli train "${MATH[@]}" "${QWEN[@]}" --adapter soft_block --init-path "$DATA/math640/warm.pt" --lr 1e-2 --steps 450 --sched-steps 450 --grpo-tis true --bank math640 --run math640/soft450_s0 --data "$DATA" --out "$OUT"
python -m shared.cli train "${MATH[@]}" "${QWEN[@]}" --adapter soft_block --init-path "$DATA/math640/warm.pt" --lr 1e-2 --steps 450 --sched-steps 450 --grpo-tis true --grpo-seed 1501 --bank math640 --run math640/soft450_s1 --data "$DATA" --out "$OUT"
python -m shared.cli train "${MATH[@]}" "${QWEN[@]}" --adapter lora "${EXPERT[@]}" --init-seed 1234 --lr 3e-5 --steps 450 --sched-steps 450 --clip-rel 3 --grpo-tis true --bank math640 --run math640/xlora450_s0 --data "$DATA" --out "$OUT"
python -m shared.cli train "${MATH[@]}" "${QWEN[@]}" --adapter lora "${EXPERT[@]}" --init-seed 1235 --lr 3e-5 --steps 450 --sched-steps 450 --clip-rel 3 --grpo-tis true --grpo-seed 1501 --bank math640 --run math640/xlora450_s1 --data "$DATA" --out "$OUT"
python -m shared.cli eval "${MATH[@]}" "${QWEN[@]}" --adapter soft_block --bank math640 --eval-run math640/soft450_s0 --eval-seed 6600 "${SEALED[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${MATH[@]}" "${QWEN[@]}" --adapter soft_block --bank math640 --eval-run math640/soft450_s1 --eval-seed 6601 "${SEALED[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${MATH[@]}" "${QWEN[@]}" --adapter lora "${EXPERT[@]}" --bank math640 --eval-run math640/xlora450_s0 --eval-seed 6800 "${SEALED[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${MATH[@]}" "${QWEN[@]}" --adapter lora "${EXPERT[@]}" --bank math640 --eval-run math640/xlora450_s1 --eval-seed 6801 "${SEALED[@]}" --data "$DATA" --out "$OUT"

# ---- B.4  Qwen contest bank (4-shot, 640 cap)
CT=(--env contest --contest-data "$DATA/contest/problems.jsonl" --contest-cap 5000)
CTG=(--grpo-tis true --clip-rel 3 --trip-logratio false --early-stop false)
python -m shared.cli bank "${CT[@]}" "${QWEN[@]}" --max-tokens 640 --bank contest640 --bank-split all --cal-max 5000 --n-chunks 4 --n-per-chunk 64 --data "$DATA" --out "$OUT"
python -m shared.cli train "${CT[@]}" "${QWEN[@]}" "${OFF[@]}" --steps 300 --sched-steps 300 --bank contest640 --run contest640/offpolicy --data "$DATA" --out "$OUT"
python $X/warm.py --warm-runs contest640/offpolicy --warm-out contest640/warm.pt --data "$DATA" --out "$OUT"
# B.4 no-soft control
python -m shared.cli eval "${CT[@]}" "${QWEN[@]}" --adapter identity --bank contest640 --eval-run contest640/nosoft --eval-seed 6200 "${SEALED[@]}" --data "$DATA" --out "$OUT"
# B.4 table: soft prompt 2e-2 s0 / s1
python -m shared.cli train "${CT[@]}" "${QWEN[@]}" "${CTG[@]}" --adapter soft_block --init-path "$DATA/contest640/warm.pt" --lr 2e-2 --bank contest640 --run contest640/soft_s0 --data "$DATA" --out "$OUT"
python -m shared.cli train "${CT[@]}" "${QWEN[@]}" "${CTG[@]}" --adapter soft_block --init-path "$DATA/contest640/warm.pt" --lr 2e-2 --grpo-seed 1501 --bank contest640 --run contest640/soft_s1 --data "$DATA" --out "$OUT"
python -m shared.cli eval "${CT[@]}" "${QWEN[@]}" --adapter soft_block --bank contest640 --eval-run contest640/soft_s0 --eval-seed 6600 "${SEALED[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${CT[@]}" "${QWEN[@]}" --adapter soft_block --bank contest640 --eval-run contest640/soft_s1 --eval-seed 6601 "${SEALED[@]}" --data "$DATA" --out "$OUT"
# B.4 table: expert LoRA r1 3e-5 s0 / s1
python -m shared.cli train "${CT[@]}" "${QWEN[@]}" "${CTG[@]}" --adapter lora "${EXPERT[@]}" --init-seed 1234 --lr 3e-5 --bank contest640 --run contest640/xlora_s0 --data "$DATA" --out "$OUT"
python -m shared.cli train "${CT[@]}" "${QWEN[@]}" "${CTG[@]}" --adapter lora "${EXPERT[@]}" --init-seed 1235 --lr 3e-5 --grpo-seed 1501 --bank contest640 --run contest640/xlora_s1 --data "$DATA" --out "$OUT"
python -m shared.cli eval "${CT[@]}" "${QWEN[@]}" --adapter lora "${EXPERT[@]}" --bank contest640 --eval-run contest640/xlora_s0 --eval-seed 6800 "${SEALED[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${CT[@]}" "${QWEN[@]}" --adapter lora "${EXPERT[@]}" --bank contest640 --eval-run contest640/xlora_s1 --eval-seed 6801 "${SEALED[@]}" --data "$DATA" --out "$OUT"

# ---- B.6  contest 640, +150 steps on a fresh cosine from the 150-step soft prompts (lr 2e-2)
python $X/warm.py --warm-runs contest640/soft_s0 --warm-out contest640/soft_s0.pt --data "$DATA" --out "$OUT"
python $X/warm.py --warm-runs contest640/soft_s1 --warm-out contest640/soft_s1.pt --data "$DATA" --out "$OUT"
python -m shared.cli train "${CT[@]}" "${QWEN[@]}" "${CTG[@]}" --adapter soft_block --init-path "$DATA/contest640/soft_s0.pt" --lr 2e-2 --bank contest640 --run contest640/soft_s0_cont150 --data "$DATA" --out "$OUT"
python -m shared.cli train "${CT[@]}" "${QWEN[@]}" "${CTG[@]}" --adapter soft_block --init-path "$DATA/contest640/soft_s1.pt" --lr 2e-2 --grpo-seed 1501 --bank contest640 --run contest640/soft_s1_cont150 --data "$DATA" --out "$OUT"
python -m shared.cli eval "${CT[@]}" "${QWEN[@]}" --adapter soft_block --bank contest640 --eval-run contest640/soft_s0_cont150 --eval-seed 6600 "${SEALED[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${CT[@]}" "${QWEN[@]}" --adapter soft_block --bank contest640 --eval-run contest640/soft_s1_cont150 --eval-seed 6601 "${SEALED[@]}" --data "$DATA" --out "$OUT"

# ---- B.5  contest bank re-calibrated at a 2048 cap
C2=(--max-tokens 2048 --max-model-len 3456)
python -m shared.cli bank "${CT[@]}" "${QWEN[@]}" "${C2[@]}" --bank contest2048 --bank-split all --cal-max 5000 --n-chunks 4 --n-per-chunk 64 --data "$DATA" --out "$OUT"
python -m shared.cli train "${CT[@]}" "${QWEN[@]}" "${C2[@]}" "${OFF[@]}" --steps 300 --sched-steps 300 --bank contest2048 --run contest2048/offpolicy --data "$DATA" --out "$OUT"
python $X/warm.py --warm-runs contest2048/offpolicy --warm-out contest2048/warm.pt --data "$DATA" --out "$OUT"
# B.5 table: soft 1e-2 s0 / s1, LoRA 3e-5 s0 / s1
python -m shared.cli train "${CT[@]}" "${QWEN[@]}" "${C2[@]}" "${CTG[@]}" --adapter soft_block --init-path "$DATA/contest2048/warm.pt" --lr 1e-2 --bank contest2048 --run contest2048/soft_s0 --data "$DATA" --out "$OUT"
python -m shared.cli train "${CT[@]}" "${QWEN[@]}" "${C2[@]}" "${CTG[@]}" --adapter soft_block --init-path "$DATA/contest2048/warm.pt" --lr 1e-2 --grpo-seed 1501 --bank contest2048 --run contest2048/soft_s1 --data "$DATA" --out "$OUT"
python -m shared.cli train "${CT[@]}" "${QWEN[@]}" "${C2[@]}" "${CTG[@]}" --adapter lora "${EXPERT[@]}" --init-seed 1234 --lr 3e-5 --bank contest2048 --run contest2048/xlora_s0 --data "$DATA" --out "$OUT"
python -m shared.cli train "${CT[@]}" "${QWEN[@]}" "${C2[@]}" "${CTG[@]}" --adapter lora "${EXPERT[@]}" --init-seed 1235 --lr 3e-5 --grpo-seed 1501 --bank contest2048 --run contest2048/xlora_s1 --data "$DATA" --out "$OUT"
python -m shared.cli eval "${CT[@]}" "${QWEN[@]}" "${C2[@]}" --adapter soft_block --bank contest2048 --eval-run contest2048/soft_s0 --eval-seed 6600 "${SEALED[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${CT[@]}" "${QWEN[@]}" "${C2[@]}" --adapter soft_block --bank contest2048 --eval-run contest2048/soft_s1 --eval-seed 6601 "${SEALED[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${CT[@]}" "${QWEN[@]}" "${C2[@]}" --adapter lora "${EXPERT[@]}" --bank contest2048 --eval-run contest2048/xlora_s0 --eval-seed 6800 "${SEALED[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${CT[@]}" "${QWEN[@]}" "${C2[@]}" --adapter lora "${EXPERT[@]}" --bank contest2048 --eval-run contest2048/xlora_s1 --eval-seed 6801 "${SEALED[@]}" --data "$DATA" --out "$OUT"

# ---- Part R  pure-K/V cartridge vs LoRA, 7 domains (zero-shot, 4096 cap, TP2, n=126 base rates) [H N=16]
ZB=("${QWEN[@]}" --tp 2 --max-model-len 5504)
ZS=("${ZB[@]}" --max-tokens 4096)
ZG=("${ZB[@]}" --max-tokens 2048)
RBANK=(--bank-split all --n-chunks 2 --n-per-chunk 63)
KVA=(--adapter kv_prefix --k 16 --row-tail true --pos0 true)
WARM=(--algo offpolicy --steps 150 --sched-steps 150 --batch 8 --eval-every 50 --train-pool solved --reward-floor 0 --early-stop false --adam-eps 1e-16)
GR=(--algo grpo --steps 40 --sched-steps 40 --eval-every 10 --probe0 true --grpo-probe-seed 777 --grpo-probe-fixed true --adam-eps 1e-16 --grpo-neg-scale 0.5 --grpo-tis true --clip-rel 3 --trip-logratio false --early-stop false)
SFT=(--algo offpolicy --adapter lora "${EXPERT[@]}" --init-seed 0 --steps 150 --sched-steps 150 --batch 8 --eval-every 15 --train-pool solved --reward-floor 0 --early-stop false --adam-eps 1e-16)
EV=(--eval-n 16 --eval-chunks 1)

# math, contest_zs
D=(--env contest --contest-data "$DATA/contest/problems.jsonl" --contest-prompt zeroshot --contest-cap 5000)
python -m shared.cli bank "${D[@]}" "${ZS[@]}" "${RBANK[@]}" --cal-max 5000 --bank contest_zs --data "$DATA" --out "$OUT"
python $X/text_kv.py "${D[@]}" --model Qwen/Qwen3-30B-A3B-Base --bank contest_zs --kv-out contest_zs/textkv.pt --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${WARM[@]}" "${KVA[@]}" --init-path "$DATA/contest_zs/textkv.pt" --kv-fix-path "$DATA/contest_zs/textkv.pt" --lr 1e-3 --bank contest_zs --run contest_zs/warm_lr1e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${WARM[@]}" "${KVA[@]}" --init-path "$DATA/contest_zs/textkv.pt" --kv-fix-path "$DATA/contest_zs/textkv.pt" --lr 3e-3 --bank contest_zs --run contest_zs/warm_lr3e-3 --data "$DATA" --out "$OUT"
python $X/warm.py --warm-runs contest_zs/warm_lr1e-3 contest_zs/warm_lr3e-3 --warm-out contest_zs/warm.pt --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/contest_zs/warm.pt" --kv-fix-path "$DATA/contest_zs/warm.pt" --lr 1e-3 --bank contest_zs --run contest_zs/kv_lr1e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/contest_zs/warm.pt" --kv-fix-path "$DATA/contest_zs/warm.pt" --lr 3e-3 --bank contest_zs --run contest_zs/kv_lr3e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/contest_zs/warm.pt" --kv-fix-path "$DATA/contest_zs/warm.pt" --kv-units "$DATA/contest_zs/units_top128.json" --lr 3e-3 --bank contest_zs --run contest_zs/topk128_lr3e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" --adapter lora "${EXPERT[@]}" --init-seed 1234 --lr 1.5e-4 --bank contest_zs --run contest_zs/lora --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${SFT[@]}" --lr 5e-5 --bank contest_zs --run contest_zs/lora_sft_5e-5 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${SFT[@]}" --lr 2e-4 --bank contest_zs --run contest_zs/lora_sft_2e-4 --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter identity --bank contest_zs --eval-run contest_zs/base "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/contest_zs/warm.pt" --kv-fix-path "$DATA/contest_zs/warm.pt" --bank contest_zs --eval-run contest_zs/kv_lr1e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/contest_zs/warm.pt" --kv-fix-path "$DATA/contest_zs/warm.pt" --bank contest_zs --eval-run contest_zs/kv_lr3e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/contest_zs/warm.pt" --kv-fix-path "$DATA/contest_zs/warm.pt" --kv-units "$DATA/contest_zs/units_top128.json" --bank contest_zs --eval-run contest_zs/topk128_lr3e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank contest_zs --eval-run contest_zs/lora "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank contest_zs --eval-run contest_zs/lora_sft_5e-5 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank contest_zs --eval-run contest_zs/lora_sft_2e-4 "${EV[@]}" --data "$DATA" --out "$OUT"

# code, mbpp_nt_zs
D=(--env mbpp --mbpp-prefill true --mbpp-hide-tests true --mbpp-cap 974)
python -m shared.cli bank "${D[@]}" "${ZS[@]}" "${RBANK[@]}" --cal-max 974 --cal-keep 400 --bank mbpp_nt_zs --data "$DATA" --out "$OUT"
python $X/text_kv.py "${D[@]}" --model Qwen/Qwen3-30B-A3B-Base --bank mbpp_nt_zs --kv-out mbpp_nt_zs/textkv.pt --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${WARM[@]}" "${KVA[@]}" --init-path "$DATA/mbpp_nt_zs/textkv.pt" --kv-fix-path "$DATA/mbpp_nt_zs/textkv.pt" --lr 1e-3 --bank mbpp_nt_zs --run mbpp_nt_zs/warm_lr1e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${WARM[@]}" "${KVA[@]}" --init-path "$DATA/mbpp_nt_zs/textkv.pt" --kv-fix-path "$DATA/mbpp_nt_zs/textkv.pt" --lr 3e-3 --bank mbpp_nt_zs --run mbpp_nt_zs/warm_lr3e-3 --data "$DATA" --out "$OUT"
python $X/warm.py --warm-runs mbpp_nt_zs/warm_lr1e-3 mbpp_nt_zs/warm_lr3e-3 --warm-out mbpp_nt_zs/warm.pt --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/mbpp_nt_zs/warm.pt" --kv-fix-path "$DATA/mbpp_nt_zs/warm.pt" --lr 1e-3 --bank mbpp_nt_zs --run mbpp_nt_zs/kv_lr1e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/mbpp_nt_zs/warm.pt" --kv-fix-path "$DATA/mbpp_nt_zs/warm.pt" --lr 3e-3 --bank mbpp_nt_zs --run mbpp_nt_zs/kv_lr3e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/mbpp_nt_zs/warm.pt" --kv-fix-path "$DATA/mbpp_nt_zs/warm.pt" --kv-units "$DATA/mbpp_nt_zs/units_top128.json" --lr 3e-3 --bank mbpp_nt_zs --run mbpp_nt_zs/topk128_lr3e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" --adapter lora "${EXPERT[@]}" --init-seed 1234 --lr 3e-5 --bank mbpp_nt_zs --run mbpp_nt_zs/lora --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${SFT[@]}" --lr 5e-5 --bank mbpp_nt_zs --run mbpp_nt_zs/lora_sft_5e-5 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${SFT[@]}" --lr 2e-4 --bank mbpp_nt_zs --run mbpp_nt_zs/lora_sft_2e-4 --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter identity --bank mbpp_nt_zs --eval-run mbpp_nt_zs/base "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/mbpp_nt_zs/warm.pt" --kv-fix-path "$DATA/mbpp_nt_zs/warm.pt" --bank mbpp_nt_zs --eval-run mbpp_nt_zs/kv_lr1e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/mbpp_nt_zs/warm.pt" --kv-fix-path "$DATA/mbpp_nt_zs/warm.pt" --bank mbpp_nt_zs --eval-run mbpp_nt_zs/kv_lr3e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/mbpp_nt_zs/warm.pt" --kv-fix-path "$DATA/mbpp_nt_zs/warm.pt" --kv-units "$DATA/mbpp_nt_zs/units_top128.json" --bank mbpp_nt_zs --eval-run mbpp_nt_zs/topk128_lr3e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank mbpp_nt_zs --eval-run mbpp_nt_zs/lora "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank mbpp_nt_zs --eval-run mbpp_nt_zs/lora_sft_5e-5 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank mbpp_nt_zs --eval-run mbpp_nt_zs/lora_sft_2e-4 "${EV[@]}" --data "$DATA" --out "$OUT"

# logic, K&K + zebra
D=(--env rg --rg-tasks knights_knaves zebra_puzzles --rg-levels level1 level2 level3 --rg-per 150 --rg-seed 7 --rg-prompt zeroshot --rg-cap 800)
python -m shared.cli bank "${D[@]}" "${ZS[@]}" "${RBANK[@]}" --cal-max 800 --cal-keep 400 --bank logic_zs --data "$DATA" --out "$OUT"
python $X/text_kv.py "${D[@]}" --model Qwen/Qwen3-30B-A3B-Base --bank logic_zs --kv-out logic_zs/textkv.pt --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${WARM[@]}" "${KVA[@]}" --init-path "$DATA/logic_zs/textkv.pt" --kv-fix-path "$DATA/logic_zs/textkv.pt" --lr 1e-3 --bank logic_zs --run logic_zs/warm_lr1e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${WARM[@]}" "${KVA[@]}" --init-path "$DATA/logic_zs/textkv.pt" --kv-fix-path "$DATA/logic_zs/textkv.pt" --lr 3e-3 --bank logic_zs --run logic_zs/warm_lr3e-3 --data "$DATA" --out "$OUT"
python $X/warm.py --warm-runs logic_zs/warm_lr1e-3 logic_zs/warm_lr3e-3 --warm-out logic_zs/warm.pt --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/logic_zs/warm.pt" --kv-fix-path "$DATA/logic_zs/warm.pt" --lr 1e-3 --bank logic_zs --run logic_zs/kv_lr1e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/logic_zs/warm.pt" --kv-fix-path "$DATA/logic_zs/warm.pt" --lr 3e-3 --bank logic_zs --run logic_zs/kv_lr3e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/logic_zs/warm.pt" --kv-fix-path "$DATA/logic_zs/warm.pt" --kv-units "$DATA/logic_zs/units_top128.json" --lr 3e-3 --bank logic_zs --run logic_zs/topk128_lr3e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" --adapter lora "${EXPERT[@]}" --init-seed 1234 --lr 5e-5 --bank logic_zs --run logic_zs/lora --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${SFT[@]}" --lr 5e-5 --bank logic_zs --run logic_zs/lora_sft_5e-5 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${SFT[@]}" --lr 2e-4 --bank logic_zs --run logic_zs/lora_sft_2e-4 --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter identity --bank logic_zs --eval-run logic_zs/base "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/logic_zs/warm.pt" --kv-fix-path "$DATA/logic_zs/warm.pt" --bank logic_zs --eval-run logic_zs/kv_lr1e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/logic_zs/warm.pt" --kv-fix-path "$DATA/logic_zs/warm.pt" --bank logic_zs --eval-run logic_zs/kv_lr3e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/logic_zs/warm.pt" --kv-fix-path "$DATA/logic_zs/warm.pt" --kv-units "$DATA/logic_zs/units_top128.json" --bank logic_zs --eval-run logic_zs/topk128_lr3e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank logic_zs --eval-run logic_zs/lora "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank logic_zs --eval-run logic_zs/lora_sft_5e-5 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank logic_zs --eval-run logic_zs/lora_sft_2e-4 "${EV[@]}" --data "$DATA" --out "$OUT"

# law
D=(--env knowledge --knowledge-data "$DATA/knowledge/law.jsonl" --knowledge-prompt zeroshot --knowledge-cap 800)
python -m shared.cli bank "${D[@]}" "${ZS[@]}" "${RBANK[@]}" --cal-max 800 --cal-keep 400 --bank know_law --data "$DATA" --out "$OUT"
python $X/text_kv.py "${D[@]}" --model Qwen/Qwen3-30B-A3B-Base --bank know_law --kv-out know_law/textkv.pt --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${WARM[@]}" "${KVA[@]}" --init-path "$DATA/know_law/textkv.pt" --kv-fix-path "$DATA/know_law/textkv.pt" --lr 1e-3 --bank know_law --run know_law/warm_lr1e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${WARM[@]}" "${KVA[@]}" --init-path "$DATA/know_law/textkv.pt" --kv-fix-path "$DATA/know_law/textkv.pt" --lr 3e-3 --bank know_law --run know_law/warm_lr3e-3 --data "$DATA" --out "$OUT"
python $X/warm.py --warm-runs know_law/warm_lr1e-3 know_law/warm_lr3e-3 --warm-out know_law/warm.pt --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/know_law/warm.pt" --kv-fix-path "$DATA/know_law/warm.pt" --lr 1e-3 --bank know_law --run know_law/kv_lr1e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/know_law/warm.pt" --kv-fix-path "$DATA/know_law/warm.pt" --lr 3e-3 --bank know_law --run know_law/kv_lr3e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/know_law/warm.pt" --kv-fix-path "$DATA/know_law/warm.pt" --kv-units "$DATA/know_law/units_top128.json" --lr 3e-3 --bank know_law --run know_law/topk128_lr3e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" --adapter lora "${EXPERT[@]}" --init-seed 1234 --lr 1.5e-4 --bank know_law --run know_law/lora --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${SFT[@]}" --lr 5e-5 --bank know_law --run know_law/lora_sft_5e-5 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${SFT[@]}" --lr 2e-4 --bank know_law --run know_law/lora_sft_2e-4 --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter identity --bank know_law --eval-run know_law/base "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/know_law/warm.pt" --kv-fix-path "$DATA/know_law/warm.pt" --bank know_law --eval-run know_law/kv_lr1e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/know_law/warm.pt" --kv-fix-path "$DATA/know_law/warm.pt" --bank know_law --eval-run know_law/kv_lr3e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/know_law/warm.pt" --kv-fix-path "$DATA/know_law/warm.pt" --kv-units "$DATA/know_law/units_top128.json" --bank know_law --eval-run know_law/topk128_lr3e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank know_law --eval-run know_law/lora "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank know_law --eval-run know_law/lora_sft_5e-5 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank know_law --eval-run know_law/lora_sft_2e-4 "${EV[@]}" --data "$DATA" --out "$OUT"

# casehold
D=(--env knowledge --knowledge-data "$DATA/knowledge/casehold.jsonl" --knowledge-prompt zeroshot --knowledge-cap 800)
python -m shared.cli bank "${D[@]}" "${ZS[@]}" "${RBANK[@]}" --cal-max 800 --cal-keep 400 --bank know_casehold --data "$DATA" --out "$OUT"
python $X/text_kv.py "${D[@]}" --model Qwen/Qwen3-30B-A3B-Base --bank know_casehold --kv-out know_casehold/textkv.pt --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${WARM[@]}" "${KVA[@]}" --init-path "$DATA/know_casehold/textkv.pt" --kv-fix-path "$DATA/know_casehold/textkv.pt" --lr 1e-3 --bank know_casehold --run know_casehold/warm_lr1e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${WARM[@]}" "${KVA[@]}" --init-path "$DATA/know_casehold/textkv.pt" --kv-fix-path "$DATA/know_casehold/textkv.pt" --lr 3e-3 --bank know_casehold --run know_casehold/warm_lr3e-3 --data "$DATA" --out "$OUT"
python $X/warm.py --warm-runs know_casehold/warm_lr1e-3 know_casehold/warm_lr3e-3 --warm-out know_casehold/warm.pt --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/know_casehold/warm.pt" --kv-fix-path "$DATA/know_casehold/warm.pt" --lr 1e-3 --bank know_casehold --run know_casehold/kv_lr1e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/know_casehold/warm.pt" --kv-fix-path "$DATA/know_casehold/warm.pt" --lr 3e-3 --bank know_casehold --run know_casehold/kv_lr3e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/know_casehold/warm.pt" --kv-fix-path "$DATA/know_casehold/warm.pt" --kv-units "$DATA/know_casehold/units_top128.json" --lr 3e-3 --bank know_casehold --run know_casehold/topk128_lr3e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" --adapter lora "${EXPERT[@]}" --init-seed 1234 --lr 1e-4 --bank know_casehold --run know_casehold/lora --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${SFT[@]}" --lr 5e-5 --bank know_casehold --run know_casehold/lora_sft_5e-5 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${SFT[@]}" --lr 2e-4 --bank know_casehold --run know_casehold/lora_sft_2e-4 --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter identity --bank know_casehold --eval-run know_casehold/base "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/know_casehold/warm.pt" --kv-fix-path "$DATA/know_casehold/warm.pt" --bank know_casehold --eval-run know_casehold/kv_lr1e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/know_casehold/warm.pt" --kv-fix-path "$DATA/know_casehold/warm.pt" --bank know_casehold --eval-run know_casehold/kv_lr3e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/know_casehold/warm.pt" --kv-fix-path "$DATA/know_casehold/warm.pt" --kv-units "$DATA/know_casehold/units_top128.json" --bank know_casehold --eval-run know_casehold/topk128_lr3e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank know_casehold --eval-run know_casehold/lora "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank know_casehold --eval-run know_casehold/lora_sft_5e-5 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank know_casehold --eval-run know_casehold/lora_sft_2e-4 "${EV[@]}" --data "$DATA" --out "$OUT"

# medqa
D=(--env knowledge --knowledge-data "$DATA/knowledge/medqa.jsonl" --knowledge-prompt zeroshot --knowledge-cap 800)
python -m shared.cli bank "${D[@]}" "${ZS[@]}" "${RBANK[@]}" --cal-max 800 --cal-keep 400 --bank know_medqa --data "$DATA" --out "$OUT"
python $X/text_kv.py "${D[@]}" --model Qwen/Qwen3-30B-A3B-Base --bank know_medqa --kv-out know_medqa/textkv.pt --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${WARM[@]}" "${KVA[@]}" --init-path "$DATA/know_medqa/textkv.pt" --kv-fix-path "$DATA/know_medqa/textkv.pt" --lr 1e-3 --bank know_medqa --run know_medqa/warm_lr1e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${WARM[@]}" "${KVA[@]}" --init-path "$DATA/know_medqa/textkv.pt" --kv-fix-path "$DATA/know_medqa/textkv.pt" --lr 3e-3 --bank know_medqa --run know_medqa/warm_lr3e-3 --data "$DATA" --out "$OUT"
python $X/warm.py --warm-runs know_medqa/warm_lr1e-3 know_medqa/warm_lr3e-3 --warm-out know_medqa/warm.pt --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/know_medqa/warm.pt" --kv-fix-path "$DATA/know_medqa/warm.pt" --lr 1e-3 --bank know_medqa --run know_medqa/kv_lr1e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/know_medqa/warm.pt" --kv-fix-path "$DATA/know_medqa/warm.pt" --lr 3e-3 --bank know_medqa --run know_medqa/kv_lr3e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/know_medqa/warm.pt" --kv-fix-path "$DATA/know_medqa/warm.pt" --kv-units "$DATA/know_medqa/units_top128.json" --lr 3e-3 --bank know_medqa --run know_medqa/topk128_lr3e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" --adapter lora "${EXPERT[@]}" --init-seed 1234 --lr 5e-5 --bank know_medqa --run know_medqa/lora --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${SFT[@]}" --lr 5e-5 --bank know_medqa --run know_medqa/lora_sft_5e-5 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${SFT[@]}" --lr 2e-4 --bank know_medqa --run know_medqa/lora_sft_2e-4 --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter identity --bank know_medqa --eval-run know_medqa/base "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/know_medqa/warm.pt" --kv-fix-path "$DATA/know_medqa/warm.pt" --bank know_medqa --eval-run know_medqa/kv_lr1e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/know_medqa/warm.pt" --kv-fix-path "$DATA/know_medqa/warm.pt" --bank know_medqa --eval-run know_medqa/kv_lr3e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/know_medqa/warm.pt" --kv-fix-path "$DATA/know_medqa/warm.pt" --kv-units "$DATA/know_medqa/units_top128.json" --bank know_medqa --eval-run know_medqa/topk128_lr3e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank know_medqa --eval-run know_medqa/lora "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank know_medqa --eval-run know_medqa/lora_sft_5e-5 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank know_medqa --eval-run know_medqa/lora_sft_2e-4 "${EV[@]}" --data "$DATA" --out "$OUT"

# trivia
D=(--env knowledge --knowledge-data "$DATA/knowledge/trivia.jsonl" --knowledge-prompt zeroshot --knowledge-cap 800)
python -m shared.cli bank "${D[@]}" "${ZS[@]}" "${RBANK[@]}" --cal-max 800 --cal-keep 400 --bank know_trivia --data "$DATA" --out "$OUT"
python $X/text_kv.py "${D[@]}" --model Qwen/Qwen3-30B-A3B-Base --bank know_trivia --kv-out know_trivia/textkv.pt --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${WARM[@]}" "${KVA[@]}" --init-path "$DATA/know_trivia/textkv.pt" --kv-fix-path "$DATA/know_trivia/textkv.pt" --lr 1e-3 --bank know_trivia --run know_trivia/warm_lr1e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${WARM[@]}" "${KVA[@]}" --init-path "$DATA/know_trivia/textkv.pt" --kv-fix-path "$DATA/know_trivia/textkv.pt" --lr 3e-3 --bank know_trivia --run know_trivia/warm_lr3e-3 --data "$DATA" --out "$OUT"
python $X/warm.py --warm-runs know_trivia/warm_lr1e-3 know_trivia/warm_lr3e-3 --warm-out know_trivia/warm.pt --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/know_trivia/warm.pt" --kv-fix-path "$DATA/know_trivia/warm.pt" --lr 1e-3 --bank know_trivia --run know_trivia/kv_lr1e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/know_trivia/warm.pt" --kv-fix-path "$DATA/know_trivia/warm.pt" --lr 3e-3 --bank know_trivia --run know_trivia/kv_lr3e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" "${KVA[@]}" --kv-graph true --init-path "$DATA/know_trivia/warm.pt" --kv-fix-path "$DATA/know_trivia/warm.pt" --kv-units "$DATA/know_trivia/units_top128.json" --lr 3e-3 --bank know_trivia --run know_trivia/topk128_lr3e-3 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZG[@]}" "${GR[@]}" --adapter lora "${EXPERT[@]}" --init-seed 1234 --lr 1.5e-4 --bank know_trivia --run know_trivia/lora --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${SFT[@]}" --lr 5e-5 --bank know_trivia --run know_trivia/lora_sft_5e-5 --data "$DATA" --out "$OUT"
python -m shared.cli train "${D[@]}" "${ZS[@]}" "${SFT[@]}" --lr 2e-4 --bank know_trivia --run know_trivia/lora_sft_2e-4 --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter identity --bank know_trivia --eval-run know_trivia/base "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/know_trivia/warm.pt" --kv-fix-path "$DATA/know_trivia/warm.pt" --bank know_trivia --eval-run know_trivia/kv_lr1e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/know_trivia/warm.pt" --kv-fix-path "$DATA/know_trivia/warm.pt" --bank know_trivia --eval-run know_trivia/kv_lr3e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" "${KVA[@]}" --init-path "$DATA/know_trivia/warm.pt" --kv-fix-path "$DATA/know_trivia/warm.pt" --kv-units "$DATA/know_trivia/units_top128.json" --bank know_trivia --eval-run know_trivia/topk128_lr3e-3 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank know_trivia --eval-run know_trivia/lora "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank know_trivia --eval-run know_trivia/lora_sft_5e-5 "${EV[@]}" --data "$DATA" --out "$OUT"
python -m shared.cli eval "${D[@]}" "${ZS[@]}" --adapter lora "${EXPERT[@]}" --bank know_trivia --eval-run know_trivia/lora_sft_2e-4 "${EV[@]}" --data "$DATA" --out "$OUT"

# ---- Part S  SFT cartridges against a rank-1 LoRA, and co-residence (RESULTS_SFT.md): runs_sft.sh
