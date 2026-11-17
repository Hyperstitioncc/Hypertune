#!/bin/bash
# Reproduces RESULTS_SFT.md section 4: SFT cartridges against a rank-1 LoRA on five domains, and co-residence.
#   runs_sft.sh DATA OUT
# 1,000 training examples a domain (function calling 700), greedy, the HF path (no vLLM), 500 test problems
# (function calling 275), one seed. One GPU a command. To split a training over N GPUs, run it as
# `torchrun --nproc_per_node=N -m shared.cli train --dist true ...` and divide --irr-n by N.
# For a quick check that every command runs: S_PASSES=1 S_DOMS="intent parse" S_FILL="syllogism word_sorting".
set -euo pipefail
DATA=$(realpath -m "${1:?usage: runs_sft.sh DATA OUT}")
OUT=$(realpath -m "${2:?usage: runs_sft.sh DATA OUT}")
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
X=experiments/soft-rlvr-qwen3moe
EXPERT=(--lora-rank 1 --lora-alpha 32 --lora-targets q_proj k_proj v_proj o_proj experts)
N=${S_PASSES:-10}

SD=$DATA/sft; RAW=$SD/raw
HF=(--model Qwen/Qwen3-30B-A3B-Base --attn eager --backend hf_eager --hf-multi true --hf-chunk 32 --temperature 0 --top-p 1)
SBANK=(--adapter identity --bank-split all --select splits --n-chunks 1 --n-per-chunk 1)
SFT=(--algo sft --sft-eos true --sft-select grade --weight-decay 0 --batch 16 --batch-order epoch --train-pool solved --early-stop false --probe0 true --trip-logratio false --reward-floor 0)
CART=(--adapter kv_prefix --k 16 --row-tail true --pos0 true)
SEV=(--eval-n 1 --eval-chunks 1 --eval-split clean)
IO=(--data "$DATA" --out "$OUT")
FILL=(${S_FILL:-family_relationships letter_counting number_sorting sentence_reordering syllogism word_sorting})
NEG=(agnews alpaca dolly gsm8k imdb mbpp squad wikitext)
DOMS=(${S_DOMS:-bf finqa funcall intent parse})
declare -A CAP=([bf]=700 [finqa]=96 [funcall]=512 [intent]=16 [parse]=128) PER=([bf]=63 [finqa]=63 [funcall]=44 [intent]=63 [parse]=63) IDX=([bf]=0 [finqa]=0 [funcall]=0 [intent]=0 [parse]=40)

# data: the domains, the filler tasks, and the off-task prompts for the write penalty
mkdir -p "$RAW/finqa" "$RAW/esolang"
for f in train dev test; do [ -s "$RAW/finqa/$f.json" ] || curl -sL -o "$RAW/finqa/$f.json" "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/$f.json"; done
[ -s "$RAW/esolang/test.jsonl" ] || curl -sL -o "$RAW/esolang/test.jsonl" https://huggingface.co/datasets/Lossfunk/Esolang-Bench/resolve/main/data/test.jsonl
for d in "${DOMS[@]}"; do
  case $d in
    funcall) python -m envs.domains.funcall build --funcall-data "$SD/funcall" --n-train 700 --n-dev 100 --n-test 275;;   # 1,075 usable rows
    finqa|bf) python -m envs.domains.$d build --$d-data "$SD/$d" --raw "$RAW";;
    *) python -m envs.domains.$d build --$d-data "$SD/$d";;
  esac
done
for t in "${FILL[@]}"; do python -m envs.domains.filler build --filler-task $t --filler-data "$SD/filler-$t" --n-train 300 --n-dev 50 --n-test 100; done
python $X/tools/build_neg_pool.py 300 "$SD/neg"
python $X/tools/irr_items.py "$SD/irr.json" "${NEG[@]/#/$SD/neg/}" "${FILL[@]/#/$SD/filler-}"
for t in "${FILL[@]}"; do
  others=(); for o in "${FILL[@]}"; do [ $o = $t ] || others+=("$SD/filler-$o"); done
  python $X/tools/irr_items.py "$SD/irr_no_$t.json" "${NEG[@]/#/$SD/neg/}" "${others[@]}"
done

# sft NAME ENV-FLAGS... : bank, start point, N passes, test score. Reads S_RUN, S_CAP, S_PER, S_IDX, S_DEC, S_MORE.
sft() {
  local name=$1; shift
  local M=("${HF[@]}" "${S_DEC[@]}" --max-tokens $S_CAP) K=$SD/$name/textkv.pt
  [ -f "$SD/$name/bank/bank.json" ] || python -m shared.cli bank "$@" "${M[@]}" "${SBANK[@]}" --bank sft/$name/bank "${IO[@]}"
  [ -f "$K" ] || python $X/text_kv.py "$@" --model Qwen/Qwen3-30B-A3B-Base --attn eager "${S_DEC[@]}" --bank sft/$name/bank --kv-out sft/$name/textkv.pt --idx $S_IDX "${IO[@]}"
  local A=("${CART[@]}" --init-path "$K" --kv-fix-path "$K")
  python -m shared.cli train "$@" "${M[@]}" "${A[@]}" "${SFT[@]}" "${S_MORE[@]}" --sft-corpus sft/$name/corpus.json --lr 1e-2 \
      --steps $((N * S_PER)) --sched-steps $((N * S_PER)) --eval-every $S_PER --bank sft/$name/bank --run sft/$name/$S_RUN "${IO[@]}"
  python -m shared.cli eval "$@" "${M[@]}" "${A[@]}" --bank sft/$name/bank --eval-run sft/$name/$S_RUN "${SEV[@]}" "${IO[@]}"
}
dec() { S_DEC=(); [ $1 = funcall ] && S_DEC=(--spell-reserved true); S_CAP=${CAP[$1]}; S_PER=${PER[$1]}; S_IDX=${IDX[$1]}; }

# single adapters: plain cartridge (N passes) and rank-1 LoRA (3 passes, or N if smaller), alone
L=$(( N < 3 ? N : 3 ))
for d in "${DOMS[@]}"; do
  dec $d; S_RUN=cart; S_MORE=()
  sft $d --env $d --$d-data "$SD/$d"
  E=(--env $d --$d-data "$SD/$d"); M=("${HF[@]}" "${S_DEC[@]}" --max-tokens $S_CAP)
  python -m shared.cli train "${E[@]}" "${M[@]}" "${SFT[@]}" --adapter lora "${EXPERT[@]}" --sft-corpus sft/$d/corpus.json --lr 1e-4 \
      --steps $((L * S_PER)) --sched-steps $((L * S_PER)) --eval-every $S_PER --bank sft/$d/bank --run sft/$d/lora "${IO[@]}"
  python -m shared.cli eval "${E[@]}" "${M[@]}" --adapter lora "${EXPERT[@]}" --bank sft/$d/bank --eval-run sft/$d/lora "${SEV[@]}" "${IO[@]}"
done

# co-residence: filler cartridges with the write penalty; each domain's cartridge with the write penalty and the
# fillers present with probability 0.3; then every domain scored with the other domains loaded
PEN=(--irr-pen 1.0 --irr-kind write --irr-resp 128 --irr-n 8)
for t in "${FILL[@]}"; do
  S_DEC=(); S_CAP=64; S_PER=19; S_IDX=0; S_RUN=cartW; S_MORE=("${PEN[@]}" --irr-items sft/irr_no_$t.json)
  sft filler-$t --env filler --filler-task $t --filler-data "$SD/filler-$t"
done
FP=("${FILL[@]/#/sft/filler-}")
python -m shared.cli stack --stack-parts "${FP[@]/%//cartW}" --stack-out sft/fillers.pt "${IO[@]}"
for d in "${DOMS[@]}"; do
  dec $d; S_RUN=cartWP; S_MORE=("${PEN[@]}" --irr-items sft/irr.json --kv-partners "$SD/fillers.pt" --kv-partner-p 0.3)
  sft $d --env $d --$d-data "$SD/$d"
done
for d in "${DOMS[@]}"; do
  dec $d; parts=(sft/$d/cartWP); for o in "${DOMS[@]}"; do [ $o = $d ] || parts+=(sft/$o/cartWP); done
  python -m shared.cli stack --stack-parts "${parts[@]}" --stack-out sft/$d/stack.pt "${IO[@]}"
  python -m shared.cli eval --env $d --$d-data "$SD/$d" "${HF[@]}" "${S_DEC[@]}" --max-tokens $S_CAP --adapter kv_prefix --k $((16 * ${#DOMS[@]})) --row-tail true --pos0 true \
      --init-path "$SD/$d/stack.pt" --kv-fix-path "$SD/$d/stack.pt" --bank sft/$d/bank --eval-run sft/$d/stack "${SEV[@]}" "${IO[@]}"
done
