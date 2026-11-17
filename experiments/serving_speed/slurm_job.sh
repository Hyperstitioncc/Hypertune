#!/bin/bash
# The serving-speed benchmark on ONE node of a Slurm cluster. Everything it reads at run time sits on that node's
# own disk (/scratch): the two models and the adapter folders.
#   slurm_job.sh gen MODEL RANK N SUBDIR [TARGETS]   CPU job: N PEFT adapters of that rank into adapters/MODEL/SUBDIR
#   slurm_job.sh work                                1-GPU job: runs configurations from configs.txt until none is left
# A configuration is one line "ID MODEL KIND N RANK [ADAPTERS] [VARIANT]"; a worker claims it by creating
# claims/ID, so several workers share the list. LoRA configurations wait until their adapter folder is complete.
#   ADAPTERS: the adapter folder under adapters/MODEL (default r<RANK>; attn_r1 = attention layers only).
#   VARIANT:  how the LoRA path is set up, parts joined by "+". "-" = vLLM's defaults; ds = LoRA on a second
#             CUDA stream (VLLM_LORA_ENABLE_DUAL_STREAM); tuned = the kernel settings found by tune_kernels.py
#             (VLLM_TUNED_CONFIG_FOLDER=tune/tuned; tuned=NAME for tune/NAME); gather = lora_gather.py in place of vLLM's GPU kernels;
#             attn = the engine applies LoRA to q/k/v/o only; greedy = temperature 0, one pass (to compare outputs).
# env: CONFIGS (default configs.txt), CLAIMS (default claims), RESULTS (default results.jsonl): the list, its claim
#      folder and the results file, under serving_speed.
set -uo pipefail
R=${CART_ROOT:?a shared folder holding venv/ and hf/ (the Hugging Face cache)}; S=$R/serving_speed; L=${CART_LIB:?the library root}; X=$L/experiments
LOCAL=${SS_LOCAL:-/scratch/$USER/serving_speed}; CF=$S/${CONFIGS:-configs.txt}; CL=$S/${CLAIMS:-claims}
mkdir -p $LOCAL/models $LOCAL/adapters $LOCAL/work $S/logs $CL
export PATH=$R/venv/bin:$PATH HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=$L:$X/serving_speed TMPDIR=$LOCAL/work
declare -A HUB=([moe]=Qwen3-30B-A3B-Base [dense]=Qwen3-32B)

stage() {       # a model's files onto the node's disk, once
  local dst=$LOCAL/models/${HUB[$1]} src
  src=$(ls -d $R/hf/hub/models--Qwen--${HUB[$1]}/snapshots/*/ | head -1)
  ( flock 9
    [ -f $dst/.done ] || { echo "staging ${HUB[$1]} to $dst $(date -u +%T)"; mkdir -p $dst && cp -L $src/* $dst/ && touch $dst/.done; echo "staged $(date -u +%T)"; }
  ) 9> $LOCAL/stage-$1.lock
  [ -f $dst/.done ]
}

if [ "$1" = gen ]; then
  m=$2; rank=$3; n=$4; sub=$5; targets=${6:-all}; out=$LOCAL/adapters/$m/$sub
  echo "START gen $m rank $rank n $n $sub on $(hostname) $(date -u +%FT%TZ)"
  stage $m || exit 1
  python $X/serving_speed/make_loras.py --model $LOCAL/models/${HUB[$m]} --rank $rank --n $n --out $out --targets $targets || exit 1
  touch $out/.done-$n; echo "END gen $(date -u +%FT%TZ)"; exit 0
fi

echo "START worker on $(hostname) job ${SLURM_JOB_ID:-?} $(date -u +%FT%TZ)"
tuned_dir() {   # the kernel-settings folder a variant names: "tuned" = tune/tuned, "tuned=NAME" = tune/NAME
  local part
  for part in ${1//+/ }; do
    case $part in tuned) echo $S/tune/tuned;; tuned=*) echo $S/tune/${part#tuned=};; esac
  done
}
need() {        # the adapter folder a configuration reads, complete?
  local m=$1 kind=$2 n=$3 rank=$4 sub=$5
  local t; t=$(tuned_dir ${6:--})
  [ -z "$t" ] || [ -f $t/NVIDIA_B200_SHRINK.json ] || return 1           # never run "tuned" on defaults
  case $kind in
    lora) ls $LOCAL/adapters/$m/$sub/.done-* 2>/dev/null | sed 's/.*done-//' | awk -v n=$n '$1 >= n {ok = 1} END {exit !ok}';;
    check) [ -f $LOCAL/adapters/$m/check_attn/.done-1 ] && [ -f $LOCAL/adapters/$m/check_mlp/.done-1 ];;
    *) true;;
  esac
}
for round in $(seq 1 240); do
  ran=0; waiting=0
  while read -r id m kind n rank sub variant; do
    [ -z "$id" ] && continue
    [ -d $CL/$id ] && continue
    sub=${sub:-r$rank}; variant=${variant:--}
    need $m $kind $n $rank $sub $variant || { waiting=1; continue; }
    mkdir $CL/$id 2>/dev/null || continue
    unset VLLM_LORA_ENABLE_DUAL_STREAM VLLM_LORA_DISABLE_PDL VLLM_TUNED_CONFIG_FOLDER; export SS_VARIANT=; extra=()
    [ $variant != - ] && export SS_VARIANT=$variant
    case +$variant+ in *+ds+*) export VLLM_LORA_ENABLE_DUAL_STREAM=1 VLLM_LORA_DISABLE_PDL=1;; esac   # PDL fails to compile on this GPU generation (vLLM's note)
    t=$(tuned_dir $variant); [ -n "$t" ] && export VLLM_TUNED_CONFIG_FOLDER=$t
    case +$variant+ in *+gather+*) extra+=(--lora-ops gather);; esac
    case +$variant+ in *+attn+*) extra+=(--targets attn);; esac
    case +$variant+ in *+greedy+*) extra+=(--temperature 0 --repeats 1);; esac
    stage $m || { echo "staging $m failed"; exit 1; }
    log=$S/logs/$id.log; model=$LOCAL/models/${HUB[$m]}
    echo "RUN $id $(date -u +%T)"
    case $kind in
      base) args=(--kind base);;
      basec) args=(--kind base --cart-engine true);;
      cart) args=(--kind cart --n $n);;
      lora) args=(--kind lora --n $n --rank $rank --adapters $LOCAL/adapters/$m/$sub);;
      check) mkdir -p $LOCAL/adapters/$m/check; ln -sfn ../check_attn $LOCAL/adapters/$m/check/attn; ln -sfn ../check_mlp $LOCAL/adapters/$m/check/mlp
             args=(--kind check --rank $rank --adapters $LOCAL/adapters/$m/check);;
      regress) (cd $L && timeout 60m python $X/tools/check_vllm_pos0.py --env intent --intent-data $R/data/intent --model Qwen/Qwen3-30B-A3B-Base \
                  --n 100 --kv-file $n --graph true) > $log 2>&1 < /dev/null
               grep -E '^RESULT' $log || { echo "FAILED $id"; grep -nE 'Error|error:' $log | tail -5 | cut -c1-260; }
               ran=1; continue;;
    esac
    timeout 90m python $X/serving_speed/bench.py --model $model "${args[@]}" "${extra[@]}" --out $S/${RESULTS:-results.jsonl} --log $log > $log 2>&1 < /dev/null
    echo "exit $? $(grep -E '^RESULT' $log | tail -1 | cut -c1-600)"
    ran=1
  done < $CF
  [ $ran = 0 ] && [ $waiting = 0 ] && break
  [ $ran = 0 ] && sleep 60
done
echo "END worker $(date -u +%FT%TZ)"
