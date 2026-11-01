#!/usr/bin/env bash
# The whole vLLM comparison into bench/ref/results/<date>/ (~2.5 h): the race on both models
# in BF16 and with INT8 weights and/or KV (alternate.py), then the speculative-decoding runs
# on the chat prompts (tileinfer_spec.jl / vllm_spec.py, the engines alternating).
#
#   bench/ref/run_all.sh [date]     # needs bench/ref/.venv and the W8A16 copies (make_w8a16.py)
set -u
cd "$(dirname "$0")/../.."
JULIA=${JULIA:-~/.juliaup/bin/julia}
D=bench/ref/results/${1:-$(date +%F)}
mkdir -p "$D/spec"

for m in Qwen3-0.6B Qwen3-4B; do
  for cfg in "bf16 bf16 bf16" "int8 bf16 int8-weights" "bf16 int8 int8-kv" "int8 int8 int8-weights-kv"; do
    set -- $cfg
    echo "=== race $m $3"
    python3 -u bench/ref/alternate.py --model Qwen/$m --rounds 3 --trials 10 --weights $1 --kv $2 \
      --out "$D/race-$m-$3.json" 2>&1 | grep -v '^    [0-9]'
  done
done

i=0
spec() {  # model method k weights kv
  local tag="$(basename $1)-$2$3-w$4-kv$5" order="vllm tileinfer"
  i=$((i + 1)); [ $((i % 2)) = 0 ] && order="tileinfer vllm"
  for e in $order; do
    echo "=== spec $e $tag"
    if [ $e = vllm ]; then
      bench/ref/.venv/bin/python bench/ref/vllm_spec.py "$@" > "$D/spec/$e-$tag.json" 2> "$D/spec/$e-$tag.err"
    else
      $JULIA --project=bench bench/ref/tileinfer_spec.jl "$@" > "$D/spec/$e-$tag.json" 2> "$D/spec/$e-$tag.err"
    fi
    echo "exit $?"; grep -E ' out [0-9]+  decode' "$D/spec/$e-$tag.err"
  done
}
for wk in "bf16 bf16" "int8 int8"; do
  spec Qwen/Qwen3-4B plain 0 $wk
  spec Qwen/Qwen3-4B dflash 7 $wk
  spec Qwen/Qwen3-4B dflash 15 $wk
  spec Qwen/Qwen3-0.6B plain 0 $wk
  spec Qwen/Qwen3-0.6B ngram 4 $wk
done
