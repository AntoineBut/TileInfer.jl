#!/usr/bin/env bash
# vLLM at the two points of docs/src/vllm.md, in the setup one would run
# for real: Qwen3-4B with DFlash (k = 15), BF16 and INT8 (W8A16) weights, greedy:
#   1k   the two 1,024-token prompts of spec_prompts.json, 128 tokens
#   8k   the two 8,192-token prompts of spec_prompts_8k.json, 2,048 tokens
# Each configuration runs in ROUNDS fresh processes into bench/ref/results/<date>/
# (~25 min in all). The TileInfer side is tileinfer_spec.jl with the same arguments.
#
#   bench/ref/run_all.sh [date]     # needs bench/ref/.venv and the W8A16 copy (make_w8a16.py)
set -u
cd "$(dirname "$0")/../.."
ROUNDS=${ROUNDS:-3}
D=bench/ref/results/${1:-$(date +%F)}
mkdir -p "$D"

# point, prompt file, tokens
for point in "1k spec_prompts.json 128" "8k spec_prompts_8k.json 2048"; do
  set -- $point
  for weights in bf16 int8; do
    for r in $(seq 1 "$ROUNDS"); do
      out="$D/vllm-$1-Qwen3-4B-dflash15-w$weights-$r"
      echo "=== vllm $1, dflash k=15, $weights weights, round $r"
      bench/ref/.venv/bin/python bench/ref/vllm_spec.py Qwen/Qwen3-4B dflash 15 "$weights" bf16 \
        --prompts "$2" --gen "$3" > "$out.json" 2> "$out.err"
      echo "exit $?"; grep -E ' out [0-9]+  decode' "$out.err"
    done
  done
done
