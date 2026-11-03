#!/usr/bin/env bash
# One engine's speculative-decoding replies under Nsight Systems, at one point of
# docs/src/vllm.md (Qwen3-4B, DFlash k = 15), then exported to CSV/JSON by nsys_export.py:
#
#   bench/ref/profile.sh ENGINE POINT WEIGHTS [GEN] [-- DRIVER ARGS...]
#     ENGINE   vllm or tileinfer (vllm_spec.py / tileinfer_spec.jl with --profile)
#     POINT    1k (spec_prompts.json) or 8k (spec_prompts_8k.json)
#     WEIGHTS  bf16 or int8
#     GEN      tokens per reply (default 128 at 1k, 256 at 8k: the step cost at that context,
#              without tracing a whole 2,048-token reply)
#   e.g. bench/ref/profile.sh vllm 8k bf16
#        TAG=prefix bench/ref/profile.sh vllm 8k bf16 256 -- --prefix-caching
#
# Writes bench/ref/results/profiles/ENGINE-POINT-wWEIGHTS[-TAG].{nsys-rep,sqlite,json,err} and the
# export directory of the same name (kernels.csv, steps.csv, summary.json; see nsys_export.py).
# Each prompt is warmed up untraced (compilation, graph capture, clocks), then one reply per
# prompt is traced: TileInfer between the CUDA profiler API's start and stop; vLLM whole (nsys
# does not honour cudaProfilerStart from its engine-core subprocess), and nsys_export.py keeps
# what was launched inside the `request NAME` ranges. Kernels launched from CUDA
# graphs are traced one by one (--cuda-graph-trace=node), and vLLM's engine-core subprocess
# is followed (--wait=primary: nsys does not wait for processes vLLM leaves behind). Tracing
# adds host overhead per launch: compare busy times and shares between engines, and take absolute speeds from vllm_spec.py / tileinfer_spec.jl.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ $# -ge 3 ] || { sed -n '2,20p' "$0"; exit 1; }
engine=$1 point=$2 weights=$3; shift 3
case $point in
  1k) prompts=spec_prompts.json gen=128 ;;
  8k) prompts=spec_prompts_8k.json gen=256 ;;
  *) echo "point: 1k or 8k" >&2; exit 1 ;;
esac
if [ $# -gt 0 ] && [ "$1" != "--" ]; then gen=$1; shift; fi
[ "${1:-}" = "--" ] && shift

NSYS=${NSYS:-$(command -v nsys || ls -d /usr/local/NVIDIA-Nsight-Compute*/host/target-linux-x64/nsys 2>/dev/null | tail -1)}
[ -x "$NSYS" ] || { echo "nsys not found; set NSYS" >&2; exit 1; }
JULIA=${JULIA:-$(command -v julia || echo ~/.juliaup/bin/julia)}

D=bench/ref/results/profiles
mkdir -p "$D"
name="$engine-$point-w$weights${TAG:+-$TAG}"
case $engine in
  vllm) range=(--trace-fork-before-exec=true)   # its engine core is a forked (not exec'd) process
        cmd=(bench/ref/.venv/bin/python bench/ref/vllm_spec.py Qwen/Qwen3-4B dflash 15 "$weights" bf16
             --prompts "$prompts" --gen "$gen" --profile "$@") ;;
  tileinfer) range=(--capture-range=cudaProfilerApi --capture-range-end=stop)
             cmd=("$JULIA" --project=bench bench/ref/tileinfer_spec.jl Qwen/Qwen3-4B dflash 15 "$weights" bf16
                  "$gen" 1 "$prompts" --profile "$@") ;;
  *) echo "engine: vllm or tileinfer" >&2; exit 1 ;;
esac

echo "=== $name: ${cmd[*]}"
"$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  "${range[@]}" --wait=primary -f true -o "$D/$name" \
  "${cmd[@]}" > "$D/$name.json" 2> "$D/$name.err" || { tail -20 "$D/$name.err"; exit 1; }
python3 bench/ref/nsys_export.py "$D/$name.nsys-rep"
