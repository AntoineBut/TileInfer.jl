# Comparison with vLLM

vLLM 0.31 at two points (two 1,024-token prompts and 128 greedy tokens, the reference
point of the performance page; two 8,192-token prompts and 2,048 tokens), in the setup one would run for real: Qwen3-4B with DFlash (k = 15), BF16
and INT8 weights. Results and their discussion:
[docs/src/vllm.md](../../docs/src/vllm.md). Runs write their JSON into
`results/<date>/` (not tracked).

| File | What it does |
|---|---|
| `spec_prompts.jl` → `spec_prompts.json`, `spec_prompts_8k.json` | the chat prompts of `bench/prompts.jl` as token ids, shared by both engines |
| `vllm_spec.py`, `tileinfer_spec.jl` | greedy replies to those prompts with plain decoding, n-gram or DFlash, in BF16 or INT8: median latency of the reply and of a 1-token request after a 1 s warm-up; decode = (n − 1) / (latency(n) − latency(1)) |
| `make_w8a16.py` | a W8A16 compressed-tensors copy of a checkpoint, quantized as TileInfer's `quantize_int8`, for vLLM's INT8 runs (into `models/`, not tracked) |
| `run_all.sh` | vLLM's two configurations at both points, each in `ROUNDS` = 3 fresh processes (~25 min) |
| `profile.sh`, `nsys_export.py` | one engine's replies under Nsight Systems (`--profile` mode of the two drivers), exported to CSV/JSON for plots, and a side-by-side summary |

## Profiles

```sh
bench/ref/profile.sh tileinfer 8k bf16        # ~3 min; ENGINE POINT WEIGHTS [GEN] [-- DRIVER ARGS]
bench/ref/profile.sh vllm 8k bf16             # ~4 min
python3 bench/ref/nsys_export.py bench/ref/results/profiles/{tileinfer,vllm}-8k-wbf16.nsys-rep
```

Each prompt is warmed up, then one reply of GEN tokens (128 at 1k, 256 at 8k) per prompt is
traced, kernels of CUDA graphs included. Into `results/profiles/NAME/`: `kernels.csv` (one
row per kernel, memcpy and memset, with its step, part and category), `steps.csv` (one row per
engine step: GPU busy and idle time, time by part and by kernel category) and `summary.json`
(means per step, top kernels), with the same columns for both engines, for plots. A step is
a target pass (prefill or verify) and what follows it up to the next (drafter, sampling). The
parts come from NVTX ranges: TileInfer's own, and in vLLM ranges `vllm_spec.py --profile` wraps
around the model runner's `execute_model`, `sample` (which includes the lm_head in vLLM) and
`DFlashSpeculator.propose`. `TAG=NAME` adds a suffix to the output name, e.g. for
`-- --prefix-caching` or `-- --attention-backend TRITON_ATTN`.

Under nsys, vLLM needs three workarounds, all in place: its engine core is forked without
`exec` (`--trace-fork-before-exec=true`), so its `cudaProfilerStart` does not open a capture
range and the whole run is traced (the export keeps the `request NAME` ranges); py-cpuinfo's
`platform.architecture()` runs `file`, which hangs with nsys injected into its seccomp sandbox
(`vllm_spec.py` stubs it); and processes it leaves behind would keep nsys waiting
(`--wait=primary`). nsys ships with Nsight Compute
(`/usr/local/NVIDIA-Nsight-Compute*/host/target-linux-x64/nsys`); `NSYS=` overrides it.

The TileInfer side, with the same arguments:

```sh
julia --project=bench bench/ref/tileinfer_spec.jl Qwen/Qwen3-4B dflash 15 bf16 bf16   # and int8 bf16
julia --project=bench bench/ref/tileinfer_spec.jl Qwen/Qwen3-4B dflash 15 bf16 bf16 2048 5 spec_prompts_8k.json
```

Setup (the environment is not tracked; `requirements.txt` pins the one the documented results came from):

```sh
uv venv --python 3.12 bench/ref/.venv
uv pip install --python bench/ref/.venv/bin/python -r bench/ref/requirements.txt
bench/ref/.venv/bin/python bench/ref/make_w8a16.py Qwen/Qwen3-4B
bench/ref/run_all.sh
```

vLLM's torch.compile finds `nvcc` in the pip-installed CUDA toolkit (no system toolkit
needed), and FlashInfer's sampler is disabled because its JIT build fails against it.
The vLLM scripts run offline from the Hugging Face cache, so pull the models first
(`julia --project -m TileInfer pull ...`, or `huggingface-cli download`).
