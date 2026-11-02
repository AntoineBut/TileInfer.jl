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
