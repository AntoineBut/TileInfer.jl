# Comparison with vLLM

TileInfer against vLLM 0.31 on the same machine and the same token ids. Results and their
discussion, with the setup they came from: "Comparison with vLLM" in
[docs/src/performance.md](../../docs/src/performance.md). Runs write their JSON into
`results/<date>/` (not tracked).

| File | What it does |
|---|---|
| `tileinfer_race.jl`, `vllm_race.py` | The race: 512-token prompt, 256 greedy tokens; latency of 1 and of 256 tokens, medians after a 1 s warm-up |
| `alternate.py` | Runs the two races in alternating fresh processes (ABBA, 3 rounds) and pools their trials; `--weights int8`, `--kv int8` |
| `make_w8a16.py` | A W8A16 compressed-tensors copy of a checkpoint, quantized as TileInfer's `quantize_int8`, for vLLM's INT8 runs (into `models/`, not tracked) |
| `spec_prompts.jl` → `spec_prompts.json` | `bench/speculative.jl`'s chat prompts as token ids, shared by both engines |
| `tileinfer_spec.jl`, `vllm_spec.py` | Greedy replies to those prompts with plain decoding, n-gram or DFlash, in BF16 or INT8 |
| `run_all.sh` | Everything above into `results/<date>/` (~2.5 h) |

Setup (the environment is not tracked; `requirements.txt` pins the one the documented results came from):

```sh
uv venv --python 3.12 bench/ref/.venv
uv pip install --python bench/ref/.venv/bin/python -r bench/ref/requirements.txt
bench/ref/.venv/bin/python bench/ref/make_w8a16.py Qwen/Qwen3-0.6B     # and Qwen/Qwen3-4B
bench/ref/run_all.sh
```

vLLM's torch.compile finds `nvcc` in the pip-installed CUDA toolkit (no system toolkit
needed), and FlashInfer's sampler is disabled because its JIT build fails against it.
The vLLM scripts run offline from the Hugging Face cache, so pull the models first
(`julia --project -m TileInfer pull ...`, or `huggingface-cli download`).
