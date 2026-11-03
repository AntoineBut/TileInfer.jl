# Comparison with vLLM

TileInfer against vLLM 0.31 on the same machine (RTX 4080 SUPER), in the setup one would
run for real: Qwen3-4B with DFlash speculative decoding (k = 15), with BF16 weights and
with INT8 weights. What each of TileInfer's features gains on its own is in
[Results by feature](performance.md).

## Setup

- **vLLM**: CUDA graphs and torch.compile, one sequence at a time, prefix caching off. Its
  INT8 runs use a W8A16 compressed-tensors copy of the checkpoint quantized as TileInfer's
  `quantize_int8` (every linear and the lm_head), on its Marlin kernels.
- **Workload**: the two tasks of `bench/prompts.jl` on Julia source code, renaming a type
  throughout it (which copies the input) and explaining it (open text), greedy, stopping at
  EOS, with the same token ids for both engines. Two points:
  - **1k**: 1,024-token prompts (1,021 and 1,018 tokens), 128 tokens, the reference point of
    [Results by feature](performance.md);
  - **8k**: 8,192-token prompts (8,181 and 8,178 tokens), 2,048 tokens.
- **Metric**: per prompt, the median latency of the reply and of a 1-token request
  ("prefill") over 5 trials after a 1 s warm-up; decode = (n − 1) / (latency(n) −
  latency(1)). Each configuration runs in fresh processes (3 at 1k, 2 at 8k), which agree
  within 1 %.
- **Scripts**: `bench/ref/` (`vllm_spec.py`, `tileinfer_spec.jl`, `run_all.sh`; setup in its
  README).

## 1k prompts

| Qwen3-4B, DFlash k = 15 | vLLM | TileInfer | tokens per step, vLLM / TileInfer |
|---|---|---|---|
| prefill (1-token request) | 100 ms | **86 ms** (INT8: 91) | |
| BF16, edit | 292 tok/s | **421 tok/s** | 4.9 / 7.1 |
| BF16, explain | **188 tok/s** | 175 tok/s | 3.2 / 2.9 |
| INT8 weights, edit | **975 tok/s** | 906 tok/s | 9.8 / 9.8 |
| INT8 weights, explain | 281 tok/s | 282 tok/s | 3.0 / 3.0 |

The replies start with the same tokens, but in BF16 the two engines' drafts are accepted
differently (presumably rounding differences between the engines' kernels; not
investigated): which engine wins follows the tokens per step. With INT8 weights both take
the same 13 and 43 steps; vLLM's step is then 7 % cheaper on the edit and the same on the
explanation. INT8 also changes the replies after a few tokens, which happens to make the
edit's drafts longer (9.8 tokens per step against 7.1).

## 8k prompts

The explanations end before 2,048 tokens, at their EOS. 8k is about half the longest prompt
TileInfer takes with DFlash in BF16 (~13k tokens): it prefills the prompt in one step, and
the drafter then reads its hidden states in one step of its own, ~400 KB of buffers per
prompt token next to 167 KB of KV caches, where the caches alone would fit ~32k positions.
vLLM prefills in chunks.

| Qwen3-4B, DFlash k = 15 | vLLM | TileInfer | tokens per step, vLLM / TileInfer |
|---|---|---|---|
| prefill (1-token request), BF16 / INT8 | 0.88 / 0.92 s | 0.88 / 0.94 s | |
| BF16, edit (2,048 tokens) | 69.5 tok/s | **95.5 tok/s (1.37×)** | 1.87 / 1.87 |
| BF16, explain (1,311 / 1,223 tokens) | 102 tok/s | **126 tok/s** | 2.70 / 2.50 |
| INT8 weights, edit (2,048 tokens) | 91.6 tok/s | **140 tok/s (1.53×)** | 1.88 / 1.88 |
| INT8 weights, explain (1,076 / 1,328 tokens) | 125 tok/s | **192 tok/s** | 2.48 / 2.57 |

- **Drafts are accepted far less**: the edit takes 1.87 tokens per step at 8k against 7.1 at
  1k, on both engines, with the same tokens (1,093 and 1,096 steps for 2,048 tokens): it
  comes from the drafter, not from an engine (why is not investigated).
- **TileInfer's steps are cheaper, by their attention**: with the same steps, a step (drafter
  pass and verify) costs 19.6 ms against vLLM's 26.9 in BF16 (1.37×; INT8 1.53×), where a
  plain decode step at 1k costs 12.7. Profiled (`bench/ref/profile.sh`, 256 tokens after the
  8k prompts), the GEMMs cost the same (15.1 ms against 15.4 ms of GPU time per step) and
  neither engine leaves the GPU idle (busy 98.5 % and 99.8 % of the step), but the verify
  step's attention takes 9.0 ms in vLLM against 2.1 ms: its FlashAttention
  `flash_fwd_splitkv_kernel` runs one block per query head (grid 1 × 1 × 32, no split of the
  8k positions over the 80 SMs), 250 µs per layer (~135 GB/s), where TileInfer's decode
  attention splits the positions in 8 (64 blocks, 56 µs per layer, ~600 GB/s). The gap grows
  with the context. Prefix caching plays no part: it is off in these runs. On Ada vLLM runs
  FlashAttention 2, whose varlen path (the only one vLLM calls) takes no split count; its
  Triton backend (`--attention-backend TRITON_ATTN`) splits only one-token steps, and costs the
  same here (10.3 ms of attention per step); FlashInfer's JIT build fails in this environment
  (nvcc 13.4 against CUDA 13.0 headers). More splits do not help TileInfer's kernel either:
  16, 32 or 64 splits are as fast or slower than 8 at 1k to 16k positions (the Float32
  partials of 64 query rows per split add traffic), and 4 are 1.6× slower.
- **Prefill ties** (~9.3k tok/s in BF16, against 12k at 1k: attention grows with the prompt).

## What is left to gain

| Where | Measured | Lever |
|---|---|---|
| DFlash step at 1k with INT8 weights | vLLM's is 7 % cheaper at the same tokens per step | profile the drafter pass and the verify step |
| Long prompts | one prefill step of the whole prompt, then one drafter step over it: ~13k-token prompts at most with DFlash in BF16 | chunked prefill, drafter included |
| DFlash drafter's attention at 8k | `attention_prefill_kernel` runs one block per head (grid 1 × 32), 232 µs per drafter layer, 1.1 ms of the 4.0 ms drafter pass | split the context positions, as the decode attention does |
