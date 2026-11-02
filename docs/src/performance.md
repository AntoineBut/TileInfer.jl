# Performance

What each feature of the engine gains, on Qwen3-4B in BFloat16 on the RTX 4080 SUPER
(736 GB/s). How the features work is described in [Architecture](architecture.md),
[KV cache](kv_cache.md), [Attention kernels](attention.md) and
[Speculative decoding](speculative.md); the order they were built in is in
[History and roadmap](roadmap.md). The engine against vLLM, at this point and at a longer
context, is on its own page: [Comparison with vLLM](vllm.md).

Qwen3-0.6B gets the same features and runs close to them (409 tok/s decode, a 1,024-token
prefill in 15.4 ms), a little further from its roofline (decode 73 % against 88 %): its
kernels are 5–7× shorter, so the fixed cost of each launch, and of filling and draining the
GPU, weighs more. For the same reason CUDA graphs and fused kernels gain more there.

## Reference point

The numbers on this page are taken at one point, unless a row or a sentence says
otherwise: **a 1,024-token prompt prefilled in one step, then 128 greedy tokens decoded
after it** (positions 1,024 to 1,151).

- **`bench/run.jl --ref --filter 4B/model --seconds 2`**: the model benchmarks of
  `bench/suite.jl`, for the engine (`tile`) and for each configuration of `MODEL_CONFIGS`,
  which switches one feature off (`bench/variants.jl`); `--quantize int8` and `--kv int8`
  for the quantized rows. Random prompt tokens. `prefill1024` is one synchronized
  `forward!` of the prompt; `decode128_ctx1024` is `generate!` of 128 tokens after 1,024
  cached positions, as tok/s. Minimum over the samples.
- **`bench/speculative.jl`**: speculative decoding on two tasks over a Julia source file
  cut to fit 1,024 prompt tokens (1,018 and 1,021 with the chat template,
  `bench/prompts.jl`): renaming a type throughout it, which copies the input, and
  explaining it, which is open text. Decode tok/s from the first new token to the last,
  best of 5.
- **Roofline**: a decode step reads the 8.04 GB of weights and ~160 MB of KV cache, 11.1 ms or
  **90 tok/s**. Prefill does 7.8 TFLOP (7.44 in the GEMMs, 0.32 in attention).

Expect ±3 % between runs (the GPU also drives the desktop).

## Summary

| | Prefill, 1,024 tokens | Decode, 128 tokens |
|---|---|---|
| **BF16** | **85.6 ms**, 12.0k tok/s (91 TFLOP/s) | **78.7 tok/s** (88 % of roofline) |
| INT8 weights | 90.9 ms, 11.3k tok/s | **138.2 tok/s** (1.76×) |
| INT8 weights + INT8 KV cache | 93.0 ms, 11.0k tok/s | 138.7 tok/s |
| Speculative decoding (DFlash, k = 15) | — | 174 tok/s explaining (2.2×), 419 editing (5.3×) |
| *For scale: reference ops (broadcasts and cuBLAS), eager* | *458 ms* | *8.6 tok/s* |

## CUDA graphs

Decode steps (≤ 8 tokens) are captured once and replayed, one launch instead of ~365
(10 kernels per layer, plus 5); pipelined, the host samples a step behind the GPU, so it
enqueues the next step before reading the token. Prefill steps run eagerly in every
configuration.

| Decode, tok/s | fused kernels | one kernel per op |
|---|---|---|
| eager steps | 76.4 | 72.3 |
| replayed from a graph, step by step | 77.9 (1.02×) | 75.2 (1.04×) |
| replayed and pipelined (the engine) | **78.7 (1.03×)** | 75.9 (1.05×) |

A 4B step takes 12.7 ms, long enough for the host to enqueue most launches while earlier
kernels run; graphs remove what is left, 0.25 ms per token, and pipelining another 0.13 ms,
the host's read of the token. On 0.6B (2.5 ms per step), graphs gained 1.22×.

## Fused kernels, FlashAttention and PagedAttention

From one plain op per kernel to the engine's kernels, with eager steps so that each row
changes one thing:

| | Prefill | Decode |
|---|---|---|
| Reference ops (`ref`): a broadcast or cuBLAS call per op; attention gathers the pages into dense K and V and materializes the scores | 458 ms | 8.6 tok/s |
| cuTile kernel per op (GEMV/GEMM, RMSNorm, RoPE, KV write, SwiGLU), same attention (`unfused-refattn-eager`) | 353 ms (1.30×) | 11.2 tok/s (1.30×) |
| + paged flash attention: online softmax over the pages in place, split-KV for decode (`unfused-eager`) | 96.4 ms (**3.7×**) | 72.3 tok/s (**6.5×**) |
| + fused kernels (`tile-eager`) | 85.6 ms (**1.13×**) | 76.4 tok/s (**1.06×**) |

The reference attention runs about 15 kernels per KV head and layer and copies the cache
into dense matrices every step; for prefill it writes the 1,024 × 1,024 score matrix of every
head. The attention kernels read each page once from the pool through the block table.

The fused kernels, bit-identical to the chains of kernels they replace:
- attention inputs: q/k RMSNorm, RoPE and the KV cache write straight from the fused q/k/v
  projection, 2 kernels instead of 8 (3 copies, 2 norms, 2 RoPE, 1 write);
- the residual add with the next RMSNorm;
- SwiGLU in the epilogue of the decode `gate_up` GEMV, and Float32 logits from the lm_head GEMV.

With graphs and pipelining, the fused kernels still gain 1.04× on decode (75.9 → 78.7 tok/s):
~0.45 ms per token of memory traffic and kernel tails.

Not kept: the attention combine fused into the split-KV kernel (the last split of a head
merges, through an acquire-release counter): 2 % slower than kernel + combine.

## Prefill GEMMs

The GEMMs are ~87 % of a prefill step's GPU time. `gemm_config` picks, per shape, the tile among 6
that minimizes waves × (TM TN)^0.7 (fitted to a sweep of 30 tiles on both models' layers at
n = 2…1,024: within 2 % of the best tile on average), and grids of ≥ 2 blocks per SM run
two blocks per SM (`gemm_occupancy`, 1.06× on a 512-token prefill). Against cuBLAS for every linear layer
(`gemmEx!`, Float32 accumulation):

| | cuTile | cuBLAS |
|---|---|---|
| Prefill | 85.6 ms (91 TFLOP/s) | **82.8 ms** (94 TFLOP/s) |
| Decode (GEMVs) | 78.7 tok/s | 78.7 tok/s |

Not kept: split-K (Float32 partials and a reduce kernel) for n = 2…64 never beat the best
direct tile; the `latency` hint on the GEMM loads.

## INT8 weights

Weight-only INT8 with one Float32 scale per output channel (`quantize = :int8`), the
embedding in BF16. Mean KL divergence to the BF16 model over the fixture prompts 0.009, top-1
agreement 49/50.

| | BF16 | INT8 weights |
|---|---|---|
| Prefill | 85.6 ms | 90.9 ms (0.94×) |
| Decode | 78.7 tok/s | **138.2 tok/s (1.76×)** |

Decode reads half the bytes. The INT8 GEMVs want half BF16's tile width (TK × TM ≈ 4,096
Int8 elements). Prefill stays compute-bound, and the Int8 weight tile is converted in
registers rather than staged through shared memory like the BF16 operands (1.6× the
instructions of the BF16 GEMM); `gemm_int8_config` has its own tiles, which took INT8
prefill on short prompts from 8–11 % slower than BF16 to 1.5× faster (64 tokens) but leave
it 6 % behind at 1,024.

INT8 also fits models that BF16 does not: Apertus 1.5 8B (16.1 GB in BF16) runs in 8.6 GB
at ~80 tok/s.

## INT8 KV cache

One Int8 scale per cached vector, keys smoothed by the key norm's weight
([KV cache](kv_cache.md#INT8-cache)), which brings the mean KL to a BF16 cache from 0.030 to
0.0051 (Qwen3-0.6B, chat prompt).

| | BF16 cache | INT8 cache |
|---|---|---|
| Prefill | 85.6 ms | 88.1 ms (0.97×) |
| Decode, BF16 weights | 78.7 tok/s | 78.8 tok/s |
| Decode, INT8 weights | 138.2 tok/s | 138.7 tok/s |
| *Long context:* decode after 8,192 positions, BF16 weights | 70.1 tok/s | 73.3 tok/s (1.05×) |
| decode after 8,192 positions, INT8 weights | 113.9 tok/s | 122.4 tok/s (1.07×) |
| decode after 32,768 positions, BF16 weights | 51.4 tok/s | **60.9 tok/s (1.18×)** |
| decode after 32,768 positions, INT8 weights | 71.6 tok/s | **91.2 tok/s (1.27×)** |

At 1,024 positions the cache is 2 % of the bytes a step reads, so halving it gains nothing:
the INT8 cache pays off as the cache grows next to the weights (BF16 cache: 1.2 GB at 8k,
4.8 GB at 32k, against 8 GB of BF16 weights or 4 GB of INT8 ones), and fits more context.
The long-context rows are `bench/run.jl --context 8192` and `--context 32768` (the
decode benchmark after that many cached positions). To keep the attention kernels
spill-free with the conversion, they load 32-key tiles and each half of the head dimension
separately (decode attention at 32k positions: 150 → 119 µs).

## Speculative decoding

Each step verifies k drafted tokens in one forward pass of k + 1 tokens, whose cost is close
to a 1-token step's, and keeps the longest accepted prefix: the tokens of plain decoding, up to BF16
rounding. Best k of each proposer:

| Decode, tok/s | edit (copies the input) | explain (open text) |
|---|---|---|
| plain | 78.7 | 78.7 |
| n-gram lookup in the context | 396 (5.03×, k = 7), 5.3 tokens per step | 83 (1.05×) |
| Qwen3-0.6B draft model | 235 (2.99×, k = 7), 7.1 | 113 (1.44×, k = 2), 2.1 |
| DFlash block drafter | **419 (5.33×, k = 15)**, 7.1 | **174 (2.22×, k = 15)**, 2.9 |

DFlash drafts a block of 15 tokens in one pass of a small model over the target's hidden
states; a longer block costs nothing more to draft and little more to verify, so the
whole block wins even on open text, where per-draft acceptance falls from 0.58 at k = 2 to
0.14 at k = 15 (1.68× and 2.22×). With sampling, drafts are verified by the p/q rule; the
picture is the same ([Speculative decoding](speculative.md#Results)).

**Attention for verify steps.** Steps of 2–16 tokens (verify steps, and DFlash's pass over
its block) run the split-KV decode kernels, with 8 splits from 9 tokens on, instead of the
prefill kernel, which runs one block per head over every page (16 tokens at 1,024 positions:
9.5 µs against 27 per layer):

| Decode, tok/s | prefill kernel | decode kernels |
|---|---|---|
| edit, DFlash k = 15 | 408 | 419 (1.03×) |
| edit, n-gram k = 4 | 288 | 299 (1.04×) |
| explain, DFlash k = 15 | 165 | 174 (1.05×) |

The gain grows with the context: on 2.2k-token prompts it was 1.08–1.10×.

## What is left to gain

| Where | Measured | Lever |
|---|---|---|
| Decode | 88 % of roofline; layer GEMVs at 90–93 % of DRAM bandwidth against the lm_head's 95 %; small kernels (norms, attention inputs) ~3 % of the step | fewer launches (norms in GEMV epilogues) |
| Prefill GEMMs | 0.97× cuBLAS (85.6 against 82.8 ms) | larger tiles or split-K for under-filled grids |
| INT8 prefill | 0.94× BF16 | a weight layout prepacked for the conversion, as Marlin does |
| Prefill, small kernels | `q_inputs_kernel` runs one block per (token, head); `silu_mul` is a kernel of its own | several tokens per block; SwiGLU in the GEMM epilogue |
| Speculative decoding | the best k depends on the prompt and the context | adaptive k, tree verification |
| Long prompts | one prefill step of the whole prompt, then one drafter step over it: ~13k-token prompts at most with DFlash in BF16, where the KV caches would hold ~32k positions | chunked prefill, drafter included |

Against vLLM: [What is left to gain](vllm.md#What-is-left-to-gain) there.
