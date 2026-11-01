# Attention kernels

```@meta
CurrentModule = TileInfer
```

Attention is the one op whose cost grows with the context, and the one that has to read the
[paged KV cache](kv_cache.md). TileInfer runs it with a family of FlashAttention-style cuTile
kernels in `src/ops/tile/attention.jl`, behind [`attention!`](@ref). This page explains how
they are organized and why, and what each choice measured.

## One kernel family, three uses

Every step attends the queries of its new tokens to all cached positions of the sequence,
causally. What changes between uses is the number of queries per step, n:

| Use | n | Kernel |
|---|---|---|
| prefill (the prompt) | tens to thousands | prefill kernel: one block per 32-query tile and head |
| decode | 1 | split-KV decode kernel + combine kernel |
| speculative verify | 2–16 | the same decode kernels |

The decode kernels need the number of query heads per KV head to be a power of two, as in
every model tested; otherwise every step runs the prefill kernel.
| DFlash drafting | new context positions + 16, not causal | prefill kernel, `causal = false` |

All of them use the same structure as FlashAttention-2: a block holds a tile of queries in
registers, walks the keys and values one tile at a time, and keeps a running maximum and sum
of the softmax (the online softmax), so the scores are never written to memory. Scores are
computed in Float32 with `exp2` (the scale is pre-multiplied by log₂ e); probabilities are
rounded to the model's element type for the PV product on the tensor cores.

## Walking the block table

The KV loop walks the sequence's block table, one page per step. A page of 64 positions of
one KV head is a dense `(head_dim, 64)` tile of the pool, so each step is one scalar load of
the page id and one aligned tile load each for keys and values:

```julia
for j in pages
    page = block_table[j]                         # scalar load from the device
    K = ct.load(k_pool; index = (1, page_column(page, kv_head)), shape = (head_dim, PAGE))
    V = ct.load(v_pool; …)
    # scores, online softmax update, accumulate V P
end
```

Positions come from device buffers, never from launch arguments: query t of a step is at
position `start + t`, where `start` is read from the device. A captured CUDA graph therefore
stays valid as the sequence grows. Only pages that overlap the diagonal are masked.

The pools are passed as 2-D views, `(head_dim, page_size × kv_heads × pages)`, rather than
4-D arrays: each array argument costs 1 + 2N kernel parameters, and from 32 parameters on,
CUDA.jl's launch takes a slow generic path (~15 µs and hundreds of allocations per launch).

## Prefill

The prefill kernel follows cuTile's FMHA example: grid (query tiles, query heads), one tile of
32 queries per block. 32-row tiles beat 64 (1.3×) and 128 at 512 tokens: on 80 SMs they give
256 blocks instead of 128, which keeps every SM busy through the causal triangle. At 512
tokens the kernel runs at ~36 TFLOP/s, 42× the eager reference.

## Decode: GQA packing and split-KV

At decode a block would hold one query row: nearly empty tiles, and only `heads` blocks to
fill 80 SMs. Two standard fixes, both from flash-decoding:

- **GQA packing.** The query heads that share a KV head (4 for Qwen3-4B: 32 query heads, 8 KV
  heads), for all n tokens of the step, form the rows of one query tile, padded to at least 16
  rows. Each page of keys is then loaded once for the whole group.
- **Split-KV.** The pages are divided among a fixed number of splits per KV head, so one
  sequence still launches enough blocks (about two per SM). Each split writes its unnormalized
  partial result (accumulator, running max and sum) to a Float32 workspace, and a small
  combine kernel merges the splits of each query row.

The grid does not depend on the context length: splits without pages write a neutral partial.
That keeps the launch shape fixed, which a CUDA graph requires, at the cost of a combine
that always merges every split. Steps of 9–16 tokens (DFlash's verify block) use 8 splits
instead of 32: with 32, the combine set a floor that lost to the prefill kernel below ~450
positions; with 8, the decode kernels win from ~200 positions on (at 2,048 positions:
14 µs against 54 µs per layer on Qwen3-4B).

Padding the packed query tile from 2 to 16 rows, with two blocks per SM, cut the decode kernel
from 15 to 10 µs at 1,024 positions: tiles with fewer rows spill registers (see
[cuTile quirks](cutile_quirks.md)). At 32k positions the BF16 decode kernel reads the cache at
642 GB/s, 87 % of the bandwidth roofline.

## Fused attention inputs

Between the q/k/v projection and attention sit the optional Qwen3 q/k RMSNorm, RoPE, and the
write of the new keys and values into the cache. [`attention_inputs!`](@ref) does all three
in two launches (queries; keys and values) instead of five or more, reading each head once.
Its results are bit-identical to the separate cuTile ops, and within 1 ulp of the reference.

## INT8 cache

With an [INT8 cache](kv_cache.md#INT8-cache), the kernels read Int8 keys and values with one
Float32 scale per vector and never write a dequantized copy:

- keys and values are converted to BFloat16 in registers for the tensor cores (|q| ≤ 127 is
  exact);
- a key's scale multiplies its row of scores, `s_k (q_kᵀ x)`, and a value's scale its
  probability before the PV product;
- the queries are multiplied by the keys' smoothing factors, which undoes the smoothing in
  the scores.

The first version converted whole `(64, 128)` pages and spilled registers (255 registers,
3.1× the instructions of the BF16 kernel). Taking 32 keys at a time, and keys, values,
queries and accumulator in two halves of the head dimension, brought the decode kernel to 191
registers without spills: 119 µs instead of 150 at 32k positions on Qwen3-4B, 581 GB/s for
half the bytes of the BF16 cache. Quantizing the queries to use Int8 tensor cores
(SageAttention's idea) was as fast but undid what key smoothing gained in accuracy, so it
was not kept.

## Numerics

The kernels keep scores and probabilities in Float32 until the PV product, while the
reference rounds them to BFloat16 as PyTorch's eager attention does. Against exact attention,
the BF16 kernels stay within ~3.5 BFloat16 ulps, closer than the reference (up to 12 ulps);
in Float32 they match exact attention to 1e-5.

## Open questions

- **Several pages per load.** Tile IR ≥ 13.3 can gather several pages in one load through
  sparse views; the kernels use one runtime-indexed load per page, and the alternative is not
  measured yet.
- **Long prompts.** At 4k–16k tokens prefill attention catches up with the GEMMs. GQA packing
  in the prefill kernel (heads of a group sharing key loads) and scheduling the heaviest
  causal blocks first are the candidates.
