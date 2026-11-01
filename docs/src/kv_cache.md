# KV cache

```@meta
CurrentModule = TileInfer
```

Every token a model has seen leaves a key and a value vector per layer and KV head, and
each later token's attention reads all of them. The KV cache keeps them on the GPU. TileInfer
stores it the way vLLM does, in fixed-size pages: a sequence's positions need not be
contiguous in memory, so it can grow, shrink and be rolled back without moving any data.

Code: `src/kv/block_manager.jl` (pages and block tables, on the host) and
`src/kv/paged_cache.jl` (the page pools, on the device).

## Pages and block tables

A [`PagedKVCache`](@ref) holds, for every layer, one pool of keys and one pool of values, each
a device array of shape `(head_dim, page_size, kv_heads, num_pages)`. A page holds
`page_size` consecutive positions of one sequence (64 by default).

A [`Sequence`](@ref) owns a **block table**: the list, in order, of the pages that hold its
positions. Position p (0-based) lives in page `block_table[p ÷ page_size + 1]`, at offset
`p % page_size`. Pages are handed out by a `BlockManager`, which keeps a stack of free page
ids:

```
pool:         [ page 1 | page 2 | page 3 | page 4 | page 5 | page 6 | … ]
sequence A:   block table [1, 2, 5]   positions 0–63 in page 1, 64–127 in page 2, 128– in page 5
free pages:   [3, 4, 6, …]
```

The steps never see the block table on the host. [`prepare_step!`](@ref) computes, for each
new token, its **slot** (a flat index into the pool: page and offset) and uploads the slots
and the block table into the step's buffers; the fused attention-input kernel writes the new
keys and values at their slots ([`store_kv!`](@ref)), and the attention kernels walk the
block table page by page.

Two details keep this compatible with CUDA graphs and cuTile's compilation:

- **The block table on the device has a fixed length**: every page of the pool, zero-padded,
  rounded up to a multiple of 16. cuTile specializes kernels on the divisibility of array
  sizes, so a table that grew with the sequence would recompile the attention kernels at page
  boundaries; a captured graph also needs its arrays to keep their size.
- **Per-step values live in device buffers**, including the position of the first new token,
  so a graph captured at one length stays valid as the sequence grows.

## Growing, rolling back, releasing

- **`reserve!`** gives a sequence pages for its next n positions before a step, and throws an
  [`OutOfPagesError`](@ref) when the pool is exhausted.
- **[`truncate!`](@ref)** drops the positions after the first n and returns the pages that
  are no longer needed. Nothing is copied or cleared: later steps overwrite the stale slots.
  Speculative decoding uses it to drop the keys and values of rejected drafts after every
  verify step, so rollback has to be free.
- **[`release!`](@ref)** returns all of a sequence's pages.

Even with one sequence, paging pays for itself: a chat can grow without reallocating the
cache, rollback is a host-side list operation, and the attention kernels are the paged
kernels a multi-sequence engine would need. A draft model or a DFlash drafter already runs
a second sequence, on a cache of its own next to the target's.

## Prefix reuse in chat

A [`ChatSession`](@ref) keeps one sequence for the whole conversation. Each turn it renders
the entire conversation with the chat template, compares the tokens with those already in the
cache, truncates the sequence to the longest common prefix, and runs only the rest. Usually
that prefix is everything up to the new user message, so a turn costs about as much as its own
tokens.

Templates can rewrite history. Qwen3's drops the reasoning (`<think>…</think>`) of earlier
assistant turns, so after a reasoning reply the common prefix ends where that reply started,
and the reply is prefilled again without its reasoning. That is the format the model was
trained on, so the session follows it rather than keeping stale tokens.

## Sizing

A cache is sized in positions when it is created: [`PagedKVCache`](@ref)`(model; max_tokens)`
allocates `cld(max_tokens, page_size)` pages for every layer. [`generate`](@ref) sizes it
for the prompt plus the token budget, a chat session for its maximum context, and the CLI for
`--max-context` (4,096 by default).

The memory per cached position is `2 × layers × kv_heads × head_dim` elements:

| Model | Per position, BFloat16 | Per position, INT8 | 4,096 positions, BFloat16 | 32,768 positions, BFloat16 |
|---|---|---|---|---|
| Qwen3-0.6B (28 layers, 8 KV heads of 128) | 112 KiB | 58 KiB | 0.47 GB | 3.8 GB |
| Qwen3-4B (36 layers, 8 KV heads of 128) | 144 KiB | 74 KiB | 0.60 GB | 4.8 GB |

At long contexts the cache rivals the weights: a Qwen3-4B decode step at 32k positions reads
~4.8 GB of keys and values against 8 GB of weights.

## INT8 cache

With `kv_quantize = :int8` ([`generate`](@ref), [`ChatSession`](@ref)) or `--kv-int8`, the
pools hold Int8 values with one Float32 scale per cached vector (one KV head at one position),
[`Int8Pages`](@ref): `x ≈ scale × q`, symmetric, with `scale = max |x| / 127`. A vector is
quantized once, when its position is written, so scales never need updating. Attention does
not dequantize the cache: a key's scale multiplies its row of scores, and a value's scale
its probability before the PV product (see [Attention kernels](attention.md#INT8-cache)).
That is 132 bytes per vector instead of 256: about half the memory, and half the bytes decode
attention reads.

**Smoothing the keys.** A few key channels are up to ~100× larger than the others (in
Qwen3-0.6B's first layer), which leaves one scale per vector with few levels for the rest.
In Qwen3 and Apertus these channels come from the per-channel weight w of the key norm
(k = rope(rmsnorm(k) · w)), so the cache stores k / c, with c = max(|wᵢ|, |wⱼ|) for each
rotary pair (i, j): the same factor on both channels a rotation mixes, so it commutes with
RoPE. Multiplying the queries by c leaves the scores unchanged (qᵀk = (c q)ᵀ(k / c)). This
needs no calibration and costs one multiply per query tile. On Qwen3-0.6B it cuts the mean KL
divergence to a BF16 cache from 0.030 to 0.005 on a chat prompt.

**Speed.** On Qwen3-4B at 32,768 positions, decode runs at 60.9 tok/s with the INT8 cache
against 51.3 with the BF16 one (1.19×); at 1,024 positions, where attention is ~4 % of a step,
both are the same.

## Not built yet

- **Prefix caching across sequences**: hash full pages of a token prefix and share them
  between sequences with reference counts, as vLLM does. It needs several sequences in flight
  first.
- **Sizing from free memory**: allocate the pages that fit after loading the weights (vLLM's
  `gpu_memory_utilization`) instead of an explicit position count.
