# Performance

What each feature of the engine gains, measured on the RTX 4080 SUPER (736 GB/s), in
BFloat16 unless noted, on Qwen3-0.6B and Qwen3-4B. How the features work is described in
[Architecture](architecture.md), [KV cache](kv_cache.md), [Attention kernels](attention.md)
and [Speculative decoding](speculative.md); the order they were built in is in
[History and roadmap](roadmap.md). Variants that were measured and not kept are listed
under each feature, with the measurement, so they are not retried blindly. What is left to
gain is collected at the end.

## How it is measured

- **`bench/run.jl`** (see CLAUDE.md): minimum time over samples, `--seconds 2`; expect
  ±3–5 % between runs at the model level (the GPU also drives the desktop). Model benchmarks:
  - `decode_ctx1024` / `decode_graph_ctx1024`: one decode step after 1,023 cached positions,
    eager or replayed from a CUDA graph, synchronized;
  - `generate64[_unpipelined|_sampled]_ctx1024`: 64 tokens after 1,023 positions, per token;
    pipelined greedy, greedy step by step, and the model's recommended sampling (Qwen3:
    temperature 0.6, top-k 20, top-p 0.95), pipelined;
  - `prefill64`, `prefill512`: a prompt in one step.
  The % of roofline counts every weight array once, the embedding table included.
- **`bench/breakdown.jl`**: GPU time per kernel inside real steps (decode replayed from
  graphs), by role (see [Profiling](profiling.md)).
- **`bench/speculative.jl`**: 256-token chat replies to fixed prompts; decode tok/s from the
  first new token to the last; best of 3.
- **The vLLM race** (`bench/ref/`, see its README): fresh processes alternating the two
  engines, medians of every trial after a 1 s warm-up. Without the warm-up, the GPU runs the
  first ~0.2 s at 900–1,600 MHz instead of 2,775 MHz, and a 10 ms request takes 16–40 ms.
- **Decode roofline at 1,024 positions**: Qwen3-0.6B reads 1.19 GB of weights + 117 MB of
  KV cache per token, 1.78 ms (~560 tok/s); Qwen3-4B 8.04 GB + 151 MB, 11.1 ms (~90 tok/s).

## Summary

`bench/run.jl --seconds 2`, decode after 1,023 cached positions, per token of 64:

| | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| Decode, greedy pipelined | 2.44 ms, **409 tok/s** (73 % of roofline) | 12.70 ms, **78.7 tok/s** (88 %) |
| Decode, sampled, pipelined | 2.45 ms, 408 tok/s | 12.72 ms, 78.6 tok/s |
| Decode, greedy step by step / graph step / eager step | 2.57 / 2.51 / 2.77 ms | 12.82 / 12.77 / 13.06 ms |
| Prefill 64 tokens | 2.73 ms, 23k tok/s | 14.5 ms, 4.4k tok/s |
| Prefill 512 tokens | 8.82 ms, **58k tok/s** (69 TFLOP/s) | 45.0 ms, **11.4k tok/s** (91 TFLOP/s) |
| Decode, INT8 weights | 1.85 ms, **542 tok/s** | 7.21 ms, **139 tok/s** |
| Prefill 64 / 512, INT8 weights | 2.49 / 8.69 ms (26k / 59k tok/s) | 9.49 / 46.1 ms (6.7k / 11.1k tok/s) |
| Speculative decoding (DFlash, k = 15, greedy) | — | 1.9× to 3.0× on open prompts, 8.5× on a code edit |

Against vLLM 0.31 on the same machine: 3.3 % faster end to end on Qwen3-0.6B, a tie on
Qwen3-4B (see [the comparison](#Comparison-with-vLLM)).

## Where the time goes

`bench/breakdown.jl`, µs per step and share of the GPU-busy time.

**Decode** (graph replay, 1,024 positions):

| Kernels | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| Layer GEMVs: qkv / o / gate_up + SwiGLU / down | 385 / 207 / 566 / 288 = 1,446 (59.1 %) | 1,683 / 1,139 / 5,346 / 2,626 = 10,793 (84.8 %) |
| lm_head GEMV | 452 (18.4 %) | 1,114 (8.8 %) |
| Attention decode + combine | 250 + 58 = 308 (12.6 %) | 322 + 75 = 396 (3.1 %) |
| Attention inputs (q, kv), add + RMSNorm ×2 | 63 + 64 + 106 = 233 (9.5 %) | 100 + 86 + 223 = 409 (3.2 %) |
| Embedding, first norm, sampling | 10 (0.4 %) | 11 (0.1 %) |
| **GPU busy per token** | **2,449** (99 % of the 2,472 µs period) | **12,723** (99.9 % of 12,738) |
| Roofline (weights + KV once) | 1,780 | 11,130 |

GEMV bandwidth inside the step: 0.6B qkv 83 %, o 77 %, gate_up + SwiGLU 85 %, down 83 %,
lm_head 94 %; 4B 91 / 90 / 91 / 93 / 95 %. A step is 10 kernels per layer plus 5, i.e. 285
(0.6B) and 365 (4B) kernels per token.

**Prefill** (eager steps):

| Kernels | 0.6B, 64 tokens | 0.6B, 512 | 4B, 64 | 4B, 512 |
|---|---|---|---|---|
| GEMMs: qkv + o + gate_up + down | 1,657 (64.5 %) | 6,292 (73.1 %) | 12,268 (86.4 %) | 39,327 (87.7 %) |
| of which gate_up | 599 (23.3 %) | 2,270 (26.4 %) | 6,052 (42.6 %) | 18,613 (41.5 %) |
| lm_head GEMV (last token) | 454 (17.7 %) | 484 (5.6 %) | 1,123 (7.9 %) | 1,141 (2.5 %) |
| Attention prefill | 96 (3.7 %) | 751 (8.7 %) | 133 (0.9 %) | 1,540 (3.4 %) |
| Attention inputs (q, kv) | 193 (7.5 %) | 729 (8.5 %) | 330 (2.3 %) | 1,460 (3.3 %) |
| add + RMSNorm ×2, silu_mul | 165 (6.4 %) | 345 (4.0 %) | 346 (2.4 %) | 1,360 (3.0 %) |
| **GPU busy** (`bench/run.jl` step) | **2,568** (2.73 ms) | **8,607** (8.82 ms) | **14,205** (14.50 ms) | **44,840** (45.0 ms) |

- At 64 tokens the GEMMs read the weights at 76–82 % of DRAM bandwidth, against 83–95 % for
  the decode GEMVs that read the same bytes.
- `q_inputs_kernel` runs one block per (token, head): at 512 tokens on 0.6B, 8,192 blocks of
  one 128-element column each, 16 µs per launch. With `kv_inputs_kernel`, the attention
  inputs cost as much as attention itself there (729 against 751 µs).
- On 0.6B a 64-token prefill keeps the GPU busy 88 % of the step's span: the host cannot
  enqueue 283 short kernels as fast as they run. 4B kernels are long enough (98 %).

## The decode step: fusion, graphs and pipelining

What each step of the decode path gained when it was added, on Qwen3-0.6B, against the
step without it:

| Feature | Decode | Other |
|---|---|---|
| Static step buffers, no allocations | eager step 3.80 → 3.49 ms (1.09×) | prefill 512 1.36× |
| Decode replayed from a CUDA graph | 3.49 → 2.87 ms (1.22×) | 539 launches per token become one |
| Fused attention inputs: q/k RMSNorm + RoPE + KV write from `qkv`, 2 kernels instead of 8 launches | graph step 2.89 → 2.58 ms (**1.12×**) | eager 1.29×, prefill 1.26× |
| Residual add + RMSNorm in one kernel | 2.58 → 2.48 ms (1.04×) | prefill 1.04× |
| SwiGLU epilogue in the gate_up GEMV, Float32 logits from the lm_head GEMV | 2.48 → 2.46 ms (1.01×) | gate_up 21.6 µs against 22.8 for GEMV + `silu_mul` |
| Pipelined decode, greedy token sampled into the next step's inputs | 2.62 → 2.47 ms per token (1.06×) | GPU busy 95 % of wall time under the tracer, from 86 % |

The fused kernels are bit-identical to the unfused cuTile ops. A first SwiGLU GEMV with
half-width column tiles (to keep registers per thread equal) was 6 % slower than unfused.

Not kept:

| Variant | Measured |
|---|---|
| GEMV tile sweep per decode shape (TK 256–2,048, TM 1–16) | every configuration within ~3 % of the default |
| Attention combine fused into the split-KV kernel (the last split of a head merges, through an acquire-release counter) | 322 µs per step against 315 for kernel + combine |
| V tile loaded before the QK product in decode attention | 252 µs per step either way |
| 80 instead of 160 target blocks for the decode splits | combine 63 → 47 µs per step (0.6 %), but fewer splits for long contexts |

## Sampling on the device

Temperature, top-k and top-p sampling inside the captured step, pipelined like greedy
(the design is in [Architecture](architecture.md#Model-runner)):

| | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| Decode, sampled: host sampling → device sampling | 3.32 → **2.46 ms per token (1.35×)** | 13.34 → **12.72 ms (1.05×)** |
| Sampled against greedy, both pipelined | +0.5 % | +0.1 % |

The sampling kernels take 17 µs per token at top-k = 20 and 5.6 µs for greedy (top-1). Their
variants, chunk kernel + final block, inside the 0.6B graph:

| Version | top-k 20 (µs) | top-1 (µs) | Kept |
|---|---|---|---|
| repeated `maximum` + `argmax` over values and ids | 10.0 + 11.0 = 21.0 | 2.3 + 3.3 | |
| `argmax` over ids only, values gathered from the logits | 8.4 + 10.8 = 19.2 | 2.2 + 4.1 | |
| `maximum`, then `minimum` of the positions holding it | 8.9 + 11.2 = 20.1 | — | no |
| argmax reducer without NaN checks (`FirstMax`) | **7.4 + 9.8 = 17.2** | 2.2 + 4.0 | **yes** |
| three levels (1,024-logit chunks, 16-list merge, final) | 6.4 + 5.4 + 7.2 = 19.0 | — | no |
| 8,192-logit chunks, 2,048-candidate final tile | 9.7 + 8.1 = 17.8 | — | no |

Under `ncu` both kernels issue an instruction every 3–4 cycles at ~8 % achieved occupancy:
they are latency-bound, and their cost is the number of sequential block-wide reductions
(see [cuTile design choices](cutile_design.md#Few-reductions-inside-loops)).

## Prefill GEMMs

**Per-shape tiles** (`gemm_config`): the tile minimizing waves × (TM TN)^0.7 among 6 shapes.
Fitted to a sweep of 30 tile shapes × 8 layers (both models) × n = 2…1,024: within 2 % of the
best tile on average (worst 1.18×), 1.22× the previous rule (64 or 128 rows × the next power
of two ≥ n columns). On 0.6B at n = 64: o 1.77×, down 1.98×, prefill64 3.62 → 3.00 ms.

**Two blocks per SM** (`gemm_occupancy`): `occupancy = 2` on grids of ≥ 160 blocks. Applied
everywhere, it slowed the small grids down:

| | everywhere | ≥ 2 blocks per SM only (kept) |
|---|---|---|
| 4B gate_up / qkv / SwiGLU, n = 512 (608 / 192 / 608 blocks) | 1.11× / 1.08× / 1.11× | 1.12× / 1.10× / 1.11× |
| 4B o / down, n = 512 (80 blocks) | 0.97× / 0.96× | 0.99× / 1.01× |
| 0.6B o / down, n = 512 (32 blocks) | **0.86× / 0.89×** | 1.00× / 0.95× |
| 0.6B gate_up / SwiGLU, n = 64 (192 blocks) | 1.22× / 1.19× | 1.22× / 1.19× |
| Prefill 512, 4B / 0.6B | 1.05× / 1.01× | **1.06× / 1.02×** |

Against cuBLAS (`bench/run.jl --ref`, speedup of the cuTile GEMM):

| n | Qwen3-0.6B: qkv / o / gate_up / down | Qwen3-4B: qkv / o / gate_up / down |
|---|---|---|
| 64 | 1.07 / 1.43 / 1.03 / 1.45× | 1.04 / 1.04 / 1.04 / 1.09× |
| 512 | 0.88 / 0.88 / 0.96 / **0.72×** | 0.96 / 0.93 / 0.96 / 0.91× |

At n = 512 on 4B, cuBLAS runs at 82–105 TFLOP/s and the cuTile GEMMs at 78–97. 0.6B down at
n = 512 gets 64 × 128 tiles: 64 blocks for 80 SMs, each looping over K = 3,072.

Not kept: **split-K** (Float32 partials and a reduce kernel) for n = 2…64 never beat the best
direct tile (down at n = 8: 13.7 µs against 12.2); the `latency` hint on any GEMM load (see
[cuTile design choices](cutile_design.md)).

## Attention for verify steps

Steps of 9–16 tokens (DFlash's verify block and the drafter's own pass) run the split-KV
decode kernels with 8 splits ([Attention kernels](attention.md)).
16-token attention per layer, µs, both kernels of a path:

| Positions | 0.6B prefill kernel | 0.6B decode, 32 splits | 0.6B, 8 splits | 4B prefill kernel | 4B decode, 32 splits | 4B, 8 splits |
|---|---|---|---|---|---|---|
| 128 | 4.5 | 8.3 | 5.5 | 4.8 | 12.4 | 6.7 |
| 256 | 7.6 | 7.6 | 5.3 | 8.1 | 12.2 | 6.7 |
| 512 | 14.3 | 7.9 | 5.7 | 14.8 | 12.6 | 7.2 |
| 1,024 | 27.4 | 8.8 | 6.7 | 26.9 | 13.6 | 9.5 |
| 2,048 | 53.4 | 11.2 | 9.8 | 53.6 | 17.4 | 14.3 |

The decode kernel does not spill at 64 packed query rows (4B: 4 heads × 16 tokens, 230
registers). End to end, Qwen3-4B with DFlash k = 15, prefill kernel → decode kernels:

| Prompt | positions | ms per verify step | tok/s |
|---|---|---|---|
| short chat prompts | 300–850 | unchanged (±2 %) | unchanged |
| long-edit (2.2k-token file) | 2,230–2,480 | 19.1 → 17.6 | 419 → **454** (+8.3 %) |
| long-explain | 2,230–2,480 | 18.9 → 17.3 | 154 → **169** (+10 %) |

Not kept: 32 splits for these steps (3.5 % slower on the rename prompt); reading the next
page's block-table entry one page ahead (1.00× at 1k, 8k and 32k positions, n = 1 and 8:
with 64-key pages long contexts already read the cache at 600–660 GB/s).

## INT8 weights

Weight-only INT8 with one scale per output channel (`quantize = :int8`, `--int8`), the
embedding in BF16. Accuracy, mean KL divergence to the BF16 model over the fixture prompts
and top-1 agreement: Qwen3-0.6B 0.032 (46/50), Qwen3-4B 0.009 (49/50). Apertus 1.5 8B in INT8
against transformers' Float32 run of the unquantized checkpoint: the argmax differs at 2 of
103 positions, both near-ties (margins 0.2 and 0.02), and 24-token greedy continuations are
identical on all three prompts.

| | BF16 decode | INT8 decode | BF16 prefill 512 | INT8 prefill 512 |
|---|---|---|---|---|
| Qwen3-0.6B | 409 tok/s | 542 tok/s (1.32×) | 58.0k tok/s | 58.9k tok/s (1.02×) |
| Qwen3-4B | 78.7 tok/s | 139 tok/s (1.76×) | 11.4k tok/s | 11.1k tok/s (0.98×) |
| Apertus 1.5 8B | does not fit (16.1 GB) | 80.5 tok/s | — | 4.8k tok/s |

Apertus 8B in INT8 holds 8.6 GB of weights (linear 7.0 GB, lm_head 0.54 GB, BF16 embedding
1.07 GB) and loads in 12 s. Its decode step (12.4 ms from a graph) reads ~7.5 GB of weights
plus the KV cache, ~82 % of the roofline.

**GEMV tiles.** TK × TM ≈ 4,096 Int8 elements (TK = 1,024, TM = 4), half BF16's width, was
best on every Apertus 8B layer; BF16's TM = 8 was 7–13 % slower, TK ≥ 2,048 with TM ≥ 4
much slower. The INT8 GEMVs run at 85–93 % of the roofline (Apertus: qkv 91 %, o 87 %,
up + xIELU 91 %, down 85 %, lm_head 93 %).

**GEMM tiles.** With BF16's tiles, INT8 prefill was 8–11 % slower than BF16. Under `ncu`
(Qwen3-4B down): at 512 tokens 255 registers and 1.6× BF16's instructions (byte shuffles and
conversions of the weight tile); at 64 tokens 29 % of DRAM peak against BF16's 90 %, with 37 %
of the stall samples on the conversion waiting for its global load, because the Int8 tile is
not staged through shared memory like the BF16 operands. A sweep of 15 tiles × occupancy 1, 2
on the layers of Qwen3-0.6B, Qwen3-4B and Apertus 8B at n = 8, 64, 512 found that INT8 blocks
always want two per SM, narrower tiles than BF16 at 512 tokens (128 × 64 × 64), and deeper K
steps at ≤ 64 tokens (32 × 32 × 256, or 64 × 64 × 128 on large grids). `gemm_int8_config`
is within 1.5 % of the best tile summed over the sweep (worst 1.11×). Qwen3-4B:

| | BF16 tiles | INT8 tiles | BF16 |
|---|---|---|---|
| GEMMs at 64 tokens: qkv / o / gate_up / down | 72 / 46 / 155 / 126 µs | 44 / 26 / 99 / 62 µs | 60 / 36 / 175 / 85 µs |
| GEMMs at 512 tokens | 237 / 155 / 580 / 369 µs | 196 / 127 / 567 / 284 µs | 203 / 116 / 521 / 266 µs |
| Prefill 64 | — | **9.5 ms (1.53× BF16)** | 14.5 ms |

Not kept: `latency` hints on the weight loads (1–8), and swapping the GEMM operands (Xᵀ W,
the weight in its natural layout as the B operand).

## INT8 KV cache

One Int8 scale per cached vector, keys smoothed by the key norm's weight
([KV cache](kv_cache.md#INT8-cache)). Why the keys need smoothing, Qwen3-0.6B, chat prompt,
relative error of the quantized cache:

| | layer 1 K | layer 2 K | layer 14 K | layer 28 K | values (any layer) |
|---|---|---|---|---|---|
| one scale per vector | 0.022 | 0.025 | 0.011 | 0.020 | 0.007–0.013 |
| one scale per channel (KIVI) | 0.005 | 0.004 | 0.008 | 0.005 | 0.008–0.016 |
| per vector, keys smoothed (kept) | **0.003** | **0.004** | 0.011 | **0.006** | — |
| largest / median channel max | 109 | 43 | 6.5 | 11 | 1.6–5 |

Per-channel scales do not fit a cache written one position at a time. Mean KL to a BF16
cache, and largest logit difference to transformers (BF16 cache: 0.63):

| Qwen3-0.6B | chat | prose | code | max logit difference |
|---|---|---|---|---|
| one scale per vector | 0.030 | 0.0089 | 0.0043 | 1.75 |
| smoothed keys (kept) | **0.0051** | **0.0040** | **0.0022** | **0.88** |

Speed on Qwen3-4B:

| | BF16 cache | INT8 cache |
|---|---|---|
| decode at 1,024 positions | 78.7 tok/s | 78.1 tok/s |
| decode at 32,768 positions (graph step) | 51.3 tok/s | **60.9 tok/s (1.19×)** |
| decode attention at 32,768 positions | 209 µs (642 GB/s) | 119 µs (581 GB/s, for 0.52× the bytes) |
| prefill 512 | 44.9 ms | 45.8 ms |

**Register pressure.** Converting whole `(64, 128)` pages spilled (255 registers, 38 % of
L1 traffic in local memory, 3.1× the BF16 kernel's instructions); converting K alone needs
178 registers, V alone 232. Variants at 32k positions (`bench/codegen/probes/08_int8_attention.jl`):

| Decode attention, Int8 K and V | µs | registers | spills |
|---|---|---|---|
| whole pages | 117–123 | 255 | 56 B |
| V stored transposed per page (another cache layout) | 112 | 255 | 8 B |
| Int8 QKᵀ: queries quantized per column, `mmai` | 109 | 206 | 0 |
| Int8 QKᵀ with two-term queries (q_hi + q_lo / 254) | 115 | 232 | 0 |
| 32-key tiles | 113 | 211 | 0 |
| K and V in halves of the head dimension | 114 | 190 | 0 |
| **32-key tiles and halves (kept)** | **111** | **184** | **0** |

In the real kernel the kept variant runs at 191 registers (prefill kernel: 232 → 218) and
took decode attention at 32k positions from 150 to 119 µs and prefill-512 attention from 84 to
67 µs (1.25×; 0.6B 1.28×). Int8 QKᵀ was not kept: the KL to the BF16 cache went from 0.0051
to 0.0070 (chat) and the largest logit difference from 0.88 to 2.0, because smoothing makes
the scaled queries heavy-tailed. `latency` hints on the loads changed nothing.

## Speculative decoding

Qwen3-4B target unless noted, `bench/speculative.jl`, speedup in decode tok/s over plain
pipelined decoding (78–80 tok/s).

**n-gram lookup**, measured with a 200-token budget, best of 5, including the prefill:

| Prompt | Model | plain | k = 1 | k = 2 | k = 4 | k = 7 | tokens per step at k = 7 (acceptance) |
|---|---|---|---|---|---|---|---|
| rename two variables in a 15-line function | 0.6B | 424 tok/s | 1.65× | 2.22× | 3.19× | **4.15×** | 4.83 (0.63) |
| | 4B | 78 tok/s | 1.65× | 2.08× | 2.87× | **3.75×** | 4.07 (0.53) |
| short story | 0.6B | 434 tok/s | 0.96× | 0.97× | 0.97× | 0.96× | 1.05 (0.02) |
| | 4B | 80 tok/s | 1.05× | 1.05× | 1.05× | 1.05× | 1.08 (0.03) |
| *sampled:* rename | 0.6B | 422 tok/s | 1.64× | 2.19× | 3.13× | **4.06×** | 4.83 (0.63) |
| | 4B | 78 tok/s | 1.65× | 2.08× | 2.86× | **3.73×** | 4.07 (0.53) |
| short story | 0.6B | 432 tok/s | 1.00× | 1.01× | 1.00× | 1.00× | 1.10 (0.03) |
| | 4B | 80 tok/s | 1.04× | 1.05× | 1.04× | 1.04× | 1.08 (0.03) |

With nothing to copy, 0.6B loses the host/GPU overlap of pipelined decoding: speculative
steps synchronize to read the accepted count. Sampled speculative decoding picks the tokens
of plain sampled decoding: identical over 200 tokens in Float32 (4 seeds); in BF16 they part
after 15–45 tokens.

**Draft model** (Qwen3-0.6B), 256 tokens:

| Prompt | plain | n-gram k = 2 / 4 / 7 | draft k = 2 / 4 / 7 | draft acceptance, tokens per step at k = 4 |
|---|---|---|---|---|
| rename two variables (greedy) | 72 tok/s | 2.29× / 3.18× / **4.22×** | 2.28× / 2.89× / 3.62× | 0.96, 4.78 |
| add a docstring and comments | 79 | 1.37× / 1.46× / 1.53× | 1.64× / **1.78×** / 1.65× | 0.56, 3.24 |
| explain hash-map collisions | 79 | 1.09× / 1.09× / 1.11× | 1.59× / **1.63×** / 1.46× | 0.49, 2.94 |
| short story | 79 | 1.04× / 1.03× / 1.03× | **1.39×** / 1.35× / 1.09× | 0.36, 2.44 |
| *sampled:* rename | 78 | 2.09× / 2.92× / **3.87×** | 2.09× / 2.65× / 3.33× | 0.96, 4.78 |
| add a docstring and comments | 79 | 1.38× / 1.45× / 1.54× | 1.71× / **1.91×** / 1.79× | 0.62, 3.46 |
| explain hash-map collisions | 79 | 1.08× / 1.08× / 1.10× | **1.47×** / 1.39× / 1.13× | 0.38, 2.53 |
| short story | 79 | 1.02× / 1.02× / 1.02× | **1.42×** / 1.37× / 1.18× | 0.37, 2.49 |

**DFlash** (z-lab/Qwen3-4B-DFlash-b16) against the other proposers, best k of each (k = 15
for DFlash in every row). EAGLE-3 (AngelSlim/Qwen3-4B_eagle3, best at k = 2) was removed
from the engine; its column is kept for comparison:

| Prompt | n-gram | 0.6B draft | EAGLE-3 | DFlash | DFlash tokens per step |
|---|---|---|---|---|---|
| rename (greedy) | 3.92× | 3.36× | 2.25× | **8.50×** | 11.0 |
| document | 1.55× | 1.80× | 1.75× | **3.00×** | 3.94 |
| explain | 1.12× | 1.63× | 1.58× | **2.50×** | 3.24 |
| story | 1.04× | 1.39× | 1.45× | **1.92×** | 2.51 |
| *sampled:* document | 1.54× | 1.92× | 1.82× | **3.17×** | 4.20 |
| explain | 1.10× | 1.48× | 1.46× | **2.30×** | 3.01 |
| story | 1.03× | 1.42× | 1.57× | **1.75×** | 2.31 |

The drafts match a Float32 transcription of z-lab's `dflash.py`, and acceptance on the
"explain" prompt is the reference's (2.78 against 2.76 tokens per step over 128 tokens).
Per-draft acceptance on "explain" falls from 0.67 at k = 2 to 0.15 at k = 15. On the two
2.2k-token prompts, the whole block still pays on the edit (5.9× over plain, 8.0 tokens per
step), while on the open one k = 7 wins (2.37× against 2.20×; 3.2 and 2.9 tokens per step).
Plain decoding runs at 77 tok/s there, and the 2.2k-token prefill takes 203 ms.

Costs of a DFlash step at 400 positions:

| | ms |
|---|---|
| drafter pass (GEMMs: 1.85 GB at ~650 GB/s, 0.78 GB of it the lm_head) | 3.17 |
| verify step of 16 tokens (1 token: 12.7) | 13.3 |
| draft sampling: 15 launches per column → one batched kernel | 0.086 → 0.050 |
| host round trip between drafter and verify, removed (drafts written on the device) | ~0.2 (+1.2 % tok/s) |

**Sampled drafts** (the p/q rule, `--sample-drafts`), with Qwen3's recommended sampling,
best k of each method, greedy drafts → sampled drafts. Each row is a single sampled reply,
so one step more or less moves it by several percent:

| Prompt | 0.6B draft | EAGLE-3 | DFlash | draft acceptance at k = 4 |
|---|---|---|---|---|
| rename | 3.35× → 3.34× | 2.25× → 2.19× | 8.42× → 7.57× | 0.96 → 0.96 |
| document | 1.92× → 1.82× | 1.82× → 1.77× | 3.17× → 2.78× | 0.62 → 0.57 |
| explain | 1.48× → **1.59×** | 1.46× → **1.58×** | 2.30× → 2.29× | 0.38 → 0.44 |
| story | 1.42× → 1.42× | 1.57× → 1.48× | 1.76× → **1.88×** (k = 4) | 0.37 → 0.39 |

The p/q kernels share the candidate selection with the plain sampler, which runs in the same
16.8 µs as before the refactor.

## Comparison with vLLM

Setup: the RTX 4080 SUPER also drove the desktop (compositor, VS Code and Discord on the
GPU); TileInfer at commit 8528c2e on Julia 1.13.3; vLLM 0.31.0 with torch 2.13.0
(`bench/ref/requirements.txt`) with CUDA graphs and torch.compile; prefix caching off and one
sequence at a time. Scripts and setup in `bench/ref/README.md`.

**The race**: 512-token prompt, 256 greedy tokens, the same first tokens on both engines.
`bench/ref/alternate.py`: 3 rounds alternating the engines (ABBA order), each a fresh process
recording 10 trials of each latency; 30 trials per engine pooled. Median [q1, q3]:

| | Qwen3-0.6B: vLLM | Qwen3-0.6B: TileInfer | Qwen3-4B: vLLM | Qwen3-4B: TileInfer |
|---|---|---|---|---|
| **End to end** (latency of 256 tokens) | **0.6439 s** [0.6419, 0.6476] | **0.6227 s** [0.6203, 0.6263] | **3.328 s** [3.296, 3.350] | **3.309 s** [3.300, 3.318] |
| 1-token request ("prefill") | 10.1 ms [9.9, 10.3] | 9.7 ms [9.6, 12.8] | 46.6 ms [45.5, 46.8] | 47.0 ms [45.4, 49.0] |
| Decode = (e2e − 1-token) / 255 | 2.49 ms/tok (402 tok/s) | 2.40 ms/tok (416 tok/s) | 12.87 ms/tok (77.7 tok/s) | 12.79 ms/tok (78.2 tok/s) |

On 4B, vLLM's fastest round matches TileInfer's median (minimum 3.295 s against 3.299 s).
TileInfer has 1–3 slow 1-token trials per process (20–70 ms, and one of 200–250 ms in every
run, on both models), which widen its q3; vLLM has none. A traced repeat showed one such
trial spending 31 ms in GC and 37 ms compiling.

**INT8.** vLLM runs a W8A16 compressed-tensors copy of each checkpoint quantized as
TileInfer's `quantize_int8` (`bench/ref/make_w8a16.py`: every linear and the untied lm_head),
with its Marlin kernels; its INT8 KV cache is `kv_cache_dtype = int8_per_token_head`, which
moves it to its Triton attention backend. Same race, end-to-end medians:

| Latency of 256 tokens | 0.6B vLLM | 0.6B TileInfer | 4B vLLM | 4B TileInfer |
|---|---|---|---|---|
| BF16 | 0.644 s | **0.623 s** (−3.3 %) | 3.328 s | 3.309 s (tie) |
| INT8 weights | 0.532 s | **0.503 s** (−5.5 %) | 1.880 s | 1.884 s (tie) |
| INT8 KV | 0.622 s | 0.625 s (tie) | **3.264 s** | 3.307 s (+1.3 %) |
| INT8 weights + KV | 0.474 s | 0.470 s (tie) | **1.837 s** | 1.871 s (+1.8 %) |

With INT8 weights the first tokens can differ between the engines (vLLM stores the scales in
BF16, TileInfer in Float32).

**Speculative decoding, best configuration per engine.** `bench/ref/tileinfer_spec.jl` and
`bench/ref/vllm_spec.py` on `bench/speculative.jl`'s six prompts with the same token ids,
greedy, up to 256 tokens stopping at EOS, median of 5 trials; decode = (n − 1) / (latency(n) −
latency(1)). Each engine ran plain, DFlash k = 7 and k = 15 (4B), n-gram k = 4 (0.6B), each in
BF16 and with INT8 weights + KV; the table keeps each engine's fastest per prompt, which on
4B is DFlash k = 15 with INT8 for both, except TileInfer's k = 7 on long-explain:

| Qwen3-4B | vLLM tok/s (× its BF16 plain) | TileInfer tok/s (× its BF16 plain) | TileInfer / vLLM | tokens per step vLLM / TileInfer |
|---|---|---|---|---|
| rename | 1,150 (14.7×) | 947 (12.2×) | 0.82 | 11.0 / 10.0 |
| document | 385 (4.9×) | 379 (4.9×) | 0.98 | 3.9 / 4.0 |
| explain | 322 (4.1×) | 290 (3.7×) | 0.90 | 3.2 / 3.0 |
| story | 255 (3.2×) | 219 (2.8×) | 0.86 | 2.5 / 2.3 |
| long-edit (2.2k) | 494 (6.5×) | **698 (9.2×)** | **1.41** | 8.0 / 8.3 |
| long-explain (2.2k) | 158 (2.1×) | **270 (3.5×)** | **1.70** | 2.6 / 3.0 |

In BF16 on rename both engines take 10 steps for 110 tokens, at 731 tok/s (vLLM) and 558
(TileInfer). At these short contexts vLLM's plain INT8 decode is 145 tok/s against
TileInfer's 139. On long-explain vLLM's DFlash reaches 158 tok/s against 134 for its plain
INT8 decode.

On Qwen3-0.6B (no DFlash drafter), with INT8 the greedy replies differ in length between the
engines on two prompts, and TileInfer's reply to long-edit ends after 5 tokens. Where the
replies match, best configuration per engine:

| Qwen3-0.6B | vLLM | TileInfer |
|---|---|---|
| rename | 1,236 tok/s (n-gram + INT8) | **1,725** (n-gram + INT8) |
| explain | **606** (plain INT8) | 563 (plain INT8) |
| long-explain | 469 (plain INT8) | **588** (n-gram + INT8) |
| story, BF16: n-gram / plain | 298 / 405 | 413 / 428 |

vLLM counts only the steps that had drafts, so its n-gram tokens per step are left out.

## What is left to gain

Measured costs that a change could recover, largest first within each group. The plan is in
[History and roadmap](roadmap.md#Next).

| Where | Measured | Lever |
|---|---|---|
| **Decode** | | |
| 0.6B GEMVs (layers of 4–12 MB each, and the lm_head) | 1,898 µs per token against a 1,619 µs floor (~11 % of the token) | fill and drain of small kernels: persistent or stream-K GEMVs (Ada has no programmatic dependent launch) |
| Small decode kernels | 0.6B 233 µs (9.5 %), 4B 409 µs (3.2 %) | fewer launches: q and kv inputs in one kernel, norms in GEMV epilogues |
| Decode attention at 1k positions | 0.6B 308 µs against ~160 to read the KV once (~6 %) | latency-bound, about one page per split |
| 4B layer GEMVs | 90–93 % of DRAM against the lm_head's 95 % (~3 %) | |
| INT8 decode GEMV | vLLM's Marlin kernels 4–7 % faster at short contexts | |
| **Prefill** | | |
| 512-token GEMMs | 0.88–0.96× cuBLAS (0.6B down 0.72×); 4B: 40.0 ms against 37.9 at cuBLAS speed | larger tiles or split-K for under-filled grids (0.6B down: 64 blocks for 80 SMs) |
| `q_inputs_kernel` at 512 tokens | 0.6B 448 µs (5.2 %) | several tokens per block |
| Short eager prefills | 0.6B prefill64: GPU idle 12 % of the step | a graph, or fewer launches |
| `silu_mul` | 4B prefill512 625 µs (1.4 %) | SwiGLU epilogue in the gate_up GEMM, as in the GEMV |
| INT8 GEMMs at 512 tokens | 0.98× BF16 on 4B (gate_up 567 against 521 µs, down 284 against 266) | weight layout prepacked for the conversion, as Marlin does |
| **Speculative decoding and host** | | |
| DFlash step at short contexts | vLLM 1.31× faster with the same steps (731 against 558 tok/s on rename, BF16) | not profiled |
| k per prompt | long-explain: k = 7 beats 15 (2.37× against 2.20×) | adaptive k, tree verification ([Speculative decoding](speculative.md#Not-built-yet)) |
| Host pauses | 1–3 slow requests per process, 20–250 ms (GC and compilation) | |
