# Performance log

What each milestone and each optimization bought, in BF16 on the RTX 4080 SUPER (736 GB/s):
on Qwen3-0.6B up to the per-shape GEMM tiles, and on both Qwen3-0.6B and Qwen3-4B from then on (see "Two
models"). Newest last. Numbers that informed a decision but did not pan out are kept,
with the reason, so they are not retried blindly.

## How things are measured

- **`bench/run.jl`** (see CLAUDE.md), minimum time over samples, `--seconds 2` for
  decisions; expect ±3–5 % between runs at the model level (the GPU also drives the desktop).
  Keys start with the model (`Qwen3-4B/model/...`) since 2026-10-29; earlier keys were 0.6B.
  - `model/decode_ctx1024/<be>`: one eager decode step after 1,023 cached positions, synced.
  - `model/decode_graph_ctx1024/tile`: the same step replayed from a CUDA graph (M6+), synced.
  - `model/generate64[_unpipelined|_sampled]_ctx1024/tile`: 64 tokens after 1,023 positions,
    per token (perf branch). Greedy pipelined overlaps host and GPU, so only it shows the
    gaps between steps disappearing; `_sampled` uses the model's recommended sampling
    (Qwen3: temperature 0.6, top-k 20, top-p 0.95), which is what `tileinfer run`/`chat` do.
  - `model/prefill{64,512}/<be>`: a 64- or 512-token prompt in one step.
- **End-to-end race** (`bench/ref/tileinfer_race.jl`, `bench/ref/vllm_race.py`): 512-token
  prompt, 256 greedy tokens, prefill = latency of generating 1 token, decode =
  (256 − 1) / (latency(256) − latency(1)). Same prompt ids and the same first tokens on both
  engines.
- **Kernel breakdowns**: `CUDA.@profile trace = true` over 20 generated tokens, summing
  kernel durations per name (the tracer adds ~5 % to wall time). Since 2026-10-29
  `bench/breakdown.jl` does this for decode and prefill steps of both models, per kernel role.
- **Roofline for decode at 1,024 positions**: Qwen3-0.6B reads 1.19 GB of weights + 117 MB
  of KV cache per token ≈ 1.78 ms, i.e. ~560 tok/s; Qwen3-4B 8.04 GB + 151 MB ≈ 11.1 ms,
  ~90 tok/s.

## Summary

Current, both models (2026-10-30, e765381, `bench/run.jl --filter /model/ --seconds 2`,
baseline `summary-2026-10-30`; decode after 1,023 cached positions, per token of 64):

| | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| Decode, greedy pipelined | 2.53 ms, **395 tok/s** (70 % of roofline) | 12.97 ms, **77.1 tok/s** (86 %) |
| Decode, sampled (Qwen3 defaults, pipelined) | 2.49 ms, 402 tok/s | 12.86 ms, 77.8 tok/s |
| Decode, greedy step by step / eager step | 2.58 / 2.72 ms | 13.11 / 13.15 ms |
| Prefill 64 tokens | 2.75 ms, 23k tok/s | 14.6 ms, 4.4k tok/s |
| Prefill 512 tokens | 8.85 ms, **58k tok/s** (69 TFLOP/s) | 45.3 ms, **11.3k tok/s** (91 TFLOP/s) |
| Speculative decoding, best proposer (4B: DFlash k = 15, sampled) | — | 1.8× (story) to 3.2× (document) on open prompts, 8.4× on a code rename: 140–660 tok/s |

0.6B greedy decode reads 2–3 % slower than the 2.44 ms of 2026-10-29, within the run-to-run
spread (the sampled run, the same kernels plus sampling, measured 2.49 ms). The M8 target
(≥ 80 % of the decode roofline) holds on 4B, not on 0.6B.

History on Qwen3-0.6B:

| Stage | Decode step | Decode tok/s | Prefill 512 | Prefill tok/s | What changed |
|---|---|---|---|---|---|
| M3 (reference ops, paged KV) | 38.89 ms | 26 | 76.3 ms | 6.7k | cuBLAS + broadcasts |
| M4 (cuTile kernels, reference attention) | 34.77 ms | 29 | 54.0 ms | 9.5k | norm, rope, kv_write, silu, embed, GEMV, GEMM |
| M5 (paged flash-attention) | 3.80 ms | 263 | 16.0 ms | 32k | prefill FA-2, split-KV decode |
| M6 (CUDA graphs) | 2.87 ms (graph) | 348 | 11.8 ms | 43k | allocation-free step, graph replay |
| perf branch | 2.47 ms (pipelined) | 405 | 9.0 ms | 57k | fusions, device greedy sampling, host/GPU overlap |
| device sampling | 2.46 ms (sampled) | 407 | 9.0 ms | 57k | top-k/top-p sampling in the graph: sampled decode as fast as greedy |
| GEMM occupancy | 2.46 ms (sampled) | 407 | 8.8 ms | 58k | two GEMM blocks per SM on large grids (4B prefill 512: 1.06×) |
| M7 (speculative decoding) | 2.49 ms (sampled) | 402 | 8.85 ms | 58k | plain decoding unchanged (re-measured); n-gram, draft-model, EAGLE-3, DFlash proposers and p/q verification on top |

From M3: decode 15.7× faster, prefill 8.5×. Race: **418 tok/s decode, 53k tok/s prefill**,
3.6 % faster end to end than vLLM on the same machine (see the end).

## M4: cuTile kernels (2026-10-27)

Every op except attention moved to cuTile; attention still used the reference.

- Decode GEMV at 68–93 % of the DRAM roofline (1.0–1.3× cuBLAS); prefill GEMM 0.70–0.92×
  cuBLAS; rope / rmsnorm / store_kv / embed 3–29× the reference.
- End to end only 1.12× (decode) and 1.41× (prefill): the reference attention (≈ 1 ms per
  layer, ~190 kernels per layer, ~6k GPU allocations per token) dominated.

## M5: paged flash-attention (2026-10-27)

| Change | Effect |
|---|---|
| FA-2 prefill kernel over the block table | prefill attention 1.27 ms → 30 µs (42×, 36 TFLOP/s) |
| Split-KV decode kernel + combine | decode attention 1.04 ms → 14 µs per layer (73×) |
| End to end | decode 34.8 → 3.8 ms (9.2×), prefill 54 → 16 ms (3.4×) |
| Prefill tile 64 → 32 query rows | prefill attention 1.28× (256 blocks instead of 128) |
| Decode query tile padded 2 → 16 rows, occupancy 2 | decode attention kernel 15.2 → 9.8 µs under `ncu` (fewer spills, 2 blocks/SM) |
| 2-D views to keep launches < 32 kernel parameters | ~10 µs less host time per launch; `store_kv` 1.7× |
| Block table padded to 16 pages | no recompiles at page boundaries: `tileinfer run` 55 → 133 tok/s on a 50-token reply |

## M6: CUDA graphs (2026-10-27)

| Change | Effect |
|---|---|
| Allocation-free step with static buffers (eager) | decode 3.80 → 3.49 ms (1.09×), prefill 16.0 → 11.8 ms (1.36×) |
| Graph replay of the decode step | 3.49 → 2.87 ms (1.22×): the host no longer enqueues 539 kernels per token |

After M6, under the tracer, the GPU was busy 2.93 ms per step, plus ~0.48 ms of gap between
steps (host sampling, uploads and the graph launch while the GPU idles). Kernel time per step:

| Kernels | µs per step |
|---|---|
| GEMV (4 per layer + lm_head) | 1,919 |
| rmsnorm (4 per layer) | 303 |
| attention decode + combine | 252 + 63 |
| CUDA.jl broadcasts and copies (q/k/v slices, residual adds, Float32 logits) | 170 |
| rope, store_kv, silu_mul | 110 + 64 + 39 |

## Perf branch (`perf-scratch`, 2026-10-27)

Baseline `p0` = M6 re-measured: graph step 2.89 ms (346 tok/s), eager 3.58 ms, prefill
11.83 ms; race 343 tok/s decode / 40k tok/s prefill. Each row is measured against the row
above it.

| # | Optimization | Decode (graph step) | Eager step | Prefill | Notes |
|---|---|---|---|---|---|
| 1 | **Fused attention inputs** (`attention_inputs!`): q/k RMSNorm + rope + KV write straight from `qkv`, 2 kernels instead of 8 launches per layer, no slice copies | 2.89 → 2.58 ms (**1.12×**) | 1.29× | 1.26× | biggest single win; bit-identical to the unfused tile ops |
| 2 | **Residual add + RMSNorm** (`add_rmsnorm!`), the next layer's input norm done by the previous layer | 2.58 → 2.48 ms (1.04×) | 1.03× | 1.04× | removes 2 broadcasts per layer |
| 3 | **GEMV epilogues**: SwiGLU in the decode gate_up GEMV, Float32 logits from the lm_head | 2.48 → 2.46 ms (1.01×) | 1.03× | 1.00× | fused gate_up 21.6 µs vs 22.8 µs for GEMV + silu_mul. A first version with half-width column tiles (to keep registers per thread equal) was 6 % slower than unfused |
| 4a | **Pinned staging, one async upload** of all step inputs | no change | — | — | groundwork for 4: uploads that don't block the host |
| 4 | **Device greedy sampling + pipelined decode** (`generate_greedy!`): argmax inside the graph writes the next step's input; two alternating graphs; the host launches step t+1 before reading token t | generate64: 2.62 ms/token with `pipeline = false` → **2.47 ms/token (1.06×)** | — | — | gaps between steps gone (GPU busy 95 % of wall time under the tracer); race decode 343 → 417 tok/s |
| 5 | GEMV tile sweep (TK ∈ 256…2048, TM ∈ 1…16) per decode shape | — | — | — | **no gain**: all configs within ~3 %. o/down (69–76 % of roofline) lose to per-kernel fill/drain, not tiling |
| 6 | Combine fused into the split-KV kernel (last split per head merges, via an acquire-release counter) | — | — | — | **reverted**: 322 µs/step vs 315 for decode + combine; the serial merge in one block costs what the launch saved |
| 7 | V load issued before the QK product in decode attention | — | — | — | **reverted**: 252 µs/step either way (the compiler already overlaps them) |
| — | Fewer decode splits (160 → 80 target blocks) | — | — | — | combine 63 → 47 µs/step (≈ 0.6 %); kept 160 for longer contexts |

Final: generate64 **2.47 ms per token (405 tok/s, 72 % of the roofline)**, graph step
2.52 ms (it syncs after each step), prefill 9.0 ms (57k tok/s); race 418 tok/s / 53k tok/s.
Overall on this branch: decode 1.17× (2.89 → 2.47 ms), prefill 1.31×.

A layer is now 10 kernels (qkv GEMV, q inputs, kv inputs, attention + combine, o GEMV,
add+norm, gate_up+SwiGLU GEMV, down GEMV, add+norm), 285 per token instead of 539. Kernel
time per step, after:

| Kernels | µs per step |
|---|---|
| GEMV (qkv, o, down, lm_head) + gate_up SwiGLU GEMV | 1,358 + 577 |
| attention decode + combine | 252 + 63 |
| add_rmsnorm (2 per layer) | 114 |
| attention inputs (q, kv) | 67 + 70 |
| argmax, embed, first rmsnorm | 9 |
| **total busy** | **2,510** (vs 2,930 at M6) |

## Where the remaining time is

Against the ~1.78 ms roofline at 1,024 positions:

- **GEMVs: 1.94 ms vs a 1.62 ms floor** for reading the weights once. The lm_head is at
  92 %; the small layer GEMVs (4–12 MB each) lose 2–3 µs each to the fill and drain of a
  kernel. Levers: weight-only INT8 (halves the bytes, the biggest one left), or split/stream-K
  and persistent kernels. Ada has no programmatic dependent launch to overlap kernel tails.
- **Attention: 0.31 ms vs ~0.16 ms** for reading the KV cache. Each split handles about one
  page, so the kernel is latency-bound; longer contexts amortize this better.
- **Small kernels: ~0.25 ms**, already fused down to 3 per layer.

## Per-shape GEMM tiles (2026-10-29)

Prefill GEMM tiling for few tokens, and two ideas that did not pay on the 4080S:

| Change | Result | Decision |
|---|---|---|
| **Per-shape GEMM tiles** (`gemm_config`): pick the tile minimizing waves × (TM TN)^0.7 among 6 shapes, instead of 64 or 128 rows × the next power of two ≥ n columns | Fitted to a sweep of 30 tile shapes × 8 layers (0.6B and 4B) × n = 2…1024: within 2 % of the best tile on average (worst 1.18×), 1.22× the old rule, 1.06× cuBLAS on average over n (ahead at small n, behind at n = 512: see "Two models"). `bench/run.jl` at n = 64: o 1.77×, down 1.98×, qkv and gate_up unchanged; **model/prefill64 3.62 → 3.00 ms (1.21×)**; n = 1 and 512 unchanged | **kept** |
| Split-K GEMM (Float32 partials + reduce kernel) for few tokens | Never beat the best direct tile at n = 2…64 (e.g. down n = 8: 13.7 µs vs 12.2 µs). K ≤ 3072 here, and 32-row tiles already give enough blocks; split-K pays with longer K or more SMs | not kept |
| Block-table entry of the next page read one page ahead in decode attention | 1.00× at 1k, 8k and 32k positions (n = 1 and 8): with 64-key pages and `ct.load` latency hints, long contexts already run at 600–660 GB/s. It matters with tiny pages, where every key is a gather | not kept |


## Comparison with vLLM (same machine, `bench/ref`)

`bench/ref/alternate.py`, 2026-10-28: 3 rounds alternating the engines (ABBA order), each a
fresh process recording 10 trials of each latency after a warm-up; 30 trials per engine
pooled. 512-token prompt, 256 greedy tokens, the same first tokens on both engines. Medians
are the primary statistic (timings are heavy-tailed), with the interquartile range.

| Median [q1, q3] | vLLM 0.31 (CUDA graphs + torch.compile) | TileInfer perf branch | |
|---|---|---|---|
| **End to end** (latency of 256 tokens) | **0.6432 s** [0.6428, 0.6437] | **0.6200 s** [0.6196, 0.6202] | **3.6 % faster** |
| 1-token request ("prefill") | 25.4 ms [24.7, 41.5] | 9.77 ms [9.74, 10.05] | |
| Decode = (e2e − 1-token) / 255 | 2.42 ms/tok (413 tok/s) | 2.39 ms/tok (418 tok/s) | |

End to end is the robust comparison. vLLM's 1-token latency is multimodal, with clusters at
~11, ~25 and ~40–47 ms within the same process (sorted trials of one run: 10.8, 11.0, 11.1,
11.1, 11.5, 28.4, 46.8, 47.2, 47.5, 47.9 ms), while its 256-token latency varies by under
1 ms. So the extra 15–35 ms are per-request overhead that the 256-token request does not pay
in full, not prefill compute. Its fast mode (~11 ms) is the best estimate of vLLM's prefill,
so the subtraction above overstates its decode speed: with 11 ms, vLLM decodes at 2.48 ms/tok
(403 tok/s), as in the earlier 10-trial runs that hit the fast mode. TileInfer's timings are
unimodal (prefill IQR 0.3 ms).

Earlier single runs: vLLM `--eager` 172 tok/s decode / 39k tok/s prefill; TileInfer M6
343 tok/s / 40k tok/s (e2e 0.755 s).

### Qwen3-4B (2026-10-28)

Same protocol (`alternate.py --model Qwen/Qwen3-4B`, 30 trials per engine), the first run of
TileInfer on this model: no kernel changes, and the same first tokens as vLLM.

| Median [q1, q3] | vLLM 0.31 | TileInfer perf branch | |
|---|---|---|---|
| **End to end** (latency of 256 tokens) | **3.3013 s** [3.3008, 3.3019] | **3.3062 s** [3.3061, 3.3068] | 0.15 % slower |
| 1-token request ("prefill") | 48.0 ms [46.7, 50.4] | 53.2 ms [49.7, 57.8] | ≈ 10 % slower |
| Decode = (e2e − 1-token) / 255 | 12.76 ms/tok (78.4 tok/s) | 12.76 ms/tok (78.4 tok/s) | tie |

At 4B both engines decode at ~87 % of the bandwidth roofline (8.1 GB of weights and KV per
token ≈ 11.1 ms, 90 tok/s): the per-kernel overheads that separated them at 0.6B are small
next to 220 MB of weights per layer. Prefill is compute-bound here, where TileInfer's GEMM
is the gap (0.87–0.92× cuBLAS at n = 512, measured in "Two models" below); its prefill also
has a wider spread (one run at 58 ms).

## Two models in the benchmark suite (2026-10-29)

`bench/run.jl` now runs every op and model benchmark on Qwen3-0.6B and Qwen3-4B (one model
after the other, ~7 min with `--seconds 2`), and adds sampled generation. Baseline saved as
`two-models` (commit 3763805 + the suite changes). The two models stress different things:
0.6B has small layers (4–12 MB per GEMV), so per-kernel overheads and host work show; 4B has
up to 100 MB per GEMV, so decode is a bandwidth test and prefill a tensor-core test. Same
context (1,024 positions), same KV shape (8 KV heads of 128), so attention costs the same.

### End to end

| | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| Decode, greedy pipelined (`generate64`) | 2.44 ms/token, 409 tok/s, **72.8 %** of roofline | 12.70 ms/token, 78.7 tok/s, **87.6 %** |
| Decode, greedy step by step | 2.60 ms/token (1.06× slower) | 12.85 ms/token (1.01×) |
| Decode, sampled (Qwen3 defaults) | 3.32 ms/token, 301 tok/s (**1.36× slower**) | 13.40 ms/token, 74.7 tok/s (1.05×) |
| Decode step, graph / eager | 2.48 / 2.63 ms | 12.78 / 13.06 ms |
| Prefill 64 tokens | 3.02 ms (54 % of DRAM: reads the weights once) | 14.6 ms (75 % of DRAM) |
| Prefill 512 tokens | 9.00 ms, 56.9k tok/s | 47.6 ms, 10.8k tok/s |
| Tile vs reference: decode / prefill 512 | 16.2× / 8.5× | 6.2× / 4.8× |

- **The M8 target (≥ 80 % of the decode roofline) is met on 4B, not on 0.6B.** The fixed
  costs that 0.6B still pays (kernel fill and drain, host sampling, graph launch) are
  constant per token, so at 4B they are drowned by 5× more bytes per token.
- **Sampled decoding costs a constant ~0.7–0.9 ms per token**: the logits (600 KB) are copied
  to the host and sampled there, step by step, with the GPU idle meanwhile. That is 36 % of a
  0.6B token and 5 % of a 4B token. Device top-k/top-p sampling inside the graph, pipelined
  like greedy, removes it; it is also what M7's rejection sampler needs.
- **Graphs matter little at 4B** (eager step 2 % slower vs 6 % at 0.6B): 363 launches cost
  the same host time, but the GPU takes 5× longer per step, so enqueueing keeps ahead.

### Decode step breakdown from the op benchmarks (per token, µs)

| | Qwen3-0.6B (28 layers) | Qwen3-4B (36 layers) |
|---|---|---|
| GEMVs per layer: qkv + o + gate_up + down | 13.0 + 8.3 + 21.6 + 11.3 = 54 | 48.4 + 32.6 + 149.8 + 77.0 = 308 |
| GEMVs, all layers + lm_head | 1,518 + 454 = 1,972 (81 % of the token) | 11,081 + 1,120 = 12,201 (96 %) |
| GEMV bandwidth | 69–88 % of DRAM, lm_head 93 % | 87–90 %, lm_head 94.5 % |
| Attention decode, all layers | 15.1 × 28 = 423 | 15.0 × 36 = 540 |

(gate_up is the plain GEMV; the decode step uses the fused SwiGLU GEMV: 21.6 µs on 0.6B vs
22.8 µs for GEMV + silu_mul, perf branch. The fused ops are not in this table: see "Full
picture" below for the whole step, kernel by kernel.)

At 4B the decode step is GEMVs at ~88 % of DRAM bandwidth. What is left: ~2–6 % by bringing
the layer GEMVs to the lm_head's 94 %, and beyond that only fewer bytes (weight-only INT8
halves them). Attention is latency-bound at 38 % of DRAM for both (one page per split at
1,024 positions); it is 4 % of a 4B token.

### Prefill GEMMs against cuBLAS

| n | Qwen3-0.6B: qkv / o / gate_up / down | Qwen3-4B: qkv / o / gate_up / down |
|---|---|---|
| 64 | 1.06 / 1.42 / 0.84 / 1.45× | 0.99 / 1.03 / 1.03 / 1.12× |
| 512 | 0.93 / 0.88 / 0.91 / **0.77×** | 0.88 / 0.92 / 0.87 / 0.91× |

- **At n = 512 every GEMM is behind cuBLAS** (82–105 TFLOP/s for cuBLAS on 4B vs 72–96 for
  ours). On 4B the four GEMMs are 1,188 µs per layer, 42.8 ms of the 47.6 ms prefill; at
  cuBLAS speed they would be 37.8 ms, which is the ~10 % prefill gap to vLLM in the race.
- **0.6B down at n = 512 (0.77×)**: `gemm_config` picks 64 × 128 tiles, i.e. 64 blocks for
  80 SMs, each looping over K = 3,072. An under-filled GPU with a long K loop is the case
  where split-K (or stream-K) pays, unlike the n ≤ 64 shapes where split-K lost (see "Per-shape GEMM tiles").
- The cost model was fitted to an average over n = 2…1024, where small n (tile wins up to
  1.45×) carries it. Tiles beyond 128 × 128, or deeper pipelining, are untested candidates for n ≥ 512.
- Prefill SwiGLU is a separate `silu_mul` kernel: 49 µs × 36 = 1.8 ms of a 4B prefill (4 %)
  and 13.5 µs × 28 = 0.4 ms on 0.6B. A SwiGLU epilogue in the gate_up GEMM, as in the GEMV,
  saves the kernel and the round trip of the (2I, n) activations.


## Full picture: every kernel of a step (2026-10-29)

Before optimizing kernels: what each one costs inside real steps (`bench/breakdown.jl`,
decode replayed from graphs at 1,024 positions), and the fused ops' own op benchmarks
(`attention_inputs`, `add_rmsnorm`, `linear_swiglu`, `argmax`; baseline `full-picture`,
all other benchmarks unchanged from `two-models`). µs per step, share of the GPU-busy time.

### Decode

| Kernels | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| Layer GEMVs: qkv / o / gate_up + SwiGLU / down | 385 / 207 / 567 / 288 = 1,447 (59.0 %) | 1,683 / 1,138 / 5,347 / 2,625 = 10,793 (84.8 %) |
| lm_head GEMV | 451 (18.4 %) | 1,114 (8.8 %) |
| Attention decode + combine | 252 + 58 = 309 (12.6 %) | 324 + 74 = 399 (3.1 %) |
| Attention inputs (q, kv), add + RMSNorm ×2 | 63 + 65 + 106 = 234 (9.5 %) | 100 + 86 + 225 = 411 (3.2 %) |
| Embedding, first norm, argmax | 9 (0.3 %) | 10 (0.1 %) |
| **GPU busy per token** | **2,450** (99 % of the 2,472 µs period) | **12,726** (99.9 % of 12,740) |
| Roofline (weights + KV once) | 1,780 | 11,130 |

GEMV bandwidth inside the step: 0.6B qkv 83 %, o 77 %, gate_up + SwiGLU 85 %, down 83 %,
lm_head 94 %; 4B 91 / 90 / 91 / 93 / 95 %. Over the roofline:

- **0.6B, +670 µs**: GEMVs +279 (1,898 vs a 1,619 floor; the 4–12 MB layers lose to fill and
  drain), small kernels +242 (~2 µs each, 116 of them), attention +149 (309 vs 160 to read
  the KV once, latency-bound with about one page per split). Three levers of similar size.
- **4B, +1,596 µs**: GEMVs +976 (11,907 vs 10,931: the layer GEMVs at 90–93 % against the
  lm_head's 95 %), small kernels +421, attention +193. GEMV bandwidth first.
- **Host**: greedy decode keeps the GPU 99 % busy. Sampled decode idles it ~0.9 ms per token
  (66 % busy on 0.6B under the tracer, 93 % on 4B; 0.88 / 0.69 ms per token in run.jl).

Op benchmarks against the same kernels inside the 0.6B decode graph (µs per call): eager
launches inflate every kernel under ~6 µs, so use the breakdown for small kernels.

| | qkv | o | gate_up + SwiGLU | down | lm_head | attention | attention inputs | add + RMSNorm | argmax |
|---|---|---|---|---|---|---|---|---|---|
| `bench/run.jl` (eager) | 13.0 | 8.3 | 21.2 | 11.3 | 453 | 15.1 | 15.6 | 5.9 | 9.8 |
| In the graph | 13.7 | 7.4 | 20.2 | 10.3 | 451 | 11.1 | 4.6 | 1.9 | 4.2 |

### Prefill

| Kernels | 0.6B, 64 tokens | 0.6B, 512 | 4B, 64 | 4B, 512 |
|---|---|---|---|---|
| GEMMs: qkv + o + gate_up + down | 1,796 (66.3 %) | 6,467 (73.6 %) | 12,367 (86.4 %) | 41,895 (88.3 %) |
| of which gate_up | 736 (27.2 %) | 2,448 (27.9 %) | 6,136 (42.9 %) | 20,653 (43.5 %) |
| lm_head GEMV (last token) | 454 (16.7 %) | 483 (5.5 %) | 1,122 (7.8 %) | 1,141 (2.4 %) |
| Attention prefill | 96 (3.5 %) | 751 (8.5 %) | 133 (0.9 %) | 1,537 (3.2 %) |
| Attention inputs (q, kv) | 193 (7.1 %) | 730 (8.3 %) | 331 (2.3 %) | 1,459 (3.1 %) |
| add + RMSNorm ×2, silu_mul | 165 (6.1 %) | 343 (3.9 %) | 354 (2.5 %) | 1,382 (2.9 %) |
| **GPU busy** (run.jl step) | **2,708** (2.89 ms) | **8,781** (9.00 ms) | **14,312** (14.55 ms) | **47,425** (47.64 ms) |

- **GEMMs are 66–88 % of prefill**, gate_up alone 28 % (0.6B) to 44 % (4B). At 512 tokens
  they are compute-bound and 0.77–0.93× cuBLAS (see "Two models"); at 64 tokens they are
  bandwidth-bound and read the weights at 67 % (0.6B) and 80 % (4B) of DRAM, against 83–93 %
  for the decode GEMVs that read the same bytes.
- **Attention inputs at 512 tokens cost as much as attention itself on 0.6B** (730 vs
  751 µs): `q_inputs_kernel` runs one block per (token, head), 8,192 blocks of one 128-element
  column each, 16 µs per launch for 8.6 MB (≈ 42 % of DRAM with kv_inputs).
- **0.6B prefill64 idles the GPU 9 % of the step** (busy 2.71 ms, span 2.98, run.jl 2.89):
  283 eager launches of mostly short kernels, which the host cannot enqueue as fast as they
  run. 4B kernels are long enough (98 % busy). A graph for short prefills would remove it.
- The lm_head reads 311 MB (0.6B) for one token's logits: 17 % of a 64-token prefill.

### What to optimize

Upper bounds from the tables above (removing the whole excess):

| Target | Where it shows | At most |
|---|---|---|
| ~~Device top-k/top-p sampling, pipelined like greedy~~ (done, see "Device sampling") | sampled decode | 0.7–0.9 ms per token: 1.36× on 0.6B, 1.05× on 4B |
| Prefill GEMMs at n = 512 to cuBLAS (larger tiles, split-K for under-filled shapes such as 0.6B down) | prefill512 | up to ~10 % of the step (4B: ~5 ms) |
| Prefill GEMMs at n = 64 to GEMV bandwidth | prefill64 | ~10–20 % of the GEMM time |
| Layer GEMVs from 90–93 % to the lm_head's 95 % (4B), fill/drain of small GEMVs (0.6B) | decode | 4B ~3 %, 0.6B ~11 % |
| Small decode kernels: fewer of them (q + kv inputs in one kernel, norms into GEMV epilogues) | decode | 0.6B ~10 %, 4B ~3 % |
| Attention decode (latency-bound at 1k positions) | decode, long contexts | 0.6B ~6 %, grows with context |
| `q_inputs_kernel` with several tokens per block | prefill512 | 0.6B ~4 % |
| Graphs (or fewer launches) for short prefills | 0.6B prefill64 | ~9 % |

## Device sampling (2026-10-29)

Temperature / top-k / top-p sampling now runs on the device inside the captured decode step
(`sample_token!`, `src/ops/tile/sampling.jl`), so sampled decoding is pipelined like greedy
(`generate_pipelined!`); greedy is top-1 through the same kernels, and `argmax_token!` is gone.
A token is a function of the logits and u = `sampling_uniform(seed, counter)` (a hash of a
per-generation seed and a device-side counter), so the device and `sample_host` pick the same
tokens and the tests compare them token by token. Device sampling covers top_k ≤ 64; other
settings (e.g. top_k = 0) still sample on the host, step by step. Baseline `sampling-before`
(a313e48), result `sampling-after`.

| | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| Decode, sampled (Qwen3 defaults: T 0.6, top-k 20, top-p 0.95) | 3.32 → **2.46 ms/token (1.35×)**, 407 tok/s | 13.34 → **12.72 ms/token (1.05×)** |
| Decode, greedy pipelined | 2.44 ms/token (unchanged) | 12.71 ms/token (unchanged) |
| Sampled vs greedy | +0.5 % | +0.1 % |

Kernel time per token inside the 0.6B graph (`bench/breakdown.jl`), chunk kernel + final block:

| Version | top-k 20 (µs) | top-1 / greedy (µs) | Note |
|---|---|---|---|
| Old argmax (greedy only) | — | 4.2 | |
| 1: repeated `maximum` + `argmax` over values and ids | 10.0 + 11.0 = 21.0 | 2.3 + 3.3 = 5.6 | |
| 2: `argmax` over ids only, values gathered from the logits | 8.4 + 10.8 = 19.2 | 2.2 + 4.1 | |
| `maximum` then `minimum` of the positions holding it | 8.9 + 11.2 = 20.1 | — | **reverted**: two plain reductions cost as much as one `argmax` |
| **3: argmax reducer without NaN handling** (`FirstMax`) | **7.4 + 9.8 = 17.2** | 2.2 + 4.0 = 6.2 | kept |
| Three levels (1024-logit chunks, 16-list merge, final) | 6.4 + 5.4 + 7.2 = 19.0 | — | **reverted** |
| 8192-logit chunks, 2048-candidate final tile | 9.7 + 8.1 = 17.8 | — | **reverted**: no better |

- **The cost is the number of sequential block-wide reductions**, not the data: selection is
  top_k repeated argmax per kernel, and one costs 0.3–0.5 µs on 128 threads whatever the
  tile (0.32 µs at 1,024 elements, 0.37 at 4,096, 0.48 at 8,192). Smaller tiles in more
  kernels only add reductions. Under `ncu` both kernels issue an instruction every 3–4
  cycles with ~8 % achieved occupancy: latency-bound, one warp per scheduler.
- **17 µs is 0.7 % of a 0.6B token**; going further needs fewer than top_k sequential
  reductions (e.g. a threshold found by counting, then sorting a small candidate set by
  pairwise comparison), not worth it at this size.

## GEMM occupancy (2026-10-29)

From the codegen audit (docs/CUTILE_QUIRKS.md): the prefill GEMM never set the `occupancy`
hint. With `occupancy = 2`, tileiras fits two blocks per SM by trimming each block's
registers and shared memory: grids of ≥ 2 blocks per SM gain latency hiding, smaller grids
only lose resources. `gemm_occupancy` applies it to grids of ≥ 160 blocks. Baseline
`gemm-occ-before`, result `gemm-occ-rule`:

| | everywhere | ≥ 2 blocks per SM only (kept) |
|---|---|---|
| 4B gate_up / qkv / SwiGLU, n = 512 (608 / 192 / 608 blocks) | 1.11× / 1.08× / 1.11× | 1.12× / 1.10× / 1.11× |
| 4B o / down, n = 512 (80 blocks) | 0.97× / 0.96× | 0.99× / 1.01× |
| 0.6B o / down, n = 512 (32 blocks) | **0.86× / 0.89×** | 1.00× / 0.95× |
| 0.6B gate_up / SwiGLU, n = 64 (192 blocks) | 1.22× / 1.19× | 1.22× / 1.19× |
| **Prefill 512**: 4B / 0.6B | 1.05× / 1.01× | **1.06× (47.8 → 45.2 ms) / 1.02×** |
| **Prefill 64**: 4B / 0.6B | 1.01× / 1.07× | 1.01× / 1.04× |

Verify-sized GEMMs (n = 2–8, bandwidth-bound) are within 1 % either way. The `latency`
hint did not help any GEMM or GEMV, and prefill attention stays at `occupancy = 2`
(3 measured the same).


## Speculative decoding, greedy with n-gram lookup (2026-10-29)

`generate!(…; speculative = NgramProposer(k = …))`: prompt lookup proposes up to k tokens,
one verify step of k + 1 tokens (all logits, argmax per position on the device, one graph
per step length) accepts the matching prefix plus a bonus token, and `truncate!` drops the
rejected keys and values. 200-token budget, BF16, best of 5, against greedy pipelined decoding
(both include the prefill):

| Prompt | Model | plain | k = 1 | k = 2 | k = 4 | k = 7 | tokens per step at k = 7 (acceptance) |
|---|---|---|---|---|---|---|---|
| rename two variables in a 15-line function (116 / 110 tokens out) | 0.6B | 424 tok/s | 1.65× | 2.22× | 3.19× | **4.15×** | 4.83 (0.63) |
| | 4B | 78 tok/s | 1.65× | 2.08× | 2.87× | **3.75×** | 4.07 (0.53) |
| short story (nothing to copy) | 0.6B | 434 tok/s | 0.96× | 0.97× | 0.97× | 0.96× | 1.05 (0.02) |
| | 4B | 80 tok/s | 1.05× | 1.05× | 1.05× | 1.05× | 1.08 (0.03) |

A verify step of 8 tokens costs little more than a decode step (the GEMMs at n = 8 still
read the weights once), so the speedup tracks tokens per step. With no matches the loss
on 0.6B is the host/GPU overlap of pipelined decoding: speculative steps synchronize to read
the accepted count before proposing. In BFloat16 the outputs can part from plain greedy at
near-ties (verify steps round differently than decode steps); in Float32 they are identical
(test/engine/speculative.jl).

**Sampling** (same day). For a deterministic proposer, rejection sampling (accept d with
probability p(d), else sample from p without d) is the same as sampling the target's token at
each position and accepting d when they agree. The verify step does that with the device
sampler and the counters plain decoding would use, so sampled speculative decoding picks the
tokens of plain sampled decoding: identical over 200 tokens in Float32 (4 seeds); in BF16 they
part after 15–45 tokens, sooner than greedy, as a sample flips whenever its uniform falls
within round-off of a cumulative boundary. With Qwen3's recommended sampling (temperature 0.6,
top-k 20, top-p 0.95), against pipelined sampled decoding:

| Prompt | Model | plain | k = 1 | k = 2 | k = 4 | k = 7 | tokens per step at k = 7 (acceptance) |
|---|---|---|---|---|---|---|---|
| rename two variables | 0.6B | 422 tok/s | 1.64× | 2.19× | 3.13× | **4.06×** | 4.83 (0.63) |
| | 4B | 78 tok/s | 1.65× | 2.08× | 2.86× | **3.73×** | 4.07 (0.53) |
| short story | 0.6B | 432 tok/s | 1.00× | 1.01× | 1.00× | 1.00× | 1.10 (0.03) |
| | 4B | 80 tok/s | 1.04× | 1.05× | 1.04× | 1.04× | 1.08 (0.03) |

The rename task is nearly deterministic at temperature 0.6 (the same steps as greedy).

## Draft-model speculative decoding (2026-10-29)

`DraftProposer(draft_model; k)`: Qwen3-0.6B drafts for Qwen3-4B on a cache of its own. Each
proposal rewinds the draft to the longest common prefix with the context, runs the one or
two new tokens through a captured step, then decodes k - 1 steps as a chain of CUDA graphs
(each samples the next draft on the device into the next step's ids); the host syncs once
per proposal. The draft decodes greedily even when the target samples, which keeps the
verification exact (the draft is a function of the context). `bench/speculative.jl`, 4B
target, 256 tokens, decode tok/s (from the first new token), best of 3:

| Prompt | plain | n-gram k = 2 / 4 / 7 | draft k = 2 / 4 / 7 | draft acceptance at k = 4, tokens per step |
|---|---|---|---|---|
| rename two variables (greedy) | 72 tok/s | 2.29× / 3.18× / **4.22×** | 2.28× / 2.89× / 3.62× | 0.96, 4.78 |
| add a docstring and comments | 79 | 1.37× / 1.46× / 1.53× | 1.64× / **1.78×** / 1.65× | 0.56, 3.24 |
| explain hash-map collisions | 79 | 1.09× / 1.09× / 1.11× | 1.59× / **1.63×** / 1.46× | 0.49, 2.94 |
| short story | 79 | 1.04× / 1.03× / 1.03× | **1.39×** / 1.35× / 1.09× | 0.36, 2.44 |
| *sampled (0.6, top-k 20, top-p 0.95):* rename | 78 | 2.09× / 2.92× / **3.87×** | 2.09× / 2.65× / 3.33× | 0.96, 4.78 |
| add a docstring and comments | 79 | 1.38× / 1.45× / 1.54× | 1.71× / **1.91×** / 1.79× | 0.62, 3.46 |
| explain hash-map collisions | 79 | 1.08× / 1.08× / 1.10× | **1.47×** / 1.39× / 1.13× | 0.38, 2.53 |
| short story | 79 | 1.02× / 1.02× / 1.02× | **1.42×** / 1.37× / 1.18× | 0.37, 2.49 |

A draft step of the 0.6B model costs about a fifth of a 4B decode step, so each draft
token must be accepted with a probability of roughly 0.2 or more to pay; at k = 7 the late
drafts on open text rarely are. n-gram lookup is free but only finds text the reply copies.
Candidates: pick k (or the proposer) from the measured acceptance, as DeepInfer's scheduler
does; combine both proposers (n-gram when it matches, the draft otherwise).

## EAGLE-3 drafting (2026-10-29)

`Eagle3Proposer(load_eagle3("AngelSlim/Qwen3-4B_eagle3", target); k)`: target steps keep
the residual stream after layers 2, 18 and 33 (`enable_aux!`, three strided copies per
step); the drafter's one Llama layer reads [norm(embed(next token)); norm(fc(those states))]
for the positions the target just verified (one graph per count), then its own output for
k - 1 chained steps (as for the draft model), over a 32k vocabulary mapped back with `d2t`.
A drafter step costs about a tenth of the 0.6B draft model's. Its acceptance matches
AngelSlim's report (accept length 2.07–2.10 at 2 drafts on vLLM; here 1.80–2.50 at k = 2)
and decays quickly along the chain, so k = 2 is best. `bench/speculative.jl`, 4B target, k = 2 / 4:

| Prompt | plain | n-gram | 0.6B draft | EAGLE-3 | EAGLE-3 tokens per step (acceptance) at k = 2 |
|---|---|---|---|---|---|
| rename (greedy) | 78 tok/s | 2.12× / 2.96× | 2.12× / 2.68× | **2.20×** / 2.25× | 2.50 (0.75) |
| document | 80 | 1.37× / 1.48× | 1.66× / **1.80×** | 1.74× / 1.75× | 2.00 (0.50) |
| explain | 80 | 1.10× / 1.10× | 1.60× / **1.64×** | 1.58× / 1.55× | 1.82 (0.41) |
| story | 80 | 1.04× / 1.04× | 1.39× / 1.36× | **1.51×** / 1.46× | 1.73 (0.36) |
| *sampled:* document | 80 | 1.39× / 1.46× | 1.72× / **1.92×** | 1.74× / 1.82× | 2.00 (0.50) |
| explain | 80 | 1.09× / 1.10× | **1.48×** / 1.41× | 1.46× / 1.40× | 1.67 (0.34) |
| story | 80 | 1.03× / 1.03× | 1.42× / 1.39× | **1.57×** / 1.54× | 1.80 (0.40) |

(n-gram at k = 7 stays best on the rename prompt: 3.92×.) EAGLE-3 trades acceptance for
cheap drafts and ends up level with the 0.6B draft model; this head was trained for two
drafts. The tests use GavinLucky/SGLang-EAGLE3-Qwen3-0.6B-SpecForge with Qwen3-0.6B.

## DFlash drafting (2026-10-30)

`DFlashProposer(load_dflash("z-lab/Qwen3-4B-DFlash-b16", target))`: five Qwen3 layers read
the residual stream after target layers 2, 10, 18, 26 and 34. One drafter step runs the
positions the target just verified (their keys and values, from `fc` of those states, stay in
the drafter's cache) and a block of [next token, 15 masks] that attends without the causal
mask (`attention(…; causal = false)`, a constant in both kernels); the target's lm_head over
the block gives 15 drafts at once. Verify steps now take up to 16 tokens (`MAX_VERIFY_TOKENS`).
The drafts match a Float32 transcription of z-lab's `dflash.py` (test/engine/dflash.jl), and
its acceptance on the "explain a hash map" prompt is the reference's (2.78 vs 2.76 tokens per
step over 128 tokens).

Costs at 400 positions: drafter step 3.17 ms whatever the context count (GEMMs 2.85 ms at
~650 GB/s for 1.85 GB of weights, 0.78 GB of which is the lm_head; 15 per-column sampling
launches 0.11 ms), verify of 16 tokens 13.3 ms against 12.7 ms for one token. A step costs
~1.3 decode steps, so DFlash pays from 1.3 tokens per step on. `bench/speculative.jl`, 4B
target, best k of each method (k = 15 for DFlash in every row):

| Prompt | plain | n-gram | 0.6B draft | EAGLE-3 | DFlash | DFlash tokens per step |
|---|---|---|---|---|---|---|
| rename (greedy) | 78 tok/s | 3.92× | 3.36× | 2.25× | **8.50×** | 11.0 |
| document | 80 | 1.55× | 1.80× | 1.75× | **3.00×** | 3.94 |
| explain | 80 | 1.12× | 1.63× | 1.58× | **2.50×** | 3.24 |
| story | 80 | 1.04× | 1.39× | 1.45× | **1.92×** | 2.51 |
| *sampled:* document | 80 | 1.54× | 1.92× | 1.82× | **3.17×** | 4.20 |
| explain | 80 | 1.10× | 1.48× | 1.46× | **2.30×** | 3.01 |
| story | 80 | 1.03× | 1.42× | 1.57× | **1.75×** | 2.31 |

Unlike the chained drafters, whose acceptance falls along the chain so that k = 2 to 4 is
best, a longer DFlash block costs nothing more to draft and little more to verify, so the
whole block always wins: per-draft acceptance drops (0.67 at k = 2 to 0.15 at k = 15 on
explain) while tokens per step keep rising. The contrast with EAGLE-3 is the point of the
method: both read the target's hidden states, but EAGLE-3 pays one pass per draft and DFlash
one pass per block. Candidates: write the drafts on the device straight into the verify
step's ids (the host now waits for them in between); one batched argmax instead of 15
sampling launches.

## Sampled drafts, the p/q rule (2026-10-30)

`DraftProposer(…; sample = true)` (also `Eagle3Proposer`, `DFlashProposer`; CLI
`--sample-drafts`): when the target samples, the drafter samples its drafts with the target's
parameters on its own logits, under a seed of its own, and records each draft's filtered
distribution q (its ≤ 64 survivors' ids and probabilities, written by the sampling kernel
itself). The verify step then runs `accept_token!` at each drafted position: accept d with
probability min(1, p(d)/q(d)), else sample from max(p − q, 0) (a (64, 64) comparison of the
two survivor lists), with a uniform stream of its own for the acceptance; the bonus position
samples as before. A rejected position never emits the draft, so the host's accept loop is
unchanged. The tokens follow the target's distribution, but are no longer plain decoding's.
The new kernels share the candidate selection with the plain sampler (`survivors`, `pick`),
which measured the same after the refactor (16.8 µs).

Greedy drafts are accepted with probability p(argmax q), sampled ones with Σ min(p, q). With
Qwen3's recommended sampling (T 0.6, top-k 20, top-p 0.95) the distributions are peaked and
the two are close. `bench/speculative.jl --sampled [--pq]`, 4B target, 256 tokens; best k of
each method, speedup over sampled plain decoding (greedy drafts → sampled drafts):

| Prompt | 0.6B draft | EAGLE-3 | DFlash | acceptance, draft k = 4 |
|---|---|---|---|---|
| rename | 3.35× → 3.34× | 2.25× → 2.19× | 8.42× → 7.57× | 0.96 → 0.96 |
| document | 1.92× → 1.82× | 1.82× → 1.77× | 3.17× → 2.78× | 0.62 → 0.57 |
| explain | 1.48× → **1.59×** | 1.46× → **1.58×** | 2.30× → 2.29× | 0.38 → 0.44 |
| story | 1.42× → 1.42× | 1.57× → 1.48× | 1.76× → **1.88×** (k = 4) | 0.37 → 0.39 |

Speedups follow tokens per step: the p/q verify costs no more than plain sampling per
column. Sampled drafts pay on the open "explain" prompt at every k (draft model k = 7: 1.13×
→ 1.33×) and lose 3–12 % on the code-documenting one; each row is a single sampled reply,
so a difference of one step moves it by several percent (DFlash k = 15 on rename: 10 → 11
steps). Greedy drafts stay the default. Sampled drafts should gain more at higher
temperatures, where p(argmax q) falls faster than Σ min(p, q).

## Weight-only INT8, and Apertus 1.5 8B (2026-10-31)

`load_model(...; quantize = :int8)`, `--int8` (CLI), `--quantize int8` (bench/run.jl, keys
`<model>-int8/...`): linear weights and the lm_head as Int8 with one Float32 scale per output
channel (symmetric, round to nearest, quantized on the GPU at load time); the embedding stays
BF16. Kernels accumulate the Int8 values in Float32 and scale the sums before the one
rounding to T: the GEMV (and its SwiGLU / xIELU epilogues) converts Int8 to Float32, the GEMM
converts Int8 tiles to BF16 for the tensor cores (|q| ≤ 127 is exact).

- **GEMV tiles**: half the BF16 width. TK × TM ≈ 4096 Int8 elements per tile (TK = 1024,
  TM = 4) was best on every Apertus 8B layer; BF16's TM = 8 was 7–13 % slower, TK ≥ 2048 with
  TM ≥ 4 much slower. With it, the INT8 GEMVs run at 85–93 % of the roofline (qkv 91 %, o 87 %,
  up + xIELU 91 %, down 85 %, lm_head 93 %).
- **Accuracy** (mean KL to the BF16 model over the fixture prompts; top-1 agreement):
  Qwen3-0.6B 0.032 (46/50), Qwen3-4B 0.009 (49/50). Apertus 1.5 8B in INT8 against
  transformers' Float32 run of the unquantized checkpoint on the CPU: argmax differs at 2 of
  103 positions, both near-ties (margins 0.2 and 0.02), and 24-token greedy continuations are
  identical on all three prompts.

End to end (`--seconds 2`, tile backend; decode at 1,024 positions):

| | BF16 decode | INT8 decode | BF16 prefill 512 | INT8 prefill 512 |
|---|---|---|---|---|
| Qwen3-0.6B | 409 tok/s | 539 tok/s (1.32×) | 58.2k tok/s | 53.9k tok/s (0.92×) |
| Qwen3-4B | 77.7 tok/s | 136.9 tok/s (1.77×) | 11.4k tok/s | 10.1k tok/s (0.89×) |
| Apertus 1.5 8B | does not fit (16.1 GB) | 80.5 tok/s | — | 4.8k tok/s |

Apertus 8B in INT8 holds 8.6 GB of weights (linear 7.0 GB, lm_head 0.54 GB, BF16 embedding
1.07 GB) and loads in 12 s. Its decode step (12.4 ms from a graph) reads ~7.5 GB of weights
plus the KV cache, ~82 % of the 736 GB/s roofline; the suite's 95 % counts the whole
embedding table as read, as it does for every model. INT8 prefill is 8–11 % slower than
BF16: the GEMM's Int8 → BF16 tile conversion is not hidden yet (down, K = 21504, is the
worst at 0.73× of its BF16 GEMM), so fused dequantization in the GEMM is the next lever.

## 16-token verify attention (2026-10-31)

Steps of 9–16 tokens (DFlash's verify block, and the drafter's own pass) ran the prefill
attention kernel: one block per query head walking every page, 16–32 blocks on 80 SMs, so
its time grows with the context (~1.7 µs per 64-key page). `DECODE_MAX_TOKENS` is now 16,
and those steps use the split-KV decode kernels with 8 splits (`VERIFY_SPLITS`) instead
of decode's 32: the combine merges every split, empty or not, and with 32 it set a floor
of 8 µs (0.6B) / 12 µs (4B) that lost to the prefill kernel below ~450 positions. The
choice cannot depend on the context, since a captured graph is replayed as it grows.

16-token attention per layer, µs (CUPTI, both kernels of a path):

| positions | 0.6B prefill kernel | 0.6B decode, 32 splits | 0.6B 8 splits | 4B prefill kernel | 4B decode, 32 splits | 4B 8 splits |
|---|---|---|---|---|---|---|
| 128 | 4.5 | 8.3 | 5.5 | 4.8 | 12.4 | 6.7 |
| 256 | 7.6 | 7.6 | 5.3 | 8.1 | 12.2 | 6.7 |
| 512 | 14.3 | 7.9 | 5.7 | 14.8 | 12.6 | 7.2 |
| 1024 | 27.4 | 8.8 | 6.7 | 26.9 | 13.6 | 9.5 |
| 2048 | 53.4 | 11.2 | 9.8 | 53.6 | 17.4 | 14.3 |

The decode kernel does not spill at 64 packed query rows (4B: group 4 × 16 tokens, 230
registers). End to end on Qwen3-4B with DFlash k = 15 (`bench/speculative.jl`, greedy),
the time per step is unchanged within the noise (±2 %; k = 7, which this does not touch,
moved as much): the chat replies stop at 300–850 positions, where the saving is 0.2–0.4 ms
of a ~16.5 ms step. 32 splits with the same limit was 3.5 % slower on the rename prompt.
The gain grows with the context: ~1.4 ms per verify step at 2,048 positions (36 layers ×
39 µs), ~8 % of a 4B verify, not measured end to end (the bench's replies are too short).

Measured since on the two long prompts of `bench/speculative.jl` (a 2.2k-token Julia file,
`bench/prompts/runner.jl`, to edit or explain; replies of 256 tokens, so 2,230–2,480
positions), greedy, `DECODE_MAX_TOKENS` 8 (before) against 16, same session:

| Prompt | plain | DFlash k = 7 (control) | DFlash k = 15 before → after | ms per verify step |
|---|---|---|---|---|
| long-edit | 77.0 / 77.0 | 367.0 / 367.1 | 419.2 → **454.2** (+8.3 %) | 19.1 → 17.6 |
| long-explain | 77.0 / 77.0 | 182.7 / 182.6 | 153.7 → **169.4** (+10 %) | 18.9 → 17.3 |

The ~1.5 ms per step matches the prediction. At these contexts the whole block still pays
on the edit prompt (8.0 tokens per step; 5.9× over plain) but not on the open one, where
k = 7 wins (2.37× vs 2.20×; 3.2 vs 2.9 tokens per step). Plain decoding runs at 77 tok/s
(80 on the short prompts), and the 2.2k-token prefill takes 203 ms (11k tok/s, ~78
TFLOP/s; 91 at 512 tokens). Rows of one run compare; between the full run (every method)
and the DFlash-only runs, long-explain's DFlash rows differed by up to 11 %.

## INT8 KV cache (2026-10-31)

`PagedKVCache(model; quantize = :int8)` (`--kv-int8` in the CLI, `--kv int8` in bench/run.jl,
keys `<model>-kvint8/...`): each cached key or value vector (one KV head at one position) as
Int8 with one Float32 scale, round to nearest, written once by the fused `attention_inputs`
kernel (`Int8Pages`, src/ops/quant.jl). Attention never dequantizes the cache: a key's scale
multiplies its row of scores after the QKᵀ product (Int8 tiles convert exactly to BF16 for
the tensor cores), and a value's scale multiplies its probability before the PV product. A
layer's keys and values share one Int8 array and one scale vector, so the kernels stay under
32 parameters (separate arrays would take the KV write kernel to 36). 132 bytes per vector
instead of 256 in BF16: Qwen3-4B holds 75k positions per GB instead of 39k.

**Accuracy: smoothing the keys.** With one scale per vector, Qwen3-0.6B's mean KL to a BF16
cache was 0.004–0.036 over the fixture prompts (INT8 weights: 0.032). The keys carry it:

| Qwen3-0.6B, chat prompt | layer 1 K | layer 2 K | layer 14 K | layer 28 K | values (any layer) |
|---|---|---|---|---|---|
| relative error, one scale per vector | 0.022 | 0.025 | 0.011 | 0.020 | 0.007–0.013 |
| one scale per channel (KIVI) | 0.005 | 0.004 | 0.008 | 0.005 | 0.008–0.016 |
| per vector, keys smoothed by the k_norm weight | **0.003** | **0.004** | 0.011 | **0.006** | — |
| largest / median channel max | 109 | 43 | 6.5 | 11 | 1.6–5 |

A few key channels are ~100× the others, which leaves few levels for the rest; per-channel
scales do not fit a cache written one position at a time. In Qwen3 (and Apertus) these
channels come from the per-channel weight w of the key norm, k = rope(rmsnorm(k) w), so the
cache stores k / c with c = max(|w_i|, |w_j|) for each rotary pair (i, j): the same factor on
both channels a rotation mixes, so it commutes with rope. Scores are unchanged by multiplying
the queries by c in the attention kernels (qᵀk = (c q)ᵀ(k / c)). No calibration; one extra
multiply per query tile. Mean KL to a BF16 cache, and largest logit difference to
transformers (BF16 cache: 0.63):

| Qwen3-0.6B, tile backend | chat | prose | code | max logit diff |
|---|---|---|---|---|
| one scale per vector | 0.030 | 0.0089 | 0.0043 | 1.75 |
| smoothed keys (kept) | **0.0051** | **0.0040** | **0.0022** | **0.88** |

**Speed** (Qwen3-4B, `--seconds 2`; pipelined greedy decode, decode attention op per layer):

| | BF16 cache | INT8 cache | |
|---|---|---|---|
| decode at 1,024 positions | 78.7 tok/s | 78.1 tok/s | 0.99× (attention is 4 % of the step) |
| decode at 32,768 positions | 51.3 tok/s | **57.8 tok/s** | **1.13×** |
| decode attention at 32,768 positions | 209 µs (642 GB/s, 87 %) | 142 µs (489 GB/s, 66 %) | 1.48× for 0.52× the bytes |
| prefill 512 | 44.9 ms | 46.0 ms | 0.98× |

The INT8 attention kernel is no longer bandwidth-bound. Under `ncu` (32k positions): 255
registers with local-memory spills (38 % of L1 traffic) and 3.1× the instructions of the
BF16 kernel (150 registers, no spills, 95 % of DRAM peak): the BF16 kernel feeds the MMAs
from shared memory, the INT8 one converts each (64, 128) tile to BF16 in registers.
Dropping the `latency` hints on its loads changed nothing. Next: convert in halves of the
head dimension, or quantize the queries to Int8 too and run QKᵀ as an Int8 MMA
(SageAttention), leaving only V to convert. At the bandwidth of the BF16 kernel, 32k-position
decode would reach ~63 tok/s.

## INT8 GEMM tiles (2026-10-31)

INT8 prefill was 8–11 % slower than BF16 although it reads half the weight bytes, and at 64
tokens, where a GEMM is bandwidth-bound, the INT8 GEMMs were 0.68–1.13× BF16. Under `ncu`
(Qwen3-4B down): at 512 tokens 255 registers and 1.6× BF16's instructions (byte shuffles
and conversions of the weight tile); at 64 tokens 29 % of DRAM peak against BF16's 90 %,
with 37 % of the stall samples on the conversion waiting for its global load: the Int8
tile is not staged through shared memory and overlapped like the BF16 operands. Neither
`latency` hints on the weight loads (1–8) nor swapping the operands (Xᵀ W, the weight in its
natural layout as the B operand) helped. Tiles and occupancy did: a sweep of 15 tiles ×
occupancy 1, 2 on the layers of Qwen3-0.6B, Qwen3-4B and Apertus 8B at n = 8, 64, 512 found
the latency-bound INT8 blocks always want two per SM, narrower tiles than BF16 at 512 tokens
(128 × 64 × 64), and at ≤ 64 tokens deeper K steps (32 × 32 × 256, or 64 × 64 × 128 on large
grids). `gemm_int8_config` encodes that: within 1.5 % of the best tile summed over the sweep
(worst case 1.11×), 1.23× faster than BF16's rule. Qwen3-4B, `--seconds 2`:

| | before | after | BF16 |
|---|---|---|---|
| GEMMs at 64 tokens (qkv / o / gate_up / down) | 72 / 46 / 155 / 126 µs | 44 / 26 / 99 / 62 µs (1.57–2.03×) | 60 / 36 / 175 / 85 µs |
| GEMMs at 512 tokens | 237 / 155 / 580 / 369 µs | 196 / 127 / 567 / 284 µs (1.02–1.30×) | 203 / 116 / 521 / 266 µs |
| prefill 64 | — | **9.55 ms (1.52× BF16)** | 14.49 ms |
| prefill 512 | 10.1k tok/s (0.89× BF16) | **11.07k tok/s (0.97× BF16)** | 11.40k tok/s |

At 512 tokens INT8 is still a few percent behind: the conversion in registers stays. A
weight layout prepacked for the conversion (as Marlin does), or conversion by bit tricks
(Int8 bytes into Float32 mantissas with a magic exponent) would be the next step.

## DFlash without a host round trip (2026-10-31)

A DFlash step was: drafter step, host waits for the drafts and uploads them into the verify
step, verify step, host waits for the target's tokens. Now the drafter's sampling writes the
drafts into a device vector (`propose_on_device!`), and `verify!(…; drafts)` copies them into
the verify step's ids on the stream, after the upload of its other inputs: the host enqueues
both steps back to back and waits once, after the verify step. The second wait remains: the
host needs the accepted count to truncate the cache and to pick the next drafter graph (one
per context length). The drafter's 15 per-position sampling calls (30 launches) became one
batched `sample_tokens!` (three launches: candidates over a (chunk, column) grid, one block
per column sampling with counter c + j − 1, and a one-block kernel advancing the counter, which
no block of the sampling kernel may do while others read it); same tokens and distributions
as the per-column calls.

Under the tracer (Qwen3-4B, k = 15, "explain"): GPU idle time inside generation fell 2.8×,
sampling 0.086 → 0.050 ms per step. End to end (`bench/speculative.jl`, greedy, same tokens and
steps before and after):

| Prompt | DFlash k = 7 | DFlash k = 15 |
|---|---|---|
| rename | — | 662.8 → 670.5 tok/s (+1.2 %) |
| explain | 192.5 → 194.8 (+1.2 %) | 202.2 → 204.9 (+1.3 %) |
| long-explain (2.2k tokens) | 182.6 → 184.7 (+1.2 %) | 169.4 → 171.5 (+1.2 %) |

As the step times predicted (3.17 ms drafting + 13.3 ms verify ≈ the 16.6 ms step), the host
was already mostly hidden: the round trip cost ~0.2 ms of a 16.5 ms step. What remains is the
step's own GPU time; the bigger lever for DFlash is choosing k per prompt (k = 7 beats 15 at long
contexts on open text) or verifying a tree of drafts.
