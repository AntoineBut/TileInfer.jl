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

| Stage | Decode step | Decode tok/s | Prefill 512 | Prefill tok/s | What changed |
|---|---|---|---|---|---|
| M3 (reference ops, paged KV) | 38.89 ms | 26 | 76.3 ms | 6.7k | cuBLAS + broadcasts |
| M4 (cuTile kernels, reference attention) | 34.77 ms | 29 | 54.0 ms | 9.5k | norm, rope, kv_write, silu, embed, GEMV, GEMM |
| M5 (paged flash-attention) | 3.80 ms | 263 | 16.0 ms | 32k | prefill FA-2, split-KV decode |
| M6 (CUDA graphs) | 2.87 ms (graph) | 348 | 11.8 ms | 43k | allocation-free step, graph replay |
| perf branch | 2.47 ms (pipelined) | 405 | 9.0 ms | 57k | fusions, device greedy sampling, host/GPU overlap |
| device sampling | 2.46 ms (sampled) | 407 | 9.0 ms | 57k | top-k/top-p sampling in the graph: sampled decode as fast as greedy |

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

