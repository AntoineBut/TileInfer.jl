# Performance log

What each milestone and each optimization bought, on Qwen3-0.6B in BF16 on the RTX 4080
SUPER (736 GB/s). Newest last. Numbers that informed a decision but did not pan out are kept,
with the reason, so they are not retried blindly.

## How things are measured

- **`bench/run.jl`** (see CLAUDE.md), minimum time over samples, `--seconds 2` for
  decisions; expect ±3–5 % between runs at the model level (the GPU also drives the desktop).
  - `model/decode_ctx1024/<be>`: one eager decode step after 1,023 cached positions, synced.
  - `model/decode_graph_ctx1024/tile`: the same step replayed from a CUDA graph (M6+), synced.
  - `model/generate64[_unpipelined]_ctx1024/tile`: 64 greedy tokens after 1,023 positions,
    per token (perf branch). The pipelined variant overlaps host and GPU, so only it shows
    the gaps between steps disappearing.
  - `model/prefill512/<be>`: a 512-token prompt in one step.
- **End-to-end race** (`bench/ref/tileinfer_race.jl`, `bench/ref/vllm_race.py`): 512-token
  prompt, 256 greedy tokens, prefill = latency of generating 1 token, decode =
  (256 − 1) / (latency(256) − latency(1)). Same prompt ids and the same first tokens on both
  engines.
- **Kernel breakdowns**: `CUDA.@profile trace = true` over 20 generated tokens, summing
  kernel durations per name (the tracer adds ~5 % to wall time).
- **Roofline for decode at 1,024 positions**: 1.19 GB of weights + 117 MB of KV cache per
  token ≈ 1.78 ms, i.e. ~560 tok/s.

## Summary

| Stage | Decode step | Decode tok/s | Prefill 512 | Prefill tok/s | What changed |
|---|---|---|---|---|---|
| M3 (reference ops, paged KV) | 38.89 ms | 26 | 76.3 ms | 6.7k | cuBLAS + broadcasts |
| M4 (cuTile kernels, reference attention) | 34.77 ms | 29 | 54.0 ms | 9.5k | norm, rope, kv_write, silu, embed, GEMV, GEMM |
| M5 (paged flash-attention) | 3.80 ms | 263 | 16.0 ms | 32k | prefill FA-2, split-KV decode |
| M6 (CUDA graphs) | 2.87 ms (graph) | 348 | 11.8 ms | 43k | allocation-free step, graph replay |
| perf branch | 2.47 ms (pipelined) | 405 | 9.0 ms | 57k | fusions, device sampling, host/GPU overlap |

From M3: decode 15.7× faster, prefill 8.5×. Race: **418 tok/s decode, 53k tok/s prefill**,
against vLLM's 404–422 tok/s and 45–47k tok/s on the same machine (see the end).

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

## Comparison with vLLM (same machine, `bench/ref`)

| Engine | Decode tok/s | Prefill tok/s |
|---|---|---|
| vLLM, CUDA graphs + torch.compile | 404–422 | 45–47k* |
| vLLM, `--eager` | 172 | 39k |
| TileInfer M6 | 343 | 40k |
| TileInfer perf branch | **418** | **53k** |

\* One vLLM run measured 13.5k tok/s prefill, presumably cold; the two later runs give 45–47k.
