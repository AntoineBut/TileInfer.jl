# History and roadmap

TileInfer was built in milestones, each a working engine that the next one sped up or
extended. This page shows where it stands, how it got there, and what comes next. The
measurements behind every step, including the ideas that did not pan out, are in the
[Results by feature](performance.md).

## Where it stands

As of 2026-11-01, milestones M0–M7 are done and M8 (performance) is in progress. On the
RTX 4080 SUPER, in BFloat16, with decode measured after 1,023 cached positions:

| | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| Decode | 409 tok/s (73 % of the bandwidth roofline) | 79 tok/s (88 %) |
| Decode with INT8 weights | 542 tok/s | 139 tok/s |
| Prefill, 512-token prompt | 58k tok/s | 11.4k tok/s (91 TFLOP/s) |
| Speculative decoding (DFlash) | — | 1.8× to 3.2× on open prompts, 8.5× on a code edit |

Models: Llama 3, Mistral, Qwen3 and Apertus (Apertus 1.5 8B runs on 16 GB with INT8
weights, at 80 tok/s).

## Milestones

| | Milestone | Done | Outcome |
|---|---|---|---|
| M0 | Environment | 2026-10-26 | Julia 1.13, CUDA.jl and cuTile.jl on the 4080 SUPER; cuTile launches capture into CUDA graphs; a launch costs ~5 µs |
| M1 | Models from the hub | 2026-10-26 | `pull`, safetensors loading without transposes, own BPE tokenizer matching Hugging Face `tokenizers`, Qwen3 chat template |
| M2 | Reference model | 2026-10-26 | the whole model on broadcasts and cuBLAS; Float32 logits match transformers to ~1e-4 |
| M3 | Paged KV cache and chat | 2026-10-26 | block manager, paged reference attention, multi-turn chat with prefix reuse |
| M4 | cuTile kernels | 2026-10-27 | norm, RoPE, KV write, SwiGLU, embedding, GEMV, GEMM; the decode GEMV at 68–93 % of the roofline (1.0–1.3× cuBLAS) |
| M5 | Paged flash-attention | 2026-10-27 | prefill and split-KV decode kernels: attention 42× (prefill) and 73× (decode) faster than the reference |
| M6 | CUDA graphs | 2026-10-27 | allocation-free steps with inputs on the device, replayed from graphs |
| M7 | Speculative decoding | 2026-10-30 | exact verification for greedy and sampled decoding; n-gram, draft-model and DFlash proposers |
| M8 | Performance | in progress | fusions, device sampling, INT8 weights and KV cache, tuned tiles (below) |

Qwen3-0.6B along the way:

| Stage | Decode | Prefill (512 tokens) | What changed |
|---|---|---|---|
| M3 | 26 tok/s | 6.7k tok/s | reference ops: cuBLAS and broadcasts |
| M4 | 29 tok/s | 9.5k tok/s | cuTile kernels, but the reference attention still dominates |
| M5 | 263 tok/s | 32k tok/s | paged flash-attention |
| M6 | 348 tok/s | 43k tok/s | static buffers, decode replayed from a CUDA graph |
| M8 so far | 395–410 tok/s | 58k tok/s | fused kernels, sampling inside the graph, host one step behind the GPU, tuned GEMM tiles |

Each stage moved the bottleneck: M4's kernels were hidden behind the reference attention
(~6k GPU allocations per token), M5 left the step bound by launching ~540 kernels from the
host, and after M6 the step was bound by the GPU itself, where only fewer, better kernels
help.

## M8 so far

| Date | Change | Gain |
|---|---|---|
| 2026-10-27 | Fusions: q/k norm + RoPE + KV write, residual add + RMSNorm, SwiGLU in the GEMV; pinned async uploads; pipelined decode | 0.6B decode 348 → 405 tok/s |
| 2026-10-29 | Top-k/top-p sampling on the device, inside the graph | sampled decoding as fast as greedy |
| 2026-10-29 | GEMM tiles chosen per shape; two blocks per SM on large grids | within 2 % of the best tile, 1.06× cuBLAS on average |
| 2026-10-31 | Weight-only INT8 linear layers | decode 1.32× (0.6B), 1.77× (4B); Apertus 1.5 8B fits on 16 GB |
| 2026-10-31 | 16-token verify steps on the split-KV decode kernels | +8–10 % for DFlash at 2k-token contexts |
| 2026-10-31 | INT8 KV cache with smoothed keys; INT8 attention without spills | 4B decode at 32k positions 1.19× |
| 2026-10-31 | INT8 GEMM tiles and occupancy of their own | INT8 prefill 0.89× → 0.97× BF16, 1.5× BF16 at 64 tokens |
| 2026-10-31 | DFlash drafts handed to the verify step on the device | +1.2 % |
| 2026-11-01 | EAGLE-3 proposer removed | DFlash beat it on every prompt |

The M8 target is 80 % of the decode bandwidth roofline: met on Qwen3-4B (88 %), not on
Qwen3-0.6B (73 %), where per-kernel overheads weigh more.

## Next

Chosen for what each teaches and lets us demonstrate, not only for raw speed.

| Item | Why |
|---|---|
| Chunked prefill | a step's buffers are sized for the whole prompt (a 32k-token prompt on Qwen3-4B needs ~1.3 GB for one activation alone); fixed-size chunks bound that memory |
| Prefill attention for long prompts | at 4k–16k tokens attention catches up with the GEMMs; GQA packing in the prefill kernel and heaviest causal blocks first are the candidates (see [Attention kernels](attention.md#Open-questions)) |
| INT8 GEMM without register conversion | INT8 prefill is still a few percent behind BF16 because weight tiles are converted in registers; a prepacked layout (as Marlin does) would hide it |
| 512-token GEMMs at cuBLAS speed | they run at 0.87–0.92× cuBLAS, ~5 ms of a 45 ms Qwen3-4B prefill |
| Adaptive k, tree verification | choose the number of drafts per step from measured acceptance; verify several candidate branches at once (see [Speculative decoding](speculative.md#Not-built-yet)) |

Later, once the single-sequence engine is done: several sequences in flight (continuous
batching), prefix caching across sequences and cache sizing from free memory (see
[KV cache](kv_cache.md#Not-built-yet)), CPU scheduling overlapped with GPU execution, and
compiling every kernel a model needs at load time.

## Tried and not kept

| Idea | Why it was dropped |
|---|---|
| EAGLE-3 drafting | level with the 0.6B draft model, and DFlash beat it on every prompt |
| Attention combine fused into the split kernel | slower than a separate combine |
| Int8 tensor cores for QKᵀ in INT8 attention | as fast as converting to BF16, but quantizing the queries undid the accuracy that key smoothing gained |
| Split-K GEMM for verify steps | did not beat the per-shape GEMM tiles at 2–64 tokens |
| `latency` hints on loads | no measurable effect on any GEMM, GEMV or attention kernel |
| Sampled drafts by default | mixed results with Qwen3's sampling settings; kept as an option |
| INT4 or group-wise weight quantization | not tried: ~1.5× faster decode than INT8, but little to learn beyond what INT8 showed |

## Open questions

- **Several KV pages per load.** Tile IR ≥ 13.3 can gather several pages in one load through
  sparse views. The attention kernels use one runtime-indexed load per page; the alternative
  is not measured yet.
