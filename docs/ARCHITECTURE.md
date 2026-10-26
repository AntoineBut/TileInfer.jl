# TileInfer — Architecture Draft

A single-GPU, single-client LLM inference engine in Julia. Every hot kernel is
written in cuTile.jl. The goal is to learn how the engine works, so the code
favours being clear and easy to instrument over supporting many features.

Status: draft v0 (2026-10-26), M0 done. Target hardware: RTX 4080 SUPER (Ada, sm_89, 16 GB).
External documentation and papers are listed in [REFERENCES.md](REFERENCES.md).

---

## 1. Design principles

1. **Copy vLLM's and DeepInfer's structure, not their size.** Keep the layers they
   have (engine → scheduler → model runner → model → kernels) and leave out
   distributed execution, the multi-tenant scheduler and the request queues.
2. **Plain Julia structs, no NN framework.** We never need autodiff. A model
   is a struct of `CuArray`s plus an explicit `forward!`. This makes every
   tensor allocation, layout and kernel launch visible.
3. **Kernels behind a narrow interface.** Each op (`rmsnorm!`, `rope!`,
   `linear!`, `attention_prefill!`, `attention_decode!`, …) has a
   *reference* implementation (CPU / cuBLAS / broadcast) and a *cuTile*
   implementation. Both are selected by a backend type parameter, so every new kernel is
   correctness-tested against the reference and benchmarked from day one.
4. **Element type as a type parameter everywhere** (`DecoderLM{T}`,
   `PagedKVCache{T}`). Only BF16 is supported early on, but FP16/FP32 (and quantized
   types later) only need a recompile of the kernels.
5. **No allocations in the decode loop.** Buffers are preallocated, so the
   decode step can be captured in a CUDA graph.

---

## 2. Layered architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│ CLI  (tileinfer pull | ls | rm | run | chat | bench)                     │
├──────────────────────────────────────────────────────────────────────────┤
│ Hub / ModelStore      HuggingFaceApi.jl → ~/.cache/huggingface/hub      │
│ Tokenizer + ChatTemplate (tokenizer.json, tokenizer_config.json)         │
├──────────────────────────────────────────────────────────────────────────┤
│ LLMEngine   add_request / step / generate (streaming iterator)           │
│   ├─ Scheduler        (single sequence: prefill chunks → decode steps)   │
│   ├─ BlockManager     (page allocator + block tables, prefix reuse later)│
│   ├─ SpecDecoder      (proposer + verifier + rejection sampler)          │
│   └─ Sampler          (greedy / temperature / top-k / top-p on GPU)      │
├──────────────────────────────────────────────────────────────────────────┤
│ ModelRunner  builds InputMetadata, owns static buffers, CUDA graphs      │
├──────────────────────────────────────────────────────────────────────────┤
│ Model  DecoderLM{T}: Embedding → N × DecoderLayer → RMSNorm → LMHead     │
│   (Llama 3 / Qwen2.5 / Qwen3 / Mistral selected by config flags)         │
├──────────────────────────────────────────────────────────────────────────┤
│ Ops (backend-dispatched)                                                 │
│   RefBackend  : CUDA.jl broadcast + cuBLAS (+ CPU for tests)             │
│   TileBackend : cuTile.jl kernels                                        │
├──────────────────────────────────────────────────────────────────────────┤
│ CUDA.jl 6 (CUDACore): CuArray, streams, events, graphs, memory pool      │
└──────────────────────────────────────────────────────────────────────────┘
```

### Mapping to DeepInfer / vLLM

| TileInfer            | DeepInfer (`deepinfer/…`)                         | vLLM (v1)                     |
|----------------------|---------------------------------------------------|-------------------------------|
| `LLMEngine`          | `engine/llm_engine.py` (`LLMEngine.step`)         | `v1/engine/core.py`           |
| `Scheduler`          | C++ scheduler (`scheduler.schedule`)              | `v1/core/sched/scheduler.py`  |
| `BlockManager`       | `worker/cache_engine.py` (GPU cache interface)    | `v1/core/kv_cache_manager.py` |
| `ModelRunner`        | `worker/model_runner.py`                          | `v1/worker/gpu_model_runner.py` |
| CUDA graph capture   | `worker/cuda_graph_runner.py` (batch-size buckets)| `CUDAGraphWrapper`            |
| `InputMetadata`      | `model_executor/input_metadata/`                  | `AttentionMetadata`           |
| `Sampler`            | `model_executor/layers/sampler.py`                | `v1/sample/sampler.py`        |
| `SpecDecoder`        | MTP / NextN draft (`mtp_info_pool.py`, `dflash`)  | `v1/spec_decode/`             |
| `ops/` cuTile kernels| `layers/tilelang_ops`, FlashMLA, DeepGEMM         | `csrc/`, FlashAttention, FlashInfer |

DeepInfer's `step()` has the same shape we want:
`schedule → (copy/clone KV blocks) → execute_model_step → sample → post_process`.
It also overlaps CPU scheduling of step *n+1* with GPU execution of step *n*
("async scheduling"). With one client and CUDA graphs this matters less, but the
overlap is worth adding later. The released DeepInfer is Blackwell- and
multi-GPU-oriented, and its KV-cache system is not included in the release. Use
vLLM as the reference for the paged KV design.

---

## 3. Module layout (proposed)

```
TileInfer/
├── Project.toml                # [apps] tileinfer = {} (Julia ≥ 1.12 Pkg apps)
├── src/
│   ├── TileInfer.jl
│   ├── config.jl               # ModelConfig parsed from config.json
│   ├── hub/
│   │   ├── store.jl            # pull/list/remove via HuggingFaceApi.jl
│   │   └── loader.jl           # safetensors (mmap) → CuArray, name mapping
│   ├── tokenizer/
│   │   ├── bpe.jl              # tokenizer.json → BytePairEncoding.jl
│   │   └── chat_template.jl    # Jinja subset / per-family templates
│   ├── model/
│   │   ├── layers.jl           # Embedding, Linear, RMSNorm, RoPE, MLP, Attention
│   │   ├── decoder.jl          # DecoderLayer, DecoderLM{T}, forward!
│   │   └── families.jl         # llama / qwen2 / qwen3 / mistral flags
│   ├── kv/
│   │   ├── paged_cache.jl      # PagedKVCache{T}: per-layer page pools
│   │   └── block_manager.jl    # free list, block tables, truncate (spec rollback)
│   ├── engine/
│   │   ├── sequence.jl         # Sequence, SamplingParams, status
│   │   ├── scheduler.jl
│   │   ├── model_runner.jl     # InputMetadata, static buffers, graph cache
│   │   ├── sampler.jl
│   │   ├── spec_decode.jl      # proposers + verifier
│   │   └── engine.jl           # LLMEngine, generate / stream
│   ├── ops/
│   │   ├── interface.jl        # abstract backend + op signatures
│   │   ├── reference.jl        # RefBackend
│   │   └── tile/               # TileBackend: one file per kernel family
│   │       ├── norm.jl  rope.jl  gemm.jl  gemv.jl  act.jl
│   │       ├── flash_attn.jl   paged_attn.jl   kv_write.jl
│   │       └── sampling.jl     spec_verify.jl
│   └── cli.jl
├── test/                       # per-op tests vs reference; e2e vs HF logits
├── bench/                      # per-kernel + end-to-end tok/s, NVTX ranges
└── docs/
```

---

## 4. Data layout conventions (Julia is column-major)

Julia arrays are column-major and PyTorch tensors are row-major. If we reverse
every shape, the memory stays identical, so that is the convention here (it is
also what cuTile.jl's FMHA example does):

| Tensor            | PyTorch (row-major)          | TileInfer (column-major)               |
|-------------------|------------------------------|----------------------------------------|
| hidden states     | `(tokens, hidden)`           | `(hidden, tokens)`                     |
| Q                 | `(tokens, heads, d)`         | `(d, heads, tokens)` or `(d, tokens, heads)` per kernel |
| Linear weight     | `(out, in)`                  | `(in, out)` → `y = Wᵀ x`               |
| Paged K/V cache   | `(pages, page, kvh, d)`      | `(d, page, kvh, pages)` per layer      |
| Block table       | `(seqs, max_pages)`          | `(max_pages, seqs)` Int32              |

- **Weights load without a transpose copy.** SafeTensors.jl 1.3 memory-maps
  the file and returns a `PermutedDimsArray` over the row-major buffer.
  `parent()` of it is exactly the `(in, out)` column-major array, so it can go
  straight into a `CuArray` (in BF16 via BFloat16s.jl).
- Put the reduction dimension first (contiguous) for GEMM/GEMV.
- One K and one V pool per layer, `CuArray{T,4}` of shape
  `(head_dim, PAGE, n_kv_heads, n_pages)`, with `PAGE` = attention `TILE_N`
  (64 or 128). A KV tile is then exactly one page, and a `ct.load` with a
  runtime page index is a single dense, aligned tile load.

---

## 5. Model

One generic `DecoderLM{T}` covers the Llama-style family. Per-family
differences are config flags:

| Flag                  | Llama 3.x | Qwen2.5 | Qwen3 | Mistral |
|-----------------------|-----------|---------|-------|---------|
| QKV bias              |           | ✓       |       |         |
| q/k RMSNorm           |           |         | ✓     |         |
| tied embeddings       | small     | small   | small |         |
| RoPE scaling          | llama3    | (yarn)  |       |         |
| sliding window        |           |         |       | (v0.1)  |

Suggested targets on 16 GB in BF16: **Qwen3-0.6B / 1.7B / 4B**, and
**Llama-3.2-1B / 3B**. For speculative decoding: Qwen3-4B target with
Qwen3-0.6B draft (same tokenizer, about 9.5 GB combined, which leaves room for
the KV cache).

Forward pass for one step (`T` = tokens in this step):

```
x = embed[ids]                                  # gather          (H, T)
for layer in layers
    h = rmsnorm(x)                              # fused add+norm after layer 0
    qkv = linear(h, Wqkv)                       # one fused projection
    q, k = rope(q, k, positions)  (+ q/k norm)  # fused
    kv_write!(cache, k, v, slot_mapping)        # scatter into pages
    o = attention(q, cache, metadata)           # prefill | decode | verify
    x = x + linear(o, Wo)
    h = rmsnorm(x)
    x = x + linear(swiglu(linear(h, Wgate_up)), Wdown)
logits = linear(rmsnorm(x)[:, last_positions], Wlm)
```

The checkpoint's separate `q_proj`, `k_proj`, `v_proj` and `gate_proj`, `up_proj` weights are concatenated once at load time
(as vLLM does), so there are fewer, larger GEMMs.

---

## 6. KV cache & block manager

- At startup, measure free memory after loading weights and allocating
  activations. Allocate `n_pages` for a configured fraction of what remains
  (vLLM's `gpu_memory_utilization`).
- `BlockManager`: a free list of page ids, and for each sequence a block table
  (`Vector{Int32}`, mirrored into a device `CuMatrix{Int32}`) plus `seq_len`.
- `slot_mapping` (token → `(page, offset)`) is computed on the host per step,
  uploaded into a static device buffer, and consumed by `kv_write!`.
- **Truncating is free.** Rolling back a sequence (rejected speculative
  tokens, or a regenerate in chat) only decrements `seq_len` and returns
  surplus pages. No data is moved.
- Later: **prefix caching** (hash full pages of the token prefix, refcount pages).
  This is useful with a single client too, since multi-turn chat re-sends the
  whole history.

Even with a single sequence, paging pays off: chat sessions can grow without
reallocation, rollback is trivial, prefix caching becomes possible, and the
attention kernels are the real thing. "Batch" in the kernels is
always ≥ 1 (target + draft, or later several sequences), so we never bake in a
batch size of 1.

---

## 7. Kernels (cuTile.jl)

| # | Kernel | Phase | Notes |
|---|--------|-------|-------|
| 1 | `rmsnorm!` (+ fused residual add) | prefill/decode | row reduction per token; start from `examples/layernorm.jl` |
| 2 | `rope!` (+ Qwen3 q/k norm) | both | in-place on Q/K, positions from metadata |
| 3 | `kv_write!` | both | `ct.scatter` or per-page `ct.store` driven by `slot_mapping` |
| 4 | `linear!` GEMM | prefill | start from `examples/matmul.jl`; cuBLAS as baseline |
| 5 | `linear!` GEMV / skinny GEMM | decode | memory-bound; split-K, wide loads; the most important kernel for decode tok/s |
| 6 | `silu_mul!` | both | later fused into the gate_up GEMM epilogue |
| 7 | `flash_attn_prefill!` | prefill | FA-2 online softmax; start from `examples/fmha.jl` (already GQA + causal); extend to varlen + paged K/V |
| 8 | `paged_attn_decode!` | decode/verify | flash-decoding: split KV across CTAs, then a combine kernel; GQA packing |
| 9 | `sample!` | both | argmax / temperature / top-k / top-p (sort-free: threshold search or Gumbel-max) |
| 10 | `spec_verify!` | spec | parallel rejection sampling over k drafts, accept-length per sequence |
| 11 | `embedding!` | both | `ct.gather` on rows |

### Attention: one kernel family, three uses

The attention variants differ in the number of query rows per sequence:

- **chunked prefill**: `M = chunk` (e.g. 512) against past pages plus the current chunk
- **decode**: `M = 1`
- **spec verify**: `M = k + 1` against past pages, with a causal mask among the k+1 tokens

So the core is one **paged FlashAttention** kernel, with the KV loop walking
the block table:

```julia
for j in 0:n_pages-1
    page = block_table[j + 1, seq]                       # scalar load
    k = ct.load(Kc; index=(1, 1, kvh, page), shape=(D, PAGE, 1, 1))
    ...online softmax, muladd(v, p, acc)...
end
```

With Tile IR ≥ 13.3, *sparse views* (`@view Kc[:, :, kvh, page_tile]`) gather
several pages in one load. Measure both.

When M is small (decode/verify), a 128-row Q tile is mostly empty. Two fixes:
- **GQA packing**: put the `group_size × (k+1)` query rows that share a KV head
  into the M dimension of one tile (Qwen3-4B: 32 Q / 8 KV heads → 4 rows per token).
- **Split-KV (flash-decoding)**: grid over `(kv_heads, splits)`, write partial
  `(acc, m, l)` in FP32, then run a small combine kernel. This is needed to fill
  80 SMs at batch 1. Use an FP32 workspace: BF16 atomic add needs sm_90.

### Ada-specific constraints (from cuTile.jl compatibility docs)

- Tile IR on sm_89 needs bytecode ≥ v13.2. cuTile.jl 1.0 targets v13.1–13.4.
  Driver ≥ 580 (this machine: 615).
- Supported element types: BF16/FP16/TF32/FP32. **No FP8 in Tile IR on Ada**,
  even though the hardware has it, so quantization would mean INT8 or
  weight-only schemes.
- No TMA, wgmma or thread-block clusters (`num_cta_in_cga = 1`), and no
  programmatic dependent launch. Kernels compile to `mma.sync`-style code, so
  tile-size tuning and the `latency` hints matter most.
- The 4080 SUPER has about 736 GB/s of memory bandwidth, so the decode roofline
  ≈ 736 GB/s ÷ weight bytes: about 90 tok/s for Qwen3-4B in BF16 and about 580 tok/s
  for 0.6B. These are the numbers to aim for.

---

## 8. Speculative decoding

Build it in stages. Each stage reuses the verifier.

1. **Verifier + rejection sampler** (model-agnostic): run the target on
   `[last_token, d1..dk]` (M = k+1 per sequence, paged causal attention),
   accept the longest valid prefix (greedy match, or Leviathan/Chen rejection
   sampling for temperature > 0), sample a bonus token, and truncate the KV of
   rejected positions.
2. **N-gram / prompt-lookup proposer**: no draft model, all on the CPU. It
   exercises the full verify path and is very effective on chat or code that
   repeats earlier context.
3. **Draft-model proposer**: Qwen3-0.6B drafting for Qwen3-4B. It has its own
   `DecoderLM` and its own `PagedKVCache` but shares the `BlockManager` logic. The k
   draft steps form a natural CUDA graph.
4. Later: **EAGLE / MTP-style heads** (DeepInfer uses MTP/NextN with
   `max_draft_len = 5`), dynamic k based on acceptance rate (DeepInfer's
   scheduler adapts using measured forward/sample times).

The engine's `step()` becomes: `propose → verify → accept → rollback → emit
1..k+1 tokens`.

---

## 9. ModelRunner & CUDA graphs

- `InputMetadata`: positions, slot_mapping, block_tables, seq_lens,
  query_start_loc (varlen), logits indices. They live in static device buffers sized
  for the maximum and are filled by `copyto!` from pinned host buffers.
- **Eager prefill**, **graph-captured decode/verify**. Graphs are keyed by
  `(num_seqs, tokens_per_seq)` buckets, as in DeepInfer's
  `build_graph_batch_sizes`. Use CUDA.jl's `capture` / `@captured` /
  `instantiate`.
- Graphs matter a lot in Julia. Each `@cuda` launch has host-side overhead
  (argument conversion, cache lookup), and a Qwen3 layer is about 10 kernels, so 28
  layers is about 300 launches per token. At around 5 µs each that is about 1.5 ms per token, a
  significant share of a 0.6B model's roughly 2 ms budget.
- Spike early: confirm that cuTile launches (`cudacall` underneath) are
  capture-safe and that no host sync or allocation happens inside the step.

---

## 10. Compilation & startup latency

- cuTile specializes on `ct.Constant` arguments (tile sizes, head_dim, flags)
  and element types. It caches compiled CUBINs across sessions in Julia's
  object cache (the `disk_cache` preference, on by default).
- `TileInfer.warmup(model)` compiles every kernel specialization the model needs
  at load time. Do it inside a `PrecompileTools` workload for the Julia side.
- CLI start time: install with **Julia 1.12 Pkg apps** (`[apps]` in
  Project.toml → `~/.julia/bin/tileinfer`). Keep `chat` as a long-lived
  process. `juliac --trim` is not realistic with the GPU compiler stack.

---

## 11. CLI

```
tileinfer pull Qwen/Qwen3-4B [--revision main]   # *.safetensors, config, tokenizer
tileinfer ls                                     # cached models (shared with Python HF cache)
tileinfer rm Qwen/Qwen3-4B
tileinfer run Qwen/Qwen3-4B -p "Hello" [--max-tokens 256 --temp 0.7 --top-p 0.9]
tileinfer chat Qwen/Qwen3-4B [--draft Qwen/Qwen3-0.6B | --ngram 3] [--k 4]
tileinfer bench Qwen/Qwen3-4B [--prompt-len 512 --gen 256] [--backend ref|tile]
```

`pull` = `HuggingFaceApi.list_repo_files` filtered to the allowed patterns, then
`hf_hub_download` per file. This gives the same cache layout as Python's
`huggingface_hub` and honours `HF_TOKEN`, `HF_HOME` and `HF_HUB_OFFLINE`.

---

## 12. Reused Julia packages

Versions are as of 2026-10-26, checked against the repos.

| Package | Use | Verdict |
|---------|-----|---------|
| **cuTile.jl** 1.0 (JuliaGPU) | all hot kernels | **Core.** Has `load/store/gather/scatter`, sparse views, `muladd` (MMA), reductions, atomics, `ct.Constant` specialization, CUBIN disk cache. FMHA, matmul, layernorm, softmax and MoE examples are good starting points. |
| **CUDA.jl** 6.3 (CUDACore + libs) | arrays, streams, events, graphs, memory pool, cuBLAS baseline, NVTX | **Core.** Held at 6.3: the registered cuTile 1.0.0 caps `CUDA_Compiler_jll` at 0.5, while CUDA 6.4 needs 0.6 (cuTile main already allows it). |
| **HuggingFaceApi.jl** 0.2 | `pull`, cache, auth, `list_repo_files` | **Use.** Light deps (Downloads, JSON, ProgressMeter), Python-compatible cache, actively maintained (Oct 2026). No snapshot function, but it takes about 10 lines on top of `list_repo_files`. |
| **SafeTensors.jl** 1.3 (FluxML) | weight loading | **Use.** Memory-maps files, supports BF16 (via BFloat16s) and FP8 (DLFP8Types). Take `parent()` of the returned permuted view for zero-transpose upload. |
| **BFloat16s.jl** | host-side BF16 | Use (already pulled in by cuTile). |
| **BytePairEncoding.jl** / **TextEncodeBase.jl** | — | **Not used.** TextEncodeBase 0.8.3 fails to precompile on Julia 1.13 (a docstring `@macroexpand` hack in `vocab.jl`). Replaced by our own byte-level BPE (`src/tokenizer/bpe.jl`), tested against HF `tokenizers` fixtures. |
| **Transformers.jl** | — | **Reference only, do not depend on it.** Last commit 2025-03. Pins `CUDA = "5"`, Flux 0.14 and Zygote, which conflicts with cuTile/CUDA 6. Built for training and research, with no paged KV. Useful for its HF config/weight-name mapping and tokenizer loading code (llama, phi, gpt-neox implementations). |
| Flux.jl / Lux.jl / NNlib.jl | — | **Skip.** No autodiff needed. Layer abstractions would hide exactly what we want to see. Plain Julia/CPU references are enough for tests. |
| NeuralAttentionlib.jl | — | Optional test oracle for attention variants; plain Julia is probably simpler. |
| **OteraEngine.jl** 1.1 | — | **Not used for now.** Chat templates are ported by hand per family (`src/tokenizer/chat_template.jl`, Qwen3 without tools so far) and checked against the model's real Jinja template rendered by jinja2 (`scripts/gen_fixtures.py chat`). |
| **Comonicon.jl** | CLI subcommands | Use, or keep a tiny hand-written argument parser plus Pkg apps. |
| JSON.jl | config.json, tokenizer.json | Use (already a transitive dep). |
| NVTX.jl, BenchmarkTools.jl | profiling (Nsight Systems/Compute), micro-benchmarks | Dev deps. |
| PythonCall.jl (test env only) | golden logits/tokens from HF `transformers` + `tokenizers` | Strongly recommended for e2e tests. |

---

## 13. Roadmap

| Milestone | Deliverable | Success criterion |
|-----------|-------------|-------------------|
| M0 ✅ | Julia + CUDA.jl + cuTile.jl installed; run `examples/fmha.jl`, `matmul.jl` on the 4080S; CUDA-graph capture spike with a cuTile kernel | done 2026-10-26: `test/smoke/` 21/21 pass, `bench/m0_launch_overhead.jl` |
| M1 ✅ | `pull/ls/rm`; config + safetensors loader; tokenizer (+ tests vs Python) | done 2026-10-26: own byte-level BPE matches HF `tokenizers` on the fixture corpus; Qwen3 chat template matches its Jinja template; weights load (fused qkv / gate_up, tied lm_head) in 0.17 s from page cache |
| M2 | `DecoderLM` with **RefBackend** (cuBLAS + broadcasts, contiguous KV) | logits match HF within BF16 tolerance; `run` produces text |
| M3 | Paged KV + BlockManager; `chat` with multi-turn | same outputs as M2 |
| M4 | cuTile kernels one by one (norm, rope, kv_write, silu, GEMV, GEMM) | each passes op tests; per-kernel bench vs cuBLAS / reference |
| M5 | Paged flash-attention (prefill, decode split-KV, verify) | correctness + % of bandwidth/FLOP roofline |
| M6 | CUDA graphs for decode; NVTX timeline | ≥ 80 % of the decode roofline on Qwen3-0.6B/4B |
| M7 | Spec decoding: verifier → n-gram → draft model | acceptance-rate + tok/s speedup report |
| M8+ | prefix caching, fused epilogues, async scheduling, INT8/weight-only quant, multi-sequence | — |

## 14. Open questions / spikes

Answered by M0 (environment: Julia 1.13.3, CUDA.jl 6.3, cuTile 1.0.0, tileiras 13.3 / bytecode v13.3, driver 615):

1. ✅ **cuTile launches capture into CUDA graphs.** Capture works when mixed with CUDA.jl
   broadcasts and cuBLAS `mul!`, and replays read the live contents of captured
   buffers (`test/smoke/graphs.jl`). Kernels must be compiled (launched once)
   before capture.
2. ✅ **Host overhead of a cuTile launch ≈ a native `@cuda` launch:** about 5.2 vs 5.5 µs
   to enqueue. `cuLaunchKernelEx` itself is about 1.8 µs; the rest is Julia-side
   argument handling. Graph replay brings enqueue cost per kernel to about 0.01 µs.
   For tiny dependent kernels, wall time is about 5–6 µs per kernel either way,
   because the GPU-side launch and DRAM round trip set the floor (possibly also
   idle clocks). Graphs therefore matter whenever per-kernel GPU time is under
   ~5 µs, which is the case for norm, rope and activation kernels at batch 1.
   Fusion still matters even with graphs.

Still open:

3. Paged KV loads: a runtime-indexed `ct.load` per page vs sparse-view gather.
4. GEMV performance in cuTile at M = 1 (and M = k+1) vs cuBLAS. This decides
   whether to keep a cuBLAS fallback for decode linears.

Resolved in M1:

5. ~~OteraEngine.jl coverage of real HF chat templates.~~ Not needed: templates are
   ported by hand per family and checked against jinja2's rendering of the real one.
