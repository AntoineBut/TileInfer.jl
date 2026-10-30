# TileInfer — Architecture Draft

A single-GPU, single-client LLM inference engine in Julia. Every hot kernel is
written in cuTile.jl. The goal is to learn how the engine works, so the code
favours being clear and easy to instrument over supporting many features.

Status (2026-10-30): M0–M7 done, and the first M8 fusions. M7 (speculative decoding): verifier
(greedy and sampled), n-gram, draft-model, EAGLE-3 and DFlash proposers, `--speculate` / `--draft` /
`--eagle` / `--dflash` and `bench/speculative.jl`; adaptive k is a later option.
Target hardware: RTX 4080 SUPER (Ada, sm_89, 16 GB). Benchmarked on Qwen3-0.6B and Qwen3-4B.
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

## 3. Module layout

As built so far; files for later milestones (scheduler, spec decoding) are marked *(planned)*.

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
│   │   ├── bpe.jl              # own byte-level BPE from tokenizer.json
│   │   └── chat_template.jl    # per-family templates, ported by hand
│   ├── model/
│   │   ├── weights.jl          # ModelWeights / LayerWeights (fused qkv, gate_up)
│   │   ├── rope.jl             # RotaryTable
│   │   └── decoder.jl          # DecoderLM{T}, StepBuffers, prepare_step!, run_step!, forward!
│   ├── kv/
│   │   ├── paged_cache.jl      # PagedKVCache{T}: per-layer page pools
│   │   └── block_manager.jl    # Sequence, free list, block tables, truncate (spec rollback)
│   ├── engine/
│   │   ├── runner.jl           # ModelRunner: CUDA graphs per token count, pipelined decode
│   │   ├── sampler.jl          # SamplingParams, sampling_uniform, host reference, TokenSampler
│   │   ├── generate.jl         # generate!, generation defaults, TextStream
│   │   ├── chat.jl             # ChatSession with prefix reuse
│   │   ├── speculative.jl      # n-gram and draft-model proposers, verify step (device sampling per position), speculative generate (M7)
│   │   ├── eagle.jl            # EAGLE-3 drafter (one layer on the target's hidden states) and its proposer
│   │   └── dflash.jl           # DFlash drafter (a block of drafts in one non-causal pass) and its proposer
│   ├── ops/
│   │   ├── interface.jl        # abstract backend + op signatures, fused-op fallbacks
│   │   ├── reference.jl        # RefBackend
│   │   └── tile/               # TileBackend: one file per kernel family
│   │       ├── norm.jl  elementwise.jl (rope, silu, kv_write, embed)  linear.jl (GEMV, GEMM)
│   │       ├── fused.jl        # attention inputs: q/k norm + rope + KV write
│   │       ├── attention.jl    # paged flash-attention: prefill, split-KV decode + combine
│   │       └── sampling.jl     # device top-k/top-p sampling, greedy as top-1 (spec_verify planned)
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
- **Prefix reuse in chat (M3):** a `ChatSession` keeps one sequence alive, re-renders
  the conversation each turn, truncates the sequence to the longest token prefix it
  shares with the new prompt, and prefills only the rest. Templates that rewrite
  history (Qwen3 drops earlier `<think>` blocks) shorten the reusable prefix to the
  start of the previous reply; that is the model's training format, so it is followed.
- Later: **prefix caching across sequences** (hash full pages of the token prefix,
  refcount pages), once several sequences share the pool.

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
| 7 | `attention_prefill!` | prefill | FA-2 online softmax from `examples/fmha.jl`, walking the block table one page (= one KV tile) per step; only diagonal pages masked |
| 8 | `attention_decode!` | decode/verify (≤ 8 tokens) | flash-decoding: GQA-packed query tile, fixed split count (grid independent of context), FP32 partials, combine kernel |
| 9 | `sample_token!` | decode | temperature / top-k (≤ 64) / top-p, greedy as top-1: per-chunk top-k by repeated argmax, then one block merges and samples by inverse transform with a counter-based uniform; done 2026-10-29 |
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
4. **EAGLE-3 drafter** (`src/engine/eagle.jl`): one layer reading the residual
   stream of three target layers, which the target's steps keep (`enable_aux!`);
   drafts chained one token at a time like the draft model.
5. **DFlash drafter** (`src/engine/dflash.jl`): five Qwen3 layers whose keys and
   values for context positions come from five target layers' states, and a block
   of [next token, 15 masks] attending without the causal mask (`attention(…;
   causal = false)`), so one pass drafts 15 tokens; verify steps take up to 16.
6. Later: dynamic k based on acceptance rate (DeepInfer's scheduler adapts using
   measured forward/sample times).

The engine's `step()` becomes: `propose → verify → accept → rollback → emit
1..k+1 tokens`.

---

## 9. ModelRunner & CUDA graphs

Implemented in M6 (`src/model/decoder.jl`, `src/engine/runner.jl`):

- **One forward pass, written in place.** `run_step!` runs a step entirely on the device,
  writing only into a `StepBuffers` (every activation, the logits) and the KV cache. Every
  op has an in-place form (`rmsnorm!`, `linear!`, …); TileBackend's allocate nothing, and
  RefBackend falls back to "compute, then copy", so both backends share the same layer
  loop. `forward!(model, cache, seq, ids)` is that step with buffers allocated for it.
- **Inputs on the device.** The step's metadata (the vLLM `InputMetadata` for one
  sequence): token ids, 0-based positions, cache slots, and the block table zero-padded to
  the pool's page count (a multiple of 16, so its length never changes). `prepare_step!`
  reserves pages and uploads them with `copyto!` before the step (synchronous pageable
  copies for now; pinned async copies later). Kernels read `start` and positions from
  these buffers, never as scalar launch arguments, which a graph would freeze.
- **`ModelRunner(model, cache)`.** Steps of up to 8 tokens (decode, spec verify) keep a
  `StepBuffers` per token count. The first step of a count runs eagerly (that compiles its
  kernels), then is captured with `CUDA.capture`/`instantiate`; later steps upload their
  inputs and `CUDA.launch` the graph. Longer steps (prefill) run eagerly. Graphs are
  keyed by tokens per step only, as there is one sequence; with several sequences the key
  becomes `(num_seqs, tokens_per_seq)` buckets, as in DeepInfer's `build_graph_batch_sizes`.
  The graph holds the addresses of weights, pools and buffers, so it is valid for the
  runner's lifetime. `generate!` and `ChatSession` (graphs persist across turns) use it.
- **Correctness:** a replayed step's logits are bit-identical to an eager step's, and a
  decode step allocates no GPU memory (`test/engine/runner.jl`).
- Graphs remove the host cost of the ~540 launches of a decode step (≈ 5 µs each), but not
  their GPU-side cost: tiny kernels still take 2–3 µs each, so fusion (M8+) is what
  reduces the step further. Replaying the graph itself costs ~0.34 ms of host time, which
  overlaps with the GPU.

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
tileinfer run Qwen/Qwen3-4B "Hello" [--max-tokens 256 --temperature 0.7 --top-k 20 --top-p 0.9 --seed 1 --thinking off --raw]
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
| M2 ✅ | `DecoderLM` with **RefBackend** (cuBLAS + broadcasts, contiguous KV) | done 2026-10-26: Float32 logits match HF to ~1e-4 with identical 24-token greedy continuations; BF16 within 0.625 (≈5 ulps), top-1 equal except one exact tie; `tileinfer run` streams text at ~27–29 tok/s decode (Qwen3-0.6B, reference path, ≈5 % of the roofline), first token after ~13 s in a fresh process (JIT) |
| M3 ✅ | Paged KV + BlockManager; `chat` with multi-turn | done 2026-10-26: paged reference path matches the HF Float32 fixtures (also with 4-position, fragmented pages); `tileinfer chat` reuses the cached common prefix across turns, and a reply with reuse equals a fresh run on the full prompt; decode ~20 tok/s (the reference attention gathers every page of the sequence at each step) |
| M4 ✅ | cuTile kernels one by one (norm, rope, kv_write, silu, GEMV, GEMM) | done 2026-10-27 (`src/ops/tile/`): element-wise kernels bit-identical to the reference, linear ≤ 1 BF16 ulp; whole model on TileBackend matches the HF Float32 fixtures (identical greedy continuations). Decode GEMV 68–93 % of the DRAM roofline (1.0–1.3× cuBLAS); prefill GEMM 0.70–0.92× cuBLAS; rope/rmsnorm/store_kv/embed 3–29× the reference. End to end: prefill 1.47×, decode only 1.13× — the reference attention fallback (≈1 ms/layer, ~6k GPU allocations per token, GC spikes) dominates until M5; tiny decode kernels sit at the ~5.5 µs launch floor until CUDA graphs (M6) |
| M5 ✅ | Paged flash-attention (prefill, decode split-KV, verify) | done 2026-10-27 (`src/ops/tile/attention.jl`): Float32 matches exact attention to 1e-5, BF16 within ~3.5 ulps (the eager reference, which rounds like PyTorch, is up to 12 ulps off); whole model on TileBackend matches the HF Float32 fixtures. Prefill (n = 512) 36 TFLOP/s, 42× the reference; decode at 1,024 positions 14 µs per call (73×), host-bound (two launches ≈ 15 µs to enqueue), with the split kernel at ~64 % of DRAM under `ncu`. End to end on Qwen3-0.6B: decode 264 tok/s (9.2× M4, 47 % of the roofline), prefill 32k tok/s (3.4×). Found along the way: launches with ≥ 32 kernel parameters take a slow host path (`store_kv` 1.7× faster with 2-D views), and a growing block table recompiled attention at page boundaries (now padded) |
| M6 ✅ | CUDA graphs for decode; NVTX timeline | done 2026-10-27: allocation-free in-place forward step with device-resident inputs (positions, slots, padded block table); `ModelRunner` captures steps of ≤ 8 tokens and replays them, bit-identical to eager steps; NVTX ranges for upload / replay / prefill / sample. Decode on Qwen3-0.6B: 348 tok/s through the graph (2.87 ms per step, 62 % of the roofline) vs 286 tok/s eager; prefill 43k tok/s (1.36× from static buffers). The step is now GPU-bound; performance work moved to M8+ |
| M7 ✅ | Spec decoding: verifier → n-gram → draft model | acceptance-rate + tok/s speedup report. Verifier + n-gram proposer done 2026-10-29 (`src/engine/speculative.jl`): verify steps (all-token logits + device sampling per position with plain decoding's counters, which for a deterministic proposer is rejection sampling) captured as graphs; same tokens and cache state as plain greedy and sampled decoding in Float32. Code-editing prompt: 4.15× (0.6B) / 3.75× (4B) at k = 7; open-ended text with nothing to copy 0.96× / 1.05×; the same with Qwen3's recommended sampling. Draft-model proposer (Qwen3-0.6B for 4B, greedy drafts as a chain of graphs) on chat prompts (`bench/speculative.jl`): 1.4–1.8× on explanations and stories, 1.8–1.9× on code edits, greedy or sampled; n-gram up to 4.2× when the reply copies the prompt. EAGLE-3 (AngelSlim/Qwen3-4B_eagle3: target layers 2/18/33 kept by the steps, one drafter layer, 32k draft vocabulary): 1.46–1.82× on open prompts, best on stories (1.57× sampled), acceptance as AngelSlim reports. DFlash (z-lab/Qwen3-4B-DFlash-b16, 2026-10-30; drafts match a transcription of z-lab's reference in Float32): one non-causal pass of five layers drafts 15 tokens for 3.2 ms (lm_head included, 80 % of its roofline), a 16-token verify costs 5 % more than a decode step; at k = 15, 8.5× on the rename prompt and 1.9–3.0× on open prompts (1.75–3.2× sampled), best of all proposers on every prompt. CLI `--speculate K`, `--draft MODEL`, `--eagle DRAFTER`, `--dflash DRAFTER`. Not done: the general p/q rule with sampled drafts |
| M8+ | Performance: fewer, fused kernels; device sampling; prefix caching, async scheduling, INT8/weight-only quant, multi-sequence | ≥ 80 % of the decode roofline on Qwen3-0.6B/4B (moved from M6). Partly done ahead of M7 (perf branch and per-shape GEMM tiles; see PERFORMANCE.md): q/k norm + rope + KV write fused, residual add + RMSNorm, SwiGLU and Float32-logit GEMV epilogues, pinned async uploads, device sampling (greedy, temperature / top-k / top-p) inside the graph with pipelined decode, per-shape GEMM tiles. Decode is at 73 % of the roofline on 0.6B and 88 % on 4B, sampled as fast as greedy (M8 target met on 4B only). Not pursued: attention combine fused into the split kernel (slower) |

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

3. Paged KV loads: a runtime-indexed `ct.load` per page vs sparse-view gather. M5 uses one
   runtime-indexed load per page; sparse views (several pages per load) are not measured yet.
4. ~~GEMV performance in cuTile at M = 1 vs cuBLAS.~~ Answered at M4: the cuTile GEMV is
   1.0–1.3× cuBLAS, so decode linears have no cuBLAS fallback. M = k+1 (verify) uses the
   GEMM with per-shape tiles; a split-K GEMM did not beat it at n = 2…64.

Resolved in M1:

5. ~~OteraEngine.jl coverage of real HF chat templates.~~ Not needed: templates are
   ported by hand per family and checked against jinja2's rendering of the real one.
