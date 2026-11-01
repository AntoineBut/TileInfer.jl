# Architecture

```@meta
CurrentModule = TileInfer
```

TileInfer is a single-GPU, single-client inference engine for Llama-style models. It keeps
the layers of production engines such as [vLLM](https://github.com/vllm-project/vllm) and
DeepSeek's DeepInfer (generation loop, model runner, model, paged KV cache, kernels) and
leaves out everything that exists to serve many users at once: request queues, a batch
scheduler, distributed execution. What remains is small enough to read in an afternoon, and
every hot kernel in it is a [cuTile.jl](https://github.com/JuliaGPU/cuTile.jl) kernel.

This page describes the layers from the top down. The three most involved parts have pages
of their own: the [KV cache](kv_cache.md), the [attention kernels](attention.md) and
[speculative decoding](speculative.md). Progress and plans are in
[History and roadmap](roadmap.md).

## Design principles

1. **Copy the structure of vLLM and DeepInfer, not their size.** The same layers, one
   sequence at a time.
2. **Plain Julia structs, no NN framework.** Nothing needs autodiff. A model is a struct of
   `CuArray`s and an explicit forward pass, so every allocation, layout and kernel launch is
   visible.
3. **Kernels behind a narrow interface.** Every op has a reference implementation (CUDA.jl
   broadcasts and cuBLAS) and a cuTile one, selected by a backend type. Each new kernel is
   tested against the reference and benchmarked from the start.
4. **The element type is a type parameter** (`DecoderLM{T}`, `PagedKVCache{T}`). BFloat16
   by default; Float32 runs are how outputs are checked against Hugging Face transformers.
5. **No allocations and no host synchronization in a decode step**, so that it can be
   captured once as a CUDA graph and replayed.

## Layers

```
┌───────────────────────────────────────────────────────────────────────┐
│ CLI         tileinfer pull | ls | rm | run | chat                     │
├───────────────────────────────────────────────────────────────────────┤
│ Generation  generate, generate!, ChatSession                          │
│             speculative decoding: proposers + verify steps            │
├───────────────────────────────────────────────────────────────────────┤
│ ModelRunner step buffers, CUDA graphs, pipelined decode,              │
│             sampling on the device                                    │
├───────────────────────────────────┬───────────────────────────────────┤
│ Model       DecoderLM{T}          │ KV cache    PagedKVCache          │
│             prepare_step!         │             BlockManager          │
│             run_step!             │             Sequence              │
├───────────────────────────────────┴───────────────────────────────────┤
│ Ops         RefBackend: CUDA.jl broadcasts + cuBLAS                   │
│             TileBackend: cuTile kernels                               │
├───────────────────────────────────────────────────────────────────────┤
│ CUDA.jl (arrays, streams, graphs, cuBLAS) and cuTile.jl (Tile IR)     │
└───────────────────────────────────────────────────────────────────────┘
  Beside the stack: Hub (pull, safetensors loading), tokenizer and chat
  templates, ModelConfig.
```

### Hub, configuration and tokenizer

[`pull`](@ref) downloads a model's safetensors, configs and tokenizer into the Hugging Face
hub cache (`~/.cache/huggingface/hub`, shared with Python), and [`model_dir`](@ref) finds them
again offline. [`load_config`](@ref) reads `config.json` into a [`ModelConfig`](@ref).
Weights are memory-mapped with SafeTensors.jl and uploaded straight to the GPU without a
transpose (see [Data layout](#Data-layout)); the q, k, v projections and the gate and up
projections are concatenated at load time, so each layer runs two fewer GEMMs.

The tokenizer is TileInfer's own byte-level BPE ([`BPETokenizer`](@ref)), read from
`tokenizer.json` and checked against Hugging Face `tokenizers`. Chat templates are ported by
hand per model family ([`chat_prompt`](@ref)) instead of running a Jinja engine, and checked
against the real templates' output.

### Generation

[`generate!`](@ref) is the loop that production engines call `LLMEngine.step`: prefill the
prompt in one forward pass, then decode one token per step until a stop token or the token
budget. With a single sequence there is nothing to schedule, so the loop is the scheduler.
[`ChatSession`](@ref) keeps one sequence alive across turns and only runs the tokens a turn
adds (see [KV cache](kv_cache.md#Prefix-reuse-in-chat)).

With a proposer, decoding becomes speculative: a cheap proposer guesses several tokens, one
forward step of the model checks them all, and the sequence is rolled back past the first
wrong guess. See [Speculative decoding](speculative.md).

### Model runner

[`ModelRunner`](@ref) owns everything that makes a step fast without changing what it
computes:

- **Static step buffers.** A step of n tokens reads and writes only a preallocated
  `StepBuffers`: its inputs (token ids, positions, cache slots, block table), every
  intermediate activation, and the logits. [`prepare_step!`](@ref)
  reserves cache pages and uploads the inputs in one asynchronous copy from pinned memory.
- **CUDA graphs.** Steps of up to 8 tokens (decode, short verify steps) run eagerly the first
  time, which compiles their kernels, and are then captured. Later steps of that size upload
  their inputs and replay the graph: one launch instead of hundreds. Kernels read every
  per-step value from device buffers, never from launch arguments, which a graph would
  freeze. Longer steps (prefill) run eagerly.
- **Sampling on the device.** Greedy, temperature, top-k and top-p sampling run inside the
  captured step ([`sample_token!`](@ref)). A sampled token is a pure function of the logits,
  the parameters and a counter-based random number, so the device sampler and its host
  reference pick the same tokens.
- **Pipelined decoding.** Two sets of step buffers alternate: each step samples its token
  straight into the next step's inputs and into host-mapped memory, so the host launches step
  n + 1 before reading the token of step n. The GPU does not wait for the host between steps.

### Model

[`DecoderLM`](@ref) is a configuration, the weights, RoPE tables and a backend. One struct
covers the Llama-style family; the differences between families are configuration flags:

| | Llama 3.x | Mistral | Qwen3 | Apertus |
|---|---|---|---|---|
| q/k RMSNorm before RoPE | | | ✓ | ✓ |
| MLP | SwiGLU | SwiGLU | SwiGLU | xIELU, ungated |
| RoPE scaling | llama3 | | | llama3 |
| tied embeddings | small models | | small models | |

Apertus 1.5 is multimodal: only its decoder is loaded, and the image and audio tokenizers
are skipped at download time. Sliding-window attention (older Mistral checkpoints) is not
supported.

A step (`run_step!`) runs entirely on the device:

```
x = embed(ids)                                      # (hidden, n)
h = rmsnorm(x)
for each layer
    qkv = linear(h, Wqkv)                           # one fused projection
    q = attention_inputs!(qkv, positions, slots)    # q/k norm, RoPE, write k and v to the cache
    o = attention!(q, cache, block_table)           # paged flash-attention
    x += linear(o, Wo);  h = rmsnorm(x)             # one fused op
    a = linear_swiglu!(h, Wgate_up)                 # GEMV with a SwiGLU epilogue at n = 1
    x += linear(a, Wdown);  h = rmsnorm(x)          # the next layer's norm, or the final one
logits = linear(h[:, last], Wlm_head)               # Float32 logits
```

### Ops and backends

Each op is a generic function with a method per backend ([`Backend`](@ref)):
[`RefBackend`](@ref) composes CUDA.jl broadcasts and cuBLAS and rounds where PyTorch rounds,
so it is the ground truth in tests; [`TileBackend`](@ref) runs cuTile kernels, one file per
kernel family in `src/ops/tile/`. Every op also has an in-place form that writes into the
step buffers, and fused ops ([`attention_inputs!`](@ref), [`add_rmsnorm!`](@ref),
[`linear_swiglu!`](@ref)) have generic fallbacks that chain the primitive ops, so a backend
only needs to implement what it wants to speed up.

| Kernel family | File | What it does |
|---|---|---|
| Normalization | `norm.jl` | RMSNorm, and residual add + RMSNorm in one pass |
| Element-wise | `elementwise.jl` | RoPE, SwiGLU, xIELU, KV-cache writes, embedding lookup |
| Linear | `linear.jl` | GEMV for decode, GEMM with per-shape tiles for prefill, BF16 and INT8 weights |
| Attention inputs | `fused.jl` | q/k norm + RoPE + KV write in one launch per step |
| Attention | `attention.jl` | paged flash-attention: prefill, split-KV decode + combine ([details](attention.md)) |
| Sampling | `sampling.jl` | top-k/top-p sampling and speculative acceptance on the device |

## Mapping to vLLM and DeepInfer

| TileInfer | vLLM (v1) | DeepInfer |
|---|---|---|
| [`generate!`](@ref), [`ChatSession`](@ref) | `v1/engine/core.py` | `engine/llm_engine.py` |
| (one sequence, no scheduler) | `v1/core/sched/scheduler.py` | C++ scheduler |
| `BlockManager`, [`PagedKVCache`](@ref) | `v1/core/kv_cache_manager.py` | `worker/cache_engine.py` |
| [`ModelRunner`](@ref) and its CUDA graphs | `v1/worker/gpu_model_runner.py`, `CUDAGraphWrapper` | `worker/model_runner.py`, `cuda_graph_runner.py` |
| `StepBuffers` (positions, slots, block table) | `AttentionMetadata` | `model_executor/input_metadata/` |
| device sampling ([`sample_token!`](@ref)) | `v1/sample/sampler.py` | `model_executor/layers/sampler.py` |
| proposers and verify steps | `v1/spec_decode/` | MTP / NextN drafts, DFlash |
| `src/ops/tile/` | `csrc/`, FlashAttention, FlashInfer | TileLang ops, FlashMLA, DeepGEMM |

DeepInfer's release targets Blackwell and several GPUs and leaves its KV-cache system out,
so vLLM is the reference for the paged KV design.

## Module layout

```
src/
├── TileInfer.jl            # module, public API, includes
├── config.jl               # ModelConfig from config.json
├── cli.jl                  # the tileinfer command line
├── hub/
│   ├── store.jl            # pull / list / remove in the hub cache
│   └── loader.jl           # safetensors (memory-mapped) → CuArray, fused projections, INT8
├── tokenizer/
│   ├── bpe.jl              # byte-level BPE from tokenizer.json
│   └── chat_template.jl    # chat templates, ported by hand per family
├── model/
│   ├── weights.jl          # ModelWeights, LayerWeights
│   ├── rope.jl             # RotaryTable
│   └── decoder.jl          # DecoderLM, StepBuffers, prepare_step!, run_step!, forward!
├── kv/
│   ├── block_manager.jl    # Sequence, BlockManager: pages, block tables, rollback
│   └── paged_cache.jl      # PagedKVCache: page pools per layer
├── engine/
│   ├── runner.jl           # ModelRunner: CUDA graphs, pipelined decode
│   ├── sampler.jl          # SamplingParams, host reference sampler, TokenSampler
│   ├── generate.jl         # generate, generate!, generation_defaults, TextStream
│   ├── chat.jl             # ChatSession with prefix reuse
│   ├── speculative.jl      # verify steps, n-gram and draft-model proposers
│   └── dflash.jl           # DFlash drafter and proposer
└── ops/
    ├── interface.jl        # Backend, op signatures, generic fallbacks
    ├── reference.jl        # RefBackend
    ├── quant.jl            # Int8Weight, Int8Pages
    └── tile/               # TileBackend: norm, elementwise, linear, fused, attention, sampling
```

## Data layout

Julia arrays are column-major and PyTorch tensors row-major. Reversing every shape keeps the
bytes identical, so that is the convention throughout (it is also what cuTile.jl's
attention example does):

| Tensor | PyTorch | TileInfer |
|---|---|---|
| hidden states | `(tokens, hidden)` | `(hidden, tokens)` |
| per-head tensors | `(tokens, heads, head_dim)` | `(head_dim, heads, tokens)` |
| linear weight | `(out, in)` | `(in, out)`, applied as `y = Wᵀx` |
| paged K or V pool, per layer | `(pages, page_size, kv_heads, head_dim)` | `(head_dim, page_size, kv_heads, pages)` |

- **Weights load without a transpose.** SafeTensors.jl returns a `PermutedDimsArray` over
  the memory-mapped row-major file; its `parent()` is already the `(in, out)` column-major
  matrix, which is uploaded as is.
- **The reduction dimension comes first**, contiguous in memory, for every GEMM and GEMV.
- **A KV tile is a page.** The keys of one page and one KV head form a dense
  `(head_dim, page_size)` tile, so the attention kernels load a page in one aligned load.

## Hardware and compilation

TileInfer is developed and tuned on an RTX 4080 SUPER (Ada, sm_89, 80 SMs, 16 GB,
~736 GB/s). On Ada, Tile IR has no FP8, no TMA, no thread-block clusters and no BF16 atomic
add, so quantization means INT8 and cross-block reductions use Float32 workspaces. At batch
size 1, decode reads every weight once per token, so its speed limit is memory bandwidth:
about 90 tok/s for Qwen3-4B in BFloat16 and 560 tok/s for Qwen3-0.6B, at 1,024 positions.

cuTile compiles a kernel on its first launch for each combination of element types, constant
arguments and array shape properties (divisibility, contiguity), and caches the result on
disk across sessions. A first reply in a fresh process therefore includes a few seconds of
compilation. Arrays that grow during decoding, such as the block table, are padded so that
their shape properties never change. Measured codegen rules are collected in
[cuTile quirks](cutile_quirks.md).

## Dependencies

| Package | Used for |
|---|---|
| cuTile.jl | every hot kernel |
| CUDA.jl | arrays, streams, events, CUDA graphs, cuBLAS (reference backend), memory pool |
| HuggingFaceApi.jl | downloads and the hub cache |
| SafeTensors.jl, BFloat16s.jl | memory-mapped weights in BFloat16 |
| JSON.jl | configs and tokenizer files |
| NVTX.jl | ranges for Nsight Systems timelines |

Deliberately not used: Flux.jl and Lux.jl (no autodiff needed, and their layers would hide
the launches this project wants to see), Transformers.jl (it pins CUDA.jl 5; used as
reference code only), and BytePairEncoding.jl (it does not precompile on Julia 1.13, hence
the own tokenizer).
