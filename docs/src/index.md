# TileInfer.jl

A single-GPU LLM inference engine in Julia, written to learn how inference engines work
from the inside and to try tile-centric GPU programming with
[cuTile.jl](https://github.com/JuliaGPU/cuTile.jl). Every hot kernel (GEMV/GEMM, paged
flash-attention, normalization, RoPE, sampling) is a cuTile kernel; everything around them
(weight loading, tokenizer, paged KV cache, CUDA graphs, speculative decoding) is plain
Julia.

!!! warning "Not production software"
    TileInfer serves one conversation on one GPU, supports a handful of model families, and
    is tuned on a single card (RTX 4080 SUPER). The code favours being easy to read and
    measure over breadth of features. Use vLLM, SGLang or llama.cpp to serve models.

## What is inside

- **Models**: Llama 3, Mistral, Qwen3 and Apertus checkpoints from the Hugging Face hub,
  in BFloat16, with optional weight-only INT8 and an INT8 KV cache.
- **Kernels**: every op has a reference implementation (CUDA.jl broadcasts and cuBLAS),
  which is the ground truth in tests, and a cuTile implementation, both selected by a
  backend type ([`RefBackend`](@ref), [`TileBackend`](@ref)).
- **Engine**: a paged KV cache with prefix reuse across chat turns, decode steps captured
  as CUDA graphs, sampling on the device and a host that runs one step behind the GPU.
- **Speculative decoding**: n-gram lookup, a draft model, or a
  [DFlash](https://arxiv.org/abs/2602.06036) block drafter, with exact verification for
  greedy and sampled decoding.
- **Tooling**: a `tileinfer` command line, per-op and end-to-end benchmarks, and
  Nsight Compute profiling scripts.

On Qwen3-4B, decode runs at 77 tok/s (86 % of the memory-bandwidth roofline) and prefill
at 11k tok/s. The [Performance log](performance.md) has the numbers for each step, including
the optimizations that did not pan out.

## Where to go next

- [Getting started](usage.md): install, the command line, and the Julia API in a few lines.
- [Architecture](architecture.md): the layers of the engine, from the command line down to
  the kernels, and how they map to vLLM's. The [KV cache](kv_cache.md),
  [attention kernels](attention.md) and [speculative decoding](speculative.md) have pages of
  their own.
- [History and roadmap](roadmap.md): where the project stands, the milestones that got it
  there, and what comes next.
- [Performance log](performance.md), [Profiling](profiling.md) and
  [cuTile quirks](cutile_quirks.md): what each optimization gained, how kernels are
  measured, and the codegen rules learned along the way.
- [Public API](api.md) and [Internals](internals.md): docstrings for the code itself.
- [References](references.md): papers, engines and package sources this builds on.
