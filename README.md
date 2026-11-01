# TileInfer.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://antoinebut.github.io/TileInfer.jl/dev/)
[![Documentation](https://github.com/AntoineBut/TileInfer.jl/actions/workflows/Documentation.yml/badge.svg)](https://github.com/AntoineBut/TileInfer.jl/actions/workflows/Documentation.yml)

A single-GPU LLM inference engine in Julia, built to learn how inference engines work on
the inside and to try out tile-centric GPU programming with
[cuTile.jl](https://github.com/JuliaGPU/cuTile.jl). The hot kernels are cuTile kernels:
GEMV/GEMM, paged flash-attention, fused norms and RoPE, and device sampling. Everything
around them is plain Julia: safetensors loading, a byte-level BPE tokenizer, a paged KV
cache, CUDA graphs and speculative decoding (n-gram, draft model, DFlash).

**This is not production software.** It serves one conversation on one GPU, supports a few
Llama-style model families (Llama 3, Mistral, Qwen3, Apertus), and has only been tuned on an
RTX 4080 SUPER. The code aims to be easy to read and measure, not to be feature-complete.

```sh
git clone https://github.com/AntoineBut/TileInfer.jl && cd TileInfer.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project -m TileInfer pull Qwen/Qwen3-0.6B
julia --project -m TileInfer chat Qwen/Qwen3-0.6B
```

Requires Julia 1.13 and an NVIDIA GPU that cuTile.jl supports.

Read the **[documentation](https://antoinebut.github.io/TileInfer.jl/dev/)** for the
Julia API, the [architecture](docs/src/architecture.md), and the
[performance log](docs/src/performance.md), which records what each optimization gained,
including the ones that did not pan out.
