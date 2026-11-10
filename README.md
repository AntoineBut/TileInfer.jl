# TileInfer.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://antoinebut.github.io/TileInfer.jl/dev/)
[![Documentation](https://github.com/AntoineBut/TileInfer.jl/actions/workflows/Documentation.yml/badge.svg)](https://github.com/AntoineBut/TileInfer.jl/actions/workflows/Documentation.yml)

A single-GPU LLM inference engine in Julia, built to learn how inference engines work on
the inside and to try out tile-centric GPU programming with
[cuTile.jl](https://github.com/JuliaGPU/cuTile.jl). The hot kernels are cuTile kernels:
GEMV/GEMM, paged flash-attention, fused norms and RoPE, and device sampling. Everything
around them is plain Julia: safetensors loading, a byte-level BPE tokenizer, a paged KV
cache, CUDA graphs and speculative decoding (n-gram, draft model, DFlash).

The code aims to be easy to read and to measure: each kernel has a reference
implementation it is tested against, a benchmark, and an entry in the performance page
saying what it gained.

Learn more by reading my [blog post](https://antoinebut.github.io/blog/tileinfer/)!

## Quick start

```sh
git clone https://github.com/AntoineBut/TileInfer.jl && cd TileInfer.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project -m TileInfer pull Qwen/Qwen3-0.6B
julia --project -m TileInfer chat Qwen/Qwen3-0.6B
```

Requires Julia 1.13 and an NVIDIA GPU that cuTile.jl supports. Qwen3-0.6B fits on any
card, Qwen3-4B needs about 9 GB, and 8B models need `--int8` on a 16 GB card.

Other commands: `run` for a single prompt, `ls` and `rm` to manage pulled models,
`help` for every option (sampling, context size, INT8 weights and KV cache, speculative
decoding). Models are stored in the Hugging Face hub cache, shared with Python's
`huggingface_hub`. To install a `tileinfer` executable instead of going through
`julia --project -m TileInfer`, run `julia --project -e 'using Pkg; Pkg.Apps.develop(path = ".")'`.

From Julia:

```julia
using TileInfer
using TileInfer: pull, load_model, load_tokenizer, ChatSession, reply!

dir = pull("Qwen/Qwen3-0.6B")
model = load_model(dir; backend = TileBackend())
session = ChatSession(model, load_tokenizer(dir); enable_thinking = false)
reply, stats = reply!(session, "Name three primary colors.")
```

## What is inside

- **Models**: Llama 3, Mistral, Qwen3 and Apertus checkpoints from the Hugging Face hub,
  in BFloat16, with optional weight-only INT8 and an INT8 KV cache.
- **Two backends**: every op has a reference implementation (CUDA.jl broadcasts and
  cuBLAS), which is the ground truth in tests, and a cuTile implementation. Float32 runs
  match the Hugging Face `transformers` logits to ~1e-4.
- **Engine**: a paged KV cache with prefix reuse across chat turns, allocation-free decode
  steps replayed from CUDA graphs, sampling on the device, and a host that runs one step
  behind the GPU.
- **Speculative decoding**: n-gram lookup, a draft model, or a
  [DFlash](https://arxiv.org/abs/2602.06036) block drafter, with exact verification for
  both greedy and sampled decoding.
- **Tooling**: per-op and end-to-end benchmarks, a per-kernel breakdown of real steps,
  Nsight Compute profiling scripts, and codegen probes that compare equivalent cuTile
  kernels down to their SASS.

## Performance

On an RTX 4080 SUPER (the card it is tuned on), in BFloat16, with decode measured after
1,023 cached positions:

| | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| Decode | 409 tok/s (73 % of the bandwidth roofline) | 79 tok/s (88 %) |
| Decode with INT8 weights | 542 tok/s | 139 tok/s |
| Prefill, 512-token prompt | 58k tok/s | 11.4k tok/s (91 TFLOP/s) |
| Speculative decoding (DFlash) | — | 1.8× to 3.2× on open prompts, 8.5× on a code edit |

Apertus 1.5 8B runs on the same 16 GB card with INT8 weights, at 80 tok/s.

The engine was built in milestones, each a working engine that the next one sped up:
reference ops on cuBLAS (26 tok/s on Qwen3-0.6B), then cuTile kernels, paged
flash-attention, CUDA graphs, and fused kernels with device sampling (409 tok/s). The
[performance page](docs/src/performance.md) records what each feature gains, including the
optimizations that did not pan out.

## Documentation

The **[documentation](https://antoinebut.github.io/TileInfer.jl/dev/)** covers:

- [Getting started](docs/src/usage.md): the command line and the Julia API.
- [Architecture](docs/src/architecture.md): the layers of the engine, from the command line
  down to the kernels, and how they map to vLLM's; with pages on the
  [KV cache](docs/src/kv_cache.md), the [attention kernels](docs/src/attention.md) and
  [speculative decoding](docs/src/speculative.md).
- [History and roadmap](docs/src/roadmap.md): milestones so far and what comes next.
- [Profiling](docs/src/profiling.md) and [cuTile design choices](docs/src/cutile_design.md):
  how kernels are measured, and the codegen rules learned along the way.
- [References](docs/src/references.md): papers, engines and package sources this builds on.

## Development

```sh
julia --project=test test/runtests.jl            # full test suite, ~4 min
julia --project=test test/runtests.jl kernels    # only matching test files
julia --project=bench bench/run.jl --model Qwen/Qwen3-0.6B   # op and model benchmarks
julia --project=docs docs/make.jl                # documentation site, no GPU needed
```

Tests that need a real checkpoint are skipped until `Qwen/Qwen3-0.6B` is pulled. Golden
outputs from the Hugging Face reference implementations live in `test/fixtures/`.

## License

MIT, see [LICENSE](LICENSE).
