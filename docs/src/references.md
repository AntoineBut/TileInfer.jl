# References

Where to look things up when working on TileInfer. Entries are grouped by topic,
and each says what it is useful for. Links were checked on 2026-10-26.

## Local sources (read these first)

Installed package sources include their docs and examples, so they always match
the versions in `Manifest.toml`. Find a package's directory with:

```sh
julia --project=. -e 'using cuTile; print(pkgdir(cuTile))'
```

| What | Path under `pkgdir(...)` |
|------|--------------------------|
| cuTile.jl manual (memory, performance, compatibility, Julia vs Python differences) | `cuTile/docs/src/man/*.md` |
| cuTile.jl API reference pages | `cuTile/docs/src/lib/*.md` |
| cuTile.jl kernel examples (fmha, matmul, batchmatmul, layernorm, softmax, moe, transpose, vadd) | `cuTile/examples/*.jl`, with the Python originals alongside as `*.py` |
| cuTile.jl language surface (`load`, `store`, `gather`, `scatter`, views, `arange`, …) | `cuTile/src/language/operations.jl` |
| CUDA graph API (`capture`, `instantiate`, `launch`, `update`, `@captured`) | `CUDA.CUDACore/lib/cudadrv/graph.jl` |
| SafeTensors.jl loader (mmap, `PermutedDimsArray` over the row-major buffer) | `SafeTensors/src/SafeTensors.jl` |
| HuggingFaceApi.jl (`hf_hub_download`, `list_repo_files`, cache layout) | `HuggingFaceApi/src/{download,api}.jl` |

## Julia packages

| Package | Repo | Docs | Notes |
|---------|------|------|-------|
| cuTile.jl | https://github.com/JuliaGPU/cuTile.jl | https://juliagpu.github.io/cuTile.jl/stable/ | Kernel DSL. `man/comparison` covers porting from cuTile Python; `man/performance` covers tuning. |
| CUDA.jl | https://github.com/JuliaGPU/CUDA.jl | https://cuda.juliagpu.org/stable/ | Arrays, streams, events, graphs, cuBLAS, `CUDA.@profile`. |
| HuggingFaceApi.jl | https://github.com/chengchingwen/HuggingFaceApi.jl | README | Hub downloads; shares the cache with Python `huggingface_hub`. |
| SafeTensors.jl | https://github.com/FluxML/SafeTensors.jl | README | Weight loading. |
| BFloat16s.jl | https://github.com/JuliaMath/BFloat16s.jl | README | Host-side BF16 type. |
| NVTX.jl | https://github.com/JuliaGPU/NVTX.jl | https://juliagpu.github.io/NVTX.jl/dev/ | Ranges for Nsight Systems timelines. |
| PrecompileTools.jl | https://github.com/JuliaLang/PrecompileTools.jl | https://julialang.github.io/PrecompileTools.jl/stable/ | Reduces startup latency. |
| Transformers.jl | https://github.com/chengchingwen/Transformers.jl | https://chengchingwen.github.io/Transformers.jl/dev/ | **Reference only**, not a dependency (pins CUDA 5). See `src/huggingface/tokenizer/fast_tkr.jl` for `tokenizer.json` loading and `src/huggingface/implementation/llama/` for HF weight names. |
| BytePairEncoding.jl / TextEncodeBase.jl | https://github.com/chengchingwen/BytePairEncoding.jl, https://github.com/chengchingwen/TextEncodeBase.jl | READMEs | BPE building blocks. TextEncodeBase 0.8.3 **fails to precompile on Julia 1.13**, so it isn't a dependency for now. |
| OteraEngine.jl | https://github.com/MommaWatasu/OteraEngine.jl | README | Jinja-like templates, a candidate for chat templates. |
| Comonicon.jl | https://github.com/comonicon/Comonicon.jl | https://comonicon.org/ | CLI framework (candidate). |
| PythonCall.jl | https://github.com/JuliaPy/PythonCall.jl | https://juliapy.github.io/PythonCall.jl/stable/ | Test-only golden references from HF `transformers` / `tokenizers`. |
| Pkg apps | — | https://pkgdocs.julialang.org/v1/apps/ | Installs the `tileinfer` executable (Julia ≥ 1.12). |

## NVIDIA Tile programming

- Tile IR specification: https://docs.nvidia.com/cuda/tile-ir/latest/
  - Stability and per-architecture support: https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html
  - Optimization guide: https://docs.nvidia.com/cuda/tile-ir/latest/sections/optimization_guide.html
  - Release notes (Ampere/Ada support added in 13.2): https://docs.nvidia.com/cuda/tile-ir/main/sections/release_notes.html
- cuTile Python, the reference implementation (more examples, and most tutorials target it):
  https://github.com/NVIDIA/cutile-python · https://docs.nvidia.com/cuda/cutile-python/
- Blog posts:
  - cuTile.jl announcement: https://developer.nvidia.com/blog/cutile-jl-brings-nvidia-cuda-tile-based-programming-to-julia
  - cuTile.jl 0.2 release notes: https://juliagpu.org/post/2026-04-08-cutile_0.2/
  - Porting cuTile Python to Julia (indexing and layout pitfalls): https://developer.nvidia.com/blog/automating-gpu-kernel-translation-with-ai-agents-cutile-python-to-cutile-jl
  - CUDA 13.2 Tile updates: https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features
- Other tile-based kernel DSLs to borrow kernel ideas from: Triton (https://triton-lang.org), TileLang (https://github.com/tile-ai/tilelang; DeepInfer's custom kernels use it).

## Inference engines (architecture references)

| Engine | Link | Look at |
|--------|------|---------|
| nano-vllm | https://github.com/GeeeekExplorer/nano-vllm | About 1.2k lines of Python with vLLM's structure. **The closest match to our scope.** |
| vLLM | https://github.com/vllm-project/vllm · https://docs.vllm.ai/en/latest/ | `vllm/v1/core/` (scheduler, KV cache manager), `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/spec_decode/`, `vllm/v1/sample/` |
| DeepInfer | https://github.com/deepseek-ai/DeepInfer | `deepinfer/engine/llm_engine.py` (`step`), `deepinfer/worker/{model_runner,cuda_graph_runner,cache_engine}.py`, `deepinfer/model_executor/layers/{sampler,attention_backend}.py`. Supports only Blackwell/Ascend, and the KV cache system is not included in the release. |
| SGLang | https://github.com/sgl-project/sglang | RadixAttention prefix caching, overlap scheduler |
| FlashInfer | https://github.com/flashinfer-ai/flashinfer | Paged/ragged attention kernels, sampling kernels (sort-free top-k/top-p) |
| llama.cpp | https://github.com/ggml-org/llama.cpp | Single-user decode optimizations, speculative decoding examples |

## Papers

Attention and kernels
- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashAttention-2 (the algorithm in cuTile's `fmha.jl`): https://arxiv.org/abs/2307.08691
- FlashAttention-3 (Hopper-specific, background only): https://arxiv.org/abs/2407.08608
- Flash-Decoding (split-KV for decode): https://crfm.stanford.edu/2023/10/12/flashdecoding.html
- Online softmax: https://arxiv.org/abs/1805.02867
- PagedAttention / vLLM: https://arxiv.org/abs/2309.06180
- SGLang / RadixAttention: https://arxiv.org/abs/2312.07104

Speculative decoding
- Leviathan et al., speculative decoding: https://arxiv.org/abs/2211.17192
- Chen et al., speculative sampling (rejection sampling scheme): https://arxiv.org/abs/2302.01318
- Prompt lookup decoding (n-gram proposer): https://github.com/apoorvumang/prompt-lookup-decoding
- Lookahead decoding: https://arxiv.org/abs/2402.02057
- Medusa: https://arxiv.org/abs/2401.10774
- EAGLE / EAGLE-2 / EAGLE-3: https://arxiv.org/abs/2401.15077 · https://arxiv.org/abs/2406.16858 · https://arxiv.org/abs/2503.01840
- Multi-token prediction (MTP) in DeepSeek-V3: https://arxiv.org/abs/2412.19437
- DFlash (block-diffusion drafter): https://arxiv.org/abs/2602.06036 · https://github.com/z-lab/dflash ·
  drafter checkpoints with the reference `dflash.py`: https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16

Model building blocks
- RoPE: https://arxiv.org/abs/2104.09864
- GQA: https://arxiv.org/abs/2305.13245
- RMSNorm: https://arxiv.org/abs/1910.07467
- SwiGLU: https://arxiv.org/abs/2002.05202
- Qwen3 technical report: https://arxiv.org/abs/2505.09388

## Hugging Face formats

- safetensors format: https://github.com/huggingface/safetensors
- Hub cache layout: https://huggingface.co/docs/huggingface_hub/guides/manage-cache
- Tokenizers (the `tokenizer.json` pipeline: normalizer, pre-tokenizer, model, post-processor, decoder): https://huggingface.co/docs/tokenizers
- Chat templates: https://huggingface.co/docs/transformers/main/en/chat_templating_writing
- Reference model code to match numerically: https://github.com/huggingface/transformers/tree/main/src/transformers/models (`llama/`, `qwen2/`, `qwen3/`, `mistral/`)
- Target models: https://huggingface.co/Qwen/Qwen3-0.6B, https://huggingface.co/Qwen/Qwen3-1.7B, https://huggingface.co/Qwen/Qwen3-4B, https://huggingface.co/meta-llama/Llama-3.2-1B (gated)

## Hardware

- GPU support matrix: https://developer.nvidia.com/cuda/gpus
- RTX 4080 SUPER: Ada AD103, sm_89, 80 SMs, 16 GB GDDR6X, ~736 GB/s. Bandwidth sets the decode roofline.

## Profiling

Workflow and scripts: [Profiling](profiling.md).

- Nsight Compute docs (UI, metrics, rules, source view): https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
- `ncu` command line options: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- Kernel profiling guide (metric names, warp stall reasons, roofline): https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- Python report interface (`ncu_report`, used by `bench/ncu_summary.py`): https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html.
  The module itself is in `/usr/local/NVIDIA-Nsight-Compute/extras/python/`.
