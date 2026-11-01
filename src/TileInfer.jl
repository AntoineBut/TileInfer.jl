"""
    TileInfer

A single-GPU LLM inference engine for Llama-style models (Llama 3, Mistral, Qwen3, Apertus),
with its hot kernels written in cuTile.jl. Built to learn how inference engines work, not
for production use.

Start with [`pull`](@ref), [`load_model`](@ref), [`load_tokenizer`](@ref) and
[`generate`](@ref) or [`ChatSession`](@ref); the `tileinfer` command line ([`cli`](@ref))
wraps the same calls.
"""
module TileInfer

using CUDA
using NVTX: NVTX
using cuTile: cuTile
import cuTile as ct

include("ops/interface.jl")
include("ops/reference.jl")
include("ops/quant.jl")
include("ops/tile/norm.jl")
include("ops/tile/elementwise.jl")
include("ops/tile/linear.jl")
include("ops/tile/attention.jl")
include("ops/tile/sampling.jl")
include("config.jl")
include("model/weights.jl")
include("model/rope.jl")
include("ops/tile/fused.jl")
include("kv/block_manager.jl")
include("kv/paged_cache.jl")
include("hub/store.jl")
include("hub/loader.jl")
include("tokenizer/bpe.jl")
include("tokenizer/chat_template.jl")
include("model/decoder.jl")
include("engine/sampler.jl")
include("engine/runner.jl")
include("engine/speculative.jl")
include("engine/dflash.jl")
include("engine/generate.jl")
include("engine/chat.jl")
include("cli.jl")

# The documented API. Nothing beyond the backend types is exported, so `using TileInfer`
# brings no common names (`encode`, `linear`, `generate`, …) into scope: qualify them
# (`TileInfer.generate`) or import them (`using TileInfer: generate`).
public
    # models and the hub
    pull, model_dir, list_models, remove_model, CachedModel,
    ModelConfig, load_config, DecoderLM, load_model, max_length, forward!,
    # tokenizer and chat templates
    BPETokenizer, load_tokenizer, encode, decode, ChatMessage, chat_prompt,
    # KV cache
    PagedKVCache, Sequence, new_sequence, truncate!, release!, OutOfPagesError,
    # generation
    SamplingParams, ModelRunner, generate, generate!, generation_defaults, TextStream,
    ChatSession, reply!, reset!, TurnStats,
    # speculative decoding
    NgramProposer, DraftProposer, DFlashProposer, DFlashDrafter, load_dflash, propose,
    SpecStats, acceptance_rate,
    # ops and backends
    embed, rmsnorm, linear, rope, silu_mul, xielu, XIELU, store_kv!, attention,
    embed!, rmsnorm!, linear!, rope!, silu_mul!, xielu!, attention!,
    attention_inputs!, linear_swiglu!, linear_xielu!, add_rmsnorm!,
    sample_token!, sample_tokens!, accept_token!, attention_workspace, supports_graphs,
    Int8Weight, quantize_int8, Int8Pages,
    # command line
    cli

function (@main)(args)
    return cli(args)
end

end
