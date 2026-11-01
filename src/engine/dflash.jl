# DFlash drafting (Chen et al., https://arxiv.org/abs/2602.06036): a small block-diffusion
# model drafts a whole block of tokens in one forward pass, where draft models (and EAGLE) run
# one pass per draft. Layout and conventions follow z-lab's reference `dflash.py`.
#
# The target keeps the residual stream after five of its layers for every token it runs;
# `fc` maps them, stacked, to one context state per position, c(p) = hidden_norm(fc(...)).
# The drafter is a stack of Qwen3 layers whose keys and values come from two sources:
#
#   - the context positions the target has run: k and v projections of c(p) (no input
#     norm), kept in the drafter's KV cache, so each position is computed once;
#   - the block: B = block_size positions after the context, the next token (known: the
#     target's last choice) followed by B - 1 mask tokens, as ordinary layer inputs.
#
# Only the block has queries, and its attention is not causal: every block position sees the
# whole context and the whole block. After the last layer, the target's own lm_head reads
# block positions 2:B, and their argmaxes are B - 1 drafts at once. The block's keys and
# values are dropped again (the cache is truncated to the context).
#
# Here one drafter step runs the new context positions and the block together, as n = c + B
# columns: every layer computes q, k and v for all of them (the context columns read c(p),
# the block columns their normed residual stream), writes all keys and values, and attends
# without the causal mask; the output projection and MLP run on the block columns only.
# The context queries are computed and thrown away, which costs less than a second read of
# the q/k/v weights. Steps with up to B context positions (every step after the prefill)
# are captured in CUDA graphs, one per context length.

"""
    DFlashDrafter

A DFlash drafter on the GPU: its Qwen3 layers, `fc` and norms, the target's embedding table
and lm_head (shared), its RoPE table, the target layers it reads (`aux_layers`, 1-based),
the block size and the mask token. Load with [`load_dflash`](@ref).
"""
struct DFlashDrafter{T,M<:AbstractMatrix{T},L,V<:AbstractVector{T},R<:RotaryTable{T},B<:Backend}
    config::ModelConfig
    aux_layers::Vector{Int}
    block_size::Int
    mask_id::Int
    fc::M                            # (aux layers × hidden, hidden)
    hidden_norm::V
    layers::Vector{LayerWeights{T,M,V}}
    norm::V
    embed::M                         # the target's (hidden, vocab)
    lm_head::L                       # the target's (Int8Weight if it is quantized)
    rope::R
    backend::B
end

max_length(d::DFlashDrafter) = length(d.rope)

"""
    load_dflash(drafter, target::DecoderLM; max_len = max_length(target)) -> DFlashDrafter

Load a DFlash drafter for `target` (a pulled repo id or a directory), in the format of
z-lab's checkpoints (`layers.*`, `fc`, `hidden_norm`, `dflash_config`).
"""
load_dflash(name::AbstractString, target::DecoderLM{T}; max_len::Integer = max_length(target)) where {T} =
    load_dflash(name, target.config, target.weights.embed, target.weights.lm_head, target.backend; max_len)

# With the target's configuration, embedding table and lm_head given separately (the tests
# run the drafter in Float32 without the whole target).
function load_dflash(name::AbstractString, tcfg::ModelConfig, embed::AbstractMatrix{T}, lm_head,
                     backend::Backend; max_len::Integer) where {T}
    dir = model_dir(name)
    raw = JSON.parsefile(joinpath(dir, "config.json"))
    cfg = parse_config(raw)
    cfg.vocab_size == tcfg.vocab_size || error("the drafter's vocabulary differs from the target's")
    cfg.hidden_size == tcfg.hidden_size ||
        error("drafter hidden size $(cfg.hidden_size) differs from the target's $(tcfg.hidden_size)")
    dcfg = get(raw, "dflash_config", Dict())
    haskey(dcfg, "mask_token_id") || error("$name has no dflash_config.mask_token_id")
    aux = dflash_layers(raw, tcfg.num_layers)
    B = raw["block_size"]
    B <= MAX_VERIFY_TOKENS || error("blocks of $B tokens exceed the verify steps' $MAX_VERIFY_TOKENS")

    c = Checkpoint(dir)
    H = cfg.hidden_size
    up(n, dims) = check_size(upload(T, c, n), dims, n)
    layers = [load_layer(T, c, cfg, "layers.$i.") for i in 0:cfg.num_layers-1]
    # The block runs B positions past the target's last one.
    return DFlashDrafter(cfg, aux, B, dcfg["mask_token_id"], up("fc.weight", (length(aux) * H, H)),
                         up("hidden_norm.weight", (H,)), layers, up("norm.weight", (H,)),
                         embed, lm_head, RotaryTable{T}(cfg, max_len + B), backend)
end

# The target layers whose outputs the drafter reads, counted from 1. DFlash's
# `target_layer_ids` index transformers' `hidden_states` minus one, i.e. number layer
# outputs from 0; without them, they are spread evenly from layer 1 to L - 3.
function dflash_layers(raw::AbstractDict, num_layers::Integer)
    ids = get(get(raw, "dflash_config", Dict()), "target_layer_ids", nothing)
    if ids === nothing
        n = raw["num_hidden_layers"]
        ids = n == 1 ? [num_layers ÷ 2] :
              [round(Int, 1 + i * (num_layers - 4) / (n - 1)) for i in 0:n-1]    # ties to even, as Python
    end
    return collect(Int, ids) .+ 1
end

# Buffers of one drafter step of c context positions and a block: the step's inputs and
# per-column activations for n = c + B columns (b.h holds the layers' input: the context
# states, then the normed block), and the target's stacked states of the context.
struct DFlashStep{T}
    b::StepBuffers{T}
    aux::CuMatrix{T}                 # (aux layers × hidden, c)
end

DFlashStep(d::DFlashDrafter{T}, cache::PagedKVCache{T}, c::Integer) where {T} =
    DFlashStep{T}(StepBuffers{T}(d.config, d.backend, cache, c + d.block_size; vocab = 1),
                  CuMatrix{T}(undef, size(d.fc, 1), c))

"""
    DFlashProposer(drafter; k = drafter.block_size - 1, sample = false, max_tokens = max_length(drafter),
                   graphs = supports_graphs(drafter.backend))

Proposes the first `k` of the `block_size - 1` tokens that the DFlash `drafter` (see
[`load_dflash`](@ref)) drafts in one pass, on a KV cache of its own for up to `max_tokens`
positions: the argmax at each block position, or, when `sample` and the target samples, a
sample with the target's parameters (each position independently: the block's positions do
not see each other's drafts, so q of draft j is the drafter's distribution at position j). Reads
the hidden states of the target's latest step, which generation enables on the target's runner.
"""
mutable struct DFlashProposer{D<:DFlashDrafter,T}
    const k::Int
    const sample::Bool
    const dists::DraftDists                      # of the drafts, when sampled
    const drafter::D
    const cache::PagedKVCache{T}
    const seq::Sequence                          # the context positions the cache holds
    const graphs::Bool
    const steps::Dict{Int,Any}                   # (DFlashStep, graph) by context length
    const x::CuMatrix{T}                         # (hidden, B): the block's residual stream
    const h::CuMatrix{T}                         # (hidden, B): its normed state (MLP input, output)
    const logits::CuMatrix{Float32}              # (vocab, B - 1)
    const sink::CuVector{Int32}                  # device copies of the drafts
    const mirror::CuVector{Int32}                # host-mapped: the drafts
    const mirror_host::Vector{Int32}
    const sampler::TokenSampler
    const scratch::CuMatrix{Int32}               # for sample_tokens!
    target::Any                                  # the target's ModelRunner
end

function DFlashProposer(d::DFlashDrafter{T}; k::Integer = d.block_size - 1, sample::Bool = false,
                        max_tokens::Integer = max_length(d),
                        graphs::Bool = supports_graphs(d.backend)) where {T}
    B, H, V = d.block_size, d.config.hidden_size, d.config.vocab_size
    1 <= k < B || throw(ArgumentError("k must be in 1:$(B - 1)"))
    cache = PagedKVCache{T}(d.config, cld(min(max_tokens + B, max_length(d)), DEFAULT_PAGE_SIZE))
    mirror = CuArray{Int32,1,CUDA.HostMemory}(undef, B - 1)
    return DFlashProposer{typeof(d),T}(k, sample, DraftDists(B - 1), d, cache, new_sequence(cache), graphs, Dict{Int,Any}(),
                                       CuMatrix{T}(undef, H, B), CuMatrix{T}(undef, H, B),
                                       CuMatrix{Float32}(undef, V, B - 1), CuVector{Int32}(undef, B - 1), mirror,
                                       unsafe_wrap(Array, mirror), TokenSampler(V), sample_cols_scratch(B - 1), nothing)
end

function attach!(p::DFlashProposer, r::ModelRunner)
    size(p.drafter.fc, 1) == length(p.drafter.aux_layers) * r.model.config.hidden_size ||
        throw(ArgumentError("the DFlash drafter does not fit this target model"))
    enable_aux!(r, p.drafter.aux_layers)
    reset_draft_sampler!(p.sampler, p.sample, r)
    p.target = r
    return p
end

draft_dists(p::DFlashProposer, r::ModelRunner) = samples_for(p.sample, r) ? p.dists : nothing

# The drafter on a prepared step whose aux holds the target's context states; leaves the
# logits of block positions 2:B in p.logits.
function dflash_step!(p::DFlashProposer{<:DFlashDrafter{T}}, s::DFlashStep{T}) where {T}
    d, cache, b = p.drafter, p.cache, s.b
    cfg, be = d.config, d.backend
    n, B, eps = num_tokens(b), d.block_size, cfg.rms_norm_eps
    c = n - B
    ctx, blk = view(b.h, :, 1:c), view(b.h, :, c+1:n)              # column views are CuArrays
    proj, act, gate_up = view(b.proj, :, 1:B), view(b.act, :, 1:B), view(b.gate_up, :, 1:B)
    linear!(be, view(b.x, :, 1:c), d.fc, s.aux)
    rmsnorm!(be, ctx, view(b.x, :, 1:c), d.hidden_norm, eps)       # the same for every layer
    embed!(be, p.x, d.embed, view(b.ids, c+1:n))
    rmsnorm!(be, blk, p.x, first(d.layers).attn_norm, eps)
    for (l, layer) in enumerate(d.layers)
        linear!(be, b.qkv, layer.wqkv, b.h)
        attention_inputs!(be, b.q, cache.k[l], cache.v[l], b.qkv, layer.q_norm, layer.k_norm, d.rope,
                          b.positions, b.slots, eps)
        attention!(be, b.attn, b.attn_workspace, b.q, cache.k[l], cache.v[l], b.block_table, b.positions,
                   1 / sqrt(cfg.head_dim); causal = false)
        linear!(be, proj, layer.wo, view(reshape(b.attn, :, n), :, c+1:n))
        add_rmsnorm!(be, p.x, p.h, proj, layer.mlp_norm, eps)
        linear_swiglu!(be, act, gate_up, layer.w_gate_up, p.h)
        linear!(be, proj, layer.w_down, act)
        # The next layer's input norm into the block columns of b.h, or the final norm.
        last = l == length(d.layers)
        add_rmsnorm!(be, p.x, last ? p.h : blk, proj, last ? d.norm : d.layers[l+1].attn_norm, eps)
    end
    linear!(be, p.logits, d.lm_head, view(p.h, :, 2:B))
    # All B - 1 positions in one call; q recorded when sampled (see DraftProposer).
    sample_tokens!(be, p.sink, p.mirror, p.logits, p.sampler, p.scratch,
                   p.sample ? (p.dists.ids, p.dists.probs) : nothing)
end

function propose(p::DFlashProposer, context::AbstractVector{<:Integer}, k::Integer)
    k = min(k, p.k)
    k >= 1 || return Int[]
    r = p.target::ModelRunner
    draft!(p, context, r.last_start, (r.last_step::StepBuffers).aux)
    synchronize()
    return Int.(p.mirror_host[1:k])
end

# The drafts stay on the device, for the verify step to copy into its inputs: the host
# enqueues the drafter's step and the target's verify step back to back and waits once.
function propose_on_device!(p::DFlashProposer, context::AbstractVector{<:Integer}, k::Integer)
    1 <= k <= p.k || return nothing
    r = p.target::ModelRunner
    draft!(p, context, r.last_start, (r.last_step::StepBuffers).aux)
    return view(p.sink, 1:k)
end

drafted(p::DFlashProposer, k::Integer) = Int.(p.mirror_host[1:k])

# Enqueue the drafting of the block after `context` into p.sink and p.mirror (and its logits
# into p.logits), where the target's latest step ran positions start:length(context)-2 and
# left their states in the first columns of `aux`. Does not wait for the drafts.
function draft!(p::DFlashProposer, context::AbstractVector{<:Integer}, start::Integer, aux::CuMatrix)
    d, seq = p.drafter, p.seq
    L, B = length(context), d.block_size
    # The target ran positions start:L-2 last (L - 1 is the newest token, the block's first).
    # Keep the context positions before that which still hold the context's tokens.
    keep = min(common_prefix(seq.tokens, view(context, 1:L-1)), start)
    truncate!(p.cache, seq, keep)
    if keep < start
        # The target ran positions the drafter never saw (the end of an earlier generation,
        # or a prefix reused without drafting): their hidden states are gone, so their keys
        # and values are left stale. Drafts get worse; the tokens stay correct.
        reserve!(p.cache.blocks, seq, start - keep)
        append!(seq.tokens, view(context, keep+1:start))
    end
    c = L - 1 - start
    # The context's tokens (only the block's are read), then the block.
    ids = [view(context, start+1:L); fill(d.mask_id, B - 1)]
    NVTX.@range "draft" begin
        s, graph = c > B ? (DFlashStep(d, p.cache, c), nothing) : get!(() -> (DFlashStep(d, p.cache, c), nothing), p.steps, c)
        prepare_step!(s.b, d, p.cache, seq, ids)
        copyto!(s.aux, 1, aux, 1, length(s.aux))         # the target's states of positions start:L-2
        step() = dflash_step!(p, s)
        if c > B
            step()
        else
            p.steps[c] = (s, run_captured!(step, p, graph))
        end
    end
    append!(seq.tokens, view(context, start+1:L-1))
    truncate!(p.cache, seq, L - 1)                       # drop the block's keys and values
    return p
end
