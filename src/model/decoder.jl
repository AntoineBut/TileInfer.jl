# Llama-style decoder (Llama 3, Mistral, Qwen2, Qwen3, Apertus) as plain structs plus an explicit
# forward pass over backend ops. Per-family differences are ModelConfig flags.

"""
    DecoderLM{T}

A model on the GPU: configuration, weights, RoPE tables and the op backend. Load with
[`load_model`](@ref); run with [`forward!`](@ref).
"""
struct DecoderLM{T,W<:ModelWeights{T},R<:RotaryTable{T},B<:Backend}
    config::ModelConfig
    weights::W
    rope::R
    backend::B
end

"""
    load_model(model; T = BFloat16, backend = RefBackend(), max_len = 4096, quantize = nothing) -> DecoderLM

Load a pulled model (repo id) or a model directory. `max_len` bounds the positions the
RoPE tables cover, i.e. the longest sequence the model can process. `quantize = :int8`
stores the linear weights as Int8 with per-channel scales (see [`Int8Weight`](@ref)).
"""
function load_model(model::AbstractString; T::Type = BFloat16, backend::Backend = RefBackend(),
                    max_len::Integer = 4096, quantize::Union{Nothing,Symbol} = nothing)
    dir = model_dir(model)
    cfg = load_config(dir)
    max_len = min(max_len, cfg.max_position_embeddings)
    return DecoderLM(cfg, load_weights(T, cfg, dir; quantize), RotaryTable{T}(cfg, max_len), backend)
end

Base.eltype(::DecoderLM{T}) where {T} = T
max_length(m::DecoderLM) = length(m.rope)

"""
    PagedKVCache(model; max_tokens = max_length(model), page_size = $DEFAULT_PAGE_SIZE)

An empty paged KV cache for `model` with room for `max_tokens` positions in total.
"""
PagedKVCache(m::DecoderLM{T}; max_tokens::Integer = max_length(m),
             page_size::Integer = DEFAULT_PAGE_SIZE) where {T} =
    PagedKVCache{T}(m.config, cld(max_tokens, page_size); page_size)

"""
    StepBuffers(model, cache, n; all_logits = false, aux_layers = Int[])

Device memory for one forward step of `n` tokens: the step's inputs (token ids, positions,
cache slots and a block table long enough for every page of the pool), every intermediate
activation, and the logits of the last token, or of all tokens with `all_logits`. With
`aux_layers`, the step also keeps the residual stream after each of those layers (1-based),
stacked in `aux`, for an EAGLE drafter (see [`Eagle3Proposer`](@ref)). A step
that only writes into these buffers allocates nothing, so on a backend that
[`supports_graphs`](@ref) it can be captured in a CUDA graph and replayed (see
[`ModelRunner`](@ref)).
"""
struct StepBuffers{T}
    # The step's inputs: views of one device vector `meta`, uploaded in one copy from the
    # pinned `staging` (written through its host view `staging_host`). Each segment starts
    # at a multiple of `segment` (32 entries, 128 bytes), so every view stays aligned.
    meta::CuVector{Int32}
    staging::CuVector{Int32}         # pinned host memory (CUDA.HostMemory)
    staging_host::Vector{Int32}
    uploaded::CuEvent                # recorded after the last copy out of `staging`
    segment::Int
    ids::CuVector{Int32}            # (n,) 0-based token ids
    positions::CuVector{Int32}      # (n,) 0-based positions
    slots::CuVector{Int32}          # (n,) cache slots, see `slots`
    block_table::CuVector{Int32}    # the sequence's pages, zero-padded to a fixed length
    x::CuMatrix{T}                  # residual stream (hidden, n)
    h::CuMatrix{T}                  # normalized input of a sublayer (hidden, n)
    qkv::CuMatrix{T}                # fused projection ((heads + 2 kv_heads) head_dim, n)
    q::CuArray{T,3}                 # attention queries (head_dim, heads, n), after q norm and rope
    attn::CuArray{T,3}              # attention output (head_dim, heads, n)
    attn_workspace::Union{Nothing,CuArray{Float32,3}}
    proj::CuMatrix{T}               # output of wo or w_down (hidden, n)
    gate_up::CuMatrix{T}            # (2 intermediate, n); (0, n) for an ungated MLP
    act::CuMatrix{T}                # (intermediate, n)
    logits::CuMatrix{Float32}
    aux_layers::Vector{Int}
    aux::Union{Nothing,CuMatrix{T}}  # (length(aux_layers) hidden, n): the layers' outputs, stacked
end

StepBuffers(m::DecoderLM{T}, cache::PagedKVCache{T}, n::Integer; kwargs...) where {T} =
    StepBuffers{T}(m.config, m.backend, cache, n; kwargs...)

# From a configuration, for drafters that are not a DecoderLM; `vocab` sizes the logits.
function StepBuffers{T}(cfg::ModelConfig, backend::Backend, cache::PagedKVCache{T}, n::Integer;
                        all_logits::Bool = false, aux_layers = Int[], vocab::Integer = cfg.vocab_size) where {T}
    H, D, nq, nkv, I = cfg.hidden_size, cfg.head_dim, cfg.num_heads, cfg.num_kv_heads, cfg.intermediate_size
    m_out = all_logits ? n : 1
    dev(S, dims...) = CuArray{S}(undef, dims...)
    # A fixed table length, a multiple of 16: cuTile kernels specialize on the divisibility
    # of each array's length, so a table that grew with the sequence would recompile them.
    table_len = cld(num_pages(cache), 16) * 16
    a = cld(n, 32) * 32
    meta = CUDA.zeros(Int32, 3a + table_len)
    staging = CuArray{Int32,1,CUDA.HostMemory}(undef, length(meta))
    staging_host = unsafe_wrap(Array, staging)
    fill!(staging_host, 0)
    q = dev(T, D, nq, n)
    return StepBuffers{T}(meta, staging, staging_host, CuEvent(), a, view(meta, 1:n), view(meta, a+1:a+n),
                          view(meta, 2a+1:2a+n), view(meta, 3a+1:3a+table_len),
                          dev(T, H, n), dev(T, H, n), dev(T, (nq + 2nkv) * D, n),
                          q, dev(T, D, nq, n),
                          attention_workspace(backend, q, first(cache.k)), dev(T, H, n),
                          dev(T, cfg.hidden_act == :silu ? 2I : 0, n),
                          dev(T, I, n), dev(Float32, vocab, m_out), collect(Int, aux_layers),
                          isempty(aux_layers) ? nothing : dev(T, length(aux_layers) * H, n))
end

num_tokens(b::StepBuffers) = length(b.ids)

"""
    prepare_step!(buffers, model, cache, seq, ids)
    prepare_step!(buffers, model, cache, seq, nothing)

Reserve cache pages for the 0-based token `ids` after the positions of `seq`, and upload
the step's inputs into `buffers` with one asynchronous copy, ordered before the step on the
stream. With `nothing`, the ids are already on the device (written by a previous step's
sampling). Host side only; `seq.tokens` is not changed.
"""
function prepare_step!(b::StepBuffers, model, cache::PagedKVCache, seq::Sequence,
                       ids::Union{Nothing,AbstractVector{<:Integer}})
    n = num_tokens(b)
    ids === nothing || length(ids) == n || throw(ArgumentError("buffers are for $n tokens, got $(length(ids))"))
    start = length(seq)
    start + n <= max_length(model) ||
        throw(ArgumentError("sequence of $(start + n) tokens exceeds the model's max length $(max_length(model))"))
    reserve!(cache.blocks, seq, n)
    synchronize(b.uploaded)          # the previous copy out of `staging` has run
    h, a = b.staging_host, b.segment
    ids === nothing || (h[1:n] .= ids)
    h[a+1:a+n] .= start:start+n-1
    h[2a+1:2a+n] .= slots(cache.blocks, seq, start, n)
    table = view(h, 3a+1:length(h))
    fill!(table, 0)
    copyto!(table, seq.block_table)
    first = ids === nothing ? a + 1 : 1
    copyto!(b.meta, first, b.staging, first, length(h) - first + 1)
    record(b.uploaded)
    return b
end

"""
    run_step!(model, cache, buffers) -> logits

The forward pass of a prepared step, entirely on the device and only writing into
`buffers` and the KV cache. Returns `buffers.logits`.
"""
function run_step!(model::DecoderLM{T}, cache::PagedKVCache{T}, b::StepBuffers{T}) where {T}
    cfg, w, be = model.config, model.weights, model.backend
    n, eps = num_tokens(b), cfg.rms_norm_eps
    D, nq, nkv = cfg.head_dim, cfg.num_heads, cfg.num_kv_heads
    scale = 1 / sqrt(D)
    flat(a) = reshape(a, :, n)

    embed!(be, b.x, w.embed, b.ids)                                 # (hidden, n)
    rmsnorm!(be, b.h, b.x, first(w.layers).attn_norm, eps)
    for (l, layer) in enumerate(w.layers)
        linear!(be, b.qkv, layer.wqkv, b.h)
        layer.bqkv === nothing || (b.qkv .+= layer.bqkv)
        attention_inputs!(be, b.q, cache.k[l], cache.v[l], b.qkv, layer.q_norm, layer.k_norm, model.rope,
                          b.positions, b.slots, eps)
        attention!(be, b.attn, b.attn_workspace, b.q, cache.k[l], cache.v[l], b.block_table,
                   b.positions, scale)
        linear!(be, b.proj, layer.wo, flat(b.attn))
        add_rmsnorm!(be, b.x, b.h, b.proj, layer.mlp_norm, eps)    # x += proj; h = norm(x)

        if layer.xielu === nothing
            linear_swiglu!(be, b.act, b.gate_up, layer.w_gate_up, b.h)
        else
            linear_xielu!(be, b.act, layer.w_gate_up, b.h, layer.xielu)
        end
        linear!(be, b.proj, layer.w_down, b.act)
        # The next layer's input norm, or the final norm after the last layer.
        add_rmsnorm!(be, b.x, b.h, b.proj, l < length(w.layers) ? w.layers[l+1].attn_norm : w.norm, eps)
        save_aux!(b, l)
    end

    last = size(b.logits, 2) == n ? b.h : view(b.h, :, n:n)        # a column view is a CuArray
    return linear!(be, b.logits, w.lm_head, last)                  # Float32 logits
end

# Keep the residual stream after layer l if it is one of the step's aux layers.
function save_aux!(b::StepBuffers, l)
    b.aux === nothing && return
    i = findfirst(==(l), b.aux_layers)
    i === nothing && return
    H = size(b.x, 1)
    view(b.aux, (i-1)*H+1:i*H, :) .= b.x
    return
end

"""
    forward!(model, cache, seq, ids; all_logits = false) -> CuMatrix{Float32}

Run the 0-based token `ids` through the model at the positions following those of `seq`,
storing their keys and values in `cache` and appending them to `seq`. Returns the logits
(vocab, n) of every token if `all_logits`, otherwise (vocab, 1) of the last token. Runs
eagerly with buffers allocated for this step; [`ModelRunner`](@ref) reuses them.
"""
function forward!(model::DecoderLM{T}, cache::PagedKVCache{T}, seq::Sequence,
                  ids::AbstractVector{<:Integer}; all_logits::Bool = false) where {T}
    n = length(ids)
    n > 0 || throw(ArgumentError("forward! needs at least one token"))
    b = StepBuffers(model, cache, n; all_logits)
    prepare_step!(b, model, cache, seq, ids)
    run_step!(model, cache, b)
    append!(seq.tokens, ids)
    return b.logits
end
