# Llama-style decoder (Llama 3, Mistral, Qwen2, Qwen3) as plain structs plus an explicit
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
    load_model(model; T = BFloat16, backend = RefBackend(), max_len = 4096) -> DecoderLM

Load a pulled model (repo id) or a model directory. `max_len` bounds the positions the
RoPE tables cover, i.e. the longest sequence the model can process.
"""
function load_model(model::AbstractString; T::Type = BFloat16, backend::Backend = RefBackend(),
                    max_len::Integer = 4096)
    dir = model_dir(model)
    cfg = load_config(dir)
    max_len = min(max_len, cfg.max_position_embeddings)
    return DecoderLM(cfg, load_weights(T, cfg, dir), RotaryTable{T}(cfg, max_len), backend)
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
    StepBuffers(model, cache, n; all_logits = false)

Device memory for one forward step of `n` tokens: the step's inputs (token ids, positions,
cache slots and a block table long enough for every page of the pool), every intermediate
activation, and the logits of the last token, or of all tokens with `all_logits`. A step
that only writes into these buffers allocates nothing, so on a backend that
[`supports_graphs`](@ref) it can be captured in a CUDA graph and replayed (see
[`ModelRunner`](@ref)).
"""
struct StepBuffers{T}
    ids::CuVector{Int32}            # (n,) 0-based token ids
    positions::CuVector{Int32}      # (n,) 0-based positions
    slots::CuVector{Int32}          # (n,) cache slots, see `slots`
    block_table::CuVector{Int32}    # the sequence's pages, zero-padded to a fixed length
    host_table::Vector{Int32}       # staging for block_table
    x::CuMatrix{T}                  # residual stream (hidden, n)
    h::CuMatrix{T}                  # normalized input of a sublayer (hidden, n)
    qkv::CuMatrix{T}                # fused projection ((heads + 2 kv_heads) head_dim, n)
    q::CuArray{T,3}                 # (head_dim, heads, n), copied out of qkv
    k::CuArray{T,3}                 # (head_dim, kv_heads, n)
    v::CuArray{T,3}
    q_normed::CuArray{T,3}          # after the Qwen3 q/k norm
    k_normed::CuArray{T,3}
    q_rot::CuArray{T,3}             # after rope
    k_rot::CuArray{T,3}
    attn::CuArray{T,3}              # attention output (head_dim, heads, n)
    attn_workspace::Union{Nothing,CuArray{Float32,3}}
    proj::CuMatrix{T}               # output of wo or w_down (hidden, n)
    gate_up::CuMatrix{T}            # (2 intermediate, n)
    act::CuMatrix{T}                # (intermediate, n)
    final::CuMatrix{T}              # final-normalized hidden states of the logit columns
    logits_T::CuMatrix{T}           # (vocab, 1 or n)
    logits::CuMatrix{Float32}
end

function StepBuffers(m::DecoderLM{T}, cache::PagedKVCache{T}, n::Integer; all_logits::Bool = false) where {T}
    cfg = m.config
    H, D, nq, nkv, I = cfg.hidden_size, cfg.head_dim, cfg.num_heads, cfg.num_kv_heads, cfg.intermediate_size
    m_out = all_logits ? n : 1
    dev(S, dims...) = CuArray{S}(undef, dims...)
    # A fixed table length, a multiple of 16: cuTile kernels specialize on the divisibility
    # of each array's length, so a table that grew with the sequence would recompile them.
    table_len = cld(num_pages(cache), 16) * 16
    q = dev(T, D, nq, n)
    return StepBuffers{T}(dev(Int32, n), dev(Int32, n), dev(Int32, n), CUDA.zeros(Int32, table_len),
                          zeros(Int32, table_len), dev(T, H, n), dev(T, H, n), dev(T, (nq + 2nkv) * D, n),
                          q, dev(T, D, nkv, n), dev(T, D, nkv, n), dev(T, D, nq, n), dev(T, D, nkv, n),
                          dev(T, D, nq, n), dev(T, D, nkv, n), dev(T, D, nq, n),
                          attention_workspace(m.backend, q, first(cache.k)), dev(T, H, n), dev(T, 2I, n),
                          dev(T, I, n), dev(T, H, m_out), dev(T, cfg.vocab_size, m_out),
                          dev(Float32, cfg.vocab_size, m_out))
end

num_tokens(b::StepBuffers) = length(b.ids)

"""
    prepare_step!(buffers, model, cache, seq, ids)

Reserve cache pages for the 0-based token `ids` after the positions of `seq`, and upload
the step's inputs into `buffers`. Host side only; `seq.tokens` is not changed.
"""
function prepare_step!(b::StepBuffers, model::DecoderLM, cache::PagedKVCache, seq::Sequence,
                       ids::AbstractVector{<:Integer})
    n = length(ids)
    n == num_tokens(b) || throw(ArgumentError("buffers are for $(num_tokens(b)) tokens, got $n"))
    start = length(seq)
    start + n <= max_length(model) ||
        throw(ArgumentError("sequence of $(start + n) tokens exceeds the model's max length $(max_length(model))"))
    reserve!(cache.blocks, seq, n)
    copyto!(b.ids, Int32.(ids))
    copyto!(b.positions, Int32.(start:start+n-1))
    copyto!(b.slots, slots(cache.blocks, seq, start, n))
    fill!(b.host_table, 0)
    copyto!(b.host_table, seq.block_table)
    copyto!(b.block_table, b.host_table)
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
    for (l, layer) in enumerate(w.layers)
        rmsnorm!(be, b.h, b.x, layer.attn_norm, eps)
        linear!(be, b.qkv, layer.wqkv, b.h)
        layer.bqkv === nothing || (b.qkv .+= layer.bqkv)
        # Copies of the q, k and v rows of the fused projection (to be fused away, M8+).
        flat(b.q) .= view(b.qkv, 1:nq*D, :)
        flat(b.k) .= view(b.qkv, nq*D+1:(nq+nkv)*D, :)
        flat(b.v) .= view(b.qkv, (nq+nkv)*D+1:(nq+2nkv)*D, :)
        q, k = b.q, b.k
        if cfg.qk_norm
            q = rmsnorm!(be, b.q_normed, q, layer.q_norm, eps)
            k = rmsnorm!(be, b.k_normed, k, layer.k_norm, eps)
        end
        rope!(be, b.q_rot, q, model.rope.cos, model.rope.sin, b.positions)
        rope!(be, b.k_rot, k, model.rope.cos, model.rope.sin, b.positions)
        store_kv!(be, cache.k[l], cache.v[l], b.k_rot, b.v, b.slots)
        attention!(be, b.attn, b.attn_workspace, b.q_rot, cache.k[l], cache.v[l], b.block_table,
                   b.positions, scale)
        linear!(be, b.proj, layer.wo, flat(b.attn))
        b.x .+= b.proj

        rmsnorm!(be, b.h, b.x, layer.mlp_norm, eps)
        linear!(be, b.gate_up, layer.w_gate_up, b.h)
        silu_mul!(be, b.act, b.gate_up)
        linear!(be, b.proj, layer.w_down, b.act)
        b.x .+= b.proj
    end

    last = size(b.logits, 2) == n ? b.x : view(b.x, :, n:n)        # a column view is a CuArray
    rmsnorm!(be, b.final, last, w.norm, eps)
    linear!(be, b.logits_T, w.lm_head, b.final)
    b.logits .= b.logits_T
    return b.logits
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
