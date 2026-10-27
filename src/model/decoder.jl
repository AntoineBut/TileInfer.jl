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
    forward!(model, cache, seq, ids; all_logits = false) -> CuMatrix{Float32}

Run the 0-based token `ids` through the model at the positions following those of `seq`,
storing their keys and values in `cache` and appending them to `seq`. Returns the logits
(vocab, n) of every token if `all_logits`, otherwise (vocab, 1) of the last token.
"""
function forward!(model::DecoderLM{T}, cache::PagedKVCache{T}, seq::Sequence,
                  ids::AbstractVector{<:Integer}; all_logits::Bool = false) where {T}
    cfg, w, be = model.config, model.weights, model.backend
    n = length(ids)
    n > 0 || throw(ArgumentError("forward! needs at least one token"))
    start = length(seq)
    start + n <= max_length(model) ||
        throw(ArgumentError("sequence of $(start + n) tokens exceeds the model's max length $(max_length(model))"))
    reserve!(cache.blocks, seq, n)
    slot_ids = CuArray(slots(cache.blocks, seq, start, n))
    # Padded to a multiple of 16 pages: cuTile kernels specialize on the divisibility of each
    # array's length, so an unpadded table would recompile attention as the sequence grows.
    block_table = CuArray([seq.block_table; zeros(Int32, -length(seq.block_table) & 15)])
    D, nq, nkv = cfg.head_dim, cfg.num_heads, cfg.num_kv_heads
    positions = start+1:start+n
    cos, sin = model.rope.cos[:, positions], model.rope.sin[:, positions]
    scale = 1 / sqrt(D)

    x = embed(be, w.embed, CuArray{Int32}(ids))                     # (hidden, n)
    for (l, layer) in enumerate(w.layers)
        h = rmsnorm(be, x, layer.attn_norm, cfg.rms_norm_eps)
        qkv = linear(be, layer.wqkv, h, layer.bqkv)                 # ((nq + 2nkv) D, n)
        q = reshape(qkv[1:nq*D, :], D, nq, n)
        k = reshape(qkv[nq*D+1:(nq+nkv)*D, :], D, nkv, n)
        v = reshape(qkv[(nq+nkv)*D+1:end, :], D, nkv, n)
        if cfg.qk_norm
            q = rmsnorm(be, q, layer.q_norm, cfg.rms_norm_eps)
            k = rmsnorm(be, k, layer.k_norm, cfg.rms_norm_eps)
        end
        q = rope(be, q, cos, sin)
        k = rope(be, k, cos, sin)
        store_kv!(be, cache.k[l], cache.v[l], k, v, slot_ids)
        o = attention(be, q, cache.k[l], cache.v[l], block_table, start, scale)   # (D, nq, n)
        x = x .+ linear(be, layer.wo, reshape(o, D * nq, n))

        h = rmsnorm(be, x, layer.mlp_norm, cfg.rms_norm_eps)
        x = x .+ linear(be, layer.w_down, silu_mul(be, linear(be, layer.w_gate_up, h)))
    end
    append!(seq.tokens, ids)

    x = all_logits ? x : x[:, n:n]
    return Float32.(linear(be, w.lm_head, rmsnorm(be, x, w.norm, cfg.rms_norm_eps)))
end
