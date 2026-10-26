# Contiguous KV cache for one sequence, used by the reference model path (M2). Paged
# caches replace it in M3.

"""
    KVCache{T}(config, max_len)

Per-layer key and value arrays (head_dim, max_len, kv_heads) and the number of positions
filled so far.
"""
mutable struct KVCache{T,A<:AbstractArray{T,3}}
    k::Vector{A}
    v::Vector{A}
    length::Int
end

function KVCache{T}(cfg::ModelConfig, max_len::Integer) where {T}
    alloc() = [CUDA.zeros(T, cfg.head_dim, max_len, cfg.num_kv_heads) for _ in 1:cfg.num_layers]
    return KVCache(alloc(), alloc(), 0)
end

capacity(c::KVCache) = size(first(c.k), 2)

"Forget all cached positions after the first `n` (rewind for regeneration)."
function truncate!(c::KVCache, n::Integer)
    0 <= n <= c.length || throw(ArgumentError("cannot truncate a cache of length $(c.length) to $n"))
    c.length = n
    return c
end

# Store k/v (head_dim, kv_heads, n) of layer `l` at positions start+1:start+n.
function store_kv!(c::KVCache, l::Integer, start::Integer, k::CuArray{T,3}, v::CuArray{T,3}) where {T}
    n = size(k, 3)
    c.k[l][:, start+1:start+n, :] .= permutedims(k, (1, 3, 2))
    c.v[l][:, start+1:start+n, :] .= permutedims(v, (1, 3, 2))
    return c
end
