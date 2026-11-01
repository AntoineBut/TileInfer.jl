# Paged KV cache: per layer, one key and one value pool of shape
# (head_dim, page_size, kv_heads, num_pages), shared by all sequences through a
# BlockManager. A page holds `page_size` consecutive positions of one sequence, so the
# keys of a page and KV head form a dense (head_dim, page_size) tile.

"Positions per KV-cache page unless the cache is built with another `page_size`."
const DEFAULT_PAGE_SIZE = 64

"""
    PagedKVCache{T}(config, num_pages; page_size = $DEFAULT_PAGE_SIZE, quantize = nothing, key_smoothing = nothing)

Key/value page pools for every layer and the allocator for their pages, for a model whose
activations are `T`. The pools hold `T`, or with `quantize = :int8` Int8 values with one
Float32 scale per cached vector ([`Int8Pages`](@ref)): about half the memory, and half the
bytes decode attention reads. `key_smoothing` holds each layer's smoothing of the keys of
an INT8 cache (see [`key_smoothing`](@ref)), or `nothing` for none.
"""
struct PagedKVCache{T,A}
    k::Vector{A}
    v::Vector{A}
    blocks::BlockManager
end

function PagedKVCache{T}(cfg::ModelConfig, num_pages::Integer; page_size::Integer = DEFAULT_PAGE_SIZE,
                         quantize::Union{Nothing,Symbol} = nothing, key_smoothing = nothing) where {T}
    blocks = BlockManager(num_pages, page_size)
    dims = (cfg.head_dim, page_size, cfg.num_kv_heads, num_pages)
    if quantize === nothing
        k, v = ([CUDA.zeros(T, dims...) for _ in 1:cfg.num_layers] for _ in 1:2)
        return PagedKVCache{T,eltype(k)}(k, v, blocks)
    end
    quantize === :int8 || throw(ArgumentError("unsupported KV cache quantization $quantize (only :int8)"))
    pairs = [int8_kv_pools(dims...; key_smoothing = key_smoothing === nothing ? nothing : key_smoothing[l])
             for l in 1:cfg.num_layers]
    return PagedKVCache{T,Int8Pages}(first.(pairs), last.(pairs), blocks)
end

"`:int8` for an INT8 cache, `nothing` for one in the activations' type."
quantization(::PagedKVCache{T,Int8Pages}) where {T} = :int8
quantization(::PagedKVCache) = nothing

"Bytes of cache memory per cached position (keys and values of every layer)."
bytes_per_token(cfg::ModelConfig, ::Type{T}) where {T} =
    2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * sizeof(T)
bytes_per_token(cfg::ModelConfig, quantize::Symbol) =
    quantize === :int8 ? 2 * cfg.num_layers * cfg.num_kv_heads * (cfg.head_dim + sizeof(Float32)) :
    throw(ArgumentError("unsupported KV cache quantization $quantize"))

"Positions per page of `cache`."
page_size(c::PagedKVCache) = c.blocks.page_size
"Total pages of `cache` (free or held by a sequence)."
num_pages(c::PagedKVCache) = c.blocks.num_pages

"A new, empty sequence in `cache`."
new_sequence(::PagedKVCache) = Sequence(Int32[], Int[])

truncate!(c::PagedKVCache, seq::Sequence, n::Integer) = truncate!(c.blocks, seq, n)
release!(c::PagedKVCache, seq::Sequence) = release!(c.blocks, seq)
