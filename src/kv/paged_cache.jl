# Paged KV cache: per layer, one key and one value pool of shape
# (head_dim, page_size, kv_heads, num_pages), shared by all sequences through a
# BlockManager. A page holds `page_size` consecutive positions of one sequence, so the
# keys of a page and KV head form a dense (head_dim, page_size) tile.

const DEFAULT_PAGE_SIZE = 64

"""
    PagedKVCache{T}(config, num_pages; page_size = $DEFAULT_PAGE_SIZE)

Key/value page pools for every layer and the allocator for their pages.
"""
struct PagedKVCache{T,A<:AbstractArray{T,4}}
    k::Vector{A}
    v::Vector{A}
    blocks::BlockManager
end

function PagedKVCache{T}(cfg::ModelConfig, num_pages::Integer;
                         page_size::Integer = DEFAULT_PAGE_SIZE) where {T}
    alloc() = [CUDA.zeros(T, cfg.head_dim, page_size, cfg.num_kv_heads, num_pages)
               for _ in 1:cfg.num_layers]
    return PagedKVCache(alloc(), alloc(), BlockManager(num_pages, page_size))
end

"Bytes of cache memory per cached position (keys and values of every layer)."
bytes_per_token(cfg::ModelConfig, ::Type{T}) where {T} =
    2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * sizeof(T)

page_size(c::PagedKVCache) = c.blocks.page_size
num_pages(c::PagedKVCache) = c.blocks.num_pages

"A new, empty sequence in `cache`."
new_sequence(::PagedKVCache) = Sequence(Int32[], Int[])

truncate!(c::PagedKVCache, seq::Sequence, n::Integer) = truncate!(c.blocks, seq, n)
release!(c::PagedKVCache, seq::Sequence) = release!(c.blocks, seq)
