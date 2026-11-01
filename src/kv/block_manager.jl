# Page allocator for the paged KV cache. The cache is a pool of fixed-size pages; each
# sequence owns a block table listing, in order, the pages that hold its positions:
# position p (0-based) lives in page block_table[p ÷ page_size + 1] at offset p % page_size.
# Pages need not be contiguous or ordered, so sequences can grow, shrink and share the
# pool without moving data.

"""
    Sequence

A sequence's place in the paged KV cache: its block table and the token ids whose keys
and values are cached, in position order.
"""
mutable struct Sequence
    block_table::Vector{Int32}       # 1-based page ids
    tokens::Vector{Int}              # 0-based token ids at positions 0, 1, …
end

Base.length(s::Sequence) = length(s.tokens)

"""
    OutOfPagesError(needed, free)

Thrown by [`reserve!`](@ref) when the KV cache has fewer free pages than a step needs. A
[`ChatSession`](@ref) can recover with [`reset!`](@ref).
"""
struct OutOfPagesError <: Exception
    needed::Int
    free::Int
end

Base.showerror(io::IO, e::OutOfPagesError) =
    print(io, "KV cache full: need $(e.needed) more pages, $(e.free) free")

"""
    BlockManager(num_pages, page_size)

Hands out the pages of a pool of `num_pages` pages of `page_size` positions each.
"""
struct BlockManager
    page_size::Int
    num_pages::Int
    free::Vector{Int32}              # stack of free page ids
end

# Reversed so that pages are handed out in increasing order (easier to read in tests).
BlockManager(num_pages::Integer, page_size::Integer) =
    BlockManager(page_size, num_pages, collect(Int32, num_pages:-1:1))

"Number of pages not held by any sequence."
num_free_pages(b::BlockManager) = length(b.free)

"""
    reserve!(blocks, seq, n)

Make sure `seq` has pages for `n` more positions; throws an [`OutOfPagesError`](@ref) when
the pool runs out.
"""
function reserve!(b::BlockManager, seq::Sequence, n::Integer)
    needed = cld(length(seq) + n, b.page_size) - length(seq.block_table)
    needed > num_free_pages(b) && throw(OutOfPagesError(needed, num_free_pages(b)))
    for _ in 1:needed
        push!(seq.block_table, pop!(b.free))
    end
    return seq
end

"""
    truncate!(blocks_or_cache, seq, n)

Drop the cached positions of `seq` after the first `n` and return its unused pages to the
pool. Speculative decoding uses it to roll back rejected drafts, and chat to roll back to
the prefix a new turn shares with the cache.
"""
function truncate!(b::BlockManager, seq::Sequence, n::Integer)
    0 <= n <= length(seq) || throw(ArgumentError("cannot truncate a sequence of length $(length(seq)) to $n"))
    resize!(seq.tokens, n)
    keep = cld(n, b.page_size)
    append!(b.free, reverse!(seq.block_table[keep+1:end]))
    resize!(seq.block_table, keep)
    return seq
end

"""
    release!(blocks_or_cache, seq)

Return all pages of `seq` to the pool.
"""
release!(b::BlockManager, seq::Sequence) = truncate!(b, seq, 0)

"""
    slots(b, seq, start, n) -> Vector{Int32}

Cache slots of positions `start:start+n-1` of `seq` (pages must be reserved). Slot
`(page - 1) * page_size + offset` with a 1-based offset identifies one position of the
pool; it is what [`store_kv!`](@ref) consumes.
"""
function slots(b::BlockManager, seq::Sequence, start::Integer, n::Integer)
    P = b.page_size
    return Int32[(seq.block_table[p ÷ P + 1] - 1) * P + p % P + 1 for p in start:start+n-1]
end
