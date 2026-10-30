# Paged causal grouped-query attention (FlashAttention-2 online softmax). The KV loop walks
# the sequence's block table, one page per step. Scores are computed in Float32 with exp2
# (the scale is pre-multiplied by log2(e)); probabilities are rounded to T for the PV product.
#
# - Prefill (many tokens): grid (query tiles, query heads), one TM-row query tile per block,
#   as in cuTile's fmha example. Only pages that overlap the diagonal are masked.
# - Decode / verify (a few tokens): GQA packing and split-KV (flash-decoding). The
#   group × n query rows that share a KV head form one tile, and a fixed number of splits
#   divide the pages between blocks, so a single sequence still fills the GPU. Each split
#   writes its unnormalized partial (acc, m, l) to a Float32 workspace; a combine kernel
#   merges them. The grid does not depend on the context length (splits without pages
#   write a neutral partial), which keeps the launch shape fixed for CUDA graphs.
#
# Positions are 1-based here: query t of the step is at position start + t and sees keys at
# positions ≤ start + t, where start (the 0-based position of the first query) is read from
# the device, so a captured CUDA graph stays valid as the sequence grows. Without CAUSAL,
# every query sees keys up to len = start + n: its position is raised to len for the mask.
#
# Arrays are passed as 2-D views to keep launches under 32 kernel parameters (see the cuTile
# notes in CLAUDE.md): a pool viewed as (head_dim, page_size × kv_heads × pages) holds the
# keys of page p and KV head h as tile column (p - 1) kv_heads + h of width page_size, and
# per-head tensors viewed as (head_dim × heads, n) hold head h of a token as tile row h.

const LOG2_E = Float32(1 / log(2))
# Finite stand-in for -∞ as the initial running max: a row whose keys are all masked in a
# split then gets m = NEG_BIG, l = 0 (no -∞ - -∞ = NaN), and weight 0 when combined.
const NEG_BIG = -floatmax(Float32)

# One page of the online softmax for the query tile q (D, M) at positions qpos (1, M).
# Keys of the page are at positions j PAGE + 1 : (j + 1) PAGE (j is 0-based).
@inline function attend_page(K, V, q, qpos, m_i, l_i, acc, col, j, qk_scale, MASK, ::Type{T},
                             D, PAGE, M) where {T}
    k = ct.load(K; index = (col, Int32(1)), shape = (PAGE, D), order = (2, 1), latency = 2)
    qk = muladd(k, q, zeros(Float32, (PAGE, M)))                  # (PAGE, M)
    if MASK
        kpos = j * Int32(PAGE) .+ reshape(ct.arange(PAGE; dtype = Int32), (PAGE, 1))
        qk = qk .+ ifelse.(kpos .<= qpos, 0.0f0, -Inf32)
    end
    m_ij = max.(m_i, maximum(qk; dims = 1) .* qk_scale)
    p, alpha = ct.@fpmode flush_to_zero = true begin
        exp2.(qk .* qk_scale .- m_ij), exp2.(m_i .- m_ij)
    end
    l_i = l_i .* alpha .+ sum(p; dims = 1)
    v = ct.load(V; index = (Int32(1), col), shape = (D, PAGE), latency = 4)
    acc = muladd(v, convert(ct.Tile{T}, p), acc .* alpha)
    return m_ij, l_i, acc
end

# Tile column of page j + 1 of the block table for KV head h.
@inline page_column(BT, j, h, KV_HEADS) = (BT[j + Int32(1)] - Int32(1)) * Int32(KV_HEADS) + h

function attention_prefill_kernel(Out::ct.TileArray{T,2}, Q::ct.TileArray{T,2}, K::ct.TileArray{T,2},
                                  V::ct.TileArray{T,2}, BT::ct.TileArray{Int32,1}, Pos::ct.TileArray{Int32,1},
                                  qk_scale::Float32, D::Int, GROUP::Int, KV_HEADS::Int, TM::Int,
                                  PAGE::Int, CAUSAL::Bool) where {T}
    ct.@compiler_options occupancy = 2
    bx, head = ct.bid(1), ct.bid(2)
    start = Pos[Int32(1)]
    h = cld(head, Int32(GROUP))
    len = start + size(Q, 2)
    first_pos = start + (bx - Int32(1)) * Int32(TM)               # queries first_pos+1 : first_pos+TM
    qpos = first_pos .+ reshape(ct.arange(TM; dtype = Int32), (1, TM))
    CAUSAL || (qpos = max.(qpos, len))
    q = ct.load(Q; index = (head, bx), shape = (D, TM))
    qk_scale = qk_scale * LOG2_E
    m_i, l_i, acc = fill(NEG_BIG, (1, TM)), zeros(Float32, (1, TM)), zeros(Float32, (D, TM))
    # Pages before mask_start hold only keys every query of the tile sees.
    mask_start = fld(CAUSAL ? first_pos : len, Int32(PAGE))
    last_key = CAUSAL ? min(first_pos + Int32(TM), len) : len
    for j in Int32(0):cld(last_key, Int32(PAGE)) - Int32(1)
        col = page_column(BT, j, h, KV_HEADS)
        if j >= mask_start
            m_i, l_i, acc = attend_page(K, V, q, qpos, m_i, l_i, acc, col, j, qk_scale, true, T, D, PAGE, TM)
        else
            m_i, l_i, acc = attend_page(K, V, q, qpos, m_i, l_i, acc, col, j, qk_scale, false, T, D, PAGE, TM)
        end
    end
    o = ct.@fpmode rounding_mode = ct.Rounding.Approx flush_to_zero = true begin
        acc ./ l_i
    end
    ct.store(Out; index = (head, bx), tile = convert(ct.Tile{T}, o))
    return nothing
end

# Grid (kv_heads, SPLITS). Column c of the packed query tile is head (h - 1) GROUP + g of
# token t, with c = g + GROUP (t - 1); NT ≥ n is the padded token count (decode_tokens).
# The partial of split s goes to column (h - 1) SPLITS + s of the workspace: acc in rows
# 1:D, m in row D + 1 and l in row D + 2.
function attention_decode_kernel(Part::ct.TileArray{Float32,3}, Q::ct.TileArray{T,2}, K::ct.TileArray{T,2},
                                 V::ct.TileArray{T,2}, BT::ct.TileArray{Int32,1}, Pos::ct.TileArray{Int32,1},
                                 qk_scale::Float32, D::Int, GROUP::Int, KV_HEADS::Int, NT::Int,
                                 PAGE::Int, SPLITS::Int, CAUSAL::Bool) where {T}
    ct.@compiler_options occupancy = 2
    h, s = ct.bid(1), ct.bid(2)
    start = Pos[Int32(1)]
    M = GROUP * NT
    pages = cld(start + size(Q, 2), Int32(PAGE))
    per_split = cld(pages, Int32(SPLITS))
    q = reshape(ct.load(Q; index = (h, Int32(1)), shape = (D * GROUP, NT),
                        padding_mode = ct.PaddingMode.Zero), (D, M))
    qpos = start .+ reshape(cld.(ct.arange(M; dtype = Int32), Int32(GROUP)), (1, M))
    CAUSAL || (qpos = max.(qpos, start + size(Q, 2)))
    qk_scale = qk_scale * LOG2_E
    m_i, l_i, acc = fill(NEG_BIG, (1, M)), zeros(Float32, (1, M)), zeros(Float32, (D, M))
    for j in (s - Int32(1)) * per_split:min(s * per_split, pages) - Int32(1)
        col = page_column(BT, j, h, KV_HEADS)
        m_i, l_i, acc = attend_page(K, V, q, qpos, m_i, l_i, acc, col, j, qk_scale, true, T, D, PAGE, M)
    end
    part = (h - Int32(1)) * Int32(SPLITS) + s
    ct.store(Part; index = (Int32(1), Int32(1), part), tile = reshape(acc, (D, M, 1)))
    ct.store(Part; index = (Int32(D + 1), Int32(1), part), tile = reshape(m_i, (1, M, 1)))
    ct.store(Part; index = (Int32(D + 2), Int32(1), part), tile = reshape(l_i, (1, M, 1)))
    return nothing
end

# Grid (GROUP n, kv_heads): merges the splits of one packed query column.
function attention_combine_kernel(Out::ct.TileArray{T,2}, Part::ct.TileArray{Float32,3},
                                  D::Int, GROUP::Int, SPLITS::Int) where {T}
    c, h = ct.bid(1), ct.bid(2)
    acc = reshape(ct.load(Part; index = (Int32(1), c, h), shape = (D, 1, SPLITS)), (D, SPLITS))
    m = reshape(ct.load(Part; index = (Int32(D + 1), c, h), shape = (1, 1, SPLITS)), (1, SPLITS))
    l = reshape(ct.load(Part; index = (Int32(D + 2), c, h), shape = (1, 1, SPLITS)), (1, SPLITS))
    w = exp2.(m .- maximum(m; dims = 2))                         # (1, SPLITS)
    o = sum(acc .* w; dims = 2) ./ sum(l .* w; dims = 2)         # (D, 1)
    head = (h - Int32(1)) * Int32(GROUP) + mod1(c, Int32(GROUP))
    ct.store(Out; index = (head, cld(c, Int32(GROUP))), tile = convert(ct.Tile{T}, o))
    return nothing
end

# Tuned on Qwen3-0.6B (bench/run.jl, ncu): 32-row prefill tiles beat 64 (1.3×) and 128 at
# n = 512 by giving 256 blocks. Decode tiles padded from 2 to 16 query rows and occupancy 2
# cut the decode kernel from 15 to 10 µs at 1024 positions (fewer spills, 2 blocks per SM).
const PREFILL_TM = 32              # query rows per prefill block
const DECODE_MAX_TOKENS = 8        # up to this many tokens use the split-KV decode kernels
const DECODE_BLOCKS = 160          # target decode blocks (2 per SM)
const DECODE_MIN_ROWS = 16         # pad packed decode query tiles to at least this many rows

decode_splits(kv_heads::Integer) = clamp(nextpow(2, cld(DECODE_BLOCKS, kv_heads)), 1, 64)

uses_decode_kernel(group::Integer, n::Integer) = ispow2(group) && n <= DECODE_MAX_TOKENS

# Padded token count of the packed decode query tile: at least DECODE_MIN_ROWS rows.
decode_tokens(group::Integer, n::Integer) = max(nextpow(2, n), cld(DECODE_MIN_ROWS, group))

"""
    decode_workspace(q, k_pages) -> CuArray{Float32,3}

Split-KV partials for decode attention of the queries `q` (head_dim, heads, n): for each
packed query row, split and KV head, the unnormalized output (rows 1:head_dim), running
max and running sum (rows head_dim + 1 and + 2).
"""
function decode_workspace(q::CuArray{<:Any,3}, k_pages::CuArray{<:Any,4})
    D, heads, n = size(q)
    kv_heads = size(k_pages, 3)
    rows = heads ÷ kv_heads * decode_tokens(heads ÷ kv_heads, n)
    return CuArray{Float32}(undef, D + 2, rows, decode_splits(kv_heads) * kv_heads)
end

# The 2-D views the kernels take (no copies).
pool_view(pages::CuArray{<:Any,4}) = reshape(pages, size(pages, 1), :)
heads_view(x::CuArray{<:Any,3}) = reshape(x, :, size(x, 3))

function attention_prefill!(::TileBackend, out::CuArray{T,3}, q::CuArray{T,3}, k_pages::CuArray{T,4},
                            v_pages::CuArray{T,4}, block_table::CuVector{Int32}, positions::CuVector{Int32},
                            scale::Real, causal::Bool) where {T}
    D, heads, n = size(q)
    P, kv_heads = size(k_pages, 2), size(k_pages, 3)
    @cuda backend = cuTile blocks = (cld(n, PREFILL_TM), heads) attention_prefill_kernel(
        heads_view(out), heads_view(q), pool_view(k_pages), pool_view(v_pages), block_table,
        positions, Float32(scale), ct.Constant(D), ct.Constant(heads ÷ kv_heads),
        ct.Constant(kv_heads), ct.Constant(PREFILL_TM), ct.Constant(P), ct.Constant(causal))
    return out
end

function attention_decode!(::TileBackend, out::CuArray{T,3}, part::CuArray{Float32,3}, q::CuArray{T,3},
                           k_pages::CuArray{T,4}, v_pages::CuArray{T,4}, block_table::CuVector{Int32},
                           positions::CuVector{Int32}, scale::Real, causal::Bool) where {T}
    D, heads, n = size(q)
    P, kv_heads = size(k_pages, 2), size(k_pages, 3)
    group, splits = heads ÷ kv_heads, size(part, 3) ÷ kv_heads
    @cuda backend = cuTile blocks = (kv_heads, splits) attention_decode_kernel(
        part, heads_view(q), pool_view(k_pages), pool_view(v_pages), block_table, positions,
        Float32(scale), ct.Constant(D), ct.Constant(group), ct.Constant(kv_heads), ct.Constant(decode_tokens(group, n)),
        ct.Constant(P), ct.Constant(splits), ct.Constant(causal))
    @cuda backend = cuTile blocks = (group * n, kv_heads) attention_combine_kernel(
        heads_view(out), part, ct.Constant(D), ct.Constant(group), ct.Constant(splits))
    return out
end

attention_workspace(::TileBackend, q::CuArray{<:Any,3}, k_pages::CuArray{<:Any,4}) =
    uses_decode_kernel(size(q, 2) ÷ size(k_pages, 3), size(q, 3)) ? decode_workspace(q, k_pages) : nothing

# The decode kernels when there is a split-KV workspace, the prefill kernel otherwise.
function attention!(be::TileBackend, out::CuArray{T,3}, workspace::Union{Nothing,CuArray{Float32,3}},
                    q::CuArray{T,3}, k_pages::CuArray{T,4}, v_pages::CuArray{T,4},
                    block_table::CuVector{Int32}, positions::CuVector{Int32}, scale::Real;
                    causal::Bool = true) where {T}
    workspace === nothing &&
        return attention_prefill!(be, out, q, k_pages, v_pages, block_table, positions, scale, causal)
    return attention_decode!(be, out, workspace, q, k_pages, v_pages, block_table, positions, scale, causal)
end

attention(be::TileBackend, q::CuArray{T,3}, k_pages::CuArray{T,4}, v_pages::CuArray{T,4},
          block_table::CuVector{Int32}, positions::CuVector{Int32}, scale::Real; causal::Bool = true) where {T} =
    attention!(be, similar(q), attention_workspace(be, q, k_pages), q, k_pages, v_pages, block_table,
               positions, scale; causal)

supports_graphs(::TileBackend) = true
