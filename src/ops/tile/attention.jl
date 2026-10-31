# Paged causal grouped-query attention (FlashAttention-2 online softmax). The KV loop walks
# the sequence's block table, one page per step (half a page for an INT8 cache). Scores are computed in Float32 with exp2
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

# The online-softmax update for the scores qk (KEYS, M) of keys at positions
# first_key + 1 : first_key + KEYS, for queries at positions qpos (1, M): the probabilities
# p (Float32), the factor alpha that rescales the accumulator, and the new running max and sum.
@inline function softmax_step(qk, qpos, m_i, l_i, first_key, qk_scale, MASK, KEYS)
    if MASK
        kpos = first_key .+ reshape(ct.arange(KEYS; dtype = Int32), (KEYS, 1))
        qk = qk .+ ifelse.(kpos .<= qpos, 0.0f0, -Inf32)
    end
    m_ij = max.(m_i, maximum(qk; dims = 1) .* qk_scale)
    p, alpha = ct.@fpmode flush_to_zero = true begin
        exp2.(qk .* qk_scale .- m_ij), exp2.(m_i .- m_ij)
    end
    return p, alpha, m_ij, l_i .* alpha .+ sum(p; dims = 1)
end

# One key tile of the online softmax for the query tile q (D, M) at positions qpos (1, M):
# its KEYS keys, at positions first_key + 1 : first_key + KEYS, are tile column kcol of K
# (width KEYS) and their values tile column vcol of V. S holds the scales of an INT8 cache
# (Int8Pages), at the same tile index as their vectors; it is `nothing` otherwise.
@inline function attend(K, V, ::Nothing, kcol, vcol, q, qpos, m_i, l_i, acc, first_key, qk_scale, MASK,
                        ::Type{T}, D, KEYS, M) where {T}
    k = ct.load(K; index = (kcol, Int32(1)), shape = (KEYS, D), order = (2, 1), latency = 2)
    qk = muladd(k, q, zeros(Float32, (KEYS, M)))                  # (KEYS, M)
    p, alpha, m_i, l_i = softmax_step(qk, qpos, m_i, l_i, first_key, qk_scale, MASK, KEYS)
    v = ct.load(V; index = (Int32(1), vcol), shape = (D, KEYS), latency = 4)
    return m_i, l_i, muladd(v, convert(ct.Tile{T}, p), acc .* alpha)
end

# An INT8 cache: the keys and values are converted to T for the tensor cores, the key's scale
# multiplies its row of scores (s_k q_kᵀ x) and the value's scale its probability
# (V diag(s_v) p). Converting whole (64, 128) pages of keys and values takes 255 registers
# and spills, so the kernels take 32 keys at a time, and K, V, the queries and the
# accumulator in two halves of the head dimension: q = (q1, q2), acc = (acc1, acc2).
@inline function attend(K, V, S, kcol, vcol, (q1, q2), qpos, m_i, l_i, (acc1, acc2), first_key, qk_scale, MASK,
                        ::Type{T}, D, KEYS, M) where {T}
    k1 = ct.load(K; index = (kcol, Int32(1)), shape = (KEYS, D ÷ 2), order = (2, 1), latency = 2)
    k2 = ct.load(K; index = (kcol, Int32(2)), shape = (KEYS, D ÷ 2), order = (2, 1), latency = 2)
    qk = muladd(convert(ct.Tile{T}, k2), q2, muladd(convert(ct.Tile{T}, k1), q1, zeros(Float32, (KEYS, M)))) .*
         reshape(ct.load(S; index = kcol, shape = (KEYS,)), (KEYS, 1))
    p, alpha, m_i, l_i = softmax_step(qk, qpos, m_i, l_i, first_key, qk_scale, MASK, KEYS)
    p = convert(ct.Tile{T}, p .* reshape(ct.load(S; index = vcol, shape = (KEYS,)), (KEYS, 1)))
    v1 = ct.load(V; index = (Int32(1), vcol), shape = (D ÷ 2, KEYS), latency = 4)
    v2 = ct.load(V; index = (Int32(2), vcol), shape = (D ÷ 2, KEYS), latency = 4)
    return m_i, l_i, (muladd(convert(ct.Tile{T}, v1), p, acc1 .* alpha), muladd(convert(ct.Tile{T}, v2), p, acc2 .* alpha))
end

# Keys per step of the KV loop: a page, or half of one for an INT8 cache.
key_tile(::Nothing, PAGE) = PAGE
key_tile(S, PAGE) = PAGE ÷ 2

# The query tile q (D, M) and a zero accumulator as `attend` takes them: as they are, or for
# an INT8 cache (C, the keys' smoothing, not `nothing`) the queries times C, rounded to T, in
# halves of the head dimension, and the accumulator too.
@inline attend_operands(q, ::Nothing, ::Type{T}, D, M) where {T} = q, zeros(Float32, (D, M))
@inline function attend_operands(q, C, ::Type{T}, D, M) where {T}
    q = convert(ct.Tile{T}, convert(ct.Tile{Float32}, q) .* reshape(ct.load(C; index = Int32(1), shape = (D,)), (D, 1)))
    half = (D ÷ 2, M)
    return (ct.extract(q, (1, 1), half), ct.extract(q, (2, 1), half)), (zeros(Float32, half), zeros(Float32, half))
end

# Tile column (width PAGE ÷ R) of the key tile j (0-based) of the sequence for KV head h:
# part j mod R of the R parts of page j ÷ R of the block table.
@inline key_column(BT, j, h, KV_HEADS, R) =
    R == 1 ? page_column(BT, j, h, KV_HEADS) :
             (page_column(BT, fld(j, Int32(R)), h, KV_HEADS) - Int32(1)) * Int32(R) + rem(j, Int32(R)) + Int32(1)

# Tile column of page j + 1 of the block table for KV head h.
@inline page_column(BT, j, h, KV_HEADS) = (BT[j + Int32(1)] - Int32(1)) * Int32(KV_HEADS) + h

# Stores the accumulator (rows 1:D of a tile at `index`, whose other indices are in units
# of the tile), whole or in its halves.
@inline store_acc(A, (i, rest...), acc) = ct.store(A; index = (i, rest...), tile = acc)
@inline function store_acc(A, (i, rest...), (acc1, acc2)::Tuple)
    ct.store(A; index = (Int32(2) * i - Int32(1), rest...), tile = acc1)
    ct.store(A; index = (Int32(2) * i, rest...), tile = acc2)
end

@inline function divide(acc, l_i, ::Type{T}) where {T}
    o = ct.@fpmode rounding_mode = ct.Rounding.Approx flush_to_zero = true begin
        acc ./ l_i
    end
    return convert(ct.Tile{T}, o)
end
@inline divide((acc1, acc2)::Tuple, l_i, ::Type{T}) where {T} = (divide(acc1, l_i, T), divide(acc2, l_i, T))

@inline partial(acc, D, M) = reshape(acc, (D, M, 1))
@inline partial((acc1, acc2)::Tuple, D, M) = (partial(acc1, D ÷ 2, M), partial(acc2, D ÷ 2, M))

# The kernels take the pools as K and V with S = nothing, or an INT8 layer as KV (keys, then
# values from VALUES key tiles on) with its scales S and the keys' smoothing C, which
# multiplies the queries (`attend_operands`).
function attention_prefill_kernel(Out::ct.TileArray{T,2}, Q::ct.TileArray{T,2}, K::ct.TileArray{T,2},
                                  V::ct.TileArray{T,2}, BT::ct.TileArray{Int32,1}, Pos::ct.TileArray{Int32,1},
                                  qk_scale::Float32, D::Int, GROUP::Int, KV_HEADS::Int, TM::Int,
                                  PAGE::Int, CAUSAL::Bool) where {T}
    ct.@compiler_options occupancy = 2
    prefill_body(Out, Q, K, V, nothing, nothing, Int32(0), BT, Pos, qk_scale, T, D, GROUP, KV_HEADS, TM, PAGE, CAUSAL)
    return nothing
end

function attention_prefill_int8_kernel(Out::ct.TileArray{T,2}, Q::ct.TileArray{T,2}, KV::ct.TileArray{Int8,2},
                                       S::ct.TileArray{Float32,1}, C::ct.TileArray{Float32,1}, BT::ct.TileArray{Int32,1},
                                       Pos::ct.TileArray{Int32,1}, qk_scale::Float32, D::Int, GROUP::Int,
                                       KV_HEADS::Int, TM::Int, PAGE::Int, CAUSAL::Bool) where {T}
    ct.@compiler_options occupancy = 2
    values = size(S, 1) ÷ Int32(2 * key_tile(S, PAGE))
    prefill_body(Out, Q, KV, KV, S, C, values, BT, Pos, qk_scale, T, D, GROUP, KV_HEADS, TM, PAGE, CAUSAL)
    return nothing
end

@inline function prefill_body(Out, Q, K, V, S, C, VALUES, BT, Pos, qk_scale, ::Type{T}, D, GROUP, KV_HEADS, TM,
                              PAGE, CAUSAL) where {T}
    bx, head = ct.bid(1), ct.bid(2)
    start = Pos[Int32(1)]
    h = cld(head, Int32(GROUP))
    len = start + size(Q, 2)
    first_pos = start + (bx - Int32(1)) * Int32(TM)               # queries first_pos+1 : first_pos+TM
    qpos = first_pos .+ reshape(ct.arange(TM; dtype = Int32), (1, TM))
    CAUSAL || (qpos = max.(qpos, len))
    q, acc = attend_operands(ct.load(Q; index = (head, bx), shape = (D, TM)), C, T, D, TM)
    qk_scale = qk_scale * LOG2_E
    m_i, l_i = fill(NEG_BIG, (1, TM)), zeros(Float32, (1, TM))
    KEYS = key_tile(S, PAGE)
    # Key tiles before mask_start hold only keys every query of the tile sees.
    mask_start = fld(CAUSAL ? first_pos : len, Int32(KEYS))
    last_key = CAUSAL ? min(first_pos + Int32(TM), len) : len
    for j in Int32(0):cld(last_key, Int32(KEYS)) - Int32(1)
        col = key_column(BT, j, h, KV_HEADS, PAGE ÷ KEYS)
        if j >= mask_start
            m_i, l_i, acc = attend(K, V, S, col, col + VALUES, q, qpos, m_i, l_i, acc, j * Int32(KEYS), qk_scale,
                                   true, T, D, KEYS, TM)
        else
            m_i, l_i, acc = attend(K, V, S, col, col + VALUES, q, qpos, m_i, l_i, acc, j * Int32(KEYS), qk_scale,
                                   false, T, D, KEYS, TM)
        end
    end
    store_acc(Out, (head, bx), divide(acc, l_i, T))
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
    decode_body(Part, Q, K, V, nothing, nothing, Int32(0), BT, Pos, qk_scale, T, D, GROUP, KV_HEADS, NT, PAGE, SPLITS, CAUSAL)
    return nothing
end

function attention_decode_int8_kernel(Part::ct.TileArray{Float32,3}, Q::ct.TileArray{T,2}, KV::ct.TileArray{Int8,2},
                                      S::ct.TileArray{Float32,1}, C::ct.TileArray{Float32,1}, BT::ct.TileArray{Int32,1},
                                      Pos::ct.TileArray{Int32,1}, qk_scale::Float32, D::Int, GROUP::Int,
                                      KV_HEADS::Int, NT::Int, PAGE::Int, SPLITS::Int, CAUSAL::Bool) where {T}
    ct.@compiler_options occupancy = 2
    values = size(S, 1) ÷ Int32(2 * key_tile(S, PAGE))
    decode_body(Part, Q, KV, KV, S, C, values, BT, Pos, qk_scale, T, D, GROUP, KV_HEADS, NT, PAGE, SPLITS, CAUSAL)
    return nothing
end

@inline function decode_body(Part, Q, K, V, S, C, VALUES, BT, Pos, qk_scale, ::Type{T}, D, GROUP, KV_HEADS, NT,
                             PAGE, SPLITS, CAUSAL) where {T}
    h, s = ct.bid(1), ct.bid(2)
    start = Pos[Int32(1)]
    M = GROUP * NT
    pages = cld(start + size(Q, 2), Int32(PAGE))
    per_split = cld(pages, Int32(SPLITS))
    q, acc = attend_operands(reshape(ct.load(Q; index = (h, Int32(1)), shape = (D * GROUP, NT),
                                             padding_mode = ct.PaddingMode.Zero), (D, M)), C, T, D, M)
    qpos = start .+ reshape(cld.(ct.arange(M; dtype = Int32), Int32(GROUP)), (1, M))
    CAUSAL || (qpos = max.(qpos, start + size(Q, 2)))
    qk_scale = qk_scale * LOG2_E
    m_i, l_i = fill(NEG_BIG, (1, M)), zeros(Float32, (1, M))
    KEYS = key_tile(S, PAGE)
    R = PAGE ÷ KEYS
    for j in (s - Int32(1)) * per_split * Int32(R):min(s * per_split, pages) * Int32(R) - Int32(1)
        col = key_column(BT, j, h, KV_HEADS, R)
        m_i, l_i, acc = attend(K, V, S, col, col + VALUES, q, qpos, m_i, l_i, acc, j * Int32(KEYS), qk_scale, true,
                               T, D, KEYS, M)
    end
    part = (h - Int32(1)) * Int32(SPLITS) + s
    store_acc(Part, (Int32(1), Int32(1), part), partial(acc, D, M))
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
const DECODE_MAX_TOKENS = 16       # up to this many tokens use the split-KV decode kernels
const DECODE_BLOCKS = 160          # target decode blocks (2 per SM)
const DECODE_MIN_ROWS = 16         # pad packed decode query tiles to at least this many rows
const VERIFY_SPLITS = 8            # splits for steps of more than 8 tokens

# 9-16 tokens (DFlash's 16-token verify): the prefill kernel runs one block per head over
# every page, 27 µs at 1,024 positions, while the decode kernels have a floor from the
# combine of all splits, empty or not. 8 splits instead of 32 cut that floor (4B, 128
# positions: 12.4 → 6.7 µs against the prefill kernel's 4.8) and win from ~200 positions
# on (1,024: 9.5 µs, 2,048: 14.3 vs 54). A captured graph keeps its kernels as the
# context grows, so the choice cannot depend on it.
decode_splits(kv_heads::Integer, n::Integer = 1) =
    n > 8 ? VERIFY_SPLITS : clamp(nextpow(2, cld(DECODE_BLOCKS, kv_heads)), 1, 64)

uses_decode_kernel(group::Integer, n::Integer) = ispow2(group) && n <= DECODE_MAX_TOKENS

# Padded token count of the packed decode query tile: at least DECODE_MIN_ROWS rows.
decode_tokens(group::Integer, n::Integer) = max(nextpow(2, n), cld(DECODE_MIN_ROWS, group))

"""
    decode_workspace(q, k_pages) -> CuArray{Float32,3}

Split-KV partials for decode attention of the queries `q` (head_dim, heads, n): for each
packed query row, split and KV head, the unnormalized output (rows 1:head_dim), running
max and running sum (rows head_dim + 1 and + 2).
"""
function decode_workspace(q::CuArray{<:Any,3}, k_pages)
    D, heads, n = size(q)
    kv_heads = size(k_pages, 3)
    rows = heads ÷ kv_heads * decode_tokens(heads ÷ kv_heads, n)
    return CuArray{Float32}(undef, D + 2, rows, decode_splits(kv_heads, n) * kv_heads)
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

function attention_prefill!(::TileBackend, out::CuArray{T,3}, q::CuArray{T,3}, k_pages::Int8Pages,
                            v_pages::Int8Pages, block_table::CuVector{Int32}, positions::CuVector{Int32},
                            scale::Real, causal::Bool) where {T}
    D, heads, n = size(q)
    P, kv_heads = size(k_pages, 2), size(k_pages, 3)
    KV, S, C = layer_arrays(k_pages, v_pages)
    @cuda backend = cuTile blocks = (cld(n, PREFILL_TM), heads) attention_prefill_int8_kernel(
        heads_view(out), heads_view(q), KV, S, C, block_table, positions, Float32(scale), ct.Constant(D),
        ct.Constant(heads ÷ kv_heads), ct.Constant(kv_heads), ct.Constant(PREFILL_TM), ct.Constant(P),
        ct.Constant(causal))
    return out
end

function attention_decode!(::TileBackend, out::CuArray{T,3}, part::CuArray{Float32,3}, q::CuArray{T,3},
                           k_pages::Int8Pages, v_pages::Int8Pages, block_table::CuVector{Int32},
                           positions::CuVector{Int32}, scale::Real, causal::Bool) where {T}
    D, heads, n = size(q)
    P, kv_heads = size(k_pages, 2), size(k_pages, 3)
    group, splits = heads ÷ kv_heads, size(part, 3) ÷ kv_heads
    KV, S, C = layer_arrays(k_pages, v_pages)
    @cuda backend = cuTile blocks = (kv_heads, splits) attention_decode_int8_kernel(
        part, heads_view(q), KV, S, C, block_table, positions, Float32(scale), ct.Constant(D), ct.Constant(group),
        ct.Constant(kv_heads), ct.Constant(decode_tokens(group, n)), ct.Constant(P), ct.Constant(splits),
        ct.Constant(causal))
    @cuda backend = cuTile blocks = (group * n, kv_heads) attention_combine_kernel(
        heads_view(out), part, ct.Constant(D), ct.Constant(group), ct.Constant(splits))
    return out
end

# A page pool: (head_dim, page_size, kv_heads, num_pages) in T, or Int8Pages.
const KVPool{T} = Union{CuArray{T,4},Int8Pages}

attention_workspace(::TileBackend, q::CuArray{<:Any,3}, k_pages::KVPool) =
    uses_decode_kernel(size(q, 2) ÷ size(k_pages, 3), size(q, 3)) ? decode_workspace(q, k_pages) : nothing

# The decode kernels when there is a split-KV workspace, the prefill kernel otherwise.
function attention!(be::TileBackend, out::CuArray{T,3}, workspace::Union{Nothing,CuArray{Float32,3}},
                    q::CuArray{T,3}, k_pages::KVPool{T}, v_pages::KVPool{T},
                    block_table::CuVector{Int32}, positions::CuVector{Int32}, scale::Real;
                    causal::Bool = true) where {T}
    workspace === nothing &&
        return attention_prefill!(be, out, q, k_pages, v_pages, block_table, positions, scale, causal)
    return attention_decode!(be, out, workspace, q, k_pages, v_pages, block_table, positions, scale, causal)
end

attention(be::TileBackend, q::CuArray{T,3}, k_pages::KVPool{T}, v_pages::KVPool{T},
          block_table::CuVector{Int32}, positions::CuVector{Int32}, scale::Real; causal::Bool = true) where {T} =
    attention!(be, similar(q), attention_workspace(be, q, k_pages), q, k_pages, v_pages, block_table,
               positions, scale; causal)

supports_graphs(::TileBackend) = true
