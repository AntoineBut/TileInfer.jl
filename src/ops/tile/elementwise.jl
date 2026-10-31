# Element-wise and data-movement kernels: rotary embedding, SwiGLU, xIELU, KV-cache writes
# and embedding lookup. All are memory-bound; each rounds to T where the reference does.

# Rotary embedding, one block per token over all heads. The two halves of head_dim are
# loaded as separate tiles (tile index 1 and 2 along the first axis), which is exactly the
# "rotate half" pairing: out1 = x1 cos1 - x2 sin1, out2 = x2 cos2 + x1 sin2. The cos/sin
# columns are looked up from the token's position, so positions can live on the device.
function rope_kernel(Y::ct.TileArray{T,3}, X::ct.TileArray{T,3}, C::ct.TileArray{T,2},
                     S::ct.TileArray{T,2}, Pos::ct.TileArray{Int32,1}, HALF::Int, HEADS::Int) where {T}
    t = ct.bid(1)
    p = Pos[t] + Int32(1)                                 # table column of the position
    f32(tile) = convert(ct.Tile{Float32}, tile)
    rnd(tile) = f32(convert(ct.Tile{T}, tile))           # round to T, keep computing in Float32
    x1 = f32(ct.load(X; index = (Int32(1), Int32(1), t), shape = (HALF, HEADS, 1)))
    x2 = f32(ct.load(X; index = (Int32(2), Int32(1), t), shape = (HALF, HEADS, 1)))
    c1 = reshape(f32(ct.load(C; index = (Int32(1), p), shape = (HALF, 1))), (HALF, 1, 1))
    c2 = reshape(f32(ct.load(C; index = (Int32(2), p), shape = (HALF, 1))), (HALF, 1, 1))
    s1 = reshape(f32(ct.load(S; index = (Int32(1), p), shape = (HALF, 1))), (HALF, 1, 1))
    s2 = reshape(f32(ct.load(S; index = (Int32(2), p), shape = (HALF, 1))), (HALF, 1, 1))
    y1 = rnd(x1 .* c1) .+ rnd(.-x2 .* s1)
    y2 = rnd(x2 .* c2) .+ rnd(x1 .* s2)
    ct.store(Y; index = (Int32(1), Int32(1), t), tile = convert(ct.Tile{T}, y1))
    ct.store(Y; index = (Int32(2), Int32(1), t), tile = convert(ct.Tile{T}, y2))
    return nothing
end

function rope!(::TileBackend, y::CuArray{T,3}, x::CuArray{T,3}, cos::CuMatrix{T}, sin::CuMatrix{T},
               positions::CuVector{Int32}) where {T}
    D, heads, n = size(x)
    # Heads are padded to a power of two: loads pad, stores clip.
    @cuda backend = cuTile blocks = n rope_kernel(y, x, cos, sin, positions, ct.Constant(D ÷ 2),
                                                  ct.Constant(nextpow(2, heads)))
    return y
end

rope(be::TileBackend, x::CuArray{T,3}, cos::CuMatrix{T}, sin::CuMatrix{T}, positions::CuVector{Int32}) where {T} =
    rope!(be, similar(x), x, cos, sin, positions)

# SwiGLU: out = silu(gate) * up, with gate and up as separate (strided) views of the
# stacked gate_up activations. Grid: (tiles along intermediate, tokens).
function silu_mul_kernel(Y::ct.TileArray{T,2}, G::ct.TileArray{T,2}, U::ct.TileArray{T,2},
                         TILE::Int) where {T}
    i, t = ct.bid(1), ct.bid(2)
    g = convert(ct.Tile{Float32}, ct.load(G; index = (i, t), shape = (TILE, 1)))
    u = convert(ct.Tile{Float32}, ct.load(U; index = (i, t), shape = (TILE, 1)))
    act = convert(ct.Tile{Float32}, convert(ct.Tile{T}, g ./ (1.0f0 .+ exp.(.-g))))
    ct.store(Y; index = (i, t), tile = convert(ct.Tile{T}, act .* u))
    return nothing
end

const ELEMENTWISE_TILE = 1024

function silu_mul!(::TileBackend, y::CuMatrix{T}, gate_up::CuMatrix{T}) where {T}
    I, n = size(y)
    tile = min(nextpow(2, I), ELEMENTWISE_TILE)
    @cuda backend = cuTile blocks = (cld(I, tile), n) silu_mul_kernel(
        y, view(gate_up, 1:I, :), view(gate_up, I+1:2I, :), ct.Constant(tile))
    return y
end

silu_mul(be::TileBackend, gate_up::CuMatrix{T}) where {T} =
    silu_mul!(be, similar(gate_up, size(gate_up, 1) ÷ 2, size(gate_up, 2)), gate_up)

# expm1 without exp(x) - 1's cancellation (Kahan): with u = exp(x) rounded, (u - 1) x / log(u)
# is accurate to a few ulps, since u's rounding error cancels between the two factors.
# Below -20, expm1 is -1 in Float32 (and log(u) underflows further down).
@inline function expm1_tile(x)
    u = exp.(x)
    um1 = u .- 1.0f0
    return ifelse.(x .< -20.0f0, -1.0f0, ifelse.(um1 .== 0.0f0, x, um1 .* x ./ log.(u)))
end

# xIELU of a Float32 tile, rounding to T after each operation as the reference does.
@inline function xielu_tile(::Type{T}, x, alpha_p, alpha_n, beta, eps) where {T}
    rnd(tile) = convert(ct.Tile{Float32}, convert(ct.Tile{T}, tile))
    bx = rnd(beta .* x)
    pos = rnd(rnd(rnd(alpha_p .* x) .* x) .+ bx)
    neg = rnd(rnd(rnd(rnd(expm1_tile(min.(x, eps))) .- x) .* alpha_n) .+ bx)
    return ifelse.(x .> 0.0f0, pos, neg)
end

# xIELU, grid (tiles along the rows, tokens). The scalars are per layer and fixed, so
# launch arguments are fine inside a CUDA graph.
function xielu_kernel(Y::ct.TileArray{T,2}, X::ct.TileArray{T,2}, alpha_p::Float32, alpha_n::Float32,
                      beta::Float32, eps::Float32, TILE::Int) where {T}
    i, t = ct.bid(1), ct.bid(2)
    x = convert(ct.Tile{Float32}, ct.load(X; index = (i, t), shape = (TILE, 1)))
    ct.store(Y; index = (i, t), tile = convert(ct.Tile{T}, xielu_tile(T, x, alpha_p, alpha_n, beta, eps)))
    return nothing
end

xielu_args(a::XIELU) = (Float32(a.alpha_p), Float32(a.alpha_n), Float32(a.beta), Float32(a.eps))

function xielu!(::TileBackend, y::CuMatrix{T}, x::CuMatrix{T}, a::XIELU{T}) where {T}
    I, n = size(y)
    tile = min(nextpow(2, I), ELEMENTWISE_TILE)
    @cuda backend = cuTile blocks = (cld(I, tile), n) xielu_kernel(y, x, xielu_args(a)..., ct.Constant(tile))
    return y
end

xielu(be::TileBackend, x::CuMatrix{T}, a::XIELU{T}) where {T} = xielu!(be, similar(x), x, a)

# KV-cache write, one block per token. Arrays are 2-D views, which keeps the launch under
# 32 kernel parameters: the pools as (head_dim, page_size × kv_heads × pages), whose column
# for (offset, head h, page), all 1-based, is offset + P (h - 1) + P H (page - 1), and k/v
# as (head_dim, kv_heads × n). Tile indices along the size-1 axis are element indices.
function store_kv_kernel(KP::ct.TileArray{T,2}, VP::ct.TileArray{T,2}, K::ct.TileArray{T,2},
                         V::ct.TileArray{T,2}, Slots::ct.TileArray{Int32,1}, D::Int, P::Int,
                         HEADS::Int) where {T}
    t = ct.bid(1)
    slot = Slots[t] - Int32(1)
    first_col = rem(slot, Int32(P)) + fld(slot, Int32(P)) * Int32(P * HEADS) + Int32(1)
    for h in Int32(1):Int32(HEADS)
        src, dst = (t - Int32(1)) * Int32(HEADS) + h, first_col + (h - Int32(1)) * Int32(P)
        ct.store(KP; index = (Int32(1), dst), tile = ct.load(K; index = (Int32(1), src), shape = (D, 1)))
        ct.store(VP; index = (Int32(1), dst), tile = ct.load(V; index = (Int32(1), src), shape = (D, 1)))
    end
    return nothing
end

function store_kv!(::TileBackend, k_pages::CuArray{T,4}, v_pages::CuArray{T,4}, k::CuArray{T,3},
                   v::CuArray{T,3}, slots::CuVector{Int32}) where {T}
    D, P, heads, _ = size(k_pages)
    flat(x) = reshape(x, D, :)
    @cuda backend = cuTile blocks = length(slots) store_kv_kernel(
        flat(k_pages), flat(v_pages), flat(k), flat(v), slots, ct.Constant(D), ct.Constant(P), ct.Constant(heads))
    return nothing
end

# INT8 KV cache (Int8Pages): the layer's keys and values as one (head_dim, 2N) Int8 array KV,
# values from column N + 1 on, their scales as one vector S of length 2N, and the keys'
# smoothing factors C (head_dim).

# The two halves of the smoothing factors, as (HALF, 1) tiles.
@inline smoothing(C, HALF) = (reshape(ct.load(C; index = Int32(1), shape = (HALF,)), (HALF, 1)),
                              reshape(ct.load(C; index = Int32(2), shape = (HALF,)), (HALF, 1)))

# Quantize the vector whose two halves are the Float32 (HALF, 1) tiles x1 and x2 into column
# `col` of KV and its scale into S[col], as quantize_columns does.
@inline function store_int8!(KV, S, col, x1, x2)
    absmax = max.(maximum(abs.(x1); dims = 1), maximum(abs.(x2); dims = 1))     # (1, 1)
    inv = ifelse.(absmax .> 0.0f0, 127.0f0 ./ absmax, 0.0f0)
    ct.store(KV; index = (Int32(1), col), tile = convert(ct.Tile{Int8}, floor.(x1 .* inv .+ 0.5f0)))
    ct.store(KV; index = (Int32(2), col), tile = convert(ct.Tile{Int8}, floor.(x2 .* inv .+ 0.5f0)))
    ct.store(S; index = col, tile = reshape(absmax ./ 127.0f0, (1,)))
    return nothing
end

function store_kv_int8_kernel(KV::ct.TileArray{Int8,2}, S::ct.TileArray{Float32,1}, C::ct.TileArray{Float32,1},
                              K::ct.TileArray{T,2}, V::ct.TileArray{T,2}, Slots::ct.TileArray{Int32,1}, D::Int,
                              P::Int, HEADS::Int) where {T}
    t = ct.bid(1)
    slot = Slots[t] - Int32(1)
    first_col = rem(slot, Int32(P)) + fld(slot, Int32(P)) * Int32(P * HEADS) + Int32(1)
    values = size(S, 1) ÷ Int32(2)                    # column offset of the values
    half(X, i, src) = convert(ct.Tile{Float32}, ct.load(X; index = (i, src), shape = (D ÷ 2, 1)))
    c1, c2 = smoothing(C, D ÷ 2)
    for h in Int32(1):Int32(HEADS)
        src, dst = (t - Int32(1)) * Int32(HEADS) + h, first_col + (h - Int32(1)) * Int32(P)
        store_int8!(KV, S, dst, half(K, Int32(1), src) ./ c1, half(K, Int32(2), src) ./ c2)
        store_int8!(KV, S, dst + values, half(V, Int32(1), src), half(V, Int32(2), src))
    end
    return nothing
end

function store_kv!(::TileBackend, k_pages::Int8Pages, v_pages::Int8Pages, k::CuArray{T,3},
                   v::CuArray{T,3}, slots::CuVector{Int32}) where {T}
    D, P, heads, _ = size(k_pages)
    KV, S, C = layer_arrays(k_pages, v_pages)
    flat(x) = reshape(x, D, :)
    @cuda backend = cuTile blocks = length(slots) store_kv_int8_kernel(
        KV, S, C, flat(k), flat(v), slots, ct.Constant(D), ct.Constant(P), ct.Constant(heads))
    return nothing
end

# Embedding lookup, one block per token, walking the column in tiles.
function embed_kernel(Y::ct.TileArray{T,2}, Table::ct.TileArray{T,2}, Ids::ct.TileArray{Int32,1},
                      TILE::Int) where {T}
    t = ct.bid(1)
    col = Ids[t] + Int32(1)
    for j in Int32(1):ct.num_tiles(Table, 1, (TILE, 1))
        ct.store(Y; index = (j, t), tile = ct.load(Table; index = (j, col), shape = (TILE, 1)))
    end
    return nothing
end

function embed!(::TileBackend, y::CuMatrix{T}, table::CuMatrix{T}, ids::CuVector{Int32}) where {T}
    H = size(table, 1)
    @cuda backend = cuTile blocks = length(ids) embed_kernel(y, table, ids,
                                                             ct.Constant(min(nextpow(2, H), ELEMENTWISE_TILE)))
    return y
end

embed(be::TileBackend, table::CuMatrix{T}, ids::CuVector{Int32}) where {T} =
    embed!(be, CuMatrix{T}(undef, size(table, 1), length(ids)), table, ids)
