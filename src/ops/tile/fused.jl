# Fused kernels: several primitive ops in one launch, rounding at the same points as the
# separate kernels, so the results are bit-identical to running those.

# --- Attention inputs: q/k RMSNorm + rope + KV-cache write, straight from the fused qkv. ---
#
# One block per (token, head). Arrays are 2-D views (see the cuTile notes in CLAUDE.md):
# qkv as (head_dim, all_heads × n), whose column for head j of token t is j + all_heads (t-1)
# (q heads first, then k, then v); the rope table as interleaved cos/sin columns.

# Head `col` of qkv, RMS-normalized with `W` when NORM (as rmsnorm_kernel: round the
# normalized value to T, scale, round again) and rotated by the angles of position `pos`,
# as two (HALF, 1) halves (as rope_kernel).
@inline function norm_rope(QKV, col, CS, pos, W, eps, NORM, ::Type{T}, D, HALF) where {T}
    f32(tile) = convert(ct.Tile{Float32}, tile)
    rnd(tile) = f32(convert(ct.Tile{T}, tile))
    x1 = f32(ct.load(QKV; index = (Int32(1), col), shape = (HALF, 1)))
    x2 = f32(ct.load(QKV; index = (Int32(2), col), shape = (HALF, 1)))
    if NORM
        x = f32(ct.load(QKV; index = (Int32(1), col), shape = (D, 1)))
        inv_rms = 1.0f0 ./ sqrt.(sum(x .* x; dims = 1) ./ Float32(D) .+ eps)
        w1 = reshape(f32(ct.load(W; index = Int32(1), shape = (HALF,))), (HALF, 1))
        w2 = reshape(f32(ct.load(W; index = Int32(2), shape = (HALF,))), (HALF, 1))
        x1 = rnd(rnd(x1 .* inv_rms) .* w1)
        x2 = rnd(rnd(x2 .* inv_rms) .* w2)
    end
    c, s = Int32(2) * pos + Int32(1), Int32(2) * pos + Int32(2)          # table columns
    c1, c2 = f32(ct.load(CS; index = (Int32(1), c), shape = (HALF, 1))), f32(ct.load(CS; index = (Int32(2), c), shape = (HALF, 1)))
    s1, s2 = f32(ct.load(CS; index = (Int32(1), s), shape = (HALF, 1))), f32(ct.load(CS; index = (Int32(2), s), shape = (HALF, 1)))
    y1 = rnd(x1 .* c1) .+ rnd(.-x2 .* s1)
    y2 = rnd(x2 .* c2) .+ rnd(x1 .* s2)
    return convert(ct.Tile{T}, y1), convert(ct.Tile{T}, y2)
end

# Grid (n, heads): query heads into Q (head_dim, heads × n).
function q_inputs_kernel(Q::ct.TileArray{T,2}, QKV::ct.TileArray{T,2}, CS::ct.TileArray{T,2},
                         W::ct.TileArray{T,1}, Pos::ct.TileArray{Int32,1}, eps::Float32, D::Int,
                         HEADS::Int, ALL_HEADS::Int, NORM::Bool) where {T}
    t, j = ct.bid(1), ct.bid(2)
    y1, y2 = norm_rope(QKV, j + Int32(ALL_HEADS) * (t - Int32(1)), CS, Pos[t], W, eps, NORM, T, D, D ÷ 2)
    col = j + Int32(HEADS) * (t - Int32(1))
    ct.store(Q; index = (Int32(1), col), tile = y1)
    ct.store(Q; index = (Int32(2), col), tile = y2)
    return nothing
end

# Grid (n, kv_heads): key head h (normalized, rotated) and value head h into the pools,
# viewed as (head_dim, page_size × kv_heads × pages) as in store_kv_kernel.
function kv_inputs_kernel(KP::ct.TileArray{T,2}, VP::ct.TileArray{T,2}, QKV::ct.TileArray{T,2},
                          CS::ct.TileArray{T,2}, W::ct.TileArray{T,1}, Pos::ct.TileArray{Int32,1},
                          Slots::ct.TileArray{Int32,1}, eps::Float32, D::Int, P::Int, Q_HEADS::Int,
                          KV_HEADS::Int, NORM::Bool) where {T}
    t, h = ct.bid(1), ct.bid(2)
    base = Int32(Q_HEADS + 2KV_HEADS) * (t - Int32(1))
    y1, y2 = norm_rope(QKV, base + Int32(Q_HEADS) + h, CS, Pos[t], W, eps, NORM, T, D, D ÷ 2)
    slot = Slots[t] - Int32(1)
    dst = rem(slot, Int32(P)) + fld(slot, Int32(P)) * Int32(P * KV_HEADS) + (h - Int32(1)) * Int32(P) + Int32(1)
    ct.store(KP; index = (Int32(1), dst), tile = y1)
    ct.store(KP; index = (Int32(2), dst), tile = y2)
    v = ct.load(QKV; index = (Int32(1), base + Int32(Q_HEADS + KV_HEADS) + h), shape = (D, 1))
    ct.store(VP; index = (Int32(1), dst), tile = v)
    return nothing
end

# The same into an INT8 cache: KV, S and C are the layer's Int8 pools, scales and the keys'
# smoothing (store_int8!).
function kv_inputs_int8_kernel(KV::ct.TileArray{Int8,2}, S::ct.TileArray{Float32,1}, C::ct.TileArray{Float32,1},
                               QKV::ct.TileArray{T,2},
                               CS::ct.TileArray{T,2}, W::ct.TileArray{T,1}, Pos::ct.TileArray{Int32,1},
                               Slots::ct.TileArray{Int32,1}, eps::Float32, D::Int, P::Int, Q_HEADS::Int,
                               KV_HEADS::Int, NORM::Bool) where {T}
    t, h = ct.bid(1), ct.bid(2)
    f32(tile) = convert(ct.Tile{Float32}, tile)
    base = Int32(Q_HEADS + 2KV_HEADS) * (t - Int32(1))
    y1, y2 = norm_rope(QKV, base + Int32(Q_HEADS) + h, CS, Pos[t], W, eps, NORM, T, D, D ÷ 2)
    slot = Slots[t] - Int32(1)
    dst = rem(slot, Int32(P)) + fld(slot, Int32(P)) * Int32(P * KV_HEADS) + (h - Int32(1)) * Int32(P) + Int32(1)
    c1, c2 = smoothing(C, D ÷ 2)
    store_int8!(KV, S, dst, f32(y1) ./ c1, f32(y2) ./ c2)
    v = base + Int32(Q_HEADS + KV_HEADS) + h
    store_int8!(KV, S, dst + size(S, 1) ÷ Int32(2),
                f32(ct.load(QKV; index = (Int32(1), v), shape = (D ÷ 2, 1))),
                f32(ct.load(QKV; index = (Int32(2), v), shape = (D ÷ 2, 1))))
    return nothing
end

function attention_inputs!(::TileBackend, q::CuArray{T,3}, k_pages::Int8Pages, v_pages::Int8Pages,
                           qkv::CuMatrix{T}, q_norm::Union{Nothing,CuVector{T}}, k_norm::Union{Nothing,CuVector{T}},
                           table::RotaryTable{T}, positions::CuVector{Int32}, slots::CuVector{Int32},
                           eps::Real) where {T}
    D, nq, n = size(q)
    P, nkv = size(k_pages, 2), size(k_pages, 3)
    KV, S, C = layer_arrays(k_pages, v_pages)
    norm = q_norm !== nothing
    unused = view(qkv, 1:D, 1)
    flat(x) = reshape(x, D, :)
    @cuda backend = cuTile blocks = (n, nq) q_inputs_kernel(
        flat(q), flat(qkv), table.cos_sin, norm ? q_norm : unused, positions, Float32(eps),
        ct.Constant(D), ct.Constant(nq), ct.Constant(nq + 2nkv), ct.Constant(norm))
    @cuda backend = cuTile blocks = (n, nkv) kv_inputs_int8_kernel(
        KV, S, C, flat(qkv), table.cos_sin, norm ? k_norm : unused, positions, slots,
        Float32(eps), ct.Constant(D), ct.Constant(P), ct.Constant(nq), ct.Constant(nkv), ct.Constant(norm))
    return q
end

function attention_inputs!(::TileBackend, q::CuArray{T,3}, k_pages::CuArray{T,4}, v_pages::CuArray{T,4},
                           qkv::CuMatrix{T}, q_norm::Union{Nothing,CuVector{T}}, k_norm::Union{Nothing,CuVector{T}},
                           table::RotaryTable{T}, positions::CuVector{Int32}, slots::CuVector{Int32},
                           eps::Real) where {T}
    D, nq, n = size(q)
    P, nkv = size(k_pages, 2), size(k_pages, 3)
    norm = q_norm !== nothing
    # Without q/k norm the weight argument is unused; any T vector will do.
    unused = view(qkv, 1:D, 1)
    flat(x) = reshape(x, D, :)
    @cuda backend = cuTile blocks = (n, nq) q_inputs_kernel(
        flat(q), flat(qkv), table.cos_sin, norm ? q_norm : unused, positions, Float32(eps),
        ct.Constant(D), ct.Constant(nq), ct.Constant(nq + 2nkv), ct.Constant(norm))
    @cuda backend = cuTile blocks = (n, nkv) kv_inputs_kernel(
        flat(k_pages), flat(v_pages), flat(qkv), table.cos_sin, norm ? k_norm : unused, positions, slots,
        Float32(eps), ct.Constant(D), ct.Constant(P), ct.Constant(nq), ct.Constant(nkv), ct.Constant(norm))
    return q
end
