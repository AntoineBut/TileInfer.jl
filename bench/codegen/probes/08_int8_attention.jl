# Decode attention over an INT8 KV cache at 32k positions (Qwen3-4B shapes: head_dim 128,
# 8 KV heads, group 4, 32 splits): why converting whole pages spilled, and what fits.
# Each variant converts Int8 keys and/or values to BF16 for the tensor cores (or not), in
# whole (64, 128) pages, 32-key tiles, or halves of the head dimension. Same kernel as
# TileInfer's decode attention otherwise; K and V are separate arrays here, so the launch
# has more than 32 parameters (host time only: the probe reports GPU time).

using Random: randperm

const KCONV_WHOLE, KCONV_HALVES, KCONV_INT8 = 0, 1, 2   # keys: one conversion, two, or Int8 QKᵀ
const VCONV_WHOLE, VCONV_HALVES, VCONV_TRANSPOSED = 0, 1, 2

# K (D, PAGE cols) in KT; V (D, PAGE cols) in VT, or with VCONV_TRANSPOSED (PAGE, D cols):
# each page of a head stored transposed. SK, SV: one scale per vector.
function int8_decode(Part::ct.TileArray{Float32,3}, Q::ct.TileArray{T,2}, K::ct.TileArray{KT,2},
                     V::ct.TileArray{VT,2}, SK::ct.TileArray{Float32,1}, SV::ct.TileArray{Float32,1},
                     BT::ct.TileArray{Int32,1}, Pos::ct.TileArray{Int32,1}, qk_scale::Float32, D::Int, GROUP::Int,
                     KV_HEADS::Int, NT::Int, PAGE::Int, SPLITS::Int, KEYS::Int, KCONV::Int, VCONV::Int) where {T,KT,VT}
    ct.@compiler_options occupancy = 2
    h, s = ct.bid(1), ct.bid(2)
    start = Pos[Int32(1)]
    M = GROUP * NT
    pages = cld(start + size(Q, 2), Int32(PAGE))
    per_split = cld(pages, Int32(SPLITS))
    q = reshape(ct.load(Q; index = (h, Int32(1)), shape = (D * GROUP, NT), padding_mode = ct.PaddingMode.Zero), (D, M))
    qpos = start .+ reshape(cld.(ct.arange(M; dtype = Int32), Int32(GROUP)), (1, M))
    R = Int32(PAGE ÷ KEYS)
    q1, q2 = ct.extract(q, (1, 1), (D ÷ 2, M)), ct.extract(q, (2, 1), (D ÷ 2, M))
    if KCONV == KCONV_INT8                                       # queries per column in Int8
        x = convert(ct.Tile{Float32}, q)
        sx = max.(maximum(abs.(x); dims = 1), 1.0f-20) ./ 127.0f0
        qx = convert(ct.Tile{Int8}, floor.(x ./ sx .+ 0.5f0))
    end
    qk_scale = qk_scale * TI.LOG2_E
    m_i, l_i, acc = fill(TI.NEG_BIG, (1, M)), zeros(Float32, (1, M)), zeros(Float32, (D, M))
    acc1, acc2 = zeros(Float32, (D ÷ 2, M)), zeros(Float32, (D ÷ 2, M))
    for j in (s - Int32(1)) * per_split * R:min(s * per_split, pages) * R - Int32(1)
        page, r = fld(j, R), rem(j, R)
        col = (TI.page_column(BT, page, h, KV_HEADS) - Int32(1)) * R + r + Int32(1)
        sk = reshape(ct.load(SK; index = col, shape = (KEYS,)), (KEYS, 1))
        if KCONV == KCONV_HALVES
            k1 = ct.load(K; index = (col, Int32(1)), shape = (KEYS, D ÷ 2), order = (2, 1), latency = 2)
            k2 = ct.load(K; index = (col, Int32(2)), shape = (KEYS, D ÷ 2), order = (2, 1), latency = 2)
            qk = muladd(convert(ct.Tile{T}, k2), q2, muladd(convert(ct.Tile{T}, k1), q1, zeros(Float32, (KEYS, M)))) .* sk
        else
            k = ct.load(K; index = (col, Int32(1)), shape = (KEYS, D), order = (2, 1), latency = 2)
            qk = KCONV == KCONV_INT8 ? convert(ct.Tile{Float32}, muladd(k, qx, zeros(Int32, (KEYS, M)))) .* sx .* sk :
                                       muladd(convert(ct.Tile{T}, k), q, zeros(Float32, (KEYS, M))) .* sk
        end
        kpos = j * Int32(KEYS) .+ reshape(ct.arange(KEYS; dtype = Int32), (KEYS, 1))
        qk = qk .+ ifelse.(kpos .<= qpos, 0.0f0, -Inf32)
        m_ij = max.(m_i, maximum(qk; dims = 1) .* qk_scale)
        p, alpha = ct.@fpmode flush_to_zero = true begin
            exp2.(qk .* qk_scale .- m_ij), exp2.(m_i .- m_ij)
        end
        l_i = l_i .* alpha .+ sum(p; dims = 1)
        p = convert(ct.Tile{T}, p .* reshape(ct.load(SV; index = col, shape = (KEYS,)), (KEYS, 1)))
        if VCONV == VCONV_HALVES
            v1 = ct.load(V; index = (Int32(1), col), shape = (D ÷ 2, KEYS), latency = 4)
            v2 = ct.load(V; index = (Int32(2), col), shape = (D ÷ 2, KEYS), latency = 4)
            acc1 = muladd(convert(ct.Tile{T}, v1), p, acc1 .* alpha)
            acc2 = muladd(convert(ct.Tile{T}, v2), p, acc2 .* alpha)
        else
            v = VCONV == VCONV_TRANSPOSED ?
                ct.load(V; index = (fld(col - Int32(1), R) + Int32(1), r + Int32(1)), shape = (D, KEYS),
                        order = (2, 1), latency = 4) :
                ct.load(V; index = (Int32(1), col), shape = (D, KEYS), latency = 4)
            acc = muladd(convert(ct.Tile{T}, v), p, acc .* alpha)
        end
        m_i = m_ij
    end
    part = (h - Int32(1)) * Int32(SPLITS) + s
    if VCONV == VCONV_HALVES
        ct.store(Part; index = (Int32(1), Int32(1), part), tile = reshape(acc1, (D ÷ 2, M, 1)))
        ct.store(Part; index = (Int32(2), Int32(1), part), tile = reshape(acc2, (D ÷ 2, M, 1)))
    else
        ct.store(Part; index = (Int32(1), Int32(1), part), tile = reshape(acc, (D, M, 1)))
    end
    ct.store(Part; index = (Int32(D + 1), Int32(1), part), tile = reshape(m_i, (1, M, 1)))
    ct.store(Part; index = (Int32(D + 2), Int32(1), part), tile = reshape(l_i, (1, M, 1)))
    return nothing
end

let D = 128, P = 64, H = 8, GROUP = 4, ctx = 32768
    pages = cld(ctx, P)
    NT, splits = TI.decode_tokens(GROUP, 1), TI.decode_splits(H, 1)
    q = CuArray(BFloat16.(randn(Float32, D * H * GROUP, 1)))
    bt = CuArray([Int32.(randperm(pages)); zeros(Int32, 16)])
    pos = CuArray(Int32[ctx - 1])
    part = CuArray{Float32}(undef, D + 2, GROUP * NT, splits * H)
    sk, sv = CuArray(rand(Float32, P * H * pages) .* 0.02f0), CuArray(rand(Float32, P * H * pages) .* 0.02f0)
    pool(T) = T == Int8 ? CuArray(rand(Int8(-127):Int8(127), D, P * H * pages)) :
                          CuArray(BFloat16.(randn(Float32, D, P * H * pages)))
    bytes(KT, VT) = (sizeof(KT) + sizeof(VT)) * D * P * H * pages + 8 * P * H * pages
    variants = [("BF16 K and V (nothing converted)", BFloat16, BFloat16, P, KCONV_WHOLE, VCONV_WHOLE),
                ("Int8 K, BF16 V", Int8, BFloat16, P, KCONV_WHOLE, VCONV_WHOLE),
                ("BF16 K, Int8 V", BFloat16, Int8, P, KCONV_WHOLE, VCONV_WHOLE),
                ("Int8 K and V, whole pages", Int8, Int8, P, KCONV_WHOLE, VCONV_WHOLE),
                ("V stored transposed", Int8, Int8, P, KCONV_WHOLE, VCONV_TRANSPOSED),
                ("Int8 QKt (Int8 queries)", Int8, Int8, P, KCONV_INT8, VCONV_WHOLE),
                ("32-key tiles", Int8, Int8, 32, KCONV_WHOLE, VCONV_WHOLE),
                ("K and V in head-dim halves", Int8, Int8, P, KCONV_HALVES, VCONV_HALVES),
                ("32-key tiles and halves (TileInfer)", Int8, Int8, 32, KCONV_HALVES, VCONV_HALVES)]
    launch(KT, VT, keys, kconv, vconv) = let get = rotating(() -> (pool(KT), vconv == VCONV_TRANSPOSED ? reshape(pool(VT), P, :) : pool(VT)),
                                                              bytes(KT, VT))
        () -> begin
            k, v = get()
            @cuda backend = cuTile blocks = (H, splits) int8_decode(part, q, k, v, sk, sv, bt, pos, 0.088f0, ct.Constant(D),
                ct.Constant(GROUP), ct.Constant(H), ct.Constant(NT), ct.Constant(P), ct.Constant(splits), ct.Constant(keys),
                ct.Constant(kconv), ct.Constant(vconv))
        end
    end
    probe!("int8_attention", "decode attention at $ctx positions over Int8 keys/values: whole pages spill, halves fit",
           (name => launch(args...) for (name, args...) in variants)...;
           bytes = [bytes(KT, VT) for (_, KT, VT) in variants])
end
