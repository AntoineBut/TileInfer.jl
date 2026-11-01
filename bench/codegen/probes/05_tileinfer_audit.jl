# TileInfer's own kernels against hand-edited copies that differ only in code layout
# (helpers inlined by hand, closures expanded), on Qwen3-0.6B shapes. A difference here means
# the code layout itself costs TileInfer time.

using TileInfer: TileInfer
const TI = TileInfer

# --- Decode attention with `attend` (formerly `attend_page`) and `page_column` inlined ---

function decode_attention_inlined(Part::ct.TileArray{Float32,3}, Q::ct.TileArray{T,2}, K::ct.TileArray{T,2},
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
    qk_scale = qk_scale * TI.LOG2_E
    m_i, l_i, acc = fill(TI.NEG_BIG, (1, M)), zeros(Float32, (1, M)), zeros(Float32, (D, M))
    for j in (s - Int32(1)) * per_split:min(s * per_split, pages) - Int32(1)
        col = (BT[j + Int32(1)] - Int32(1)) * Int32(KV_HEADS) + h
        k = ct.load(K; index = (col, Int32(1)), shape = (PAGE, D), order = (2, 1), latency = 2)
        qk = muladd(k, q, zeros(Float32, (PAGE, M)))
        kpos = j * Int32(PAGE) .+ reshape(ct.arange(PAGE; dtype = Int32), (PAGE, 1))
        qk = qk .+ ifelse.(kpos .<= qpos, 0.0f0, -Inf32)
        m_ij = max.(m_i, maximum(qk; dims = 1) .* qk_scale)
        p, alpha = ct.@fpmode flush_to_zero = true begin
            exp2.(qk .* qk_scale .- m_ij), exp2.(m_i .- m_ij)
        end
        l_i = l_i .* alpha .+ sum(p; dims = 1)
        v = ct.load(V; index = (Int32(1), col), shape = (D, PAGE), latency = 4)
        acc = muladd(v, convert(ct.Tile{T}, p), acc .* alpha)
        m_i = m_ij
    end
    part = (h - Int32(1)) * Int32(SPLITS) + s
    ct.store(Part; index = (Int32(1), Int32(1), part), tile = reshape(acc, (D, M, 1)))
    ct.store(Part; index = (Int32(D + 1), Int32(1), part), tile = reshape(m_i, (1, M, 1)))
    ct.store(Part; index = (Int32(D + 2), Int32(1), part), tile = reshape(l_i, (1, M, 1)))
    return nothing
end

let (D, heads, kv, P, ctx) = (128, 16, 8, 64, 1024)
    pages = cld(ctx, P)
    kp = CuArray(BFloat16.(randn(Float32, D, P, kv, pages)))
    vp = CuArray(BFloat16.(randn(Float32, D, P, kv, pages)))
    q = CuArray(BFloat16.(randn(Float32, D, heads, 1)))
    bt = CuVector{Int32}([1:pages; zeros(Int32, 16 - pages % 16)])
    pos = CuVector{Int32}([ctx - 1])
    part = TI.decode_workspace(q, kp)
    group, splits = heads ÷ kv, size(part, 3) ÷ kv
    launch(k) = () -> @cuda backend = cuTile blocks = (kv, splits) k(
        part, TI.heads_view(q), TI.pool_view(kp), TI.pool_view(vp), bt, pos, Float32(1 / sqrt(D)), ct.Constant(D),
        ct.Constant(group), ct.Constant(kv), ct.Constant(TI.decode_tokens(group, 1)), ct.Constant(P), ct.Constant(splits), ct.Constant(true))
    probe!("audit_decode_attention", "TileInfer's decode attention (1,024 positions) vs a copy with its helpers inlined",
           "TileInfer (attend helper)" => launch(TI.attention_decode_kernel), "helpers inlined" => launch(decode_attention_inlined);
           output = () -> Array(part))
end

# --- q inputs with `norm_rope` and its closures inlined ---

function q_inputs_inlined(Q::ct.TileArray{T,2}, QKV::ct.TileArray{T,2}, CS::ct.TileArray{T,2},
                          W::ct.TileArray{T,1}, Pos::ct.TileArray{Int32,1}, eps::Float32, D::Int,
                          HEADS::Int, ALL_HEADS::Int, NORM::Bool) where {T}
    t, j = ct.bid(1), ct.bid(2)
    HALF = D ÷ 2
    col = j + Int32(ALL_HEADS) * (t - Int32(1))
    pos = Pos[t]
    x1 = convert(ct.Tile{Float32}, ct.load(QKV; index = (Int32(1), col), shape = (HALF, 1)))
    x2 = convert(ct.Tile{Float32}, ct.load(QKV; index = (Int32(2), col), shape = (HALF, 1)))
    x = convert(ct.Tile{Float32}, ct.load(QKV; index = (Int32(1), col), shape = (D, 1)))
    inv_rms = 1.0f0 ./ sqrt.(sum(x .* x; dims = 1) ./ Float32(D) .+ eps)
    w1 = reshape(convert(ct.Tile{Float32}, ct.load(W; index = Int32(1), shape = (HALF,))), (HALF, 1))
    w2 = reshape(convert(ct.Tile{Float32}, ct.load(W; index = Int32(2), shape = (HALF,))), (HALF, 1))
    x1 = convert(ct.Tile{Float32}, convert(ct.Tile{T}, convert(ct.Tile{Float32}, convert(ct.Tile{T}, x1 .* inv_rms)) .* w1))
    x2 = convert(ct.Tile{Float32}, convert(ct.Tile{T}, convert(ct.Tile{Float32}, convert(ct.Tile{T}, x2 .* inv_rms)) .* w2))
    c, s = Int32(2) * pos + Int32(1), Int32(2) * pos + Int32(2)
    c1 = convert(ct.Tile{Float32}, ct.load(CS; index = (Int32(1), c), shape = (HALF, 1)))
    c2 = convert(ct.Tile{Float32}, ct.load(CS; index = (Int32(2), c), shape = (HALF, 1)))
    s1 = convert(ct.Tile{Float32}, ct.load(CS; index = (Int32(1), s), shape = (HALF, 1)))
    s2 = convert(ct.Tile{Float32}, ct.load(CS; index = (Int32(2), s), shape = (HALF, 1)))
    y1 = convert(ct.Tile{Float32}, convert(ct.Tile{T}, x1 .* c1)) .+ convert(ct.Tile{Float32}, convert(ct.Tile{T}, .-x2 .* s1))
    y2 = convert(ct.Tile{Float32}, convert(ct.Tile{T}, x2 .* c2)) .+ convert(ct.Tile{Float32}, convert(ct.Tile{T}, x1 .* s2))
    out = j + Int32(HEADS) * (t - Int32(1))
    ct.store(Q; index = (Int32(1), out), tile = convert(ct.Tile{T}, y1))
    ct.store(Q; index = (Int32(2), out), tile = convert(ct.Tile{T}, y2))
    return nothing
end

let (D, nq, nkv, n, maxlen) = (128, 16, 8, 512, 4096)
    qkv = CuArray(BFloat16.(randn(Float32, D * (nq + 2nkv), n)))
    q = CuArray{BFloat16}(undef, D, nq * n)
    cs = CuArray(BFloat16.(rand(Float32, D, 2maxlen)))
    w = CuArray(BFloat16.(randn(Float32, D)))
    pos = CuVector{Int32}(0:n-1)
    launch(k) = () -> @cuda backend = cuTile blocks = (n, nq) k(q, reshape(qkv, D, :), cs, w, pos, 1f-6,
                                                               ct.Constant(D), ct.Constant(nq), ct.Constant(nq + 2nkv),
                                                               ct.Constant(true))
    probe!("audit_q_inputs", "TileInfer's q inputs (512 tokens, norm + rope) vs a copy with norm_rope and its closures inlined",
           "TileInfer (norm_rope helper)" => launch(TI.q_inputs_kernel), "inlined" => launch(q_inputs_inlined);
           output = () -> Array(q))
end

# --- Attention combine: the (D, 1, SPLITS) partials reshaped to 2-D, or reduced as 3-D ---

function combine_3d(Out::ct.TileArray{T,2}, Part::ct.TileArray{Float32,3}, D::Int, GROUP::Int, SPLITS::Int) where {T}
    c, h = ct.bid(1), ct.bid(2)
    acc = ct.load(Part; index = (Int32(1), c, h), shape = (D, 1, SPLITS))
    m = ct.load(Part; index = (Int32(D + 1), c, h), shape = (1, 1, SPLITS))
    l = ct.load(Part; index = (Int32(D + 2), c, h), shape = (1, 1, SPLITS))
    w = exp2.(m .- maximum(m; dims = 3))
    o = reshape(sum(acc .* w; dims = 3) ./ sum(l .* w; dims = 3), (D, 1))
    head = (h - Int32(1)) * Int32(GROUP) + mod1(c, Int32(GROUP))
    ct.store(Out; index = (head, cld(c, Int32(GROUP))), tile = convert(ct.Tile{T}, o))
    return nothing
end

let (D, heads, kv) = (128, 16, 8)
    group = heads ÷ kv
    splits = TI.decode_splits(kv)
    part = CUDA.randn(Float32, D + 2, group * TI.decode_tokens(group, 1), splits * kv)
    out = CuArray{BFloat16}(undef, D * heads, 1)
    launch(k) = () -> @cuda backend = cuTile blocks = (group, kv) k(out, part, ct.Constant(D), ct.Constant(group),
                                                                   ct.Constant(splits))
    probe!("audit_combine", "attention combine of $splits splits: TileInfer's 2-D reshape vs reducing the 3-D (D, 1, SPLITS) tile",
           "TileInfer (2-D)" => launch(TI.attention_combine_kernel), "3-D, reduce dims = 3" => launch(combine_3d);
           output = () -> Array(out))
end

# --- RMSNorm: 2× the SASS of add_rmsnorm (1960 vs 1032 instructions) for less work ---

# The loads through a closure, as add_rmsnorm_kernel does.
function rmsnorm_closure(Y::ct.TileArray{T,2}, X::ct.TileArray{T,2}, W::ct.TileArray{T,1},
                         eps::Float32, TILE::Int) where {T}
    col = ct.bid(1)
    ntiles = ct.num_tiles(X, 1, (TILE, 1))
    f32(tile) = convert(ct.Tile{Float32}, tile)
    @inline loaded(j) = f32(ct.load(X; index = (j, col), shape = (TILE, 1), padding_mode = ct.PaddingMode.Zero))
    acc = zeros(Float32, (TILE, 1))
    for j in Int32(1):ntiles
        x = loaded(j)
        acc = acc .+ x .* x
    end
    inv_rms = 1.0f0 ./ sqrt.(sum(acc; dims = 1) ./ Float32(size(X, 1)) .+ eps)
    for j in Int32(1):ntiles
        x = loaded(j)
        w = reshape(ct.load(W; index = j, shape = (TILE,), padding_mode = ct.PaddingMode.Zero), (TILE, 1))
        normed = f32(convert(ct.Tile{T}, x .* inv_rms))
        ct.store(Y; index = (j, col), tile = convert(ct.Tile{T}, normed .* f32(w)))
    end
    return nothing
end

# The same with the output written before the weight is loaded, as add_rmsnorm orders it.
function rmsnorm_store_first(Y::ct.TileArray{T,2}, X::ct.TileArray{T,2}, W::ct.TileArray{T,1},
                             eps::Float32, TILE::Int) where {T}
    col = ct.bid(1)
    ntiles = ct.num_tiles(X, 1, (TILE, 1))
    acc = zeros(Float32, (TILE, 1))
    for j in Int32(1):ntiles
        x = convert(ct.Tile{Float32}, ct.load(X; index = (j, col), shape = (TILE, 1), padding_mode = ct.PaddingMode.Zero))
        acc = acc .+ x .* x
    end
    inv_rms = 1.0f0 ./ sqrt.(sum(acc; dims = 1) ./ Float32(size(X, 1)) .+ eps)
    for j in Int32(1):ntiles
        x = convert(ct.Tile{Float32}, ct.load(X; index = (j, col), shape = (TILE, 1), padding_mode = ct.PaddingMode.Zero))
        normed = convert(ct.Tile{Float32}, convert(ct.Tile{T}, x .* inv_rms))
        w = convert(ct.Tile{Float32}, reshape(ct.load(W; index = j, shape = (TILE,), padding_mode = ct.PaddingMode.Zero), (TILE, 1)))
        ct.store(Y; index = (j, col), tile = convert(ct.Tile{T}, normed .* w))
    end
    return nothing
end

for (H, n) in ((1024, 512), (2560, 512))
    X = CuArray(BFloat16.(randn(Float32, H, n)))
    W = CuArray(BFloat16.(randn(Float32, H)))
    Y = CuArray{BFloat16}(undef, H, n)
    tile = min(nextpow(2, H), 1024)
    l(k) = () -> @cuda backend = cuTile blocks = n k(Y, X, W, 1f-6, ct.Constant(tile))
    probe!("audit_rmsnorm_H$H", "TileInfer's rmsnorm (hidden $H, $n tokens) vs rewrites of its loop bodies",
           "TileInfer" => l(TI.rmsnorm_kernel), "loads through a closure" => l(rmsnorm_closure),
           "weight converted after normalizing" => l(rmsnorm_store_first);
           output = () -> Array(Y), bytes = fill(2.0 * 2H * n, 3))
end
