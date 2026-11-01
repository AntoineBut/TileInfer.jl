# Tile shapes and axis order for the same computation.

# --- Reducing a 3-D tile over its last axis vs the same data as a 2-D tile ---
# Sum over S of a (D, H, S) block: Y[:, :, b] = sum(A[:, :, :, b]; dims = 3).

function sum3_last(Y::ct.TileArray{Float32,2}, A::ct.TileArray{Float32,4}, D::Int, H::Int, S::Int)
    b = ct.bid(1)
    t = reshape(ct.load(A; index = (Int32(1), Int32(1), Int32(1), b), shape = (D, H, S, 1)), (D, H, S))
    ct.store(Y; index = (Int32(1), b), tile = reshape(sum(t; dims = 3), (D * H, 1)))
    return nothing
end

function sum3_first(Y::ct.TileArray{Float32,2}, A::ct.TileArray{Float32,4}, D::Int, H::Int, S::Int)
    b = ct.bid(1)
    # The same block loaded with the reduced axis first.
    t = reshape(ct.load(A; index = (Int32(1), Int32(1), Int32(1), b), shape = (S, D, H, 1), order = (3, 1, 2, 4)),
                (S, D, H))
    ct.store(Y; index = (Int32(1), b), tile = reshape(sum(t; dims = 1), (D * H, 1)))
    return nothing
end

function sum2_last(Y::ct.TileArray{Float32,2}, A::ct.TileArray{Float32,3}, D::Int, H::Int, S::Int)
    b = ct.bid(1)
    t = reshape(ct.load(A; index = (Int32(1), Int32(1), b), shape = (D * H, S, 1)), (D * H, S))
    ct.store(Y; index = (Int32(1), b), tile = sum(t; dims = 2))
    return nothing
end

let (D, H, S, B) = (64, 8, 16, 160)
    A = CUDA.randn(Float32, D, H, S, B)
    A3 = reshape(A, D * H, S, B)
    Y = CUDA.zeros(Float32, D * H, B)
    l4(k) = () -> @cuda backend = cuTile blocks = B k(Y, A, ct.Constant(D), ct.Constant(H), ct.Constant(S))
    probe!("reduce_axis", "sum over S of (D=64, H=8, S=16) blocks: 3-D tile over its last axis, loaded with S first, or as a 2-D (D H, S) tile",
           "3-D, reduce dims = 3" => l4(sum3_last), "3-D loaded S first, reduce dims = 1" => l4(sum3_first),
           "2-D (512, 16), reduce dims = 2" => () -> @cuda backend = cuTile blocks = B sum2_last(Y, A3, ct.Constant(D), ct.Constant(H), ct.Constant(S));
           output = () -> Array(Y), rtol = 1e-5)
end

# --- A size-1 axis kept or dropped (RMSNorm of one token per block) ---

function norm_keep(Y::ct.TileArray{T,2}, X::ct.TileArray{T,2}, W::ct.TileArray{T,2}, H::Int) where {T}
    b = ct.bid(1)
    x = convert(ct.Tile{Float32}, ct.load(X; index = (Int32(1), b), shape = (H, 1)))
    w = convert(ct.Tile{Float32}, ct.load(W; index = (Int32(1), Int32(1)), shape = (H, 1)))
    r = sqrt.(sum(x .* x; dims = 1) ./ Float32(H) .+ 1f-6)
    ct.store(Y; index = (Int32(1), b), tile = convert(ct.Tile{T}, x ./ r .* w))
    return nothing
end

function norm_drop(Y::ct.TileArray{T,2}, X::ct.TileArray{T,2}, W::ct.TileArray{T,2}, H::Int) where {T}
    b = ct.bid(1)
    x = reshape(convert(ct.Tile{Float32}, ct.load(X; index = (Int32(1), b), shape = (H, 1))), (H,))
    w = reshape(convert(ct.Tile{Float32}, ct.load(W; index = (Int32(1), Int32(1)), shape = (H, 1))), (H,))
    r = sqrt.(sum(x .* x; dims = 1) ./ Float32(H) .+ 1f-6)
    ct.store(Y; index = (Int32(1), b), tile = reshape(convert(ct.Tile{T}, x ./ r .* w), (H, 1)))
    return nothing
end

function norm_dropdims(Y::ct.TileArray{T,2}, X::ct.TileArray{T,2}, W::ct.TileArray{T,2}, H::Int) where {T}
    b = ct.bid(1)
    x = dropdims(convert(ct.Tile{Float32}, ct.load(X; index = (Int32(1), b), shape = (H, 1))); dims = 2)
    w = dropdims(convert(ct.Tile{Float32}, ct.load(W; index = (Int32(1), Int32(1)), shape = (H, 1))); dims = 2)
    r = sqrt.(sum(x .* x; dims = 1) ./ Float32(H) .+ 1f-6)
    ct.store(Y; index = (Int32(1), b), tile = reshape(convert(ct.Tile{T}, x ./ r .* w), (H, 1)))
    return nothing
end

let (H, n) = (1024, 512)
    X = CuArray(BFloat16.(randn(Float32, H, n)))
    W = CuArray(BFloat16.(randn(Float32, H, 1)))
    Y = CuArray{BFloat16}(undef, H, n)
    l(k) = () -> @cuda backend = cuTile blocks = n k(Y, X, W, ct.Constant(H))
    probe!("size1_axis", "RMSNorm of one 1024-element token per block, as (H, 1) tiles or with the size-1 axis dropped",
           "(H, 1) throughout" => l(norm_keep), "reshape to (H,)" => l(norm_drop), "dropdims to (H,)" => l(norm_dropdims);
           output = () -> Array(Y))
end

# --- Narrow matmul operands ---
# Scores for 16 pages of 64 keys (head_dim 128) against a query tile of R columns,
# accumulated in Float32 as decode attention does. In Julia's column-major terms the
# product is K (64 × 128) · q (128 × R), which cuTile.jl emits as a row-major
# `mmaf` of (R × 128) · (128 × 64): R is the M of the Tile IR op. The transposed product
# qᵀ (R × 128) · Kᵀ (128 × 64) puts R on the N side instead. Only 2 columns are stored.

function scores(Y::ct.TileArray{Float32,2}, K::ct.TileArray{T,2}, Q::ct.TileArray{T,2}, D::Int, P::Int, R::Int) where {T}
    b = ct.bid(1)
    q = ct.load(Q; index = (Int32(1), Int32(1)), shape = (D, R), padding_mode = ct.PaddingMode.Zero)
    acc = zeros(Float32, P, R)
    for j in Int32(1):Int32(16)
        k = ct.load(K; index = ((b - Int32(1)) * Int32(16) + j, Int32(1)), shape = (P, D), order = (2, 1))
        acc = muladd(k, q, acc)
    end
    ct.store(Y; index = (b, Int32(1)), tile = ct.extract(acc, (1, 1), (P, 2)))
    return nothing
end

function scores_t(Y::ct.TileArray{Float32,2}, K::ct.TileArray{T,2}, Q::ct.TileArray{T,2}, D::Int, P::Int, R::Int) where {T}
    b = ct.bid(1)
    q = ct.load(Q; index = (Int32(1), Int32(1)), shape = (R, D), order = (2, 1), padding_mode = ct.PaddingMode.Zero)
    acc = zeros(Float32, R, P)
    for j in Int32(1):Int32(16)
        k = ct.load(K; index = (Int32(1), (b - Int32(1)) * Int32(16) + j), shape = (D, P))
        acc = muladd(q, k, acc)
    end
    ct.store(Y; index = (b, Int32(1)), tile = permutedims(ct.extract(acc, (1, 1), (2, P))))
    return nothing
end

let (D, P, B) = (128, 64, 160)
    K = CuArray(BFloat16.(randn(Float32, D, P * 16 * B)))
    Q = CuArray(BFloat16.(randn(Float32, D, 2)))
    Y = CUDA.zeros(Float32, P * B, 2)
    l(k, R) = () -> @cuda backend = cuTile blocks = B k(Y, K, Q, ct.Constant(D), ct.Constant(P), ct.Constant(R))
    probe!("narrow_matmul", "K q over 16 pages of 64 keys with a query tile of R columns (2 used): R is the Tile IR mmaf's M",
           ("R = $R" => l(scores, R) for R in (32, 16, 8, 4, 2))...; output = () -> Array(Y), rtol = 1e-5)
    probe!("narrow_matmul_t", "the same as qᵀ Kᵀ: R is the Tile IR mmaf's N",
           ("R = $R" => l(scores_t, R) for R in (32, 16, 8, 4, 2))...; output = () -> Array(Y), rtol = 1e-5)
end

# --- A Float64 literal in Float32 code ---

function scale_f32(Y::ct.TileArray{Float32,1}, X::ct.TileArray{Float32,1}, N::Int)
    b = ct.bid(1)
    ct.store(Y; index = b, tile = exp.(ct.load(X; index = b, shape = (N,)) .* 0.7f0))
    return nothing
end

function scale_f64(Y::ct.TileArray{Float32,1}, X::ct.TileArray{Float32,1}, N::Int)
    b = ct.bid(1)
    ct.store(Y; index = b, tile = exp.(ct.load(X; index = b, shape = (N,)) .* 0.7))
    return nothing
end

let N = 1024
    X = CUDA.randn(Float32, 1 << 22)
    Y = CUDA.zeros(Float32, 1 << 22)
    l(k) = () -> @cuda backend = cuTile blocks = length(X) ÷ N k(Y, X, ct.Constant(N))
    probe!("f64_literal", "exp.(x .* 0.7f0) vs exp.(x .* 0.7) on a Float32 tile (Julia promotes the second to Float64)",
           "0.7f0" => l(scale_f32), "0.7" => l(scale_f64); output = () -> Array(Y), rtol = 1e-6)
end
