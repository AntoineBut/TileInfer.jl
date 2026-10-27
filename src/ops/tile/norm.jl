# RMSNorm: one block per column (token, or head of a token for q/k norm). Two passes over
# the column: the sum of squares, then normalize and scale. The column is walked in tiles,
# so hidden sizes that are not powers of two work; the second pass re-reads it from L1/L2.

function rmsnorm_kernel(Y::ct.TileArray{T,2}, X::ct.TileArray{T,2}, W::ct.TileArray{T,1},
                        eps::Float32, TILE::Int) where {T}
    col = ct.bid(1)
    H = size(X, 1)
    ntiles = ct.num_tiles(X, 1, (TILE, 1))
    acc = zeros(Float32, (TILE, 1))
    for j in Int32(1):ntiles
        x = convert(ct.Tile{Float32}, ct.load(X; index = (j, col), shape = (TILE, 1),
                                              padding_mode = ct.PaddingMode.Zero))
        acc = acc .+ x .* x
    end
    inv_rms = 1.0f0 ./ sqrt.(sum(acc; dims = 1) ./ Float32(H) .+ eps)     # (1, 1)
    for j in Int32(1):ntiles
        x = convert(ct.Tile{Float32}, ct.load(X; index = (j, col), shape = (TILE, 1),
                                              padding_mode = ct.PaddingMode.Zero))
        w = reshape(ct.load(W; index = j, shape = (TILE,), padding_mode = ct.PaddingMode.Zero), (TILE, 1))
        # Round the normalized value to T before scaling, as transformers does.
        normed = convert(ct.Tile{Float32}, convert(ct.Tile{T}, x .* inv_rms))
        ct.store(Y; index = (j, col), tile = convert(ct.Tile{T}, normed .* convert(ct.Tile{Float32}, w)))
    end
    return nothing
end

# Columns longer than this are processed in several tiles.
const NORM_MAX_TILE = 1024

function rmsnorm!(::TileBackend, y::CuArray{T}, x::CuArray{T}, w::CuVector{T}, eps::Real) where {T}
    H = size(x, 1)
    cols = length(x) ÷ H
    tile = min(nextpow(2, H), NORM_MAX_TILE)
    @cuda backend = cuTile blocks = cols rmsnorm_kernel(reshape(y, H, cols), reshape(x, H, cols), w,
                                                       Float32(eps), ct.Constant(tile))
    return y
end

rmsnorm(be::TileBackend, x::CuArray{T}, w::CuVector{T}, eps::Real) where {T} =
    rmsnorm!(be, similar(x), x, w, eps)

# Residual add + RMSNorm, one block per column: pass 1 sums the squares of x + delta, pass
# 2 recomputes it, stores it into x and the normalized value into H. Each tile is read in
# pass 2 before it is overwritten, and pass 1 never sees stored values.
function add_rmsnorm_kernel(X::ct.TileArray{T,2}, H::ct.TileArray{T,2}, Delta::ct.TileArray{T,2},
                            W::ct.TileArray{T,1}, eps::Float32, TILE::Int) where {T}
    col = ct.bid(1)
    ntiles = ct.num_tiles(X, 1, (TILE, 1))
    f32(tile) = convert(ct.Tile{Float32}, tile)
    @inline added(j) = f32(convert(ct.Tile{T},
        f32(ct.load(X; index = (j, col), shape = (TILE, 1), padding_mode = ct.PaddingMode.Zero)) .+
        f32(ct.load(Delta; index = (j, col), shape = (TILE, 1), padding_mode = ct.PaddingMode.Zero))))
    acc = zeros(Float32, (TILE, 1))
    for j in Int32(1):ntiles
        x = added(j)
        acc = acc .+ x .* x
    end
    inv_rms = 1.0f0 ./ sqrt.(sum(acc; dims = 1) ./ Float32(size(X, 1)) .+ eps)
    for j in Int32(1):ntiles
        x = added(j)
        ct.store(X; index = (j, col), tile = convert(ct.Tile{T}, x))
        w = reshape(ct.load(W; index = j, shape = (TILE,), padding_mode = ct.PaddingMode.Zero), (TILE, 1))
        normed = f32(convert(ct.Tile{T}, x .* inv_rms))
        ct.store(H; index = (j, col), tile = convert(ct.Tile{T}, normed .* f32(w)))
    end
    return nothing
end

function add_rmsnorm!(::TileBackend, x::CuMatrix{T}, h::CuMatrix{T}, delta::CuMatrix{T}, w::CuVector{T},
                      eps::Real) where {T}
    @cuda backend = cuTile blocks = size(x, 2) add_rmsnorm_kernel(
        x, h, delta, w, Float32(eps), ct.Constant(min(nextpow(2, size(x, 1)), NORM_MAX_TILE)))
    return h
end
