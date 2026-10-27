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
