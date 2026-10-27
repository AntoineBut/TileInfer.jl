# Linear layers y = Wᵀx for weights stored (in, out). Two kernels:
#
# - GEMV for a single token (decode). Memory-bound: each block reads TM contiguous weight
#   columns in TK-long tiles and reduces them against x in Float32. TK covers the whole
#   input dimension when it is ≤ 1024, and TM is chosen to give ≥ 256 blocks so the 80 SMs
#   stay busy even for small layers. 70–93 % of the DRAM roofline on Qwen3-0.6B's layers.
# - GEMM for several tokens (prefill). Compute-bound, on tensor cores: W tiles are loaded
#   transposed (order = (2, 1)) so the product is a plain (TM × TK) · (TK × TN) muladd with
#   a Float32 accumulator, and blocks are swizzled for L2 reuse. 0.8–0.9× cuBLAS.
#
# Both round once to T at the end, like cuBLAS. A bias is added afterwards in T, as in
# the reference backend.

function gemv_kernel(Y::ct.TileArray{T,2}, W::ct.TileArray{T,2}, X::ct.TileArray{T,2},
                     TK::Int, TM::Int) where {T}
    bid = ct.bid(1)
    acc = zeros(Float32, (TK, TM))
    for k in Int32(1):ct.num_tiles(W, 1, (TK, TM))
        w = convert(ct.Tile{Float32}, ct.load(W; index = (k, bid), shape = (TK, TM),
                                              padding_mode = ct.PaddingMode.Zero))
        x = convert(ct.Tile{Float32}, ct.load(X; index = (k, Int32(1)), shape = (TK, 1),
                                              padding_mode = ct.PaddingMode.Zero))
        acc = acc .+ w .* x
    end
    y = reshape(convert(ct.Tile{T}, sum(acc; dims = 1)), (TM, 1))
    ct.store(Y; index = (bid, Int32(1)), tile = y)
    return nothing
end

# 1-based (block row, block column) of block `bid`, walking groups of GROUP block rows
# column by column so that consecutive blocks share W and X tiles in L2.
function swizzle_2d(M, N, TM, TN, GROUP, bid)
    num_m, num_n = cld(M, Int32(TM)), cld(N, Int32(TN))
    in_group = Int32(GROUP) * num_n
    b = bid - Int32(1)
    first_m = fld(b, in_group) * Int32(GROUP)
    group_m = min(num_m - first_m, Int32(GROUP))
    return first_m + rem(b, group_m) + Int32(1), fld(rem(b, in_group), group_m) + Int32(1)
end

function gemm_kernel(Y::ct.TileArray{T,2}, W::ct.TileArray{T,2}, X::ct.TileArray{T,2},
                     TM::Int, TN::Int, TK::Int) where {T}
    bm, bn = swizzle_2d(size(W, 2), size(X, 2), TM, TN, 8, ct.bid(1))
    acc = zeros(Float32, TM, TN)
    for k in Int32(1):ct.num_tiles(X, 1, (TK, TN))
        a = ct.load(W; index = (bm, k), shape = (TM, TK), order = (2, 1), padding_mode = ct.PaddingMode.Zero)
        b = ct.load(X; index = (k, bn), shape = (TK, TN), padding_mode = ct.PaddingMode.Zero)
        acc = muladd(a, b, acc)
    end
    ct.store(Y; index = (bm, bn), tile = convert(ct.Tile{T}, acc))
    return nothing
end

const GEMV_BLOCKS = 256           # target block count for decode GEMVs
const GEMM_BLOCKS = 120           # use 128 × 128 output tiles from this many blocks (1.5 × SMs)

function gemv_config(in::Integer, out::Integer)
    TK = min(nextpow(2, in), 1024)
    TM = clamp(prevpow(2, max(out ÷ GEMV_BLOCKS, 1)), 2, 8)
    return TK, TM
end

function gemm_config(out::Integer, n::Integer)
    TN = clamp(nextpow(2, n), 16, 128)
    big = cld(out, 128) * cld(n, TN) >= GEMM_BLOCKS
    return (big ? 128 : 64), TN, 64
end

function linear!(::TileBackend, y::CuMatrix{T}, W::CuMatrix{T}, x::CuMatrix{T}) where {T}
    in, out = size(W)
    n = size(x, 2)
    if n == 1
        TK, TM = gemv_config(in, out)
        @cuda backend = cuTile blocks = cld(out, TM) gemv_kernel(y, W, x, ct.Constant(TK), ct.Constant(TM))
    else
        TM, TN, TK = gemm_config(out, n)
        @cuda backend = cuTile blocks = cld(out, TM) * cld(n, TN) gemm_kernel(
            y, W, x, ct.Constant(TM), ct.Constant(TN), ct.Constant(TK))
    end
    return y
end

function linear(be::TileBackend, W::CuMatrix{T}, x::CuMatrix{T}, bias = nothing) where {T}
    y = linear!(be, CuMatrix{T}(undef, size(W, 2), size(x, 2)), W, x)
    bias === nothing || (y .+= bias)
    return y
end
