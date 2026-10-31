# Linear layers y = Wᵀx for weights stored (in, out). Two kernels:
#
# - GEMV for a single token (decode). Memory-bound: each block reads TM contiguous weight
#   columns in TK-long tiles and reduces them against x in Float32. TK covers the whole
#   input dimension when it is ≤ 1024, and TM is chosen to give ≥ 256 blocks so the 80 SMs
#   stay busy even for small layers. 70–93 % of the DRAM roofline on Qwen3-0.6B's layers.
# - GEMM for several tokens (prefill). Compute-bound, on tensor cores: W tiles are loaded
#   transposed (order = (2, 1)) so the product is a plain (TM × TK) · (TK × TN) muladd with
#   a Float32 accumulator, and blocks are swizzled for L2 reuse. The tile shape depends on
#   the layer and token count (gemm_config).
#
# Both round once to T at the end, like cuBLAS, then convert to the output's element type
# S (Float32 logits straight from the lm_head, as the reference rounds then converts). A
# bias is added afterwards in T, as in the reference backend.

function gemv_kernel(Y::ct.TileArray{S,2}, W::ct.TileArray{T,2}, X::ct.TileArray{T,2},
                     TK::Int, TM::Int) where {S,T}
    bid = ct.bid(1)
    acc = zeros(Float32, (TK, TM))
    for k in Int32(1):ct.num_tiles(W, 1, (TK, TM))
        w = convert(ct.Tile{Float32}, ct.load(W; index = (k, bid), shape = (TK, TM),
                                              padding_mode = ct.PaddingMode.Zero))
        x = convert(ct.Tile{Float32}, ct.load(X; index = (k, Int32(1)), shape = (TK, 1),
                                              padding_mode = ct.PaddingMode.Zero))
        acc = acc .+ w .* x
    end
    y = reshape(convert(ct.Tile{S}, convert(ct.Tile{T}, sum(acc; dims = 1))), (TM, 1))
    ct.store(Y; index = (bid, Int32(1)), tile = y)
    return nothing
end

# GEMV with a SwiGLU epilogue for the stacked gate_up weight (in, 2I): block b reduces TM
# gate columns and the TM up columns I apart, and writes silu(gate) * up, rounding as
# silu_mul_kernel does after a plain GEMV. Needs I divisible by TM.
function gemv_swiglu_kernel(Y::ct.TileArray{T,2}, W::ct.TileArray{T,2}, X::ct.TileArray{T,2},
                            TK::Int, TM::Int) where {T}
    bid = ct.bid(1)
    up = bid + ct.num_tiles(Y, 1, (TM, 1))                 # tile column of the up weights
    acc_gate, acc_up = zeros(Float32, (TK, TM)), zeros(Float32, (TK, TM))
    f32(tile) = convert(ct.Tile{Float32}, tile)
    for k in Int32(1):ct.num_tiles(W, 1, (TK, TM))
        x = f32(ct.load(X; index = (k, Int32(1)), shape = (TK, 1), padding_mode = ct.PaddingMode.Zero))
        acc_gate = acc_gate .+ f32(ct.load(W; index = (k, bid), shape = (TK, TM), padding_mode = ct.PaddingMode.Zero)) .* x
        acc_up = acc_up .+ f32(ct.load(W; index = (k, up), shape = (TK, TM), padding_mode = ct.PaddingMode.Zero)) .* x
    end
    g = f32(convert(ct.Tile{T}, sum(acc_gate; dims = 1)))
    u = f32(convert(ct.Tile{T}, sum(acc_up; dims = 1)))
    act = f32(convert(ct.Tile{T}, g ./ (1.0f0 .+ exp.(.-g))))
    ct.store(Y; index = (bid, Int32(1)), tile = reshape(convert(ct.Tile{T}, act .* u), (TM, 1)))
    return nothing
end

# GEMV with an xIELU epilogue (Apertus' up projection): the plain GEMV, then xielu_tile on
# the rounded sums, as xielu_kernel computes after it.
function gemv_xielu_kernel(Y::ct.TileArray{T,2}, W::ct.TileArray{T,2}, X::ct.TileArray{T,2},
                           alpha_p::Float32, alpha_n::Float32, beta::Float32, eps::Float32,
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
    u = convert(ct.Tile{Float32}, convert(ct.Tile{T}, sum(acc; dims = 1)))
    y = xielu_tile(T, u, alpha_p, alpha_n, beta, eps)
    ct.store(Y; index = (bid, Int32(1)), tile = reshape(convert(ct.Tile{T}, y), (TM, 1)))
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

function gemm_kernel(Y::ct.TileArray{S,2}, W::ct.TileArray{T,2}, X::ct.TileArray{T,2},
                     TM::Int, TN::Int, TK::Int) where {S,T}
    bm, bn = swizzle_2d(size(W, 2), size(X, 2), TM, TN, 8, ct.bid(1))
    acc = zeros(Float32, TM, TN)
    for k in Int32(1):ct.num_tiles(X, 1, (TK, TN))
        a = ct.load(W; index = (bm, k), shape = (TM, TK), order = (2, 1), padding_mode = ct.PaddingMode.Zero)
        b = ct.load(X; index = (k, bn), shape = (TK, TN), padding_mode = ct.PaddingMode.Zero)
        acc = muladd(a, b, acc)
    end
    ct.store(Y; index = (bm, bn), tile = convert(ct.Tile{S}, convert(ct.Tile{T}, acc)))
    return nothing
end

const GEMV_BLOCKS = 256           # target block count for decode GEMVs

function gemv_config(in::Integer, out::Integer)
    TK = min(nextpow(2, in), 1024)
    TM = clamp(prevpow(2, max(out ÷ GEMV_BLOCKS, 1)), 2, 8)
    return TK, TM
end

# GEMM tiles (TM, TN, TK), small to large. gemm_config picks the one with the lowest cost
# waves × (TM TN)^0.7, where a wave is one block per SM: a larger tile does more work per
# block, but more efficiently, and leaves fewer blocks to fill the GPU, or a partial last
# wave. Columns past n are wasted work, so tiles wider than n are not considered.
# Fitted to a sweep of 30 tile shapes on Qwen3-0.6B and 4B layers at n = 2…1024 on the
# 4080S: within 2 % of the best tile on average (worst 1.18×), and 1.06× cuBLAS.
const GEMM_TILES = ((32, 16, 256), (32, 32, 64), (64, 32, 64), (64, 64, 64), (64, 128, 64), (128, 128, 64))

num_sms() = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)

function gemm_config(out::Integer, n::Integer, sms::Integer = num_sms())
    best, best_cost = first(GEMM_TILES), Inf
    for (TM, TN, TK) in GEMM_TILES
        TN <= max(16, nextpow(2, n)) || continue
        cost = cld(cld(out, TM) * cld(n, TN), sms) * (TM * TN)^0.7
        cost <= best_cost && ((best, best_cost) = ((TM, TN, TK), cost))  # ties: the larger tile
    end
    return best
end

# Two GEMM blocks per SM when the grid has enough of them: the second hides the first's
# latency (4B gate_up at n = 512: 1.11×), but tileiras trims registers and shared memory to
# fit both, which slows grids that leave one block per SM (0.6B o/down at n = 512: 0.86×).
gemm_occupancy(blocks::Integer, sms::Integer = num_sms()) = blocks >= 2sms ? 2 : 1

function linear!(::TileBackend, y::CuMatrix{S}, W::CuMatrix{T}, x::CuMatrix{T}) where {S,T}
    in, out = size(W)
    n = size(x, 2)
    if n == 1
        TK, TM = gemv_config(in, out)
        @cuda backend = cuTile blocks = cld(out, TM) gemv_kernel(y, W, x, ct.Constant(TK), ct.Constant(TM))
    else
        TM, TN, TK = gemm_config(out, n)
        blocks = cld(out, TM) * cld(n, TN)
        @cuda backend = cuTile blocks occupancy = gemm_occupancy(blocks) gemm_kernel(
            y, W, x, ct.Constant(TM), ct.Constant(TN), ct.Constant(TK))
    end
    return y
end

function linear(be::TileBackend, W::CuMatrix{T}, x::CuMatrix{T}, bias = nothing) where {T}
    y = linear!(be, CuMatrix{T}(undef, size(W, 2), size(x, 2)), W, x)
    bias === nothing || (y .+= bias)
    return y
end

# SwiGLU of the gate_up projection: fused into the GEMV for one token, otherwise a GEMM
# into `gate_up` followed by silu_mul.
function linear_swiglu!(be::TileBackend, y::CuMatrix{T}, gate_up::CuMatrix{T}, W::CuMatrix{T},
                        x::CuMatrix{T}) where {T}
    in, I = size(W, 1), size(y, 1)
    TK, TM = gemv_config(in, 2I)          # as the plain GEMV; halving TM was 6 % slower
    if size(x, 2) == 1 && I % TM == 0
        @cuda backend = cuTile blocks = I ÷ TM gemv_swiglu_kernel(y, W, x, ct.Constant(TK), ct.Constant(TM))
        return y
    end
    return silu_mul!(be, y, linear!(be, gate_up, W, x))
end

# xIELU of the up projection: fused into the GEMV for one token, otherwise a GEMM followed
# by xielu! in place.
function linear_xielu!(be::TileBackend, y::CuMatrix{T}, W::CuMatrix{T}, x::CuMatrix{T}, a::XIELU{T}) where {T}
    if size(x, 2) == 1
        TK, TM = gemv_config(size(W)...)
        @cuda backend = cuTile blocks = cld(size(W, 2), TM) gemv_xielu_kernel(
            y, W, x, xielu_args(a)..., ct.Constant(TK), ct.Constant(TM))
        return y
    end
    return xielu!(be, y, linear!(be, y, W, x), a)
end

# INT8 weights (Int8Weight): the same kernels reading Int8 values, which convert exactly to
# Float32 (GEMV) or to T (GEMM, on tensor cores: |q| ≤ 127 fits BFloat16's 8-bit
# significand). The per-column scales multiply the Float32 sums before the one rounding to
# T, so `y = T(s qᵀx)`.

# Float32 sums of TM weight columns against x, from column tile `col`: the GEMV loop.
@inline function gemv_sums(W, X, col, TK, TM)
    acc = zeros(Float32, (TK, TM))
    for k in Int32(1):ct.num_tiles(W, 1, (TK, TM))
        w = convert(ct.Tile{Float32}, ct.load(W; index = (k, col), shape = (TK, TM), padding_mode = ct.PaddingMode.Zero))
        x = convert(ct.Tile{Float32}, ct.load(X; index = (k, Int32(1)), shape = (TK, 1), padding_mode = ct.PaddingMode.Zero))
        acc = acc .+ w .* x
    end
    return sum(acc; dims = 1)                                # (1, TM)
end

# Scaled, rounded GEMV outputs of column tile `col`, as Float32 (1, TM).
@inline function gemv_int8_out(::Type{T}, W, Scale, X, col, TK, TM) where {T}
    s = reshape(ct.load(Scale; index = col, shape = (TM,), padding_mode = ct.PaddingMode.Zero), (1, TM))
    return convert(ct.Tile{Float32}, convert(ct.Tile{T}, gemv_sums(W, X, col, TK, TM) .* s))
end

function gemv_int8_kernel(Y::ct.TileArray{S,2}, W::ct.TileArray{Int8,2}, Scale::ct.TileArray{Float32,1},
                          X::ct.TileArray{T,2}, TK::Int, TM::Int) where {S,T}
    bid = ct.bid(1)
    y = gemv_int8_out(T, W, Scale, X, bid, TK, TM)
    ct.store(Y; index = (bid, Int32(1)), tile = reshape(convert(ct.Tile{S}, y), (TM, 1)))
    return nothing
end

function gemv_int8_swiglu_kernel(Y::ct.TileArray{T,2}, W::ct.TileArray{Int8,2}, Scale::ct.TileArray{Float32,1},
                                 X::ct.TileArray{T,2}, TK::Int, TM::Int) where {T}
    bid = ct.bid(1)
    g = gemv_int8_out(T, W, Scale, X, bid, TK, TM)
    u = gemv_int8_out(T, W, Scale, X, bid + ct.num_tiles(Y, 1, (TM, 1)), TK, TM)
    act = convert(ct.Tile{Float32}, convert(ct.Tile{T}, g ./ (1.0f0 .+ exp.(.-g))))
    ct.store(Y; index = (bid, Int32(1)), tile = reshape(convert(ct.Tile{T}, act .* u), (TM, 1)))
    return nothing
end

function gemv_int8_xielu_kernel(Y::ct.TileArray{T,2}, W::ct.TileArray{Int8,2}, Scale::ct.TileArray{Float32,1},
                                X::ct.TileArray{T,2}, alpha_p::Float32, alpha_n::Float32, beta::Float32,
                                eps::Float32, TK::Int, TM::Int) where {T}
    bid = ct.bid(1)
    y = xielu_tile(T, gemv_int8_out(T, W, Scale, X, bid, TK, TM), alpha_p, alpha_n, beta, eps)
    ct.store(Y; index = (bid, Int32(1)), tile = reshape(convert(ct.Tile{T}, y), (TM, 1)))
    return nothing
end

function gemm_int8_kernel(Y::ct.TileArray{S,2}, W::ct.TileArray{Int8,2}, Scale::ct.TileArray{Float32,1},
                          X::ct.TileArray{T,2}, TM::Int, TN::Int, TK::Int) where {S,T}
    bm, bn = swizzle_2d(size(W, 2), size(X, 2), TM, TN, 8, ct.bid(1))
    acc = zeros(Float32, TM, TN)
    for k in Int32(1):ct.num_tiles(X, 1, (TK, TN))
        a = ct.load(W; index = (bm, k), shape = (TM, TK), order = (2, 1), padding_mode = ct.PaddingMode.Zero)
        b = ct.load(X; index = (k, bn), shape = (TK, TN), padding_mode = ct.PaddingMode.Zero)
        acc = muladd(convert(ct.Tile{T}, a), b, acc)
    end
    s = reshape(ct.load(Scale; index = bm, shape = (TM,), padding_mode = ct.PaddingMode.Zero), (TM, 1))
    ct.store(Y; index = (bm, bn), tile = convert(ct.Tile{S}, convert(ct.Tile{T}, acc .* s)))
    return nothing
end

# Int8 GEMV tiles of half the BF16 width: TK × TM ≈ 4096 elements was fastest on Apertus
# 8B's layers (TK = 1024, TM = 4: 600–670 GB/s; TM = 8 was 7–13 % slower).
function gemv_int8_config(in::Integer, out::Integer)
    TK, TM = gemv_config(in, out)
    return TK, max(TM ÷ 2, 2)
end

function linear!(::TileBackend, y::CuMatrix{S}, W::Int8Weight, x::CuMatrix{T}) where {S,T}
    in, out = size(W)
    n = size(x, 2)
    if n == 1
        TK, TM = gemv_int8_config(in, out)
        @cuda backend = cuTile blocks = cld(out, TM) gemv_int8_kernel(y, W.q, W.scale, x, ct.Constant(TK), ct.Constant(TM))
    else
        TM, TN, TK = gemm_config(out, n)
        blocks = cld(out, TM) * cld(n, TN)
        @cuda backend = cuTile blocks occupancy = gemm_occupancy(blocks) gemm_int8_kernel(
            y, W.q, W.scale, x, ct.Constant(TM), ct.Constant(TN), ct.Constant(TK))
    end
    return y
end

function linear_swiglu!(be::TileBackend, y::CuMatrix{T}, gate_up::CuMatrix{T}, W::Int8Weight, x::CuMatrix{T}) where {T}
    in, I = size(W, 1), size(y, 1)
    TK, TM = gemv_int8_config(in, 2I)
    if size(x, 2) == 1 && I % TM == 0
        @cuda backend = cuTile blocks = I ÷ TM gemv_int8_swiglu_kernel(y, W.q, W.scale, x, ct.Constant(TK), ct.Constant(TM))
        return y
    end
    return silu_mul!(be, y, linear!(be, gate_up, W, x))
end

function linear_xielu!(be::TileBackend, y::CuMatrix{T}, W::Int8Weight, x::CuMatrix{T}, a::XIELU{T}) where {T}
    if size(x, 2) == 1
        TK, TM = gemv_int8_config(size(W)...)
        @cuda backend = cuTile blocks = cld(size(W, 2), TM) gemv_int8_xielu_kernel(
            y, W.q, W.scale, x, xielu_args(a)..., ct.Constant(TK), ct.Constant(TM))
        return y
    end
    return xielu!(be, y, linear!(be, y, W, x), a)
end
