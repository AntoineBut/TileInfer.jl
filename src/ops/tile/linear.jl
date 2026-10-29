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

function linear!(::TileBackend, y::CuMatrix{S}, W::CuMatrix{T}, x::CuMatrix{T}) where {S,T}
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
