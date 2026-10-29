# Codegen knobs never swept on TileInfer's heaviest kernels: the `latency` hint of the
# weight loads, and the `occupancy` hint. Real shapes, weights rotated through DRAM
# (`rotating`). Variants of a kernel are copies differing only in the knob.

for lat in (1, 4, 10)
    @eval function $(Symbol(:gemv_lat, lat))(Y::ct.TileArray{S,2}, W::ct.TileArray{T,2}, X::ct.TileArray{T,2},
                                              TK::Int, TM::Int) where {S,T}
        bid = ct.bid(1)
        acc = zeros(Float32, (TK, TM))
        for k in Int32(1):ct.num_tiles(W, 1, (TK, TM))
            w = convert(ct.Tile{Float32}, ct.load(W; index = (k, bid), shape = (TK, TM),
                                                  padding_mode = ct.PaddingMode.Zero, latency = $lat))
            x = convert(ct.Tile{Float32}, ct.load(X; index = (k, Int32(1)), shape = (TK, 1),
                                                  padding_mode = ct.PaddingMode.Zero))
            acc = acc .+ w .* x
        end
        y = reshape(convert(ct.Tile{S}, convert(ct.Tile{T}, sum(acc; dims = 1))), (TM, 1))
        ct.store(Y; index = (bid, Int32(1)), tile = y)
        return nothing
    end
end

for (label, din, out) in (("0.6B qkv", 1024, 4096), ("0.6B down", 3072, 1024), ("4B gate_up", 2560, 19456),
                         ("4B down", 9728, 2560))
    W = rotating(() -> CuArray(BFloat16.(randn(Float32, din, out) ./ 32)), 2din * out)
    X = CuArray(BFloat16.(randn(Float32, din, 1)))
    Y = CuArray{BFloat16}(undef, out, 1)
    TK, TM = TI.gemv_config(din, out)
    l(k) = () -> @cuda backend = cuTile blocks = cld(out, TM) k(Y, W(), X, ct.Constant(TK), ct.Constant(TM))
    lo(occ) = () -> @cuda backend = cuTile blocks = cld(out, TM) occupancy = occ TI.gemv_kernel(Y, W(), X, ct.Constant(TK), ct.Constant(TM))
    probe!("knobs_gemv_$(replace(label, ' ' => '_'))", "decode GEMV $label ($din → $out): latency hint on W, occupancy",
           "TileInfer (no hint)" => l(TI.gemv_kernel), "latency = 1" => l(gemv_lat1), "latency = 4" => l(gemv_lat4),
           "latency = 10" => l(gemv_lat10), "occupancy = 2" => lo(2), "occupancy = 4" => lo(4);
           output = () -> Array(Y), bytes = fill(2.0 * din * out, 6))
end

for lat in (1, 4, 10)
    @eval function $(Symbol(:gemm_lat, lat))(Y::ct.TileArray{S,2}, W::ct.TileArray{T,2}, X::ct.TileArray{T,2},
                                              TM::Int, TN::Int, TK::Int) where {S,T}
        bm, bn = TI.swizzle_2d(size(W, 2), size(X, 2), TM, TN, 8, ct.bid(1))
        acc = zeros(Float32, TM, TN)
        for k in Int32(1):ct.num_tiles(X, 1, (TK, TN))
            a = ct.load(W; index = (bm, k), shape = (TM, TK), order = (2, 1), padding_mode = ct.PaddingMode.Zero,
                        latency = $lat)
            b = ct.load(X; index = (k, bn), shape = (TK, TN), padding_mode = ct.PaddingMode.Zero, latency = $lat)
            acc = muladd(a, b, acc)
        end
        ct.store(Y; index = (bm, bn), tile = convert(ct.Tile{S}, convert(ct.Tile{T}, acc)))
        return nothing
    end
end

for (label, din, out, n) in (("0.6B gate_up", 1024, 6144, 512), ("4B gate_up", 2560, 19456, 512), ("4B qkv", 2560, 6144, 64))
    W = rotating(() -> CuArray(BFloat16.(randn(Float32, din, out) ./ 32)), 2din * out)
    X = CuArray(BFloat16.(randn(Float32, din, n)))
    Y = CuArray{BFloat16}(undef, out, n)
    TM, TN, TK = TI.gemm_config(out, n)
    blocks = cld(out, TM) * cld(n, TN)
    l(k) = () -> @cuda backend = cuTile blocks = blocks k(Y, W(), X, ct.Constant(TM), ct.Constant(TN), ct.Constant(TK))
    lo(occ) = () -> @cuda backend = cuTile blocks = blocks occupancy = occ TI.gemm_kernel(Y, W(), X, ct.Constant(TM), ct.Constant(TN), ct.Constant(TK))
    flops = 2.0 * din * out * n
    probe!("knobs_gemm_$(replace(label, ' ' => '_'))_n$n",
           "prefill GEMM $label ($din → $out, n = $n, tile $((TM, TN, TK)), $(round(flops / 1e9; digits = 1)) GFLOP): latency hints, occupancy",
           "TileInfer (no hint)" => l(TI.gemm_kernel), "latency = 1" => l(gemm_lat1), "latency = 4" => l(gemm_lat4),
           "latency = 10" => l(gemm_lat10), "occupancy = 1" => lo(1), "occupancy = 2" => lo(2);
           output = () -> Array(Y), rtol = 1e-2)
end

# Verify-sized steps (spec decoding: 2–8 tokens) on the GEMM: occupancy 1 vs 2 where
# gemm_occupancy picks 2 (≥ 2 blocks per SM).
for (label, din, out, n) in (("0.6B gate_up", 1024, 6144, 4), ("4B gate_up", 2560, 19456, 8), ("4B qkv", 2560, 6144, 2))
    W = rotating(() -> CuArray(BFloat16.(randn(Float32, din, out) ./ 32)), 2din * out)
    X = CuArray(BFloat16.(randn(Float32, din, n)))
    Y = CuArray{BFloat16}(undef, out, n)
    TM, TN, TK = TI.gemm_config(out, n)
    blocks = cld(out, TM) * cld(n, TN)
    lo(occ) = () -> @cuda backend = cuTile blocks = blocks occupancy = occ TI.gemm_kernel(Y, W(), X, ct.Constant(TM), ct.Constant(TN), ct.Constant(TK))
    probe!("knobs_gemm_$(replace(label, ' ' => '_'))_n$n",
           "verify-step GEMM $label ($din → $out, n = $n, tile $((TM, TN, TK)), $blocks blocks; gemm_occupancy picks $(TI.gemm_occupancy(blocks)))",
           "occupancy = 1" => lo(1), "occupancy = 2" => lo(2);
           output = () -> Array(Y), bytes = fill(2.0 * din * out, 2))
end

# Prefill attention (512 tokens from position 0) at occupancy 1–4; the kernel asks for 2.
for (label, heads, kv) in (("0.6B", 16, 8), ("4B", 32, 8))
    D, P, n = 128, 64, 512
    pages = cld(n, P)
    kp, vp = CuArray(BFloat16.(randn(Float32, D, P, kv, pages))), CuArray(BFloat16.(randn(Float32, D, P, kv, pages)))
    q, out = CuArray(BFloat16.(randn(Float32, D, heads, n))), CuArray{BFloat16}(undef, D, heads, n)
    bt, pos = CuVector{Int32}([1:pages; zeros(Int32, 16 - pages)]), CuVector{Int32}([0])
    args = (TI.heads_view(out), TI.heads_view(q), TI.pool_view(kp), TI.pool_view(vp), bt, pos, Float32(1 / sqrt(D)),
            ct.Constant(D), ct.Constant(heads ÷ kv), ct.Constant(kv), ct.Constant(TI.PREFILL_TM), ct.Constant(P))
    grid = (cld(n, TI.PREFILL_TM), heads)
    probe!("knobs_attention_prefill_$label", "prefill attention of $label, $n tokens: occupancy (the kernel asks for 2)",
           "occupancy = 1" => () -> @cuda(backend = cuTile, blocks = grid, occupancy = 1, TI.attention_prefill_kernel(args...)),
           "occupancy = 2" => () -> @cuda(backend = cuTile, blocks = grid, occupancy = 2, TI.attention_prefill_kernel(args...)),
           "occupancy = 3" => () -> @cuda(backend = cuTile, blocks = grid, occupancy = 3, TI.attention_prefill_kernel(args...)),
           "occupancy = 4" => () -> @cuda(backend = cuTile, blocks = grid, occupancy = 4, TI.attention_prefill_kernel(args...));
           output = () -> Array(out))
end
