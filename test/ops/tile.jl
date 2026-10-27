# TileBackend kernels against the reference backend on random inputs, including shapes
# that are not powers of two (heads = 24, hidden = 2560 or 100, odd token counts). The
# element-wise kernels round at the same points as the reference and must be
# bit-identical; reductions may differ by accumulation order.

@testset "tile kernels" begin
    ref, tile = RefBackend(), TileBackend()
    host(x) = Float32.(Array(x))
    # Largest difference in units of T's precision at each element's size. The unit is
    # floored at the output's RMS: an entry near zero comes from cancellation, where a
    # different Float32 accumulation order legitimately moves it by many of its own ulps.
    function ulps(a, b, T)
        a, b = host(a), host(b)
        floor_ulp = eps(T(sqrt(sum(abs2, a) / length(a))))
        return maximum(abs.(a .- b) ./ max.(eps.(T.(abs.(a))), floor_ulp); init = 0f0)
    end

    @testset "$T" for T in (Float32, BFloat16)
        rnd(dims...; scale = 1) = CuArray(T.(randn(Float32, dims...) .* scale))

        @testset "rmsnorm $dims" for dims in ((1024, 1), (1024, 37), (128, 16, 3), (2560, 4), (100, 3))
            x, w = rnd(dims...), rnd(dims[1])
            y_ref, y = TileInfer.rmsnorm(ref, x, w, 1e-6), TileInfer.rmsnorm(tile, x, w, 1e-6)
            @test size(y) == size(x)
            @test ulps(y_ref, y, T) <= (T == BFloat16 ? 1 : 8)  # sum of squares in another order
        end

        @testset "rope $dims" for dims in ((128, 16, 1), (128, 8, 7), (64, 24, 3))
            D, _, n = dims
            x, θ = rnd(dims...), rand(Float32, D ÷ 2, n) .* 100
            c, s = CuArray(T.(vcat(cos.(θ), cos.(θ)))), CuArray(T.(vcat(sin.(θ), sin.(θ))))
            @test host(TileInfer.rope(tile, x, c, s)) == host(TileInfer.rope(ref, x, c, s))
        end

        @testset "silu_mul $dims" for dims in ((6144, 1), (6144, 5), (19456, 2), (200, 3))
            gu = rnd(dims...; scale = 3)
            @test host(TileInfer.silu_mul(tile, gu)) == host(TileInfer.silu_mul(ref, gu))
        end

        @testset "store_kv" begin
            D, P, H, pages, n = 128, 4, 8, 6, 9
            k, v = rnd(D, H, n), rnd(D, H, n)
            slots = CuArray(Int32[13, 14, 15, 16, 5, 6, 7, 8, 21])   # out of order, across pages
            pools = [CUDA.zeros(T, D, P, H, pages) for _ in 1:4]
            TileInfer.store_kv!(ref, pools[1], pools[2], k, v, slots)
            TileInfer.store_kv!(tile, pools[3], pools[4], k, v, slots)
            @test host(pools[3]) == host(pools[1]) && host(pools[4]) == host(pools[2])
        end

        @testset "embed H = $H" for H in (1024, 2560)
            table, ids = rnd(H, 50), CuArray(Int32[0, 49, 7, 7])
            @test host(TileInfer.embed(tile, table, ids)) == host(TileInfer.embed(ref, table, ids))
        end

        @testset "linear ($in, $out) n = $n" for (in, out) in ((1024, 4096), (2048, 1024), (1000, 37)),
                                                 n in (1, 3, 17, 512)
            W, x = rnd(in, out; scale = 1 / 30), rnd(in, n)
            y_ref, y = TileInfer.linear(ref, W, x), TileInfer.linear(tile, W, x)
            @test size(y) == (out, n)
            if T == BFloat16
                @test ulps(y_ref, y, T) <= 1          # one rounding of a Float32 sum
            else
                @test host(y) ≈ host(y_ref) rtol = 1e-3
            end
        end

        @testset "linear with bias" begin
            W, x, b = rnd(64, 32), rnd(64, 2), rnd(32)
            @test host(TileInfer.linear(tile, W, x, b)) ≈ host(TileInfer.linear(ref, W, x, b)) rtol = 1e-2
        end
    end

    @test !TileInfer.has_kernel(tile, TileInfer.attention)        # reference fallback until M5
    @test TileInfer.has_kernel(tile, TileInfer.linear)
end
