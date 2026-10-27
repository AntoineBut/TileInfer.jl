# TileBackend kernels against the reference backend on random inputs, including shapes
# that are not powers of two (heads = 24, hidden = 2560 or 100, odd token counts). The
# element-wise kernels round at the same points as the reference and must be
# bit-identical; reductions may differ by accumulation order.

using Random: randperm

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
            x, θ = rnd(dims...), rand(Float32, D ÷ 2, 50) .* 100
            c, s = CuArray(T.(vcat(cos.(θ), cos.(θ)))), CuArray(T.(vcat(sin.(θ), sin.(θ))))
            positions = CuArray(Int32.(rand(0:49, n)))
            @test host(TileInfer.rope(tile, x, c, s, positions)) == host(TileInfer.rope(ref, x, c, s, positions))
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

        # (head_dim, heads, kv_heads, page_size, start, n): decode and verify (split-KV, a
        # step that crosses a page boundary so some splits see no key of a query), prefill
        # with and without cached context, group 4, and a group of 3 (prefill kernel only).
        @testset "attention $case" for case in ((128, 16, 8, 64, 0, 1), (128, 16, 8, 64, 1023, 1),
                                                 (128, 16, 8, 64, 63, 2), (128, 16, 8, 64, 100, 8),
                                                 (128, 16, 8, 64, 0, 37), (128, 16, 8, 64, 0, 512),
                                                 (128, 16, 8, 64, 200, 300), (128, 32, 8, 64, 700, 3),
                                                 (64, 12, 4, 16, 40, 5))
            D, H, KVH, P, start, n = case
            len = start + n
            pages = cld(len, P) + 2
            # Fragmented, out of order, and zero-padded to 16 entries as forward! does.
            table = CuArray([Int32.(randperm(pages)[1:cld(len, P)]); zeros(Int32, 16 - cld(len, P))])
            positions = CuArray(Int32.(start:start+n-1))
            q, k_pages, v_pages = rnd(D, H, n), rnd(D, P, KVH, pages), rnd(D, P, KVH, pages)
            y = TileInfer.attention(tile, q, k_pages, v_pages, table, positions, 1 / sqrt(D))
            @test size(y) == size(q)
            # Exact attention of the same inputs. The reference rounds scores and
            # probabilities to T as PyTorch's eager attention does (up to ~12 BFloat16 ulps
            # from exact); the flash kernels keep them in Float32 until the PV product.
            f32(x) = CuArray(host(x))
            exact = TileInfer.attention(ref, f32(q), f32(k_pages), f32(v_pages), table, positions, 1 / sqrt(D))
            if T == BFloat16
                @test ulps(exact, y, T) <= 5      # ≤ 3.5 over 30 seeds at n = 512
            else
                @test host(y) ≈ host(exact) rtol = 1e-5
            end
        end
    end

    @test TileInfer.has_kernel(tile, TileInfer.attention)
end
