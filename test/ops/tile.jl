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
                                                 n in (1, 3, 17, 100, 512)
            W, x = rnd(in, out; scale = 1 / 30), rnd(in, n)
            y_ref, y = TileInfer.linear(ref, W, x), TileInfer.linear(tile, W, x)
            @test size(y) == (out, n)
            if T == BFloat16
                @test ulps(y_ref, y, T) <= 1          # one rounding of a Float32 sum
            else
                @test host(y) ≈ host(y_ref) rtol = 1e-3
            end
        end

        # Fused q/k norm + rope + KV write vs the same tile ops run one by one (the generic
        # fallback): bit-identical; and vs the reference within the rmsnorm tolerance.
        @testset "attention_inputs (qk_norm = $qk_norm)" for qk_norm in (true, false)
            D, nq, nkv, P, pages, L = 128, 16, 8, 16, 6, 64
            positions = Int32[3, 17, 18, 40, 0]                      # any order, across pages
            n = length(positions)
            slots = CuArray(Int32[13, 14, 15, 16, 70])
            θ = rand(Float32, D ÷ 2, L) .* 100
            angles = vcat(θ, θ)
            c, s = T.(cos.(angles)), T.(sin.(angles))
            table = TileInfer.RotaryTable(CuArray(c), CuArray(s),
                                          CuArray(reshape(permutedims(cat(c, s; dims = 3), (1, 3, 2)), D, :)))
            qkv = rnd((nq + 2nkv) * D, n)
            qn, kn = qk_norm ? (rnd(D), rnd(D)) : (nothing, nothing)
            pos = CuArray(positions)
            function run(be; fused = true)
                q, kp, vp = CUDA.zeros(T, D, nq, n), CUDA.zeros(T, D, P, nkv, pages), CUDA.zeros(T, D, P, nkv, pages)
                args = (be, q, kp, vp, qkv, qn, kn, table, pos, slots, 1e-6)
                fused ? TileInfer.attention_inputs!(args...) :
                        invoke(TileInfer.attention_inputs!, Tuple{TileInfer.Backend, ntuple(_ -> Any, 10)...}, args...)
                return host.((q, kp, vp))
            end
            fused, unfused, reference = run(tile), run(tile; fused = false), run(ref)
            @test fused == unfused
            worst = maximum(ulps(r, f, T) for (r, f) in zip(reference, fused))
            @test worst <= (T == BFloat16 ? 1 : 32)    # rmsnorm's sum order, amplified by rope
        end

        @testset "add_rmsnorm H = $H" for H in (1024, 2560, 100)
            x, d, w = rnd(H, 5), rnd(H, 5), rnd(H)
            x1, x2, h1, h2 = copy(x), copy(x), similar(x), similar(x)
            TileInfer.add_rmsnorm!(tile, x1, h1, d, w, 1e-6)
            invoke(TileInfer.add_rmsnorm!, Tuple{TileInfer.Backend, ntuple(_ -> Any, 5)...}, tile, x2, h2, d, w, 1e-6)
            @test host(x1) == host(x2) && host(h1) == host(h2)       # same as add, then rmsnorm
        end

        @testset "linear_swiglu ($in, 2 × $I) n = $n" for (in, I) in ((1024, 3072), (200, 100)), n in (1, 3)
            W, x = rnd(in, 2I; scale = 1 / 30), rnd(in, n)
            y, y_unfused, scratch = CuMatrix{T}(undef, I, n), CuMatrix{T}(undef, I, n), CuMatrix{T}(undef, 2I, n)
            TileInfer.linear_swiglu!(tile, y, scratch, W, x)
            invoke(TileInfer.linear_swiglu!, Tuple{TileInfer.Backend, ntuple(_ -> Any, 4)...}, tile, y_unfused, scratch, W, x)
            # Same as GEMV then silu_mul; bit-identical when both use the same column tile
            # (as at Qwen3 sizes), else the reductions may round differently.
            @test ulps(y_unfused, y, T) <= (T == BFloat16 ? 1 : 8)
        end

        @testset "linear into Float32 n = $n" for n in (1, 17)
            W, x = rnd(1024, 300; scale = 1 / 30), rnd(1024, n)
            y = TileInfer.linear!(tile, CuMatrix{Float32}(undef, 300, n), W, x)
            @test Array(y) == host(TileInfer.linear(tile, W, x))       # rounded to T, then converted
        end

        @testset "linear with bias" begin
            W, x, b = rnd(64, 32), rnd(64, 2), rnd(32)
            @test host(TileInfer.linear(tile, W, x, b)) ≈ host(TileInfer.linear(ref, W, x, b)) rtol = 1e-2
        end

        # (head_dim, heads, kv_heads, page_size, start, n, causal): decode and verify (split-KV,
        # a step that crosses a page boundary so some splits see no key of a query), prefill
        # with and without cached context, group 4, and a group of 3 (prefill kernel only);
        # non-causal (DFlash blocks) on both kernels, with a partial last page.
        @testset "attention $case" for case in ((128, 16, 8, 64, 0, 1, true), (128, 16, 8, 64, 1023, 1, true),
                                                 (128, 16, 8, 64, 63, 2, true), (128, 16, 8, 64, 100, 8, true),
                                                 (128, 16, 8, 64, 0, 37, true), (128, 16, 8, 64, 0, 512, true),
                                                 (128, 16, 8, 64, 200, 300, true), (128, 32, 8, 64, 700, 3, true),
                                                 (64, 12, 4, 16, 40, 5, true),
                                                 (128, 16, 8, 64, 60, 4, false), (128, 16, 8, 64, 0, 20, false),
                                                 (128, 32, 8, 64, 300, 30, false), (128, 16, 8, 64, 10, 100, false))
            D, H, KVH, P, start, n, causal = case
            len = start + n
            pages = cld(len, P) + 2
            # Fragmented, out of order, and zero-padded to 16 entries as forward! does.
            table = CuArray([Int32.(randperm(pages)[1:cld(len, P)]); zeros(Int32, 16 - cld(len, P))])
            positions = CuArray(Int32.(start:start+n-1))
            q, k_pages, v_pages = rnd(D, H, n), rnd(D, P, KVH, pages), rnd(D, P, KVH, pages)
            y = TileInfer.attention(tile, q, k_pages, v_pages, table, positions, 1 / sqrt(D); causal)
            @test size(y) == size(q)
            # Exact attention of the same inputs. The reference rounds scores and
            # probabilities to T as PyTorch's eager attention does (up to ~12 BFloat16 ulps
            # from exact); the flash kernels keep them in Float32 until the PV product.
            f32(x) = CuArray(host(x))
            exact = TileInfer.attention(ref, f32(q), f32(k_pages), f32(v_pages), table, positions, 1 / sqrt(D); causal)
            if T == BFloat16
                @test ulps(exact, y, T) <= 5      # ≤ 3.5 over 30 seeds at n = 512
            else
                @test host(y) ≈ host(exact) rtol = 1e-5
            end
        end
    end

    # The device sampler must pick the token sample_host picks with the same uniform number,
    # except where Float32 round-off can move a decision: u or top_p within 1e-5 of a
    # cumulative-mass boundary (checked by perturbing them on the host).
    @testset "sample_token (vocab = $V)" for V in (151936, 5000, 4096)
        ids, mirror = CUDA.zeros(Int32, 1), CUDA.zeros(Int32, 1)
        s = TileInfer.TokenSampler(V)
        function robust(logits, p, u)
            t = TileInfer.sample_host(logits, p, u)
            alts = [(u - 1f-5, p.top_p), (u + 1f-5, p.top_p), (u, p.top_p - 1e-5), (u, min(1, p.top_p + 1e-5))]
            all(TileInfer.sample_host(logits, TileInfer.SamplingParams(p.temperature, p.top_k, tp), clamp(v, 0, 1)) == t
                for (v, tp) in alts) || return nothing
            return t
        end
        checked = skipped = 0
        for case in 1:6, (temperature, top_k, top_p) in ((0, 0, 1), (1, 1, 1), (1, 2, 1), (0.6, 20, 0.95),
                                                           (1.5, 64, 1), (0.3, 64, 0.5), (1, 37, 0.99))
            logits = randn(Float32, V) .* (case == 2 ? 0.01f0 : 3)                 # case 2: nearly flat
            case == 3 && (logits[[17, V ÷ 2, V]] .= 50)                              # ties
            case == 4 && (logits[end-63:end] .= 20 .+ randn(Float32, 64))           # all in the last chunk
            case == 5 && (logits[1:2:200] .= 10)                                     # ties across the k-th
            case == 6 && (logits .= round.(logits))                                  # many ties everywhere
            p = TileInfer.SamplingParams(; temperature, top_k, top_p)
            seed = rand(UInt32)
            TileInfer.reset_sampler!(s, p, seed)
            for counter in UInt32(0):UInt32(9)
                TileInfer.sample_token!(tile, ids, mirror, CuArray(reshape(logits, V, 1)), s)
                @test Array(ids)[1] == Array(mirror)[1]
                expected = robust(logits, p, TileInfer.sampling_uniform(seed, counter))
                expected === nothing && (skipped += 1; continue)
                checked += 1
                Array(ids)[1] == expected || @test Array(ids)[1] == expected
            end
            @test Array(s.state)[3] == 10                                            # counter advanced
        end
        @test skipped < checked ÷ 20       # mostly case 5, whose equal tokens put top_p on a boundary
        # The fallback runs sample_host itself, so it is exact.
        logits = CUDA.randn(Float32, V, 1) .* 3
        p = TileInfer.SamplingParams(temperature = 0.6, top_k = 20, top_p = 0.95)
        TileInfer.reset_sampler!(s, p, UInt32(5))
        TileInfer.sample_token!(ref, ids, mirror, logits, s)
        @test Array(ids)[1] == Array(mirror)[1] ==
              TileInfer.sample_host(vec(Array(logits)), p, TileInfer.sampling_uniform(UInt32(5), UInt32(0)))
        @test Array(s.state)[3] == 1
    end

    @test TileInfer.has_kernel(tile, TileInfer.attention)
end
