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

    # Key smoothing factors for an INT8 cache, the same on both channels of a rotary pair (as
    # key_smoothing makes them), spanning two orders of magnitude with an RMS of 1, so that
    # keys scaled by them keep their size.
    function smoothing(D)
        c = exp.(randn(Float32, D ÷ 2) .* 1.5f0)
        c ./= sqrt(sum(abs2, c) / length(c))
        return CuArray([c; c])
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

        @testset "xielu $dims" for dims in ((21504, 1), (21504, 5), (200, 3))
            x = rnd(dims...; scale = 3)
            copyto!(x, 1, CuArray(T[-30, -1f-7, 0, 1f-7]), 1, 4)
            a = TileInfer.XIELU{T}(1.3, 0.9, 0.5, -1e-6)
            # Bit-identical in BFloat16 up to expm1, which is Kahan's form here and CUDA's
            # expm1f in the reference (a few Float32 ulps apart, about 1 in 1000 roundings).
            @test ulps(TileInfer.xielu(ref, x, a), TileInfer.xielu(tile, x, a), T) <= (T == BFloat16 ? 1 : 8)
        end

        @testset "store_kv" begin
            D, P, H, pages, n = 128, 4, 8, 6, 9
            k, v = rnd(D, H, n), rnd(D, H, n)
            slots = CuArray(Int32[13, 14, 15, 16, 5, 6, 7, 8, 21])   # out of order, across pages
            pools = [CUDA.zeros(T, D, P, H, pages) for _ in 1:4]
            TileInfer.store_kv!(ref, pools[1], pools[2], k, v, slots)
            TileInfer.store_kv!(tile, pools[3], pools[4], k, v, slots)
            @test host(pools[3]) == host(pools[1]) && host(pools[4]) == host(pools[2])
            # INT8 cache: the same quantization (values and scales) as the reference.
            a, b = TileInfer.int8_kv_pools(D, P, H, pages), TileInfer.int8_kv_pools(D, P, H, pages)
            TileInfer.store_kv!(ref, a..., k, v, slots)
            TileInfer.store_kv!(tile, b..., k, v, slots)
            @test Array(a[1].layer_q) == Array(b[1].layer_q) && Array(a[1].layer_scale) == Array(b[1].layer_scale)
            # Round to nearest: within half a step of the input, scaled to ±127.
            x = host(k)[:, 1, 1]
            s = Array(a[1].scale)[1, 1, 4]                       # slot 13: offset 1, page 4
            @test s ≈ maximum(abs, x) / 127
            @test maximum(abs.(s .* Array(a[1].q)[:, 1, 1, 4] .- x)) <= s / 2 * (1 + 1e-5)
            @test maximum(abs, Array(a[1].q)[:, 1, 1, 4]) == 127
            # With smoothed keys.
            c = smoothing(D)
            a = TileInfer.int8_kv_pools(D, P, H, pages; key_smoothing = c)
            b = TileInfer.int8_kv_pools(D, P, H, pages; key_smoothing = c)
            TileInfer.store_kv!(ref, a..., k, v, slots)
            TileInfer.store_kv!(tile, b..., k, v, slots)
            @test Array(a[1].layer_q) == Array(b[1].layer_q) && Array(a[1].layer_scale) == Array(b[1].layer_scale)
        end

        @testset "embed H = $H" for H in (1024, 2560)
            table, ids = rnd(H, 50), CuArray(Int32[0, 49, 7, 7])
            @test host(TileInfer.embed(tile, table, ids)) == host(TileInfer.embed(ref, table, ids))
        end

        # Float32 (a correctness check, not a production type): fewer token counts.
        @testset "linear ($in, $out) n = $n" for (in, out) in ((1024, 4096), (2048, 1024), (1000, 37)),
                                                 n in (T == BFloat16 ? (1, 3, 17, 100, 512) : (1, 17, 512))
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

            # Into an INT8 cache: the fused kernel quantizes as store_kv! does.
            c = smoothing(D)
            function run_int8(be; fused = true)
                q, (kp, vp) = CUDA.zeros(T, D, nq, n), TileInfer.int8_kv_pools(D, P, nkv, pages; key_smoothing = c)
                args = (be, q, kp, vp, qkv, qn, kn, table, pos, slots, 1e-6)
                fused ? TileInfer.attention_inputs!(args...) :
                        invoke(TileInfer.attention_inputs!, Tuple{TileInfer.Backend, ntuple(_ -> Any, 10)...}, args...)
                return host(q), Array(kp.layer_q), Array(kp.layer_scale)
            end
            @test run_int8(tile) == run_int8(tile; fused = false)
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

        @testset "linear_xielu ($in, $I) n = $n" for (in, I) in ((4096, 21504), (200, 100)), n in (1, 3)
            W, x = rnd(in, I; scale = 1 / 30), rnd(in, n)
            a = TileInfer.XIELU{T}(18.1, 1.2, 0.5, -1e-6)
            y, y_unfused = CuMatrix{T}(undef, I, n), CuMatrix{T}(undef, I, n)
            TileInfer.linear_xielu!(tile, y, W, x, a)
            invoke(TileInfer.linear_xielu!, Tuple{TileInfer.Backend, ntuple(_ -> Any, 4)...}, tile, y_unfused, W, x, a)
            @test ulps(y_unfused, y, T) <= (T == BFloat16 ? 1 : 8)    # same as GEMV, then xielu
            y_ref = TileInfer.xielu(ref, TileInfer.linear(ref, W, x), a)
            @test ulps(y_ref, y, T) <= (T == BFloat16 ? 2 : 1000) # the linear's 1 ulp, through αp x²
        end

        # INT8 weights: same Float32 sums of the Int8 values, scaled, then one rounding.
        @testset "int8 linear ($in, $out) n = $n" for (in, out) in ((4096, 6144), (1024, 3000), (1000, 37)),
                                                      n in (T == BFloat16 ? (1, 3, 100, 512) : (1, 512))
            W, x = TileInfer.quantize_int8(rnd(in, out)), rnd(in, n)
            y_ref, y = TileInfer.linear(ref, W, x), TileInfer.linear(tile, W, x)
            if T == BFloat16
                @test ulps(y_ref, y, T) <= 1
            else
                @test host(y) ≈ host(y_ref) rtol = 1e-4       # Float32 sums in another order
            end
            if n == 1
                y32 = TileInfer.linear!(tile, CuMatrix{Float32}(undef, out, 1), W, x)
                @test Array(y32) == host(y)
            end
        end

        @testset "int8 linear_swiglu / linear_xielu ($in, $I) n = $n" for (in, I) in ((4096, 21504), (200, 96)), n in (1, 3)
            a = TileInfer.XIELU{T}(18.1, 1.2, 0.5, -1e-6)
            Wg, Wu, x = TileInfer.quantize_int8(rnd(in, 2I)), TileInfer.quantize_int8(rnd(in, I)), rnd(in, n)
            fused, unfused, scratch = CuMatrix{T}(undef, I, n), CuMatrix{T}(undef, I, n), CuMatrix{T}(undef, 2I, n)
            TileInfer.linear_swiglu!(tile, fused, scratch, Wg, x)
            invoke(TileInfer.linear_swiglu!, Tuple{TileInfer.Backend, ntuple(_ -> Any, 4)...}, tile, unfused, scratch, Wg, x)
            @test ulps(unfused, fused, T) <= (T == BFloat16 ? 1 : 8)
            TileInfer.linear_xielu!(tile, fused, Wu, x, a)
            invoke(TileInfer.linear_xielu!, Tuple{TileInfer.Backend, ntuple(_ -> Any, 4)...}, tile, unfused, Wu, x, a)
            @test ulps(unfused, fused, T) <= (T == BFloat16 ? 1 : 8)
        end

        @testset "linear into Float32 n = $n" for n in (1, 17)
            W, x = rnd(1024, 300; scale = 1 / 30), rnd(1024, n)
            y = TileInfer.linear!(tile, CuMatrix{Float32}(undef, 300, n), W, x)
            @test Array(y) == host(TileInfer.linear(tile, W, x))       # rounded to T, then converted
        end

        # (head_dim, heads, kv_heads, page_size, start, n, causal): decode and verify (split-KV,
        # a step that crosses a page boundary so some splits see no key of a query; 8 tokens at
        # group 2, 16 (DFlash's verify) at group 4, and 12 padded to 16), prefill with and
        # without cached context, group 4, and a group of 3 (prefill kernel only); non-causal
        # (DFlash blocks) on both kernels, with a partial last page. Each case compiles kernels
        # of its own, so the list covers every path once.
        @testset "attention $case" for case in ((128, 16, 8, 64, 1023, 1, true), (128, 16, 8, 64, 63, 2, true),
                                                 (128, 16, 8, 64, 100, 8, true), (128, 16, 8, 64, 0, 512, true),
                                                 (128, 16, 8, 64, 200, 300, true), (128, 32, 8, 64, 700, 3, true),
                                                 (64, 12, 4, 16, 40, 5, true), (128, 32, 8, 64, 950, 16, true),
                                                 (128, 32, 8, 64, 63, 12, true), (128, 32, 8, 64, 1000, 16, false),
                                                 (128, 32, 8, 64, 300, 30, false))
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

            # INT8 cache (same cases, BFloat16 only: models with an INT8 cache run in BFloat16),
            # with keys whose channels scale like the smoothing factors (as real keys do): against
            # exact attention over the dequantized cache, which the reference computes with
            # Float32 queries.
            if T == BFloat16
                c = smoothing(D)
                k_src = k_pages .* reshape(T.(c), D, 1, 1, 1)
                kq, vq = TileInfer.int8_kv_pools(D, P, KVH, pages; key_smoothing = c)
                every = CuArray(Int32.(1:P*pages))
                TileInfer.store_kv!(ref, kq, vq, reshape(permutedims(k_src, (1, 3, 2, 4)), D, KVH, :),
                                    reshape(permutedims(v_pages, (1, 3, 2, 4)), D, KVH, :), every)
                y8 = TileInfer.attention(tile, q, kq, vq, table, positions, 1 / sqrt(D); causal)
                exact8 = TileInfer.attention(ref, f32(q), kq, vq, table, positions, 1 / sqrt(D); causal)
                @test ulps(exact8, y8, T) <= 10   # the kernels round the smoothed queries c q to T
                # Quantization error of the whole op, against the unquantized keys and values.
                exact_src = TileInfer.attention(ref, f32(q), f32(k_src), f32(v_pages), table, positions, 1 / sqrt(D); causal)
                @test sqrt(sum(abs2, host(y8) .- host(exact_src)) / sum(abs2, host(exact_src))) < 0.02
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

    # Speculative sampling: a drafter's sample with its distribution q, and the verifier's
    # p/q decision, against the host reference with the same uniform numbers.
    # Several columns in one call (a DFlash block): the tokens, distributions and counter of
    # sample_token! on each column in turn (same kernels' arithmetic, so identical).
    @testset "sample_tokens (vocab = $V, $cols columns)" for (V, cols) in ((151936, 15), (5000, 3))
        K = TileInfer.MAX_TOP_K
        scratch = TileInfer.sample_cols_scratch(cols)
        for (temperature, top_k, top_p) in ((0, 0, 1), (0.6, 20, 0.95), (1.5, 64, 1)), with_q in (false, true)
            p = TileInfer.SamplingParams(; temperature, top_k, top_p)
            logits = CuArray(randn(Float32, V, cols) .* 3)
            seed = rand(UInt32)
            s1, s2 = TileInfer.TokenSampler(V), TileInfer.TokenSampler(V)
            TileInfer.reset_sampler!(s1, p, seed)
            TileInfer.reset_sampler!(s2, p, seed)
            TileInfer.set_counter!(s1, 7)
            TileInfer.set_counter!(s2, 7)
            one, one_m = CUDA.zeros(Int32, cols), CUDA.zeros(Int32, cols)
            all, all_m = CUDA.zeros(Int32, cols), CUDA.zeros(Int32, cols)
            q1 = (CUDA.zeros(Int32, K, cols), CUDA.zeros(Float32, K, cols))
            q2 = (CUDA.zeros(Int32, K, cols), CUDA.zeros(Float32, K, cols))
            for j in 1:cols
                args = (tile, view(one, j:j), view(one_m, j:j), logits[:, j:j], s1)
                with_q ? TileInfer.sample_token!(args..., (view(q1[1], :, j), view(q1[2], :, j))) :
                         TileInfer.sample_token!(args...)
            end
            TileInfer.sample_tokens!(tile, all, all_m, logits, s2, scratch, with_q ? q2 : nothing)
            @test Array(all) == Array(one) == Array(all_m)
            @test Array(s2.state)[3] == Array(s1.state)[3] == 7 + cols
            with_q && @test Array(q2[1]) == Array(q1[1]) && Array(q2[2]) == Array(q1[2])
            # The generic fallback: the reference sampler on each column (on the host); greedy
            # only, where Float32 round-off cannot move a decision.
            temperature == 0 || continue
            fall, fall_m = CUDA.zeros(Int32, cols), CUDA.zeros(Int32, cols)
            TileInfer.reset_sampler!(s2, p, seed); TileInfer.set_counter!(s2, 7)
            TileInfer.sample_tokens!(ref, fall, fall_m, logits, s2, scratch)
            @test Array(fall) == Array(one)
        end
    end

    @testset "sample_token with q, accept_token (vocab = $V)" for V in (151936, 5000)
        K = TileInfer.MAX_TOP_K
        ids, mirror, draft = CUDA.zeros(Int32, 1), CUDA.zeros(Int32, 1), CUDA.zeros(Int32, 1)
        q_ids, q_probs = CUDA.zeros(Int32, K), CUDA.zeros(Float32, K)
        s, sq = TileInfer.TokenSampler(V), TileInfer.TokenSampler(V)
        checked = skipped = accepted = 0
        for case in 1:4, (temperature, top_k, top_p) in ((0, 0, 1), (1, 2, 1), (0.6, 20, 0.95), (1.5, 64, 1))
            p = TileInfer.SamplingParams(; temperature, top_k, top_p)
            seed = rand(UInt32)
            TileInfer.reset_sampler!(s, p, seed)
            TileInfer.reset_sampler!(sq, p, TileInfer.draft_seed(seed))
            for counter in UInt32(0):UInt32(4)
                # The draft's logits: the target's, perturbed (case 1: equal; case 4: unrelated).
                logits = randn(Float32, V) .* 3
                qlogits = case == 1 ? logits : case == 4 ? randn(Float32, V) .* 3 : logits .+ randn(Float32, V) .* case
                TileInfer.sample_token!(tile, draft, mirror, CuArray(reshape(qlogits, V, 1)), sq, (q_ids, q_probs))
                t, q = TileInfer.filtered_dist(qlogits, p)
                @test Array(q_ids) == [t .- 1; fill(-1, K - length(t))]
                @test Array(q_probs) ≈ [q; zeros(K - length(q))] atol = 1e-5
                d = Array(draft)[1]
                @test d == Array(mirror)[1] && d + 1 in t
                TileInfer.accept_token!(tile, ids, mirror, CuArray(reshape(logits, V, 1)), s, draft, (q_ids, q_probs))
                @test Array(ids)[1] == Array(mirror)[1]
                ua, u = TileInfer.accept_uniform(seed, counter), TileInfer.sampling_uniform(seed, counter)
                expected = TileInfer.accept_host(logits, p, d, Array(q_ids), Array(q_probs), ua, u)
                # Skip decisions Float32 round-off could move, as for sample_token.
                if any(TileInfer.accept_host(logits, p, d, Array(q_ids), Array(q_probs), clamp(a, 0, 1), clamp(b, 0, 1)) != expected
                       for (a, b) in ((ua - 1f-5, u), (ua + 1f-5, u), (ua, u - 1f-5), (ua, u + 1f-5)))
                    skipped += 1
                    continue
                end
                checked += 1
                accepted += expected == d
                Array(ids)[1] == expected || @test Array(ids)[1] == expected
            end
            @test Array(s.state)[3] == Array(sq.state)[3] == 5
        end
        @test skipped < checked ÷ 10
        @test 0 < accepted < checked                     # both branches exercised
        # The fallbacks run the host reference itself.
        logits = CUDA.randn(Float32, V, 1) .* 3
        p = TileInfer.SamplingParams(temperature = 0.6, top_k = 20, top_p = 0.95)
        TileInfer.reset_sampler!(s, p, UInt32(5))
        TileInfer.sample_token!(ref, draft, mirror, logits, s, (q_ids, q_probs))
        t, q = TileInfer.filtered_dist(vec(Array(logits)), p)
        @test Array(q_ids)[1:length(t)] == t .- 1 && Array(q_probs)[1:length(q)] ≈ q
        @test Array(draft)[1] == TileInfer.sample_host(vec(Array(logits)), p, TileInfer.sampling_uniform(UInt32(5), UInt32(0)))
        TileInfer.accept_token!(ref, ids, mirror, logits, s, draft, (q_ids, q_probs))
        @test Array(ids)[1] == Array(draft)[1]           # q = p: always accepted
        @test Array(s.state)[3] == 2
    end

    @test TileInfer.has_kernel(tile, TileInfer.attention)
end
