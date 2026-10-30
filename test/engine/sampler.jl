using Random: Xoshiro

@testset "sampler" begin
    P = TileInfer.SamplingParams
    host(logits, p, u) = TileInfer.sample_host(logits, p, u)
    logits = Float32[0.0, 3.0, 1.0, 2.5, -1.0]

    @testset "sample_host" begin
        us = range(0, 1; length = 2001)[1:end-1]
        @test all(host(logits, P(), u) == 1 for u in us)                       # greedy, 0-based
        @test all(host(logits, P(temperature = 2.0, top_k = 1), u) == 1 for u in us)
        # Inverse transform sampling: the share of u values picking a token is its probability.
        share(p, id) = count(u -> host(logits, p, u) == id, us) / length(us)
        @test Set(host(logits, P(temperature = 1.0, top_k = 2), u) for u in us) == Set([1, 3])
        @test share(P(temperature = 1.0, top_k = 2), 1) ≈ 1 / (1 + exp(-0.5)) atol = 1e-3
        probs = exp.(logits) ./ sum(exp.(logits))
        @test [share(P(temperature = 1.0), id) for id in 0:4] ≈ probs atol = 1e-3
        # Survivors are taken best first, so small u picks the best token.
        @test host(logits, P(temperature = 1.0), 0.0) == 1
        @test host(logits, P(temperature = 1.0), 0.9999) == 4
        # top_p keeps the smallest prefix reaching the mass: here only the best token, then two.
        @test all(host(logits, P(temperature = 0.1, top_p = 0.5), u) == 1 for u in us)
        @test Set(host(logits, P(temperature = 1.0, top_p = 0.7), u) for u in us) == Set([1, 3])
        # Ties go to the smaller id, also across the top-k boundary.
        tied = Float32[1, 5, 2, 5, 5]
        @test TileInfer.top_candidates(tied, 2) == [2, 4]
        @test TileInfer.top_candidates(tied, 5) == [2, 4, 5, 3, 1]
        @test host(tied, P(temperature = 1.0, top_k = 2), 0.0) == 1
        @test host(tied, P(temperature = 1.0, top_k = 2), 0.99) == 3
    end

    @testset "filtered_dist" begin
        ids, probs = TileInfer.filtered_dist(logits, P(temperature = 1.0, top_k = 2))
        @test ids == [2, 4] && probs ≈ [1, exp(-0.5)] ./ (1 + exp(-0.5))
        @test TileInfer.filtered_dist(logits, P()) == ([2], [1.0])
        @test TileInfer.filtered_dist(logits, P(temperature = 1.0, top_p = 0.7))[1] == [2, 4]
    end

    # Speculative sampling: whatever q the drafts come from, the emitted tokens are distributed
    # as the target's p, and a rejected draft is never emitted.
    @testset "accept_host (q from $name)" for (name, qlogits, qp) in (
            ("another model", Float32[2.0, 0.5, 1.0, 2.5, 0.0], P(temperature = 1.0, top_k = 4)),
            ("the target", logits, P(temperature = 0.8, top_k = 3, top_p = 0.9)),
            ("a greedy draft", Float32[0, 0, 0, 0, 1], P()))
        p = P(temperature = 0.8, top_k = 3, top_p = 0.9)
        q_ids, q_probs = TileInfer.filtered_dist(qlogits, qp)
        q_ids .-= 1
        rng = Xoshiro(3)
        n = 200_000
        counts = zeros(Int, length(logits))
        accepted = 0
        for _ in 1:n
            d = TileInfer.sample_host(qlogits, qp, rand(rng))
            t = TileInfer.accept_host(logits, p, d, q_ids, q_probs, rand(rng), rand(rng))
            counts[t+1] += 1
            accepted += t == d
        end
        ids, probs = TileInfer.filtered_dist(logits, p)
        expected = zeros(length(logits))
        expected[ids] .= probs
        @test counts ./ n ≈ expected atol = 4e-3
        # The acceptance rate is Σ min(p, q).
        qfull = zeros(length(logits))
        qfull[q_ids .+ 1] .= q_probs
        @test accepted / n ≈ sum(min.(expected, qfull)) atol = 4e-3
    end

    @testset "sampling_uniform" begin
        u = [TileInfer.sampling_uniform(UInt32(seed), UInt32(c)) for seed in 1:20 for c in 0:999]
        @test all(x -> 0 <= x < 1, u)
        @test sum(u) / length(u) ≈ 0.5 atol = 0.01
        counts = [count(x -> (b - 1) / 10 <= x < b / 10, u) for b in 1:10]   # 2000 expected each
        @test all(c -> abs(c - 2000) < 200, counts)
        @test allunique(u[1:1000])
        @test TileInfer.sampling_uniform(UInt32(1), UInt32(0)) != TileInfer.sampling_uniform(UInt32(2), UInt32(0))
    end

    @testset "parameters" begin
        @test TileInfer.on_device(P())
        @test TileInfer.on_device(P(temperature = 0.6, top_k = 20, top_p = 0.95))
        @test !TileInfer.on_device(P(temperature = 0.6))                            # no top-k
        @test !TileInfer.on_device(P(temperature = 0.6, top_k = TileInfer.MAX_TOP_K + 1))
        @test_throws ArgumentError TileInfer.reset_sampler!(TileInfer.TokenSampler(10), P(temperature = 1.0),
                                                            UInt32(0))
        @test_throws ArgumentError P(top_p = 0)
        @test_throws ArgumentError P(temperature = -1)
    end
end
