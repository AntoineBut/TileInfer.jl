using Random: Xoshiro

@testset "sampler" begin
    logits = Float32[0.0, 3.0, 1.0, 2.5, -1.0]
    rng = Xoshiro(1)
    @test TileInfer.sample(CuArray(logits), TileInfer.SamplingParams(), rng) == 1   # greedy, 0-based
    # top_k = 1 is greedy at any temperature.
    @test all(TileInfer.sample(logits, TileInfer.SamplingParams(temperature = 2.0, top_k = 1), rng) == 1
              for _ in 1:50)
    # top_k = 2 only ever picks the two best tokens, both with substantial frequency.
    picks = [TileInfer.sample(logits, TileInfer.SamplingParams(temperature = 1.0, top_k = 2), rng) for _ in 1:2000]
    @test Set(picks) == Set([1, 3])
    @test 0.5 < count(==(1), picks) / 2000 < 0.75      # softmax([3, 2.5]) ≈ (0.62, 0.38)
    # top_p keeps the smallest prefix reaching the mass: here only the best token.
    p = TileInfer.SamplingParams(temperature = 0.1, top_p = 0.5)
    @test all(TileInfer.sample(logits, p, rng) == 1 for _ in 1:50)
    @test_throws ArgumentError TileInfer.SamplingParams(top_p = 0)
    @test_throws ArgumentError TileInfer.SamplingParams(temperature = -1)
end
