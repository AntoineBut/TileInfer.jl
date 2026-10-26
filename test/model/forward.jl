# Full forward pass against Hugging Face transformers' logits
# (scripts/gen_model_fixtures.py <model_dir> qwen3-0.6b, eager attention, CPU).

const LOGITS = TileInfer.JSON.parsefile(fixture("logits", "qwen3-0.6b.json"))["cases"]

# Our logits at HF's top-k ids for every position: (max |Δ|, positions whose argmax differs
# from HF's although HF's top-1/top-2 gap is at least `tie`).
function compare_logits(logits, ref; tie)
    maxdiff, mismatches = 0.0, 0
    for t in axes(logits, 2)
        ids = Int.(ref["topk_ids"][t]) .+ 1
        values = Float64.(ref["topk_logits"][t])
        maxdiff = max(maxdiff, maximum(abs.(logits[ids, t] .- values)))
        gap = values[1] - values[2]
        mismatches += argmax(logits[:, t]) != ids[1] && gap >= tie
    end
    return maxdiff, mismatches
end

dir = test_model_dir()
if dir !== nothing
    @testset "forward vs transformers" begin
        @testset "float32" begin
            # Same math in the same precision: agreement down to accumulation order.
            m = TileInfer.load_model(TEST_MODEL; T = Float32, max_len = 128)
            for (name, case) in LOGITS
                ids = Int.(case["ids"])
                logits = Array(TileInfer.forward!(m, TileInfer.KVCache(m), ids; all_logits = true))
                maxdiff, mismatches = compare_logits(logits, case["float32"]; tie = 0)
                @test maxdiff < 1e-2
                @test mismatches == 0
                @test TileInfer.generate(m, ids; max_new_tokens = 24) == case["float32"]["greedy"]
            end

            @testset "incremental decode matches one-shot prefill" begin
                ids = Int.(LOGITS["chat"]["ids"])
                full = Array(TileInfer.forward!(m, TileInfer.KVCache(m), ids; all_logits = true))
                cache = TileInfer.KVCache(m)
                TileInfer.forward!(m, cache, ids[1:10])
                for t in 11:length(ids)
                    @test Array(TileInfer.forward!(m, cache, ids[t:t]))[:, 1] ≈ full[:, t] rtol = 1e-4
                end
                # Rewinding the cache and re-feeding a token reproduces its logits.
                TileInfer.truncate!(cache, length(ids) - 1)
                @test Array(TileInfer.forward!(m, cache, ids[end:end]))[:, 1] ≈ full[:, end] rtol = 1e-4
            end
            m = nothing; GC.gc(); CUDA.reclaim()
        end

        @testset "bfloat16" begin
            # Rounding to BFloat16 at the same points as PyTorch, but with GPU vs CPU
            # accumulation order: logits within a few BFloat16 ulps (0.125 at |logit| ~ 20),
            # argmax equal except on near-ties. Greedy continuations are not compared:
            # they legitimately diverge after the first near-tie.
            m = TileInfer.load_model(TEST_MODEL; T = BFloat16, max_len = 128)
            for (name, case) in LOGITS
                ids = Int.(case["ids"])
                logits = Array(TileInfer.forward!(m, TileInfer.KVCache(m), ids; all_logits = true))
                maxdiff, mismatches = compare_logits(logits, case["bfloat16"]; tie = 0.25)
                @test maxdiff < 1.0
                @test mismatches == 0
            end
            m = nothing; GC.gc(); CUDA.reclaim()
        end
    end
end
