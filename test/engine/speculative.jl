# Speculative decoding: the n-gram proposer, the verify step against eager forward steps, and
# speculative greedy generation against the plain greedy loop.

@testset "n-gram proposer" begin
    p = TileInfer.NgramProposer(k = 3, max_ngram = 2)
    @test TileInfer.propose(p, [1, 2, 3, 4, 1, 2], 3) == [3, 4, 1]          # bigram match
    @test TileInfer.propose(p, [1, 2, 3, 4, 1, 2], 1) == [3]
    @test TileInfer.propose(p, [5, 2, 7, 9, 2, 8, 2], 3) == [8, 2]          # latest unigram match, cut at the end
    @test TileInfer.propose(p, [7, 1, 2, 9, 1, 2, 5, 1, 2], 2) == [5, 1]    # latest of two bigram matches
    @test TileInfer.propose(p, [1, 2, 3], 3) == Int[]                       # no match
    @test TileInfer.propose(p, [1], 3) == Int[]
    @test_throws ArgumentError TileInfer.NgramProposer(k = TileInfer.MAX_GRAPH_TOKENS)
end

if test_model_dir() !== nothing
    @testset "speculative decoding" begin
        m = TileInfer.load_model(TEST_MODEL; backend = TileBackend(), max_len = 512)
        new_cache() = TileInfer.PagedKVCache(m; max_tokens = 512, page_size = 16)
        # A prompt that repeats itself, so that prompt lookup finds drafts.
        prompt = repeat(collect(1000:1023), 3)

        @testset "verify steps return the argmax of every position" begin
            c, eager_c = new_cache(), new_cache()
            r = TileInfer.ModelRunner(m, c)
            s, eager_s = TileInfer.new_sequence(c), TileInfer.new_sequence(eager_c)
            TileInfer.forward!(r, s, prompt)
            TileInfer.forward!(m, eager_c, eager_s, prompt)
            for ids in ([5], [6, 7, 8], [9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19])
                logits = Array(TileInfer.forward!(m, eager_c, eager_s, ids; all_logits = true))
                @test TileInfer.verify!(r, s, ids) == [argmax(col) - 1 for col in eachcol(logits)]
            end
            @test s.tokens == eager_s.tokens
            @test TileInfer.num_graphs(r) == 3                     # 1, 3 and 8 tokens
        end

        # Generations are compared in Float32: in BFloat16, verify steps of several tokens round
        # differently from decode steps, and the two runs part at the first near-tie (an exact
        # BFloat16 tie after 46 tokens with this prompt).
        m = TileInfer.load_model(TEST_MODEL; T = Float32, backend = TileBackend(), max_len = 512)

        function plain(max_new_tokens, stop_ids)
            c = new_cache()
            s = TileInfer.new_sequence(c)
            out = TileInfer.generate!(TileInfer.ModelRunner(m, c), s, prompt; max_new_tokens, stop_ids, pipeline = false)
            return out, copy(s.tokens), copy(s.block_table)
        end
        function speculative(max_new_tokens, stop_ids; k = 4, stats = TileInfer.SpecStats())
            c = new_cache()
            s = TileInfer.new_sequence(c)
            out = TileInfer.generate!(TileInfer.ModelRunner(m, c), s, prompt; max_new_tokens, stop_ids,
                                      speculative = TileInfer.NgramProposer(; k), spec_stats = stats)
            return out, copy(s.tokens), copy(s.block_table)
        end

        @testset "same tokens and cache as plain greedy decoding (k = $k)" for k in (1, 4, 7)
            stats = TileInfer.SpecStats()
            expected = plain(60, Int[])
            @test speculative(60, Int[]; k, stats) == expected
            @test stats.accepted > 0
            @test stats.steps < 60
            @test 0 < TileInfer.acceptance_rate(stats) <= 1
        end

        @testset "stops like plain greedy decoding" begin
            full, = plain(30, Int[])
            # Stop tokens early and late in the output, and every budget up to 12.
            for stop in (full[2], full[10], full[25])
                @test speculative(30, [stop]) == plain(30, [stop])
            end
            @test all(speculative(n, Int[]) == plain(n, Int[]) for n in 1:12)
            @test speculative(0, Int[])[1] == Int[]
        end

        @testset "sampled speculative decoding is rejected" begin
            c = new_cache()
            @test_throws ArgumentError TileInfer.generate!(TileInfer.ModelRunner(m, c), TileInfer.new_sequence(c), prompt;
                                                           sampling = TileInfer.SamplingParams(temperature = 0.6, top_k = 20),
                                                           speculative = TileInfer.NgramProposer())
        end

        @testset "reference backend" begin
            ref = TileInfer.DecoderLM(m.config, m.weights, m.rope, RefBackend())
            c = TileInfer.PagedKVCache(ref; max_tokens = 128)
            out = TileInfer.generate!(TileInfer.ModelRunner(ref, c), TileInfer.new_sequence(c), prompt;
                                      max_new_tokens = 10, stop_ids = Int[], speculative = TileInfer.NgramProposer())
            c = TileInfer.PagedKVCache(ref; max_tokens = 128)
            @test out == TileInfer.generate!(TileInfer.ModelRunner(ref, c), TileInfer.new_sequence(c), prompt;
                                             max_new_tokens = 10, stop_ids = Int[])
        end
        GC.gc(); CUDA.reclaim()
    end
end
