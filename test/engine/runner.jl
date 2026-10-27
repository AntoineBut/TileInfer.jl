# ModelRunner: steps replayed from CUDA graphs against eager forward! steps on a second
# cache. Both run the same kernels on the same inputs, so the logits must be bit-identical.

if test_model_dir() !== nothing
    @testset "model runner" begin
        m = TileInfer.load_model(TEST_MODEL; backend = TileBackend(), max_len = 256)
        prompt = collect(1000:1030)
        # 16-position pages, so the decode steps below cross several page boundaries.
        new_cache() = TileInfer.PagedKVCache(m; max_tokens = 256, page_size = 16)
        cache, eager_cache = new_cache(), new_cache()
        runner = TileInfer.ModelRunner(m, cache)
        seq, eager_seq = TileInfer.new_sequence(cache), TileInfer.new_sequence(eager_cache)

        @testset "graph replay matches eager steps" begin
            logits = Array(TileInfer.forward!(runner, seq, prompt))                   # prefill, eager
            @test logits == Array(TileInfer.forward!(m, eager_cache, eager_seq, prompt))
            same = true
            for _ in 1:40
                id = argmax(vec(logits)) - 1
                logits = Array(TileInfer.forward!(runner, seq, [id]))
                same &= logits == Array(TileInfer.forward!(m, eager_cache, eager_seq, [id]))
            end
            @test same
            @test seq.tokens == eager_seq.tokens
            # Verify-sized steps get a graph of their own; the second one replays it.
            for ids in ([5, 6, 7], [8, 9, 10])
                @test Array(TileInfer.forward!(runner, seq, ids)) ==
                      Array(TileInfer.forward!(m, eager_cache, eager_seq, ids))
            end
            @test TileInfer.num_graphs(runner) == 2
        end

        @testset "a decode step allocates no GPU memory" begin
            b = runner.steps[1].buffers
            TileInfer.prepare_step!(b, m, cache, seq, [42])
            TileInfer.run_step!(m, cache, b)
            @test CUDA.@allocated(TileInfer.run_step!(m, cache, b)) == 0
        end

        @testset "generation with and without graphs" begin
            function gen(graphs)
                c = new_cache()
                return TileInfer.generate!(TileInfer.ModelRunner(m, c; graphs), TileInfer.new_sequence(c), prompt;
                                           max_new_tokens = 30, stop_ids = Int[])
            end
            @test gen(true) == gen(false)                         # pipelined vs step by step
        end

        @testset "pipelined greedy decoding stops like the plain loop" begin
            function run(graphs, stop_ids)
                c = new_cache()
                s = TileInfer.new_sequence(c)
                out = TileInfer.generate!(TileInfer.ModelRunner(m, c; graphs), s, prompt;
                                          max_new_tokens = 12, stop_ids)
                return out, copy(s.tokens), copy(s.block_table)
            end
            full, = run(false, Int[])
            for stop_ids in ([full[6]], Int[])                         # stop token, budget
                @test run(true, stop_ids) == run(false, stop_ids)
            end
        end

        @testset "reference backend steps run eagerly" begin
            ref = TileInfer.DecoderLM(m.config, m.weights, m.rope, RefBackend())
            ref_cache() = TileInfer.PagedKVCache(ref; max_tokens = 64)
            @test_throws ArgumentError TileInfer.ModelRunner(ref, ref_cache(); graphs = true)
            c1, c2 = ref_cache(), ref_cache()
            r = TileInfer.ModelRunner(ref, c1)
            @test Array(TileInfer.forward!(r, TileInfer.new_sequence(c1), prompt[1:5])) ==
                  Array(TileInfer.forward!(ref, c2, TileInfer.new_sequence(c2), prompt[1:5]))
            @test TileInfer.num_graphs(r) == 0
        end
        runner = nothing; GC.gc(); CUDA.reclaim()
    end
end
