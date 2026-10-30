# DFlash: the drafter against a transcription of z-lab's reference (scripts/gen_dflash_fixtures.py,
# Float32 on the CPU, from the same BFloat16-rounded target states), and speculative
# generation with it. Needs Qwen/Qwen3-4B and the drafter; skipped when they are not pulled.

using Random: Xoshiro

const DFLASH_TARGET = "Qwen/Qwen3-4B"
const DFLASH_DRAFTER = "z-lab/Qwen3-4B-DFlash-b16"

pulled(name) = try TileInfer.model_dir(name); true catch; false end

@testset "target layers" begin
    # dflash.py's build_target_layer_ids when the config has none, plus one.
    @test TileInfer.dflash_layers(Dict("num_hidden_layers" => 5), 36) == [2, 10, 18, 26, 34]
    @test TileInfer.dflash_layers(Dict("num_hidden_layers" => 1), 36) == [19]
    @test TileInfer.dflash_layers(Dict("num_hidden_layers" => 5, "dflash_config" => Dict("target_layer_ids" => [0, 3])), 36) == [1, 4]
end

if !(pulled(DFLASH_TARGET) && pulled(DFLASH_DRAFTER))
    @info "$DFLASH_TARGET or $DFLASH_DRAFTER not pulled; skipping the DFlash tests"
else
    GC.gc(); CUDA.reclaim()                         # the 4B target needs most of the GPU
    meta = TileInfer.JSON.parsefile(fixture("dflash", "qwen3-4b-dflash-b16.json"))
    raw = reinterpret(BFloat16, read(fixture("dflash", "qwen3-4b-dflash-b16.bin")))
    tdir = TileInfer.model_dir(DFLASH_TARGET)
    tcfg = TileInfer.load_config(tdir)
    H = tcfg.hidden_size

    # In functions, so that each model is freed before the next one loads (the locals of
    # top-level code stay rooted until the whole block ends).
    function check_drafter(be)
        embed = TileInfer.upload(Float32, TileInfer.Checkpoint(tdir), "model.embed_tokens.weight")
        d = TileInfer.load_dflash(DFLASH_DRAFTER, tcfg, embed, embed, be; max_len = 256)   # tied lm_head
        @test d.aux_layers == meta["layers"] == [2, 10, 18, 26, 34]
        @test d.block_size == meta["block_size"] && d.mask_id == meta["mask_id"]
        p = TileInfer.DFlashProposer(d; max_tokens = 256)
        @test p.k == 15
        # Twice: the second pass starts the sequence over and replays the captured graphs.
        for _ in 1:2
            offset = 0
            for block in meta["blocks"]
                c = block["ctx_len"]
                states = Float32.(reshape(raw[offset*5H+1:(offset+c)*5H], 5H, c))
                offset += c
                TileInfer.draft!(p, Int.(block["context"]), block["ctx_start"], CuArray(states))
                logits = Array(p.logits)
                maxdiff = maximum(1:15) do j
                    ids = Int.(block["topk_ids"][j]) .+ 1
                    maximum(abs.(logits[ids, j] .- Float64.(block["topk_logits"][j])))
                end
                @test maxdiff < 2e-3                  # logits of ~±30: ~1e-4 relative
                @test Int.(p.mirror_host) == block["drafts"]
                @test length(p.seq) == length(block["context"]) - 1     # the block is dropped again
            end
        end
        be isa TileBackend && @test length(p.steps) == 2                  # contexts of 6 and 1 positions
    end
    @testset "drafter vs reference ($(nameof(typeof(be))))" for be in (TileBackend(), RefBackend())
        check_drafter(be)
        GC.gc(); CUDA.reclaim()
    end

    function check_generation()
        m = TileInfer.load_model(DFLASH_TARGET; backend = TileBackend(), max_len = 512)
        d = TileInfer.load_dflash(DFLASH_DRAFTER, m; max_len = 512)
        prompt = Int.(first(meta["blocks"])["context"][1:end-1])             # the fixture's chat prompt
        function gen(proposer; n = 40, stop_ids = Int[], sampling = TileInfer.SamplingParams(), stats = TileInfer.SpecStats())
            c = TileInfer.PagedKVCache(m; max_tokens = 512)
            s = TileInfer.new_sequence(c)
            kw = proposer === nothing ? (; pipeline = false) : (; speculative = proposer, spec_stats = stats)
            out = TileInfer.generate!(TileInfer.ModelRunner(m, c), s, prompt; max_new_tokens = n, stop_ids, sampling,
                                      rng = Xoshiro(1), kw...)
            return out, copy(s.tokens)
        end
        # In BFloat16, verify steps of 16 tokens round differently from decode steps, so the two
        # can part at a near-tie; with this prompt they do not within 40 tokens (as in the
        # reference, the prime list is drafted whole).
        plain = gen(nothing)
        for k in (15, 4)
            stats = TileInfer.SpecStats()
            @test gen(TileInfer.DFlashProposer(d; k, max_tokens = 512); stats) == plain
            @test stats.steps < 20
        end
        p = TileInfer.DFlashProposer(d; max_tokens = 512)
        stop = plain[1][9]
        @test gen(p; stop_ids = [stop]) == gen(nothing; stop_ids = [stop])
        @test all(gen(p; n) == gen(nothing; n) for n in (1, 2, 17))
        sampled = TileInfer.SamplingParams(temperature = 0.7, top_k = 20, top_p = 0.8)
        @test gen(p; sampling = sampled) == gen(nothing; sampling = sampled)
        # Sampled drafts (p/q rule): other tokens than plain decoding's, and the cache consistent
        # with them.
        ps = TileInfer.DFlashProposer(d; sample = true, max_tokens = 512)
        stats = TileInfer.SpecStats()
        out, tokens = gen(ps; sampling = sampled, stats)
        @test length(out) == 40 && tokens == [prompt; out[1:end-1]]
        @test 0.2 < TileInfer.acceptance_rate(stats) < 1
        @test gen(ps; sampling = sampled) == (out, tokens)                   # reproducible
        @test gen(ps) == plain                                               # greedy target
        @test all(sum(Array(ps.dists.probs); dims = 1) .≈ 1)                 # q of every block position

        @testset "continuing a sequence, as chat turns do" begin
            function turns(proposer)
                c = TileInfer.PagedKVCache(m; max_tokens = 512)
                r, s = TileInfer.ModelRunner(m, c), TileInfer.new_sequence(c)
                out = Int[]
                for ids in (prompt, [1500, 1501, 1502], collect(2000:2011))
                    kw = proposer === nothing ? (; pipeline = false) : (; speculative = proposer)
                    append!(out, TileInfer.generate!(r, s, ids; max_new_tokens = 12, stop_ids = Int[], kw...))
                end
                return out, copy(s.tokens)
            end
            @test turns(p) == turns(nothing)
        end
    end
    @testset "speculative generation" begin
        check_generation()
        GC.gc(); CUDA.reclaim()
    end
end
