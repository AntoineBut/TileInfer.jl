dir = test_model_dir()
if dir !== nothing
    @testset "chat session" begin
        m = TileInfer.load_model(TEST_MODEL; T = Float32, max_len = 512)
        tok = TileInfer.load_tokenizer(dir)
        s = TileInfer.ChatSession(m, tok; enable_thinking = false)   # greedy

        reply1, stats1 = TileInfer.reply!(s, "Name a primary color."; max_new_tokens = 16)
        @test stats1.reused == 0
        reply2, stats2 = TileInfer.reply!(s, "And another one?"; max_new_tokens = 16)
        @test length(s.messages) == 4 && s.messages[end] == TileInfer.ChatMessage("assistant", reply2)
        # Everything before the first reply is reused. (The template drops the empty think
        # block of earlier turns, so the cached reply itself is recomputed.)
        @test stats2.reused > 0

        # The second reply is what a fresh run on the full re-rendered prompt produces.
        prompt = TileInfer.encode(tok, TileInfer.chat_prompt("qwen3", s.messages[1:3]; enable_thinking = false))
        @test stats2.reused + stats2.prefilled == length(prompt)
        @test TileInfer.decode(tok, TileInfer.generate(m, prompt; max_new_tokens = 16)) == reply2

        TileInfer.reset!(s)
        @test isempty(s.messages) && length(s.seq) == 0
        @test TileInfer.num_free_pages(s.cache.blocks) == TileInfer.num_pages(s.cache)

        # A conversation that outgrows the cache fails without changing the history. (Qwen
        # tokenizes digits one by one, so this message alone is ~290 tokens.)
        small = TileInfer.ChatSession(m, tok; enable_thinking = false, max_tokens = 64)
        TileInfer.reply!(small, "Hi"; max_new_tokens = 4)
        @test_throws TileInfer.OutOfPagesError TileInfer.reply!(small, join(1:100, " "); max_new_tokens = 4)
        @test length(small.messages) == 2

        @testset "chat_loop" begin
            input = IOBuffer("Hi\n\n/reset\nHello\n/exit\nnot read\n")
            out = IOBuffer()
            redirect_stderr(devnull) do
                TileInfer.chat_loop(s, input, out; max_new_tokens = 4)
            end
            text = String(take!(out))
            @test occursin("[conversation cleared]", text)
            @test length(s.messages) == 2                  # only "Hello" after the reset
            @test !eof(input)                              # stopped at /exit
        end
        m = nothing; GC.gc(); CUDA.reclaim()
    end
end
