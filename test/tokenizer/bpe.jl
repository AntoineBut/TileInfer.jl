# Compares against golden ids from Hugging Face `tokenizers`
# (scripts/gen_fixtures.py tokenizer <model_dir> qwen3, and apertus-1.5).

@testset "tokenizer" begin
    @testset "byte-level alphabet" begin
        @test length(unique(TileInfer.BYTE_TO_CHAR)) == 256
        @test TileInfer.byte_level(" hi\n") == "Ġhi" * "Ċ"
        @test all(TileInfer.CHAR_TO_BYTE[TileInfer.BYTE_TO_CHAR[b+1]] == b for b in 0x00:0xff)
    end

    dir = test_model_dir()
    if dir !== nothing
        tok = TileInfer.load_tokenizer(dir)
        cases = TileInfer.JSON.parsefile(fixture("tokenizer", "qwen3.json"))["cases"]
        @testset "matches HF tokenizers" begin
            for case in cases
                text = case["text"]
                @test TileInfer.encode(tok, text) == case["ids"]
                @test TileInfer.encode(tok, text; add_special_tokens = false) == case["ids_no_special"]
            end
        end
        @testset "decode round trip" begin
            for case in cases
                text = case["text"]
                # NFC normalization is not reversible; compare against the normalized text.
                @test TileInfer.decode(tok, case["ids"]) == TileInfer.Unicode.normalize(text, :NFC)
            end
            im_end = tok.added["<|im_end|>"]
            @test TileInfer.decode(tok, [im_end]; skip_special_tokens = true) == ""
        end
    end

    @testset "added tokens" begin
        added = TileInfer.AddedTokens(Dict("<a>" => 1, "<ab>" => 2, "<|é|>" => 3))
        pieces = []
        TileInfer.foreach_piece((p, id) -> push!(pieces, (String(p), id)), added, "x<ab><a>é<|é|><|é|")
        @test pieces == [("x", nothing), ("<ab>", 2), ("<a>", 1), ("é", nothing), ("<|é|>", 3), ("<|é|", nothing)]
    end

    # Apertus 1.5: 136k added tokens, and a normalizer that rewrites reasoning markers
    # (<think>, <channel|>, …) before matching its `normalized` added tokens.
    dir = test_model_dir(APERTUS_TEST_MODEL)
    if dir !== nothing
        tok = TileInfer.load_tokenizer(dir)
        @test length(tok.added) == 136366
        @testset "matches HF tokenizers ($APERTUS_TEST_MODEL)" begin
            for case in TileInfer.JSON.parsefile(fixture("tokenizer", "apertus-1.5.json"))["cases"]
                text = case["text"]
                @test TileInfer.encode(tok, text) == case["ids"]
                @test TileInfer.encode(tok, text; add_special_tokens = false) == case["ids_no_special"]
            end
        end
    end
end
