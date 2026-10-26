# Compares against golden ids from Hugging Face `tokenizers`
# (scripts/gen_fixtures.py tokenizer <model_dir> qwen3).

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
end
