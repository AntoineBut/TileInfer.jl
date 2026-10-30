@testset "store" begin
    @testset "wanted files" begin
        @test TileInfer.wanted_file("model.safetensors")
        @test TileInfer.wanted_file("model-00001-of-00002.safetensors")
        @test TileInfer.wanted_file("model.safetensors.index.json")
        @test TileInfer.wanted_file("tokenizer.json")
        @test TileInfer.wanted_file("merges.txt")
        @test TileInfer.wanted_file("chat_template.jinja")
        @test !TileInfer.wanted_file("pytorch_model.bin")
        @test !TileInfer.wanted_file("README.md")
        @test !TileInfer.wanted_file("onnx/model.safetensors")
    end

    @testset "skipped shards" begin
        weight_map = Dict("lm_head.weight" => "model-apertus-model-00003-of-00004.safetensors",
                          "model.language_model.norm.weight" => "model-apertus-model-00004-of-00004.safetensors",
                          "model.vision_tokenizer.encoder.conv_in.weight" => "model-vision_tokenizer-model.safetensors",
                          "model.audio_tokenizer.encoder.conv.weight" => "model-wavtokenizer-model.safetensors",
                          # a skipped tensor in a shard that is needed anyway
                          "model.vision_tokenizer.quantize.embedding.weight" => "model-apertus-model-00004-of-00004.safetensors")
        @test TileInfer.skipped_shards(weight_map) ==
              Set(["model-vision_tokenizer-model.safetensors", "model-wavtokenizer-model.safetensors"])
        @test isempty(TileInfer.skipped_shards(Dict("model.norm.weight" => "model.safetensors")))
    end

    @testset "local directory" begin
        mktempdir() do dir
            @test TileInfer.model_dir(dir) == abspath(dir)
        end
        @test_throws ErrorException TileInfer.model_dir("nobody/not-a-pulled-model")
    end

    dir = test_model_dir()
    if dir !== nothing
        @testset "pulled model" begin
            models = TileInfer.list_models()
            m = only(filter(m -> m.repo_id == TEST_MODEL && m.revision == "main", models))
            @test "config.json" in m.files && "tokenizer.json" in m.files
            @test m.bytes > 10^9
        end
    end
end
