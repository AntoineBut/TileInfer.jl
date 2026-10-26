@testset "store" begin
    @testset "wanted files" begin
        @test TileInfer.wanted_file("model.safetensors")
        @test TileInfer.wanted_file("model-00001-of-00002.safetensors")
        @test TileInfer.wanted_file("model.safetensors.index.json")
        @test TileInfer.wanted_file("tokenizer.json")
        @test TileInfer.wanted_file("merges.txt")
        @test !TileInfer.wanted_file("pytorch_model.bin")
        @test !TileInfer.wanted_file("README.md")
        @test !TileInfer.wanted_file("onnx/model.safetensors")
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
