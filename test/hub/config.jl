const QWEN3_CONFIG = Dict{String,Any}(
    "model_type" => "qwen3", "vocab_size" => 151936, "hidden_size" => 1024,
    "intermediate_size" => 3072, "num_hidden_layers" => 28, "num_attention_heads" => 16,
    "num_key_value_heads" => 8, "head_dim" => 128, "rms_norm_eps" => 1e-6,
    "rope_theta" => 1000000, "rope_scaling" => nothing, "max_position_embeddings" => 40960,
    "tie_word_embeddings" => true, "attention_bias" => false, "torch_dtype" => "bfloat16",
    "hidden_act" => "silu", "use_sliding_window" => false, "sliding_window" => nothing,
    "bos_token_id" => 151643, "eos_token_id" => 151645)

const LLAMA32_CONFIG = Dict{String,Any}(
    "model_type" => "llama", "vocab_size" => 128256, "hidden_size" => 2048,
    "intermediate_size" => 8192, "num_hidden_layers" => 16, "num_attention_heads" => 32,
    "num_key_value_heads" => 8, "rms_norm_eps" => 1e-5, "rope_theta" => 500000.0,
    "rope_scaling" => Dict("factor" => 32.0, "high_freq_factor" => 4.0, "low_freq_factor" => 1.0,
                           "original_max_position_embeddings" => 8192, "rope_type" => "llama3"),
    "max_position_embeddings" => 131072, "tie_word_embeddings" => true, "mlp_bias" => false,
    "torch_dtype" => "bfloat16", "hidden_act" => "silu", "bos_token_id" => 128000,
    "eos_token_id" => [128001, 128008, 128009])

@testset "config" begin
    q = TileInfer.parse_config(QWEN3_CONFIG)
    @test q.head_dim == 128          # explicit, ≠ hidden ÷ heads = 64
    @test q.qk_norm && !q.qkv_bias
    @test q.tie_word_embeddings
    @test q.dtype == BFloat16
    @test q.eos_token_ids == [151645]

    l = TileInfer.parse_config(LLAMA32_CONFIG)
    @test l.head_dim == 64           # derived
    @test !l.qk_norm && !l.qkv_bias
    @test l.rope_scaling == TileInfer.Llama3RopeScaling(32.0, 1.0, 4.0, 8192)
    @test l.eos_token_ids == [128001, 128008, 128009]

    qwen2 = TileInfer.parse_config(merge(QWEN3_CONFIG, Dict("model_type" => "qwen2")))
    @test qwen2.qkv_bias && !qwen2.qk_norm

    # transformers ≥ 5 layout
    v5 = merge(filter(kv -> !(first(kv) in ("rope_theta", "rope_scaling", "torch_dtype")), QWEN3_CONFIG),
               Dict("rope_parameters" => Dict("rope_theta" => 1e6, "rope_type" => "default"),
                    "dtype" => "float16"))
    @test TileInfer.parse_config(v5).rope_theta == 1e6
    @test TileInfer.parse_config(v5).dtype == Float16

    @test_throws ErrorException TileInfer.parse_config(merge(QWEN3_CONFIG, Dict("model_type" => "gemma3")))
    @test_throws ErrorException TileInfer.parse_config(merge(QWEN3_CONFIG,
        Dict("use_sliding_window" => true, "sliding_window" => 4096)))

    dir = test_model_dir()
    if dir !== nothing
        loaded = TileInfer.load_config(dir)
        @test all(getfield(loaded, f) == getfield(q, f) for f in fieldnames(TileInfer.ModelConfig))
    end
end
