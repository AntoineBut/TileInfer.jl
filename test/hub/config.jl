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

# Apertus 1.5: the decoder's settings nested under text_config (abridged).
const APERTUS15_CONFIG = Dict{String,Any}(
    "model_type" => "apertus1p5", "architectures" => ["Apertus1p5ForConditionalGeneration"],
    "tie_word_embeddings" => false, "image_token_id" => 131079,
    "vision_tokenizer_config" => Dict("model_type" => "apertus1p5_vision_tokenizer"),
    "text_config" => Dict{String,Any}(
        "model_type" => "apertus1p5_text", "vocab_size" => 266752, "output_vocab_size" => 131072,
        "hidden_size" => 4096, "intermediate_size" => 21504, "num_hidden_layers" => 32,
        "num_attention_heads" => 32, "num_key_value_heads" => 8, "rms_norm_eps" => 1e-5,
        "rope_parameters" => Dict("factor" => 32.0, "high_freq_factor" => 4.0, "low_freq_factor" => 1.0,
                                  "original_max_position_embeddings" => 8192, "rope_theta" => 4000000,
                                  "rope_type" => "llama3"),
        "max_position_embeddings" => 262144, "tie_word_embeddings" => false, "attention_bias" => false,
        "mlp_bias" => false, "qk_norm" => true, "post_norm" => false, "hidden_act" => "xielu",
        "dtype" => "bfloat16", "bos_token_id" => 1, "eos_token_id" => [2, 68, 72]))

@testset "config" begin
    q = TileInfer.parse_config(QWEN3_CONFIG)
    @test q.head_dim == 128          # explicit, ≠ hidden ÷ heads = 64
    @test q.qk_norm
    @test q.tie_word_embeddings
    @test q.dtype == BFloat16
    @test q.eos_token_ids == [151645]

    l = TileInfer.parse_config(LLAMA32_CONFIG)
    @test l.head_dim == 64           # derived
    @test !l.qk_norm
    @test l.rope_scaling == TileInfer.Llama3RopeScaling(32.0, 1.0, 4.0, 8192)
    @test l.eos_token_ids == [128001, 128008, 128009]

    @test_throws ErrorException TileInfer.parse_config(merge(QWEN3_CONFIG, Dict("model_type" => "qwen2")))
    @test_throws ErrorException TileInfer.parse_config(merge(QWEN3_CONFIG, Dict("attention_bias" => true)))

    # transformers ≥ 5 layout
    v5 = merge(filter(kv -> !(first(kv) in ("rope_theta", "rope_scaling", "torch_dtype")), QWEN3_CONFIG),
               Dict("rope_parameters" => Dict("rope_theta" => 1e6, "rope_type" => "default"),
                    "dtype" => "float16"))
    @test TileInfer.parse_config(v5).rope_theta == 1e6
    @test TileInfer.parse_config(v5).dtype == Float16

    a = TileInfer.parse_config(APERTUS15_CONFIG)
    @test a.model_type == "apertus1p5"
    @test a.vocab_size == 131072     # text rows only
    @test a.hidden_act == :xielu && a.qk_norm
    @test (a.num_layers, a.num_heads, a.num_kv_heads, a.head_dim) == (32, 32, 8, 128)
    @test a.rope_theta == 4e6 && a.rope_scaling == TileInfer.Llama3RopeScaling(32.0, 1.0, 4.0, 8192)
    @test a.eos_token_ids == [2, 68, 72]
    @test q.hidden_act == :silu

    # Apertus 1 (Apertus-8B-Instruct-2509): flat, transformers 4 layout.
    v1 = merge(APERTUS15_CONFIG["text_config"], Dict("model_type" => "apertus", "vocab_size" => 131072,
               "rope_theta" => 12000000, "max_position_embeddings" => 65536, "eos_token_id" => 68,
               "rope_scaling" => Dict("factor" => 8.0, "high_freq_factor" => 4.0, "low_freq_factor" => 1.0,
                                      "original_max_position_embeddings" => 8192, "rope_type" => "llama3")))
    delete!(v1, "rope_parameters"); delete!(v1, "output_vocab_size")
    a1 = TileInfer.parse_config(v1)
    @test a1.vocab_size == 131072 && a1.hidden_act == :xielu && a1.qk_norm
    @test a1.rope_theta == 1.2e7 && a1.rope_scaling.factor == 8.0
    @test_throws ErrorException TileInfer.parse_config(merge(v1, Dict("post_norm" => true)))
    @test_throws ErrorException TileInfer.parse_config(merge(v1, Dict("hidden_act" => "gelu")))

    @test_throws ErrorException TileInfer.parse_config(merge(QWEN3_CONFIG, Dict("model_type" => "gemma3")))
    @test_throws ErrorException TileInfer.parse_config(merge(QWEN3_CONFIG,
        Dict("use_sliding_window" => true, "sliding_window" => 4096)))

    dir = test_model_dir()
    if dir !== nothing
        loaded = TileInfer.load_config(dir)
        @test all(getfield(loaded, f) == getfield(q, f) for f in fieldnames(TileInfer.ModelConfig))
    end
end
