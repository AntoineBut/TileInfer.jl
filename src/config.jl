# Model hyperparameters, parsed from a Hugging Face config.json. One struct covers the
# Llama-style decoder family (and Apertus, whose MLP is ungated); per-family differences
# are flags.

using JSON: JSON
using BFloat16s: BFloat16

const SUPPORTED_MODEL_TYPES = ("llama", "mistral", "qwen3", "apertus", "apertus1p5")

"Llama 3.x frequency-dependent RoPE scaling (`rope_type = \"llama3\"`)."
struct Llama3RopeScaling
    factor::Float64
    low_freq_factor::Float64
    high_freq_factor::Float64
    original_max_position_embeddings::Int
end

"""
    ModelConfig

Hyperparameters of a Llama-style decoder. Build from a model directory with
[`load_config`](@ref).
"""
Base.@kwdef struct ModelConfig
    model_type::String
    vocab_size::Int            # rows of lm_head; Apertus 1.5's embedding has more (image/audio codes)
    hidden_size::Int
    intermediate_size::Int
    num_layers::Int
    num_heads::Int
    num_kv_heads::Int
    head_dim::Int              # Qwen3 sets it explicitly; it is not always hidden ÷ heads
    rms_norm_eps::Float64
    rope_theta::Float64
    rope_scaling::Union{Nothing,Llama3RopeScaling}
    max_position_embeddings::Int
    tie_word_embeddings::Bool
    qk_norm::Bool              # Qwen3, Apertus: RMSNorm over head_dim on q and k before RoPE
    hidden_act::Symbol         # :silu (gated SwiGLU MLP) or :xielu (Apertus: ungated xIELU MLP)
    dtype::DataType            # checkpoint element type
    bos_token_id::Union{Nothing,Int}
    eos_token_ids::Vector{Int}
end

const DTYPES = Dict("bfloat16" => BFloat16, "float16" => Float16, "float32" => Float32)

"Read `config.json` from a model directory (see [`model_dir`](@ref))."
load_config(dir::AbstractString) = parse_config(JSON.parsefile(joinpath(dir, "config.json")))

function parse_config(c::AbstractDict)
    model_type = c["model_type"]
    model_type in SUPPORTED_MODEL_TYPES ||
        error("unsupported model_type \"$model_type\"; supported: $(join(SUPPORTED_MODEL_TYPES, ", "))")
    # Apertus 1.5 is multimodal: the decoder's settings are nested, and its image and audio
    # tokenizers (which TileInfer does not run) are configured beside them.
    model_type == "apertus1p5" && (c = c["text_config"])
    hidden_act = get(c, "hidden_act", "silu")
    hidden_act in ("silu", "xielu") || error("unsupported hidden_act \"$hidden_act\"")
    get(c, "post_norm", false) && error("Apertus post-norm layers are not supported")
    sliding = get(c, "use_sliding_window", model_type == "mistral") &&
              get(c, "sliding_window", nothing) !== nothing
    sliding && error("sliding-window attention is not supported yet")
    get(c, "attention_bias", false) && error("attention biases are not supported")
    get(c, "mlp_bias", false) && error("MLP biases are not supported")

    # transformers ≥ 5 nests RoPE settings under `rope_parameters`.
    rope = get(c, "rope_parameters", nothing)
    rope_theta = rope === nothing ? c["rope_theta"] : rope["rope_theta"]
    scaling = rope === nothing ? get(c, "rope_scaling", nothing) : rope

    num_heads = c["num_attention_heads"]
    eos = get(c, "eos_token_id", nothing)
    dtype_name = get(c, "dtype", get(c, "torch_dtype", "bfloat16"))
    haskey(DTYPES, dtype_name) || error("unsupported checkpoint dtype \"$dtype_name\"")

    return ModelConfig(;
        model_type,
        # Apertus 1.5 embeds image and audio codes but only predicts text tokens.
        vocab_size = something(get(c, "output_vocab_size", nothing), c["vocab_size"]),
        hidden_size = c["hidden_size"],
        intermediate_size = c["intermediate_size"],
        num_layers = c["num_hidden_layers"],
        num_heads,
        num_kv_heads = get(c, "num_key_value_heads", num_heads),
        head_dim = something(get(c, "head_dim", nothing), c["hidden_size"] ÷ num_heads),
        rms_norm_eps = c["rms_norm_eps"],
        rope_theta,
        rope_scaling = parse_rope_scaling(scaling),
        max_position_embeddings = c["max_position_embeddings"],
        tie_word_embeddings = get(c, "tie_word_embeddings", false),
        qk_norm = get(c, "qk_norm", model_type == "qwen3"),
        hidden_act = Symbol(hidden_act),
        dtype = DTYPES[dtype_name],
        bos_token_id = get(c, "bos_token_id", nothing),
        eos_token_ids = eos === nothing ? Int[] : eos isa Integer ? [eos] : collect(Int, eos),
    )
end

function parse_rope_scaling(s)
    s === nothing && return nothing
    type = get(s, "rope_type", get(s, "type", "default"))
    type == "default" && return nothing
    type == "llama3" || error("unsupported RoPE scaling \"$type\"")
    return Llama3RopeScaling(s["factor"], s["low_freq_factor"], s["high_freq_factor"],
                             s["original_max_position_embeddings"])
end
