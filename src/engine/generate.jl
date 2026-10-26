# Single-sequence generation loop on the reference path: prefill the prompt in one
# forward pass, then decode one token per step until a stop token or the token budget.

"""
    generate(model, prompt_ids; max_new_tokens = 256, sampling = SamplingParams(),
             stop_ids = model.config.eos_token_ids, rng = Random.default_rng(),
             on_token = nothing) -> Vector{Int}

Generate a continuation of the 0-based `prompt_ids`. Returns the new token ids, without
the stop token. `on_token(id)` is called after each new token, for streaming.
"""
function generate(model::DecoderLM, prompt_ids::AbstractVector{<:Integer};
                  max_new_tokens::Integer = 256, sampling::SamplingParams = SamplingParams(),
                  stop_ids = model.config.eos_token_ids, rng::AbstractRNG = Random.default_rng(),
                  on_token = nothing)
    isempty(prompt_ids) && throw(ArgumentError("empty prompt"))
    total = length(prompt_ids) + max_new_tokens
    total <= max_length(model) ||
        throw(ArgumentError("prompt + max_new_tokens = $total exceeds the model's max length $(max_length(model))"))
    cache = KVCache(model, total)
    generated = Int[]
    logits = forward!(model, cache, prompt_ids)
    for _ in 1:max_new_tokens
        id = sample(vec(logits), sampling, rng)
        id in stop_ids && break
        push!(generated, id)
        on_token === nothing || on_token(id)
        length(generated) == max_new_tokens && break
        logits = forward!(model, cache, [id])
    end
    return generated
end

"""
    generation_defaults(model_dir, config) -> (SamplingParams, stop_ids)

The model authors' recommended sampling settings and stop tokens, from
`generation_config.json` when the model ships one; greedy and the config's EOS otherwise.
"""
function generation_defaults(dir::AbstractString, cfg::ModelConfig)
    path = joinpath(dir, "generation_config.json")
    isfile(path) || return SamplingParams(), cfg.eos_token_ids
    g = JSON.parsefile(path)
    eos = get(g, "eos_token_id", nothing)
    stop_ids = unique(vcat(cfg.eos_token_ids, eos === nothing ? Int[] : eos))
    get(g, "do_sample", false) || return SamplingParams(), stop_ids
    sampling = SamplingParams(temperature = get(g, "temperature", 1.0), top_k = get(g, "top_k", 0),
                              top_p = get(g, "top_p", 1.0))
    return sampling, stop_ids
end

"""
    TextStream(tokenizer, io)

Prints generated tokens to `io` as they arrive. Bytes of a UTF-8 character split across
tokens are held back until the character is complete.
"""
mutable struct TextStream
    tokenizer::BPETokenizer
    io::IO
    ids::Vector{Int}
    printed::Int          # bytes of decode(ids) already written
end

TextStream(tok::BPETokenizer, io::IO = stdout) = TextStream(tok, io, Int[], 0)

function (s::TextStream)(id::Integer)
    push!(s.ids, id)
    text = decode(s.tokenizer, s.ids)
    isvalid(text) || return
    write(s.io, codeunits(text)[s.printed+1:end])
    flush(s.io)
    s.printed = ncodeunits(text)
    return
end
