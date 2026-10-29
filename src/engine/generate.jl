# Single-sequence generation loop: prefill the new tokens in one forward pass, then decode
# one token per step until a stop token or the token budget. Steps go through a
# ModelRunner, so decode steps replay a CUDA graph on backends that support it.

"""
    generate!(runner, seq, ids; max_new_tokens = 256, sampling = SamplingParams(),
              stop_ids = model.config.eos_token_ids, rng = Random.default_rng(),
              on_token = nothing, pipeline = true, speculative = nothing,
              spec_stats = SpecStats()) -> Vector{Int}
    generate!(model, cache, seq, ids; kwargs...)

Continue `seq` with the 0-based token `ids` and generate up to `max_new_tokens` tokens.
Returns the new token ids, without the stop token. `on_token(id)` is called after each
new token, for streaming. Afterwards `seq` holds `ids` and every generated token except
the last one, which was sampled but never run through the model.

The tokens are a function of the logits and one seed drawn from `rng` (see
[`sample_host`](@ref)), however they are sampled: on the device when the sampling is
[`on_device`](@ref), pipelined ([`generate_pipelined!`](@ref)) on a runner with CUDA graphs
unless `pipeline = false`; otherwise on the host, step by step.

With a proposer as `speculative` (such as [`NgramProposer`](@ref)), decoding checks
its drafts several tokens per step ([`generate_speculative!`](@ref)) and counts them in
`spec_stats`; the sampling must be [`on_device`](@ref).
"""
function generate!(r::ModelRunner, seq::Sequence, ids::AbstractVector{<:Integer};
                   max_new_tokens::Integer = 256, sampling::SamplingParams = SamplingParams(),
                   stop_ids = r.model.config.eos_token_ids, rng::AbstractRNG = Random.default_rng(),
                   on_token = nothing, pipeline::Bool = true, speculative = nothing,
                   spec_stats::SpecStats = SpecStats())
    isempty(ids) && throw(ArgumentError("no tokens to process"))
    seed = rand(rng, UInt32)
    device = on_device(sampling)
    speculative === nothing || device ||
        throw(ArgumentError("speculative decoding needs sampling on the device (top_k = $(sampling.top_k))"))
    if device
        reset_sampler!(r.sampler, sampling, seed)
        speculative === nothing ||
            return generate_speculative!(r, seq, ids, speculative; max_new_tokens, stop_ids, on_token, stats = spec_stats)
        pipeline && r.graphs && return generate_pipelined!(r, seq, ids; max_new_tokens, stop_ids, on_token)
    end
    generated = Int[]
    logits = forward!(r, seq, ids)
    for counter in 0:max_new_tokens-1
        id = NVTX.@range "sample" if device
            Int(Array(sample_token!(r.model.backend, r.token, r.token, logits, r.sampler))[1])
        else
            sample_host(vec(Array(logits)), sampling, sampling_uniform(seed, UInt32(counter)))
        end
        id in stop_ids && break
        push!(generated, id)
        on_token === nothing || on_token(id)
        length(generated) == max_new_tokens && break
        logits = forward!(r, seq, [id])
    end
    return generated
end

generate!(model::DecoderLM, cache::PagedKVCache, seq::Sequence, ids::AbstractVector{<:Integer}; kwargs...) =
    generate!(ModelRunner(model, cache), seq, ids; kwargs...)

"""
    generate(model, prompt_ids; kwargs...) -> Vector{Int}

One-shot [`generate!`](@ref) in a fresh cache sized for the prompt and `max_new_tokens`.
"""
function generate(model::DecoderLM, prompt_ids::AbstractVector{<:Integer};
                  max_new_tokens::Integer = 256, kwargs...)
    total = length(prompt_ids) + max_new_tokens
    total <= max_length(model) ||
        throw(ArgumentError("prompt + max_new_tokens = $total exceeds the model's max length $(max_length(model))"))
    cache = PagedKVCache(model; max_tokens = total)
    return generate!(model, cache, new_sequence(cache), prompt_ids; max_new_tokens, kwargs...)
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
