# Runs the forward steps of one model on one KV cache. A step of a few tokens (decode, and
# speculative verification later) is launched as a CUDA graph: the first step of each token
# count runs eagerly, which compiles its kernels, and is then captured; later steps of that
# count only upload their inputs into the step's static buffers and replay the graph, one
# launch instead of ~540. Longer steps (prefill) run eagerly with buffers of their own.
#
# A captured graph holds the device addresses of the weights, the cache pools and the static
# buffers, and the kernels read every per-step value (ids, positions, slots, block table)
# from those buffers, so it stays valid as long as the runner does.

"Steps of up to this many tokens are captured in CUDA graphs."
const MAX_GRAPH_TOKENS = 8

mutable struct CapturedStep{T}
    buffers::StepBuffers{T}
    graph::Union{Nothing,CUDA.CuGraphExec}
end

"""
    ModelRunner(model, cache; graphs = supports_graphs(model.backend))

Forward steps of `model` on `cache`, with CUDA graphs for steps of up to
$MAX_GRAPH_TOKENS tokens when `graphs`. Run steps with `forward!(runner, seq, ids)`.
"""
mutable struct ModelRunner{T,M<:DecoderLM{T},C<:PagedKVCache{T}}
    const model::M
    const cache::C
    const graphs::Bool
    const steps::Dict{Int,CapturedStep{T}}
    const verify_steps::Dict{Tuple{Int,UInt},Any}   # VerifyStep{T} by token count and drafts' q, see `verify!`
    const sampler::TokenSampler      # device sampling state, see `sample_token!`
    const token::CuVector{Int32}     # the token sampled by a step-by-step generation
    pipeline::Any                    # DecodePipeline, created on first use
    const aux_layers::Vector{Int}    # layers whose outputs steps keep, see `enable_aux!`
    last_step::Any                   # StepBuffers of the latest forward or verify step,
    last_start::Int                  # and the position of its first token
end

function ModelRunner(model::DecoderLM{T}, cache::PagedKVCache{T};
                     graphs::Bool = supports_graphs(model.backend)) where {T}
    graphs && !supports_graphs(model.backend) &&
        throw(ArgumentError("$(nameof(typeof(model.backend))) steps cannot be captured in CUDA graphs"))
    return ModelRunner(model, cache, graphs, Dict{Int,CapturedStep{T}}(), Dict{Tuple{Int,UInt},Any}(), TokenSampler(model.config.vocab_size),
                       CuVector{Int32}(undef, 1), nothing, Int[], nothing, 0)
end

"""
    enable_aux!(runner, layers)

Make every later step keep the residual stream after `layers` (1-based) in its buffers'
`aux`, for an EAGLE drafter; the latest step's buffers are `runner.last_step`. Drops the
captured steps when the layers change.
"""
function enable_aux!(r::ModelRunner, layers)
    r.aux_layers == layers && return r
    copy!(r.aux_layers, layers)
    empty!(r.steps)
    empty!(r.verify_steps)
    r.pipeline = nothing
    r.last_step = nothing
    return r
end

"""
    forward!(runner, seq, ids) -> CuMatrix{Float32}

[`forward!`](@ref) through `runner`: the logits (vocab, 1) of the last token. For short
steps they live in the step's static buffers and are overwritten by the next step of the
same length, so read them before running another step.
"""
function forward!(r::ModelRunner, seq::Sequence, ids::AbstractVector{<:Integer})
    n = length(ids)
    n > 0 || throw(ArgumentError("forward! needs at least one token"))
    r.last_start = length(seq)
    if n > MAX_GRAPH_TOKENS
        NVTX.@range "prefill" begin
            b = StepBuffers(r.model, r.cache, n; r.aux_layers)
            prepare_step!(b, r.model, r.cache, seq, ids)
            run_step!(r.model, r.cache, b)
        end
        append!(seq.tokens, ids)
        r.last_step = b
        return b.logits
    end
    step = get!(() -> CapturedStep(StepBuffers(r.model, r.cache, n; r.aux_layers), nothing), r.steps, n)
    r.last_step = step.buffers
    NVTX.@range "upload inputs" prepare_step!(step.buffers, r.model, r.cache, seq, ids)
    if step.graph !== nothing
        NVTX.@range "replay step" CUDA.launch(step.graph)
    else
        NVTX.@range "eager step" run_step!(r.model, r.cache, step.buffers)
        if r.graphs
            # Records the launches without running them: the eager step already did.
            step.graph = NVTX.@range "capture step" begin
                CUDA.instantiate(CUDA.capture(() -> run_step!(r.model, r.cache, step.buffers)))
            end
        end
    end
    append!(seq.tokens, ids)
    return step.buffers.logits
end

"Number of captured CUDA graphs (forward and verify steps)."
num_graphs(r::ModelRunner) =
    count(s -> s.graph !== nothing, values(r.steps)) + count(s -> s.graph !== nothing, values(r.verify_steps))

# --- Decoding with the host one step behind the GPU ---
#
# Two sets of step buffers alternate. The step on set k samples its token on the device
# (`sample_token!` with the runner's sampler) straight into the ids of set 3 - k, and into
# `mirror[3 - k]` in host-mapped memory. So the host can launch the step that consumes a
# token before reading it: while the GPU runs that step, the host waits for the previous
# one, reads its token, checks for a stop token and streams it. The GPU no longer idles
# between steps.

mutable struct DecodePipeline{T}
    buffers::NTuple{2,StepBuffers{T}}
    graphs::Vector{Union{Nothing,CUDA.CuGraphExec}}
    mirror::CuVector{Int32}          # host-mapped: mirror[k] is the token in buffers[k].ids
    mirror_host::Vector{Int32}
    done::NTuple{2,CuEvent}          # recorded after the step on each set
end

function decode_pipeline(r::ModelRunner{T}) where {T}
    r.pipeline === nothing || return r.pipeline::DecodePipeline{T}
    mirror = CuArray{Int32,1,CUDA.HostMemory}(undef, 2)
    bufs = (StepBuffers(r.model, r.cache, 1), StepBuffers(r.model, r.cache, 1))
    r.pipeline = DecodePipeline{T}(bufs, [nothing, nothing], mirror, unsafe_wrap(Array, mirror),
                                   (CuEvent(), CuEvent()))
    return r.pipeline
end

# Sample the token of `logits` into set k (its ids and mirror slot).
sample_into!(r::ModelRunner, p::DecodePipeline, k, logits) =
    sample_token!(r.model.backend, p.buffers[k].ids, view(p.mirror, k:k), logits, r.sampler)

function run_pipelined_step!(r::ModelRunner, p::DecodePipeline, k)
    b = p.buffers[k]
    step() = (run_step!(r.model, r.cache, b); sample_into!(r, p, 3 - k, b.logits))
    if p.graphs[k] === nothing
        NVTX.@range "eager step" step()
        p.graphs[k] = NVTX.@range "capture step" CUDA.instantiate(CUDA.capture(step))
    else
        NVTX.@range "replay step" CUDA.launch(p.graphs[k])
    end
    record(p.done[k])
end

"""
    generate_pipelined!(runner, seq, ids; max_new_tokens, stop_ids, on_token) -> Vector{Int}

[`generate!`](@ref) with the runner's sampler (set it with [`reset_sampler!`](@ref)),
pipelined so that the host runs one step behind the GPU. Needs a runner with CUDA graphs.
"""
function generate_pipelined!(r::ModelRunner, seq::Sequence, ids::AbstractVector{<:Integer};
                             max_new_tokens::Integer, stop_ids, on_token = nothing)
    r.graphs || throw(ArgumentError("pipelined decoding needs CUDA graphs"))
    p = decode_pipeline(r)
    generated = Int[]
    max_new_tokens > 0 || return generated
    logits = forward!(r, seq, ids)
    sample_into!(r, p, 1, logits)
    ready = CuEvent()
    record(ready)                    # the first token, in set 1
    k = 1                            # the set whose ids hold the next token to read
    while true
        more = length(generated) + 1 < max_new_tokens    # will a token follow this one?
        if more                      # run the token before knowing it
            NVTX.@range "upload inputs" prepare_step!(p.buffers[k], r.model, r.cache, seq, nothing)
            run_pipelined_step!(r, p, k)
            push!(seq.tokens, -1)    # filled in below
        end
        NVTX.@range "wait for token" synchronize(ready)
        id = Int(p.mirror_host[k])
        more && (seq.tokens[end] = id)
        if id in stop_ids
            if more                  # the stop token went through the model: drop it again
                synchronize()
                truncate!(r.cache, seq, length(seq) - 1)
            end
            break
        end
        push!(generated, id)
        on_token === nothing || on_token(id)
        more || break
        ready, k = p.done[k], 3 - k
    end
    return generated
end

# ---- src/engine/generate.jl ----

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

One-shot [`generate!`](@ref) in a fresh cache sized for the prompt and `max_new_tokens`,
quantized as `kv_quantize` (see [`PagedKVCache`](@ref)).
"""
function generate(model::DecoderLM, prompt_ids::AbstractVector{<:Integer};
                  max_new_tokens::Integer = 256, kv_quantize::Union{Nothing,Symbol} = nothing, kwargs...)
    total = length(prompt_ids) + max_new_tokens
    total <= max_length(model) ||
        throw(ArgumentError("prompt + max_new_tokens = $total exceeds the model's max length $(max_length(model))"))
    cache = PagedKVCache(model; max_tokens = total, quantize = kv_quantize)
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

# ---- src/engine/sampler.jl ----

# Token sampling. A sampled token is a deterministic function of the logits, the sampling
# parameters and one uniform number u = sampling_uniform(seed, counter): the seed is drawn
# from the caller's RNG once per generation, and the counter numbers the tokens sampled
# since. The device sampler (`sample_token!`, inside the captured decode step) keeps the
# counter on the device, and the host reference (`sample_host`) follows the same rule, so
# both pick the same tokens from the same logits.

using Random: Random, AbstractRNG

"""
    SamplingParams(; temperature = 0, top_k = 0, top_p = 1)

`temperature = 0` is greedy decoding. `top_k = 0` and `top_p = 1` disable those filters.
"""
Base.@kwdef struct SamplingParams
    temperature::Float64 = 0.0
    top_k::Int = 0
    top_p::Float64 = 1.0
    function SamplingParams(temperature, top_k, top_p)
        temperature >= 0 || throw(ArgumentError("temperature must be ≥ 0"))
        top_k >= 0 || throw(ArgumentError("top_k must be ≥ 0"))
        0 < top_p <= 1 || throw(ArgumentError("top_p must be in (0, 1]"))
        return new(temperature, top_k, top_p)
    end
end

"Whether tokens can be sampled with `p` on the device: greedy, or top-k with k ≤ $MAX_TOP_K."
on_device(p::SamplingParams) = p.temperature == 0 || 1 <= p.top_k <= MAX_TOP_K

"""
lowbias32 (Chris Wellons): a bijective mix of 32 bits with low avalanche bias. (cuTile
shifts need a shift count of the operand's type.)
"""
@inline function mix32(x::UInt32)
    x = xor(x, x >> UInt32(16)) * 0x7feb352d
    x = xor(x, x >> UInt32(15)) * 0x846ca68b
    return xor(x, x >> UInt32(16))
end

"""
    sampling_uniform(seed, counter) -> Float32

The uniform number in [0, 1) that decides the `counter`-th token sampled under `seed`. A
pure function of its arguments, evaluated identically on the host and in kernels.
"""
@inline sampling_uniform(seed::UInt32, counter::UInt32) =
    Float32(mix32(xor(mix32(seed), counter)) >> UInt32(8)) * 5.9604645f-8   # 24 bits × 2^-24

"""
The ids of the k largest logits, best first, ties to the smaller id.
"""
function top_candidates(logits::AbstractVector{<:Real}, k::Integer)
    k == length(logits) && return sortperm(logits; rev = true)             # stable
    t = partialsort(logits, k; rev = true)                                 # the k-th largest
    above = findall(>(t), logits)
    ids = [above; findall(==(t), logits)[1:k-length(above)]]
    return sort!(ids; by = i -> (-logits[i], i))
end

"""
    filtered_dist(logits, p) -> (ids, probs)

The distribution [`sample_host`](@ref) samples from: the 1-based ids of the tokens that
survive temperature, then top-k, then top-p (nucleus) over the renormalized survivors, as in
transformers' default logits-processor order, best first, and their probabilities. Greedy
decoding keeps only the argmax.
"""
function filtered_dist(logits::AbstractVector{<:Real}, p::SamplingParams)
    p.temperature == 0 && return [argmax(logits)], [1.0]
    ids = top_candidates(logits, p.top_k == 0 ? length(logits) : min(p.top_k, length(logits)))
    scaled = Float64.(logits[ids]) ./ p.temperature
    probs = exp.(scaled .- scaled[1])
    probs ./= sum(probs)
    if p.top_p < 1
        # Keep the smallest prefix whose mass reaches top_p (always at least one token).
        n = something(findfirst(>=(p.top_p), cumsum(probs)), length(probs))
        ids, probs = ids[1:n], probs[1:n] ./ sum(probs[1:n])
    end
    return ids, probs
end

"""
Inverse transform sampling over weights `w` (best first) by u ∈ [0, 1): the first entry whose
cumulative weight reaches u × the total, among those with positive weight (the last of them
on round-off).
"""
function pick_host(w::AbstractVector{<:Real}, u::Real)
    cum = cumsum(w)
    return something(findfirst(i -> w[i] > 0 && cum[i] >= u * cum[end], eachindex(w)), findlast(>(0), w))
end

"""
    sample_host(logits, p, u) -> Int

The 0-based token sampled from `logits` with `p` by the uniform number `u` ∈ [0, 1): the
reference for the device samplers. Inverse transform sampling over
[`filtered_dist`](@ref), best first.
"""
function sample_host(logits::AbstractVector{<:Real}, p::SamplingParams, u::Real)
    ids, probs = filtered_dist(logits, p)
    return ids[pick_host(probs, u)] - 1
end

# Speculative sampling with drafts sampled from a proposer's distribution q (Leviathan et
# al., Chen et al.): the draft d is accepted with probability min(1, p(d) / q(d)), where p
# is the target's filtered distribution; otherwise the token is sampled from the residual
# max(p - q, 0), renormalized. Either way it is distributed as p, and a resampled token
# never equals d (its residual is 0 when it is rejected). Acceptance draws its uniform
# number from a stream of its own, and drafters sample under a seed of their own.

"The uniform number that decides whether the `counter`-th token accepts its draft."
@inline accept_uniform(seed::UInt32, counter::UInt32) = sampling_uniform(xor(seed, 0x3c6ef372), counter)

"The seed a drafter samples its drafts with, for a generation sampled under `seed`."
draft_seed(seed::UInt32) = xor(seed, 0x9e3779b9)

"""
    accept_host(logits, p, draft, q_ids, q_probs, ua, u) -> Int

The 0-based token emitted at a position drafted as `draft` (0-based), drawn from a
distribution q (0-based `q_ids` with probabilities `q_probs`; other ids have probability
0): `draft` when `ua * q(draft) < p(draft)`, else a token sampled by `u` from max(p - q, 0),
renormalized over p's survivors best first. The reference for the device verifier.
"""
function accept_host(logits::AbstractVector{<:Real}, p::SamplingParams, draft::Integer,
                     q_ids::AbstractVector{<:Integer}, q_probs::AbstractVector{<:Real}, ua::Real, u::Real)
    ids, probs = filtered_dist(logits, p)
    q(t) = sum(q_probs[i] for i in eachindex(q_ids) if q_ids[i] == t; init = 0.0)
    pd = sum(probs[i] for i in eachindex(ids) if ids[i] == draft + 1; init = 0.0)
    r = [max(probs[i] - q(ids[i] - 1), 0.0) for i in eachindex(ids)]
    (ua * q(draft) < pd || sum(r) <= 0) && return Int(draft)
    return ids[pick_host(r, u)] - 1
end

"""
    TokenSampler(vocab)

Device state for [`sample_token!`](@ref): the sampling parameters, the seed and the
counter of tokens sampled so far, and the backend's scratch. Set with
[`reset_sampler!`](@ref) before a generation; each sampled token advances the counter on
the device, so a captured step samples a fresh token on every replay.
"""
mutable struct TokenSampler
    sampling::SamplingParams
    seed::UInt32
    const params::CuVector{Float32}    # [1 / temperature, top_p]
    const state::CuVector{UInt32}      # [top_k, seed, counter]
    const scratch::CuVector{Int32}     # from `sample_scratch(vocab)`
end

TokenSampler(vocab::Integer) = TokenSampler(SamplingParams(), 0, CUDA.ones(Float32, 2),
                                            CuVector{UInt32}([1, 0, 0]), sample_scratch(vocab))

"""
    reset_sampler!(s, p, seed)

Sample with `p` (which must be [`on_device`](@ref)) from counter 0 under `seed`. Greedy
decoding samples the top-1 token.
"""
function reset_sampler!(s::TokenSampler, p::SamplingParams, seed::UInt32)
    on_device(p) || throw(ArgumentError("top_k = $(p.top_k) cannot be sampled on the device (1 ≤ top_k ≤ $MAX_TOP_K)"))
    greedy = p.temperature == 0
    s.sampling, s.seed = p, seed
    copyto!(s.params, Float32[greedy ? 1 : 1 / p.temperature, greedy ? 1 : p.top_p])
    copyto!(s.state, UInt32[greedy ? 1 : p.top_k, seed, 0])
    return s
end

"""
    set_counter!(s, counter)

Sample the next token with `counter`, as when rewinding to a token that was sampled ahead.
Ordered on the stream before the steps that follow.
"""
set_counter!(s::TokenSampler, counter::Integer) = copyto!(view(s.state, 3:3), UInt32[counter])

# ---- src/engine/speculative.jl ----

# Speculative decoding: a cheap proposer guesses the next k tokens, and one forward step of
# the target model on [last token, d1..dk] checks them all at once. The step samples the
# target's own choice after each prefix, the token it would have sampled one step at a time,
# so the drafts are accepted up to the first one that differs from it, and the target's
# choice at that point comes for free (the "bonus" token): each step emits 1 to k + 1 tokens for the
# cost of one step of k + 1 tokens, which at decode sizes costs little more than one token
# (the step reads the weights once either way). The keys and values of rejected drafts are
# dropped again by truncating the sequence: its pages go back to the pool, and later steps
# overwrite the stale slots.
#
# Sampling. Rejection sampling (Leviathan et al., Chen et al.) accepts a draft d drawn from
# the proposer's q with probability min(1, p(d) / q(d)), where p is the target's (filtered)
# distribution, and otherwise samples from max(p - q, 0) renormalized. The n-gram proposer is
# deterministic, q is one-hot on d: that is "accept d with probability p(d), else sample from
# p without d", the same as sampling t ~ p and accepting d if t == d. So the verify step
# samples each position with the device sampler and the counter plain decoding would use for
# that token (`sampling_uniform(seed, counter)`): the drafts are accepted with probability
# p(d), and the tokens are those of plain sampled decoding, up to round-off. Greedy decoding
# is the case p = one-hot on the argmax. Model proposers draft greedily by default, so this
# rule is exact for them too. With `sample = true` they sample their drafts from their own
# filtered distribution q (the target's sampling parameters on the drafter's logits) and
# record it (`DraftDists`); the verify step then applies the general rule at each draft
# position (`accept_token!`), and the target's own sample only at the bonus position. The
# tokens are then distributed as plain sampled decoding's, but are no longer the same.
#
# Proposers: n-gram (prompt) lookup below, where the drafts are the tokens that followed the
# latest earlier occurrence of the sequence's last n tokens; a draft model (end of this
# file); DFlash (dflash.jl).

"""
    NgramProposer(; k = 4, max_ngram = 3, min_ngram = 1)

Proposes up to `k` tokens by prompt lookup: find the latest earlier occurrence of the last
n tokens (n from `max_ngram` down to `min_ngram`) and propose the tokens that followed it.
"""
Base.@kwdef struct NgramProposer
    k::Int = 4
    max_ngram::Int = 3
    min_ngram::Int = 1
    function NgramProposer(k, max_ngram, min_ngram)
        1 <= k < MAX_GRAPH_TOKENS || throw(ArgumentError("k must be in 1:$(MAX_GRAPH_TOKENS - 1)"))
        1 <= min_ngram <= max_ngram || throw(ArgumentError("need 1 ≤ min_ngram ≤ max_ngram"))
        return new(k, max_ngram, min_ngram)
    end
end

"""
    propose(proposer, tokens, k) -> Vector{Int}

Up to `k` draft tokens to follow `tokens` (possibly none).
"""
function propose(p::NgramProposer, tokens::AbstractVector{<:Integer}, k::Integer)
    L = length(tokens)
    for n in min(p.max_ngram, L - 1):-1:p.min_ngram
        tail = view(tokens, L-n+1:L)
        # Latest match first: recent context predicts best. The match must end before the tail.
        for s in L-n:-1:1
            if view(tokens, s:s+n-1) == tail
                return collect(Int, view(tokens, s+n:min(s + n + k - 1, L)))
            end
        end
    end
    return Int[]
end

"""
    propose_on_device!(proposer, tokens, k) -> Union{Nothing,CuVector{Int32}}

Enqueue drafting `k` tokens to follow `tokens` without waiting for them, and return the
device vector they land in (0-based ids), for [`verify!`](@ref) to read on the device; the
host reads them with [`drafted`](@ref) once the stream has run that far. `nothing` when the
proposer cannot (then [`propose`](@ref) runs instead).
"""
propose_on_device!(::Any, ::AbstractVector{<:Integer}, ::Integer) = nothing

"""
    drafted(proposer, k) -> Vector{Int}

The `k` drafts of the latest [`propose_on_device!`](@ref), after the stream has run them.
"""
function drafted end

"""
    attach!(proposer, runner)

Prepare `proposer` to draft for the target model of `runner`, before a generation.
"""
attach!(::Any, ::ModelRunner) = nothing

"""
Run a proposer's `step`: replay `graph`, or run it eagerly and capture it when the proposer
uses graphs. Returns the graph.
"""
function run_captured!(step, p, graph)
    p.graphs || return (step(); nothing)
    graph === nothing || (CUDA.launch(graph); return graph)
    step()
    return CUDA.instantiate(CUDA.capture(step))
end

"""
    SpecStats()

Counters of a speculative generation: verify steps, drafted and accepted tokens.
"""
Base.@kwdef mutable struct SpecStats
    steps::Int = 0
    drafted::Int = 0
    accepted::Int = 0
end

"Fraction of drafted tokens that were accepted (`NaN` before any draft)."
acceptance_rate(s::SpecStats) = s.drafted == 0 ? NaN : s.accepted / s.drafted

"""
    DraftDists(k)

The distributions q that up to `k` sampled drafts were drawn from: column j holds the
0-based ids of draft j's candidates (-1 past them) and their probabilities, as
`sample_token!(…, q)` writes them.
"""
struct DraftDists
    ids::CuMatrix{Int32}             # (MAX_TOP_K, k)
    probs::CuMatrix{Float32}
end

DraftDists(k::Integer) = DraftDists(CuMatrix{Int32}(undef, MAX_TOP_K, k), CuMatrix{Float32}(undef, MAX_TOP_K, k))

"Draft j's distribution, as the `q` argument of `sample_token!` and `accept_token!`."
column(d::DraftDists, j) = (view(d.ids, :, j), view(d.probs, :, j))

"""
    draft_dists(proposer, runner) -> Union{DraftDists,Nothing}

The distributions of the drafts `proposer` samples for the target of `runner`, or nothing
when its drafts are a function of the context (then verification compares them with the
target's own samples).
"""
draft_dists(::Any, ::ModelRunner) = nothing

"""
Whether a proposer that can sample its drafts does so for `runner`: only when the target
samples (with greedy decoding, both draft and target take the argmax).
"""
samples_for(sample::Bool, r::ModelRunner) = sample && r.sampler.sampling.temperature > 0

# Reset a drafter's sampler for a generation of `runner`'s: the target's parameters when it
# samples its drafts, greedy otherwise.
reset_draft_sampler!(s::TokenSampler, sample::Bool, r::ModelRunner) =
    reset_sampler!(s, samples_for(sample, r) ? r.sampler.sampling : SamplingParams(), draft_seed(r.sampler.seed))

"Verify steps take up to this many tokens (a DFlash block); all are captured in CUDA graphs."
const MAX_VERIFY_TOKENS = 16

"""
A step that returns the token the target model samples after each of its n inputs: the
forward step with the logits of every token, then the runner's sampler on each column
(counters c, c + 1, …), all captured together in a CUDA graph. With the distributions of
sampled drafts, the columns of drafted positions apply the p/q rule instead. The tokens
land in host-mapped memory.
"""
mutable struct VerifyStep{T}
    buffers::StepBuffers{T}
    dists::Union{Nothing,DraftDists}
    graph::Union{Nothing,CUDA.CuGraphExec}
    targets::CuVector{Int32}         # device copies (`sample_token!` writes both)
    mirror::CuVector{Int32}          # host-mapped
    mirror_host::Vector{Int32}
end

function VerifyStep(r::ModelRunner{T}, n::Integer, dists) where {T}
    mirror = CuArray{Int32,1,CUDA.HostMemory}(undef, n)
    return VerifyStep{T}(StepBuffers(r.model, r.cache, n; all_logits = true, r.aux_layers), dists, nothing,
                         CuVector{Int32}(undef, n),
                         mirror, unsafe_wrap(Array, mirror))
end

function run_verify_step!(r::ModelRunner, v::VerifyStep)
    b, be = v.buffers, r.model.backend
    run_step!(r.model, r.cache, b)
    n = num_tokens(b)
    for j in 1:n
        ids, mirror, logits = view(v.targets, j:j), view(v.mirror, j:j), view(b.logits, :, j:j)
        if v.dists === nothing || j == n
            sample_token!(be, ids, mirror, logits, r.sampler)
        else                         # the position drafted as input j + 1
            accept_token!(be, ids, mirror, logits, r.sampler, view(b.ids, j+1:j+1), column(v.dists, j))
        end
    end
end

"""
    verify!(runner, seq, ids, counter; dists = nothing, drafts = nothing) -> Vector{Int}

Run the 0-based token `ids` (the last sampled token, then the drafts) through the model,
appending them to `seq`, and return the token the runner's sampler picks after each of them,
with counters `counter`, `counter + 1`, … (the argmax when greedy). When the drafts were
sampled from the distributions `dists` (column j for draft j), the token after each input
but the last follows the p/q rule instead ([`accept_token!`](@ref)): the next draft if
accepted, else a different token. Steps of each length, up to $MAX_VERIFY_TOKENS tokens, are
captured in a CUDA graph like [`forward!`](@ref) steps. With `drafts`, a device vector
(see [`propose_on_device!`](@ref)), the drafts are read from it on the device rather than
from `ids[2:end]`, which then only hold places (and go into `seq.tokens` as they are).
"""
function verify!(r::ModelRunner{T}, seq::Sequence, ids::AbstractVector{<:Integer}, counter::Integer;
                 dists::Union{Nothing,DraftDists} = nothing,
                 drafts::Union{Nothing,CuVector{Int32}} = nothing) where {T}
    n = length(ids)
    1 <= n <= MAX_VERIFY_TOKENS || throw(ArgumentError("verify! takes 1 to $MAX_VERIFY_TOKENS tokens, got $n"))
    dists === nothing || n - 1 <= size(dists.ids, 2) || throw(ArgumentError("$(n - 1) drafts but $(size(dists.ids, 2)) distributions"))
    key = (n, dists === nothing ? UInt(0) : objectid(dists))
    v = get!(() -> VerifyStep(r, n, dists), r.verify_steps, key)::VerifyStep{T}
    r.last_step, r.last_start = v.buffers, length(seq)
    NVTX.@range "upload inputs" begin
        prepare_step!(v.buffers, r.model, r.cache, seq, ids)
        # Ordered after the upload on the stream, which wrote placeholders there.
        drafts === nothing || copyto!(view(v.buffers.ids, 2:n), drafts)
        set_counter!(r.sampler, counter)
    end
    if v.graph !== nothing
        NVTX.@range "replay verify" CUDA.launch(v.graph)
    else
        NVTX.@range "eager verify" run_verify_step!(r, v)
        if r.graphs
            v.graph = NVTX.@range "capture verify" CUDA.instantiate(CUDA.capture(() -> run_verify_step!(r, v)))
        end
    end
    append!(seq.tokens, ids)
    NVTX.@range "wait for tokens" synchronize()
    return Int.(v.mirror_host)
end

"""
    generate_speculative!(runner, seq, ids, proposer; max_new_tokens, stop_ids, on_token,
                          stats = SpecStats()) -> Vector{Int}

[`generate!`](@ref) with the runner's sampler (set it with [`reset_sampler!`](@ref)) and drafts
from `proposer` checked by [`verify!`](@ref): the same tokens as plain decoding with that
sampler up to near-ties (a verify step of k + 1 tokens runs other kernels than a decode
step, so the logits differ in the last bits), and `seq` ends in the same state. A proposer
that samples its drafts ([`draft_dists`](@ref)) yields tokens with the same distribution
instead, not the same tokens. Counts steps and accepted drafts in `stats`.
"""
function generate_speculative!(r::ModelRunner, seq::Sequence, ids::AbstractVector{<:Integer}, proposer;
                               max_new_tokens::Integer, stop_ids, on_token = nothing,
                               stats::SpecStats = SpecStats())
    generated = Int[]
    max_new_tokens > 0 || return generated
    attach!(proposer, r)
    dists = draft_dists(proposer, r)
    logits = forward!(r, seq, ids)
    # Sampled with counter 0, not yet in the cache.
    last = Int(Array(sample_token!(r.model.backend, r.token, r.token, logits, r.sampler))[1])
    context = copy(seq.tokens)       # seq.tokens and `last`, for the proposer
    while true
        last in stop_ids && break
        push!(generated, last)
        on_token === nothing || on_token(last)
        remaining = max_new_tokens - length(generated)
        remaining == 0 && break
        push!(context, last)
        # A step emits at most k + 1 tokens: draft no more than the budget needs.
        k = remaining > 1 ? min(proposer.k, remaining - 1) : 0
        start = length(seq)
        # The token after `last` is generated token number length(generated) (0-based). A
        # proposer that drafts on the device hands its drafts to the verify step there, so
        # the host waits once per step.
        on_device = k > 0 ? propose_on_device!(proposer, context, k) : nothing
        if on_device === nothing
            drafts = k > 0 ? propose(proposer, context, k) : Int[]
            targets = verify!(r, seq, [last; drafts], length(generated); dists)
        else
            targets = verify!(r, seq, [last; zeros(Int, k)], length(generated); dists, drafts = on_device)
            drafts = drafted(proposer, k)
            seq.tokens[end-k+1:end] .= drafts
        end
        # (With sampled drafts, a rejected draft's position holds a token that differs from it.)
        accepted = 0
        while accepted < length(drafts) && drafts[accepted+1] == targets[accepted+1]
            accepted += 1
        end
        stats.steps += 1
        stats.drafted += length(drafts)
        stats.accepted += accepted
        # Emit the accepted drafts, then the bonus token, which becomes the next `last`.
        # Stop early on a stop token or the budget, keeping in the cache only the tokens
        # before the last emitted one, as the plain loop does.
        emitted, stopped = 0, false
        while emitted < accepted && !stopped
            emitted += 1
            t = targets[emitted]
            if t in stop_ids
                stopped = true
            else
                push!(generated, t)
                push!(context, t)
                on_token === nothing || on_token(t)
                stopped = length(generated) == max_new_tokens
            end
        end
        # The cache holds `last` and the drafts; keep `last` and the drafts emitted (all of
        # them but one when stopping, as the stopping token is never run).
        truncate!(r.cache, seq, start + 1 + (stopped ? emitted - 1 : accepted))
        stopped && break
        last = targets[accepted+1]
    end
    return generated
end

# --- Draft-model proposer ---
#
# A small model with the same vocabulary (Qwen3-0.6B for Qwen3-4B) runs ahead of the target
# on a cache of its own. Its sequence follows the generation's context lazily: each proposal
# first drops what the target rejected (the longest common prefix with the context is kept)
# and runs the one or two new tokens, then decodes k - 1 more steps. Those steps form a chain
# of CUDA graphs: step i reads draft i from its buffers and samples draft i + 1 straight into
# the ids of step i + 1 and into host-mapped memory, so the host enqueues the whole chain
# and synchronizes once.
#
# The draft decodes greedily by default, even when the target samples: a greedy draft is a
# function of the context, so the verification above stays exact. With `sample = true` and
# a sampling target, it samples with the target's parameters instead and records each
# draft's q for the general rule.

"""
    DraftProposer(draft_model; k = 4, sample = false, max_tokens = max_length(draft_model),
                  graphs = supports_graphs(draft_model.backend))

Proposes the next `k` tokens that `draft_model` decodes greedily, on a KV cache of its own
for up to `max_tokens` positions; or, with `sample` and a target that samples, that it
samples with the target's sampling parameters, verified by the p/q rule. The draft model
must share the target's vocabulary.
"""
mutable struct DraftProposer{R<:ModelRunner,T}
    const k::Int
    const sample::Bool
    const dists::DraftDists                      # of the drafts, when sampled
    const runner::R
    const seq::Sequence
    const buffers::Vector{StepBuffers{T}}        # step i runs draft i, for i in 1:k-1
    const graphs::Vector{Union{Nothing,CUDA.CuGraphExec}}
    const sink::CuVector{Int32}                  # ids written by the last step of the chain
    const mirror::CuVector{Int32}                # host-mapped: the drafts
    const mirror_host::Vector{Int32}
end

function DraftProposer(m::DecoderLM{T}; k::Integer = 4, sample::Bool = false, max_tokens::Integer = max_length(m),
                       graphs::Bool = supports_graphs(m.backend)) where {T}
    1 <= k < MAX_GRAPH_TOKENS || throw(ArgumentError("k must be in 1:$(MAX_GRAPH_TOKENS - 1)"))
    cache = PagedKVCache(m; max_tokens = min(max_tokens, max_length(m)))
    r = ModelRunner(m, cache; graphs)
    mirror = CuArray{Int32,1,CUDA.HostMemory}(undef, k)
    return DraftProposer{typeof(r),T}(k, sample, DraftDists(k), r, new_sequence(cache), [StepBuffers(m, cache, 1) for _ in 1:k-1],
                                      fill!(Vector{Union{Nothing,CUDA.CuGraphExec}}(undef, k - 1), nothing),
                                      CuVector{Int32}(undef, 1), mirror, unsafe_wrap(Array, mirror))
end

attach!(p::DraftProposer, r::ModelRunner) = (reset_draft_sampler!(p.runner.sampler, p.sample, r); p)
draft_dists(p::DraftProposer, r::ModelRunner) = samples_for(p.sample, r) ? p.dists : nothing

"""
Where the sample of chain step i (or of the forward step before the chain, i = 0) goes.
A proposer that can sample always records q (one-hot when greedy), so that its captured
chain serves greedy and sampled generations alike.
"""
next_ids(p::DraftProposer, i) = i < p.k - 1 ? p.buffers[i+1].ids : p.sink
function sample_draft!(p::DraftProposer, i, logits)
    args = (p.runner.model.backend, next_ids(p, i), view(p.mirror, i+1:i+1), logits, p.runner.sampler)
    return p.sample ? sample_token!(args..., column(p.dists, i + 1)) : sample_token!(args...)
end

function run_draft_step!(p::DraftProposer, i)
    r, b = p.runner, p.buffers[i]
    step() = (run_step!(r.model, r.cache, b); sample_draft!(p, i, b.logits))
    if !r.graphs
        step()
    elseif p.graphs[i] === nothing
        NVTX.@range "eager draft step" step()
        p.graphs[i] = NVTX.@range "capture draft step" CUDA.instantiate(CUDA.capture(step))
    else
        NVTX.@range "replay draft step" CUDA.launch(p.graphs[i])
    end
end

function propose(p::DraftProposer, context::AbstractVector{<:Integer}, k::Integer)
    k = min(k, p.k)
    k >= 1 || return Int[]
    r, seq = p.runner, p.seq
    # Keep what the draft shares with the context, but run at least one token for logits.
    keep = min(common_prefix(seq.tokens, context), length(context) - 1)
    truncate!(r.cache, seq, keep)
    logits = forward!(r, seq, view(context, keep+1:length(context)))
    NVTX.@range "draft" begin
        sample_draft!(p, 0, logits)
        for i in 1:k-1
            prepare_step!(p.buffers[i], r.model, r.cache, seq, nothing)
            push!(seq.tokens, -1)                # draft i, read below
            run_draft_step!(p, i)
        end
        synchronize()
    end
    drafts = Int.(p.mirror_host[1:k])
    seq.tokens[end-k+2:end] .= drafts[1:k-1]
    return drafts
end

# ---- src/engine/dflash.jl ----

# DFlash drafting (Chen et al., https://arxiv.org/abs/2602.06036): a small block-diffusion
# model drafts a whole block of tokens in one forward pass, where draft models (and EAGLE) run
# one pass per draft. Layout and conventions follow z-lab's reference `dflash.py`.
#
# The target keeps the residual stream after five of its layers for every token it runs;
# `fc` maps them, stacked, to one context state per position, c(p) = hidden_norm(fc(...)).
# The drafter is a stack of Qwen3 layers whose keys and values come from two sources:
#
#   - the context positions the target has run: k and v projections of c(p) (no input
#     norm), kept in the drafter's KV cache, so each position is computed once;
#   - the block: B = block_size positions after the context, the next token (known: the
#     target's last choice) followed by B - 1 mask tokens, as ordinary layer inputs.
#
# Only the block has queries, and its attention is not causal: every block position sees the
# whole context and the whole block. After the last layer, the target's own lm_head reads
# block positions 2:B, and their argmaxes are B - 1 drafts at once. The block's keys and
# values are dropped again (the cache is truncated to the context).
#
# Here one drafter step runs the new context positions and the block together, as n = c + B
# columns: every layer computes q, k and v for all of them (the context columns read c(p),
# the block columns their normed residual stream), writes all keys and values, and attends
# without the causal mask; the output projection and MLP run on the block columns only.
# The context queries are computed and thrown away, which costs less than a second read of
# the q/k/v weights. Steps with up to B context positions (every step after the prefill)
# are captured in CUDA graphs, one per context length.

"""
    DFlashDrafter

A DFlash drafter on the GPU: its Qwen3 layers, `fc` and norms, the target's embedding table
and `lm_head` (shared), its RoPE table, the target layers it reads (`aux_layers`, 1-based),
the block size and the mask token. Load with [`load_dflash`](@ref).
"""
struct DFlashDrafter{T,M<:AbstractMatrix{T},L,V<:AbstractVector{T},R<:RotaryTable{T},B<:Backend}
    config::ModelConfig
    aux_layers::Vector{Int}
    block_size::Int
    mask_id::Int
    fc::M                            # (aux layers × hidden, hidden)
    hidden_norm::V
    layers::Vector{LayerWeights{T,M,V}}
    norm::V
    embed::M                         # the target's (hidden, vocab)
    lm_head::L                       # the target's (Int8Weight if it is quantized)
    rope::R
    backend::B
end

max_length(d::DFlashDrafter) = length(d.rope)

"""
    load_dflash(drafter, target::DecoderLM; max_len = max_length(target)) -> DFlashDrafter

Load a DFlash drafter for `target` (a pulled repo id or a directory), in the format of
z-lab's checkpoints (`layers.*`, `fc`, `hidden_norm`, `dflash_config`).
"""
load_dflash(name::AbstractString, target::DecoderLM{T}; max_len::Integer = max_length(target)) where {T} =
    load_dflash(name, target.config, target.weights.embed, target.weights.lm_head, target.backend; max_len)

"""
    load_dflash(drafter, target_config, embed, lm_head, backend; max_len) -> DFlashDrafter

The same, with the target's configuration, embedding table and `lm_head` given separately
(the tests run the drafter in Float32 without the whole target).
"""
function load_dflash(name::AbstractString, tcfg::ModelConfig, embed::AbstractMatrix{T}, lm_head,
                     backend::Backend; max_len::Integer) where {T}
    dir = model_dir(name)
    raw = JSON.parsefile(joinpath(dir, "config.json"))
    cfg = parse_config(raw)
    cfg.vocab_size == tcfg.vocab_size || error("the drafter's vocabulary differs from the target's")
    cfg.hidden_size == tcfg.hidden_size ||
        error("drafter hidden size $(cfg.hidden_size) differs from the target's $(tcfg.hidden_size)")
    dcfg = get(raw, "dflash_config", Dict())
    haskey(dcfg, "mask_token_id") || error("$name has no dflash_config.mask_token_id")
    aux = dflash_layers(raw, tcfg.num_layers)
    B = raw["block_size"]
    B <= MAX_VERIFY_TOKENS || error("blocks of $B tokens exceed the verify steps' $MAX_VERIFY_TOKENS")

    c = Checkpoint(dir)
    H = cfg.hidden_size
    up(n, dims) = check_size(upload(T, c, n), dims, n)
    layers = [load_layer(T, c, cfg, "layers.$i.") for i in 0:cfg.num_layers-1]
    # The block runs B positions past the target's last one.
    return DFlashDrafter(cfg, aux, B, dcfg["mask_token_id"], up("fc.weight", (length(aux) * H, H)),
                         up("hidden_norm.weight", (H,)), layers, up("norm.weight", (H,)),
                         embed, lm_head, RotaryTable{T}(cfg, max_len + B), backend)
end

"""
The target layers whose outputs the drafter reads, counted from 1. DFlash's
`target_layer_ids` index transformers' `hidden_states` minus one, i.e. number layer
outputs from 0; without them, they are spread evenly from layer 1 to L - 3.
"""
function dflash_layers(raw::AbstractDict, num_layers::Integer)
    ids = get(get(raw, "dflash_config", Dict()), "target_layer_ids", nothing)
    if ids === nothing
        n = raw["num_hidden_layers"]
        ids = n == 1 ? [num_layers ÷ 2] :
              [round(Int, 1 + i * (num_layers - 4) / (n - 1)) for i in 0:n-1]    # ties to even, as Python
    end
    return collect(Int, ids) .+ 1
end

"""
Buffers of one drafter step of c context positions and a block: the step's inputs and
per-column activations for n = c + B columns (b.h holds the layers' input: the context
states, then the normed block), and the target's stacked states of the context.
"""
struct DFlashStep{T}
    b::StepBuffers{T}
    aux::CuMatrix{T}                 # (aux layers × hidden, c)
end

DFlashStep(d::DFlashDrafter{T}, cache::PagedKVCache{T}, c::Integer) where {T} =
    DFlashStep{T}(StepBuffers{T}(d.config, d.backend, cache, c + d.block_size; vocab = 1),
                  CuMatrix{T}(undef, size(d.fc, 1), c))

"""
    DFlashProposer(drafter; k = drafter.block_size - 1, sample = false, max_tokens = max_length(drafter),
                   graphs = supports_graphs(drafter.backend))

Proposes the first `k` of the `block_size - 1` tokens that the DFlash `drafter` (see
[`load_dflash`](@ref)) drafts in one pass, on a KV cache of its own for up to `max_tokens`
positions: the argmax at each block position, or, when `sample` and the target samples, a
sample with the target's parameters (each position independently: the block's positions do
not see each other's drafts, so q of draft j is the drafter's distribution at position j). Reads
the hidden states of the target's latest step, which generation enables on the target's runner.
"""
mutable struct DFlashProposer{D<:DFlashDrafter,T}
    const k::Int
    const sample::Bool
    const dists::DraftDists                      # of the drafts, when sampled
    const drafter::D
    const cache::PagedKVCache{T}
    const seq::Sequence                          # the context positions the cache holds
    const graphs::Bool
    const steps::Dict{Int,Any}                   # (DFlashStep, graph) by context length
    const x::CuMatrix{T}                         # (hidden, B): the block's residual stream
    const h::CuMatrix{T}                         # (hidden, B): its normed state (MLP input, output)
    const logits::CuMatrix{Float32}              # (vocab, B - 1)
    const sink::CuVector{Int32}                  # device copies of the drafts
    const mirror::CuVector{Int32}                # host-mapped: the drafts
    const mirror_host::Vector{Int32}
    const sampler::TokenSampler
    const scratch::CuMatrix{Int32}               # for sample_tokens!
    target::Any                                  # the target's ModelRunner
end

function DFlashProposer(d::DFlashDrafter{T}; k::Integer = d.block_size - 1, sample::Bool = false,
                        max_tokens::Integer = max_length(d),
                        graphs::Bool = supports_graphs(d.backend)) where {T}
    B, H, V = d.block_size, d.config.hidden_size, d.config.vocab_size
    1 <= k < B || throw(ArgumentError("k must be in 1:$(B - 1)"))
    cache = PagedKVCache{T}(d.config, cld(min(max_tokens + B, max_length(d)), DEFAULT_PAGE_SIZE))
    mirror = CuArray{Int32,1,CUDA.HostMemory}(undef, B - 1)
    return DFlashProposer{typeof(d),T}(k, sample, DraftDists(B - 1), d, cache, new_sequence(cache), graphs, Dict{Int,Any}(),
                                       CuMatrix{T}(undef, H, B), CuMatrix{T}(undef, H, B),
                                       CuMatrix{Float32}(undef, V, B - 1), CuVector{Int32}(undef, B - 1), mirror,
                                       unsafe_wrap(Array, mirror), TokenSampler(V), sample_cols_scratch(B - 1), nothing)
end

function attach!(p::DFlashProposer, r::ModelRunner)
    size(p.drafter.fc, 1) == length(p.drafter.aux_layers) * r.model.config.hidden_size ||
        throw(ArgumentError("the DFlash drafter does not fit this target model"))
    enable_aux!(r, p.drafter.aux_layers)
    reset_draft_sampler!(p.sampler, p.sample, r)
    p.target = r
    return p
end

draft_dists(p::DFlashProposer, r::ModelRunner) = samples_for(p.sample, r) ? p.dists : nothing

"""
The drafter on a prepared step whose aux holds the target's context states; leaves the
logits of block positions 2:B in p.logits.
"""
function dflash_step!(p::DFlashProposer{<:DFlashDrafter{T}}, s::DFlashStep{T}) where {T}
    d, cache, b = p.drafter, p.cache, s.b
    cfg, be = d.config, d.backend
    n, B, eps = num_tokens(b), d.block_size, cfg.rms_norm_eps
    c = n - B
    ctx, blk = view(b.h, :, 1:c), view(b.h, :, c+1:n)              # column views are CuArrays
    proj, act, gate_up = view(b.proj, :, 1:B), view(b.act, :, 1:B), view(b.gate_up, :, 1:B)
    linear!(be, view(b.x, :, 1:c), d.fc, s.aux)
    rmsnorm!(be, ctx, view(b.x, :, 1:c), d.hidden_norm, eps)       # the same for every layer
    embed!(be, p.x, d.embed, view(b.ids, c+1:n))
    rmsnorm!(be, blk, p.x, first(d.layers).attn_norm, eps)
    for (l, layer) in enumerate(d.layers)
        linear!(be, b.qkv, layer.wqkv, b.h)
        attention_inputs!(be, b.q, cache.k[l], cache.v[l], b.qkv, layer.q_norm, layer.k_norm, d.rope,
                          b.positions, b.slots, eps)
        attention!(be, b.attn, b.attn_workspace, b.q, cache.k[l], cache.v[l], b.block_table, b.positions,
                   1 / sqrt(cfg.head_dim); causal = false)
        linear!(be, proj, layer.wo, view(reshape(b.attn, :, n), :, c+1:n))
        add_rmsnorm!(be, p.x, p.h, proj, layer.mlp_norm, eps)
        linear_swiglu!(be, act, gate_up, layer.w_gate_up, p.h)
        linear!(be, proj, layer.w_down, act)
        # The next layer's input norm into the block columns of b.h, or the final norm.
        last = l == length(d.layers)
        add_rmsnorm!(be, p.x, last ? p.h : blk, proj, last ? d.norm : d.layers[l+1].attn_norm, eps)
    end
    linear!(be, p.logits, d.lm_head, view(p.h, :, 2:B))
    # All B - 1 positions in one call; q recorded when sampled (see DraftProposer).
    sample_tokens!(be, p.sink, p.mirror, p.logits, p.sampler, p.scratch,
                   p.sample ? (p.dists.ids, p.dists.probs) : nothing)
end

function propose(p::DFlashProposer, context::AbstractVector{<:Integer}, k::Integer)
    k = min(k, p.k)
    k >= 1 || return Int[]
    r = p.target::ModelRunner
    draft!(p, context, r.last_start, (r.last_step::StepBuffers).aux)
    synchronize()
    return Int.(p.mirror_host[1:k])
end

"""
The drafts stay on the device, for the verify step to copy into its inputs: the host
enqueues the drafter's step and the target's verify step back to back and waits once.
"""
function propose_on_device!(p::DFlashProposer, context::AbstractVector{<:Integer}, k::Integer)
    1 <= k <= p.k || return nothing
    r = p.target::ModelRunner
    draft!(p, context, r.last_start, (r.last_step::StepBuffers).aux)
    return view(p.sink, 1:k)
end

drafted(p::DFlashProposer, k::Integer) = Int.(p.mirror_host[1:k])

"""
Enqueue the drafting of the block after `context` into p.sink and p.mirror (and its logits
into p.logits), where the target's latest step ran positions start:length(context)-2 and
left their states in the first columns of `aux`. Does not wait for the drafts.
"""
function draft!(p::DFlashProposer, context::AbstractVector{<:Integer}, start::Integer, aux::CuMatrix)
    d, seq = p.drafter, p.seq
    L, B = length(context), d.block_size
    # The target ran positions start:L-2 last (L - 1 is the newest token, the block's first).
    # Keep the context positions before that which still hold the context's tokens.
    keep = min(common_prefix(seq.tokens, view(context, 1:L-1)), start)
    truncate!(p.cache, seq, keep)
    if keep < start
        # The target ran positions the drafter never saw (the end of an earlier generation,
        # or a prefix reused without drafting): their hidden states are gone, so their keys
        # and values are left stale. Drafts get worse; the tokens stay correct.
        reserve!(p.cache.blocks, seq, start - keep)
        append!(seq.tokens, view(context, keep+1:start))
    end
    c = L - 1 - start
    # The context's tokens (only the block's are read), then the block.
    ids = [view(context, start+1:L); fill(d.mask_id, B - 1)]
    NVTX.@range "draft" begin
        s, graph = c > B ? (DFlashStep(d, p.cache, c), nothing) : get!(() -> (DFlashStep(d, p.cache, c), nothing), p.steps, c)
        prepare_step!(s.b, d, p.cache, seq, ids)
        copyto!(s.aux, 1, aux, 1, length(s.aux))         # the target's states of positions start:L-2
        step() = dflash_step!(p, s)
        if c > B
            step()
        else
            p.steps[c] = (s, run_captured!(step, p, graph))
        end
    end
    append!(seq.tokens, view(context, start+1:L-1))
    truncate!(p.cache, seq, L - 1)                       # drop the block's keys and values
    return p
end

# ---- src/model/decoder.jl ----

# Llama-style decoder (Llama 3, Mistral, Qwen3, Apertus) as plain structs plus an explicit
# forward pass over backend ops. Per-family differences are ModelConfig flags.

"""
    DecoderLM{T}

A model on the GPU: configuration, weights, RoPE tables and the op backend. Load with
[`load_model`](@ref); run with [`forward!`](@ref).
"""
struct DecoderLM{T,W<:ModelWeights{T},R<:RotaryTable{T},B<:Backend}
    config::ModelConfig
    weights::W
    rope::R
    backend::B
end

"""
    load_model(model; T = BFloat16, backend = RefBackend(), max_len = 4096, quantize = nothing) -> DecoderLM

Load a pulled model (repo id) or a model directory. `max_len` bounds the positions the
RoPE tables cover, i.e. the longest sequence the model can process. `quantize = :int8`
stores the linear weights as Int8 with per-channel scales (see [`Int8Weight`](@ref)).
"""
function load_model(model::AbstractString; T::Type = BFloat16, backend::Backend = RefBackend(),
                    max_len::Integer = 4096, quantize::Union{Nothing,Symbol} = nothing)
    dir = model_dir(model)
    cfg = load_config(dir)
    max_len = min(max_len, cfg.max_position_embeddings)
    return DecoderLM(cfg, load_weights(T, cfg, dir; quantize), RotaryTable{T}(cfg, max_len), backend)
end

Base.eltype(::DecoderLM{T}) where {T} = T
"""
    max_length(model) -> Int

The longest sequence `model` can process: the positions its RoPE tables cover, set by
`max_len` in [`load_model`](@ref).
"""
max_length(m::DecoderLM) = length(m.rope)

"""
    PagedKVCache(model; max_tokens = max_length(model), page_size = $DEFAULT_PAGE_SIZE, quantize = nothing)

An empty paged KV cache for `model` with room for `max_tokens` positions in total;
`quantize = :int8` stores it as Int8 (see [`Int8Pages`](@ref)), with keys smoothed by the
layers' key norms when the model has them.
"""
function PagedKVCache(m::DecoderLM{T}; max_tokens::Integer = max_length(m), page_size::Integer = DEFAULT_PAGE_SIZE,
                      quantize::Union{Nothing,Symbol} = nothing) where {T}
    norms = [layer.k_norm for layer in m.weights.layers]
    smoothing = quantize === nothing || any(isnothing, norms) ? nothing : key_smoothing.(norms)
    return PagedKVCache{T}(m.config, cld(max_tokens, page_size); page_size, quantize, key_smoothing = smoothing)
end

"""
    StepBuffers(model, cache, n; all_logits = false, aux_layers = Int[])

Device memory for one forward step of `n` tokens: the step's inputs (token ids, positions,
cache slots and a block table long enough for every page of the pool), every intermediate
activation, and the logits of the last token, or of all tokens with `all_logits`. With
`aux_layers`, the step also keeps the residual stream after each of those layers (1-based),
stacked in `aux`, for a DFlash drafter (see [`DFlashProposer`](@ref)). A step
that only writes into these buffers allocates nothing, so on a backend that
[`supports_graphs`](@ref) it can be captured in a CUDA graph and replayed (see
[`ModelRunner`](@ref)).
"""
struct StepBuffers{T}
    # The step's inputs: views of one device vector `meta`, uploaded in one copy from the
    # pinned `staging` (written through its host view `staging_host`). Each segment starts
    # at a multiple of `segment` (32 entries, 128 bytes), so every view stays aligned.
    meta::CuVector{Int32}
    staging::CuVector{Int32}         # pinned host memory (CUDA.HostMemory)
    staging_host::Vector{Int32}
    uploaded::CuEvent                # recorded after the last copy out of `staging`
    segment::Int
    ids::CuVector{Int32}            # (n,) 0-based token ids
    positions::CuVector{Int32}      # (n,) 0-based positions
    slots::CuVector{Int32}          # (n,) cache slots, see `slots`
    block_table::CuVector{Int32}    # the sequence's pages, zero-padded to a fixed length
    x::CuMatrix{T}                  # residual stream (hidden, n)
    h::CuMatrix{T}                  # normalized input of a sublayer (hidden, n)
    qkv::CuMatrix{T}                # fused projection ((heads + 2 kv_heads) head_dim, n)
    q::CuArray{T,3}                 # attention queries (head_dim, heads, n), after q norm and rope
    attn::CuArray{T,3}              # attention output (head_dim, heads, n)
    attn_workspace::Union{Nothing,CuArray{Float32,3}}
    proj::CuMatrix{T}               # output of wo or w_down (hidden, n)
    gate_up::CuMatrix{T}            # (2 intermediate, n); (0, n) for an ungated MLP
    act::CuMatrix{T}                # (intermediate, n)
    logits::CuMatrix{Float32}
    aux_layers::Vector{Int}
    aux::Union{Nothing,CuMatrix{T}}  # (length(aux_layers) hidden, n): the layers' outputs, stacked
end

StepBuffers(m::DecoderLM{T}, cache::PagedKVCache{T}, n::Integer; kwargs...) where {T} =
    StepBuffers{T}(m.config, m.backend, cache, n; kwargs...)

"""
    StepBuffers{T}(config, backend, cache, n; all_logits = false, aux_layers = Int[], vocab = config.vocab_size)

[`StepBuffers`](@ref) from a configuration, for drafters that are not a DecoderLM; `vocab` sizes the logits.
"""
function StepBuffers{T}(cfg::ModelConfig, backend::Backend, cache::PagedKVCache{T}, n::Integer;
                        all_logits::Bool = false, aux_layers = Int[], vocab::Integer = cfg.vocab_size) where {T}
    H, D, nq, nkv, I = cfg.hidden_size, cfg.head_dim, cfg.num_heads, cfg.num_kv_heads, cfg.intermediate_size
    m_out = all_logits ? n : 1
    dev(S, dims...) = CuArray{S}(undef, dims...)
    # A fixed table length, a multiple of 16: cuTile kernels specialize on the divisibility
    # of each array's length, so a table that grew with the sequence would recompile them.
    table_len = cld(num_pages(cache), 16) * 16
    a = cld(n, 32) * 32
    meta = CUDA.zeros(Int32, 3a + table_len)
    staging = CuArray{Int32,1,CUDA.HostMemory}(undef, length(meta))
    staging_host = unsafe_wrap(Array, staging)
    fill!(staging_host, 0)
    q = dev(T, D, nq, n)
    return StepBuffers{T}(meta, staging, staging_host, CuEvent(), a, view(meta, 1:n), view(meta, a+1:a+n),
                          view(meta, 2a+1:2a+n), view(meta, 3a+1:3a+table_len),
                          dev(T, H, n), dev(T, H, n), dev(T, (nq + 2nkv) * D, n),
                          q, dev(T, D, nq, n),
                          attention_workspace(backend, q, first(cache.k)), dev(T, H, n),
                          dev(T, cfg.hidden_act == :silu ? 2I : 0, n),
                          dev(T, I, n), dev(Float32, vocab, m_out), collect(Int, aux_layers),
                          isempty(aux_layers) ? nothing : dev(T, length(aux_layers) * H, n))
end

num_tokens(b::StepBuffers) = length(b.ids)

"""
    prepare_step!(buffers, model, cache, seq, ids)
    prepare_step!(buffers, model, cache, seq, nothing)

Reserve cache pages for the 0-based token `ids` after the positions of `seq`, and upload
the step's inputs into `buffers` with one asynchronous copy, ordered before the step on the
stream. With `nothing`, the ids are already on the device (written by a previous step's
sampling). Host side only; `seq.tokens` is not changed.
"""
function prepare_step!(b::StepBuffers, model, cache::PagedKVCache, seq::Sequence,
                       ids::Union{Nothing,AbstractVector{<:Integer}})
    n = num_tokens(b)
    ids === nothing || length(ids) == n || throw(ArgumentError("buffers are for $n tokens, got $(length(ids))"))
    start = length(seq)
    start + n <= max_length(model) ||
        throw(ArgumentError("sequence of $(start + n) tokens exceeds the model's max length $(max_length(model))"))
    reserve!(cache.blocks, seq, n)
    synchronize(b.uploaded)          # the previous copy out of `staging` has run
    h, a = b.staging_host, b.segment
    ids === nothing || (h[1:n] .= ids)
    h[a+1:a+n] .= start:start+n-1
    h[2a+1:2a+n] .= slots(cache.blocks, seq, start, n)
    table = view(h, 3a+1:length(h))
    fill!(table, 0)
    copyto!(table, seq.block_table)
    first = ids === nothing ? a + 1 : 1
    copyto!(b.meta, first, b.staging, first, length(h) - first + 1)
    record(b.uploaded)
    return b
end

"""
    run_step!(model, cache, buffers) -> logits

The forward pass of a prepared step, entirely on the device and only writing into
`buffers` and the KV cache. Returns `buffers.logits`.
"""
function run_step!(model::DecoderLM{T}, cache::PagedKVCache{T}, b::StepBuffers{T}) where {T}
    cfg, w, be = model.config, model.weights, model.backend
    n, eps = num_tokens(b), cfg.rms_norm_eps
    D, nq, nkv = cfg.head_dim, cfg.num_heads, cfg.num_kv_heads
    scale = 1 / sqrt(D)
    flat(a) = reshape(a, :, n)

    embed!(be, b.x, w.embed, b.ids)                                 # (hidden, n)
    rmsnorm!(be, b.h, b.x, first(w.layers).attn_norm, eps)
    for (l, layer) in enumerate(w.layers)
        linear!(be, b.qkv, layer.wqkv, b.h)
        attention_inputs!(be, b.q, cache.k[l], cache.v[l], b.qkv, layer.q_norm, layer.k_norm, model.rope,
                          b.positions, b.slots, eps)
        attention!(be, b.attn, b.attn_workspace, b.q, cache.k[l], cache.v[l], b.block_table,
                   b.positions, scale)
        linear!(be, b.proj, layer.wo, flat(b.attn))
        add_rmsnorm!(be, b.x, b.h, b.proj, layer.mlp_norm, eps)    # x += proj; h = norm(x)

        if layer.xielu === nothing
            linear_swiglu!(be, b.act, b.gate_up, layer.w_gate_up, b.h)
        else
            linear_xielu!(be, b.act, layer.w_gate_up, b.h, layer.xielu)
        end
        linear!(be, b.proj, layer.w_down, b.act)
        # The next layer's input norm, or the final norm after the last layer.
        add_rmsnorm!(be, b.x, b.h, b.proj, l < length(w.layers) ? w.layers[l+1].attn_norm : w.norm, eps)
        save_aux!(b, l)
    end

    last = size(b.logits, 2) == n ? b.h : view(b.h, :, n:n)        # a column view is a CuArray
    return linear!(be, b.logits, w.lm_head, last)                  # Float32 logits
end

"""
Keep the residual stream after layer l if it is one of the step's aux layers.
"""
function save_aux!(b::StepBuffers, l)
    b.aux === nothing && return
    i = findfirst(==(l), b.aux_layers)
    i === nothing && return
    H = size(b.x, 1)
    view(b.aux, (i-1)*H+1:i*H, :) .= b.x
    return
end

"""
    forward!(model, cache, seq, ids; all_logits = false) -> CuMatrix{Float32}

Run the 0-based token `ids` through the model at the positions following those of `seq`,
storing their keys and values in `cache` and appending them to `seq`. Returns the logits
(vocab, n) of every token if `all_logits`, otherwise (vocab, 1) of the last token. Runs
eagerly with buffers allocated for this step; [`ModelRunner`](@ref) reuses them.
"""
function forward!(model::DecoderLM{T}, cache::PagedKVCache{T}, seq::Sequence,
                  ids::AbstractVector{<:Integer}; all_logits::Bool = false) where {T}
    n = length(ids)
    n > 0 || throw(ArgumentError("forward! needs at least one token"))
    b = StepBuffers(model, cache, n; all_logits)
    prepare_step!(b, model, cache, seq, ids)
    run_step!(model, cache, b)
    append!(seq.tokens, ids)
    return b.logits
end

# ---- src/kv/paged_cache.jl ----

# Paged KV cache: per layer, one key and one value pool of shape
# (head_dim, page_size, kv_heads, num_pages), shared by all sequences through a
# BlockManager. A page holds `page_size` consecutive positions of one sequence, so the
# keys of a page and KV head form a dense (head_dim, page_size) tile.

"Positions per KV-cache page unless the cache is built with another `page_size`."
const DEFAULT_PAGE_SIZE = 64

"""
    PagedKVCache{T}(config, num_pages; page_size = $DEFAULT_PAGE_SIZE, quantize = nothing, key_smoothing = nothing)

Key/value page pools for every layer and the allocator for their pages, for a model whose
activations are `T`. The pools hold `T`, or with `quantize = :int8` Int8 values with one
Float32 scale per cached vector ([`Int8Pages`](@ref)): about half the memory, and half the
bytes decode attention reads. `key_smoothing` holds each layer's smoothing of the keys of
an INT8 cache (see [`key_smoothing`](@ref)), or `nothing` for none.
"""
struct PagedKVCache{T,A}
    k::Vector{A}
    v::Vector{A}
    blocks::BlockManager
end

function PagedKVCache{T}(cfg::ModelConfig, num_pages::Integer; page_size::Integer = DEFAULT_PAGE_SIZE,
                         quantize::Union{Nothing,Symbol} = nothing, key_smoothing = nothing) where {T}
    blocks = BlockManager(num_pages, page_size)
    dims = (cfg.head_dim, page_size, cfg.num_kv_heads, num_pages)
    if quantize === nothing
        k, v = ([CUDA.zeros(T, dims...) for _ in 1:cfg.num_layers] for _ in 1:2)
        return PagedKVCache{T,eltype(k)}(k, v, blocks)
    end
    quantize === :int8 || throw(ArgumentError("unsupported KV cache quantization $quantize (only :int8)"))
    pairs = [int8_kv_pools(dims...; key_smoothing = key_smoothing === nothing ? nothing : key_smoothing[l])
             for l in 1:cfg.num_layers]
    return PagedKVCache{T,Int8Pages}(first.(pairs), last.(pairs), blocks)
end

"`:int8` for an INT8 cache, `nothing` for one in the activations' type."
quantization(::PagedKVCache{T,Int8Pages}) where {T} = :int8
quantization(::PagedKVCache) = nothing

"Bytes of cache memory per cached position (keys and values of every layer)."
bytes_per_token(cfg::ModelConfig, ::Type{T}) where {T} =
    2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * sizeof(T)
bytes_per_token(cfg::ModelConfig, quantize::Symbol) =
    quantize === :int8 ? 2 * cfg.num_layers * cfg.num_kv_heads * (cfg.head_dim + sizeof(Float32)) :
    throw(ArgumentError("unsupported KV cache quantization $quantize"))

"Positions per page of `cache`."
page_size(c::PagedKVCache) = c.blocks.page_size
"Total pages of `cache` (free or held by a sequence)."
num_pages(c::PagedKVCache) = c.blocks.num_pages

"A new, empty sequence in `cache`."
new_sequence(::PagedKVCache) = Sequence(Int32[], Int[])

truncate!(c::PagedKVCache, seq::Sequence, n::Integer) = truncate!(c.blocks, seq, n)
release!(c::PagedKVCache, seq::Sequence) = release!(c.blocks, seq)

# ---- src/kv/block_manager.jl ----

# Page allocator for the paged KV cache. The cache is a pool of fixed-size pages; each
# sequence owns a block table listing, in order, the pages that hold its positions:
# position p (0-based) lives in page block_table[p ÷ page_size + 1] at offset p % page_size.
# Pages need not be contiguous or ordered, so sequences can grow, shrink and share the
# pool without moving data.

"""
    Sequence

A sequence's place in the paged KV cache: its block table and the token ids whose keys
and values are cached, in position order.
"""
mutable struct Sequence
    block_table::Vector{Int32}       # 1-based page ids
    tokens::Vector{Int}              # 0-based token ids at positions 0, 1, …
end

Base.length(s::Sequence) = length(s.tokens)

"""
    OutOfPagesError(needed, free)

Thrown by [`reserve!`](@ref) when the KV cache has fewer free pages than a step needs. A
[`ChatSession`](@ref) can recover with [`reset!`](@ref).
"""
struct OutOfPagesError <: Exception
    needed::Int
    free::Int
end

Base.showerror(io::IO, e::OutOfPagesError) =
    print(io, "KV cache full: need $(e.needed) more pages, $(e.free) free")

"""
    BlockManager(num_pages, page_size)

Hands out the pages of a pool of `num_pages` pages of `page_size` positions each.
"""
struct BlockManager
    page_size::Int
    num_pages::Int
    free::Vector{Int32}              # stack of free page ids
end

# Reversed so that pages are handed out in increasing order (easier to read in tests).
BlockManager(num_pages::Integer, page_size::Integer) =
    BlockManager(page_size, num_pages, collect(Int32, num_pages:-1:1))

"Number of pages not held by any sequence."
num_free_pages(b::BlockManager) = length(b.free)

"""
    reserve!(blocks, seq, n)

Make sure `seq` has pages for `n` more positions; throws an [`OutOfPagesError`](@ref) when
the pool runs out.
"""
function reserve!(b::BlockManager, seq::Sequence, n::Integer)
    needed = cld(length(seq) + n, b.page_size) - length(seq.block_table)
    needed > num_free_pages(b) && throw(OutOfPagesError(needed, num_free_pages(b)))
    for _ in 1:needed
        push!(seq.block_table, pop!(b.free))
    end
    return seq
end

"""
    truncate!(blocks_or_cache, seq, n)

Drop the cached positions of `seq` after the first `n` and return its unused pages to the
pool. Speculative decoding uses it to roll back rejected drafts, and chat to roll back to
the prefix a new turn shares with the cache.
"""
function truncate!(b::BlockManager, seq::Sequence, n::Integer)
    0 <= n <= length(seq) || throw(ArgumentError("cannot truncate a sequence of length $(length(seq)) to $n"))
    resize!(seq.tokens, n)
    keep = cld(n, b.page_size)
    append!(b.free, reverse!(seq.block_table[keep+1:end]))
    resize!(seq.block_table, keep)
    return seq
end

"""
    release!(blocks_or_cache, seq)

Return all pages of `seq` to the pool.
"""
release!(b::BlockManager, seq::Sequence) = truncate!(b, seq, 0)

"""
    slots(b, seq, start, n) -> Vector{Int32}

Cache slots of positions `start:start+n-1` of `seq` (pages must be reserved). Slot
`(page - 1) * page_size + offset` with a 1-based offset identifies one position of the
pool; it is what [`store_kv!`](@ref) consumes.
"""
function slots(b::BlockManager, seq::Sequence, start::Integer, n::Integer)
    P = b.page_size
    return Int32[(seq.block_table[p ÷ P + 1] - 1) * P + p % P + 1 for p in start:start+n-1]
end
