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

# lowbias32 (Chris Wellons): a bijective mix of 32 bits with low avalanche bias. (cuTile
# shifts need a shift count of the operand's type.)
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

# The ids of the k largest logits, best first, ties to the smaller id.
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

# Inverse transform sampling over weights `w` (best first) by u ∈ [0, 1): the first entry whose
# cumulative weight reaches u × the total, among those with positive weight (the last of them
# on round-off).
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
