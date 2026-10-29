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
    sample_host(logits, p, u) -> Int

The 0-based token sampled from `logits` with `p` by the uniform number `u` ∈ [0, 1): the
reference for the device samplers. Temperature, then top-k, then top-p (nucleus) over the
renormalized survivors, as in transformers' default logits-processor order; then inverse
transform sampling over the survivors, best first.
"""
function sample_host(logits::AbstractVector{<:Real}, p::SamplingParams, u::Real)
    p.temperature == 0 && return argmax(logits) - 1
    ids = top_candidates(logits, p.top_k == 0 ? length(logits) : min(p.top_k, length(logits)))
    scaled = Float64.(logits[ids]) ./ p.temperature
    probs = exp.(scaled .- scaled[1])
    probs ./= sum(probs)
    cum = cumsum(probs)
    if p.top_p < 1
        # Keep the smallest prefix whose mass reaches top_p (always at least one token).
        cum = cum[1:something(findfirst(>=(p.top_p), cum), length(cum))]
    end
    return ids[something(findfirst(>=(u * cum[end]), cum), length(cum))] - 1
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
