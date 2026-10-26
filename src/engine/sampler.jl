# Token sampling. Greedy decoding runs on the device; stochastic sampling copies the
# logits to the host (reference implementation; GPU sampling kernels come later).

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

"Sample a 0-based token id from `logits` (a vector, on the device or the host)."
function sample(logits::AbstractVector{Float32}, p::SamplingParams, rng::AbstractRNG)
    p.temperature == 0 && return argmax(logits) - 1
    return sample_host(Array(logits), p, rng)
end

# Temperature, then top-k, then top-p (nucleus) over the renormalized survivors, as in
# transformers' default logits-processor order.
function sample_host(logits::Vector{Float32}, p::SamplingParams, rng::AbstractRNG)
    k = p.top_k == 0 ? length(logits) : min(p.top_k, length(logits))
    order = partialsortperm(logits, 1:k; rev = true)          # candidate ids, best first
    scaled = Float64.(logits[order]) ./ p.temperature
    probs = exp.(scaled .- maximum(scaled))
    probs ./= sum(probs)
    if p.top_p < 1
        # Keep the smallest prefix whose mass reaches top_p (always at least one token).
        keep = something(findfirst(>=(p.top_p), cumsum(probs)), length(probs))
        probs = probs[1:keep] ./ sum(probs[1:keep])
    end
    r = rand(rng) * sum(probs)
    i = something(findfirst(>=(r), cumsum(probs)), length(probs))
    return order[i] - 1
end
