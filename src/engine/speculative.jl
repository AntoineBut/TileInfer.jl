# Speculative decoding: a cheap proposer guesses the next k tokens, and one forward step of
# the target model on [last token, d1..dk] checks them all at once. With greedy decoding the
# target's own choice after each prefix is the argmax of that position's logits, so the
# drafts are accepted up to the first one that differs from it, and the target's choice at
# that point comes for free (the "bonus" token): each step emits 1 to k + 1 tokens for the
# cost of one step of k + 1 tokens, which at decode sizes costs little more than one token
# (the step reads the weights once either way). The keys and values of rejected drafts are
# dropped again by truncating the sequence: its pages go back to the pool, and later steps
# overwrite the stale slots.
#
# The proposer here is n-gram (prompt) lookup: no draft model, the drafts are the tokens
# that followed the latest earlier occurrence of the sequence's last n tokens.

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
    SpecStats()

Counters of a speculative generation: verify steps, drafted and accepted tokens.
"""
Base.@kwdef mutable struct SpecStats
    steps::Int = 0
    drafted::Int = 0
    accepted::Int = 0
end

acceptance_rate(s::SpecStats) = s.drafted == 0 ? NaN : s.accepted / s.drafted

# A step that returns the target model's greedy token after each of its n inputs: the
# forward step with the logits of every token, then an argmax per column, all captured
# together in a CUDA graph. The tokens land in host-mapped memory.
mutable struct VerifyStep{T}
    buffers::StepBuffers{T}
    graph::Union{Nothing,CUDA.CuGraphExec}
    targets::CuVector{Int32}         # device copies (`sample_token!` writes both)
    mirror::CuVector{Int32}          # host-mapped
    mirror_host::Vector{Int32}
    sampler::TokenSampler            # greedy
end

function VerifyStep(r::ModelRunner{T}, n::Integer) where {T}
    mirror = CuArray{Int32,1,CUDA.HostMemory}(undef, n)
    return VerifyStep{T}(StepBuffers(r.model, r.cache, n; all_logits = true), nothing, CuVector{Int32}(undef, n),
                         mirror, unsafe_wrap(Array, mirror), TokenSampler(r.model.config.vocab_size))
end

function run_verify_step!(r::ModelRunner, v::VerifyStep)
    b, be = v.buffers, r.model.backend
    run_step!(r.model, r.cache, b)
    for j in 1:num_tokens(b)
        sample_token!(be, view(v.targets, j:j), view(v.mirror, j:j), view(b.logits, :, j:j), v.sampler)
    end
end

"""
    verify!(runner, seq, ids) -> Vector{Int}

Run the 0-based token `ids` (the last sampled token, then the drafts) through the model,
appending them to `seq`, and return the greedy token after each of them. Steps of each
length are captured in a CUDA graph like [`forward!`](@ref) steps.
"""
function verify!(r::ModelRunner{T}, seq::Sequence, ids::AbstractVector{<:Integer}) where {T}
    n = length(ids)
    1 <= n <= MAX_GRAPH_TOKENS || throw(ArgumentError("verify! takes 1 to $MAX_GRAPH_TOKENS tokens, got $n"))
    v = get!(() -> VerifyStep(r, n), r.verify_steps, n)::VerifyStep{T}
    NVTX.@range "upload inputs" prepare_step!(v.buffers, r.model, r.cache, seq, ids)
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

Greedy [`generate!`](@ref) with drafts from `proposer` checked by [`verify!`](@ref): the
same tokens as plain greedy decoding up to near-ties (a verify step of k + 1 tokens runs
other kernels than a decode step, so the logits differ in the last bits), and `seq` ends in
the same state. Counts steps and accepted drafts in `stats`.
"""
function generate_speculative!(r::ModelRunner, seq::Sequence, ids::AbstractVector{<:Integer}, proposer;
                               max_new_tokens::Integer, stop_ids, on_token = nothing,
                               stats::SpecStats = SpecStats())
    generated = Int[]
    max_new_tokens > 0 || return generated
    logits = forward!(r, seq, ids)
    last = argmax(vec(Array(logits))) - 1    # sampled, not yet in the cache
    context = copy(seq.tokens)       # seq.tokens and `last`, for the proposer
    while true
        last in stop_ids && break
        push!(generated, last)
        on_token === nothing || on_token(last)
        remaining = max_new_tokens - length(generated)
        remaining == 0 && break
        push!(context, last)
        # A step emits at most k + 1 tokens: draft no more than the budget needs.
        drafts = remaining > 1 ? propose(proposer, context, min(proposer.k, remaining - 1)) : Int[]
        start = length(seq)
        targets = verify!(r, seq, [last; drafts])
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
