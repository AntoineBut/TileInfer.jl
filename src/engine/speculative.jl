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
# file); EAGLE-3 (eagle.jl).

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
    attach!(proposer, runner)

Prepare `proposer` to draft for the target model of `runner`, before a generation.
"""
attach!(::Any, ::ModelRunner) = nothing

# Run a proposer's `step`: replay `graph`, or run it eagerly and capture it when the proposer
# uses graphs. Returns the graph.
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

# Whether a proposer that can sample its drafts does so for `runner`: only when the target
# samples (with greedy decoding, both draft and target take the argmax).
samples_for(sample::Bool, r::ModelRunner) = sample && r.sampler.sampling.temperature > 0

# Reset a drafter's sampler for a generation of `runner`'s: the target's parameters when it
# samples its drafts, greedy otherwise.
reset_draft_sampler!(s::TokenSampler, sample::Bool, r::ModelRunner) =
    reset_sampler!(s, samples_for(sample, r) ? r.sampler.sampling : SamplingParams(), draft_seed(r.sampler.seed))

"Verify steps take up to this many tokens (a DFlash block); all are captured in CUDA graphs."
const MAX_VERIFY_TOKENS = 16

# A step that returns the token the target model samples after each of its n inputs: the
# forward step with the logits of every token, then the runner's sampler on each column
# (counters c, c + 1, …), all captured together in a CUDA graph. With the distributions of
# sampled drafts, the columns of drafted positions apply the p/q rule instead. The tokens
# land in host-mapped memory.
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
    verify!(runner, seq, ids, counter; dists = nothing) -> Vector{Int}

Run the 0-based token `ids` (the last sampled token, then the drafts) through the model,
appending them to `seq`, and return the token the runner's sampler picks after each of them,
with counters `counter`, `counter + 1`, … (the argmax when greedy). When the drafts were
sampled from the distributions `dists` (column j for draft j), the token after each input
but the last follows the p/q rule instead ([`accept_token!`](@ref)): the next draft if
accepted, else a different token. Steps of each length, up to $MAX_VERIFY_TOKENS tokens, are
captured in a CUDA graph like [`forward!`](@ref) steps.
"""
function verify!(r::ModelRunner{T}, seq::Sequence, ids::AbstractVector{<:Integer}, counter::Integer;
                 dists::Union{Nothing,DraftDists} = nothing) where {T}
    n = length(ids)
    1 <= n <= MAX_VERIFY_TOKENS || throw(ArgumentError("verify! takes 1 to $MAX_VERIFY_TOKENS tokens, got $n"))
    dists === nothing || n - 1 <= size(dists.ids, 2) || throw(ArgumentError("$(n - 1) drafts but $(size(dists.ids, 2)) distributions"))
    key = (n, dists === nothing ? UInt(0) : objectid(dists))
    v = get!(() -> VerifyStep(r, n, dists), r.verify_steps, key)::VerifyStep{T}
    r.last_step, r.last_start = v.buffers, length(seq)
    NVTX.@range "upload inputs" begin
        prepare_step!(v.buffers, r.model, r.cache, seq, ids)
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
        drafts = remaining > 1 ? propose(proposer, context, min(proposer.k, remaining - 1)) : Int[]
        start = length(seq)
        # The token after `last` is generated token number length(generated) (0-based).
        targets = verify!(r, seq, [last; drafts], length(generated); dists)
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

# Where the sample of chain step i (or of the forward step before the chain, i = 0) goes.
# A proposer that can sample always records q (one-hot when greedy), so that its captured
# chain serves greedy and sampled generations alike.
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
