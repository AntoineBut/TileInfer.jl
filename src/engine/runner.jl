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
    greedy::Any                      # GreedyPipeline, created on first use
end

function ModelRunner(model::DecoderLM{T}, cache::PagedKVCache{T};
                     graphs::Bool = supports_graphs(model.backend)) where {T}
    graphs && !supports_graphs(model.backend) &&
        throw(ArgumentError("$(nameof(typeof(model.backend))) steps cannot be captured in CUDA graphs"))
    return ModelRunner(model, cache, graphs, Dict{Int,CapturedStep{T}}(), nothing)
end

"""
    forward!(runner, seq, ids) -> CuMatrix{Float32}

[`forward!`](@ref) through `runner`: the logits (vocab, 1) of the last token. For short
steps they live in the step's static buffers and are overwritten by the next step of the
same length, so read them before running another step.
"""
function forward!(r::ModelRunner, seq::Sequence, ids::AbstractVector{<:Integer})
    n = length(ids)
    n > MAX_GRAPH_TOKENS && return NVTX.@range "prefill" forward!(r.model, r.cache, seq, ids)
    n > 0 || throw(ArgumentError("forward! needs at least one token"))
    step = get!(() -> CapturedStep(StepBuffers(r.model, r.cache, n), nothing), r.steps, n)
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

"Number of token counts with a captured CUDA graph."
num_graphs(r::ModelRunner) = count(s -> s.graph !== nothing, values(r.steps))

# --- Greedy decoding with the host one step behind the GPU ---
#
# Two sets of step buffers alternate. The step on set k samples its token on the device
# (argmax) straight into the ids of set 3 - k, and into `mirror[3 - k]` in host-mapped
# memory. So the host can launch the step that consumes a token before reading it: while
# the GPU runs that step, the host waits for the previous one, reads its token, checks for
# a stop token and streams it. The GPU no longer idles between steps.

mutable struct GreedyPipeline{T}
    buffers::NTuple{2,StepBuffers{T}}
    graphs::Vector{Union{Nothing,CUDA.CuGraphExec}}
    mirror::CuVector{Int32}          # host-mapped: mirror[k] is the token in buffers[k].ids
    mirror_host::Vector{Int32}
    scratch::Any                     # for argmax_token!
    done::NTuple{2,CuEvent}          # recorded after the step on each set
end

function greedy_pipeline(r::ModelRunner{T}) where {T}
    r.greedy === nothing || return r.greedy::GreedyPipeline{T}
    mirror = CuArray{Int32,1,CUDA.HostMemory}(undef, 2)
    bufs = (StepBuffers(r.model, r.cache, 1), StepBuffers(r.model, r.cache, 1))
    r.greedy = GreedyPipeline{T}(bufs, [nothing, nothing], mirror, unsafe_wrap(Array, mirror),
                                 argmax_scratch(r.model.config.vocab_size), (CuEvent(), CuEvent()))
    return r.greedy
end

# Sample the token of `logits` into set k (its ids and mirror slot).
sample_into!(r::ModelRunner, p::GreedyPipeline, k, logits) =
    argmax_token!(r.model.backend, p.buffers[k].ids, view(p.mirror, k:k), logits, p.scratch)

function run_greedy_step!(r::ModelRunner, p::GreedyPipeline, k)
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
    generate_greedy!(runner, seq, ids; max_new_tokens, stop_ids, on_token) -> Vector{Int}

[`generate!`](@ref) with greedy sampling, pipelined so that the host runs one step behind
the GPU. Needs a runner with CUDA graphs.
"""
function generate_greedy!(r::ModelRunner, seq::Sequence, ids::AbstractVector{<:Integer};
                          max_new_tokens::Integer, stop_ids, on_token = nothing)
    r.graphs || throw(ArgumentError("pipelined greedy decoding needs CUDA graphs"))
    p = greedy_pipeline(r)
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
            run_greedy_step!(r, p, k)
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
