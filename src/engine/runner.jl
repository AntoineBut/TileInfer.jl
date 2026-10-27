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
struct ModelRunner{T,M<:DecoderLM{T},C<:PagedKVCache{T}}
    model::M
    cache::C
    graphs::Bool
    steps::Dict{Int,CapturedStep{T}}
end

function ModelRunner(model::DecoderLM{T}, cache::PagedKVCache{T};
                     graphs::Bool = supports_graphs(model.backend)) where {T}
    graphs && !supports_graphs(model.backend) &&
        throw(ArgumentError("$(nameof(typeof(model.backend))) steps cannot be captured in CUDA graphs"))
    return ModelRunner(model, cache, graphs, Dict{Int,CapturedStep{T}}())
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
