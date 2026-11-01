# Device-resident weights of a Llama-style decoder, in TileInfer's column-major layout:
# a linear layer is stored (in, out) and applied as y = Wᵀx. Projections that read the
# same input are fused into one matrix at load time (q/k/v, gate/up). Linear weights `M`
# are matrices of T, or Int8Weight when the model is loaded with `quantize = :int8`.

"""
    LayerWeights{T,M,V}

The weights of one decoder layer: its two RMSNorm weights, the fused `wqkv` and `w_gate_up`
projections, `wo` and `w_down`, the optional Qwen3 q/k norms and the optional xIELU scalars
(Apertus). Linear weights are stored `(in, out)`; see the field comments for shapes.
"""
struct LayerWeights{T,M,V<:AbstractVector{T}}
    attn_norm::V                 # (hidden,)
    wqkv::M                      # (hidden, (heads + 2 kv_heads) * head_dim): q | k | v columns
    q_norm::Union{Nothing,V}     # (head_dim,), Qwen3 only
    k_norm::Union{Nothing,V}     # (head_dim,), Qwen3 only
    wo::M                        # (heads * head_dim, hidden)
    mlp_norm::V                  # (hidden,)
    w_gate_up::M                 # (hidden, 2 intermediate): gate | up columns; only up,
                                 # (hidden, intermediate), for an ungated MLP (`xielu` set)
    w_down::M                    # (intermediate, hidden)
    xielu::Union{Nothing,XIELU{T}}  # Apertus
end

"""
    ModelWeights{T,E,M,V}

All weights of a decoder on the device: the embedding table `(hidden, vocab)`, the
[`LayerWeights`](@ref) of each layer, the final norm and the `lm_head`. Made by
[`load_weights`](@ref).
"""
struct ModelWeights{T,E<:AbstractMatrix{T},M,V<:AbstractVector{T}}
    embed::E                     # (hidden, vocab): column t is the embedding of token t-1
    layers::Vector{LayerWeights{T,M,V}}
    norm::V                      # (hidden,)
    lm_head::M                   # (hidden, vocab); the same array as `embed` when tied, unless
end                              # quantized (the embedding gather stays in T)

Base.eltype(::ModelWeights{T}) where {T} = T

"""
The distinct device arrays and Int8 weights of `w` (a tied `lm_head` is listed once).
"""
function distinct_weights(w::ModelWeights)
    arrays = IdDict{Any,Nothing}()
    for a in (w.embed, w.norm, w.lm_head)
        arrays[a] = nothing
    end
    for l in w.layers, name in fieldnames(LayerWeights)
        a = getfield(l, name)
        a isa Union{AbstractArray,Int8Weight} && (arrays[a] = nothing)
    end
    return keys(arrays)
end

"Total bytes of the distinct device arrays in `w` (a tied `lm_head` is counted once)."
weight_bytes(w::ModelWeights) = sum(sizeof, distinct_weights(w))

"Number of weight elements (parameters) in `w`; an Int8 weight's scales are not counted."
weight_count(w::ModelWeights) = sum(a -> a isa Int8Weight ? length(a.q) : length(a), distinct_weights(w))

"`:int8` if the model's linear weights are quantized, else `nothing`."
quantization(w::ModelWeights) = w.lm_head isa Int8Weight ? :int8 : nothing
