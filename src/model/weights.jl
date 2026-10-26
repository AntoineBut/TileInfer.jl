# Device-resident weights of a Llama-style decoder, in TileInfer's column-major layout:
# a linear layer is stored (in, out) and applied as y = Wᵀx. Projections that read the
# same input are fused into one matrix at load time (q/k/v, gate/up).

struct LayerWeights{T,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    attn_norm::V                 # (hidden,)
    wqkv::M                      # (hidden, (heads + 2 kv_heads) * head_dim): q | k | v columns
    bqkv::Union{Nothing,V}       # ((heads + 2 kv_heads) * head_dim,), Qwen2 only
    q_norm::Union{Nothing,V}     # (head_dim,), Qwen3 only
    k_norm::Union{Nothing,V}     # (head_dim,), Qwen3 only
    wo::M                        # (heads * head_dim, hidden)
    mlp_norm::V                  # (hidden,)
    w_gate_up::M                 # (hidden, 2 intermediate): gate | up columns
    w_down::M                    # (intermediate, hidden)
end

struct ModelWeights{T,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    embed::M                     # (hidden, vocab): column t is the embedding of token t-1
    layers::Vector{LayerWeights{T,M,V}}
    norm::V                      # (hidden,)
    lm_head::M                   # (hidden, vocab); the same array as `embed` when tied
end

Base.eltype(::ModelWeights{T}) where {T} = T

"Total bytes of the distinct device arrays in `w` (a tied lm_head is counted once)."
function weight_bytes(w::ModelWeights)
    arrays = IdDict{Any,Nothing}()
    for a in (w.embed, w.norm, w.lm_head)
        arrays[a] = nothing
    end
    for l in w.layers, name in fieldnames(LayerWeights)
        a = getfield(l, name)
        a === nothing || (arrays[a] = nothing)
    end
    return sum(sizeof, keys(arrays))
end
