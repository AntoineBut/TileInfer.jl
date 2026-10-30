# Loads safetensors checkpoints straight into device arrays.
#
# Safetensors stores tensors row-major. SafeTensors.jl presents them in their logical
# (PyTorch) shape as a lazy PermutedDimsArray over the memory-mapped file; its parent is
# the same bytes seen column-major with the shape reversed, which is exactly TileInfer's
# layout. So weights are uploaded from the parent, without any transpose.

using SafeTensors: SafeTensors

"""
    Checkpoint

Lazily opened, memory-mapped tensors of every `*.safetensors` file in a model directory,
by name. Nothing is read from disk until a tensor is uploaded.
"""
struct Checkpoint
    tensors::Dict{String,AbstractArray}
end

function Checkpoint(dir::AbstractString)
    files = filter(endswith(".safetensors"), readdir(dir; join = true))
    isempty(files) && error("no .safetensors files in $dir")
    tensors = Dict{String,AbstractArray}()
    for file in files, (name, tensor) in SafeTensors.deserialize(file; mmap = true)
        haskey(tensors, name) && error("tensor $name appears in more than one shard")
        tensors[name] = tensor
    end
    return Checkpoint(tensors)
end

Base.haskey(c::Checkpoint, name) = haskey(c.tensors, name)
Base.keys(c::Checkpoint) = keys(c.tensors)

"Tensor `name` in file layout: column-major with the PyTorch shape reversed."
function file_layout(c::Checkpoint, name::AbstractString)
    haskey(c, name) || error("checkpoint has no tensor $name")
    # Unwrap SafeTensors' little-endian mapping, then its row-major permutation.
    return parent(parent(c.tensors[name]))
end

# Bytes of `x` as a host Array{T} that can be memcpy'd to the device: zero-copy when the
# file already holds T, a converting copy otherwise.
host_array(::Type{T}, x::AbstractArray{T}) where {T} = x
host_array(::Type{T}, x::AbstractArray) where {T} = Array{T}(x)

# Copies `x` into consecutive elements of `dst` starting at linear index `offset`.
function upload!(dst::CuArray{T}, offset::Int, x::AbstractArray) where {T}
    src = host_array(T, x)
    n = Int(length(src))   # views into the mmap'd file are indexed with UInt64
    GC.@preserve src begin
        # The mmap'd file is contiguous, so a raw pointer + length describes the tensor.
        host = unsafe_wrap(Array, pointer(src), n)
        copyto!(dst, offset, host, 1, n)
    end
    return offset + n
end

"Upload tensor `name` as a device array of element type `T`, in file layout."
function upload(::Type{T}, c::Checkpoint, name::AbstractString) where {T}
    x = file_layout(c, name)
    dst = CuArray{T}(undef, size(x))
    upload!(dst, 1, x)
    return dst
end

"""
Upload several 2-D tensors as one matrix whose columns are the tensors' columns side by
side. In file layout a PyTorch (out, in) weight is (in, out), so this concatenates along
the output dimension: the fused q|k|v and gate|up projections.
"""
function upload_fused(::Type{T}, c::Checkpoint, names) where {T}
    parts = [file_layout(c, n) for n in names]
    rows = size(first(parts), 1)
    all(p -> ndims(p) == ndims(first(parts)) && size(p, 1) == rows, parts) ||
        error("cannot fuse tensors of shapes $(map(size, parts))")
    dims = ndims(first(parts)) == 1 ? (sum(length, parts),) : (rows, sum(p -> size(p, 2), parts))
    dst = CuArray{T}(undef, dims)
    offset = 1
    for p in parts
        offset = upload!(dst, offset, p)
    end
    return dst
end

function check_size(x, expected, name)
    size(x) == expected || error("$name has size $(size(x)), expected $expected from config")
    return x
end

"""
    load_weights(T, config, dir) -> ModelWeights{T}

Read the checkpoint in `dir` onto the GPU with element type `T`, fusing q/k/v and gate/up.
"""
function load_weights(::Type{T}, cfg::ModelConfig, dir::AbstractString) where {T}
    c = Checkpoint(dir)
    H = cfg.hidden_size
    layers = [load_layer(T, c, cfg, "model.layers.$i.") for i in 0:cfg.num_layers-1]
    embed = check_size(upload(T, c, "model.embed_tokens.weight"), (H, cfg.vocab_size), "embed")
    # Some tied checkpoints (Qwen3-0.6B) still store a copy of lm_head; like transformers,
    # ignore it rather than spend a second vocab × hidden matrix of memory.
    lm_head = if cfg.tie_word_embeddings
        embed
    else
        check_size(upload(T, c, "lm_head.weight"), (H, cfg.vocab_size), "lm_head")
    end
    norm = check_size(upload(T, c, "model.norm.weight"), (H,), "norm")
    return ModelWeights(embed, layers, norm, lm_head)
end

"The weights of the decoder layer whose tensors are named `prefix * \"self_attn.q_proj.weight\"`, …"
function load_layer(::Type{T}, c::Checkpoint, cfg::ModelConfig, p::AbstractString) where {T}
    H, I, D = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim
    attn = p * "self_attn."
    qkv_names = [attn * "$(x)_proj.weight" for x in ("q", "k", "v")]
    return LayerWeights(
        check_size(upload(T, c, p * "input_layernorm.weight"), (H,), "attn_norm"),
        check_size(upload_fused(T, c, qkv_names), (H, (cfg.num_heads + 2cfg.num_kv_heads) * D), "wqkv"),
        cfg.qkv_bias ? upload_fused(T, c, replace.(qkv_names, "weight" => "bias")) : nothing,
        cfg.qk_norm ? check_size(upload(T, c, attn * "q_norm.weight"), (D,), "q_norm") : nothing,
        cfg.qk_norm ? check_size(upload(T, c, attn * "k_norm.weight"), (D,), "k_norm") : nothing,
        check_size(upload(T, c, attn * "o_proj.weight"), (cfg.num_heads * D, H), "wo"),
        check_size(upload(T, c, p * "post_attention_layernorm.weight"), (H,), "mlp_norm"),
        check_size(upload_fused(T, c, [p * "mlp.gate_proj.weight", p * "mlp.up_proj.weight"]),
                   (H, 2I), "w_gate_up"),
        check_size(upload(T, c, p * "mlp.down_proj.weight"), (I, H), "w_down"),
    )
end
