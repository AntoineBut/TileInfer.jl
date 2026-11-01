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

"""
Copies `x` into consecutive elements of `dst` starting at linear index `offset`.
"""
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

"""
Upload tensor `name` as a device array of element type `T`, in file layout. With `cols`,
only its first `cols` columns (rows of the PyTorch tensor), which are contiguous in the file.
"""
function upload(::Type{T}, c::Checkpoint, name::AbstractString; cols::Union{Nothing,Integer} = nothing) where {T}
    x = file_layout(c, name)
    if cols !== nothing
        cols <= size(x, 2) || error("$name has $(size(x, 2)) columns, cannot take the first $cols")
        x = view(x, :, 1:cols)
    end
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

check_quantize(quantize) = quantize in (nothing, :int8) ||
    throw(ArgumentError("unsupported quantization $quantize; supported: :int8"))

"""
Quantize a freshly uploaded weight and free its T copy at once, so that loading a model
that only fits quantized never holds more than one T matrix at a time.
"""
function quantized(W::CuMatrix, quantize::Union{Nothing,Symbol})
    check_quantize(quantize)
    quantize === nothing && return W
    Wq = quantize_int8(W)
    CUDA.unsafe_free!(W)
    return Wq
end

const QUANTIZE_COLUMNS = 16384   # 128 MB of BFloat16 at 4096 rows

"""
Tensor `name` uploaded in file layout and stored as an Int8Weight (or as a T matrix when
`quantize === nothing`), with only its first `cols` columns if given. Quantization is per
column, so it is done block by block: the T copy on the device never exceeds
`QUANTIZE_COLUMNS` columns (a 131k-token `lm_head` is 1 GB in BFloat16).
"""
function upload_weight(::Type{T}, c::Checkpoint, name::AbstractString, quantize::Union{Nothing,Symbol};
                       cols::Union{Nothing,Integer} = nothing) where {T}
    check_quantize(quantize)
    quantize === nothing && return upload(T, c, name; cols)
    x = file_layout(c, name)
    n = something(cols, size(x, 2))
    n <= size(x, 2) || error("$name has $(size(x, 2)) columns, cannot take the first $n")
    W = Int8Weight(CuMatrix{Int8}(undef, size(x, 1), n), CuVector{Float32}(undef, n))
    for block in Iterators.partition(1:n, QUANTIZE_COLUMNS)
        part = CuMatrix{T}(undef, size(x, 1), length(block))
        upload!(part, 1, view(x, :, block))
        q = quantized(part, :int8)
        copyto!(view(W.q, :, block), q.q)
        copyto!(view(W.scale, block), q.scale)
        CUDA.unsafe_free!(q.q)
    end
    return W
end

"""
    load_weights(T, config, dir; quantize = nothing) -> ModelWeights{T}

Read the checkpoint in `dir` onto the GPU with element type `T`, fusing q/k/v and gate/up.
With `quantize = :int8`, the linear weights and the `lm_head` are stored as [`Int8Weight`](@ref)
(the embedding table stays in `T`).
"""
function load_weights(::Type{T}, cfg::ModelConfig, dir::AbstractString;
                      quantize::Union{Nothing,Symbol} = nothing) where {T}
    c = Checkpoint(dir)
    H = cfg.hidden_size
    # Multimodal checkpoints (Apertus 1.5) keep the decoder under `model.language_model.`.
    m = haskey(c, "model.language_model.embed_tokens.weight") ? "model.language_model." : "model."
    layers = [load_layer(T, c, cfg, "$(m)layers.$i."; quantize) for i in 0:cfg.num_layers-1]
    # Only the text rows: Apertus 1.5 also embeds image and audio codes after them.
    embed = check_size(upload(T, c, m * "embed_tokens.weight"; cols = cfg.vocab_size), (H, cfg.vocab_size), "embed")
    # Some tied checkpoints (Qwen3-0.6B) still store a copy of lm_head; like transformers,
    # ignore it rather than spend a second vocab × hidden matrix of memory.
    lm_head = if cfg.tie_word_embeddings
        quantize === nothing ? embed : upload_weight(T, c, m * "embed_tokens.weight", quantize; cols = cfg.vocab_size)
    else
        check_size(upload_weight(T, c, "lm_head.weight", quantize), (H, cfg.vocab_size), "lm_head")
    end
    norm = check_size(upload(T, c, m * "norm.weight"), (H,), "norm")
    return ModelWeights(embed, layers, norm, lm_head)
end

"The weights of the decoder layer whose tensors are named `prefix * \"self_attn.q_proj.weight\"`, …"
function load_layer(::Type{T}, c::Checkpoint, cfg::ModelConfig, p::AbstractString;
                    quantize::Union{Nothing,Symbol} = nothing) where {T}
    q(W) = quantized(W, quantize)
    H, I, D = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim
    attn = p * "self_attn."
    qkv_names = [attn * "$(x)_proj.weight" for x in ("q", "k", "v")]
    xielu = cfg.hidden_act == :xielu
    # Apertus names the two norms after their sublayers.
    attn_norm, mlp_norm = xielu ? ("attention_layernorm", "feedforward_layernorm") :
                                  ("input_layernorm", "post_attention_layernorm")
    mlp_in = xielu ? [p * "mlp.up_proj.weight"] : [p * "mlp.gate_proj.weight", p * "mlp.up_proj.weight"]
    return LayerWeights(
        check_size(upload(T, c, p * attn_norm * ".weight"), (H,), "attn_norm"),
        q(check_size(upload_fused(T, c, qkv_names), (H, (cfg.num_heads + 2cfg.num_kv_heads) * D), "wqkv")),
        cfg.qk_norm ? check_size(upload(T, c, attn * "q_norm.weight"), (D,), "q_norm") : nothing,
        cfg.qk_norm ? check_size(upload(T, c, attn * "k_norm.weight"), (D,), "k_norm") : nothing,
        check_size(upload_weight(T, c, attn * "o_proj.weight", quantize), (cfg.num_heads * D, H), "wo"),
        check_size(upload(T, c, p * mlp_norm * ".weight"), (H,), "mlp_norm"),
        q(check_size(upload_fused(T, c, mlp_in), (H, length(mlp_in) * I), "w_gate_up")),
        check_size(upload_weight(T, c, p * "mlp.down_proj.weight", quantize), (I, H), "w_down"),
        xielu ? load_xielu(T, c, p * "mlp.act_fn.") : nothing,
    )
end

"""
The activation's scalars as transformers' XIELUActivation computes them in dtype T.
"""
function load_xielu(::Type{T}, c::Checkpoint, p::AbstractString) where {T}
    scalar(name) = Float32(only(file_layout(c, p * name)))
    softplus(x) = log1p(exp(x))
    beta = T(scalar("beta"))
    return XIELU{T}(T(softplus(scalar("alpha_p"))), T(beta + T(softplus(scalar("alpha_n")))),
                    beta, T(scalar("eps")))
end
