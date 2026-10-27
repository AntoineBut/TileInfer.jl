# Reference backend: plain CUDA.jl broadcasts and cuBLAS, written for clarity and to
# reproduce Hugging Face transformers' numerics, not for speed. Each op rounds to the
# element type `T` at the same points as the PyTorch implementation (eager attention), so
# with T = BFloat16 the results match transformers' bfloat16 run up to accumulation
# order. It allocates freely; the TileBackend path is the one kept allocation-free.

using LinearAlgebra: LinearAlgebra
using CUDA: CUBLAS

embed(::RefBackend, table::CuMatrix, ids::CuVector{<:Integer}) = table[:, ids .+ 1]

# Qwen3RMSNorm / LlamaRMSNorm: normalize in Float32, round to T, then scale in T.
function rmsnorm(::RefBackend, x::CuArray{T}, w::CuVector{T}, eps::Real) where {T}
    x32 = Float32.(x)
    inv_rms = 1f0 ./ sqrt.(sum(abs2, x32; dims = 1) ./ size(x, 1) .+ Float32(eps))
    return w .* T.(x32 .* inv_rms)
end

# cuBLAS GEMM with Float32 accumulation. Not `mul!`: for BFloat16, CUDA.jl's `mul!` falls
# back to a generic kernel that accumulates in BFloat16 (~30 % error at K = 1024).
linear!(::RefBackend, y::CuMatrix{T}, W::CuMatrix{T}, x::CuMatrix{T}) where {T} =
    CUBLAS.gemmEx!('T', 'N', true, W, x, false, y)

function linear(be::RefBackend, W::CuMatrix{T}, x::CuMatrix{T}, bias = nothing) where {T}
    y = linear!(be, CuMatrix{T}(undef, size(W, 2), size(x, 2)), W, x)
    bias === nothing || (y .+= bias)
    return y
end

function rope(::RefBackend, x::CuArray{T,3}, cos::CuMatrix{T}, sin::CuMatrix{T},
              positions::CuVector{Int32}) where {T}
    half, n = size(x, 1) ÷ 2, size(x, 3)
    rotated = vcat(-x[half+1:end, :, :], x[1:half, :, :])
    c = reshape(cos[:, positions .+ Int32(1)], size(cos, 1), 1, n)
    s = reshape(sin[:, positions .+ Int32(1)], size(sin, 1), 1, n)
    return x .* c .+ rotated .* s       # each op rounds to T, as in PyTorch
end

function silu_mul(::RefBackend, gate_up::CuMatrix{T}) where {T}
    n = size(gate_up, 1) ÷ 2
    gate, up = view(gate_up, 1:n, :), view(gate_up, n+1:2n, :)
    return T.(silu.(Float32.(gate))) .* up
end

silu(x) = x / (1 + exp(-x))

# Scatter through a flat column view of the pools: the column of (offset o, head h, page p),
# all 0-based, in (head_dim, page_size × kv_heads × num_pages) is o + P h + P H p + 1.
function store_kv!(::RefBackend, k_pages::CuArray{T,4}, v_pages::CuArray{T,4},
                   k::CuArray{T,3}, v::CuArray{T,3}, slots::CuVector{Int32}) where {T}
    D, P, H, _ = size(k_pages)
    n = length(slots)
    page, offset = (slots .- 1) .÷ P, (slots .- 1) .% P
    heads = CuArray(0:H-1)
    cols = vec(reshape(offset, 1, n) .+ P .* heads .+ (P * H) .* reshape(page, 1, n) .+ 1)  # (h, t)
    reshape(k_pages, D, :)[:, cols] = reshape(k, D, H * n)
    reshape(v_pages, D, :)[:, cols] = reshape(v, D, H * n)
    return nothing
end

# Keys (or values) of positions 1:len of one KV head, gathered from the sequence's pages
# into a dense (head_dim, len) matrix.
function gather_pages(pages::CuArray{T,4}, block_table::CuVector{Int32}, h::Integer, len::Integer) where {T}
    D, P = size(pages, 1), size(pages, 2)
    used = view(block_table, 1:cld(len, P))
    return view(reshape(pages[:, :, h, used], D, :), :, 1:len)
end

# Eager attention, per KV head over its group of query heads at once:
#   S = round(round(KᵀQ) * scale),  P = round(softmax(S + mask) in Float32),  O = round(V P)
function attention(::RefBackend, q::CuArray{T,3}, k_pages::CuArray{T,4}, v_pages::CuArray{T,4},
                   block_table::CuVector{Int32}, positions::CuVector{Int32}, scale::Real) where {T}
    D, heads, n = size(q)
    start = Int(Array(view(positions, 1:1))[1])              # host sync: not graph-capturable
    kv_heads = size(k_pages, 3)
    group = heads ÷ kv_heads
    len = start + n
    q_by_head = permutedims(q, (1, 3, 2))                    # (D, n, heads)
    out = similar(q_by_head)
    # Key i is visible to query j (position start + j) iff i <= start + j.
    visible = CuArray([i <= start + j for i in 1:len, j in 1:n])
    for h in 1:kv_heads
        heads_h = (h-1)*group+1:h*group
        K = Float32.(gather_pages(k_pages, block_table, h, len))   # (D, len)
        V = Float32.(gather_pages(v_pages, block_table, h, len))
        Q = Float32.(reshape(q_by_head[:, :, heads_h], D, n * group))
        S = T.(T.(K' * Q) .* T(scale))                       # (len, n * group)
        S = ifelse.(reshape(visible, len, n, 1), Float32.(reshape(S, len, n, group)), -Inf32)
        E = exp.(S .- maximum(S; dims = 1))
        P = T.(E ./ sum(E; dims = 1))
        O = T.(V * Float32.(reshape(P, len, n * group)))     # (D, n * group)
        out[:, :, heads_h] .= reshape(O, D, n, group)
    end
    return permutedims(out, (1, 3, 2))                       # (D, heads, n)
end
