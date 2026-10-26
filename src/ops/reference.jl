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
function linear(::RefBackend, W::CuMatrix{T}, x::CuMatrix{T}, bias = nothing) where {T}
    y = CuMatrix{T}(undef, size(W, 2), size(x, 2))
    CUBLAS.gemmEx!('T', 'N', true, W, x, false, y)
    bias === nothing || (y .+= bias)
    return y
end

function rope(::RefBackend, x::CuArray{T,3}, cos::CuMatrix{T}, sin::CuMatrix{T}) where {T}
    half = size(x, 1) ÷ 2
    rotated = vcat(-x[half+1:end, :, :], x[1:half, :, :])
    c = reshape(cos, size(cos, 1), 1, size(cos, 2))
    s = reshape(sin, size(sin, 1), 1, size(sin, 2))
    return x .* c .+ rotated .* s       # each op rounds to T, as in PyTorch
end

function silu_mul(::RefBackend, gate_up::CuMatrix{T}) where {T}
    n = size(gate_up, 1) ÷ 2
    gate, up = view(gate_up, 1:n, :), view(gate_up, n+1:2n, :)
    return T.(silu.(Float32.(gate))) .* up
end

silu(x) = x / (1 + exp(-x))

# Eager attention, per KV head over its group of query heads at once:
#   S = round(round(KᵀQ) * scale),  P = round(softmax(S + mask) in Float32),  O = round(V P)
function attention(::RefBackend, q::CuArray{T,3}, k_cache::CuArray{T,3}, v_cache::CuArray{T,3},
                   start::Integer, scale::Real) where {T}
    D, heads, n = size(q)
    kv_heads = size(k_cache, 3)
    group = heads ÷ kv_heads
    len = start + n
    q_by_head = permutedims(q, (1, 3, 2))                    # (D, n, heads)
    out = similar(q_by_head)
    # Key i is visible to query j (position start + j) iff i <= start + j.
    visible = CuArray([i <= start + j for i in 1:len, j in 1:n])
    for h in 1:kv_heads
        heads_h = (h-1)*group+1:h*group
        K = Float32.(view(k_cache, :, 1:len, h))             # (D, len)
        V = Float32.(view(v_cache, :, 1:len, h))
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
