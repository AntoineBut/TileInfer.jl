# A backend for measuring what one feature of the TileBackend gains: the same cuTile kernels
# with one feature switched back to the plain version, so that a step differs from the full
# engine's in that feature only. Benchmarks only (bench/features.jl, bench/speculative.jl).

using CUDA
using CUDA: CUBLAS
using TileInfer
using TileInfer: Backend, RefBackend, TileBackend

const TILE = TileBackend()

"""
    Variant(; fused = true, attention = :tile, linear = :tile, verify = :decode)

TileBackend with features switched off:
- `fused = false`: chains of primitive cuTile kernels instead of the fused ones (attention
  inputs as 3 copies + q/k RMSNorm + 2 RoPE + KV write; residual add as a broadcast before
  RMSNorm; SwiGLU after the gate_up GEMV);
- `attention = :ref`: the reference attention (pages gathered into dense K/V, scores
  materialized, softmax by broadcasts; reads positions on the host, so no CUDA graphs);
- `linear = :cublas`: cuBLAS (`gemmEx!`, Float32 accumulation) for every linear layer;
- `verify = :prefill`: the prefill attention kernel for every step of more than one token,
  instead of the split-KV decode kernels for steps of up to 16 tokens.
"""
struct Variant <: Backend
    fused::Bool
    attention::Symbol
    linear::Symbol
    verify::Symbol
    scratch::IdDict{Any,Any}       # unfused intermediates, allocated by the first (eager) step
end

Variant(; fused = true, attention = :tile, linear = :tile, verify = :decode) =
    Variant(fused, attention, linear, verify, IdDict{Any,Any}())

TileInfer.supports_graphs(v::Variant) = v.attention == :tile

scratch(v::Variant, T, name, dims...) = get!(() -> CuArray{T}(undef, dims...), v.scratch, (name, T, dims))

# Ops that are the same kernels as TileBackend's.
for f in (:embed!, :rmsnorm!, :rope!, :silu_mul!, :xielu!, :store_kv!, :sample_token!, :sample_tokens!,
          :accept_token!, :embed, :rmsnorm, :rope, :silu_mul)
    @eval TileInfer.$f(::Variant, a, args...) = TileInfer.$f(TILE, a, args...)
end

# cuBLAS's alpha and beta, on the device (CUDA.jl's handles use device pointer mode).
# `gemmEx!` allocates them per call, which a CUDA graph capture cannot free.
const CUBLAS_ONE = CuRef{Float32}(1f0)
const CUBLAS_ZERO = CuRef{Float32}(0f0)

"""
`y = Wᵀx` by `cublasGemmEx` with Float32 accumulation, allocating nothing (capturable).
"""
function cublas_linear!(y::CuMatrix, W::CuMatrix{T}, x::CuMatrix{T}) where {T}
    (in, out), n = size(W), size(x, 2)
    CUBLAS.cublasGemmEx(CUBLAS.handle(), 'T', 'N', out, n, in, CUBLAS_ONE, W, T, stride(W, 2), x, T, stride(x, 2),
                        CUBLAS_ZERO, y, eltype(y), stride(y, 2), CUBLAS.CUBLAS_COMPUTE_32F, CUBLAS.CUBLAS_GEMM_DEFAULT)
    return y
end

function TileInfer.linear!(v::Variant, y::CuMatrix, W::CuMatrix{T}, x::CuMatrix{T}) where {T}
    v.linear == :cublas || return TileInfer.linear!(TILE, y, W, x)
    return cublas_linear!(y, W, x)
end

TileInfer.linear!(::Variant, y::CuMatrix, W::TileInfer.Int8Weight, x::CuMatrix) = TileInfer.linear!(TILE, y, W, x)
TileInfer.linear(v::Variant, W::CuMatrix{T}, x::CuMatrix{T}) where {T} =
    TileInfer.linear!(v, CuMatrix{T}(undef, size(W, 2), size(x, 2)), W, x)

function TileInfer.linear_swiglu!(v::Variant, y::CuMatrix{T}, gate_up::CuMatrix{T}, W, x::CuMatrix{T}) where {T}
    v.fused && v.linear == :tile && return TileInfer.linear_swiglu!(TILE, y, gate_up, W, x)
    return TileInfer.silu_mul!(TILE, y, TileInfer.linear!(v, gate_up, W, x))
end

function TileInfer.add_rmsnorm!(v::Variant, x::CuMatrix{T}, h::CuMatrix{T}, delta::CuMatrix{T}, w::CuVector{T}, eps::Real) where {T}
    v.fused && return TileInfer.add_rmsnorm!(TILE, x, h, delta, w, eps)
    x .+= delta
    return TileInfer.rmsnorm!(TILE, h, x, w, eps)
end

function TileInfer.attention_inputs!(v::Variant, q::CuArray{T,3}, k_pages, v_pages, qkv::CuMatrix{T}, q_norm, k_norm,
                                     table, positions, slots, eps) where {T}
    v.fused && return TileInfer.attention_inputs!(TILE, q, k_pages, v_pages, qkv, q_norm, k_norm, table, positions, slots, eps)
    D, nq, n = size(q)
    nkv = size(k_pages, 3)
    qs, k, vs, kr = scratch(v, T, :q, D, nq, n), scratch(v, T, :k, D, nkv, n), scratch(v, T, :v, D, nkv, n),
                    scratch(v, T, :kr, D, nkv, n)
    # The split of the fused projection, one copy each.
    reshape(qs, :, n) .= view(qkv, 1:nq*D, :)
    reshape(k, :, n) .= view(qkv, nq*D+1:(nq+nkv)*D, :)
    reshape(vs, :, n) .= view(qkv, (nq+nkv)*D+1:size(qkv, 1), :)
    if q_norm !== nothing
        qn, kn = scratch(v, T, :qn, D, nq, n), scratch(v, T, :kn, D, nkv, n)
        TileInfer.rmsnorm!(TILE, qn, qs, q_norm, eps)
        TileInfer.rmsnorm!(TILE, kn, k, k_norm, eps)
        qs, k = qn, kn
    end
    TileInfer.rope!(TILE, kr, k, table.cos, table.sin, positions)
    TileInfer.store_kv!(TILE, k_pages, v_pages, kr, vs, slots)
    return TileInfer.rope!(TILE, q, qs, table.cos, table.sin, positions)
end

function TileInfer.attention_workspace(v::Variant, q::CuArray{<:Any,3}, k_pages)
    v.attention == :ref && return nothing
    v.verify == :prefill && size(q, 3) > 1 && return nothing          # the prefill kernel
    return TileInfer.attention_workspace(TILE, q, k_pages)
end

function TileInfer.attention!(v::Variant, out::CuArray{T,3}, workspace, q::CuArray{T,3}, k_pages, v_pages,
                              block_table::CuVector{Int32}, positions::CuVector{Int32}, scale::Real;
                              causal::Bool = true) where {T}
    v.attention == :ref &&
        return copyto!(out, TileInfer.attention(RefBackend(), q, k_pages, v_pages, block_table, positions, scale; causal))
    return TileInfer.attention!(TILE, out, workspace, q, k_pages, v_pages, block_table, positions, scale; causal)
end
