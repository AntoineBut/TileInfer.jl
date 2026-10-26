# Every op has a reference implementation (CUDA.jl broadcasts + cuBLAS, CPU in tests)
# and a cuTile implementation; the backend is selected by dispatch on these types.
#
# Layouts (column-major, PyTorch shapes reversed): hidden states (hidden, tokens), per-head
# tensors (head_dim, heads, tokens), linear weights (in, out) applied as y = Wᵀx,
# contiguous KV cache per layer (head_dim, max_len, kv_heads).

abstract type Backend end

"Reference ops: CUDA.jl broadcasts and cuBLAS. Ground truth for correctness tests."
struct RefBackend <: Backend end

"cuTile.jl kernels."
struct TileBackend <: Backend end

"""
    embed(backend, table, ids) -> (hidden, tokens)

Columns of `table` (hidden, vocab) for the 0-based token `ids` (a device vector).
"""
function embed end

"""
    rmsnorm(backend, x, w, eps)

RMS-normalize `x` along its first dimension and scale by `w`; `x` is (hidden, tokens) or
(head_dim, heads, tokens).
"""
function rmsnorm end

"""
    linear(backend, W, x, bias = nothing) -> (out, tokens)

`Wᵀx (+ bias)` for a weight stored (in, out).
"""
function linear end

"""
    rope(backend, x, cos, sin) -> same shape

Rotary embedding of `x` (head_dim, heads, tokens) with the `cos`/`sin` columns
(head_dim, tokens) of the tokens' positions. Uses the "rotate half" pairing of Llama/Qwen:
element i is paired with element i + head_dim/2.
"""
function rope end

"""
    silu_mul(backend, gate_up) -> (intermediate, tokens)

SwiGLU activation `silu(gate) .* up`, where `gate_up` stacks gate rows over up rows.
"""
function silu_mul end

"""
    attention(backend, q, k_cache, v_cache, start, scale) -> (head_dim, heads, tokens)

Causal grouped-query attention of the queries `q` (head_dim, heads, n), at positions
`start+1:start+n`, over the first `start + n` positions of the caches
(head_dim, max_len, kv_heads).
"""
function attention end

export Backend, RefBackend, TileBackend
