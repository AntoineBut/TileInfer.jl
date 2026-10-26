# Every op has a reference implementation (CUDA.jl broadcasts + cuBLAS, CPU in tests)
# and a cuTile implementation; the backend is selected by dispatch on these types.
#
# Layouts (column-major, PyTorch shapes reversed): hidden states (hidden, tokens), per-head
# tensors (head_dim, heads, tokens), linear weights (in, out) applied as y = Wᵀx,
# paged KV pools per layer (head_dim, page_size, kv_heads, num_pages).

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
    store_kv!(backend, k_pages, v_pages, k, v, slots)

Write the keys and values (head_dim, kv_heads, n) of n tokens into the page pools
(head_dim, page_size, kv_heads, num_pages) at the 1-based cache `slots` (a device vector,
see [`slots`](@ref)): slot s is offset `(s - 1) % page_size` of page `(s - 1) ÷ page_size + 1`.
"""
function store_kv! end

"""
    attention(backend, q, k_pages, v_pages, block_table, start, scale) -> (head_dim, heads, n)

Causal grouped-query attention of the queries `q` (head_dim, heads, n), at positions
`start+1:start+n` of one sequence, over its first `start + n` cached positions. The
sequence's keys and values are in the pages listed by `block_table` (a device vector).
"""
function attention end

export Backend, RefBackend, TileBackend
