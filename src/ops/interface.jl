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
    rope(backend, x, cos, sin, positions) -> same shape

Rotary embedding of `x` (head_dim, heads, n) for tokens at the 0-based `positions` (a
device vector of n `Int32`). `cos`/`sin` are the model's full tables (head_dim, max_len),
whose column p + 1 is position p. Uses the "rotate half" pairing of Llama/Qwen: element i is
paired with element i + head_dim/2.
"""
function rope end

"""
    silu_mul(backend, gate_up) -> (intermediate, tokens)

SwiGLU activation `silu(gate) .* up`, where `gate_up` stacks gate rows over up rows.
"""
function silu_mul end

"""
    XIELU{T}

The learned scalars of a layer's xIELU activation (Apertus, arXiv:2411.13010), in the form
transformers applies them, each rounded to `T` as there:
`x > 0 ? αp x² + β x : (expm1(min(x, ε)) - x) αn + β x`.
"""
struct XIELU{T}
    alpha_p::T                   # softplus(alpha_p) of the checkpoint
    alpha_n::T                   # β + softplus(alpha_n) of the checkpoint
    beta::T
    eps::T
end

"""
    xielu(backend, x, act::XIELU) -> same shape

The xIELU activation of an ungated MLP (Apertus) with a layer's scalars `act`, rounding to
T after each operation as transformers' XIELUActivation does.
"""
function xielu end

"""
    store_kv!(backend, k_pages, v_pages, k, v, slots)

Write the keys and values (head_dim, kv_heads, n) of n tokens into the page pools
(head_dim, page_size, kv_heads, num_pages) at the 1-based cache `slots` (a device vector,
see [`slots`](@ref)): slot s is offset `(s - 1) % page_size` of page `(s - 1) ÷ page_size + 1`.
"""
function store_kv! end

"""
    attention(backend, q, k_pages, v_pages, block_table, positions, scale; causal = true) -> (head_dim, heads, n)

Causal grouped-query attention of the queries `q` (head_dim, heads, n) of one sequence, at
the consecutive 0-based `positions` (a device vector of n `Int32`), over its first
`positions[end] + 1` cached positions. The sequence's keys and values are in the pages
listed by `block_table` (a device vector, which may be padded past the sequence's pages).
With `causal = false`, every query sees all `positions[end] + 1` positions, including those
of the later queries of the step (a DFlash drafter's block).
"""
function attention end

"""
    embed!(backend, y, table, ids)
    rmsnorm!(backend, y, x, w, eps)
    linear!(backend, y, W, x)
    rope!(backend, y, x, cos, sin, positions)
    silu_mul!(backend, y, gate_up)
    xielu!(backend, y, x, act)
    attention!(backend, out, workspace, q, k_pages, v_pages, block_table, positions, scale; causal = true)

In-place forms of the ops, writing into the preallocated `y` (or `out`) and returning it.
TileBackend writes directly and allocates nothing, so a step made of these calls can be
captured in a CUDA graph. The fallbacks here run the allocating op and copy its result.
`workspace` is the backend's scratch for `attention!`, from [`attention_workspace`](@ref).
"""
embed!(be::Backend, y, args...) = copyto!(y, embed(be, args...))
rmsnorm!(be::Backend, y, args...) = copyto!(y, rmsnorm(be, args...))
linear!(be::Backend, y, args...) = copyto!(y, linear(be, args...))
rope!(be::Backend, y, args...) = copyto!(y, rope(be, args...))
silu_mul!(be::Backend, y, args...) = copyto!(y, silu_mul(be, args...))
xielu!(be::Backend, y, args...) = copyto!(y, xielu(be, args...))
attention!(be::Backend, out, workspace, args...; kw...) = copyto!(out, attention(be, args...; kw...))

"""
    attention_inputs!(backend, q, k_pages, v_pages, qkv, q_norm, k_norm, table, positions, slots, eps)

Everything between the fused q/k/v projection `qkv` ((heads + 2 kv_heads) head_dim, n) and
attention, as one op: the optional Qwen3 q/k RMSNorm (`q_norm`/`k_norm` or `nothing`),
rotary embedding with the [`RotaryTable`](@ref) `table` at `positions`, and writing the keys
and values into the page pools at `slots`. Writes the queries into `q` (head_dim, heads, n)
and returns it. The fallback chains the primitive ops; TileBackend fuses them.
"""
function attention_inputs!(be::Backend, q, k_pages, v_pages, qkv, q_norm, k_norm, table, positions, slots, eps)
    D, nq, n = size(q)
    nkv = size(k_pages, 3)
    split(rows, heads) = reshape(qkv[rows, :], D, heads, n)
    qs, k, v = split(1:nq*D, nq), split(nq*D+1:(nq+nkv)*D, nkv), split((nq+nkv)*D+1:size(qkv, 1), nkv)
    if q_norm !== nothing
        qs, k = rmsnorm(be, qs, q_norm, eps), rmsnorm(be, k, k_norm, eps)
    end
    store_kv!(be, k_pages, v_pages, rope(be, k, table.cos, table.sin, positions), v, slots)
    return copyto!(q, rope(be, qs, table.cos, table.sin, positions))
end

"""
    linear_swiglu!(backend, y, gate_up, W, x)

`y = silu_mul(linear(W, x))` for the stacked gate/up weight `W`, in one op where the
backend can fuse it; `gate_up` is scratch for backends that cannot. Returns `y`.
"""
linear_swiglu!(be::Backend, y, gate_up, W, x) = silu_mul!(be, y, linear!(be, gate_up, W, x))

"""
    linear_xielu!(backend, y, W, x, act)

`y = xielu(linear(W, x), act)` for the up weight `W` of an ungated MLP, in one op where the
backend can fuse it. Returns `y`.
"""
linear_xielu!(be::Backend, y, W, x, act) = xielu!(be, y, linear!(be, y, W, x), act)

"""
    add_rmsnorm!(backend, x, h, delta, w, eps)

The residual update `x .+= delta` (rounded to T), then `h = rmsnorm(x, w, eps)`, in one
op. Returns `h`.
"""
function add_rmsnorm!(be::Backend, x, h, delta, w, eps)
    x .+= delta
    return rmsnorm!(be, h, x, w, eps)
end

"""
    sample_token!(backend, ids, mirror, logits, sampler)

Sample a token from the logits (vocab, 1) with the parameters and seed of the
[`TokenSampler`](@ref) `sampler` and its next counter, which it advances. The 0-based token
goes to `ids[1]` and `mirror[1]`, as [`sample_host`](@ref) picks it. On TileBackend this runs
on the device and can be captured in a graph.
"""
function sample_token!(::Backend, ids, mirror, logits, s)
    state = Array(s.state)
    token = Int32[sample_host(vec(Array(logits)), s.sampling, sampling_uniform(s.seed, state[3]))]
    copyto!(ids, token)
    copyto!(mirror, token)
    copyto!(s.state, UInt32[state[1], state[2], state[3] + 1])
    return ids
end

"""
    sample_token!(backend, ids, mirror, logits, sampler, (q_ids, q_probs))

[`sample_token!`](@ref) that also writes the distribution the token is sampled from, for
speculative sampling ([`accept_token!`](@ref)): the 0-based ids of the
[`filtered_dist`](@ref) survivors, best first, into `q_ids` (-1 past them) and their
probabilities into `q_probs` (0 past them), vectors of `MAX_TOP_K` entries.
"""
function sample_token!(be::Backend, ids, mirror, logits, s, (q_ids, q_probs))
    t, q = filtered_dist(vec(Array(logits)), s.sampling)
    pad = length(q_ids) - length(t)
    copyto!(q_ids, [Int32.(t .- 1); fill(Int32(-1), pad)])
    copyto!(q_probs, [Float32.(q); zeros(Float32, pad)])
    return sample_token!(be, ids, mirror, logits, s)
end

"""
    sample_tokens!(backend, ids, mirror, logits, sampler, scratch, dists = nothing)

[`sample_token!`](@ref) on each column j of `logits` (vocab, n) in turn, into `ids[j]` and
`mirror[j]` (and with `dists`, a pair of (MAX_TOP_K, n) matrices, column j's distribution
into their column j), in a few launches on TileBackend. `scratch` is a zeroed
[`sample_cols_scratch`](@ref).
"""
function sample_tokens!(be::Backend, ids, mirror, logits, s, scratch, dists = nothing)
    for j in axes(logits, 2)
        args = (be, view(ids, j:j), view(mirror, j:j), view(logits, :, j:j), s)
        dists === nothing ? sample_token!(args...) : sample_token!(args..., (view(dists[1], :, j), view(dists[2], :, j)))
    end
    return ids
end

"""
    accept_token!(backend, ids, mirror, logits, sampler, draft, (q_ids, q_probs))

The token a verify step emits at a position drafted as `draft[1]` from the distribution
`(q_ids, q_probs)` (see [`accept_host`](@ref)): `draft[1]` with probability
min(1, p/q) under the sampler's parameters, else a token from max(p - q, 0) that differs
from it. Uses and advances the sampler's counter like [`sample_token!`](@ref).
"""
function accept_token!(::Backend, ids, mirror, logits, s, draft, (q_ids, q_probs))
    state = Array(s.state)
    token = Int32[accept_host(vec(Array(logits)), s.sampling, Array(draft)[1], Array(q_ids), Array(q_probs),
                              accept_uniform(s.seed, state[3]), sampling_uniform(s.seed, state[3]))]
    copyto!(ids, token)
    copyto!(mirror, token)
    copyto!(s.state, UInt32[state[1], state[2], state[3] + 1])
    return ids
end

"""
    map_token!(backend, ids, mirror, offsets)

Map the 0-based token `ids[1]` of a reduced vocabulary to `ids[1] + offsets[ids[1] + 1]`, into
`ids[1]` and `mirror[1]`: an EAGLE drafter's token to the target's (its `d2t` table).
"""
function map_token!(::Backend, ids, mirror, offsets)
    id = Int(Array(ids)[1])
    token = Int32[id + Array(offsets)[id+1]]
    copyto!(ids, token)
    copyto!(mirror, token)
    return ids
end

# With the distribution the token was sampled from, whose ids are mapped too.
function map_token!(be::Backend, ids, mirror, offsets, (q_ids, _))
    q, off = Array(q_ids), Array(offsets)
    copyto!(q_ids, [t < 0 ? t : t + off[t+1] for t in q])
    return map_token!(be, ids, mirror, offsets)
end

"""
    attention_workspace(backend, q, k_pages)

Scratch memory `attention!` needs for queries shaped like `q`, or `nothing`.
"""
attention_workspace(::Backend, q, k_pages) = nothing

"""
    supports_graphs(backend) -> Bool

Whether a forward step on `backend` can be captured in a CUDA graph: its in-place ops
neither allocate nor synchronize with the host.
"""
supports_graphs(::Backend) = false

"""
    has_kernel(backend, op) -> Bool

Whether `backend` implements `op` itself, rather than delegating to another backend.
"""
has_kernel(::Backend, op) = true

export Backend, RefBackend, TileBackend
