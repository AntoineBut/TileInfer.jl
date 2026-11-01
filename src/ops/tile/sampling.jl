# Sampling on the device, so that it can be part of a captured decode step: temperature,
# top-k with k ≤ MAX_TOP_K, then top-p, as `sample_host` does with u = sampling_uniform(seed,
# counter); greedy decoding is top-1. Two kernels: each chunk of the vocabulary selects the
# ids of its top_k logits, then one block selects the top_k of those candidates and samples
# among them. Selection is repeated argmax, so the survivors come out best first with ties
# to the smaller id, the order `sample_host` uses; values are gathered from the logits by
# id. Each argmax is a block-wide reduction of 0.3–0.5 µs that barely depends on the tile
# size, so a token costs about 2 × top_k of them (17 µs at top_k = 20). The parameters, seed and
# counter live in device buffers (a `TokenSampler`), so the kernels can be captured. The
# 0-based token goes to `ids` (the next step's input) and to `mirror`, which may live in
# host-mapped memory for the host to read.

const MAX_TOP_K = 64
const SAMPLE_CHUNK = 4096
const SAMPLE_PARTS = 64            # ≥ chunks of the vocabulary: MAX_TOP_K × SAMPLE_PARTS candidates

"""
The (value, position) pair that is larger, ties to the smaller position: `argmax` without
its NaN handling (logits are never NaN, padding is -Inf), which shortens the reduction.
"""
struct FirstMax end
@inline function (::FirstMax)((va, ia), (vb, ib))
    a = (va > vb) | ((va == vb) & (ia < ib))
    return ifelse(a, va, vb), ifelse(a, ia, ib)
end

"""
The 1-based positions of the k (≤ K) largest entries of the N-tile x, best first, ties to
the smaller position; 0 past k.
"""
@inline function top_k_positions(x, k, K::Int, N::Int)
    pos, slot = ct.arange((N,)), ct.arange((K,))
    at = zeros(Int32, K)
    for r in Int32(1):k
        _, i = mapreduce(identity, FirstMax(), x, pos; dims = 1, init = (-Inf32, Int32(0)))
        at = ifelse.(slot .== r, i, at)
        x = ifelse.(pos .== i, -Inf32, x)
    end
    return at
end

function topk_chunks_kernel(Ids::ct.TileArray{Int32,1}, Logits::ct.TileArray{Float32,1},
                            State::ct.TileArray{UInt32,1}, CHUNK::Int, K::Int)
    b = ct.bid(1)
    x = ct.load(Logits; index = b, shape = (CHUNK,), padding_mode = ct.PaddingMode.NegInf)
    at = top_k_positions(x, State[Int32(1)] % Int32, K, CHUNK)
    ct.store(Ids; index = b, tile = ifelse.(at .> Int32(0), at .+ (b - Int32(1)) * Int32(CHUNK), Int32(0)))
    return nothing
end

"""
The survivors among the chunks' candidates `Ids`: their 1-based ids (best first, 0 past
`top_k`), their probabilities under temperature and top-k, zeroed outside the top-p prefix,
and the mask of survivors. With several columns of logits (`sample_tokens!`), Ids and
Logits are the flat arrays and `col` the column: its candidates are entries
(col - 1) N + 1 : col N of Ids, and its logits start at (col - 1) V + 1.
"""
@inline function survivors(Ids, Logits, Params, State, K::Int, N::Int, col = Int32(1), V = Int32(0))
    k = State[Int32(1)] % Int32
    shift(i, offset) = ifelse.(i .> Int32(0), i .+ offset, Int32(0))       # 0 stays padding
    lo = (col - Int32(1)) * V
    # Candidate ids, chunk after chunk; 0 (and ids past the vocabulary) gather -Inf.
    cand = ct.load(Ids; index = col, shape = (N,), padding_mode = ct.PaddingMode.Zero)
    at = top_k_positions(ct.gather(Logits, shift(cand, lo); padding_value = -Inf32), k, K, N)
    ids = ct.gather(Ids, shift(at, (col - Int32(1)) * Int32(N)))
    vals = ct.gather(Logits, shift(ids, lo); padding_value = -Inf32)
    slot = ct.arange((K,))
    inv_t, top_p = Params[Int32(1)], Params[Int32(2)]
    w = exp.((vals .- maximum(vals; dims = 1)) .* inv_t)
    p = w ./ sum(w; dims = 1)
    # The smallest prefix whose mass reaches top_p: tokens whose better ones hold less.
    keep = (slot .<= k) .& ((slot .== Int32(1)) .| (top_p >= 1f0) .| ((cumsum(p; dims = 1) .- p) .< top_p))
    return ids, ifelse.(keep, p, 0f0), keep
end

"""
The id at the first slot whose cumulative weight reaches u × the total, among the slots
with positive weight (the last of them on round-off), as `pick_host`.
"""
@inline function pick(ids, w, u, K::Int)
    slot = ct.arange((K,))
    live = w .> 0f0
    target = u .* sum(w; dims = 1)
    first = minimum(ifelse.(live .& (cumsum(w; dims = 1) .>= target), slot, Int32(K + 1)); dims = 1)
    chosen = min.(first, maximum(ifelse.(live, slot, Int32(0)); dims = 1))
    return sum(ifelse.(slot .== chosen, ids, Int32(0)))
end

function topk_sample_kernel(Token::ct.TileArray{Int32,1}, Mirror::ct.TileArray{Int32,1},
                            Ids::ct.TileArray{Int32,1}, Logits::ct.TileArray{Float32,1},
                            Params::ct.TileArray{Float32,1}, State::ct.TileArray{UInt32,1},
                            K::Int, N::Int)
    ids, q, _ = survivors(Ids, Logits, Params, State, K, N)
    counter = State[Int32(3)]
    token = pick(ids, q, sampling_uniform(State[Int32(2)], counter), K) - Int32(1)
    Token[Int32(1)] = token
    Mirror[Int32(1)] = token
    State[Int32(3)] = counter + UInt32(1)
    return nothing
end

"""
`topk_sample_kernel` that also writes the distribution it samples from: the survivors'
0-based ids (-1 past them) and their probabilities.
"""
function topk_sample_dist_kernel(Token::ct.TileArray{Int32,1}, Mirror::ct.TileArray{Int32,1},
                                 QIds::ct.TileArray{Int32,1}, QProbs::ct.TileArray{Float32,1},
                                 Ids::ct.TileArray{Int32,1}, Logits::ct.TileArray{Float32,1},
                                 Params::ct.TileArray{Float32,1}, State::ct.TileArray{UInt32,1},
                                 K::Int, N::Int)
    ids, q, keep = survivors(Ids, Logits, Params, State, K, N)
    ct.store(QIds; index = Int32(1), tile = ifelse.(keep, ids .- Int32(1), Int32(-1)))
    ct.store(QProbs; index = Int32(1), tile = q ./ sum(q; dims = 1))
    counter = State[Int32(3)]
    token = pick(ids, q, sampling_uniform(State[Int32(2)], counter), K) - Int32(1)
    Token[Int32(1)] = token
    Mirror[Int32(1)] = token
    State[Int32(3)] = counter + UInt32(1)
    return nothing
end

"""
The token emitted at a position drafted as Draft[1] from the distribution (QIds, QProbs),
as `accept_host`: the draft with probability min(1, p/q), else a sample of max(p - q, 0).
"""
function accept_kernel(Token::ct.TileArray{Int32,1}, Mirror::ct.TileArray{Int32,1},
                       Draft::ct.TileArray{Int32,1}, QIds::ct.TileArray{Int32,1}, QProbs::ct.TileArray{Float32,1},
                       Ids::ct.TileArray{Int32,1}, Logits::ct.TileArray{Float32,1},
                       Params::ct.TileArray{Float32,1}, State::ct.TileArray{UInt32,1},
                       K::Int, N::Int)
    ids, w, keep = survivors(Ids, Logits, Params, State, K, N)
    p = w ./ sum(w; dims = 1)
    tokens = ifelse.(keep, ids .- Int32(1), Int32(-1))
    q_ids = ct.load(QIds; index = Int32(1), shape = (K,))
    q_probs = ct.load(QProbs; index = Int32(1), shape = (K,))
    d = Draft[Int32(1)]
    pd = sum(ifelse.(tokens .== d, p, 0f0))
    qd = sum(ifelse.(q_ids .== d, q_probs, 0f0))
    # q at each of p's survivors: a (K, K) comparison of the two id lists.
    q_at = sum(ifelse.(reshape(tokens, (K, 1)) .== reshape(q_ids, (1, K)), reshape(q_probs, (1, K)), 0f0); dims = 2)
    r = max.(p .- reshape(q_at, (K,)), 0f0)
    seed, counter = State[Int32(2)], State[Int32(3)]
    accept = (accept_uniform(seed, counter) * qd < pd) | (sum(r) <= 0f0)
    token = ifelse(accept, d, pick(ids, r, sampling_uniform(seed, counter), K) - Int32(1))
    Token[Int32(1)] = token
    Mirror[Int32(1)] = token
    State[Int32(3)] = counter + UInt32(1)
    return nothing
end

# --- Several columns at once (a DFlash drafter's block): column j samples with the counter
# c + j - 1 as `sample_token!` on each column in turn would, in three launches instead of
# two per column. A last one-block kernel advances the counter, after every column has read it.

function topk_chunks_cols_kernel(Ids::ct.TileArray{Int32,2}, Logits::ct.TileArray{Float32,2},
                                 State::ct.TileArray{UInt32,1}, CHUNK::Int, K::Int)
    b, j = ct.bid(1), ct.bid(2)
    x = reshape(ct.load(Logits; index = (b, j), shape = (CHUNK, 1), padding_mode = ct.PaddingMode.NegInf), (CHUNK,))
    at = top_k_positions(x, State[Int32(1)] % Int32, K, CHUNK)
    ct.store(Ids; index = (b, j), tile = reshape(ifelse.(at .> Int32(0), at .+ (b - Int32(1)) * Int32(CHUNK), Int32(0)), (K, 1)))
    return nothing
end

function topk_sample_cols_kernel(Token::ct.TileArray{Int32,1}, Mirror::ct.TileArray{Int32,1},
                                 Ids::ct.TileArray{Int32,1}, Logits::ct.TileArray{Float32,1},
                                 Params::ct.TileArray{Float32,1}, State::ct.TileArray{UInt32,1},
                                 K::Int, N::Int)
    j = ct.bid(1)
    ids, q, _ = survivors(Ids, Logits, Params, State, K, N, j, size(Logits, 1) ÷ size(Token, 1))
    token = pick(ids, q, sampling_uniform(State[Int32(2)], State[Int32(3)] + (j - Int32(1)) % UInt32), K) - Int32(1)
    Token[j] = token
    Mirror[j] = token
    return nothing
end

function topk_sample_dist_cols_kernel(Token::ct.TileArray{Int32,1}, Mirror::ct.TileArray{Int32,1},
                                      QIds::ct.TileArray{Int32,1}, QProbs::ct.TileArray{Float32,1},
                                      Ids::ct.TileArray{Int32,1}, Logits::ct.TileArray{Float32,1},
                                      Params::ct.TileArray{Float32,1}, State::ct.TileArray{UInt32,1},
                                      K::Int, N::Int)
    j = ct.bid(1)
    ids, q, keep = survivors(Ids, Logits, Params, State, K, N, j, size(Logits, 1) ÷ size(Token, 1))
    ct.store(QIds; index = j, tile = ifelse.(keep, ids .- Int32(1), Int32(-1)))
    ct.store(QProbs; index = j, tile = q ./ sum(q; dims = 1))
    token = pick(ids, q, sampling_uniform(State[Int32(2)], State[Int32(3)] + (j - Int32(1)) % UInt32), K) - Int32(1)
    Token[j] = token
    Mirror[j] = token
    return nothing
end

function advance_counter_kernel(State::ct.TileArray{UInt32,1}, COUNT::Int)
    State[Int32(3)] = State[Int32(3)] + UInt32(COUNT)
    return nothing
end

"Scratch for [`sample_tokens!`](@ref) over `cols` columns: each column's candidates, zero-padded."
sample_cols_scratch(cols::Integer) = CUDA.zeros(Int32, MAX_TOP_K * SAMPLE_PARTS, cols)

function sample_tokens!(::TileBackend, ids::CuVector{Int32}, mirror::CuVector{Int32}, logits::CuMatrix{Float32},
                        s, scratch::CuMatrix{Int32}, dists = nothing)
    V, cols = size(logits)
    chunks = cld(V, SAMPLE_CHUNK)
    @assert chunks <= SAMPLE_PARTS && size(scratch) == (MAX_TOP_K * SAMPLE_PARTS, cols)
    K, N = ct.Constant(MAX_TOP_K), ct.Constant(MAX_TOP_K * SAMPLE_PARTS)
    @cuda backend = cuTile blocks = (chunks, cols) topk_chunks_cols_kernel(scratch, logits, s.state,
                                                                          ct.Constant(SAMPLE_CHUNK), K)
    if dists === nothing
        @cuda backend = cuTile blocks = cols topk_sample_cols_kernel(ids, mirror, vec(scratch), vec(logits),
                                                                     s.params, s.state, K, N)
    else
        q_ids, q_probs = dists
        @cuda backend = cuTile blocks = cols topk_sample_dist_cols_kernel(ids, mirror, vec(q_ids), vec(q_probs),
                                                                          vec(scratch), vec(logits), s.params,
                                                                          s.state, K, N)
    end
    @cuda backend = cuTile advance_counter_kernel(s.state, ct.Constant(cols))
    return ids
end

"Scratch for [`sample_token!`](@ref): the top-k candidate ids of every vocabulary chunk."
sample_scratch(vocab::Integer) = CuVector{Int32}(undef, cld(vocab, SAMPLE_CHUNK) * MAX_TOP_K)

"""
The candidates of every chunk of the vocabulary, into the sampler's scratch.
"""
function topk_chunks!(logits, s)
    cand = s.scratch
    chunks = length(cand) ÷ MAX_TOP_K
    @assert chunks <= SAMPLE_PARTS
    @cuda backend = cuTile blocks = chunks topk_chunks_kernel(cand, vec(logits), s.state, ct.Constant(SAMPLE_CHUNK),
                                                              ct.Constant(MAX_TOP_K))
    return cand
end

function sample_token!(::TileBackend, ids::CuVector{Int32}, mirror::CuVector{Int32}, logits::CuMatrix{Float32},
                       s)
    cand = topk_chunks!(logits, s)
    @cuda backend = cuTile topk_sample_kernel(ids, mirror, cand, vec(logits), s.params, s.state,
                                              ct.Constant(MAX_TOP_K), ct.Constant(MAX_TOP_K * SAMPLE_PARTS))
    return ids
end

function sample_token!(::TileBackend, ids::CuVector{Int32}, mirror::CuVector{Int32}, logits::CuMatrix{Float32},
                       s, (q_ids, q_probs))
    cand = topk_chunks!(logits, s)
    @cuda backend = cuTile topk_sample_dist_kernel(ids, mirror, q_ids, q_probs, cand, vec(logits), s.params, s.state,
                                                   ct.Constant(MAX_TOP_K), ct.Constant(MAX_TOP_K * SAMPLE_PARTS))
    return ids
end

function accept_token!(::TileBackend, ids::CuVector{Int32}, mirror::CuVector{Int32}, logits::CuMatrix{Float32},
                       s, draft::CuVector{Int32}, (q_ids, q_probs))
    cand = topk_chunks!(logits, s)
    @cuda backend = cuTile accept_kernel(ids, mirror, draft, q_ids, q_probs, cand, vec(logits), s.params, s.state,
                                         ct.Constant(MAX_TOP_K), ct.Constant(MAX_TOP_K * SAMPLE_PARTS))
    return ids
end
