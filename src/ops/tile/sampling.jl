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

# The (value, position) pair that is larger, ties to the smaller position: `argmax` without
# its NaN handling (logits are never NaN, padding is -Inf), which shortens the reduction.
struct FirstMax end
@inline function (::FirstMax)((va, ia), (vb, ib))
    a = (va > vb) | ((va == vb) & (ia < ib))
    return ifelse(a, va, vb), ifelse(a, ia, ib)
end

# The 1-based positions of the k (≤ K) largest entries of the N-tile x, best first, ties to
# the smaller position; 0 past k.
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

function topk_sample_kernel(Token::ct.TileArray{Int32,1}, Mirror::ct.TileArray{Int32,1},
                            Ids::ct.TileArray{Int32,1}, Logits::ct.TileArray{Float32,1},
                            Params::ct.TileArray{Float32,1}, State::ct.TileArray{UInt32,1},
                            K::Int, N::Int)
    k = State[Int32(1)] % Int32
    # Candidate ids, chunk after chunk; 0 (and ids past the vocabulary) gather -Inf.
    cand = ct.load(Ids; index = Int32(1), shape = (N,), padding_mode = ct.PaddingMode.Zero)
    ids = ct.gather(Ids, top_k_positions(ct.gather(Logits, cand; padding_value = -Inf32), k, K, N))
    vals = ct.gather(Logits, ids; padding_value = -Inf32)
    slot = ct.arange((K,))
    inv_t, top_p = Params[Int32(1)], Params[Int32(2)]
    w = exp.((vals .- maximum(vals; dims = 1)) .* inv_t)
    p = w ./ sum(w; dims = 1)
    # The smallest prefix whose mass reaches top_p: tokens whose better ones hold less.
    keep = (slot .<= k) .& ((slot .== Int32(1)) .| (top_p >= 1f0) .| ((cumsum(p; dims = 1) .- p) .< top_p))
    q = ifelse.(keep, p, 0f0)
    counter = State[Int32(3)]
    target = sampling_uniform(State[Int32(2)], counter) .* sum(q; dims = 1)
    # The first survivor whose cumulative mass reaches the target (the last on round-off).
    first = minimum(ifelse.(keep .& (cumsum(q; dims = 1) .>= target), slot, Int32(K + 1)); dims = 1)
    chosen = min.(first, count(keep; dims = 1))
    token = sum(ifelse.(slot .== chosen, ids, Int32(0))) - Int32(1)
    Token[Int32(1)] = token
    Mirror[Int32(1)] = token
    State[Int32(3)] = counter + UInt32(1)
    return nothing
end

"Scratch for [`sample_token!`](@ref): the top-k candidate ids of every vocabulary chunk."
sample_scratch(vocab::Integer) = CuVector{Int32}(undef, cld(vocab, SAMPLE_CHUNK) * MAX_TOP_K)

function sample_token!(::TileBackend, ids::CuVector{Int32}, mirror::CuVector{Int32}, logits::CuMatrix{Float32},
                       s)
    cand = s.scratch
    chunks = length(cand) ÷ MAX_TOP_K
    @assert chunks <= SAMPLE_PARTS
    @cuda backend = cuTile blocks = chunks topk_chunks_kernel(cand, vec(logits), s.state, ct.Constant(SAMPLE_CHUNK),
                                                              ct.Constant(MAX_TOP_K))
    @cuda backend = cuTile topk_sample_kernel(ids, mirror, cand, vec(logits), s.params, s.state,
                                              ct.Constant(MAX_TOP_K), ct.Constant(MAX_TOP_K * SAMPLE_PARTS))
    return ids
end
