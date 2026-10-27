# Greedy sampling on the device, so that it can be part of a captured decode step: the
# argmax of the logits in two kernels (per-chunk maxima, then the maximum of those), ties to
# the smallest index as Julia's argmax. The 0-based token goes to `ids` (the next step's
# input) and to `mirror`, which may live in host-mapped memory for the host to read.

const ARGMAX_CHUNK = 4096
const ARGMAX_PARTS = 64            # ≥ chunks of the vocabulary (Qwen3: 38)

function argmax_chunks_kernel(Val::ct.TileArray{Float32,1}, Idx::ct.TileArray{Int32,1},
                              Logits::ct.TileArray{Float32,2}, CHUNK::Int)
    b = ct.bid(1)
    x = reshape(ct.load(Logits; index = (b, Int32(1)), shape = (CHUNK, 1),
                        padding_mode = ct.PaddingMode.NegInf), (CHUNK,))
    ct.store(Val; index = b, tile = maximum(x; dims = 1))
    ct.store(Idx; index = b, tile = (b - Int32(1)) * Int32(CHUNK) .+ argmax(x; dims = 1))
    return nothing
end

function argmax_final_kernel(Ids::ct.TileArray{Int32,1}, Mirror::ct.TileArray{Int32,1},
                             Val::ct.TileArray{Float32,1}, Idx::ct.TileArray{Int32,1}, PARTS::Int)
    v = ct.load(Val; index = Int32(1), shape = (PARTS,), padding_mode = ct.PaddingMode.NegInf)
    token = Idx[argmax(v; dims = 1)[1]] - Int32(1)
    Ids[Int32(1)] = token
    Mirror[Int32(1)] = token
    return nothing
end

"Scratch for [`argmax_token!`](@ref): per-chunk maxima and their indices."
argmax_scratch(vocab::Integer) =
    (CuVector{Float32}(undef, cld(vocab, ARGMAX_CHUNK)), CuVector{Int32}(undef, cld(vocab, ARGMAX_CHUNK)))

function argmax_token!(::TileBackend, ids::CuVector{Int32}, mirror::CuVector{Int32}, logits::CuMatrix{Float32},
                       scratch)
    val, idx = scratch
    @assert length(val) <= ARGMAX_PARTS
    @cuda backend = cuTile blocks = length(val) argmax_chunks_kernel(val, idx, logits, ct.Constant(ARGMAX_CHUNK))
    @cuda backend = cuTile argmax_final_kernel(ids, mirror, val, idx, ct.Constant(ARGMAX_PARTS))
    return ids
end
