# Weight-only INT8 quantization: symmetric, one scale per output channel. A linear weight
# (in, out) is stored as Int8 values q and Float32 scales s with W[:, j] ≈ s[j] q[:, j], so
# y = Wᵀx = s .* (qᵀx): the kernels accumulate qᵀx in Float32 from the Int8 values and
# scale the sums once, before rounding to T. Activations stay in T. Halves the bytes a
# decode GEMV reads compared with BFloat16.

"""
    Int8Weight

A linear weight (in, out) as Int8 values `q` with one Float32 scale per output column:
`W[:, j] ≈ scale[j] * q[:, j]`. Made by [`quantize_int8`](@ref).
"""
struct Int8Weight
    q::CuMatrix{Int8}
    scale::CuVector{Float32}
end

Base.size(W::Int8Weight, d...) = size(W.q, d...)
Base.sizeof(W::Int8Weight) = sizeof(W.q) + sizeof(W.scale)

"""
    quantize_int8(W) -> Int8Weight

Round-to-nearest symmetric quantization of each column of `W` (in, out) to [-127, 127],
with the scale `max |W[:, j]| / 127` (1 for an all-zero column).
"""
function quantize_int8(W::CuMatrix)
    absmax = vec(mapreduce(x -> abs(Float32(x)), max, W; dims = 1))
    scale = ifelse.(absmax .> 0, absmax ./ 127, 1.0f0)
    q = Int8.(clamp.(round.(Float32.(W) ./ reshape(scale, 1, :)), -127, 127))
    return Int8Weight(q, scale)
end

"The weight `quantize_int8` stands for, in element type `T`."
dequantize(::Type{T}, W::Int8Weight) where {T} = T.(Float32.(W.q) .* reshape(W.scale, 1, :))

# Reference: Float32 cuBLAS GEMM of the Int8 values, then the scales, rounded to T (the
# activations' type) and converted to the output's. Column blocks bound the Float32 copy
# of the weight (2 GB for a whole 131k-row lm_head).
const REF_INT8_COLUMNS = 8192

function linear!(::RefBackend, y::CuMatrix{S}, W::Int8Weight, x::CuMatrix{T}) where {S,T}
    x32 = Float32.(x)
    for cols in Iterators.partition(1:size(W, 2), REF_INT8_COLUMNS)
        acc = CUBLAS.gemmEx!('T', 'N', true, Float32.(view(W.q, :, cols)), x32, false,
                             CuMatrix{Float32}(undef, length(cols), size(x, 2)))
        view(y, cols, :) .= S.(T.(view(W.scale, cols) .* acc))
    end
    return y
end

function linear(be::Backend, W::Int8Weight, x::CuMatrix{T}, bias = nothing) where {T}
    y = linear!(be, CuMatrix{T}(undef, size(W, 2), size(x, 2)), W, x)
    bias === nothing || (y .+= bias)
    return y
end

# --- INT8 KV cache ---
#
# Each cached key or value vector (one KV head at one position) is stored as Int8 with its
# own Float32 scale, symmetric: x ≈ s q with s = max |x| / 127. A vector is quantized once,
# when its position is written, so the scale never needs updating. Attention applies the
# scales to Float32 results rather than dequantizing the cache: a key's scale multiplies its
# row of scores (s_k q_kᵀ x = s_k (q_kᵀ x)), and a value's scale multiplies its probability
# before the PV product. Halves the bytes decode attention reads compared with BFloat16.
#
# Keys have outlier channels (in Qwen3-0.6B's first layer, one channel's largest value is
# ~100× the median channel's), so one scale per vector leaves few levels for the other
# channels. They come from the per-channel weight of the key norm (k = rope(rmsnorm(k) w)),
# so keys are stored smoothed: divided by a fixed factor c per channel, max(|w_i|, |w_j|)
# for each rotary pair (i, j), which commutes with the rotation. Scores are unchanged when
# the queries are multiplied by c instead (qᵀk = (c q)ᵀ(k / c)). This brings the keys'
# quantization error down to that of one scale per channel (3–7× lower in the outlier
# layers), without calibration.

"""
    Int8Pages

A key or value page pool stored as Int8 values `q` (head_dim, page_size, kv_heads, num_pages)
with one Float32 `scale` (page_size, kv_heads, num_pages) per cached vector:
`pool[:, o, h, p] ≈ scale[o, h, p] * q[:, o, h, p]`. A layer's keys and values are the two
halves of one Int8 array and one scale array (`layer_q`, `layer_scale`, keys first), so a
kernel takes both pools as a single argument and stays under 32 parameters. Vectors are
stored divided by `smooth` (head_dim), the keys' smoothing factors (ones for values):
`pool[:, o, h, p] ≈ smooth .* scale[o, h, p] .* q[:, o, h, p]`. Made in pairs by
[`int8_kv_pools`](@ref).
"""
struct Int8Pages
    q::CuArray{Int8,4}
    scale::CuArray{Float32,3}
    smooth::CuVector{Float32}
    layer_q::CuArray{Int8,5}          # (head_dim, page_size, kv_heads, num_pages, 2)
    layer_scale::CuArray{Float32,4}   # (page_size, kv_heads, num_pages, 2)
end

Base.size(p::Int8Pages) = size(p.q)
Base.size(p::Int8Pages, d::Integer) = size(p.q, d)
Base.sizeof(p::Int8Pages) = sizeof(p.q) + sizeof(p.scale)

"""
    int8_kv_pools(head_dim, page_size, kv_heads, num_pages; key_smoothing = nothing) -> (keys, values)

The key and value [`Int8Pages`](@ref) of one layer, zeroed; the keys smoothed by
`key_smoothing` (see [`key_smoothing`](@ref)), or not.
"""
function int8_kv_pools(D::Integer, P::Integer, H::Integer, pages::Integer;
                       key_smoothing::Union{Nothing,CuVector{Float32}} = nothing)
    q, scale = CUDA.zeros(Int8, D, P, H, pages, 2), CUDA.zeros(Float32, P, H, pages, 2)
    half(i, smooth) = Int8Pages(view(q, :, :, :, :, i), view(scale, :, :, :, i), smooth, q, scale)
    return half(1, something(key_smoothing, CUDA.ones(Float32, D))), half(2, CUDA.ones(Float32, D))
end

"""
    key_smoothing(k_norm) -> CuVector{Float32}

The factors that keys normalized by the weight `k_norm` are divided by in an INT8 cache:
max(|w_i|, |w_j|) for each rotary pair (i, j = i + head_dim/2), so the factor is the same
on both channels a rotation mixes (1 where both are zero).
"""
function key_smoothing(k_norm::CuVector)
    w = abs.(Float32.(k_norm))
    half = length(w) ÷ 2
    c = max.(view(w, 1:half), view(w, half+1:2half))
    c = ifelse.(c .> 0, c, 1.0f0)
    return vcat(c, c)
end

# The layer arrays behind a key/value pair, for the kernels: Int8 values as
# (head_dim, 2 page_size kv_heads num_pages), keys' columns first, the scales as one vector,
# and the keys' smoothing.
function layer_arrays(k::Int8Pages, v::Int8Pages)
    k.layer_q === v.layer_q && pointer(k.q) == pointer(k.layer_q) ||
        throw(ArgumentError("keys and values must be the halves of one layer's pools (int8_kv_pools)"))
    return reshape(k.layer_q, size(k.q, 1), :), vec(k.layer_scale), k.smooth
end

# Quantization of the columns of x (Float32): q = floor(x 127 / max|x| + 1/2) and
# s = max|x| / 127, as the kernels compute them (0 for an all-zero column).
function quantize_columns(x::CuMatrix{Float32})
    absmax = maximum(abs, x; dims = 1)
    inv = ifelse.(absmax .> 0, 127.0f0 ./ absmax, 0.0f0)
    return Int8.(floor.(x .* inv .+ 0.5f0)), vec(absmax ./ 127.0f0)
end

function store_kv!(::RefBackend, k_pages::Int8Pages, v_pages::Int8Pages, k::CuArray{T,3}, v::CuArray{T,3},
                   slots::CuVector{Int32}) where {T}
    D, P, H, _ = size(k_pages)
    n = length(slots)
    page, offset = (slots .- 1) .÷ P, (slots .- 1) .% P
    cols = vec(reshape(offset, 1, n) .+ P .* CuArray(0:H-1) .+ (P * H) .* reshape(page, 1, n) .+ 1)  # (h, t)
    for (pool, x) in ((k_pages, k), (v_pages, v))
        q, s = quantize_columns(Float32.(reshape(x, D, H * n)) ./ pool.smooth)
        reshape(pool.q, D, :)[:, cols] = q
        vec(pool.scale)[cols] = s
    end
    return nothing
end

# Dequantized and unsmoothed in Float32 (not rounded to T): the kernels scale Float32 scores
# and probabilities.
function gather_pages(pages::Int8Pages, block_table::CuVector{Int32}, h::Integer, len::Integer)
    D, P = size(pages, 1), size(pages, 2)
    used = view(block_table, 1:cld(len, P))
    x = Float32.(pages.q[:, :, h, used]) .* reshape(pages.scale[:, h, used], 1, P, :) .* pages.smooth
    return view(reshape(x, D, :), :, 1:len)
end
