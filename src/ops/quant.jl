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
