# Rotary position embedding tables, computed once on the host the way transformers does:
# inverse frequencies and angles in Float32, cos/sin rounded to the model's element type.

"""
Per-position `cos`/`sin` tables (head_dim, max_len) on the device; column p+1 is position p.
`cos_sin` holds both, interleaved as (head_dim, 2 max_len): columns 2p+1 and 2p+2 are the
cos and sin of position p, so a fused kernel needs one array argument for them.
"""
struct RotaryTable{T,M<:AbstractMatrix{T}}
    cos::M
    sin::M
    cos_sin::M
end

function inverse_frequencies(cfg::ModelConfig)
    D = cfg.head_dim
    inv_freq = 1f0 ./ (Float32(cfg.rope_theta) .^ (Float32.(0:2:D-1) ./ Float32(D)))
    s = cfg.rope_scaling
    s === nothing && return inv_freq
    return llama3_scaled(inv_freq, s)
end

# transformers' `_compute_llama3_parameters`: long wavelengths are divided by `factor`,
# short ones kept, and the band in between interpolated smoothly.
function llama3_scaled(inv_freq::Vector{Float32}, s::Llama3RopeScaling)
    factor, low, high = Float32(s.factor), Float32(s.low_freq_factor), Float32(s.high_freq_factor)
    old_len = Float32(s.original_max_position_embeddings)
    low_freq_wavelen, high_freq_wavelen = old_len / low, old_len / high
    return map(inv_freq) do f
        wavelen = 2f0 * Float32(π) / f
        wavelen < high_freq_wavelen && return f
        wavelen > low_freq_wavelen && return f / factor
        smooth = (old_len / wavelen - low) / (high - low)
        return (1 - smooth) * f / factor + smooth * f
    end
end

function RotaryTable{T}(cfg::ModelConfig, max_len::Integer) where {T}
    inv_freq = inverse_frequencies(cfg)
    angles = [Float32(p) * f for f in inv_freq, p in 0:max_len-1]   # (head_dim/2, max_len)
    angles = vcat(angles, angles)                                   # both halves share angles
    c = T.(Float32.(cos.(Float64.(angles))))
    s = T.(Float32.(sin.(Float64.(angles))))
    cs = reshape(permutedims(cat(c, s; dims = 3), (1, 3, 2)), size(c, 1), :)
    return RotaryTable(CuArray(c), CuArray(s), CuArray(cs))
end

Base.length(r::RotaryTable) = size(r.cos, 2)
