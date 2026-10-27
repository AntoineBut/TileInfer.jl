# Attention runs on the reference implementation until the paged FlashAttention kernels
# (M5). `has_kernel` tells benchmarks not to report this fallback as a TileBackend result.

attention(::TileBackend, args...) = attention(RefBackend(), args...)
has_kernel(::TileBackend, ::typeof(attention)) = false
