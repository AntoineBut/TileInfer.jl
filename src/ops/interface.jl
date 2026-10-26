# Every op has a reference implementation (CUDA.jl broadcasts + cuBLAS, CPU in tests)
# and a cuTile implementation; the backend is selected by dispatch on these types.

abstract type Backend end

"Reference ops: CUDA.jl broadcasts and cuBLAS. Ground truth for correctness tests."
struct RefBackend <: Backend end

"cuTile.jl kernels."
struct TileBackend <: Backend end

export Backend, RefBackend, TileBackend
