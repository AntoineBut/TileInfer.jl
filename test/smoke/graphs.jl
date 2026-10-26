# Can cuTile launches be captured into a CUDA graph, alongside CUDA.jl broadcasts and
# cuBLAS calls, and does replay read the live contents of the captured buffers?

using LinearAlgebra

@testset "CUDA graphs" begin
    n = 1 << 16
    a = CUDA.rand(Float32, n)
    b = CUDA.rand(Float32, n)
    c = CUDA.zeros(Float32, n)
    W = CUDA.rand(Float32, 256, 256)
    x = CUDA.rand(Float32, 256)
    y = CUDA.zeros(Float32, 256)

    step!() = begin
        vadd!(c, a, b)          # cuTile
        vadd!(c, c, b)          # cuTile, depends on the previous launch
        c .*= 2f0               # CUDA.jl broadcast
        mul!(y, W, x)           # cuBLAS
    end

    # Compilation and cuBLAS handle creation cannot happen during capture.
    step!()
    synchronize()

    graph = CUDA.capture(step!)
    exec = CUDA.instantiate(graph)

    for _ in 1:3
        copyto!(a, rand(Float32, n))
        copyto!(x, rand(Float32, 256))
        CUDA.launch(exec)
        synchronize()
        @test Array(c) ≈ 2 .* (Array(a) .+ 2 .* Array(b))
        @test Array(y) ≈ Array(W) * Array(x)
    end
end
