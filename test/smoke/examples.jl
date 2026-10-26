# Run cuTile.jl's own examples on this GPU. Each example's `main()` verifies its
# results against a reference implementation and throws on mismatch.

const EXAMPLES = ["vadd", "matmul", "batchmatmul", "transpose", "layernorm",
                  "softmax", "fmha", "moe"]

@testset "cuTile examples" begin
    dir = joinpath(pkgdir(cuTile), "examples")
    @testset "$name" for name in EXAMPLES
        mod = Module(Symbol("example_", name))
        Base.include(mod, joinpath(dir, "$name.jl"))
        @test (Base.invokelatest(mod.main); true)
    end
end
