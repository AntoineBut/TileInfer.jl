# Small kernels shared by several test files.

function vadd_kernel(a::ct.TileArray{T,1}, b::ct.TileArray{T,1}, c::ct.TileArray{T,1},
                     tile::Int) where {T}
    pid = ct.bid(1)
    ta = ct.load(a; index=pid, shape=(tile,))
    tb = ct.load(b; index=pid, shape=(tile,))
    ct.store(c; index=pid, tile=ta + tb)
    return nothing
end

function vadd!(c, a, b; tile=1024)
    @cuda backend=cuTile blocks=cld(length(c), tile) vadd_kernel(a, b, c, ct.Constant(tile))
    return c
end

using Random: Random

# Seeds Julia's and CUDA.jl's global RNGs (`rand`, `randn`, `randperm`; `CUDA.rand`,
# `CUDA.randn`). Test files that draw random inputs call it first, so each file sees the
# same inputs on every run, whichever files run before it.
function seed_rngs!(seed::Integer = 1)
    Random.seed!(seed)
    CUDA.seed!(seed)
    return nothing
end

# Model used by tests that need a real checkpoint; they are skipped when it is not pulled.
const TEST_MODEL = "Qwen/Qwen3-0.6B"

# Apertus 1.5, for its tokenizer only: the model does not fit the GPU in BFloat16.
const APERTUS_TEST_MODEL = "swiss-ai/Apertus-v1.5-8B"

function test_model_dir(repo = TEST_MODEL)
    try
        return TileInfer.model_dir(repo)
    catch
        @info "$repo not pulled; skipping tests that need it (tileinfer pull $repo)"
        return nothing
    end
end

fixture(path...) = joinpath(@__DIR__, "fixtures", path...)
