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

# Model used by tests that need a real checkpoint; they are skipped when it is not pulled.
const TEST_MODEL = "Qwen/Qwen3-0.6B"
# An EAGLE-3 drafter for TEST_MODEL; its tests are skipped when it is not pulled.
const EAGLE_TEST_HEAD = "GavinLucky/SGLang-EAGLE3-Qwen3-0.6B-SpecForge"

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
