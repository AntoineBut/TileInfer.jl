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

function test_model_dir()
    try
        return TileInfer.model_dir(TEST_MODEL)
    catch
        @info "$TEST_MODEL not pulled; skipping tests that need it (tileinfer pull $TEST_MODEL)"
        return nothing
    end
end

fixture(path...) = joinpath(@__DIR__, "fixtures", path...)
