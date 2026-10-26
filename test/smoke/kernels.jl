# Minimal hand-written cuTile kernels: tiled load/store, element-type specialization
# (`vadd!` from helpers.jl), scalar loads with runtime indices (the access pattern of
# paged KV lookups).

# Copies whole pages of `src` (D, PAGE, NPAGES) into `dst` following an Int32 page table:
# a scalar load picks the page, then one dense tile load moves it.
function page_copy_kernel(src::ct.TileArray{T,3}, dst::ct.TileArray{T,3},
                          table::ct.TileArray{Int32,1}, D::Int, PAGE::Int) where {T}
    i = ct.bid(1)
    page = table[i]
    tile = ct.load(src; index=(Int32(1), Int32(1), page), shape=(D, PAGE, 1))
    ct.store(dst; index=(Int32(1), Int32(1), i), tile=tile)
    return nothing
end

@testset "kernels" begin
    @testset "vadd $T" for T in (Float32, Float16, BFloat16)
        n = 1 << 20 + 512   # not a multiple of the tile: exercises bounds masking
        a = CUDA.rand(T, n)
        b = CUDA.rand(T, n)
        c = CUDA.zeros(T, n)
        vadd!(c, a, b)
        @test Array(c) == Array(a) .+ Array(b)
    end

    @testset "page table lookup" begin
        D, PAGE, NPAGES = 128, 64, 16
        src = CUDA.rand(BFloat16, D, PAGE, NPAGES)
        table = Int32[7, 3, 16, 1]
        dst = CUDA.zeros(BFloat16, D, PAGE, length(table))
        @cuda backend=cuTile blocks=length(table) page_copy_kernel(
            src, dst, CuArray(table), ct.Constant(D), ct.Constant(PAGE))
        @test Array(dst) == Array(src)[:, :, table]
    end
end
