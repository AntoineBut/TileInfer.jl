# Host-side cost of a kernel launch: native CUDA.jl vs cuTile vs CUDA-graph replay.
# A decode step is ~300 launches, so this bounds achievable tok/s without graphs.

using CUDA
using cuTile: cuTile
import cuTile as ct
using Printf

function native_kernel(a, b, c)
    i = threadIdx().x
    @inbounds c[i] = a[i] + b[i]
    return nothing
end

function tile_kernel(a::ct.TileArray{T,1}, b::ct.TileArray{T,1}, c::ct.TileArray{T,1},
                     tile::Int) where {T}
    pid = ct.bid(1)
    ct.store(c; index=pid, tile=ct.load(a; index=pid, shape=(tile,)) +
                                ct.load(b; index=pid, shape=(tile,)))
    return nothing
end

const N = 1024
a, b, c = CUDA.rand(Float32, N), CUDA.rand(Float32, N), CUDA.zeros(Float32, N)

launch_native() = @cuda threads=N native_kernel(a, b, c)
launch_tile() = @cuda backend=cuTile blocks=1 tile_kernel(a, b, c, ct.Constant(N))

# Returns (enqueue µs per launch, wall µs per launch including GPU completion).
function measure(f, iters)
    f(); synchronize()
    t0 = time_ns()
    for _ in 1:iters
        f()
    end
    t1 = time_ns()
    synchronize()
    t2 = time_ns()
    return (t1 - t0) / 1e3 / iters, (t2 - t0) / 1e3 / iters
end

function main(; iters=10_000, graph_len=300)
    for _ in 1:3  # warm up the compiler and the measurement loop itself
        measure(launch_native, 100); measure(launch_tile, 100)
    end

    graph = CUDA.capture() do
        for _ in 1:graph_len
            launch_tile()
        end
    end
    exec = CUDA.instantiate(graph)
    replays = cld(iters, graph_len)
    measure(() -> CUDA.launch(exec), 10)
    g_enq, g_wall = measure(() -> CUDA.launch(exec), replays)

    @printf("%-28s %12s %12s\n", "launch kind", "enqueue µs", "wall µs")
    for (name, f) in (("native @cuda", launch_native), ("cuTile @cuda", launch_tile))
        enq, wall = measure(f, iters)
        @printf("%-28s %12.2f %12.2f\n", name, enq, wall)
    end
    @printf("%-28s %12.2f %12.2f\n", "graph replay (per kernel)", g_enq / graph_len,
            g_wall / graph_len)
    @printf("\nper %d-kernel step: cuTile eager %.0f µs, graph %.0f µs\n", graph_len,
            measure(launch_tile, iters)[2] * graph_len, g_wall)
end

main()
