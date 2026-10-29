# What a block-wide reduction costs: `ITERS` dependent reductions over a tile held in
# registers (each removes the element it found, as a top-k selection does), by kind of
# reduction and by tile size. A variant with 0 iterations gives the load/store baseline.

struct FirstMaxPair end
@inline function (::FirstMaxPair)((va, ia), (vb, ib))
    a = (va > vb) | ((va == vb) & (ia < ib))
    return ifelse(a, va, vb), ifelse(a, ia, ib)
end

for (name, step) in (
        :red_max => :(m = maximum(x; dims = 1); x = ifelse.(x .== m, -Inf32, x); acc = acc .+ m),
        :red_sum => :(m = sum(x; dims = 1); x = x .- m .* 1f-3; acc = acc .+ m),
        :red_argmax => :(i = argmax(x; dims = 1); x = ifelse.(pos .== i, -Inf32, x); acc = acc .+ convert(ct.Tile{Float32}, i)),
        :red_argmax_nan => :(i = argmax(x; dims = 1, propagate_nan = true); x = ifelse.(pos .== i, -Inf32, x); acc = acc .+ convert(ct.Tile{Float32}, i)),
        :red_firstmax => :((_, i) = mapreduce(identity, FirstMaxPair(), x, pos; dims = 1, init = (-Inf32, Int32(0)));
                           x = ifelse.(pos .== i, -Inf32, x); acc = acc .+ convert(ct.Tile{Float32}, i)),
        :red_max_then_min => :(m = maximum(x; dims = 1); i = minimum(ifelse.(x .== m, pos, Int32(N + 1)); dims = 1);
                               x = ifelse.(pos .== i, -Inf32, x); acc = acc .+ convert(ct.Tile{Float32}, i)))
    @eval function $name(Out::ct.TileArray{Float32,1}, In::ct.TileArray{Float32,1}, Iters::ct.TileArray{Int32,1}, N::Int)
        b = ct.bid(1)
        x = ct.load(In; index = b, shape = (N,), padding_mode = ct.PaddingMode.NegInf)
        pos = ct.arange((N,))
        acc = zeros(Float32, 1)
        for _ in Int32(1):Iters[Int32(1)]
            $step
        end
        ct.store(Out; index = b, tile = acc)
        return nothing
    end
end

const RED_BLOCKS = 38                     # as many as TileInfer's sampling chunk kernel

function reduction_launch(kernel, N, iters)
    input = CUDA.randn(Float32, RED_BLOCKS * N)
    out = CUDA.zeros(Float32, RED_BLOCKS)
    it = CuVector{Int32}([iters])
    return () -> @cuda backend = cuTile blocks = RED_BLOCKS kernel(out, input, it, ct.Constant(N))
end

probe!("reduce_kind", "20 dependent reductions over a 4096-element Float32 tile, by kind",
       "load + store only (0 iterations)" => reduction_launch(red_max, 4096, 0),
       "maximum" => reduction_launch(red_max, 4096, 20),
       "sum" => reduction_launch(red_sum, 4096, 20),
       "argmax" => reduction_launch(red_argmax, 4096, 20),
       "argmax propagate_nan" => reduction_launch(red_argmax_nan, 4096, 20),
       "mapreduce, no NaN handling" => reduction_launch(red_firstmax, 4096, 20),
       "maximum, then minimum position" => reduction_launch(red_max_then_min, 4096, 20))

probe!("reduce_size", "20 dependent `maximum` reductions, by tile size (38 blocks)",
       ("N = $N" => reduction_launch(red_max, N, 20) for N in (256, 1024, 4096, 16384))...)

# Can a hint change the block size tileiras picks (always 128 threads so far)? The same
# kernel with `occupancy` and `opt_level` hints. (The Tile IR and SASS columns show the
# kernel compiled without hints; time, block size and registers are the hinted launch's.)
let input = CUDA.randn(Float32, RED_BLOCKS * 4096), out = CUDA.zeros(Float32, RED_BLOCKS), it = CuVector{Int32}([20])
    args = (out, input, it, ct.Constant(4096))
    probe!("hint_block_size", "20 argmax reductions over 4096 elements with compiler hints",
           "no hint" => () -> @cuda(backend = cuTile, blocks = RED_BLOCKS, red_argmax(args...)),
           "occupancy = 1" => () -> @cuda(backend = cuTile, blocks = RED_BLOCKS, occupancy = 1, red_argmax(args...)),
           "occupancy = 4" => () -> @cuda(backend = cuTile, blocks = RED_BLOCKS, occupancy = 4, red_argmax(args...)),
           "occupancy = 16" => () -> @cuda(backend = cuTile, blocks = RED_BLOCKS, occupancy = 16, red_argmax(args...)),
           "opt_level = 1" => () -> @cuda(backend = cuTile, blocks = RED_BLOCKS, opt_level = 1, red_argmax(args...)))
end
