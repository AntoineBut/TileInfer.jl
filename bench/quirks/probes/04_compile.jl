# Valid Julia that cuTile.jl does or does not compile.

# --- Shifts of an unsigned integer by an Int literal ---

function shift_int(Y::ct.TileArray{UInt32,1}, X::ct.TileArray{UInt32,1})
    x = X[Int32(1)]
    Y[Int32(1)] = xor(x, x >> 16)
    return nothing
end

function shift_same_type(Y::ct.TileArray{UInt32,1}, X::ct.TileArray{UInt32,1})
    x = X[Int32(1)]
    Y[Int32(1)] = xor(x, x >> UInt32(16))
    return nothing
end

function shift_left_int(Y::ct.TileArray{UInt32,1}, X::ct.TileArray{UInt32,1})
    x = X[Int32(1)]
    Y[Int32(1)] = x << 3
    return nothing
end

function shift_tile_int(Y::ct.TileArray{UInt32,1}, X::ct.TileArray{UInt32,1})
    x = ct.load(X; index = Int32(1), shape = (16,))
    ct.store(Y; index = Int32(1), tile = x .>> 16)
    return nothing
end

let X = CuVector{UInt32}(1:16), Y = CUDA.zeros(UInt32, 16)
    compile_probe!("shift_count_type", "x >> 16 on a UInt32 (Julia allows any integer shift count)",
                   "UInt32 >> 16 (Int literal)" => () -> @cuda(backend = cuTile, shift_int(Y, X)),
                   "UInt32 >> UInt32(16)" => () -> @cuda(backend = cuTile, shift_same_type(Y, X)),
                   "UInt32 << 3 (Int literal)" => () -> @cuda(backend = cuTile, shift_left_int(Y, X)),
                   "tile .>> 16 (Int literal)" => () -> @cuda(backend = cuTile, shift_tile_int(Y, X)))
end

# --- Loops over a tuple of arrays ---

function tuple_loop(Y::ct.TileArray{Float32,1}, A::ct.TileArray{Float32,1}, B::ct.TileArray{Float32,1})
    acc = zeros(Float32, 16)
    for X in (A, B)
        acc = acc .+ ct.load(X; index = Int32(1), shape = (16,))
    end
    ct.store(Y; index = Int32(1), tile = acc)
    return nothing
end

@inline add_all(acc) = acc
@inline add_all(acc, X, rest...) = add_all(acc .+ ct.load(X; index = Int32(1), shape = (16,)), rest...)

function tuple_recursion(Y::ct.TileArray{Float32,1}, A::ct.TileArray{Float32,1}, B::ct.TileArray{Float32,1})
    ct.store(Y; index = Int32(1), tile = add_all(zeros(Float32, 16), A, B))
    return nothing
end

let A = CUDA.ones(Float32, 16), B = CUDA.ones(Float32, 16), Y = CUDA.zeros(Float32, 16)
    compile_probe!("tuple_loop", "accumulating over a tuple of arrays",
                   "for X in (A, B)" => () -> @cuda(backend = cuTile, tuple_loop(Y, A, B)),
                   "recursive @inline helper" => () -> @cuda(backend = cuTile, tuple_recursion(Y, A, B)))
end
