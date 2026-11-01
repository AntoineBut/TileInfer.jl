# Does moving a kernel's code into a function change what it compiles to? Each kernel is
# written inline, through an @inline helper, through a plain (not annotated) helper, and
# with a local closure, all computing the same thing.

# --- GEMV loop body (TileInfer's decode GEMV, qkv shape of Qwen3-0.6B: 1024 → 2048) ---

function gemv_inline(Y::ct.TileArray{Float32,2}, W::ct.TileArray{T,2}, X::ct.TileArray{T,2},
                     TK::Int, TM::Int) where {T}
    bid = ct.bid(1)
    acc = zeros(Float32, (TK, TM))
    for k in Int32(1):ct.num_tiles(W, 1, (TK, TM))
        w = convert(ct.Tile{Float32}, ct.load(W; index = (k, bid), shape = (TK, TM), padding_mode = ct.PaddingMode.Zero))
        x = convert(ct.Tile{Float32}, ct.load(X; index = (k, Int32(1)), shape = (TK, 1), padding_mode = ct.PaddingMode.Zero))
        acc = acc .+ w .* x
    end
    ct.store(Y; index = (bid, Int32(1)), tile = reshape(sum(acc; dims = 1), (TM, 1)))
    return nothing
end

@inline function gemv_step_inline(acc, W, X, k, bid, TK, TM)
    w = convert(ct.Tile{Float32}, ct.load(W; index = (k, bid), shape = (TK, TM), padding_mode = ct.PaddingMode.Zero))
    x = convert(ct.Tile{Float32}, ct.load(X; index = (k, Int32(1)), shape = (TK, 1), padding_mode = ct.PaddingMode.Zero))
    return acc .+ w .* x
end

function gemv_step_plain(acc, W, X, k, bid, TK, TM)
    w = convert(ct.Tile{Float32}, ct.load(W; index = (k, bid), shape = (TK, TM), padding_mode = ct.PaddingMode.Zero))
    x = convert(ct.Tile{Float32}, ct.load(X; index = (k, Int32(1)), shape = (TK, 1), padding_mode = ct.PaddingMode.Zero))
    return acc .+ w .* x
end

for (name, step) in ((:gemv_helper_inline, :gemv_step_inline), (:gemv_helper_plain, :gemv_step_plain))
    @eval function $name(Y::ct.TileArray{Float32,2}, W::ct.TileArray{T,2}, X::ct.TileArray{T,2},
                         TK::Int, TM::Int) where {T}
        bid = ct.bid(1)
        acc = zeros(Float32, (TK, TM))
        for k in Int32(1):ct.num_tiles(W, 1, (TK, TM))
            acc = $step(acc, W, X, k, bid, TK, TM)
        end
        ct.store(Y; index = (bid, Int32(1)), tile = reshape(sum(acc; dims = 1), (TM, 1)))
        return nothing
    end
end

function gemv_closure(Y::ct.TileArray{Float32,2}, W::ct.TileArray{T,2}, X::ct.TileArray{T,2},
                      TK::Int, TM::Int) where {T}
    bid = ct.bid(1)
    f32(tile) = convert(ct.Tile{Float32}, tile)
    acc = zeros(Float32, (TK, TM))
    for k in Int32(1):ct.num_tiles(W, 1, (TK, TM))
        w = f32(ct.load(W; index = (k, bid), shape = (TK, TM), padding_mode = ct.PaddingMode.Zero))
        x = f32(ct.load(X; index = (k, Int32(1)), shape = (TK, 1), padding_mode = ct.PaddingMode.Zero))
        acc = acc .+ w .* x
    end
    ct.store(Y; index = (bid, Int32(1)), tile = reshape(sum(acc; dims = 1), (TM, 1)))
    return nothing
end

let (in, out, TK, TM) = (1024, 2048, 1024, 8)
    W = CuArray(BFloat16.(randn(Float32, in, out) ./ 32))
    X = CuArray(BFloat16.(randn(Float32, in, 1)))
    Y = CUDA.zeros(Float32, out, 1)
    launch(k) = () -> @cuda backend = cuTile blocks = out ÷ TM k(Y, W, X, ct.Constant(TK), ct.Constant(TM))
    probe!("helpers_gemv", "GEMV loop body inline, in an @inline helper, a plain helper, or using a closure",
           "inline" => launch(gemv_inline), "@inline helper" => launch(gemv_helper_inline),
           "plain helper" => launch(gemv_helper_plain), "closure f32" => launch(gemv_closure);
           output = () -> Array(Y))
end

# --- Top-k selection loop (TileInfer's sampling chunk kernel, top_k = 20) ---

function topk_inline(Ids::ct.TileArray{Int32,1}, Logits::ct.TileArray{Float32,1}, Kp::ct.TileArray{Int32,1},
                     N::Int, K::Int)
    b = ct.bid(1)
    x = ct.load(Logits; index = b, shape = (N,), padding_mode = ct.PaddingMode.NegInf)
    pos, slot = ct.arange((N,)), ct.arange((K,))
    at = zeros(Int32, K)
    for r in Int32(1):Kp[Int32(1)]
        i = argmax(x; dims = 1)
        at = ifelse.(slot .== r, i, at)
        x = ifelse.(pos .== i, -Inf32, x)
    end
    ct.store(Ids; index = b, tile = at)
    return nothing
end

@inline function topk_positions_inline(x, k, K, N)
    pos, slot = ct.arange((N,)), ct.arange((K,))
    at = zeros(Int32, K)
    for r in Int32(1):k
        i = argmax(x; dims = 1)
        at = ifelse.(slot .== r, i, at)
        x = ifelse.(pos .== i, -Inf32, x)
    end
    return at
end

function topk_positions_plain(x, k, K, N)
    pos, slot = ct.arange((N,)), ct.arange((K,))
    at = zeros(Int32, K)
    for r in Int32(1):k
        i = argmax(x; dims = 1)
        at = ifelse.(slot .== r, i, at)
        x = ifelse.(pos .== i, -Inf32, x)
    end
    return at
end

for (name, helper) in ((:topk_helper_inline, :topk_positions_inline), (:topk_helper_plain, :topk_positions_plain))
    @eval function $name(Ids::ct.TileArray{Int32,1}, Logits::ct.TileArray{Float32,1}, Kp::ct.TileArray{Int32,1},
                         N::Int, K::Int)
        b = ct.bid(1)
        x = ct.load(Logits; index = b, shape = (N,), padding_mode = ct.PaddingMode.NegInf)
        ct.store(Ids; index = b, tile = $helper(x, Kp[Int32(1)], K, N))
        return nothing
    end
end

let (V, N, K) = (151936, 4096, 64)
    logits = CUDA.randn(Float32, V)
    ids = CUDA.zeros(Int32, cld(V, N) * K)
    kp = CuVector{Int32}([20])
    launch(k) = () -> @cuda backend = cuTile blocks = cld(V, N) k(ids, logits, kp, ct.Constant(N), ct.Constant(K))
    probe!("helpers_topk", "top-k selection loop inline, in an @inline helper, or in a plain helper",
           "inline" => launch(topk_inline), "@inline helper" => launch(topk_helper_inline),
           "plain helper" => launch(topk_helper_plain);
           output = () -> Array(ids))
end
