# RefBackend ops on small random inputs against straightforward CPU loops.

using LinearAlgebra: dot

@testset "reference ops" begin
    be = RefBackend()

    @testset "linear accumulates in Float32" begin
        # Guards against CUDA.jl's generic BFloat16 matmul (BFloat16 accumulation, ~30 % off).
        W = rand(Float32, 1024, 96); x = rand(Float32, 1024, 5)
        y = Array(TileInfer.linear(be, CuArray(BFloat16.(W)), CuArray(BFloat16.(x))))
        ref = Float32.(BFloat16.(W))' * Float32.(BFloat16.(x))
        @test maximum(abs.(Float32.(y) .- ref) ./ ref) < 1e-2
        b = rand(Float32, 96)
        @test Array(TileInfer.linear(be, CuArray(W), CuArray(x), CuArray(b))) ≈ W' * x .+ b
    end

    @testset "rmsnorm" begin
        x = randn(Float32, 64, 3, 5); w = randn(Float32, 64)
        ref = similar(x)
        for j in 1:3, t in 1:5
            col = x[:, j, t]
            ref[:, j, t] = w .* col ./ sqrt(sum(abs2, col) / 64 + 1f-6)
        end
        @test Array(TileInfer.rmsnorm(be, CuArray(x), CuArray(w), 1e-6)) ≈ ref
    end

    @testset "rope" begin
        D, H, L = 8, 2, 10
        positions = [7, 2, 9]                       # 0-based, looked up in tables of L positions
        n = length(positions)
        x = randn(Float32, D, H, n); θ = rand(Float32, D ÷ 2, L)
        c, s = vcat(cos.(θ), cos.(θ)), vcat(sin.(θ), sin.(θ))
        y = Array(TileInfer.rope(be, CuArray(x), CuArray(c), CuArray(s), CuArray(Int32.(positions))))
        for h in 1:H, t in 1:n, i in 1:D÷2
            # Element i rotates together with element i + D/2 by angle θ[i, position + 1].
            a, b, φ = x[i, h, t], x[i+D÷2, h, t], θ[i, positions[t]+1]
            @test y[i, h, t] ≈ a * cos(φ) - b * sin(φ)
            @test y[i+D÷2, h, t] ≈ b * cos(φ) + a * sin(φ)
        end
    end

    @testset "silu_mul" begin
        gu = randn(Float32, 6, 4)
        g, u = gu[1:3, :], gu[4:6, :]
        @test Array(TileInfer.silu_mul(be, CuArray(gu))) ≈ g ./ (1 .+ exp.(-g)) .* u
    end

    @testset "int8 weights" begin
        W = randn(Float32, 300, 40) .* rand(Float32, 1, 40)          # columns of different scales
        W[:, 7] .= 0
        Wq = TileInfer.quantize_int8(CuArray(W))
        q, scale = Array(Wq.q), Array(Wq.scale)
        @test size(Wq) == (300, 40) && maximum(abs, q) == 127
        @test scale[7] == 1 && all(iszero, q[:, 7])
        # Round to nearest: within half a step of each column's scale.
        @test all(abs.(Array(TileInfer.dequantize(Float32, Wq)) .- W) .<= scale' ./ 2 .* (1 + 1e-6))
        x = randn(Float32, 300, 3)
        exact = scale .* (Float64.(q)' * Float64.(x))
        @test Array(TileInfer.linear(be, Wq, CuArray(x))) ≈ exact rtol = 1e-5
        xb = CuArray(BFloat16.(x))
        @test Array(TileInfer.linear(be, Wq, xb)) ==
              BFloat16.(Array(TileInfer.linear(be, Wq, CuArray(Float32.(Array(xb))))))  # one rounding to T
    end

    @testset "xielu" begin
        a = TileInfer.XIELU{Float32}(1.3f0, 0.9f0, 0.5f0, -1f-6)
        x = Float32[-30, -2, -0.1, -1f-3, -1f-7, 0, 1f-7, 0.5, 3]
        f(x) = x > 0 ? 1.3 * x^2 + 0.5x : (expm1(min(Float64(x), -1e-6)) - x) * 0.9 + 0.5x
        @test Array(TileInfer.xielu(be, CuArray(x), a)) ≈ f.(x) rtol = 1e-6
        # In BFloat16 every operation rounds, as in PyTorch.
        b = TileInfer.XIELU{BFloat16}(1.3, 0.9, 0.5, -1e-6)
        xb = BFloat16.(x)
        stepwise(x) = x > 0 ? BFloat16(BFloat16(BFloat16(1.3) * x) * x) + BFloat16(0.5) * x :
                      BFloat16(BFloat16(BFloat16(expm1(Float32(min(x, BFloat16(-1e-6))))) - x) * BFloat16(0.9)) + BFloat16(0.5) * x
        @test Array(TileInfer.xielu(be, CuArray(xb), b)) == stepwise.(xb)
    end

    # Pages of a (head_dim, len, kv_heads) contiguous KV tensor, scattered over a pool in the
    # order given by `table`, through store_kv!.
    function paged(kv::Array{Float32,3}, table::Vector{Int32}, page_size::Integer, num_pages::Integer)
        D, len, H = size(kv)
        pool = CUDA.zeros(Float32, D, page_size, H, num_pages)
        slot_ids = Int32[(table[p ÷ page_size + 1] - 1) * page_size + p % page_size + 1 for p in 0:len-1]
        TileInfer.store_kv!(be, pool, CUDA.zeros(Float32, size(pool)), CuArray(permutedims(kv, (1, 3, 2))),
                            CUDA.zeros(Float32, D, H, len), CuArray(slot_ids))
        return pool
    end

    @testset "store_kv!" begin
        kv = randn(Float32, 8, 10, 2)
        table = Int32[5, 2, 4]                       # 3 pages of 4 positions, out of order
        pool = Array(paged(kv, table, 4, 5))
        for p in 0:9, h in 1:2
            @test pool[:, p % 4 + 1, h, table[p ÷ 4 + 1]] == kv[:, p + 1, h]
        end
        @test all(iszero, pool[:, :, :, [1, 3]])      # pages not in the table are untouched
    end

    @testset "attention (GQA, causal = $causal, start = $start)" for start in (0, 5), causal in (true, false)
        D, H, KVH, n, P = 16, 4, 2, 3, 4
        len = start + n
        q = randn(Float32, D, H, n)
        kc = randn(Float32, D, len, KVH); vc = randn(Float32, D, len, KVH)
        table = Int32[3, 1, 6][1:cld(len, P)]
        k_pool, v_pool = paged(kc, table, P, 6), paged(vc, table, P, 6)
        scale = 1 / sqrt(D)
        positions = CuArray(Int32.(start:start+n-1))
        padded = CuArray([table; zeros(Int32, 5)])  # entries past the sequence's pages are ignored
        o = Array(TileInfer.attention(be, CuArray(q), k_pool, v_pool, padded, positions, scale; causal))
        for h in 1:H, t in 1:n
            g = cld(h, H ÷ KVH)                        # query heads share KV heads in blocks
            keys = causal ? (1:start+t) : (1:len)      # causal: up to the query's own position
            s = [dot(kc[:, i, g], q[:, h, t]) * scale for i in keys]
            p = exp.(s .- maximum(s)); p ./= sum(p)
            @test o[:, h, t] ≈ vc[:, keys, g] * p
        end
    end
end
