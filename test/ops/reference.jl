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
        D, H, n = 8, 2, 3
        x = randn(Float32, D, H, n); θ = rand(Float32, D ÷ 2, n)
        c, s = vcat(cos.(θ), cos.(θ)), vcat(sin.(θ), sin.(θ))
        y = Array(TileInfer.rope(be, CuArray(x), CuArray(c), CuArray(s)))
        for h in 1:H, t in 1:n, i in 1:D÷2
            # Element i rotates together with element i + D/2 by angle θ[i, t].
            a, b = x[i, h, t], x[i+D÷2, h, t]
            @test y[i, h, t] ≈ a * cos(θ[i, t]) - b * sin(θ[i, t])
            @test y[i+D÷2, h, t] ≈ b * cos(θ[i, t]) + a * sin(θ[i, t])
        end
    end

    @testset "silu_mul" begin
        gu = randn(Float32, 6, 4)
        g, u = gu[1:3, :], gu[4:6, :]
        @test Array(TileInfer.silu_mul(be, CuArray(gu))) ≈ g ./ (1 .+ exp.(-g)) .* u
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

    @testset "attention (GQA, causal, start = $start)" for start in (0, 5)
        D, H, KVH, n, P = 16, 4, 2, 3, 4
        len = start + n
        q = randn(Float32, D, H, n)
        kc = randn(Float32, D, len, KVH); vc = randn(Float32, D, len, KVH)
        table = Int32[3, 1, 6][1:cld(len, P)]
        k_pool, v_pool = paged(kc, table, P, 6), paged(vc, table, P, 6)
        scale = 1 / sqrt(D)
        o = Array(TileInfer.attention(be, CuArray(q), k_pool, v_pool, CuArray(table), start, scale))
        for h in 1:H, t in 1:n
            g = cld(h, H ÷ KVH)                        # query heads share KV heads in blocks
            keys = 1:start+t                           # causal: up to the query's own position
            s = [dot(kc[:, i, g], q[:, h, t]) * scale for i in keys]
            p = exp.(s .- maximum(s)); p ./= sum(p)
            @test o[:, h, t] ≈ vc[:, keys, g] * p
        end
    end
end
