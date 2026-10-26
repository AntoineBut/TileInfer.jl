using SafeTensors: SafeTensors

@testset "loader" begin
    @testset "file layout and fusion" begin
        mktempdir() do dir
            # PyTorch-shaped (out, in) weights, as a checkpoint stores them.
            q = rand(Float32, 6, 4)
            k = rand(Float32, 2, 4)
            b = rand(Float32, 6)
            SafeTensors.serialize(joinpath(dir, "model.safetensors"),
                                  Dict("q" => q, "k" => k, "b" => b))
            c = TileInfer.Checkpoint(dir)
            # Column-major layout is the transpose of the logical shape, without a copy.
            @test size(TileInfer.file_layout(c, "q")) == (4, 6)
            @test Array(TileInfer.upload(Float32, c, "q")) == permutedims(q)
            @test Array(TileInfer.upload(BFloat16, c, "q")) == BFloat16.(permutedims(q))
            @test Array(TileInfer.upload_fused(Float32, c, ["q", "k"])) == hcat(permutedims(q), permutedims(k))
            @test Array(TileInfer.upload_fused(Float32, c, ["b", "b"])) == [b; b]
            @test_throws ErrorException TileInfer.upload(Float32, c, "missing")
        end
    end

    dir = test_model_dir()
    if dir !== nothing
        @testset "$TEST_MODEL" begin
            cfg = TileInfer.load_config(dir)
            w = TileInfer.load_weights(BFloat16, cfg, dir)
            @test length(w.layers) == cfg.num_layers
            @test w.lm_head === w.embed
            # Qwen3-0.6B: 0.6B parameters (tied embeddings counted once), 2 bytes each.
            @test 1.1e9 < TileInfer.weight_bytes(w) < 1.3e9

            # Spot-check against the raw checkpoint: column t of embed is token t-1.
            c = TileInfer.Checkpoint(dir)
            emb = c.tensors["model.embed_tokens.weight"]           # logical (vocab, hidden)
            @test Array(w.embed[:, 1001]) == emb[1001, :]
            qp = c.tensors["model.layers.3.self_attn.q_proj.weight"]   # (heads*hd, hidden)
            vp = c.tensors["model.layers.3.self_attn.v_proj.weight"]   # (kv*hd, hidden)
            wqkv = Array(w.layers[4].wqkv)
            nq, nkv = cfg.num_heads * cfg.head_dim, cfg.num_kv_heads * cfg.head_dim
            @test wqkv[:, 1:nq] == permutedims(qp)
            @test wqkv[:, nq+nkv+1:end] == permutedims(vp)
        end
    end
end
