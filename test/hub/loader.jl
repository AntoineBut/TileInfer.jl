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

    @testset "Apertus 1.5 layout" begin
        # A tiny checkpoint named and shaped like Apertus 1.5: nested decoder, ungated xIELU
        # MLP, sublayer-named norms, q/k norms and an embedding with extra (image) rows.
        H, I, D, nq, nkv, V, Vin = 8, 12, 4, 2, 1, 16, 24
        text = Dict{String,Any}("model_type" => "apertus1p5_text", "vocab_size" => Vin, "output_vocab_size" => V,
                                "hidden_size" => H, "intermediate_size" => I, "num_hidden_layers" => 2,
                                "num_attention_heads" => nq, "num_key_value_heads" => nkv, "head_dim" => D,
                                "rms_norm_eps" => 1e-5, "max_position_embeddings" => 64, "qk_norm" => true,
                                "hidden_act" => "xielu", "dtype" => "bfloat16", "eos_token_id" => [2],
                                "rope_parameters" => Dict("rope_theta" => 4e6, "rope_type" => "default"))
        m = "model.language_model."
        tensors = Dict{String,Array{Float32}}("lm_head.weight" => rand(Float32, V, H),
                                              m * "embed_tokens.weight" => rand(Float32, Vin, H),
                                              m * "norm.weight" => rand(Float32, H),
                                              "model.vision_tokenizer.quantize.weight" => rand(Float32, 3, 2))
        for l in 0:1
            p = "$(m)layers.$l."
            for (name, dims) in (("attention_layernorm", (H,)), ("feedforward_layernorm", (H,)),
                                 ("self_attn.q_proj", (nq * D, H)), ("self_attn.k_proj", (nkv * D, H)),
                                 ("self_attn.v_proj", (nkv * D, H)), ("self_attn.o_proj", (H, nq * D)),
                                 ("self_attn.q_norm", (D,)), ("self_attn.k_norm", (D,)),
                                 ("mlp.up_proj", (I, H)), ("mlp.down_proj", (H, I)))
                tensors[p * name * ".weight"] = rand(Float32, dims...)
            end
            # As stored: raw parameters, the inverse softplus of αp = 0.8 and of αn - β = 0.3.
            for (name, x) in (("alpha_p", log(expm1(0.8f0))), ("alpha_n", log(expm1(0.3f0))),
                              ("beta", 0.5f0), ("eps", -1f-6))
                tensors[p * "mlp.act_fn." * name] = [x]
            end
        end
        mktempdir() do dir
            SafeTensors.serialize(joinpath(dir, "model.safetensors"), tensors)
            write(joinpath(dir, "config.json"),
                  TileInfer.JSON.json(Dict("model_type" => "apertus1p5", "text_config" => text)))
            cfg = TileInfer.load_config(dir)
            @test cfg.vocab_size == V
            w = TileInfer.load_weights(Float32, cfg, dir)
            @test Array(w.embed) == permutedims(tensors[m * "embed_tokens.weight"][1:V, :])
            @test Array(w.lm_head) == permutedims(tensors["lm_head.weight"])
            l = w.layers[2]
            p = "$(m)layers.1."
            @test Array(l.attn_norm) == tensors[p * "attention_layernorm.weight"]
            @test Array(l.mlp_norm) == tensors[p * "feedforward_layernorm.weight"]
            @test Array(l.q_norm) == tensors[p * "self_attn.q_norm.weight"]
            @test Array(l.w_gate_up) == permutedims(tensors[p * "mlp.up_proj.weight"])
            @test size(l.w_down) == (I, H)
            @test l.xielu.alpha_p ≈ 0.8f0 rtol = 1e-4
            @test l.xielu.alpha_n ≈ 0.5f0 + 0.3f0 rtol = 1e-4
            @test (l.xielu.beta, l.xielu.eps) == (0.5f0, -1f-6)
            @test TileInfer.load_weights(BFloat16, cfg, dir).layers[1].xielu isa TileInfer.XIELU{BFloat16}
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
