# Census of TileInfer's kernels: every op a step runs, at the shapes of Qwen3-0.6B and 4B,
# for decode (1 token at 1,024 positions) and prefill (64 and 512 tokens), as run_step!
# calls them. Not a comparison: the report lists each kernel's time, bandwidth, block size,
# registers, spills and SASS, to spot outliers. Weights rotate through copies (`rotating`),
# so they come from DRAM as in the model; activations stay in L2.

using TileInfer: TileInfer
const TI = TileInfer

function census!(repo)
    label = basename(repo)
    cfg = TI.load_config(TI.model_dir(repo))
    T = BFloat16
    Hd, I, D, nq, nkv, V = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim, cfg.num_heads, cfg.num_kv_heads, cfg.vocab_size
    be = TI.TileBackend()
    rnd(dims...) = CuArray(T.(randn(Float32, dims...) ./ 8))
    B = 2.0                                                     # bytes per element
    P, ctx = 64, 1024
    pages = cld(ctx + 512, P)
    kp, vp = rnd(D, P, nkv, pages), rnd(D, P, nkv, pages)
    bt = CuVector{Int32}([1:pages; zeros(Int32, cld(pages, 16) * 16 - pages)])
    table = TI.RotaryTable{T}(cfg, 4096)
    norm_w, qn, kn = rnd(Hd), rnd(D), rnd(D)
    embed_table = rnd(Hd, V)
    w(in, out) = rotating(() -> rnd(in, out), in * out * B)
    Wqkv, Wo, Wgu, Wd = w(Hd, (nq + 2nkv) * D), w(nq * D, Hd), w(Hd, 2I), w(I, Hd)
    lm_head = embed_table                                       # one copy: 311 MB (0.6B) to 778 MB
    sampler = TI.reset_sampler!(TI.TokenSampler(V), TI.SamplingParams(temperature = 0.6, top_k = 20, top_p = 0.95),
                                UInt32(1))

    for (step, n) in (("decode", 1), ("prefill64", 64), ("prefill512", 512))
        start = n == 1 ? ctx - 1 : 0
        positions, slots = CuVector{Int32}(start:start+n-1), CuVector{Int32}(start+1:start+n)
        ids = CuVector{Int32}(rand(0:V-1, n))
        x, h, delta = rnd(Hd, n), rnd(Hd, n), rnd(Hd, n)
        qkv, q, attn = rnd((nq + 2nkv) * D, n), rnd(D, nq, n), rnd(D, nq, n)
        gate_up, act, proj = rnd(2I, n), rnd(I, n), rnd(Hd, n)
        ws = TI.attention_workspace(be, q, kp)
        variants, bytes = Pair{String,Function}[], Float64[]
        add(name, f, b) = (push!(variants, name => f); push!(bytes, b))
        add("embed", () -> TI.embed!(be, x, embed_table, ids), 2Hd * n * B)
        add("rmsnorm", () -> TI.rmsnorm!(be, h, x, norm_w, 1e-6), (2Hd * n + Hd) * B)
        linear(W, y, xin) = () -> TI.linear!(be, y, W(), xin)
        add("linear qkv", linear(Wqkv, qkv, h), Hd * (nq + 2nkv) * D * B)
        add("attention inputs", () -> TI.attention_inputs!(be, q, kp, vp, qkv, qn, kn, table, positions, slots, 1e-6),
            (2nq + 4nkv) * D * n * B)
        kv_len = start + n
        add("attention", () -> TI.attention!(be, attn, ws, q, kp, vp, bt, positions, 1 / sqrt(D)),
            (2kv_len * nkv * D + 2nq * D * n) * B)
        add("linear o", linear(Wo, proj, reshape(attn, :, n)), nq * D * Hd * B)
        add("add_rmsnorm", () -> TI.add_rmsnorm!(be, x, h, delta, norm_w, 1e-6), (4Hd * n + Hd) * B)
        add("linear gate_up + SwiGLU", () -> TI.linear_swiglu!(be, act, gate_up, Wgu(), h), Hd * 2I * B)
        add("linear down", linear(Wd, proj, act), I * Hd * B)
        if n == 1
            logits = CuMatrix{Float32}(undef, V, 1)
            add("lm_head (Float32 logits)", () -> TI.linear!(be, logits, lm_head, h), Hd * V * B)
            token = CuVector{Int32}(undef, 1)
            add("sample (top-k 20, top-p 0.95)", () -> TI.sample_token!(be, token, token, logits, sampler), 4V)
        end
        probe!("census_$(label)_$step", "TileInfer's kernels for one $step step of $label (n = $n)",
               variants...; bytes)
    end
end

census!("Qwen/Qwen3-0.6B")
census!("Qwen/Qwen3-4B")
