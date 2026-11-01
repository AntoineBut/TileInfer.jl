# Census of TileInfer's kernels: every op a step runs, at the shapes of Qwen3-0.6B and 4B
# (BFloat16 and INT8 weights) and Apertus 1.5 8B (INT8, as it runs on 16 GB), for decode
# (1 token at 1,024 positions), spec-decoding verify steps (8 and 16 tokens at 1,024
# positions: k = 7 drafts, DFlash's block of 16) and prefill (64 and 512 tokens), as
# run_step! calls them. Not a comparison: the report lists each kernel's time, bandwidth,
# block size, registers, spills and SASS, to spot outliers. Weights rotate through copies
# (`rotating`), so they come from DRAM as in the model; activations stay in L2.

using TileInfer: TileInfer
const TI = TileInfer

function census!(repo; quantize::Union{Nothing,Symbol} = nothing,
                 steps = (("decode", 1), ("verify8", 8), ("verify16", 16), ("prefill64", 64), ("prefill512", 512)))
    label = basename(repo) * (quantize === nothing ? "" : "-$quantize")
    cfg = TI.load_config(TI.model_dir(repo))
    T = BFloat16
    Hd, I, D, nq, nkv, V = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim, cfg.num_heads, cfg.num_kv_heads, cfg.vocab_size
    be = TI.TileBackend()
    rnd(dims...) = CuArray(T.(randn(Float32, dims...) ./ 8))
    B = 2.0                                                     # bytes per activation element
    WB = quantize === :int8 ? 1.0 : B                           # bytes per weight element
    weight(in, out) = quantize === :int8 ? TI.quantize_int8(rnd(in, out)) : rnd(in, out)
    P, ctx = 64, 1024
    pages = cld(ctx + 512, P)
    kp, vp = rnd(D, P, nkv, pages), rnd(D, P, nkv, pages)
    bt = CuVector{Int32}([1:pages; zeros(Int32, cld(pages, 16) * 16 - pages)])
    table = TI.RotaryTable{T}(cfg, 4096)
    norm_w, qn, kn = rnd(Hd), rnd(D), rnd(D)
    embed_table = rnd(Hd, V)
    w(in, out) = rotating(() -> weight(in, out), in * out * WB)
    xielu = cfg.hidden_act === :xielu
    mlp_out = xielu ? I : 2I                                    # ungated xIELU: up only
    Wqkv, Wo, Wgu, Wd = w(Hd, (nq + 2nkv) * D), w(nq * D, Hd), w(Hd, mlp_out), w(I, Hd)
    a = TI.XIELU{T}(0.8, 0.8, 0.5, -1e-6)
    # One copy: 311 MB (0.6B) to 778 MB (4B) in BF16; the embedding stays BF16 when quantized.
    lm_head = quantize === :int8 ? TI.quantize_int8(embed_table) : embed_table
    sampler = TI.reset_sampler!(TI.TokenSampler(V), TI.SamplingParams(temperature = 0.6, top_k = 20, top_p = 0.95),
                                UInt32(1))

    for (step, n) in steps
        start = n <= 16 ? ctx - n : 0
        positions, slots = CuVector{Int32}(start:start+n-1), CuVector{Int32}(start+1:start+n)
        ids = CuVector{Int32}(rand(0:V-1, n))
        x, h, delta = rnd(Hd, n), rnd(Hd, n), rnd(Hd, n)
        qkv, q, attn = rnd((nq + 2nkv) * D, n), rnd(D, nq, n), rnd(D, nq, n)
        gate_up, act, proj = rnd(mlp_out, n), rnd(I, n), rnd(Hd, n)
        ws = TI.attention_workspace(be, q, kp)
        variants, bytes = Pair{String,Function}[], Float64[]
        add(name, f, b) = (push!(variants, name => f); push!(bytes, b))
        add("embed", () -> TI.embed!(be, x, embed_table, ids), 2Hd * n * B)
        add("rmsnorm", () -> TI.rmsnorm!(be, h, x, norm_w, 1e-6), (2Hd * n + Hd) * B)
        linear(W, y, xin) = () -> TI.linear!(be, y, W(), xin)
        add("linear qkv", linear(Wqkv, qkv, h), Hd * (nq + 2nkv) * D * WB)
        add("attention inputs", () -> TI.attention_inputs!(be, q, kp, vp, qkv, qn, kn, table, positions, slots, 1e-6),
            (2nq + 4nkv) * D * n * B)
        kv_len = start + n
        add("attention", () -> TI.attention!(be, attn, ws, q, kp, vp, bt, positions, 1 / sqrt(D)),
            (2kv_len * nkv * D + 2nq * D * n) * B)
        add("linear o", linear(Wo, proj, reshape(attn, :, n)), nq * D * Hd * WB)
        add("add_rmsnorm", () -> TI.add_rmsnorm!(be, x, h, delta, norm_w, 1e-6), (4Hd * n + Hd) * B)
        if xielu
            add("linear up + xIELU", () -> TI.linear_xielu!(be, act, Wgu(), h, a), Hd * I * WB)
        else
            add("linear gate_up + SwiGLU", () -> TI.linear_swiglu!(be, act, gate_up, Wgu(), h), Hd * 2I * WB)
        end
        add("linear down", linear(Wd, proj, act), I * Hd * WB)
        if n <= 16
            # A decode step's logits for the last token; a verify step's for every token.
            logits = CuMatrix{Float32}(undef, V, n)
            add("lm_head (Float32 logits)", () -> TI.linear!(be, logits, lm_head, h), Hd * V * WB + 4V * n)
            if n == 1
                token = CuVector{Int32}(undef, 1)
                add("sample (top-k 20, top-p 0.95)", () -> TI.sample_token!(be, token, token, logits, sampler), 4V)
            end
        end
        probe!("census_$(label)_$step", "TileInfer's kernels for one $step step of $label (n = $n)",
               variants...; bytes)
    end
end

census!("Qwen/Qwen3-0.6B")
census!("Qwen/Qwen3-4B")
census!("Qwen/Qwen3-0.6B"; quantize = :int8)
census!("Qwen/Qwen3-4B"; quantize = :int8)
# Apertus 1.5 8B only fits the GPU in INT8; its 512-token prefill activations are small next
# to the weights, so it runs every step too.
census!("swiss-ai/Apertus-v1.5-8B"; quantize = :int8)
