# Benchmark suite: every op at the model's real shapes, at decode (1 token) and prefill
# (512 tokens) sizes, and end-to-end prefill and decode steps (eager, and replayed from a
# CUDA graph where the backend supports it), for every backend that implements them. Used
# by bench/run.jl.
#
# Each benchmark also records the bytes it must move and the FLOPs it must do, so results
# can be read against the GPU's roofline. Op samples run OP_REPS calls back to back and
# synchronize once: a sync per call would add more time than a small decode kernel takes.

using BenchmarkTools
using BenchmarkTools: leaves
using CUDA
using BFloat16s: BFloat16
using TileInfer
using TileInfer: DecoderLM, PagedKVCache, RefBackend, TileBackend

const OP_REPS = 20
const PREFILL = 512
const CONTEXT = 1024        # cached positions seen by decode attention / decode steps
const GEN = 64              # tokens per generation benchmark

"What a benchmark measures, for the report."
struct Workload
    reps::Int               # calls per sample
    bytes::Float64          # minimum device memory traffic of one call
    flops::Float64          # arithmetic of one call
    tokens::Int             # tokens processed by one call (model benchmarks)
end

const BACKENDS = (RefBackend(), TileBackend())

backend_name(::RefBackend) = "ref"
backend_name(::TileBackend) = "tile"

# Stores `value` at `path`, creating the intermediate groups.
function setleaf!(g::BenchmarkGroup, path, value)
    for k in path[1:end-1]
        haskey(g, k) || (g[k] = BenchmarkGroup())
        g = g[k]
    end
    g[path[end]] = value
end

const L2_BYTES = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_L2_CACHE_SIZE)

"""
    add_op!(suite, info, filter, path, op, make_args; bytes, flops = 0)

Benchmark `op(backend, args...)` for each backend that has a method for these arguments.
`bytes` is the call's minimum DRAM traffic. `make_args()` returns a fresh argument tuple;
a sample cycles through enough independent copies (≥ 2× the L2 cache) that each call
reads from DRAM, as it would in a model where 28 layers of weights stream through L2.
"""
function add_op!(suite, info, filter, path, op, make_args; bytes, flops = 0.0)
    keys = [join([path..., backend_name(be)], "/") for be in BACKENDS]
    any(k -> occursin(filter, k), keys) || return
    copies = clamp(cld(2 * L2_BYTES, round(Int, bytes)), 1, OP_REPS)
    argsets = [make_args() for _ in 1:copies]
    calls = [argsets[mod1(i, copies)] for i in 1:OP_REPS]
    for be in BACKENDS
        key = [path..., backend_name(be)]
        # Skip backends without the op, and fallbacks that would only repeat another backend.
        occursin(filter, join(key, "/")) && applicable(op, be, first(argsets)...) &&
            TileInfer.has_kernel(be, op) || continue
        setleaf!(suite, key, @benchmarkable begin
            for args in $calls
                $op($be, args...)
            end
            CUDA.synchronize()
        end evals = 1)
        info[join(key, "/")] = Workload(OP_REPS, bytes, flops, 0)
    end
end

"""
    build_suite(model; filter = r"") -> (suite, info)

Benchmarks for `model` (a DecoderLM; its backend is ignored), keyed
"ops/<op>/<size>/<backend>" and "model/<step>/<backend>", restricted to keys matching
`filter`. `info` maps keys to their [`Workload`](@ref).
"""
function build_suite(model::DecoderLM{T}; filter::Regex = r"") where {T}
    cfg, w = model.config, model.weights
    H, I, D, nq, nkv = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim, cfg.num_heads, cfg.num_kv_heads
    eps, B = cfg.rms_norm_eps, sizeof(T)
    suite, info = BenchmarkGroup(), Dict{String,Workload}()
    rnd(dims...) = CUDA.rand(T, dims...)

    for (label, n) in (("decode", 1), ("prefill$PREFILL", PREFILL))
        ops(name) = ["ops", name, label]
        ids() = CuArray(rand(Int32(0):Int32(cfg.vocab_size - 1), n))
        # Reads one table column per token and writes it: the table is shared, not copied.
        add_op!(suite, info, filter, ops("embed"), TileInfer.embed, () -> (w.embed, ids()); bytes = 2.0 * H * n * B)
        add_op!(suite, info, filter, ops("rmsnorm"), TileInfer.rmsnorm, () -> (rnd(H, n), rnd(H), eps);
                bytes = (2.0 * H * n + H) * B)
        for (name, (in, out)) in (("qkv", size(first(w.layers).wqkv)), ("o", size(first(w.layers).wo)),
                                  ("gate_up", size(first(w.layers).w_gate_up)),
                                  ("down", size(first(w.layers).w_down)), ("lm_head", size(w.lm_head)))
            name == "lm_head" && n > 1 && continue              # prefill only needs the last token
            # The lm_head is 311 MB: use the real one rather than allocating copies.
            weight() = name == "lm_head" ? w.lm_head : rnd(in, out)
            add_op!(suite, info, filter, ops("linear_$name"), TileInfer.linear, () -> (weight(), rnd(in, n));
                    bytes = (in * out + (in + out) * n) * B, flops = 2.0 * in * out * n)
        end
        # The model's own rope tables: a step reads n of their columns.
        positions() = CuArray(Int32.(rand(0:length(model.rope)-1, n)))
        add_op!(suite, info, filter, ops("rope"), TileInfer.rope,
                () -> (rnd(D, nq, n), model.rope.cos, model.rope.sin, positions());
                bytes = (2.0 * D * nq * n + 2.0 * D * n) * B)
        add_op!(suite, info, filter, ops("silu_mul"), TileInfer.silu_mul, () -> (rnd(2I, n),); bytes = 3.0 * I * n * B)

        # One layer's page pools, holding `len` positions of a single sequence.
        start = n == 1 ? CONTEXT - 1 : 0
        len = start + n
        P = TileInfer.DEFAULT_PAGE_SIZE
        pages = cld(len, P)
        pools() = (rnd(D, P, nkv, pages), rnd(D, P, nkv, pages))
        kv_bytes(positions) = 2.0 * positions * D * nkv * B
        add_op!(suite, info, filter, ops("store_kv"), TileInfer.store_kv!,
                () -> (pools()..., rnd(D, nkv, n), rnd(D, nkv, n), CuArray(Int32.(start+1:start+n)));
                bytes = 2 * kv_bytes(n))                        # read k, v and write n slots
        # Causal prefill visits about half of the (len × n) score matrix.
        visited = n == 1 ? len : n * (n + 1) / 2
        add_op!(suite, info, filter, ops("attention"), TileInfer.attention,
                () -> (rnd(D, nq, n), pools()..., CuArray(Int32.(1:pages)), CuArray(Int32.(start:len-1)), 1 / sqrt(D));
                bytes = kv_bytes(len) + 2.0 * D * nq * n * B, flops = 4.0 * D * nq * visited)
    end

    weights = Float64(TileInfer.weight_bytes(w))
    params = weights / sizeof(T)
    for be in BACKENDS
        name = backend_name(be)
        prefill_key, decode_key = "model/prefill$PREFILL/$name", "model/decode_ctx$CONTEXT/$name"
        graph_key = "model/decode_graph_ctx$CONTEXT/$name"           # through a ModelRunner
        gen_keys = ("model/generate$(GEN)_ctx$CONTEXT/$name", "model/generate$(GEN)_unpipelined_ctx$CONTEXT/$name")
        keys = TileInfer.supports_graphs(be) ? (prefill_key, decode_key, graph_key, gen_keys...) : (prefill_key, decode_key)
        any(k -> occursin(filter, k), keys) || continue
        m = DecoderLM(cfg, w, model.rope, be)
        cache = PagedKVCache(m; max_tokens = CONTEXT + PREFILL)
        prompt = rand(0:cfg.vocab_size-1, PREFILL)
        seq = TileInfer.new_sequence(cache)
        try
            TileInfer.forward!(m, cache, seq, prompt[1:CONTEXT÷2])   # also checks that every op exists
        catch e
            e isa MethodError || rethrow()
            continue
        end
        TileInfer.release!(cache, seq)

        if occursin(filter, prefill_key)
            setleaf!(suite, split(prefill_key, "/"), @benchmarkable begin
                seq = TileInfer.new_sequence($cache)
                TileInfer.forward!($m, $cache, seq, $prompt)
                CUDA.synchronize()
                TileInfer.release!($cache, seq)
            end evals = 1)
            info[prefill_key] = Workload(1, weights, 2 * params * PREFILL, PREFILL)
        end

        # One new token after CONTEXT - 1 cached positions, rewound after each sample: eagerly
        # (buffers allocated per step), and replaying a CUDA graph through a ModelRunner.
        ctx = TileInfer.new_sequence(cache)
        TileInfer.forward!(m, cache, ctx, rand(0:cfg.vocab_size-1, CONTEXT - 1))
        kv = Float64(CONTEXT * TileInfer.bytes_per_token(cfg, T))
        if occursin(filter, decode_key)
            setleaf!(suite, split(decode_key, "/"), @benchmarkable begin
                TileInfer.forward!($m, $cache, $ctx, [42])
                CUDA.synchronize()
                TileInfer.truncate!($cache, $ctx, $(CONTEXT - 1))
            end evals = 1)
            info[decode_key] = Workload(1, weights + kv, 2 * params, 1)
        end
        if graph_key in keys && any(k -> occursin(filter, k), (graph_key, gen_keys...))
            runner = TileInfer.ModelRunner(m, cache)
            TileInfer.forward!(runner, ctx, [42])                     # captures the graph
            TileInfer.truncate!(cache, ctx, CONTEXT - 1)
            if occursin(filter, graph_key)
                setleaf!(suite, split(graph_key, "/"), @benchmarkable begin
                    TileInfer.forward!($runner, $ctx, [42])
                    CUDA.synchronize()
                    TileInfer.truncate!($cache, $ctx, $(CONTEXT - 1))
                end evals = 1)
                info[graph_key] = Workload(1, weights + kv, 2 * params, 1)
            end
            # Greedy generation of GEN tokens (times are per call; tok/s per token), with the
            # host one step behind the GPU and, for comparison, step by step.
            for pipeline in (true, false)
                key = "model/generate$(GEN)$(pipeline ? "" : "_unpipelined")_ctx$CONTEXT/$name"
                occursin(filter, key) || continue
                gen() = TileInfer.generate!(runner, ctx, [42]; max_new_tokens = GEN, stop_ids = Int[], pipeline)
                gen(); CUDA.synchronize(); TileInfer.truncate!(cache, ctx, CONTEXT - 1)
                setleaf!(suite, split(key, "/"), @benchmarkable begin
                    $gen()
                    CUDA.synchronize()
                    TileInfer.truncate!($cache, $ctx, $(CONTEXT - 1))
                end evals = 1)
                info[key] = Workload(1, GEN * (weights + kv), 2 * params * GEN, GEN)
            end
        end
    end

    return suite, info
end
