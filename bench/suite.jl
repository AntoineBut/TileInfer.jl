# Benchmark suite: every op at a model's real shapes (the primitive ops, and the fused ones a
# step runs), at decode (1 token) and prefill (64 and 512 tokens) sizes, and end-to-end prefill and decode steps (eager, and replayed from a
# CUDA graph where the backend supports it), for every backend that implements them. Keys
# start with the model ("Qwen3-4B/ops/linear_qkv/decode/tile"); bench/run.jl and
# bench/profile.jl run the suite for each model of MODELS in turn.
#
# Each benchmark also records the bytes it must move and the FLOPs it must do, so results
# can be read against the GPU's roofline. Op samples run OP_REPS calls back to back and
# synchronize once: a sync per call would add more time than a small decode kernel takes.

using BenchmarkTools
using BenchmarkTools: leaves
using CUDA
using BFloat16s: BFloat16
using TileInfer
using TileInfer: DecoderLM, PagedKVCache, RefBackend, SamplingParams, TileBackend
using Random: Xoshiro

"Models benchmarked by default: a small one, dominated by per-kernel overheads, and a large one."
const MODELS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B")

"The first segment of a model's benchmark keys: the repo name without its owner."
model_label(repo::AbstractString) = basename(repo)

const OP_REPS = 20
const PREFILL = 512
const SHORT = 64            # a short prompt, or a follow-up turn of a chat
const CONTEXT = 1024        # cached positions seen by decode attention / decode steps
const GEN = 64              # tokens per generation benchmark

"""
    OpArgs(make_args, copies)

The argument tuples of an op benchmark: `copies` independent ones, cycled through OP_REPS
calls. Allocated on first use and dropped by `free!` once the benchmark has run, so that
only one benchmark's copies (≥ 2× L2, or one 200 MB weight) occupy the GPU at a time next
to the model's weights.
"""
mutable struct OpArgs{F}
    const make_args::F
    const copies::Int
    calls::Union{Nothing,Vector}
end

OpArgs(make_args, copies) = OpArgs(make_args, copies, nothing)

function calls!(a::OpArgs)
    if a.calls === nothing
        argsets = [a.make_args() for _ in 1:a.copies]
        a.calls = [argsets[mod1(i, a.copies)] for i in 1:OP_REPS]
    end
    return a.calls
end

free!(a::OpArgs) = (a.calls = nothing; a)

# One op sample: OP_REPS calls back to back, then one sync. Behind a function barrier, so the
# loop is compiled for the concrete type of `calls`.
function run_calls(op, be, calls)
    for args in calls
        op(be, args...)
    end
    CUDA.synchronize()
end

"What a benchmark measures, for the report."
struct Workload
    reps::Int               # calls per sample
    bytes::Float64          # minimum device memory traffic of one call
    flops::Float64          # arithmetic of one call
    tokens::Int             # tokens processed by one call (model benchmarks)
    args::Union{Nothing,OpArgs}   # an op benchmark's arguments, freed after it has run
end

Workload(reps, bytes, flops, tokens) = Workload(reps, bytes, flops, tokens, nothing)

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
    argtypes = map(typeof, make_args())                 # one set, only for its types
    args = OpArgs(make_args, copies)
    for be in BACKENDS
        key = [path..., backend_name(be)]
        # Skip backends without the op, and fallbacks that would only repeat another backend.
        occursin(filter, join(key, "/")) && hasmethod(op, Tuple{typeof(be), argtypes...}) &&
            TileInfer.has_kernel(be, op) || continue
        setleaf!(suite, key, @benchmarkable run_calls($op, $be, calls!($args)) evals = 1)
        info[join(key, "/")] = Workload(OP_REPS, bytes, flops, 0, args)
    end
end

"""
    build_suite(model, label; filter = r"", sampling = SamplingParams()) -> (suite, info)

Benchmarks for `model` (a DecoderLM; its backend is ignored), keyed
"<label>/ops/<op>/<size>/<backend>" and "<label>/model/<step>/<backend>", restricted to keys
matching `filter`. `info` maps keys to their [`Workload`](@ref). `sampling` is the model's
recommended sampling (see `TileInfer.generation_defaults`), for a sampled generation
benchmark when it is not greedy.
"""
function build_suite(model::DecoderLM{T}, label::AbstractString; filter::Regex = r"",
                     sampling::SamplingParams = SamplingParams()) where {T}
    cfg, w = model.config, model.weights
    H, I, D, nq, nkv = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim, cfg.num_heads, cfg.num_kv_heads
    eps, B = cfg.rms_norm_eps, sizeof(T)
    suite, info = BenchmarkGroup(), Dict{String,Workload}()
    rnd(dims...) = CUDA.rand(T, dims...)

    for (size_label, n) in (("decode", 1), ("prefill$SHORT", SHORT), ("prefill$PREFILL", PREFILL))
        ops(name) = [label, "ops", name, size_label]
        ids() = CuArray(rand(Int32(0):Int32(cfg.vocab_size - 1), n))
        # Reads one table column per token and writes it: the table is shared, not copied.
        add_op!(suite, info, filter, ops("embed"), TileInfer.embed, () -> (w.embed, ids()); bytes = 2.0 * H * n * B)
        add_op!(suite, info, filter, ops("rmsnorm"), TileInfer.rmsnorm, () -> (rnd(H, n), rnd(H), eps);
                bytes = (2.0 * H * n + H) * B)
        for (name, (in, out)) in (("qkv", size(first(w.layers).wqkv)), ("o", size(first(w.layers).wo)),
                                  ("gate_up", size(first(w.layers).w_gate_up)),
                                  ("down", size(first(w.layers).w_down)))
            add_op!(suite, info, filter, ops("linear_$name"), TileInfer.linear, () -> (rnd(in, out), rnd(in, n));
                    bytes = (in * out + (in + out) * n) * B, flops = 2.0 * in * out * n)
        end
        # The lm_head as a step runs it: the last token only, into Float32 logits. The weight
        # is the real one (311 MB for 0.6B, 778 MB for 4B) rather than copies.
        # (`let`: argument closures run lazily, after later assignments in this loop body.)
        n == 1 && let (in, out) = size(w.lm_head)
            add_op!(suite, info, filter, ops("linear_lm_head"), TileInfer.linear!,
                    () -> (CuMatrix{Float32}(undef, out, 1), w.lm_head, rnd(in, 1));
                    bytes = (in * out + in) * B + 4.0 * out, flops = 2.0 * in * out)
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

        # The fused ops of a step, in place as run_step! calls them (decode: 10 kernels per
        # layer, of which 6 are these). Their bytes are the unfused minimum, so % peak shows
        # what the fusion achieves.
        layer = first(w.layers)
        add_op!(suite, info, filter, ops("add_rmsnorm"), TileInfer.add_rmsnorm!,
                () -> (rnd(H, n), rnd(H, n), rnd(H, n), rnd(H), eps);
                bytes = (4.0 * H * n + H) * B)                   # read x, delta; write x, h
        let (in, out) = size(layer.w_gate_up)
            add_op!(suite, info, filter, ops("linear_swiglu"), TileInfer.linear_swiglu!,
                    () -> (rnd(out ÷ 2, n), rnd(out, n), rnd(in, out), rnd(in, n));
                    bytes = (in * out + (in + out ÷ 2) * n) * B, flops = 2.0 * in * out * n)
        end
        # q/k norm + rope from the fused qkv, keys and values into the pools.
        add_op!(suite, info, filter, ops("attention_inputs"), TileInfer.attention_inputs!,
                () -> (rnd(D, nq, n), pools()..., rnd((nq + 2nkv) * D, n), layer.q_norm, layer.k_norm, model.rope,
                       CuArray(Int32.(start:len-1)), CuArray(Int32.(start+1:start+n)), eps);
                bytes = (2.0 * nq + 4.0 * nkv + 2.0) * D * n * B)  # qkv in, q and k/v out, rope table
        # Device sampling with the model's recommended settings (top-1 if it is greedy).
        n == 1 && add_op!(suite, info, filter, ops("sample"), TileInfer.sample_token!,
                          () -> (CuVector{Int32}(undef, 1), CuVector{Int32}(undef, 1),
                                 CUDA.rand(Float32, cfg.vocab_size, 1),
                                 TileInfer.reset_sampler!(TileInfer.TokenSampler(cfg.vocab_size),
                                                          TileInfer.on_device(sampling) ? sampling : SamplingParams(),
                                                          UInt32(1)));
                          bytes = 4.0 * cfg.vocab_size)
    end

    weights = Float64(TileInfer.weight_bytes(w))
    params = weights / sizeof(T)
    for be in BACKENDS
        name = backend_name(be)
        prefill_keys = ("$label/model/prefill$SHORT/$name", "$label/model/prefill$PREFILL/$name")
        decode_key = "$label/model/decode_ctx$CONTEXT/$name"
        graph_key = "$label/model/decode_graph_ctx$CONTEXT/$name"    # through a ModelRunner
        # Greedy generation, pipelined and step by step, and with the model's sampling.
        gen_modes = [("", SamplingParams(), true), ("_unpipelined", SamplingParams(), false)]
        sampling.temperature > 0 && push!(gen_modes, ("_sampled", sampling, true))
        gen_keys = ["$label/model/generate$(GEN)$(suffix)_ctx$CONTEXT/$name" for (suffix, _, _) in gen_modes]
        keys = TileInfer.supports_graphs(be) ? (prefill_keys..., decode_key, graph_key, gen_keys...) :
                                               (prefill_keys..., decode_key)
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

        for (key, p) in zip(prefill_keys, (prompt[1:SHORT], prompt))
            occursin(filter, key) || continue
            setleaf!(suite, split(key, "/"), @benchmarkable begin
                seq = TileInfer.new_sequence($cache)
                TileInfer.forward!($m, $cache, seq, $p)
                CUDA.synchronize()
                TileInfer.release!($cache, seq)
            end evals = 1)
            info[key] = Workload(1, weights, 2 * params * length(p), length(p))
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
            # Generation of GEN tokens (times are per call; tok/s per token): greedy with the
            # host one step behind the GPU, greedy step by step, and with the model's
            # sampling (pipelined when it can be sampled on the device).
            for (key, (_, sp, pipeline)) in zip(gen_keys, gen_modes)
                occursin(filter, key) || continue
                rng = Xoshiro(1)
                gen() = TileInfer.generate!(runner, ctx, [42]; max_new_tokens = GEN, stop_ids = Int[],
                                            sampling = sp, rng, pipeline)
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

"""
    run_suite(suite, info) -> results

Warm up and run the benchmarks of `suite` one at a time, freeing each op benchmark's
arguments once it has run (see [`OpArgs`](@ref)).
"""
function run_suite(suite, info)
    results = BenchmarkGroup()
    for (path, b) in leaves(suite)
        warmup(b; verbose = false)
        setleaf!(results, path, run(b))
        free_args!(info[join(path, "/")])
    end
    return results
end

free_args!(w::Workload) = (w.args === nothing || free!(w.args); nothing)
