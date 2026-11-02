# Benchmark suite: the ops a step runs at a model's real shapes (fused, and the primitive ones
# they are built from; rope and store_kv, which only the unfused fallback calls, are left out),
# at decode (1 token) and prefill (64 and 1,024 tokens) sizes, on the cuTile backend and
# optionally the reference one; and the model at the reference point of
# docs/src/performance.md (a 1,024-token prefill, 128 tokens decoded after 1,024 positions),
# for the engine and optionally for configurations with one feature switched off. Keys
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
include("variants.jl")

"Models benchmarked by default: a small one, dominated by per-kernel overheads, and a large one."
const MODELS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B")

"The first segment of a model's benchmark keys: the repo name without its owner."
model_label(repo::AbstractString) = basename(repo)

const OP_REPS = 20
# The reference point of docs/src/performance.md: a PREFILL-token prompt in one step, then
# GEN greedy tokens decoded after CONTEXT cached positions.
const PREFILL = 1024
const CONTEXT = 1024        # cached positions seen by decode attention / decode steps
const GEN = 128             # tokens per decode benchmark
const SHORT = 64            # op benchmarks only: a short prompt, a follow-up turn, a verify step

"Positions a model and its cache need for the model benchmarks after `context` positions."
model_len(context) = context + PREFILL + GEN + 64

"""
    ModelConfig(name, backend; graphs = true, pipeline = graphs, prefill = true)

An engine configuration of the model benchmarks: its backend, and whether decode steps
replay CUDA graphs and run pipelined (the host one step behind the GPU). Prefill steps run
eagerly whatever the graph settings, so `prefill = false` skips a prefill benchmark that
would repeat another configuration's.
"""
struct ModelConfig
    name::String
    backend::TileInfer.Backend
    graphs::Bool
    pipeline::Bool
    prefill::Bool
end

ModelConfig(name, backend; graphs = true, pipeline = graphs, prefill = true) =
    ModelConfig(name, backend, graphs, pipeline, prefill)

"""
The model benchmarks' configurations: "tile" is the engine, the others (run with `--ref`)
switch features off (see bench/variants.jl), from the reference ops up.
"""
const MODEL_CONFIGS = [
    ModelConfig("tile", TileBackend()),
    ModelConfig("tile-steps", TileBackend(); pipeline = false, prefill = false),   # graphs, not pipelined
    ModelConfig("tile-eager", TileBackend(); graphs = false, prefill = false),
    ModelConfig("unfused", Variant(fused = false)),                                # a kernel per op
    ModelConfig("unfused-steps", Variant(fused = false); pipeline = false, prefill = false),
    ModelConfig("unfused-eager", Variant(fused = false); graphs = false, prefill = false),
    ModelConfig("unfused-refattn-eager", Variant(fused = false, attention = :ref); graphs = false),
    ModelConfig("cublas", Variant(linear = :cublas)),                    # cuBLAS for every linear
    ModelConfig("ref", RefBackend(); graphs = false),
]

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
"Backends benchmarked by default: tracking the cuTile kernels is the point; the reference is opt-in."
const DEFAULT_BACKENDS = (TileBackend(),)

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
    add_op!(suite, info, filter, backends, path, op, make_args; bytes, flops = 0)

Benchmark `op(backend, args...)` for each of `backends` that has a method for these arguments.
`bytes` is the call's minimum DRAM traffic. `make_args()` returns a fresh argument tuple;
a sample cycles through enough independent copies (≥ 2× the L2 cache) that each call
reads from DRAM, as it would in a model where 28 layers of weights stream through L2.
"""
function add_op!(suite, info, filter, backends, path, op, make_args; bytes, flops = 0.0)
    keys = [join([path..., backend_name(be)], "/") for be in backends]
    any(k -> occursin(filter, k), keys) || return
    copies = clamp(cld(2 * L2_BYTES, round(Int, bytes)), 1, OP_REPS)
    argtypes = map(typeof, make_args())                 # one set, only for its types
    args = OpArgs(make_args, copies)
    for be in backends
        key = [path..., backend_name(be)]
        # Skip backends without the op, and fallbacks that would only repeat another backend.
        occursin(filter, join(key, "/")) && hasmethod(op, Tuple{typeof(be), argtypes...}) &&
            TileInfer.has_kernel(be, op) || continue
        setleaf!(suite, key, @benchmarkable run_calls($op, $be, calls!($args)) evals = 1)
        info[join(key, "/")] = Workload(OP_REPS, bytes, flops, 0, args)
    end
end

# INT8 pools (TileInfer.Int8Pages) of random values, as the attention ops read them.
function int8_pools(D, P, nkv, pages)
    k, v = TileInfer.int8_kv_pools(D, P, nkv, pages)
    copyto!(k.layer_q, rand(Int8, size(k.layer_q)))
    k.layer_scale .= 0.01f0
    return k, v
end

# `len` random tokens into `seq`, prefilled in chunks: a 32k-token context in one step would
# need gigabytes of activations.
function fill_context!(m, cache, seq, len; chunk = 4096)
    for part in Iterators.partition(1:len, chunk)
        TileInfer.forward!(m, cache, seq, rand(0:m.config.vocab_size-1, length(part)))
    end
    return seq
end

"""
    build_suite(model, label; filter = r"", sampling = SamplingParams(), kv_quantize = nothing,
                context = CONTEXT, backends = DEFAULT_BACKENDS) -> (suite, info)

Benchmarks for `model` (a DecoderLM; its backend is ignored), keyed
"<label>/ops/<op>/<size>/<backend>" and "<label>/model/<benchmark>/<config>", restricted to
keys matching `filter` and to `backends` (the cuTile one by default; `BACKENDS` adds the
reference ops, and the model benchmarks of every [`ModelConfig`](@ref) of `MODEL_CONFIGS`).
`info` maps keys to their [`Workload`](@ref). `sampling` is the model's recommended
sampling (see `TileInfer.generation_defaults`), for the sampling op. `kv_quantize = :int8`
uses INT8 KV caches and pools. The model benchmarks are a `PREFILL`-token prefill, `GEN`
greedy tokens decoded after `context` cached positions, and one eager decode step there
(to profile); decode attention ops run after `context - 1`.
"""
function build_suite(model::DecoderLM{T}, label::AbstractString; filter::Regex = r"",
                     sampling::SamplingParams = SamplingParams(), kv_quantize::Union{Nothing,Symbol} = nothing,
                     context::Integer = CONTEXT, backends = DEFAULT_BACKENDS) where {T}
    cfg, w = model.config, model.weights
    H, I, D, nq, nkv = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim, cfg.num_heads, cfg.num_kv_heads
    eps, B = cfg.rms_norm_eps, sizeof(T)
    suite, info = BenchmarkGroup(), Dict{String,Workload}()
    rnd(dims...) = CUDA.rand(T, dims...)
    # Weights like the model's: Int8 (1 byte each, plus a scale per column) when it is quantized.
    int8 = TileInfer.quantization(w) === :int8
    weight(in, out) = int8 ? TileInfer.quantize_int8(rnd(in, out)) : rnd(in, out)
    WB = int8 ? 1 : B

    for (size_label, n) in (("decode", 1), ("prefill$SHORT", SHORT), ("prefill$PREFILL", PREFILL))
        ops(name) = [label, "ops", name, size_label]
        ids() = CuArray(rand(Int32(0):Int32(cfg.vocab_size - 1), n))
        # Reads one table column per token and writes it: the table is shared, not copied.
        add_op!(suite, info, filter, backends, ops("embed"), TileInfer.embed, () -> (w.embed, ids()); bytes = 2.0 * H * n * B)
        add_op!(suite, info, filter, backends, ops("rmsnorm"), TileInfer.rmsnorm, () -> (rnd(H, n), rnd(H), eps);
                bytes = (2.0 * H * n + H) * B)
        for (name, (in, out)) in (("qkv", size(first(w.layers).wqkv)), ("o", size(first(w.layers).wo)),
                                  ("gate_up", size(first(w.layers).w_gate_up)),
                                  ("down", size(first(w.layers).w_down)))
            # Decode runs gate_up only fused (linear_swiglu / linear_xielu below); prefill's
            # fused op is this GEMM, then the activation.
            name == "gate_up" && n == 1 && continue
            add_op!(suite, info, filter, backends, ops("linear_$name"), TileInfer.linear, () -> (weight(in, out), rnd(in, n));
                    bytes = in * out * WB + (in + out) * n * B, flops = 2.0 * in * out * n)
        end
        # The lm_head as a step runs it: the last token only, into Float32 logits. The weight
        # is the real one (311 MB for 0.6B, 778 MB for 4B) rather than copies.
        # (`let`: argument closures run lazily, after later assignments in this loop body.)
        n == 1 && let (in, out) = size(w.lm_head)
            add_op!(suite, info, filter, backends, ops("linear_lm_head"), TileInfer.linear!,
                    () -> (CuMatrix{Float32}(undef, out, 1), w.lm_head, rnd(in, 1));
                    bytes = in * out * WB + in * B + 4.0 * out, flops = 2.0 * in * out)
        end
        layer = first(w.layers)
        if layer.xielu === nothing
            add_op!(suite, info, filter, backends, ops("silu_mul"), TileInfer.silu_mul, () -> (rnd(2I, n),); bytes = 3.0 * I * n * B)
        else
            add_op!(suite, info, filter, backends, ops("xielu"), TileInfer.xielu, () -> (rnd(I, n), layer.xielu); bytes = 2.0 * I * n * B)
        end

        # One layer's page pools, holding `len` positions of a single sequence.
        start = n == 1 ? context - 1 : 0
        len = start + n
        P = TileInfer.DEFAULT_PAGE_SIZE
        pages = cld(len, P)
        pools(pages = pages) = kv_quantize === nothing ? (rnd(D, P, nkv, pages), rnd(D, P, nkv, pages)) :
                                                         int8_pools(D, P, nkv, pages)
        # Ops that only write the step's n slots get pools of n positions (slots 1:n): copies
        # of a long context's pools would not fit next to the model.
        write_pools() = pools(cld(n, P))
        kv_bytes(positions) = 2.0 * positions * nkv * (kv_quantize === nothing ? D * B : D + 4)
        # Causal prefill visits about half of the (len × n) score matrix.
        visited = n == 1 ? len : n * (n + 1) / 2
        add_op!(suite, info, filter, backends, ops("attention"), TileInfer.attention,
                () -> (rnd(D, nq, n), pools()..., CuArray(Int32.(1:pages)), CuArray(Int32.(start:len-1)), 1 / sqrt(D));
                bytes = kv_bytes(len) + 2.0 * D * nq * n * B, flops = 4.0 * D * nq * visited)

        # The fused ops of a step, in place as run_step! calls them (decode: 10 kernels per
        # layer, of which 6 are these). Their bytes are the unfused minimum, so % peak shows
        # what the fusion achieves.
        add_op!(suite, info, filter, backends, ops("add_rmsnorm"), TileInfer.add_rmsnorm!,
                () -> (rnd(H, n), rnd(H, n), rnd(H, n), rnd(H), eps);
                bytes = (4.0 * H * n + H) * B)                   # read x, delta; write x, h
        let (in, out) = size(layer.w_gate_up)
            if layer.xielu === nothing
                add_op!(suite, info, filter, backends, ops("linear_swiglu"), TileInfer.linear_swiglu!,
                        () -> (rnd(out ÷ 2, n), rnd(out, n), weight(in, out), rnd(in, n));
                        bytes = in * out * WB + (in + out ÷ 2) * n * B, flops = 2.0 * in * out * n)
            else
                add_op!(suite, info, filter, backends, ops("linear_xielu"), TileInfer.linear_xielu!,
                        () -> (rnd(out, n), weight(in, out), rnd(in, n), layer.xielu);
                        bytes = in * out * WB + (in + out) * n * B, flops = 2.0 * in * out * n)
            end
        end
        # q/k norm + rope from the fused qkv, keys and values into the pools.
        add_op!(suite, info, filter, backends, ops("attention_inputs"), TileInfer.attention_inputs!,
                () -> (rnd(D, nq, n), write_pools()..., rnd((nq + 2nkv) * D, n), layer.q_norm, layer.k_norm, model.rope,
                       CuArray(Int32.(start:len-1)), CuArray(Int32.(1:n)), eps);
                bytes = (2.0 * nq + 4.0 * nkv + 2.0) * D * n * B)  # qkv in, q and k/v out, rope table
        # Device sampling with the model's recommended settings (top-1 if it is greedy).
        n == 1 && add_op!(suite, info, filter, backends, ops("sample"), TileInfer.sample_token!,
                          () -> (CuVector{Int32}(undef, 1), CuVector{Int32}(undef, 1),
                                 CUDA.rand(Float32, cfg.vocab_size, 1),
                                 TileInfer.reset_sampler!(TileInfer.TokenSampler(cfg.vocab_size),
                                                          TileInfer.on_device(sampling) ? sampling : SamplingParams(),
                                                          UInt32(1)));
                          bytes = 4.0 * cfg.vocab_size)
    end

    weights = Float64(TileInfer.weight_bytes(w))
    params = Float64(TileInfer.weight_count(w))
    configs = [c for c in MODEL_CONFIGS if c.name == "tile" || RefBackend() in backends]
    keys_of(c) = ("$label/model/prefill$PREFILL/$(c.name)", "$label/model/decode$(GEN)_ctx$context/$(c.name)")
    step_key = "$label/model/decode_step_ctx$context/tile"
    any(c -> any(k -> occursin(filter, k), keys_of(c)), configs) || occursin(filter, step_key) ||
        return suite, info
    # One cache for every configuration, holding the decode context and a prompt.
    tile = DecoderLM(cfg, w, model.rope, TileBackend())
    cache = PagedKVCache(tile; max_tokens = model_len(context), quantize = kv_quantize)
    ctx = fill_context!(tile, cache, TileInfer.new_sequence(cache), context)
    prompt = rand(Xoshiro(1), 0:cfg.vocab_size-1, PREFILL)
    kv = Float64((context + GEN ÷ 2) * TileInfer.bytes_per_token(cfg, something(kv_quantize, T)))

    for c in configs
        int8 && c.backend isa Variant && c.backend.linear == :cublas && continue   # cuBLAS has no Int8 GEMM here
        prefill_key, decode_key = keys_of(c)
        any(k -> occursin(filter, k), (prefill_key, decode_key)) || c.name == "tile" && occursin(filter, step_key) ||
            continue
        m = DecoderLM(cfg, w, model.rope, c.backend)
        # The reference attention allocates gigabytes per prefill: from an empty memory pool,
        # or what an earlier benchmark left in it slows it down by up to 1.7×. (Emptying the
        # pool costs the others their step buffers: 0.6B prefill 16 → 50 ms.)
        reference = c.backend isa RefBackend || c.backend isa Variant && c.backend.attention == :ref
        clean = reference ? () -> (GC.gc(true); CUDA.reclaim()) : () -> nothing
        runner = TileInfer.ModelRunner(m, cache; c.graphs)
        if c.prefill && occursin(filter, prefill_key)
            setleaf!(suite, split(prefill_key, "/"), @benchmarkable begin
                seq = TileInfer.new_sequence($cache)
                TileInfer.forward!($runner, seq, $prompt)
                CUDA.synchronize()
                TileInfer.release!($cache, seq)
            end setup = ($clean)() evals = 1)
            info[prefill_key] = Workload(1, weights, 2 * params * PREFILL, PREFILL)
        end
        # GEN greedy tokens after the context, rewound after each sample.
        if occursin(filter, decode_key)
            gen = () -> TileInfer.generate!(runner, ctx, [42]; max_new_tokens = GEN, stop_ids = Int[],
                                            sampling = SamplingParams(), c.pipeline)
            gen(); CUDA.synchronize(); TileInfer.truncate!(cache, ctx, context)    # compile and capture
            setleaf!(suite, split(decode_key, "/"), @benchmarkable begin
                $gen()
                CUDA.synchronize()
            end setup = ($clean)() teardown = TileInfer.truncate!($cache, $ctx, $context) evals = 1)
            info[decode_key] = Workload(1, GEN * (weights + kv), 2 * params * GEN, GEN)
        end
    end
    # One eager decode step, to profile every kernel of a step (bench/profile.jl).
    if occursin(filter, step_key)
        setleaf!(suite, split(step_key, "/"), @benchmarkable begin
            TileInfer.forward!($tile, $cache, $ctx, [42])
            CUDA.synchronize()
        end teardown = TileInfer.truncate!($cache, $ctx, $context) evals = 1)
        info[step_key] = Workload(1, weights + kv, 2 * params, 1)
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
