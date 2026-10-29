# Where the GPU time of whole steps goes, kernel by kernel. The op benchmarks of bench/run.jl
# launch each op eagerly, so a small kernel there measures its ~5.5 µs launch rather than its
# GPU time. Here CUDA's profiler traces real steps (decode replayed from CUDA graphs, as in
# generation) and sums the GPU time of each kernel per step.
#
#   julia --project=bench bench/breakdown.jl [--model REPO,...] [--filter REGEX]
#
# Steps, for each model on TileBackend:
#   decode           GEN greedy tokens after CONTEXT - 1 cached positions, pipelined
#   decode_sampled   the same with the model's recommended sampling
#   prefill64/512    a SHORT- or PREFILL-token prompt
#
# Each step is split into a prologue (embedding, first norm), num_layers repeats of the
# layer's kernel sequence, and an epilogue (lm_head, sampling). A layer kernel is named by
# its role: GEMV/GEMM launches by the projection they compute, in run_step!'s order (qkv, o,
# gate_up, down), other repeated kernels by their occurrence in the layer (#1, #2).
# `busy` is the kernel time per step; `span` runs from a step's first kernel start to its
# last kernel end, so span - busy is time the GPU idled inside the step. `period` (decode)
# runs from one step's start to the next: period - span is the gap between steps.
# Clocks: each case runs for a second, then ~TRACE_SECONDS of it are traced. Starting the
# profiler idles the GPU long enough to drop its clocks (it idles at 210 MHz), so the first
# steps of a trace run up to 3× slow: only the second half of the steps is reported (the
# first half's busy time is printed for comparison). The tracer adds a little to each
# launch: take absolute times from bench/run.jl, and shares from here.

include("suite.jl")
using Printf: @printf, @sprintf

const TRACE_SECONDS = 0.4
const PROJECTIONS = ("qkv", "o", "gate_up", "down")
const LINEAR_KERNELS = ("gemv_kernel", "gemm_kernel")

struct Launch
    name::String            # kernel function name
    grid::NTuple{3,Int}
    start::Float64          # s
    stop::Float64
end

# Kernel launches of a profile, in start order. Names are shortened to the function name.
function launches(r)
    d = r.device
    ks = [Launch(String(first(split(d.name[i], '('))), (Int(d.grid[i].x), Int(d.grid[i].y), Int(d.grid[i].z)),
                 d.start[i], d.stop[i]) for i in eachindex(d.name) if d.grid[i] !== missing]
    return sort!(ks; by = k -> k.start)
end

# Split into steps at each embedding launch; launches before the first one join step 1.
function split_steps(ks::Vector{Launch})
    firsts = findall(k -> k.name == "embed_kernel", ks)
    isempty(firsts) && return [ks]
    bounds = [1; firsts[2:end]; length(ks) + 1]
    return [ks[bounds[i]:bounds[i+1]-1] for i in 1:length(bounds)-1]
end

# (prologue length, layer length) such that the step's launches after the prologue start
# with `layers` repeats of the same kernel sequence, or nothing.
function layer_structure(step::Vector{Launch}, layers::Int)
    key(k) = (k.name, k.grid)
    for p in 0:4, b in 1:40
        p + layers * b <= length(step) || continue      # (`break` would leave both loops)
        all(key(step[p+i]) == key(step[p+mod1(i, b)]) for i in 1:layers*b) && return p, b
    end
    return nothing
end

# Role of each launch of one layer (`block`), as described in the header.
function layer_roles(block::Vector{Launch})
    roles = String[]
    k = 1                                   # next projection
    for (j, l) in enumerate(block)
        if l.name == "gemv_swiglu_kernel"
            push!(roles, "gate_up + SwiGLU")
            k = 4
        elseif l.name in LINEAR_KERNELS && k <= length(PROJECTIONS)
            push!(roles, PROJECTIONS[k])
            k += 1
        else
            n = count(b -> b.name == l.name, block)
            i = count(b -> b.name == l.name, block[1:j])
            push!(roles, n == 1 ? l.name : "$(l.name) #$i")
        end
    end
    return [r == l.name || startswith(r, l.name) ? "layer: $r" : "layer: $r ($(l.name))" for (r, l) in zip(roles, block)]
end

# Rows (role, launches per step, µs per step) in step order, and per-step totals.
function breakdown(steps::Vector{Vector{Launch}}, layers::Int)
    structure = layer_structure(steps[end], layers)
    rows = Dict{String,Tuple{Int,Float64,Int}}()           # role => (launches, seconds, order)
    for step in steps
        roles = if structure === nothing || layer_structure(step, layers) != structure
            [l.name for l in step]
        else
            p, b = structure
            lr = layer_roles(step[p+1:p+b])
            post = [l.name in LINEAR_KERNELS ? "post: lm_head ($(l.name))" : "post: $(l.name)" for l in step[p+layers*b+1:end]]
            [["pre: $(l.name)" for l in step[1:p]]; repeat(lr, layers); post]
        end
        for (i, (role, l)) in enumerate(zip(roles, step))
            n, t, o = get(rows, role, (0, 0.0, i))
            rows[role] = (n + 1, t + (l.stop - l.start), min(o, i))
        end
    end
    ns = length(steps)
    sorted = sort!(collect(rows); by = r -> r[2][3])
    table = [(role, n / ns, t / ns * 1e6) for (role, (n, t, _)) in sorted]
    busy = sum(sum(l.stop - l.start for l in s) for s in steps) / ns * 1e6
    span = sum(s[end].stop - s[1].start for s in steps) / ns * 1e6
    period = ns > 1 ? (steps[end][1].start - steps[1][1].start) / (ns - 1) * 1e6 : NaN
    return table, busy, span, period
end

function report(title, all_steps, layers)
    half = length(all_steps) ÷ 2
    steps = all_steps[half+1:end]
    table, busy, span, period = breakdown(steps, layers)
    warm = sum(sum(l.stop - l.start for l in s) for s in all_steps[1:half]) / half * 1e6
    println("\n$title: $(length(steps)) steps, $(sum(length, steps) ÷ length(steps)) kernels per step",
            @sprintf(" (first %d steps, discarded: busy %.0f µs per step)", half, warm))
    @printf("  busy %.0f µs, span %.0f µs (GPU busy %.1f %% of the span)", busy, span, 100 * busy / span)
    isnan(period) || @printf(", period %.0f µs (busy %.1f %%)", period, 100 * busy / period)
    println()
    @printf("  %-48s %8s %10s %10s %7s\n", "kernel (role)", "#/step", "µs/step", "µs/launch", "share")
    for (role, n, t) in table
        @printf("  %-48s %8.1f %10.1f %10.2f %6.1f%%\n", role, n, t, t / n, 100 * t / busy)
    end
end

function parse_cli(args)
    opts = Dict{String,String}()
    for i in 1:2:length(args)
        i < length(args) && startswith(args[i], "--") || error("usage: see the header of bench/breakdown.jl")
        opts[args[i][3:end]] = args[i+1]
    end
    issubset(keys(opts), ("model", "filter")) || error("unknown options; see the header of bench/breakdown.jl")
    return opts
end

function main(args)
    opts = parse_cli(args)
    repos = haskey(opts, "model") ? String.(split(opts["model"], ",")) : collect(MODELS)
    filter = Regex(get(opts, "filter", ""))
    for repo in repos
        label = model_label(repo)
        cases = ["decode", "decode_sampled", "prefill$SHORT", "prefill$PREFILL"]
        any(c -> occursin(filter, "$label/$c"), cases) || continue
        model = TileInfer.load_model(repo; backend = TileBackend(), max_len = CONTEXT + PREFILL)
        cfg = model.config
        sampling, _ = TileInfer.generation_defaults(TileInfer.model_dir(repo), cfg)
        cache = PagedKVCache(model; max_tokens = CONTEXT + PREFILL)
        runner = TileInfer.ModelRunner(model, cache)
        ctx = TileInfer.new_sequence(cache)
        TileInfer.forward!(model, cache, ctx, rand(0:cfg.vocab_size-1, CONTEXT - 1))
        function decode(sp)
            TileInfer.generate!(runner, ctx, [42]; max_new_tokens = GEN, stop_ids = Int[], sampling = sp,
                                rng = Xoshiro(1))
            CUDA.synchronize()
            TileInfer.truncate!(cache, ctx, CONTEXT - 1)
        end
        function prefill(p)
            seq = TileInfer.new_sequence(cache)
            TileInfer.forward!(model, cache, seq, p)
            CUDA.synchronize()
            TileInfer.release!(cache, seq)
        end
        prompt = rand(0:cfg.vocab_size-1, PREFILL)
        runs = ["decode" => () -> decode(SamplingParams()), "decode_sampled" => () -> decode(sampling),
                "prefill$SHORT" => () -> prefill(prompt[1:SHORT]), "prefill$PREFILL" => () -> prefill(prompt)]
        for (case, f) in runs
            occursin(filter, "$label/$case") || continue
            case == "decode_sampled" && sampling.temperature == 0 && continue
            # Compile and capture graphs, bring the clocks up, then trace ~TRACE_SECONDS.
            f()
            calls, t0 = 0, time()
            while time() - t0 < 1
                f()
                calls += 1
            end
            reps = max(2, ceil(Int, TRACE_SECONDS * calls / (time() - t0)))
            r = CUDA.@profile trace = true for _ in 1:reps
                f()
            end
            report("$label/$case", split_steps(launches(r)), cfg.num_layers)
        end
        model = cache = runner = nothing
        GC.gc(true); CUDA.reclaim()
    end
end

main(ARGS)
