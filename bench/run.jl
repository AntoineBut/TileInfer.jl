# Op and model benchmarks with BenchmarkTools.
#
#   julia --project=bench bench/run.jl [options]
#
#   --filter REGEX     only benchmarks whose key matches, e.g. "ops/linear", "4B/.*decode", "/tile"
#   --save NAME        store results in bench/results/NAME.json (with git commit and GPU info)
#   --compare NAME     compare against bench/results/NAME.json (minimum times; changes
#                      within ±10 % are reported as invariant)
#   --seconds S        time budget per benchmark (default 1)
#   --model REPO,...   models whose shapes and weights are used (default: Qwen/Qwen3-0.6B and
#                      Qwen/Qwen3-4B, one after the other). Keys start with the model name.
#   --quantize int8    load the models with Int8 linear weights (keys start "<model>-int8"; the
#                      op benchmarks of weight-reading kernels use Int8 weights)
#
# Times are per call. GB/s and % roofline use the minimum memory traffic of the call
# against the RTX 4080 SUPER's ~736 GB/s; TFLOP/s is reported for calls doing real math.
# "x ref" is the speedup over the reference backend's version of the same benchmark.

include("suite.jl")
using JSON: JSON
using Printf: @printf, @sprintf

const PEAK_BANDWIDTH = 736e9
const RESULTS = joinpath(@__DIR__, "results")
const TIME_TOLERANCE = 0.10      # GPU timings vary by a few percent between runs

function parse_cli(args)
    opts = Dict{String,String}()
    i = 1
    while i <= length(args)
        startswith(args[i], "--") && i < length(args) || error("usage: see the header of bench/run.jl")
        opts[args[i][3:end]] = args[i+1]
        i += 2
    end
    unknown = setdiff(keys(opts), ("filter", "save", "compare", "seconds", "model", "quantize"))
    isempty(unknown) || error("unknown options: $(join(unknown, ", "))")
    return opts
end

result_path(name) = joinpath(RESULTS, "$name.json")

function environment(repos)
    commit = try readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`) catch; "unknown" end
    dirty = try !isempty(readchomp(`git -C $(@__DIR__) status --porcelain`)) catch; false end
    return Dict("commit" => commit * (dirty ? "+dirty" : ""), "gpu" => CUDA.name(CUDA.device()),
                "date" => string(round(Int, time())), "julia" => string(VERSION), "models" => repos)
end

function save_results(name, results, env)
    mkpath(RESULTS)
    BenchmarkTools.save(result_path(name), results)
    write(joinpath(RESULTS, "$name.env.json"), JSON.json(env))
end

function load_results(name)
    path = result_path(name)
    isfile(path) || error("no saved results $path")
    env_path = joinpath(RESULTS, "$name.env.json")
    env = isfile(env_path) ? JSON.parsefile(env_path) : Dict()
    results = BenchmarkTools.load(path)[1]
    # Results saved before keys started with the model were all for Qwen3-0.6B.
    if haskey(results, "ops") || haskey(results, "model")
        old, results = results, BenchmarkGroup()
        results[model_label(first(MODELS))] = old
    end
    return results, env
end

# The baseline trial at `path`, or nothing. (Indexing a BenchmarkGroup with a missing key
# returns an empty group rather than failing.)
function lookup(g::BenchmarkGroup, path)
    for k in path
        haskey(g, k) || return nothing
        g = g[k]
    end
    return g isa BenchmarkTools.Trial ? g : nothing
end

format_time(t) = t < 1e3 ? @sprintf("%.0f ns", t) : t < 1e6 ? @sprintf("%.1f µs", t / 1e3) :
                 @sprintf("%.2f ms", t / 1e6)

function report(results, info; baseline = nothing)
    rows = sort!(collect(leaves(results)); by = r -> join(first(r), "/"))
    @printf("%-56s %10s %10s %9s %6s %8s %8s %7s", "benchmark", "min", "median", "GB/s", "%peak", "TFLOP/s", "tok/s", "x ref")
    baseline === nothing || @printf(" %9s  %s", "speedup", "vs baseline")
    println()
    for (path, trial) in rows
        key = join(path, "/")
        w = info[key]
        tmin, tmed = minimum(trial).time / w.reps, median(trial).time / w.reps   # ns per call
        bandwidth = w.bytes / (tmin * 1e-9)
        @printf("%-56s %10s %10s %9.1f %5.1f%% %8s %8s", key, format_time(tmin), format_time(tmed),
                bandwidth / 1e9, 100 * bandwidth / PEAK_BANDWIDTH,
                w.flops >= 1e9 ? @sprintf("%.2f", w.flops / (tmin * 1e-9) / 1e12) : "",
                w.tokens > 0 ? @sprintf("%.1f", w.tokens / (tmin * 1e-9)) : "")
        # Speedup over the reference backend's version of the same benchmark in this run.
        ref = path[end] == "ref" ? nothing : lookup(results, [path[1:end-1]; "ref"])
        print(ref === nothing ? "        " : @sprintf(" %6.2fx", minimum(ref).time / minimum(trial).time))
        if baseline !== nothing
            old = lookup(baseline, path)
            if old === nothing
                print("         -  (new)")
            else
                j = judge(minimum(trial), minimum(old); time_tolerance = TIME_TOLERANCE)
                @printf(" %8.2fx  %s", minimum(old).time / minimum(trial).time, string(j.time))
            end
        end
        println()
    end
end

function main(args)
    opts = parse_cli(args)
    repos = haskey(opts, "model") ? String.(split(opts["model"], ",")) : collect(MODELS)
    filter = Regex(get(opts, "filter", ""))
    seconds = parse(Float64, get(opts, "seconds", "1"))
    env = environment(repos)

    # One model at a time, so that only one set of weights is on the GPU.
    results, info = BenchmarkGroup(), Dict{String,Workload}()
    for repo in repos
        quantize = haskey(opts, "quantize") ? Symbol(opts["quantize"]) : nothing
        model = TileInfer.load_model(repo; max_len = CONTEXT + PREFILL, quantize)
        sampling, _ = TileInfer.generation_defaults(TileInfer.model_dir(repo), model.config)
        label = model_label(repo) * (quantize === nothing ? "" : "-$quantize")
        suite, model_info = build_suite(model, label; filter, sampling)
        if !isempty(leaves(suite))
            foreach(((_, b),) -> (b.params.seconds = seconds; b.params.samples = 10_000), leaves(suite))
            println("Running $(length(leaves(suite))) $label benchmarks on $(env["gpu"]) at $(env["commit"])...")
            results[label] = run_suite(suite, model_info)[label]
            merge!(info, model_info)
        end
        model = suite = nothing
        GC.gc(true); CUDA.reclaim()
    end
    isempty(leaves(results)) && error("no benchmark matches the filter")

    baseline = nothing
    if haskey(opts, "compare")
        baseline, base_env = load_results(opts["compare"])
        println("Baseline \"$(opts["compare"])\": commit $(get(base_env, "commit", "?")) on $(get(base_env, "gpu", "?"))")
    end
    if haskey(opts, "save")             # before reporting, so a report error loses nothing
        save_results(opts["save"], results, env)
        println("Saved to $(result_path(opts["save"]))")
    end
    report(results, info; baseline)
end

main(ARGS)
