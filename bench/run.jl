# Op and model benchmarks with BenchmarkTools.
#
#   julia --project=bench bench/run.jl [options]
#
#   --filter REGEX     only benchmarks whose key matches, e.g. "ops/linear", "decode", "/tile"
#   --save NAME        store results in bench/results/NAME.json (with git commit and GPU info)
#   --compare NAME     compare against bench/results/NAME.json (minimum times; changes
#                      within ±10 % are reported as invariant)
#   --seconds S        time budget per benchmark (default 1)
#   --model REPO       model whose shapes and weights are used (default Qwen/Qwen3-0.6B)
#
# Times are per call. GB/s and % roofline use the minimum memory traffic of the call
# against the RTX 4080 SUPER's ~736 GB/s; TFLOP/s is reported for calls doing real math.

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
    unknown = setdiff(keys(opts), ("filter", "save", "compare", "seconds", "model"))
    isempty(unknown) || error("unknown options: $(join(unknown, ", "))")
    return opts
end

result_path(name) = joinpath(RESULTS, "$name.json")

function environment()
    commit = try readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`) catch; "unknown" end
    dirty = try !isempty(readchomp(`git -C $(@__DIR__) status --porcelain`)) catch; false end
    return Dict("commit" => commit * (dirty ? "+dirty" : ""), "gpu" => CUDA.name(CUDA.device()),
                "date" => string(round(Int, time())), "julia" => string(VERSION))
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
    return BenchmarkTools.load(path)[1], env
end

format_time(t) = t < 1e3 ? @sprintf("%.0f ns", t) : t < 1e6 ? @sprintf("%.1f µs", t / 1e3) :
                 @sprintf("%.2f ms", t / 1e6)

function report(results, info; baseline = nothing)
    rows = sort!(collect(leaves(results)); by = r -> join(first(r), "/"))
    @printf("%-40s %10s %10s %9s %6s %8s %8s", "benchmark", "min", "median", "GB/s", "%peak", "TFLOP/s", "tok/s")
    baseline === nothing || @printf(" %9s  %s", "speedup", "vs baseline")
    println()
    for (path, trial) in rows
        key = join(path, "/")
        w = info[key]
        tmin, tmed = minimum(trial).time / w.reps, median(trial).time / w.reps   # ns per call
        bandwidth = w.bytes / (tmin * 1e-9)
        @printf("%-40s %10s %10s %9.1f %5.1f%% %8s %8s", key, format_time(tmin), format_time(tmed),
                bandwidth / 1e9, 100 * bandwidth / PEAK_BANDWIDTH,
                w.flops >= 1e9 ? @sprintf("%.2f", w.flops / (tmin * 1e-9) / 1e12) : "",
                w.tokens > 0 ? @sprintf("%.1f", w.tokens / (tmin * 1e-9)) : "")
        if baseline !== nothing
            old = try baseline[path] catch; nothing end
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
    model = TileInfer.load_model(get(opts, "model", "Qwen/Qwen3-0.6B"); max_len = CONTEXT + PREFILL)
    suite, info = build_suite(model; filter = Regex(get(opts, "filter", "")))
    isempty(leaves(suite)) && error("no benchmark matches the filter")
    seconds = parse(Float64, get(opts, "seconds", "1"))
    foreach(((_, b),) -> (b.params.seconds = seconds; b.params.samples = 10_000), leaves(suite))

    env = environment()
    println("Running $(length(leaves(suite))) benchmarks on $(env["gpu"]) at $(env["commit"])...")
    warmup(suite; verbose = false)
    results = run(suite; verbose = false)

    baseline = nothing
    if haskey(opts, "compare")
        baseline, base_env = load_results(opts["compare"])
        println("Baseline \"$(opts["compare"])\": commit $(get(base_env, "commit", "?")) on $(get(base_env, "gpu", "?"))")
    end
    report(results, info; baseline)
    if haskey(opts, "save")
        save_results(opts["save"], results, env)
        println("Saved to $(result_path(opts["save"]))")
    end
end

main(ARGS)
