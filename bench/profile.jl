# Runs benchmarks from bench/suite.jl once each under Nsight Compute, for kernel-level
# metrics (memory throughput, occupancy, stall reasons) rather than timings.
#
#   ncu --profile-from-start off [ncu options] \
#       julia --project=bench bench/profile.jl [--model REPO,...] FILTER
#
# FILTER is a regex over the keys printed by bench/run.jl, e.g.
# "Qwen3-0.6B/ops/rmsnorm/decode/tile". Models default to bench/run.jl's (MODELS); a
# filter naming one model still loads the others, so pass --model to skip them.
# Each matching benchmark is warmed up (so compilation and first-launch work stay outside
# the profile), then one sample runs between cuProfilerStart/Stop. An op sample is
# OP_REPS = 20 calls of the same kernel, so pass `-c 1` to profile one of them; a model
# step is every kernel of a prefill or decode step (~300 for decode, a few minutes with
# `--set full`). Useful ncu options:
#
#   -o bench/results/NAME      write NAME.ncu-rep, to open in ncu-ui
#   --set full                 all sections (default: a basic set)
#   -c N                       profile only the first N kernel launches
#   -k regex:NAME              only kernels whose name matches
#
# ncu locks clocks to base frequency while profiling: its durations are slower than
# bench/run.jl's and are for comparing kernels with each other, not with the benchmarks.

include("suite.jl")

function main(args)
    repos = collect(MODELS)
    if length(args) == 3 && args[1] == "--model"
        repos = String.(split(args[2], ","))
    elseif length(args) != 1
        error("usage: see the header of bench/profile.jl")
    end
    found = false
    for repo in repos
        model = TileInfer.load_model(repo; max_len = CONTEXT + PREFILL)
        sampling, _ = TileInfer.generation_defaults(TileInfer.model_dir(repo), model.config)
        suite, info = build_suite(model, model_label(repo); filter = Regex(args[end]), sampling)
        benchmarks = sort!(collect(leaves(suite)); by = b -> join(first(b), "/"))
        found |= !isempty(benchmarks)

        for (path, b) in benchmarks
            key = join(path, "/")
            run(b; samples = 1, seconds = 1)                # compile and warm up
            println("profiling ", key)
            CUDA.@profile external = true run(b; samples = 1, seconds = 1, warmup = false)
            free_args!(info[key])
        end
        model = suite = benchmarks = nothing
        GC.gc(true); CUDA.reclaim()
    end
    found || error("no benchmark matches the filter")
end

main(ARGS)
