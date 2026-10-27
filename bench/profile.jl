# Runs benchmarks from bench/suite.jl once each under Nsight Compute, for kernel-level
# metrics (memory throughput, occupancy, stall reasons) rather than timings.
#
#   ncu --profile-from-start off [ncu options] \
#       julia --project=bench bench/profile.jl [--model REPO] FILTER
#
# FILTER is a regex over the keys printed by bench/run.jl, e.g. "ops/rmsnorm/decode/tile".
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
    model_repo = "Qwen/Qwen3-0.6B"
    if length(args) == 3 && args[1] == "--model"
        model_repo = args[2]
    elseif length(args) != 1
        error("usage: see the header of bench/profile.jl")
    end
    model = TileInfer.load_model(model_repo; max_len = CONTEXT + PREFILL)
    suite, _ = build_suite(model; filter = Regex(args[end]))
    benchmarks = sort!(collect(leaves(suite)); by = b -> join(first(b), "/"))
    isempty(benchmarks) && error("no benchmark matches the filter")

    for (_, b) in benchmarks
        run(b; samples = 1, seconds = 1)                    # compile and warm up
    end
    for (path, b) in benchmarks
        println("profiling ", join(path, "/"))
        CUDA.@profile external = true run(b; samples = 1, seconds = 1, warmup = false)
    end
end

main(ARGS)
