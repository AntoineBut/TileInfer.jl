# Usage:
#   julia --project=test test/runtests.jl                  # everything, in two processes
#   julia --project=test test/runtests.jl kernels graphs   # files whose path contains an arg
#   julia --project=test test/runtests.jl smoke            # a whole directory
#   julia --project=test test/runtests.jl --serial [...]   # in this process only
#
# The files form two groups, which run in parallel processes when both are selected: kernel
# tests (small arrays; their time is kernel compilation) and model tests (they load
# checkpoints up to Qwen3-4B, so they run one after the other, never two big models at once).
# Each group's output is printed when it finishes.

using Test

# In run order within each group; slow files go last.
const KERNEL_TESTS = [
    "smoke/environment.jl",
    "smoke/kernels.jl",
    "smoke/graphs.jl",
    "hub/store.jl",
    "hub/config.jl",
    "hub/loader.jl",
    "tokenizer/bpe.jl",
    "tokenizer/chat_template.jl",
    "kv/block_manager.jl",
    "engine/sampler.jl",
    "ops/reference.jl",
    "ops/tile.jl",
]
const MODEL_TESTS = [
    "engine/runner.jl",
    "engine/chat.jl",
    "cli.jl",
    "model/forward.jl",
    "engine/speculative.jl",
    "engine/dflash.jl",     # Qwen3-4B + DFlash drafter, when pulled
]
const TESTS = [KERNEL_TESTS; MODEL_TESTS]

const WORKER = "--worker" in ARGS                   # run exactly the files given
const SERIAL = "--serial" in ARGS
const FILTERS = filter(a -> !(a in ("--worker", "--serial")), ARGS)
const SELECTED = WORKER || isempty(FILTERS) ? (WORKER ? FILTERS : TESTS) :
    filter(t -> any(occursin(arg, t) for arg in FILTERS), TESTS)

isempty(SELECTED) && error("no test file matches $(FILTERS); available: $(TESTS)")

const GROUPS = filter(g -> !isempty(last(g)),
                      ["kernels" => intersect(KERNEL_TESTS, SELECTED), "models" => intersect(MODEL_TESTS, SELECTED)])

# Run each group in a process of its own, with this process's options (coverage included).
function run_groups(groups)
    julia = `$(Base.julia_cmd()) --color=$(get(stdout, :color, false) ? "yes" : "no") --project=$(Base.active_project())`
    start = time()
    tasks = map(groups) do (name, files)
        out = IOBuffer()
        p = run(pipeline(ignorestatus(`$julia $(@__FILE__) --worker $files`); stdout = out, stderr = out); wait = false)
        name => @async (wait(p); (success(p), String(take!(out)), time() - start))
    end
    passed = true
    for (name, task) in tasks
        ok, output, seconds = fetch(task)
        println("── $name tests ($(round(Int, seconds)) s) ", "─"^40)
        print(output)
        ok || println("── $name tests FAILED")
        passed &= ok
    end
    println("Both groups ", passed ? "passed" : "did not pass", " in $(round(Int, time() - start)) s")
    passed || exit(1)
end

if WORKER || SERIAL || length(GROUPS) == 1
    using CUDA
    using BFloat16s
    using cuTile: cuTile
    import cuTile as ct
    using TileInfer

    include("helpers.jl")

    @testset "TileInfer" begin
        for file in SELECTED
            @testset "$(splitext(file)[1])" begin
                include(file)
            end
        end
    end
else
    run_groups(GROUPS)
end
