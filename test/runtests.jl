# Usage:
#   julia --project=test test/runtests.jl                  # everything
#   julia --project=test test/runtests.jl kernels graphs   # files whose path contains an arg
#   julia --project=test test/runtests.jl smoke            # a whole directory

using Test
using CUDA
using BFloat16s
using cuTile: cuTile
import cuTile as ct
using TileInfer

include("helpers.jl")

# In run order; slow files go last.
const TESTS = [
    "smoke/environment.jl",
    "smoke/kernels.jl",
    "smoke/graphs.jl",
    "hub/store.jl",
    "hub/config.jl",
    "hub/loader.jl",
    "tokenizer/bpe.jl",
    "tokenizer/chat_template.jl",
    "kv/block_manager.jl",
    "ops/reference.jl",
    "ops/tile.jl",
    "model/forward.jl",
    "engine/sampler.jl",
    "engine/runner.jl",
    "engine/chat.jl",
    "cli.jl",
    "smoke/examples.jl",    # ~40 s: compiles all cuTile examples
]

const SELECTED = isempty(ARGS) ? TESTS :
    filter(t -> any(occursin(arg, t) for arg in ARGS), TESTS)

isempty(SELECTED) && error("no test file matches $(ARGS); available: $(TESTS)")

@testset "TileInfer" begin
    for file in SELECTED
        @testset "$(splitext(file)[1])" begin
            include(file)
        end
    end
end
