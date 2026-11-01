module TileInfer

using CUDA
using NVTX: NVTX
using cuTile: cuTile
import cuTile as ct

include("ops/interface.jl")
include("ops/reference.jl")
include("ops/quant.jl")
include("ops/tile/norm.jl")
include("ops/tile/elementwise.jl")
include("ops/tile/linear.jl")
include("ops/tile/attention.jl")
include("ops/tile/sampling.jl")
include("config.jl")
include("model/weights.jl")
include("model/rope.jl")
include("ops/tile/fused.jl")
include("kv/block_manager.jl")
include("kv/paged_cache.jl")
include("hub/store.jl")
include("hub/loader.jl")
include("tokenizer/bpe.jl")
include("tokenizer/chat_template.jl")
include("model/decoder.jl")
include("engine/sampler.jl")
include("engine/runner.jl")
include("engine/speculative.jl")
include("engine/dflash.jl")
include("engine/generate.jl")
include("engine/chat.jl")
include("cli.jl")

function (@main)(args)
    return cli(args)
end

end
