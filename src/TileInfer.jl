module TileInfer

using CUDA
using cuTile: cuTile
import cuTile as ct

include("ops/interface.jl")
include("ops/reference.jl")
include("config.jl")
include("model/weights.jl")
include("model/rope.jl")
include("kv/block_manager.jl")
include("kv/paged_cache.jl")
include("hub/store.jl")
include("hub/loader.jl")
include("tokenizer/bpe.jl")
include("tokenizer/chat_template.jl")
include("model/decoder.jl")
include("engine/sampler.jl")
include("engine/generate.jl")
include("engine/chat.jl")
include("cli.jl")

function (@main)(args)
    return cli(args)
end

end
