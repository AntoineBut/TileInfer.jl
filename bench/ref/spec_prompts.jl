# Export the chat prompts of bench/prompts.jl as token ids with the model's stop ids, so that
# vllm_spec.py and tileinfer_spec.jl feed both engines the same tokens: at most `len` tokens
# each (default 1024, the reference point; 8192 for the long-context one).
#
#   julia --project=bench bench/ref/spec_prompts.jl [model] [len] > bench/ref/spec_prompts.json
#   julia --project=bench bench/ref/spec_prompts.jl Qwen/Qwen3-4B 8192 > bench/ref/spec_prompts_8k.json

using TileInfer, JSON
using TileInfer: generation_defaults, load_config, load_tokenizer, model_dir
include(joinpath(@__DIR__, "..", "prompts.jl"))

dir = model_dir(get(ARGS, 1, "Qwen/Qwen3-4B"))
cfg = load_config(dir)
_, stop_ids = generation_defaults(dir, cfg)
prompts = [Dict("name" => name, "ids" => ids) for (name, ids) in chat_prompts(load_tokenizer(dir), cfg.model_type;
                                                                       len = parse(Int, get(ARGS, 2, string(PROMPT_TOKENS))))]
JSON.print(stdout, Dict("stop_ids" => collect(stop_ids), "prompts" => prompts))
println()
