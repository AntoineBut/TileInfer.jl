# Export bench/speculative.jl's chat prompts as token ids (chat template applied, thinking
# off) with the model's stop ids, so that vllm_spec.py and tileinfer_spec.jl feed both
# engines the same tokens. Qwen3-0.6B and Qwen3-4B share the tokenizer.
#
#   julia --project=bench bench/ref/spec_prompts.jl [model] > bench/ref/spec_prompts.json

using TileInfer, JSON
using TileInfer: ChatMessage, chat_prompt, encode, generation_defaults, load_config, load_tokenizer, model_dir

# Only the `const` definitions of bench/speculative.jl (CODE, LONG_DOCUMENT, PROMPTS), not its main.
const SPEC = joinpath(@__DIR__, "..", "speculative.jl")
const SRC = replace(read(SPEC, String), "@__DIR__" => repr(dirname(abspath(SPEC))))
for ex in Meta.parseall(SRC; filename = SPEC).args
    ex isa Expr && ex.head === :const && Core.eval(Main, ex)
end

dir = model_dir(get(ARGS, 1, "Qwen/Qwen3-4B"))
cfg = load_config(dir)
tok = load_tokenizer(dir)
_, stop_ids = generation_defaults(dir, cfg)
prompts = [Dict("name" => name,
                "ids" => encode(tok, chat_prompt(cfg.model_type, [ChatMessage("user", text)]; enable_thinking = false);
                                add_special_tokens = false))
           for (name, text) in PROMPTS]
JSON.print(stdout, Dict("stop_ids" => collect(stop_ids), "prompts" => prompts))
println()
