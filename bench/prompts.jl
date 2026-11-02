# The chat prompts of the speculative-decoding benchmarks (bench/speculative.jl, and both
# engines of bench/ref/ through spec_prompts*.json): two tasks on Julia source code (a frozen
# copy of src/engine/runner.jl followed by the generation, speculative-decoding, decoder
# and KV cache sources, ~19k tokens, in bench/prompts/engine.jl, long enough for longer reference points), cut to fit the prompt
# length of a reference point of docs/src/performance.md. Acceptance depends on the text:
# the edit copies its input, the explanation is open text.

using TileInfer: ChatMessage, chat_prompt, encode

"The document the tasks work on."
const DOCUMENT = read(joinpath(@__DIR__, "prompts", "engine.jl"), String)

"Prompt tokens of the reference point, chat template included."
const PROMPT_TOKENS = 1024

"""
Prompt tokens of the long-context reference point. With DFlash, Qwen3-4B in BF16 takes
prompts of up to ~13k tokens: the prompt is prefilled in one step, and the drafter then
reads its states in one step of its own, ~400 KB per prompt token in all next to 167 KB
of KV caches.
"""
const LONG_PROMPT_TOKENS = 8192

"The tasks, each followed by the document's first lines."
const TASKS = [
    "edit" => "Rename `ModelRunner` to `StepRunner` throughout this Julia code. Reply with the full code only.\n\n",
    "explain" => "Explain how this Julia code decides when to capture a CUDA graph and when to replay one.\n\n",
]

"""
    fit_document(tokens, task, len) -> text

`task` followed by the most lines of `DOCUMENT` whose prompt `tokens(text)` has at most
`len` tokens.
"""
function fit_document(tokens, task, len)
    lines = split(DOCUMENT, '\n')
    text(k) = task * join(lines[1:k], '\n')
    lo, hi = 0, length(lines)                # the answer is in lo:hi
    while lo < hi
        mid = cld(lo + hi, 2)
        length(tokens(text(mid))) <= len ? (lo = mid) : (hi = mid - 1)
    end
    return text(lo)
end

"""
    chat_prompts(tok, model_type; len = PROMPT_TOKENS) -> [name => ids]

The 0-based token ids of each task's chat prompt (thinking off), of at most `len` tokens.
"""
function chat_prompts(tok, model_type; len::Integer = PROMPT_TOKENS)
    tokens(text) = encode(tok, chat_prompt(model_type, [ChatMessage("user", text)]; enable_thinking = false);
                          add_special_tokens = false)
    return [name => tokens(fit_document(tokens, task, len)) for (name, task) in TASKS]
end
