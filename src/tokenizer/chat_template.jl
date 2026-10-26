# Chat prompt formatting. Hugging Face models ship a Jinja template in tokenizer_config.json;
# instead of a Jinja engine, each supported family's template is ported by hand and checked
# against the real template's output (test/fixtures/chat/).

"One turn of a conversation. `role` is \"system\", \"user\" or \"assistant\"."
struct ChatMessage
    role::String
    content::String
end

"""
    chat_prompt(model_type, messages; add_generation_prompt = true, enable_thinking = true)

Render `messages` (a vector of [`ChatMessage`](@ref)) into the prompt string the model was
trained on. `add_generation_prompt` opens an assistant turn for the model to complete.
"""
chat_prompt(model_type::AbstractString, messages; kwargs...) =
    chat_prompt(Val(Symbol(model_type)), messages; kwargs...)

chat_prompt(::Val{T}, messages; kwargs...) where {T} =
    error("no chat template for model type \"$T\" yet")

# Port of Qwen3's template without tool calling. Two details matter:
# - assistant turns before the last user message drop their <think>…</think> reasoning,
#   so earlier reasoning does not use up context;
# - enable_thinking = false pre-fills an empty think block, which makes the model answer
#   directly.
function chat_prompt(::Val{:qwen3}, messages::AbstractVector{ChatMessage};
                     add_generation_prompt::Bool = true, enable_thinking::Bool = true)
    io = IOBuffer()
    last_query = something(findlast(m -> m.role == "user", messages), length(messages))
    for (i, m) in enumerate(messages)
        if m.role == "user" || m.role == "system"
            print(io, "<|im_start|>", m.role, "\n", m.content, "<|im_end|>\n")
        elseif m.role == "assistant"
            content, reasoning = m.content, ""
            if occursin("</think>", content)
                parts = split(content, "</think>")
                reasoning = lstrip(last(split(rstrip(first(parts), '\n'), "<think>")), '\n')
                content = lstrip(last(parts), '\n')
            end
            print(io, "<|im_start|>assistant\n")
            if i > last_query && (i == length(messages) || !isempty(reasoning))
                print(io, "<think>\n", strip(reasoning, '\n'), "\n</think>\n\n", lstrip(content, '\n'))
            else
                print(io, content)
            end
            print(io, "<|im_end|>\n")
        else
            error("unsupported chat role \"$(m.role)\"")
        end
    end
    if add_generation_prompt
        print(io, "<|im_start|>assistant\n")
        enable_thinking || print(io, "<think>\n\n</think>\n\n")
    end
    return String(take!(io))
end
