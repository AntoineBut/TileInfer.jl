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
trained on. `add_generation_prompt` opens an assistant turn for the model to complete. The
prompt includes any BOS token the template writes, so encode it with
`add_special_tokens = false`, as transformers' `apply_chat_template` does.
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

const APERTUS15_DEFAULT_SYSTEM = "You are Apertus 1.5 Omni, a multimodal assistant developed by the Swiss AI \
    Initiative. Extended from Apertus 1 via continued pretraining, you understand images and audio and respond in text."

# Port of Apertus 1.5's template (chat_template.jinja) for text messages without tools. A
# system message may only come first; without one, the template's default is used. The
# developer turn says whether the model may deliberate (think between <|inner_prefix|> and
# <|inner_suffix|>). Consecutive assistant messages form one turn. Earlier reasoning is kept.
function chat_prompt(::Val{:apertus1p5}, messages::AbstractVector{ChatMessage};
                     add_generation_prompt::Bool = true, enable_thinking::Bool = true)
    io = IOBuffer()
    print(io, "<s><|system_start|>")
    rest = messages
    if !isempty(messages) && first(messages).role == "system"
        print(io, first(messages).content)
        rest = messages[2:end]
    else
        print(io, APERTUS15_DEFAULT_SYSTEM)
    end
    print(io, "<|system_end|><|developer_start|>Deliberation: ", enable_thinking ? "enabled" : "disabled",
          "\nTool Capabilities: disabled<|developer_end|>")
    in_assistant = false
    for m in rest
        if m.role == "user"
            in_assistant && print(io, "<|assistant_end|>")
            in_assistant = false
            print(io, "<|user_start|>", m.content, "<|user_end|>")
        elseif m.role == "assistant"
            in_assistant || print(io, "<|assistant_start|>")
            in_assistant = true
            print(io, m.content)
        else
            error("unsupported chat role \"$(m.role)\" (a system message must come first)")
        end
    end
    in_assistant && print(io, "<|assistant_end|>")
    add_generation_prompt && print(io, "<|assistant_start|>")
    return String(take!(io))
end
