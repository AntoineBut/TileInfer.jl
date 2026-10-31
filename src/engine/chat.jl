# Multi-turn chat on one long-lived sequence. Each turn re-renders the whole conversation
# with the chat template, keeps the longest prefix of tokens already in the KV cache, and
# runs the model only on the rest. Usually that prefix is everything up to the new user
# message; it is shorter when the template rewrites history (Qwen3 drops the reasoning of
# earlier assistant turns), and the cache is then rolled back to where they diverge.

"""
    ChatSession(model, tokenizer; sampling, stop_ids, enable_thinking = true,
                max_tokens = max_length(model), rng = Random.default_rng(), speculative = nothing,
                kv_quantize = nothing)

A conversation with `model`, with its own paged KV cache for up to `max_tokens` positions,
quantized as `kv_quantize` (see [`PagedKVCache`](@ref)).
Send messages with [`reply!`](@ref). `speculative` is a proposer for speculative decoding
(see [`generate!`](@ref)).
"""
mutable struct ChatSession{M<:DecoderLM,C<:PagedKVCache,R<:ModelRunner}
    model::M
    tokenizer::BPETokenizer
    cache::C
    runner::R                    # keeps its CUDA graphs across turns
    seq::Sequence
    messages::Vector{ChatMessage}
    sampling::SamplingParams
    stop_ids::Vector{Int}
    enable_thinking::Bool
    rng::AbstractRNG
    speculative::Any                 # a proposer or nothing
end

function ChatSession(model::DecoderLM, tok::BPETokenizer; sampling::SamplingParams = SamplingParams(),
                     stop_ids = model.config.eos_token_ids, enable_thinking::Bool = true,
                     max_tokens::Integer = max_length(model), rng::AbstractRNG = Random.default_rng(),
                     speculative = nothing, kv_quantize::Union{Nothing,Symbol} = nothing)
    cache = PagedKVCache(model; max_tokens = min(max_tokens, max_length(model)), quantize = kv_quantize)
    return ChatSession(model, tok, cache, ModelRunner(model, cache), new_sequence(cache), ChatMessage[],
                       sampling, collect(Int, stop_ids), enable_thinking, rng, speculative)
end

"Statistics of one chat turn: prompt tokens reused from the cache and newly prefilled."
struct TurnStats
    reused::Int
    prefilled::Int
    generated::Int
    spec::SpecStats                  # speculative steps and drafts (all zero without)
end

"Clear the conversation and free its cache pages."
function reset!(s::ChatSession)
    empty!(s.messages)
    release!(s.cache, s.seq)
    return s
end

"Length of the longest common prefix of `a` and `b`."
common_prefix(a, b) = something(findfirst(i -> a[i] != b[i], 1:min(length(a), length(b))),
                                min(length(a), length(b)) + 1) - 1

"""
    reply!(session, text; max_new_tokens = 1024, on_token = nothing) -> (reply, TurnStats)

Add the user message `text`, generate the assistant's reply and add it to the
conversation. On error (e.g. the cache is full) the conversation is left unchanged.
"""
function reply!(s::ChatSession, text::AbstractString; max_new_tokens::Integer = 1024, on_token = nothing)
    messages = [s.messages; ChatMessage("user", text)]
    prompt = encode(s.tokenizer, chat_prompt(s.model.config.model_type, messages;
                                             enable_thinking = s.enable_thinking); add_special_tokens = false)
    # Reuse the cached prefix, but always run at least one token to get logits.
    reused = min(common_prefix(s.seq.tokens, prompt), length(prompt) - 1)
    truncate!(s.cache, s.seq, reused)
    spec = SpecStats()
    generated = generate!(s.runner, s.seq, prompt[reused+1:end]; max_new_tokens,
                          sampling = s.sampling, stop_ids = s.stop_ids, rng = s.rng, on_token,
                          s.speculative, spec_stats = spec)
    reply = decode(s.tokenizer, generated)
    append!(s.messages, (ChatMessage("user", text), ChatMessage("assistant", reply)))
    return reply, TurnStats(reused, length(prompt) - reused, length(generated), spec)
end
